"""
Rolling Daily Optuna Pipeline — 24-hour continuous hyperparameter optimization.

Runs daily 12:00 CET → 09:00 CET with phased GPU allocation:
  Day   (12:00-22:00): P40 = 1 worker, GTX 1050 = 3 workers  (bot on P40)
  Night (22:00-09:00): P40 = 6 workers, GTX 1050 = 3 workers  (full gas)

Warm-starts from persistent per-timeframe SQLite DBs, adds ~30 trials/symbol/day.
Rotates through timeframes: M15 → M30 → H1 → H4 (one per day).

Usage:
    python3 research/rolling_optuna.py              # normal run
    python3 research/rolling_optuna.py --dry-run    # show plan, don't execute
"""
from __future__ import annotations

import csv
import fcntl
import gc
import os
import signal
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.optuna_orchestrator import (
    DATA_ROOTS,
    discover_symbols_from_data,
    load_active_symbols,
    run_symbol,
)
from research.train_ml_strategy import (
    process_symbol as wfo_process_symbol,
    load_symbol_ticks,
    make_time_bars,
    infer_spread_bps,
    infer_slippage_bps,
    sanitize_training_frame,
    calc_metrics,
    fit_xgb_with_fallback,
    recency_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CET = timezone(timedelta(hours=1))  # CET = UTC+1 (winter), close enough
LOCKFILE = "/tmp/rolling_optuna.lock"
ROLLING_DIR = REPO_ROOT / "models" / "optuna_results" / "rolling"
AUDIT_DB = REPO_ROOT / "audit" / "sovereign_log.db"

TIMEFRAME_ROTATION = ["M15", "M30", "H1", "H4"]

# GPU IDs — P40 = 0, GTX 1050 = 1
GPU_P40 = 0
GPU_GTX = 1

# Phase configs: (p40_workers, gtx_workers)
PHASE_DAY = (1, 3)      # 12:00-22:00 — bot runs on P40
PHASE_NIGHT = (6, 3)     # 22:00-09:00 — full gas
PHASE_DONE = (0, 0)      # 09:00-12:00 — rest

# Per-symbol trial budget
TRIALS_PER_SYMBOL = 30
TRIAL_JOBS = 1  # sequential trials within symbol (avoids GPU contention)

# Skip hopeless symbols: >500 trials and best EV < 0
HOPELESS_TRIAL_THRESHOLD = 500

# Shared XGB/pipeline settings (match sunday_ritual.sh)
# WFO validation args (match train_ml_strategy defaults)
WFO_ARGS = {
    "data_roots": ",".join(DATA_ROOTS),
    "years": "",
    "symbols": "",
    "symbols_file": "",
    "timeframes": "H1",  # overridden per run
    "target_bars_per_day": 96,
    "z_threshold": 1.5,
    "pt_mult": 2.0,
    "sl_mult": 1.5,
    "horizon_bars": 24,
    "gap_bars": 24,
    "wf_train_ratio": 0.75,
    "wf_step_ratio": 0.25,
    "fee_bps": 3.0,
    "slippage_bps": 5.0,
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "symbol_workers": 1,
    "xgb_jobs": 1,
    "device": "cuda",
    "meta_threshold": 0.5,
    "out_dir": str(REPO_ROOT / "models" / "optuna_results" / "rolling" / "wfo"),
}

# Minimum WFO thresholds to consider "positive"
WFO_MIN_SHARPE = 0.0
WFO_MIN_PROFIT_FACTOR = 1.0

PIPELINE_ARGS = {
    "data_roots": DATA_ROOTS,
    "z_threshold": 1.0,
    "trials": TRIALS_PER_SYMBOL,
    "trial_jobs": TRIAL_JOBS,
    "train_size": 2400,
    "test_size": 600,
    "purge": 24,
    "embargo": 24,
    "pt_mult": 2.0,
    "sl_mult": 1.5,
    "horizon_bars": 24,
    "discovery_cutoff": TRIALS_PER_SYMBOL,  # no separate discovery phase for 30 trials
    "htf": False,
    "bar_roots": "",  # empty = use tick data (positive EV vs negative with bar data)
}

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"\n[rolling] Received signal {signum} — shutting down gracefully...")


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_discord():
    """Load Discord notifier (lazy, cached)."""
    try:
        import json as _json
        cfg_path = REPO_ROOT / "config" / "discord_config.json"
        if not cfg_path.exists():
            return None
        with open(cfg_path) as f:
            url = _json.load(f).get("webhook_url", "")
        if not url:
            return None
        from tools.discord_notifier import DiscordNotifier
        return DiscordNotifier(webhook_url=url)
    except Exception:
        return None


_discord = None  # lazy singleton


def get_discord():
    global _discord
    if _discord is None:
        _discord = _load_discord()
    return _discord


def _simulate_atr_exit(bars_close, bars_high, bars_low, entry_idx, direction,
                       atr_val, atr_sl_mult, atr_tp_mult, max_bars=48,
                       breakeven_atr=0.0, trail_activation_atr=0.0,
                       trail_distance_atr=0.0):
    """Simulate a trade with ATR SL/TP + breakeven + trailing stop.
    Matches live bot position_manager logic.
    Returns (pnl_pct, exit_bar_offset, exit_reason)."""
    entry_price = bars_close[entry_idx]
    sl_dist = atr_val * atr_sl_mult
    tp_dist = atr_val * atr_tp_mult

    if direction == "BUY":
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    breakeven_done = False
    trailing_active = False
    best_price = entry_price

    for offset in range(1, min(max_bars, len(bars_close) - entry_idx)):
        idx = entry_idx + offset
        hi = bars_high[idx]
        lo = bars_low[idx]
        cl = bars_close[idx]

        # Track best price for trailing
        if direction == "BUY":
            best_price = max(best_price, hi)
            current_profit_atr = (best_price - entry_price) / atr_val
        else:
            best_price = min(best_price, lo)
            current_profit_atr = (entry_price - best_price) / atr_val

        # Breakeven: move SL to entry after X ATR profit
        if not breakeven_done and breakeven_atr > 0 and current_profit_atr >= breakeven_atr:
            if direction == "BUY":
                sl = entry_price + atr_val * 0.01  # tiny buffer above entry
            else:
                sl = entry_price - atr_val * 0.01
            breakeven_done = True

        # Trailing stop: activate after X ATR, trail at Y ATR distance
        if trail_activation_atr > 0 and current_profit_atr >= trail_activation_atr:
            trailing_active = True

        if trailing_active and trail_distance_atr > 0:
            trail_dist = atr_val * trail_distance_atr
            if direction == "BUY":
                new_sl = best_price - trail_dist
                sl = max(sl, new_sl)
            else:
                new_sl = best_price + trail_dist
                sl = min(sl, new_sl)

        # Check SL/TP hits
        if direction == "BUY":
            if lo <= sl:
                pnl = (sl - entry_price) / entry_price
                return pnl, offset, "SL" if not breakeven_done else "BE"
            if hi >= tp:
                return tp_dist / entry_price, offset, "TP"
        else:
            if hi >= sl:
                pnl = (entry_price - sl) / entry_price
                return pnl, offset, "SL" if not breakeven_done else "BE"
            if lo <= tp:
                return tp_dist / entry_price, offset, "TP"

    # Max bars reached — exit at close
    exit_price = bars_close[min(entry_idx + max_bars, len(bars_close) - 1)]
    if direction == "BUY":
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price
    return pnl, max_bars, "TIMEOUT"


def run_wfo_validation(symbol: str, timeframe: str, optuna_ev: float,
                       optuna_result: dict | None = None) -> dict | None:
    """Realistic WFO: train XGBoost → predict proba → threshold filter → ATR SL/TP exit simulation.
    Matches actual bot behavior. Returns metrics dict or None on failure."""
    import json as _json
    import numpy as np
    import polars as pl
    from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
    from engine.labeling import apply_triple_barrier

    try:
        # Load symbol config for ATR mults and threshold
        cfg_path = REPO_ROOT / "config" / "sovereign_configs.json"
        with open(cfg_path) as f:
            all_cfg = _json.load(f)
        sym_cfg = all_cfg.get(symbol, {})
        atr_sl_mult = sym_cfg.get("atr_sl_mult", 1.2)
        atr_tp_mult = sym_cfg.get("atr_tp_mult", 3.6)
        prob_threshold = sym_cfg.get("prob_threshold", 0.55)
        exit_horizon = sym_cfg.get("exit_horizon", 24)

        # Load tick data → bars
        roots = DATA_ROOTS
        ticks = load_symbol_ticks(symbol, roots, set())
        if ticks is None or ticks.height < 1000:
            print(f"  WFO {symbol}: insufficient data")
            return None

        bars = make_time_bars(ticks, timeframe)
        if bars.height < 500:
            print(f"  WFO {symbol}: insufficient bars ({bars.height})")
            return None

        # Build features
        feat = build_bar_features(bars, z_threshold=1.0)
        feat = feat.with_columns(pl.col("time").dt.year().alias("year"))

        # Triple barrier for training labels only (bot trains on these)
        tb = apply_triple_barrier(
            close=feat["close"].to_numpy(),
            vol_proxy=feat["vol20"].to_numpy(),
            side=feat["primary_side"].to_numpy(),
            horizon=exit_horizon,
            pt_mult=2.0,
            sl_mult=1.5,
        )
        feat = feat.with_columns([
            pl.Series("label", tb.label),
            pl.Series("tb_ret", tb.tb_ret),
            pl.Series("upside", tb.upside),
            pl.Series("downside", tb.downside),
        ])
        feat = feat.filter(pl.col("label").is_finite())
        feat = sanitize_training_frame(feat)
        if feat.height < 400:
            print(f"  WFO {symbol}: insufficient labeled bars ({feat.height})")
            return None

        # Arrays
        X = feat.select(FEATURE_COLUMNS).to_numpy()
        y = feat["label"].cast(pl.Int8).to_numpy()
        yr = feat["year"].to_numpy()
        close = feat["close"].to_numpy()
        high = feat["high"].to_numpy()
        low = feat["low"].to_numpy()
        z20 = feat.select("z20").to_numpy().ravel()
        w = recency_weights(yr)

        # Costs
        spread_bps = float(infer_spread_bps(ticks))
        slippage_bps = float(infer_slippage_bps(symbol))
        fee_bps = optuna_result.get("fee_bps", 3.0) if optuna_result else 3.0
        cost_pct = (fee_bps + spread_bps + slippage_bps) / 1e4

        # ATR array (14-period on close, same as bot)
        atr = np.full(len(close), np.nan)
        for i in range(14, len(close)):
            tr_vals = []
            for j in range(i - 13, i + 1):
                tr = max(high[j] - low[j],
                         abs(high[j] - close[j - 1]),
                         abs(low[j] - close[j - 1]))
                tr_vals.append(tr)
            atr[i] = np.mean(tr_vals)

        # XGBoost params from Optuna
        if optuna_result:
            model_params = dict(
                n_estimators=optuna_result.get("best_num_boost_round", 500),
                max_depth=optuna_result.get("best_max_depth", 6),
                learning_rate=optuna_result.get("best_eta", 0.04),
                subsample=optuna_result.get("best_subsample", 0.9),
                colsample_bytree=optuna_result.get("best_colsample_bytree", 0.9),
                reg_alpha=optuna_result.get("best_reg_alpha", 0.05),
                reg_lambda=optuna_result.get("best_reg_lambda", 1.0),
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=1,
                random_state=42,
            )
        else:
            model_params = dict(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.05, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                n_jobs=1, random_state=42,
            )

        # Walk-forward: 75% train, 25% test
        n = len(feat)
        split = int(n * 0.75)
        gap = 24  # embargo
        tr_idx = np.arange(0, split)
        te_start = split + gap
        if te_start >= n - 50:
            print(f"  WFO {symbol}: not enough test data")
            return None
        te_idx = np.arange(te_start, n)

        # Train
        ytr = y[tr_idx]
        if len(np.unique(ytr)) < 2:
            print(f"  WFO {symbol}: single class in training")
            return None

        pos = max(int(ytr.sum()), 1)
        neg = max(int(len(ytr) - ytr.sum()), 1)
        model_params["scale_pos_weight"] = neg / pos
        model, device = fit_xgb_with_fallback(model_params, X[tr_idx], ytr, w[tr_idx], "cuda")

        # Predict on test
        probas = model.predict_proba(X[te_idx])[:, 1]

        # --- Simulate trades helper ---
        def _run_simulation(indices, proba_arr, sl_m, tp_m, be_atr, trail_act, trail_dist):
            """Run trade simulation on given bar indices. Returns list of net returns."""
            rets = []
            i = 0
            while i < len(indices):
                bar_idx = indices[i]
                p = proba_arr[i]
                if p < prob_threshold or np.isnan(atr[bar_idx]):
                    i += 1
                    continue
                if z20[bar_idx] < 0:
                    d = "BUY"
                elif z20[bar_idx] > 0:
                    d = "SELL"
                else:
                    i += 1
                    continue
                pnl, held, _ = _simulate_atr_exit(
                    close, high, low, bar_idx, d,
                    atr[bar_idx], sl_m, tp_m,
                    max_bars=exit_horizon,
                    breakeven_atr=be_atr,
                    trail_activation_atr=trail_act,
                    trail_distance_atr=trail_dist,
                )
                rets.append(pnl - cost_pct)
                i += max(held, 1)
            return rets

        # --- Optimize SL/TP/exit on TRAIN data ---
        train_probas = model.predict_proba(X[tr_idx])[:, 1]

        SL_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]
        TP_GRID = [1.5, 2.0, 2.5, 3.5, 4.5]
        BE_GRID = [0.0, 0.2, 0.5]
        # Trail params coupled: (activation, distance)
        TRAIL_GRID = [(0.0, 0.0), (0.8, 0.4), (1.2, 0.5)]

        best_score = -999.0
        best_exit_params = (atr_sl_mult, atr_tp_mult, 0.2, 0.8, 0.4)

        for sl_m in SL_GRID:
            for tp_m in TP_GRID:
                if tp_m <= sl_m:
                    continue  # TP must be > SL
                for be in BE_GRID:
                    for trail_act, trail_dist in TRAIL_GRID:
                        rets = _run_simulation(
                            tr_idx, train_probas, sl_m, tp_m, be, trail_act, trail_dist)
                        if len(rets) < 10:
                            continue
                        s = calc_metrics(rets)
                        # Score: Sharpe weighted by profit factor
                        score = s["sharpe"] * min(s["profit_factor"], 2.0)
                        if score > best_score:
                            best_score = score
                            best_exit_params = (sl_m, tp_m, be, trail_act, trail_dist)

        opt_sl, opt_tp, opt_be, opt_trail_act, opt_trail_dist = best_exit_params
        print(f"  WFO {symbol}: optimized exits SL={opt_sl}×ATR TP={opt_tp}×ATR "
              f"BE={opt_be} trail={opt_trail_act}/{opt_trail_dist} "
              f"(train score={best_score:.2f})")

        # --- Validate on TEST data with optimized params ---
        trade_returns = _run_simulation(
            te_idx, probas, opt_sl, opt_tp, opt_be, opt_trail_act, opt_trail_dist)

        if len(trade_returns) < 5:
            print(f"  WFO {symbol}: only {len(trade_returns)} trades in test")
            return None

        # Metrics
        stats = calc_metrics(trade_returns)
        avg = {
            "n_trades": stats["n_trades"],
            "win_rate": stats["win_rate"],
            "sharpe": stats["sharpe"],
            "sortino": stats["sortino"],
            "max_dd": stats["max_drawdown_pct"],
            "profit_factor": stats["profit_factor"],
            "test_bars": len(te_idx),
            "device": device,
            "opt_sl_mult": opt_sl,
            "opt_tp_mult": opt_tp,
            "opt_breakeven_atr": opt_be,
            "opt_trail_activation": opt_trail_act,
            "opt_trail_distance": opt_trail_dist,
        }

        is_positive = (avg["sharpe"] > WFO_MIN_SHARPE
                       and avg["profit_factor"] > WFO_MIN_PROFIT_FACTOR
                       and avg["n_trades"] >= 10)

        tag = "PASS" if is_positive else "FAIL"
        print(f"  WFO {symbol}: {tag} | Sharpe={avg['sharpe']:.2f} "
              f"PF={avg['profit_factor']:.2f} WR={avg['win_rate']:.0%} "
              f"trades={avg['n_trades']} thresh={prob_threshold} "
              f"SL={opt_sl}×ATR TP={opt_tp}×ATR BE={opt_be} "
              f"trail={opt_trail_act}/{opt_trail_dist}")

        if is_positive:
            discord = get_discord()
            if discord:
                discord.send(
                    title=f"WFO Pass: {symbol} ({timeframe})",
                    description=(
                        f"**Optuna EV:** {optuna_ev:.6f}\n"
                        f"**Sharpe:** {avg['sharpe']:.2f} | **PF:** {avg['profit_factor']:.2f}\n"
                        f"**Win rate:** {avg['win_rate']:.0%} | **Trades:** {avg['n_trades']}\n"
                        f"**Max DD:** {avg['max_dd']:.1%} | **Sortino:** {avg['sortino']:.2f}\n"
                        f"**Optimized exits:** SL={opt_sl}×ATR TP={opt_tp}×ATR\n"
                        f"BE={opt_be}×ATR trail={opt_trail_act}/{opt_trail_dist}×ATR"
                    ),
                    color="green",
                )

        return avg
    except Exception as e:
        import traceback
        print(f"  WFO {symbol}: ERROR {e}")
        traceback.print_exc()
        return None


def get_current_phase() -> str:
    """Return current phase based on CET time."""
    hour = datetime.now(CET).hour
    if 12 <= hour < 22:
        return "day"
    elif hour >= 22 or hour < 9:
        return "night"
    else:
        return "done"  # 09:00-12:00 = rest


def get_phase_workers(phase: str) -> tuple[int, int]:
    """Return (p40_workers, gtx_workers) for given phase."""
    if phase == "day":
        return PHASE_DAY
    elif phase == "night":
        return PHASE_NIGHT
    return PHASE_DONE


def past_deadline() -> bool:
    """True if past 09:00 CET (next morning)."""
    now = datetime.now(CET)
    # If it's the next day between 09:00 and 12:00, we're done
    return now.hour >= 9 and now.hour < 12


def todays_timeframe() -> str:
    """Rotate through timeframes based on day-of-year."""
    doy = datetime.now(CET).timetuple().tm_yday
    return TIMEFRAME_ROTATION[doy % len(TIMEFRAME_ROTATION)]


def check_open_positions() -> int:
    """Query latest heartbeat from audit DB for open position count."""
    if not AUDIT_DB.exists():
        return 0
    try:
        conn = sqlite3.connect(str(AUDIT_DB), timeout=5)
        row = conn.execute(
            "SELECT open_positions FROM heartbeats "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def is_sunday_ritual_active() -> bool:
    """Check if sunday-ritual.service is currently running."""
    try:
        ret = os.system("systemctl --user is-active --quiet sunday-ritual.service")
        return ret == 0
    except Exception:
        return False


def is_hopeless(symbol: str, timeframe: str, db_path: str) -> bool:
    """Check if a symbol has >500 trials with best EV < 0 in the study DB."""
    if not os.path.exists(db_path):
        return False
    try:
        import optuna
        storage_url = f"sqlite:///{db_path}"
        study_name = f"{symbol}_{timeframe}"
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
        )
        n_trials = len(study.trials)
        if n_trials < HOPELESS_TRIAL_THRESHOLD:
            return False
        best_val = study.best_trial.value if study.best_trial else -1.0
        return best_val < 0
    except Exception:
        return False


def get_live_tickers() -> list[str]:
    """Load live tickers from sovereign_configs.json (dynamic, not hardcoded)."""
    try:
        return load_active_symbols()
    except SystemExit:
        return []


def build_symbol_queue(timeframe: str) -> list[str]:
    """Build prioritized symbol queue: live tickers first, then rest alphabetically.
    Skip hopeless symbols."""
    all_symbols = discover_symbols_from_data(DATA_ROOTS)
    if not all_symbols:
        return []

    live_tickers = get_live_tickers()
    db_path = str(ROLLING_DIR / timeframe / "optuna_studies.db")

    # Separate live from rest
    live_set = set(live_tickers)
    live = [s for s in live_tickers if s in set(all_symbols)]
    rest = [s for s in all_symbols if s not in live_set]

    # Filter out hopeless symbols
    queue = []
    skipped = 0
    for sym in live + rest:
        if is_hopeless(sym, timeframe, db_path):
            skipped += 1
        else:
            queue.append(sym)

    if skipped:
        print(f"[rolling] Skipped {skipped} hopeless symbols (>{HOPELESS_TRIAL_THRESHOLD} trials, EV<0)")

    return queue


def _run_symbol_with_gpu(symbol: str, args_dict: dict, gpu_id: int) -> dict:
    """Worker wrapper: set CUDA_VISIBLE_DEVICES before running."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        return run_symbol(symbol, args_dict)
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e)}
    finally:
        gc.collect()


def update_rolling_summary(results: list[dict], timeframe: str):
    """Merge new results into rolling_summary.csv — keep best EV per (symbol, TF)."""
    summary_path = ROLLING_DIR / "rolling_summary.csv"

    # Load existing
    existing: dict[tuple[str, str], dict] = {}
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["symbol"], row["timeframe"])
                existing[key] = row

    # Merge new results
    for r in results:
        if r.get("status") != "ok":
            continue
        sym = r["symbol"]
        key = (sym, timeframe)
        ev = float(r["best_ev"])
        old_ev = float(existing[key]["best_ev"]) if key in existing else -999.0
        if ev > old_ev:
            existing[key] = {
                "symbol": sym,
                "timeframe": timeframe,
                "best_ev": f"{ev:.6f}",
                "calmar_score": f"{r.get('calmar_score', ev):.6f}",
                "trials": str(r.get("trials", 0)),
                "cluster": r.get("cluster", ""),
                "fee_bps": f"{r.get('fee_bps', 0):.1f}",
                "spread_bps": f"{r.get('spread_bps', 0):.1f}",
                "slippage_bps": f"{r.get('slippage_bps', 0):.1f}",
                "updated": datetime.now(CET).strftime("%Y-%m-%d %H:%M"),
            }

    # Write back
    if not existing:
        return

    fieldnames = ["symbol", "timeframe", "best_ev", "calmar_score", "trials",
                  "cluster", "fee_bps", "spread_bps", "slippage_bps", "updated"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing.keys()):
            writer.writerow(existing[key])

    print(f"[rolling] Updated {summary_path} ({len(existing)} entries)")


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def run_rolling_pipeline(dry_run: bool = False):
    """Main entry point for the rolling daily Optuna pipeline."""
    start_time = datetime.now(CET)
    timeframe = todays_timeframe()
    out_dir = str(ROLLING_DIR / timeframe)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[rolling] ═══════════════════════════════════════════════════════")
    print(f"[rolling] Rolling Daily Optuna Pipeline")
    print(f"[rolling] Start:     {start_time.strftime('%Y-%m-%d %H:%M CET')}")
    print(f"[rolling] Deadline:  09:00 CET tomorrow")
    print(f"[rolling] Timeframe: {timeframe}")
    print(f"[rolling] Trials:    {TRIALS_PER_SYMBOL} per symbol (warm-start)")
    print(f"[rolling] DB:        {out_dir}/optuna_studies.db")
    print(f"[rolling] ═══════════════════════════════════════════════════════")

    # Build queue
    queue = build_symbol_queue(timeframe)
    if not queue:
        print("[rolling] No symbols to process — exiting")
        return

    live_tickers = get_live_tickers()
    print(f"[rolling] Queue: {len(queue)} symbols "
          f"(live first: {[s for s in queue if s in live_tickers]})")

    if dry_run:
        print("[rolling] DRY RUN — would process:")
        for i, sym in enumerate(queue):
            print(f"  {i+1:3d}. {sym}")
        return

    # Prepare args
    args_dict = dict(PIPELINE_ARGS)
    args_dict["timeframe"] = timeframe
    args_dict["out_dir"] = out_dir

    results: list[dict] = []
    processed = 0
    remaining_queue = list(queue)

    while remaining_queue and not _shutdown:
        # Check deadline
        if past_deadline():
            print(f"[rolling] Deadline reached (09:00 CET) — stopping with "
                  f"{len(remaining_queue)} symbols remaining")
            break

        # Check phase and workers
        phase = get_current_phase()
        if phase == "done":
            print("[rolling] Phase=done (09:00-12:00) — stopping")
            break

        p40_workers, gtx_workers = get_phase_workers(phase)

        # Safety: if many open positions, reduce P40 load
        open_pos = check_open_positions()
        if open_pos > 10:
            print(f"[rolling] WARNING: {open_pos} open positions — P40 workers → 0")
            p40_workers = 0

        total_workers = p40_workers + gtx_workers
        if total_workers == 0:
            print("[rolling] No workers available — waiting 60s...")
            time.sleep(60)
            continue

        # Build GPU assignment for this batch
        batch_size = min(total_workers, len(remaining_queue))
        batch = remaining_queue[:batch_size]
        remaining_queue = remaining_queue[batch_size:]

        # Assign GPUs: first gtx_workers get GTX, rest get P40
        gpu_assignments = []
        for i in range(batch_size):
            if i < gtx_workers:
                gpu_assignments.append(GPU_GTX)
            else:
                gpu_assignments.append(GPU_P40)

        phase_label = f"[{phase}] P40={p40_workers} GTX={gtx_workers}"
        print(f"\n[rolling] {phase_label} — Batch {processed+1}-{processed+batch_size} "
              f"of {len(queue)} | {len(remaining_queue)} remaining")

        # Submit batch
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            futures = {}
            for sym, gpu_id in zip(batch, gpu_assignments):
                f = executor.submit(_run_symbol_with_gpu, sym, args_dict, gpu_id)
                futures[f] = (sym, gpu_id)

            for future in as_completed(futures):
                if _shutdown:
                    print("[rolling] Shutdown requested — cancelling pending futures")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                sym, gpu_id = futures[future]
                try:
                    r = future.result(timeout=600)  # 10 min max per symbol
                    status = r.get("status", "unknown")
                    ev = r.get("best_ev", 0)
                    trials = r.get("trials", 0)
                    if status == "ok":
                        print(f"  ✓ {sym:20s} EV={ev:.6f} trials={trials} gpu={gpu_id}")
                    elif status == "killed_negative_ev":
                        print(f"  ✗ {sym:20s} KILLED EV={ev:.6f} trials={trials}")
                    else:
                        print(f"  - {sym:20s} {status}")
                    # Run WFO on everything that produced a model (ok or killed)
                    if status in ("ok", "killed_negative_ev"):
                        run_wfo_validation(sym, timeframe, ev, optuna_result=r)
                    results.append(r)
                except Exception as e:
                    print(f"  ! {sym:20s} ERROR: {e}")
                    results.append({"symbol": sym, "status": "error", "error": str(e)})

                processed += 1

    # Write results
    elapsed = datetime.now(CET) - start_time
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    killed_count = sum(1 for r in results if r.get("status") == "killed_negative_ev")
    error_count = sum(1 for r in results if r.get("status") == "error")

    print(f"\n[rolling] ═══════════════════════════════════════════════════════")
    print(f"[rolling] DONE — {processed}/{len(queue)} symbols in {elapsed}")
    print(f"[rolling] OK={ok_count} Killed={killed_count} Error={error_count}")
    print(f"[rolling] ═══════════════════════════════════════════════════════")

    # Update rolling summary
    if results:
        update_rolling_summary(results, timeframe)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Rolling Daily Optuna Pipeline")
    p.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = p.parse_args()

    # Check if sunday-ritual is active
    if is_sunday_ritual_active():
        print("[rolling] Sunday ritual is active — skipping")
        sys.exit(0)

    # Lockfile
    try:
        lock_fd = open(LOCKFILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
    except (IOError, OSError):
        print(f"[rolling] Another instance is running (lockfile: {LOCKFILE}) — exiting")
        sys.exit(0)

    try:
        run_rolling_pipeline(dry_run=args.dry_run)
    finally:
        # Release lockfile
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            os.unlink(LOCKFILE)
        except Exception:
            pass


if __name__ == "__main__":
    main()
