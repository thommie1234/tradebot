"""
Exit-parameter Optuna orchestrator — second-stage optimization.

Optimizes 5 exit parameters per symbol × timeframe via vectorized trade
simulation with walk-forward validation and realistic broker costs.

Usage:
    python3 research/exit_optuna.py --active --trials 400
    python3 research/exit_optuna.py --symbols SOLUSD --trials 10
    python3 research/exit_optuna.py --active --trials 400 --workers 28 --update-configs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from trading_prop.ml.integrated_pipeline import (
    purged_walk_forward_splits,
    set_polars_threads,
)
from trading_prop.ml.train_ml_strategy import (
    infer_spread_bps,
    make_time_bars,
)
from trading_prop.production.optuna_orchestrator import (
    DATA_ROOTS,
    broker_commission_bps,
    broker_slippage_bps,
    calmar_weighted_score,
    cluster_for_symbol,
    discover_symbols_from_data,
    load_symbol_ticks_lf,
)
from research.exit_simulator import ExitParams, simulate_trades

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = REPO_ROOT / "config" / "sovereign_configs.json"
MODEL_DIR = REPO_ROOT / "models" / "sovereign_models"
BAR_ROOT = Path("/home/tradebot/ssd_data_2/bars")

ALL_TIMEFRAMES = ["M1", "M15", "M30", "H1", "H4"]

# Walk-forward sizes scaled per timeframe (more bars on smaller TFs)
TF_WF_SIZES = {
    "M1":  {"train_size": 10000, "test_size": 2500},
    "M5":  {"train_size": 4800,  "test_size": 1200},
    "M15": {"train_size": 4800,  "test_size": 1200},
    "M30": {"train_size": 3000,  "test_size": 800},
    "H1":  {"train_size": 1000,  "test_size": 300},
    "H4":  {"train_size": 250,   "test_size": 80},
}

# ---------------------------------------------------------------------------
# Search spaces per asset class
# ---------------------------------------------------------------------------

EXIT_SEARCH_SPACES = {
    "equity": {
        "atr_sl_mult": (0.5, 2.5),
        "atr_tp_mult": (2.0, 8.0),
        "breakeven_atr": (0.3, 1.5),
        "trail_activation_atr": (1.0, 4.0),
        "trail_distance_atr": (0.3, 2.0),
        "horizon": (6, 36),
    },
    "crypto": {
        "atr_sl_mult": (0.8, 3.0),
        "atr_tp_mult": (3.0, 10.0),
        "breakeven_atr": (0.3, 2.0),
        "trail_activation_atr": (1.5, 5.0),
        "trail_distance_atr": (0.5, 2.5),
        "horizon": (6, 48),
    },
    "index": {
        "atr_sl_mult": (0.5, 2.5),
        "atr_tp_mult": (2.0, 8.0),
        "breakeven_atr": (0.3, 1.5),
        "trail_activation_atr": (1.0, 4.0),
        "trail_distance_atr": (0.3, 2.0),
        "horizon": (6, 36),
    },
}
EXIT_SEARCH_SPACES["forex"] = EXIT_SEARCH_SPACES["index"]
EXIT_SEARCH_SPACES["commodity"] = EXIT_SEARCH_SPACES["index"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_model_path(symbol: str) -> Path | None:
    """Resolve symbol to XGBoost model .json path."""
    candidates = [
        MODEL_DIR / f"{symbol}.json",
        MODEL_DIR / f"{symbol.replace('_', '')}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_bars_from_parquets(symbol: str, timeframe: str) -> pl.DataFrame | None:
    """Load pre-downloaded bar parquets from /ssd_data_2/bars/{TF}/{SYMBOL}/.

    Tries symbol name variants (SYMBOL, SYMBOL with _ removed, / replaced).
    Returns DataFrame with [time, open, high, low, close, volume] or None.
    """
    variants = [
        symbol,
        symbol.replace("_", "/"),
        symbol.replace("_", ""),
    ]
    for variant in variants:
        bar_dir = BAR_ROOT / timeframe / variant
        if not bar_dir.is_dir():
            continue
        pq_files = sorted(bar_dir.glob("*.parquet"))
        if not pq_files:
            continue
        dfs = []
        for pf in pq_files:
            try:
                df = pl.read_parquet(pf)
                dfs.append(df)
            except Exception:
                continue
        if not dfs:
            continue
        combined = pl.concat(dfs, how="vertical").sort("time")
        # Normalise column names — MT5 bars use lowercase already
        rename = {}
        for col in combined.columns:
            if col.lower() in ("time", "open", "high", "low", "close", "volume",
                                "tick_volume", "spread", "real_volume"):
                rename[col] = col.lower()
        combined = combined.rename(rename)
        # Use tick_volume as volume if volume is missing or zero
        if "volume" not in combined.columns and "tick_volume" in combined.columns:
            combined = combined.rename({"tick_volume": "volume"})
        elif "volume" in combined.columns and "tick_volume" in combined.columns:
            vol_sum = combined.select(pl.col("volume").sum()).item()
            if vol_sum == 0:
                combined = combined.drop("volume").rename({"tick_volume": "volume"})
        # Ensure time is datetime
        if combined["time"].dtype == pl.Int64 or combined["time"].dtype == pl.UInt64:
            combined = combined.with_columns(
                (pl.col("time") * 1_000_000).cast(pl.Datetime("us", "UTC")).alias("time")
            )
        cols = ["time", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in combined.columns]
        if missing:
            continue
        return combined.select(cols).drop_nulls()
    return None


def compute_atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute ATR(14) from OHLC arrays."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(13, n):
        atr[i] = np.mean(tr[i - 13 : i + 1])
    return atr


# ---------------------------------------------------------------------------
# Per-symbol-timeframe worker
# ---------------------------------------------------------------------------

def run_symbol_tf(symbol: str, timeframe: str, args_dict: dict) -> dict:
    """Optimize exit params for one symbol on one timeframe."""
    set_polars_threads(2)

    cluster = cluster_for_symbol(symbol)
    roots = list(args_dict["data_roots"])
    data_source = "ticks"

    # 1. Try pre-downloaded bar parquets first
    bars = load_bars_from_parquets(symbol, timeframe)
    if bars is not None and bars.height >= 500:
        data_source = "bars"
        # Trim to max_years of history
        max_years = args_dict.get("max_years", 10)
        cutoff = datetime.now(__import__("datetime").timezone.utc) - __import__("datetime").timedelta(days=max_years * 365)
        # Cast cutoff to match bar time dtype
        time_dtype = bars["time"].dtype
        cutoff_lit = pl.lit(cutoff).cast(time_dtype)
        bars = bars.filter(pl.col("time") >= cutoff_lit)
        # Spread not available from bar data — use heuristic
        ticks_df = None
    else:
        # Fallback: tick data → aggregate to bars
        ticks_lf = load_symbol_ticks_lf(symbol, roots)
        if ticks_lf is None:
            return {"symbol": symbol, "timeframe": timeframe,
                    "status": "no_data", "cluster": cluster}

        ticks_df = ticks_lf.collect()
        if ticks_df.height == 0:
            return {"symbol": symbol, "timeframe": timeframe,
                    "status": "no_data", "cluster": cluster}
        if ticks_df.select(pl.col("size").sum()).item() <= 0:
            ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

        bars = make_time_bars(
            ticks_df.select(["time", "price", "size"]), timeframe,
        )

    feat = build_bar_features(bars, z_threshold=args_dict["z_threshold"])

    min_rows = 500
    if feat.height < min_rows:
        return {"symbol": symbol, "timeframe": timeframe,
                "status": "insufficient_rows", "rows": feat.height,
                "cluster": cluster}

    feat_clean = feat.drop_nulls(FEATURE_COLUMNS)

    # 2. Entry signals from primary_side (z-score)
    primary_side = feat_clean["primary_side"].to_numpy().astype(np.int32)
    signal_mask = primary_side != 0
    entry_indices = np.where(signal_mask)[0]
    directions = primary_side[signal_mask]

    # 3. Optional model filter (per-symbol threshold from configs)
    model_path = find_model_path(symbol)
    if model_path is not None:
        try:
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            x = feat_clean.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
            dmat = xgb.DMatrix(x)
            probas = bst.predict(dmat)
            if probas.std() > 0.01:
                threshold = args_dict.get("sym_thresholds", {}).get(
                    symbol, args_dict["prob_threshold"])
                model_ok = (probas >= threshold) | (probas <= (1.0 - threshold))
                model_filter = model_ok[signal_mask]
                entry_indices = entry_indices[model_filter]
                directions = directions[model_filter]
        except Exception:
            pass

    if len(entry_indices) < 20:
        return {"symbol": symbol, "timeframe": timeframe,
                "status": "too_few_signals",
                "n_signals": len(entry_indices), "cluster": cluster}

    # 4. OHLCV + ATR arrays
    open_arr = feat_clean["open"].to_numpy().astype(np.float64)
    high_arr = feat_clean["high"].to_numpy().astype(np.float64)
    low_arr = feat_clean["low"].to_numpy().astype(np.float64)
    close_arr = feat_clean["close"].to_numpy().astype(np.float64)
    atr_arr = compute_atr14(high_arr, low_arr, close_arr)

    # 5. Realistic transaction costs
    if ticks_df is not None:
        spread_bps = infer_spread_bps(ticks_df)
    else:
        # Heuristic spread from asset class when using bar data
        spread_bps = {"crypto": 15.0, "forex": 2.0, "equity": 3.0,
                      "index": 2.0, "commodity": 4.0}.get(cluster, 3.0)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)
    cost_pct = (fee_bps + spread_bps + slippage_bps * 2.0) / 1e4

    # 6. Walk-forward splits (scaled per TF)
    wf = TF_WF_SIZES.get(timeframe, TF_WF_SIZES["H1"])
    n_samples = len(feat_clean)
    splits = purged_walk_forward_splits(
        n_samples=n_samples,
        train_size=wf["train_size"],
        test_size=wf["test_size"],
        purge=args_dict["purge"],
        embargo=args_dict["embargo"],
    )
    if not splits:
        return {"symbol": symbol, "timeframe": timeframe,
                "status": "no_folds", "rows": n_samples, "cluster": cluster}

    # 7. Optuna study — in-memory (no SQLite lock contention)
    space = EXIT_SEARCH_SPACES.get(cluster, EXIT_SEARCH_SPACES["index"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    min_rrr = args_dict.get("min_rrr", 0.0)
    use_sizing = args_dict.get("use_sizing", False)
    account_size = args_dict.get("account_size", 100_000.0)

    def objective(trial) -> float:
        sl = trial.suggest_float("atr_sl_mult", *space["atr_sl_mult"])
        tp = trial.suggest_float("atr_tp_mult", *space["atr_tp_mult"])
        be = trial.suggest_float("breakeven_atr", *space["breakeven_atr"])
        ta = trial.suggest_float("trail_activation_atr", *space["trail_activation_atr"])
        td = trial.suggest_float("trail_distance_atr", *space["trail_distance_atr"])
        hz = trial.suggest_int("horizon", *space["horizon"])

        # Position sizing: risk % of account per trade
        if use_sizing:
            risk_pct = trial.suggest_float("risk_pct", 0.005, 0.03)
        else:
            risk_pct = 0.0

        # Constraints: trail > breakeven, tp > sl, RRR check
        if ta <= be or tp <= sl:
            raise optuna.TrialPruned()
        if min_rrr > 0 and tp < sl * min_rrr:
            raise optuna.TrialPruned()

        ep = ExitParams(
            atr_sl_mult=sl, atr_tp_mult=tp,
            breakeven_atr=be, trail_activation_atr=ta,
            trail_distance_atr=td, horizon=hz,
        )

        ev_scores: list[float] = []
        for fold_idx, (_, te_idx) in enumerate(splits):
            fold_mask = np.isin(entry_indices, te_idx)
            fold_entries = entry_indices[fold_mask]
            fold_dirs = directions[fold_mask]

            if len(fold_entries) < 5:
                continue

            pnl, _, _ = simulate_trades(
                fold_entries, fold_dirs,
                open_arr, high_arr, low_arr, close_arr, atr_arr,
                ep, cost_pct,
            )
            if len(pnl) == 0:
                continue

            if use_sizing:
                # Convert % returns to dollar PnL with position sizing
                # SL distance as fraction of price
                sl_fracs = atr_arr[fold_entries] * sl / close_arr[fold_entries]
                sl_fracs = np.clip(sl_fracs, 1e-6, None)
                # Position $ = risk_amount / sl_frac; dollar_pnl = position * pnl
                risk_amount = account_size * risk_pct
                dollar_pnl = risk_amount * pnl / sl_fracs[:len(pnl)]
                # Build equity curve, compute Calmar
                equity = np.cumsum(dollar_pnl)
                peak = np.maximum.accumulate(equity)
                drawdown = peak - equity
                max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
                total_return = float(equity[-1]) if len(equity) > 0 else 0.0
                # FTMO penalty: max 10% total drawdown
                dd_pct = max_dd / account_size
                if dd_pct > 0.10:
                    ev_scores.append(-abs(total_return / account_size))
                    continue
                # Calmar = annualized return / max drawdown
                calmar = total_return / max(max_dd, 1.0)
                ev_scores.append(calmar)
            else:
                ev_scores.append(float(np.mean(pnl)))

            trial.report(float(np.mean(ev_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not ev_scores:
            return -999.0

        return calmar_weighted_score(ev_scores)

    study.optimize(
        objective,
        n_trials=args_dict["trials"],
        n_jobs=1,
        show_progress_bar=False,
    )

    if not study.best_trial:
        return {"symbol": symbol, "timeframe": timeframe,
                "status": "no_completed_trials", "cluster": cluster}

    best = study.best_trial
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "ok",
        "cluster": cluster,
        "data_source": data_source,
        "n_signals": int(len(entry_indices)),
        "n_folds": len(splits),
        "rows": n_samples,
        "trials": len(study.trials),
        "best_ev": float(best.value),
        "best_atr_sl_mult": float(best.params["atr_sl_mult"]),
        "best_atr_tp_mult": float(best.params["atr_tp_mult"]),
        "best_breakeven_atr": float(best.params["breakeven_atr"]),
        "best_trail_activation_atr": float(best.params["trail_activation_atr"]),
        "best_trail_distance_atr": float(best.params["trail_distance_atr"]),
        "best_horizon": int(best.params["horizon"]),
        "cost_bps": round((fee_bps + spread_bps + slippage_bps * 2.0), 1),
        "fee_bps": fee_bps,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
    }
    if use_sizing and "risk_pct" in best.params:
        result["best_risk_pct"] = float(best.params["risk_pct"])
    return result


def _worker(task: tuple) -> dict:
    """Wrapper for ProcessPoolExecutor — unpacks (symbol, tf, args_dict)."""
    symbol, tf, args_dict = task
    return run_symbol_tf(symbol, tf, args_dict)


# ---------------------------------------------------------------------------
# Config updater — picks best TF per symbol
# ---------------------------------------------------------------------------

def update_sovereign_configs(results: list[dict]) -> None:
    """Merge best exit params (across TFs) into sovereign_configs.json."""
    if not CONFIG_PATH.exists():
        print(f"WARNING: {CONFIG_PATH} not found — skipping config update")
        return

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    # Pick best TF per symbol by EV
    best_per_sym: dict[str, dict] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        sym = r["symbol"]
        if sym not in best_per_sym or r["best_ev"] > best_per_sym[sym]["best_ev"]:
            best_per_sym[sym] = r

    updated = 0
    for sym, r in best_per_sym.items():
        if sym not in configs:
            continue
        configs[sym]["atr_sl_mult"] = round(r["best_atr_sl_mult"], 3)
        configs[sym]["atr_tp_mult"] = round(r["best_atr_tp_mult"], 3)
        configs[sym]["breakeven_atr"] = round(r["best_breakeven_atr"], 3)
        configs[sym]["trail_activation_atr"] = round(r["best_trail_activation_atr"], 3)
        configs[sym]["trail_distance_atr"] = round(r["best_trail_distance_atr"], 3)
        configs[sym]["exit_horizon"] = r["best_horizon"]
        configs[sym]["exit_timeframe"] = r["timeframe"]
        if "best_risk_pct" in r:
            configs[sym]["risk_per_trade"] = round(r["best_risk_pct"], 4)
        updated += 1

    with open(CONFIG_PATH, "w") as f:
        json.dump(configs, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[exit-optuna] Updated {updated} symbols in {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Symbol discovery
# ---------------------------------------------------------------------------

def load_active_symbols() -> list[str]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"sovereign_configs.json not found at {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        configs = json.load(f)
    return sorted(configs.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exit-parameter Optuna optimization (multi-TF, per-symbol)",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--active", action="store_true",
                     help="Use symbols from sovereign_configs.json")
    grp.add_argument("--all", action="store_true",
                     help="Auto-discover all symbols with tick data")
    grp.add_argument("--symbols", type=str, default="",
                     help="Comma-separated symbol list")

    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--timeframes", type=str, default="M1,M15,M30,H1,H4",
                   help="Comma-separated timeframes (default: M1,M15,M30,H1,H4)")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--purge", type=int, default=8)
    p.add_argument("--embargo", type=int, default=8)
    p.add_argument("--z-threshold", type=float, default=1.0)
    p.add_argument("--prob-threshold", type=float, default=0.55)
    p.add_argument("--min-rrr", type=float, default=0.0,
                   help="Minimum risk-reward ratio: TP >= min_rrr * SL (e.g. 3.0 for 1:3)")
    p.add_argument("--use-sizing", action="store_true",
                   help="Optimize position sizing (risk_pct 0.5%%-3.0%%) with Calmar objective")
    p.add_argument("--account-size", type=float, default=100_000.0,
                   help="Account size in USD for position sizing simulation")
    p.add_argument("--update-configs", action="store_true",
                   help="Write best params back to sovereign_configs.json")
    p.add_argument("--max-years", type=int, default=10,
                   help="Maximum years of history to use (default: 10)")
    p.add_argument("--out-dir", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    elif args.all:
        symbols = discover_symbols_from_data(DATA_ROOTS)
    else:
        symbols = load_active_symbols()

    timeframes = [tf.strip().upper() for tf in args.timeframes.split(",")]

    if not symbols:
        raise SystemExit("No symbols found")

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"models/optuna_results/exit_{ts}"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load per-symbol thresholds from sovereign_configs
    sym_thresholds = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _cfgs = json.load(f)
        for s, c in _cfgs.items():
            if "prob_threshold" in c:
                sym_thresholds[s] = c["prob_threshold"]

    args_dict = {
        "data_roots": DATA_ROOTS,
        "z_threshold": args.z_threshold,
        "prob_threshold": args.prob_threshold,
        "sym_thresholds": sym_thresholds,
        "trials": args.trials,
        "purge": args.purge,
        "embargo": args.embargo,
        "out_dir": out_dir,
        "min_rrr": args.min_rrr,
        "use_sizing": args.use_sizing,
        "account_size": args.account_size,
        "max_years": args.max_years,
    }

    # Build task list: symbol × timeframe
    tasks = [(sym, tf, args_dict) for sym in symbols for tf in timeframes]

    print(f"[exit-optuna] {len(symbols)} symbols × {len(timeframes)} TFs = "
          f"{len(tasks)} jobs | {args.trials} trials/job | {args.workers} workers")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Output: {out_dir}")
    for sym in symbols:
        model = find_model_path(sym)
        status = "model" if model else "z-score"
        print(f"  {sym:20s} -> {cluster_for_symbol(sym):9s}  [{status}]")
    print()

    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_worker, task): task for task in tasks
        }
        for future in as_completed(futures):
            sym, tf, _ = futures[future]
            done += 1
            try:
                r = future.result()
                status = r.get("status", "unknown")
                cluster = r.get("cluster", "?")
                if status == "ok":
                    risk_str = ""
                    if "best_risk_pct" in r:
                        risk_str = f"  risk={r['best_risk_pct']*100:.1f}%"
                    print(
                        f"  [{done}/{len(tasks)}] ✓ {sym:15s} {tf:4s} "
                        f"[{cluster:9s}] EV={r['best_ev']:+.6f}  "
                        f"SL={r['best_atr_sl_mult']:.2f} TP={r['best_atr_tp_mult']:.2f} "
                        f"BE={r['best_breakeven_atr']:.2f} "
                        f"trail={r['best_trail_activation_atr']:.2f}/{r['best_trail_distance_atr']:.2f} "
                        f"H={r['best_horizon']}  "
                        f"cost={r['cost_bps']:.0f}bp{risk_str}  "
                        f"signals={r['n_signals']}  folds={r['n_folds']}"
                    )
                else:
                    print(f"  [{done}/{len(tasks)}] - {sym:15s} {tf:4s} "
                          f"[{cluster:9s}] {status}")
            except Exception as e:
                r = {"symbol": sym, "timeframe": tf, "status": "error",
                     "error": str(e)}
                print(f"  [{done}/{len(tasks)}] ✗ {sym:15s} {tf:4s} ERROR: {e}")
            rows.append(r)

    # Save all results
    out_df = pl.from_dicts(rows).sort(["symbol", "timeframe"])
    csv_path = os.path.join(out_dir, "exit_summary.csv")
    out_df.write_csv(csv_path)
    print(f"\n[exit-optuna] Saved: {csv_path}")

    # Print best TF per symbol
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    best_per_sym: dict[str, dict] = {}
    for r in ok_rows:
        sym = r["symbol"]
        if sym not in best_per_sym or r["best_ev"] > best_per_sym[sym]["best_ev"]:
            best_per_sym[sym] = r

    if best_per_sym:
        print(f"\n  Best TF per symbol:")
        for sym in sorted(best_per_sym):
            r = best_per_sym[sym]
            print(f"    {sym:20s} → {r['timeframe']:4s}  EV={r['best_ev']:+.6f}")

    # Update configs if requested
    if args.update_configs:
        update_sovereign_configs(rows)

    print(f"\n[exit-optuna] Done — {len(ok_rows)}/{len(tasks)} jobs succeeded, "
          f"{len(best_per_sym)} symbols optimized")


if __name__ == "__main__":
    main()
