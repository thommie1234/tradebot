"""
Unified Walk-Forward Out-of-Sample Validation — replaces 6 WFO scripts.

Takes Optuna entry params + exit params → trains XGBoost per fold → runs
exit_simulator.simulate_trades() on OOS → produces per-symbol metrics and
a filtered portfolio.

Usage:
    python3 common/research/wfo_validate.py \
        --account ftmo_100k \
        --entry-csv ftmo/optuna/ritual_combined_20260224_enriched.csv \
        --exit-csv ftmo/optuna/exit_sizing_combined/exit_best_per_symbol.csv \
        --out-dir ftmo/optuna/wfo_validated_TIMESTAMP \
        [--symbols NVDA,TSLA] [--workers 6] [--account-size 100000]

3-stage pipeline position: Entry Optuna → Exit Optuna → **WFO Validate**
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import xgboost as xgb

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from engine.labeling import apply_triple_barrier
from research.cost_model import get_cost_model
from research.exit_simulator import ExitParams, simulate_trades
from research.integrated_pipeline import (
    make_ev_custom_objective,
    purged_walk_forward_splits,
    set_polars_threads,
    tesla_p40_xgb_params,
)
from research.optuna_orchestrator import (
    DATA_ROOTS,
    cluster_for_symbol,
    load_symbol_ticks_lf,
)
from research.train_ml_strategy import (
    infer_spread_bps,
    make_time_bars,
    sanitize_training_frame,
)

# ---------------------------------------------------------------------------
# WFO sizes per timeframe
# ---------------------------------------------------------------------------

TF_WF_SIZES = {
    "M1":  {"train_size": 10000, "test_size": 2500},
    "M5":  {"train_size": 4800,  "test_size": 1200},
    "M15": {"train_size": 4800,  "test_size": 1200},
    "M30": {"train_size": 3000,  "test_size": 800},
    "H1":  {"train_size": 1000,  "test_size": 300},
    "H4":  {"train_size": 250,   "test_size": 80},
}

TF_BARS_PER_YEAR = {
    "M1": 525600, "M5": 105120, "M15": 35040, "M30": 17520,
    "H1": 8760, "H4": 2190, "D1": 365,
}

# ---------------------------------------------------------------------------
# Correlation groups & trading sessions (from wfo_portfolio_backtest.py)
# ---------------------------------------------------------------------------

CORRELATION_GROUPS = {
    "us_tech":   ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"],
    "eu_equity": ["ALVG", "BAYGn", "DBKGn", "LVMH", "RACE"],
    "us_idx":    ["US100.cash", "US30.cash", "US500.cash", "US2000.cash"],
    "eu_idx":    ["EU50.cash", "FRA40.cash", "GER40.cash", "SPN35.cash", "UK100.cash"],
    "apac_idx":  ["AUS200.cash", "HK50.cash", "JP225.cash"],
    "jpy_fx":    ["AUD_JPY", "CAD_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "USD_JPY"],
    "usd_fx":    ["AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD"],
    "gold":      ["XAU_AUD", "XAU_EUR", "XAU_USD"],
    "silver":    ["XAG_AUD", "XAG_EUR", "XAG_USD"],
    "oil":       ["UKOIL.cash", "USOIL.cash"],
    "defi":      ["AAVUSD", "UNIUSD", "LNKUSD"],
    "alt_crypto": ["ALGUSD", "DASHUSD", "ETCUSD", "FETUSD", "GALUSD",
                   "GRTUSD", "ICPUSD", "IMXUSD", "MANUSD", "NERUSD",
                   "SOLUSD", "VECUSD", "XLMUSD", "BARUSD", "SANUSD",
                   "AVAUSD", "BNBUSD"],
}

TRADING_SESSIONS = {
    "crypto":    {"open": 0, "close": 24, "days": "Mon-Sun"},
    "forex":     {"open": 0, "close": 24, "days": "Mon-Fri"},
    "metals":    {"open": 1, "close": 24, "days": "Mon-Fri"},
    "us_equity": {"open": 16, "close": 23, "days": "Mon-Fri"},
    "eu_equity": {"open": 10, "close": 18, "days": "Mon-Fri"},
    "us_idx":    {"open": 1, "close": 24, "days": "Mon-Fri"},
    "eu_idx":    {"open": 9, "close": 23, "days": "Mon-Fri"},
    "apac_idx":  {"open": 1, "close": 23, "days": "Mon-Fri"},
    "oil":       {"open": 1, "close": 24, "days": "Mon-Fri"},
    "agri":      {"open": 12, "close": 20, "days": "Mon-Fri"},
}


def _session_for_cluster(cluster: str) -> str:
    """Map cluster name to trading session key."""
    return {
        "crypto": "crypto",
        "forex": "forex",
        "commodity": "agri",
        "index": "us_idx",
        "equity": "us_equity",
    }.get(cluster, "forex")


# ---------------------------------------------------------------------------
# Per-symbol WFO validation
# ---------------------------------------------------------------------------

def validate_symbol(symbol: str, args_dict: dict) -> dict:
    """Full WFO validation for one symbol:
    1. Load tick data → time bars
    2. Build features + triple-barrier labels
    3. Purged walk-forward splits
    4. Per fold: XGBoost train with Optuna params → predict OOS → exit simulator
    5. Aggregate metrics across folds
    """
    set_polars_threads(4)

    entry_row = args_dict["entry_rows"].get(symbol)
    exit_row = args_dict["exit_rows"].get(symbol)
    account = args_dict["account"]
    cm = get_cost_model(account)

    if entry_row is None:
        return {"symbol": symbol, "status": "no_entry_params"}
    if exit_row is None:
        return {"symbol": symbol, "status": "no_exit_params"}

    cluster = cluster_for_symbol(symbol)
    session = _session_for_cluster(cluster)
    timeframe = entry_row.get("timeframe", "H1")

    # --- 1. Load tick data → time bars ---
    roots = list(args_dict["data_roots"])
    ticks_lf = load_symbol_ticks_lf(symbol, roots)
    if ticks_lf is None:
        return {"symbol": symbol, "status": "no_data", "cluster": cluster}

    ticks_df = ticks_lf.collect()
    if ticks_df.height == 0:
        return {"symbol": symbol, "status": "no_data", "cluster": cluster}

    if ticks_df.select(pl.col("size").sum()).item() <= 0:
        ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks_df)
    bars = make_time_bars(
        ticks_df.select(["time", "price", "size"]), timeframe,
    )
    del ticks_df

    if bars.height < 200:
        return {"symbol": symbol, "status": "insufficient_bars", "cluster": cluster}

    # Ensure time is datetime
    if bars["time"].dtype == pl.Int64:
        bars = bars.with_columns(
            pl.from_epoch(pl.col("time"), time_unit="s").alias("time")
        )

    # --- 2. Features + labels ---
    z_threshold = args_dict.get("z_threshold", 1.0)
    feat = build_bar_features(bars, z_threshold=z_threshold)

    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=24,
        pt_mult=2.0,
        sl_mult=1.5,
    )
    feat = feat.with_columns([
        pl.Series("label", tb.label),
        pl.Series("target", tb.label),
        pl.Series("tb_ret", tb.tb_ret),
        pl.Series("upside", tb.upside),
        pl.Series("downside", tb.downside),
        pl.Series("avg_win", tb.upside),
        pl.Series("avg_loss", tb.downside),
    ]).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    # Cost from account's cost model
    fee_bps = cm.commission_bps(symbol)
    slippage_bps_val = cm.slippage_bps(symbol)
    if spread_bps is None or spread_bps <= 0:
        spread_bps = fee_bps + slippage_bps_val
    cost_pct = (fee_bps + spread_bps + slippage_bps_val * 2.0) / 1e4

    # --- 3. Purged walk-forward splits ---
    wf = TF_WF_SIZES.get(timeframe, TF_WF_SIZES["H1"])
    purge = args_dict.get("purge", 8)
    embargo = args_dict.get("embargo", 8)

    # Dynamic sizing
    available = feat.height - (purge + embargo)
    if available < 200:
        return {"symbol": symbol, "status": "insufficient_rows",
                "rows": feat.height, "cluster": cluster}

    train_size = min(wf["train_size"], int(available * 0.8))
    test_size = min(wf["test_size"], available - train_size)
    train_size = max(200, train_size)
    test_size = max(50, test_size)

    splits = purged_walk_forward_splits(
        n_samples=feat.height,
        train_size=train_size,
        test_size=test_size,
        purge=purge,
        embargo=embargo,
    )
    if not splits:
        return {"symbol": symbol, "status": "no_folds",
                "rows": feat.height, "cluster": cluster}

    # --- 4. Prepare XGBoost params from entry CSV ---
    xgb_params = tesla_p40_xgb_params()
    try:
        # GPU probe
        probe = xgb.DMatrix(
            np.array([[0.0], [1.0]]), label=np.array([0.0, 1.0]),
        )
        xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"},
            probe, num_boost_round=1,
        )
    except Exception:
        xgb_params["device"] = "cpu"
        xgb_params["sampling_method"] = "uniform"

    # Override with Optuna best params
    for param in ("max_depth", "gamma", "subsample", "colsample_bytree",
                   "reg_alpha", "reg_lambda", "min_child_weight", "eta",
                   "colsample_bylevel"):
        key = f"best_{param}"
        if key in entry_row:
            val = entry_row[key]
            if param == "max_depth":
                xgb_params[param] = int(val)
            else:
                xgb_params[param] = float(val)

    num_boost_round = int(entry_row.get("best_num_boost_round", 500))

    # Exit params
    ep = ExitParams(
        atr_sl_mult=float(exit_row["best_atr_sl_mult"]),
        atr_tp_mult=float(exit_row["best_atr_tp_mult"]),
        breakeven_atr=float(exit_row["best_breakeven_atr"]),
        trail_activation_atr=float(exit_row["best_trail_activation_atr"]),
        trail_distance_atr=float(exit_row["best_trail_distance_atr"]),
        horizon=int(exit_row["best_horizon"]),
    )
    risk_pct = float(exit_row.get("best_risk_pct", 0.005))

    # Compute ATR14 from OHLC (not exported by build_bar_features)
    _high = feat["high"].to_numpy()
    _low = feat["low"].to_numpy()
    _close = feat["close"].to_numpy()
    _n = len(_close)
    _tr = np.empty(_n, dtype=np.float64)
    _tr[0] = _high[0] - _low[0]
    for _i in range(1, _n):
        _tr[_i] = max(_high[_i] - _low[_i],
                      abs(_high[_i] - _close[_i - 1]),
                      abs(_low[_i] - _close[_i - 1]))
    _atr14 = np.full(_n, np.nan, dtype=np.float64)
    for _i in range(13, _n):
        _atr14[_i] = np.mean(_tr[_i - 13 : _i + 1])
    feat = feat.with_columns(pl.Series("atr14", _atr14))

    # Prepare numpy arrays
    feature_cols = FEATURE_COLUMNS
    extra = ["target", "avg_win", "avg_loss",
             "close", "open", "high", "low", "atr14"]
    cols = list(dict.fromkeys(feature_cols + extra))  # deduplicate, preserve order
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        return {"symbol": symbol, "status": f"missing_cols:{','.join(missing)}",
                "cluster": cluster}

    df = feat.select(cols).drop_nulls(feature_cols + ["target"])
    x_all = df.select(feature_cols).to_numpy().astype(np.float32)
    y_all = df["target"].to_numpy().astype(np.float32)
    avg_win_all = df["avg_win"].to_numpy().astype(np.float64)
    avg_loss_all = df["avg_loss"].to_numpy().astype(np.float64)

    open_arr = df["open"].to_numpy().astype(np.float64)
    high_arr = df["high"].to_numpy().astype(np.float64)
    low_arr = df["low"].to_numpy().astype(np.float64)
    close_arr = df["close"].to_numpy().astype(np.float64)
    atr_arr = df["atr14"].to_numpy().astype(np.float64)
    primary_side = df["primary_side"].to_numpy().astype(np.int32)

    # --- 5. Per-fold: train XGBoost → predict OOS → exit simulator ---
    threshold_sweep = args_dict.get("threshold_sweep", False)
    sweep_thresholds = args_dict.get("sweep_thresholds", [])
    default_threshold = args_dict.get("prob_threshold", 0.55)

    if threshold_sweep and sweep_thresholds:
        thresholds_to_test = sweep_thresholds
    else:
        thresholds_to_test = [default_threshold]

    # Collect per-fold OOS predictions + sides (train once, sweep later)
    fold_oos_data = []  # list of (te_idx, probas, oos_side)

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        x_train, y_train = x_all[tr_idx], y_all[tr_idx]
        aw_train = avg_win_all[tr_idx]
        al_train = avg_loss_all[tr_idx]
        fold_cost = float(cost_pct)

        obj = make_ev_custom_objective(
            float(np.mean(aw_train)),
            float(np.mean(al_train)),
            fold_cost,
        )

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_all[te_idx], label=y_all[te_idx])

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=[(dtest, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        probas = bst.predict(dtest)
        del bst
        oos_side = primary_side[te_idx]
        fold_oos_data.append((te_idx, probas, oos_side))

    gc.collect()

    # --- Sweep thresholds on collected OOS predictions ---
    sweep_results = []

    for prob_threshold in thresholds_to_test:
        all_pnl = []
        all_bars = []
        all_exits = []
        all_entry_bars = []

        for te_idx, probas, oos_side in fold_oos_data:
            signal_mask = oos_side != 0
            if probas.std() > 0.01:
                model_ok = (probas >= prob_threshold) | (probas <= (1.0 - prob_threshold))
                signal_mask = signal_mask & model_ok

            oos_entry_indices = te_idx[signal_mask]
            oos_directions = oos_side[signal_mask]

            if len(oos_entry_indices) < 2:
                continue

            pnl, bars_h, exits = simulate_trades(
                oos_entry_indices, oos_directions,
                open_arr, high_arr, low_arr, close_arr, atr_arr,
                ep, cost_pct,
            )

            if len(pnl) == 0:
                continue

            all_pnl.append(pnl)
            all_bars.append(bars_h)
            all_exits.append(exits)
            all_entry_bars.append(oos_entry_indices[:len(pnl)])

        if not all_pnl:
            if len(thresholds_to_test) == 1:
                return {"symbol": symbol, "status": "no_trades", "cluster": cluster,
                        "timeframe": timeframe, "n_folds": len(splits)}
            continue

        sweep_results.append((prob_threshold, all_pnl, all_bars, all_exits, all_entry_bars))

    if not sweep_results:
        return {"symbol": symbol, "status": "no_trades", "cluster": cluster,
                "timeframe": timeframe, "n_folds": len(splits)}

    # --- 6. Aggregate metrics across folds per threshold, pick best ---
    def _compute_metrics(prob_thr, all_pnl, all_bars, all_exits, all_entry_bars):
        pnl = np.concatenate(all_pnl)
        bars_held = np.concatenate(all_bars)
        exit_types = np.concatenate(all_exits)
        entry_bar_idxs = np.concatenate(all_entry_bars)
        n_trades = len(pnl)
        if n_trades == 0:
            return None

        account_size = args_dict.get("account_size", 100_000.0)
        sl_fracs = atr_arr[entry_bar_idxs] * ep.atr_sl_mult / close_arr[entry_bar_idxs]
        sl_fracs = np.clip(sl_fracs, 1e-6, None)
        risk_amount = account_size * risk_pct
        dollar_pnl = risk_amount * pnl / sl_fracs[:len(pnl)]

        equity = np.cumsum(dollar_pnl)
        peak = np.maximum.accumulate(np.concatenate([[0], equity]))
        dd = peak[1:] - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        total_return = float(equity[-1]) if len(equity) > 0 else 0.0

        wins = int(np.sum(pnl > 0))
        losses = int(np.sum(pnl < 0))
        win_rate = float(wins / n_trades)

        gross_profit = float(np.sum(dollar_pnl[dollar_pnl > 0]))
        gross_loss = float(np.abs(np.sum(dollar_pnl[dollar_pnl < 0])))
        profit_factor = gross_profit / max(gross_loss, 1.0)

        bpy = TF_BARS_PER_YEAR.get(timeframe, 8760)
        avg_bh = float(np.mean(bars_held)) if len(bars_held) > 0 else 1.0
        tpy = bpy / max(avg_bh, 1.0)
        sharpe = float(
            (np.mean(dollar_pnl) / np.std(dollar_pnl)) * np.sqrt(tpy)
        ) if np.std(dollar_pnl) > 0 else 0.0

        calmar = total_return / max(max_dd, 1.0)
        dd_pct = max_dd / account_size * 100
        ftmo_ok = dd_pct < 10.0

        n_sl = int(np.sum(exit_types == 0))
        n_tp = int(np.sum(exit_types == 1))
        n_be = int(np.sum(exit_types == 2))
        n_trail = int(np.sum(exit_types == 3))
        n_horizon = int(np.sum(exit_types == 4))

        return {
            "symbol": symbol,
            "status": "ok",
            "cluster": cluster,
            "session": session,
            "timeframe": timeframe,
            "prob_threshold": round(prob_thr, 3),
            "n_folds": len(splits),
            "n_trades": n_trades,
            "total_return": round(total_return, 2),
            "max_dd": round(max_dd, 2),
            "dd_pct": round(dd_pct, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "profit_factor": round(profit_factor, 2),
            "win_rate": round(win_rate * 100, 1),
            "wins": wins,
            "losses": losses,
            "avg_bars_held": round(avg_bh, 1),
            "ftmo_ok": ftmo_ok,
            "risk_pct": round(risk_pct * 100, 2),
            "fee_bps": round(fee_bps, 1),
            "spread_bps": round(spread_bps, 2),
            "slippage_bps": round(slippage_bps_val, 1),
            "cost_pct": round(cost_pct * 100, 4),
            "sl": ep.atr_sl_mult,
            "tp": ep.atr_tp_mult,
            "be": ep.breakeven_atr,
            "trail_act": ep.trail_activation_atr,
            "trail_dist": ep.trail_distance_atr,
            "horizon": ep.horizon,
            "pct_sl": round(n_sl / n_trades * 100, 1),
            "pct_tp": round(n_tp / n_trades * 100, 1),
            "pct_be": round(n_be / n_trades * 100, 1),
            "pct_trail": round(n_trail / n_trades * 100, 1),
            "pct_horizon": round(n_horizon / n_trades * 100, 1),
            "account_id": args_dict["account"],
        }

    # Compute metrics for each threshold
    all_threshold_results = []
    for prob_thr, a_pnl, a_bars, a_exits, a_eb in sweep_results:
        m = _compute_metrics(prob_thr, a_pnl, a_bars, a_exits, a_eb)
        if m is not None:
            all_threshold_results.append(m)

    if not all_threshold_results:
        return {"symbol": symbol, "status": "no_trades", "cluster": cluster,
                "timeframe": timeframe}

    # If threshold sweep: return all results + pick best by Calmar
    if threshold_sweep and len(all_threshold_results) > 1:
        best = max(all_threshold_results, key=lambda r: r.get("calmar", -999))
        best["best_of_sweep"] = True
        best["sweep_results"] = all_threshold_results
        return best

    return all_threshold_results[0]


# ---------------------------------------------------------------------------
# Portfolio selection
# ---------------------------------------------------------------------------

def build_portfolio(
    results: list[dict],
    max_corr_per_group: int = 2,
    min_sharpe: float = 0.5,
    max_dd_pct: float = 10.0,
) -> dict:
    """Select diversified portfolio with 24h session coverage."""
    ok = [
        r for r in results
        if r.get("status") == "ok"
        and r.get("total_return", 0) > 0
        and r.get("ftmo_ok", False)
        and r.get("sharpe", 0) >= min_sharpe
        and r.get("dd_pct", 100) <= max_dd_pct
    ]
    if not ok:
        return {"portfolio": [], "sessions": {}}

    ok.sort(key=lambda x: x.get("calmar", 0), reverse=True)

    selected = []
    group_counts: dict[str, int] = {}

    for r in ok:
        sym = r["symbol"]
        blocked = False
        for grp_name, grp_syms in CORRELATION_GROUPS.items():
            if sym in grp_syms:
                if group_counts.get(grp_name, 0) >= max_corr_per_group:
                    blocked = True
                    break
        if blocked:
            continue

        selected.append(r)
        for grp_name, grp_syms in CORRELATION_GROUPS.items():
            if sym in grp_syms:
                group_counts[grp_name] = group_counts.get(grp_name, 0) + 1

    # Session coverage
    sessions_covered: dict[str, list[str]] = {
        "asia_night (00-08)": [],
        "europe (08-16)": [],
        "us (16-23)": [],
        "weekend": [],
    }
    for r in selected:
        sess = r.get("session", "unknown")
        sym = r["symbol"]
        sess_info = TRADING_SESSIONS.get(sess, {})
        o = sess_info.get("open", 0)
        c = sess_info.get("close", 24)
        days = sess_info.get("days", "Mon-Fri")
        if o <= 8:
            sessions_covered["asia_night (00-08)"].append(sym)
        if o <= 16 and c >= 8:
            sessions_covered["europe (08-16)"].append(sym)
        if c >= 16:
            sessions_covered["us (16-23)"].append(sym)
        if "Sun" in days or "Sat" in days:
            sessions_covered["weekend"].append(sym)

    return {"portfolio": selected, "sessions": sessions_covered}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified WFO validation (entry Optuna + exit Optuna → OOS backtest)",
    )
    p.add_argument("--account", type=str, required=True,
                   help="Account ID for cost model (e.g. ftmo_100k, bright_100k)")
    p.add_argument("--entry-csv", type=str, required=True,
                   help="Entry Optuna results CSV (from optuna_orchestrator.py)")
    p.add_argument("--exit-csv", type=str, default="",
                   help="Exit Optuna results CSV (from exit_optuna.py)")
    p.add_argument("--out-dir", type=str, default="",
                   help="Output directory (default: timestamped)")
    p.add_argument("--symbols", type=str, default="",
                   help="Comma-separated symbol filter (default: all in entry CSV)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--account-size", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.005,
                   help="Default risk per trade as fraction (overridden by exit CSV)")
    p.add_argument("--min-sharpe", type=float, default=0.5,
                   help="Minimum Sharpe for portfolio inclusion")
    p.add_argument("--max-dd-pct", type=float, default=10.0,
                   help="Maximum drawdown %% for portfolio inclusion")
    p.add_argument("--max-corr", type=int, default=2,
                   help="Max symbols per correlation group")
    p.add_argument("--prob-threshold", type=float, default=0.55)
    p.add_argument("--threshold-sweep", action="store_true",
                   help="Sweep probability thresholds (0.50-0.70 step 0.02) per symbol")
    p.add_argument("--sweep-min", type=float, default=0.50,
                   help="Min threshold for sweep (default 0.50)")
    p.add_argument("--sweep-max", type=float, default=0.70,
                   help="Max threshold for sweep (default 0.70)")
    p.add_argument("--sweep-step", type=float, default=0.02,
                   help="Threshold step for sweep (default 0.02)")
    p.add_argument("--purge", type=int, default=8)
    p.add_argument("--embargo", type=int, default=8)
    p.add_argument("--z-threshold", type=float, default=1.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load entry CSV
    entry_df = pl.read_csv(args.entry_csv)
    entry_df = entry_df.filter(pl.col("status") == "ok")
    entry_rows = {row["symbol"]: row for row in entry_df.iter_rows(named=True)}

    # Load exit CSV
    exit_rows: dict[str, dict] = {}
    if args.exit_csv:
        exit_df = pl.read_csv(args.exit_csv)
        exit_df = exit_df.filter(pl.col("status") == "ok")
        exit_rows = {row["symbol"]: row for row in exit_df.iter_rows(named=True)}

    # Symbol selection
    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    else:
        symbols = sorted(entry_rows.keys())

    # Filter to symbols that have both entry and exit params
    if exit_rows:
        symbols = [s for s in symbols if s in entry_rows and s in exit_rows]
    else:
        symbols = [s for s in symbols if s in entry_rows]

    if not symbols:
        raise SystemExit("No symbols found with matching entry+exit params")

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"wfo_validated_{ts}"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build threshold sweep list
    if args.threshold_sweep:
        sweep_thresholds = list(np.round(
            np.arange(args.sweep_min, args.sweep_max + 0.001, args.sweep_step), 3
        ))
    else:
        sweep_thresholds = []

    args_dict = {
        "data_roots": DATA_ROOTS,
        "account": args.account,
        "account_size": args.account_size,
        "risk": args.risk,
        "prob_threshold": args.prob_threshold,
        "threshold_sweep": args.threshold_sweep,
        "sweep_thresholds": sweep_thresholds,
        "purge": args.purge,
        "embargo": args.embargo,
        "z_threshold": args.z_threshold,
        "entry_rows": entry_rows,
        "exit_rows": exit_rows,
    }

    sweep_info = (
        f" | threshold sweep {args.sweep_min:.2f}→{args.sweep_max:.2f} "
        f"(step {args.sweep_step:.2f}, {len(sweep_thresholds)} levels)"
        if args.threshold_sweep else f" | threshold={args.prob_threshold}"
    )
    print(f"[wfo-validate] {len(symbols)} symbols | account={args.account} | "
          f"{args.workers} workers{sweep_info}")
    print(f"  Entry CSV: {args.entry_csv}")
    print(f"  Exit CSV:  {args.exit_csv or '(none)'}")
    print(f"  Output:    {out_dir}")
    print()

    # Run validation
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_symbol, sym, args_dict): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                r = future.result()
                status = r.get("status", "unknown")
                if status == "ok":
                    thr_info = (
                        f"  thr={r.get('prob_threshold', 0):.2f}"
                        if args.threshold_sweep else ""
                    )
                    print(
                        f"  OK  {sym:20s} [{r['cluster']:9s}] "
                        f"${r['total_return']:>9,.0f}  "
                        f"DD={r['dd_pct']:5.1f}%  "
                        f"Sharpe={r['sharpe']:5.2f}  "
                        f"Calmar={r['calmar']:6.1f}  "
                        f"PF={r['profit_factor']:5.2f}  "
                        f"WR={r['win_rate']:4.1f}%  "
                        f"N={r['n_trades']}{thr_info}"
                    )
                else:
                    print(f"  --  {sym:20s} {status}")
            except Exception as e:
                r = {"symbol": sym, "status": "error", "error": str(e)}
                print(f"  ERR {sym:20s} {e}")
            results.append(r)

    # Save raw results
    ok_results = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*100}")
    print(f"Results: {len(ok_results)}/{len(results)} symbols completed successfully")

    # Collect and save sweep details if threshold sweep was enabled
    if args.threshold_sweep:
        all_sweep = []
        for r in ok_results:
            sweep_data = r.pop("sweep_results", None)
            r.pop("best_of_sweep", None)
            if sweep_data:
                all_sweep.extend(sweep_data)
        if all_sweep:
            sweep_df = pl.from_dicts(all_sweep).sort(["symbol", "prob_threshold"])
            sweep_csv = os.path.join(out_dir, "threshold_sweep.csv")
            sweep_df.write_csv(sweep_csv)
            print(f"Saved threshold sweep: {sweep_csv} ({len(all_sweep)} rows)")

    if ok_results:
        # Remove non-serializable fields before saving
        save_results = []
        for r in ok_results:
            r2 = {k: v for k, v in r.items()
                  if k not in ("sweep_results", "best_of_sweep")}
            save_results.append(r2)
        raw_df = pl.from_dicts(save_results).sort("calmar", descending=True)
        raw_csv = os.path.join(out_dir, "wfo_results.csv")
        raw_df.write_csv(raw_csv)
        print(f"Saved: {raw_csv}")

    # Build portfolio
    portfolio = build_portfolio(
        results,
        max_corr_per_group=args.max_corr,
        min_sharpe=args.min_sharpe,
        max_dd_pct=args.max_dd_pct,
    )
    selected = portfolio["portfolio"]
    sessions = portfolio["sessions"]

    if selected:
        # Print portfolio
        total_return = 0
        total_trades = 0
        thr_col = " {'Thr':>5s}" if args.threshold_sweep else ""
        print(f"\n{'SELECTED PORTFOLIO':=^110}")
        print(f"{'#':>3}  {'Symbol':20s} {'Session':15s} {'TF':4s} "
              f"{'Return':>10s} {'MaxDD%':>7s} {'Sharpe':>7s} {'Calmar':>8s} "
              f"{'PF':>6s} {'WR%':>5s} {'Trades':>6s}"
              + (f" {'Thr':>5s}" if args.threshold_sweep else ""))
        print("-" * 110)

        for i, r in enumerate(selected, 1):
            total_return += r["total_return"]
            total_trades += r["n_trades"]
            thr_str = f" {r.get('prob_threshold', 0.55):5.2f}" if args.threshold_sweep else ""
            print(
                f"{i:3d}  {r['symbol']:20s} {r['session']:15s} {r['timeframe']:4s} "
                f"${r['total_return']:>9,.0f} {r['dd_pct']:6.1f}% "
                f"{r['sharpe']:6.2f} {r['calmar']:7.1f} "
                f"{r['profit_factor']:5.2f} {r['win_rate']:4.1f}% "
                f"{r['n_trades']:5d}{thr_str}"
            )

        print("-" * 100)
        print(f"     TOTAAL: ${total_return:>9,.0f}  |  "
              f"{total_trades} trades  |  {len(selected)} symbols")

        # Session coverage
        print(f"\n{'SESSION COVERAGE':=^80}")
        for sess, syms in sessions.items():
            print(f"  {sess:25s}: {len(syms):2d} symbols — {', '.join(syms[:8])}"
                  + (f" +{len(syms)-8} more" if len(syms) > 8 else ""))

        # Correlation group usage
        print(f"\n{'CORRELATION GROUPS':=^80}")
        for grp_name, grp_syms in CORRELATION_GROUPS.items():
            in_portfolio = [s for s in grp_syms
                           if any(r["symbol"] == s for r in selected)]
            if in_portfolio:
                print(f"  {grp_name:15s}: {len(in_portfolio)}/{args.max_corr} "
                      f"— {', '.join(in_portfolio)}")

        # Save portfolio CSV
        port_df = pl.from_dicts(selected)
        port_csv = os.path.join(out_dir, "portfolio_selected.csv")
        port_df.write_csv(port_csv)
        print(f"\nSaved: {port_csv}")

        # Save portfolio JSON config
        port_config = {}
        for r in selected:
            sym = r["symbol"]
            port_config[sym] = {
                "asset_class": r["cluster"],
                "sector": r["session"],
                "atr_period": 14,
                "atr_timeframe": "H1",
                "atr_sl_mult": r["sl"],
                "atr_tp_mult": r["tp"],
                "breakeven_atr": r["be"],
                "trail_activation_atr": r["trail_act"],
                "trail_distance_atr": r["trail_dist"],
                "exit_horizon": r["horizon"],
                "exit_timeframe": r["timeframe"],
                "risk_per_trade": round(r["risk_pct"] / 100, 4),
                "max_spread_pct": (
                    0.005 if r["cluster"] == "crypto"
                    else 0.001 if r["cluster"] == "equity"
                    else 0.0005
                ),
                "magic_number": 2000,
                "wfo_sharpe": r["sharpe"],
                "wfo_calmar": r["calmar"],
                "wfo_profit_factor": r["profit_factor"],
                "wfo_total_return": r["total_return"],
                "wfo_max_dd_pct": r["dd_pct"],
                "wfo_n_trades": r["n_trades"],
                "prob_threshold": r.get("prob_threshold", 0.55),
            }

        config_path = os.path.join(out_dir, "portfolio_configs.json")
        with open(config_path, "w") as f:
            json.dump(port_config, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved: {config_path}")
    else:
        print("\nNo symbols passed portfolio filters.")

    positive = sum(1 for r in ok_results if r.get("total_return", 0) > 0)
    negative = sum(1 for r in ok_results if r.get("total_return", 0) <= 0)
    print(f"\n[wfo-validate] Done — {len(ok_results)} backtested, "
          f"{positive} profitable, {negative} unprofitable, "
          f"{len(selected)} in portfolio")


if __name__ == "__main__":
    main()
