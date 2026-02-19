#!/usr/bin/env python3
"""
Institutional-grade portfolio analysis for the 8-symbol sovereign fleet.

Answers:
  1. Total trades + trades/year
  2. Exact CAGR (compounded)
  3. Long vs short breakdown (count + PnL)
  4. Buy & hold comparison per symbol
  5. Cost sensitivity at 1.5× costs
  6. Worst rolling 12-month return
  7. Per-year regime analysis
  8. Drawdown duration

Usage:
    python3 research/institutional_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from trading_prop.ml.integrated_pipeline import (
    purged_walk_forward_splits,
    set_polars_threads,
)
from trading_prop.ml.train_ml_strategy import infer_spread_bps, make_time_bars
from trading_prop.production.optuna_orchestrator import (
    DATA_ROOTS,
    cluster_for_symbol,
    load_symbol_ticks_lf,
)
from research.exit_simulator import ExitParams, simulate_trades
from research.wfo_portfolio_backtest import (
    TF_WF_SIZES,
    compute_atr14,
    find_model_path,
    ftmo_cost_pct,
)

import argparse

ACCOUNT = 100_000.0

# ── Load configs ─────────────────────────────────────────────────────────
CONFIG_PATH = REPO_ROOT / "config" / "sovereign_configs.json"
DEFAULT_EXIT_CSV = REPO_ROOT / "models" / "optuna_results" / "exit_all_20260215_203654" / "exit_active_8.csv"


def reconstruct_trade_entries(
    entry_indices: np.ndarray,
    bars_held: np.ndarray,
    open_arr: np.ndarray,
    atr_arr: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    """Mirror simulate_trades skip logic to find which entries became trades.

    Returns array of length len(bars_held) with the TRUE entry index per trade.
    """
    result = []
    position_end = -1
    trade_count = 0
    n_trades = len(bars_held)

    for i in range(len(entry_indices)):
        if trade_count >= n_trades:
            break
        idx = int(entry_indices[i])
        if idx <= position_end:
            continue
        entry_bar = idx + 1
        if entry_bar >= n_bars:
            continue
        entry_px = open_arr[entry_bar]
        if entry_px <= 0 or not np.isfinite(entry_px):
            continue
        atr = atr_arr[idx]
        if atr <= 0 or not np.isfinite(atr):
            continue

        # This entry became trade #trade_count
        result.append(idx)
        exit_bar = entry_bar + int(bars_held[trade_count])
        position_end = exit_bar
        trade_count += 1

    return np.array(result, dtype=np.int64)


def analyse_symbol(symbol: str, cfg: dict, exit_row: dict) -> dict | None:
    """Full WFO analysis for one symbol with proper trade-level detail."""
    import xgboost as xgb
    set_polars_threads(2)

    cluster = cluster_for_symbol(symbol)
    timeframe = exit_row["timeframe"]
    risk_pct = cfg["risk_per_trade"]

    # 1. Load data
    ticks_lf = load_symbol_ticks_lf(symbol, list(DATA_ROOTS))
    if ticks_lf is None:
        print(f"  {symbol}: no data")
        return None
    ticks_df = ticks_lf.collect()
    if ticks_df.height == 0:
        return None
    if ticks_df.select(pl.col("size").sum()).item() <= 0:
        ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks_df.select(["time", "price", "size"]), timeframe)
    feat = build_bar_features(bars, z_threshold=1.0)
    if feat.height < 500:
        print(f"  {symbol}: only {feat.height} bars")
        return None
    feat_clean = feat.drop_nulls(FEATURE_COLUMNS)

    # Timestamps
    time_arr = feat_clean["time"].to_numpy()

    # 2. Entry signals + ML filter
    primary_side = feat_clean["primary_side"].to_numpy().astype(np.int32)
    signal_mask = primary_side != 0
    entry_indices = np.where(signal_mask)[0]
    directions = primary_side[signal_mask]

    model_path = find_model_path(symbol)
    if model_path is not None:
        try:
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            x = feat_clean.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
            dmat = xgb.DMatrix(x)
            probas = bst.predict(dmat)
            if probas.std() > 0.01:
                threshold = cfg.get("prob_threshold", 0.55)
                model_ok = (probas >= threshold) | (probas <= (1.0 - threshold))
                model_filter = model_ok[signal_mask]
                entry_indices = entry_indices[model_filter]
                directions = directions[model_filter]
        except Exception as e:
            print(f"  {symbol}: model error {e}")

    if len(entry_indices) < 10:
        return None

    # 3. OHLCV + ATR
    open_arr = feat_clean["open"].to_numpy().astype(np.float64)
    high_arr = feat_clean["high"].to_numpy().astype(np.float64)
    low_arr = feat_clean["low"].to_numpy().astype(np.float64)
    close_arr = feat_clean["close"].to_numpy().astype(np.float64)
    atr_arr = compute_atr14(high_arr, low_arr, close_arr)

    # 4. Costs
    spread_bps = infer_spread_bps(ticks_df)
    cost_pct_1x = ftmo_cost_pct(symbol, cluster, spread_bps)
    cost_pct_15x = cost_pct_1x * 1.5

    # 5. Exit params
    ep = ExitParams(
        atr_sl_mult=exit_row["best_atr_sl_mult"],
        atr_tp_mult=exit_row["best_atr_tp_mult"],
        breakeven_atr=exit_row["best_breakeven_atr"],
        trail_activation_atr=exit_row["best_trail_activation_atr"],
        trail_distance_atr=exit_row["best_trail_distance_atr"],
        horizon=int(exit_row["best_horizon"]),
    )

    # 6. Walk-forward OOS
    n_samples = len(feat_clean)
    wf = TF_WF_SIZES.get(timeframe, TF_WF_SIZES["H1"])
    splits = purged_walk_forward_splits(
        n_samples=n_samples,
        train_size=wf["train_size"],
        test_size=wf["test_size"],
        purge=8, embargo=8,
    )
    if not splits:
        return None

    # Collect trade-level data across all OOS folds
    trades = []  # list of dicts per trade

    for _, te_idx in splits:
        fold_mask = np.isin(entry_indices, te_idx)
        fold_entries = entry_indices[fold_mask]
        fold_dirs = directions[fold_mask]
        if len(fold_entries) < 2:
            continue

        # Run at 1x costs
        pnl_1x, bars_held, exit_types = simulate_trades(
            fold_entries, fold_dirs,
            open_arr, high_arr, low_arr, close_arr, atr_arr,
            ep, cost_pct_1x,
        )
        if len(pnl_1x) == 0:
            continue

        # Run at 1.5x costs
        pnl_15x, _, _ = simulate_trades(
            fold_entries, fold_dirs,
            open_arr, high_arr, low_arr, close_arr, atr_arr,
            ep, cost_pct_15x,
        )

        # Reconstruct true entry indices
        true_entries = reconstruct_trade_entries(
            fold_entries, bars_held, open_arr, atr_arr, n_samples
        )
        if len(true_entries) != len(pnl_1x):
            # Fallback: skip this fold if reconstruction fails
            print(f"  {symbol}: mapping mismatch fold ({len(true_entries)} vs {len(pnl_1x)})")
            continue

        # Reconstruct trade directions
        # true_entries are indices into feat_clean, directions are at signal positions
        # We need the direction for each true entry
        true_dirs = primary_side[true_entries]

        # Position sizing
        sl_fracs = atr_arr[true_entries] * ep.atr_sl_mult / close_arr[true_entries]
        sl_fracs = np.clip(sl_fracs, 1e-6, None)
        risk_amount = ACCOUNT * risk_pct
        dollar_pnl_1x = risk_amount * pnl_1x / sl_fracs
        dollar_pnl_15x = risk_amount * pnl_15x / sl_fracs

        for j in range(len(pnl_1x)):
            entry_bar = true_entries[j] + 1  # +1 because entry at next bar open
            trades.append({
                "symbol": symbol,
                "timestamp": time_arr[min(entry_bar, n_samples - 1)],
                "direction": int(true_dirs[j]),
                "pnl_pct": float(pnl_1x[j]),
                "dollar_pnl_1x": float(dollar_pnl_1x[j]),
                "dollar_pnl_15x": float(dollar_pnl_15x[j]),
                "bars_held": int(bars_held[j]),
                "exit_type": int(exit_types[j]),
            })

    if not trades:
        return None

    # Buy & hold: first to last OOS bar
    oos_bars = set()
    for _, te_idx in splits:
        oos_bars.update(te_idx.tolist())
    oos_bars = sorted(oos_bars)
    if oos_bars:
        first_oos = min(oos_bars)
        last_oos = max(oos_bars)
        bh_start_px = close_arr[first_oos]
        bh_end_px = close_arr[last_oos]
        bh_return_pct = (bh_end_px - bh_start_px) / bh_start_px * 100
        oos_start_date = time_arr[first_oos]
        oos_end_date = time_arr[min(last_oos, n_samples - 1)]
    else:
        bh_return_pct = 0.0
        oos_start_date = time_arr[0]
        oos_end_date = time_arr[-1]

    print(f"  {symbol}: {len(trades)} trades, OOS {oos_start_date} → {oos_end_date}, B&H={bh_return_pct:+.1f}%")
    return {
        "symbol": symbol,
        "trades": trades,
        "bh_return_pct": bh_return_pct,
        "oos_start": oos_start_date,
        "oos_end": oos_end_date,
        "cost_bps_1x": round(cost_pct_1x * 1e4, 1),
        "spread_bps": round(spread_bps, 1),
        "cluster": cluster,
        "risk_pct": risk_pct,
    }


def main():
    p = argparse.ArgumentParser(description="Institutional-grade portfolio analysis")
    p.add_argument("--results-csv", type=str, default=str(DEFAULT_EXIT_CSV),
                   help="Path to exit results CSV (default: active 8 from exit_optuna)")
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    exit_csv = Path(args.results_csv)
    exit_df = pl.read_csv(str(exit_csv))
    exit_rows = {row["symbol"]: row for row in exit_df.iter_rows(named=True)}

    symbols = sorted(configs.keys())
    print(f"Analysing {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Account: ${ACCOUNT:,.0f}\n")

    # Run per-symbol analysis (sequential for reliability)
    all_results = {}
    for sym in symbols:
        if sym not in exit_rows:
            print(f"  {sym}: not in exit CSV, skipping")
            continue
        r = analyse_symbol(sym, configs[sym], exit_rows[sym])
        if r is not None:
            all_results[sym] = r

    if not all_results:
        print("No results!")
        return

    # ── Aggregate all trades ─────────────────────────────────────────────
    all_trades = []
    for r in all_results.values():
        all_trades.extend(r["trades"])

    # Sort by timestamp
    all_trades.sort(key=lambda t: t["timestamp"])

    # Convert timestamps to datetime for binning
    timestamps = []
    for t in all_trades:
        ts = t["timestamp"]
        if hasattr(ts, 'item'):
            ts = ts.item()
        if isinstance(ts, (int, float)):
            ts = datetime.utcfromtimestamp(ts / 1e9 if ts > 1e12 else ts)
        elif isinstance(ts, np.datetime64):
            ts = ts.astype('datetime64[s]').astype(datetime)
        timestamps.append(ts)

    dollar_pnl_1x = np.array([t["dollar_pnl_1x"] for t in all_trades])
    dollar_pnl_15x = np.array([t["dollar_pnl_15x"] for t in all_trades])
    directions = np.array([t["direction"] for t in all_trades])

    # Check for NaN contamination
    nan_count_1x = np.isnan(dollar_pnl_1x).sum()
    nan_count_15x = np.isnan(dollar_pnl_15x).sum()
    if nan_count_1x > 0 or nan_count_15x > 0:
        print(f"\n⚠  NaN detected: {nan_count_1x} in 1x, {nan_count_15x} in 1.5x — filtering out")
        valid = ~np.isnan(dollar_pnl_1x) & ~np.isnan(dollar_pnl_15x)
        dollar_pnl_1x = dollar_pnl_1x[valid]
        dollar_pnl_15x = dollar_pnl_15x[valid]
        directions = directions[valid]
        timestamps = [ts for ts, v in zip(timestamps, valid) if v]
        all_trades = [t for t, v in zip(all_trades, valid) if v]

    n_trades = len(all_trades)
    print(f"\n{'='*90}")
    print(f"INSTITUTIONAL ANALYSIS — {n_trades} trades across {len(all_results)} symbols")
    print(f"{'='*90}")

    # ── 1. Period & CAGR ─────────────────────────────────────────────────
    first_date = min(timestamps)
    last_date = max(timestamps)

    # Handle both datetime and Timestamp types
    if hasattr(first_date, 'to_pydatetime'):
        first_date = first_date.to_pydatetime()
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()

    period_days = (last_date - first_date).days
    period_years = period_days / 365.25

    total_return_1x = float(np.sum(dollar_pnl_1x))
    total_return_15x = float(np.sum(dollar_pnl_15x))

    final_equity_1x = ACCOUNT + total_return_1x
    final_equity_15x = ACCOUNT + total_return_15x

    if period_years > 0 and final_equity_1x > 0:
        cagr_1x = (final_equity_1x / ACCOUNT) ** (1.0 / period_years) - 1.0
    else:
        cagr_1x = 0.0

    if period_years > 0 and final_equity_15x > 0:
        cagr_15x = (final_equity_15x / ACCOUNT) ** (1.0 / period_years) - 1.0
    else:
        cagr_15x = 0.0

    print(f"\n{'─ 1. PERIOD & CAGR ─':─<60}")
    print(f"  OOS period:         {first_date.strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')}")
    print(f"  Duration:           {period_days} days = {period_years:.2f} years")
    print(f"  Total return (1x):  ${total_return_1x:>12,.0f}  ({total_return_1x/ACCOUNT*100:+.1f}%)")
    print(f"  CAGR (1x costs):    {cagr_1x*100:>8.1f}%")
    print(f"  CAGR (1.5× costs):  {cagr_15x*100:>8.1f}%")

    # ── 2. Trade Counts ──────────────────────────────────────────────────
    trades_per_year = n_trades / max(period_years, 0.01)

    print(f"\n{'─ 2. TRADE COUNTS ─':─<60}")
    print(f"  Total trades:       {n_trades}")
    print(f"  Trades/year:        {trades_per_year:.0f}")
    print(f"  Trades/month:       {trades_per_year/12:.0f}")

    # Per symbol
    sym_counts = defaultdict(int)
    for t in all_trades:
        sym_counts[t["symbol"]] += 1
    for sym in sorted(sym_counts, key=sym_counts.get, reverse=True):
        print(f"    {sym:20s}: {sym_counts[sym]:4d} trades ({sym_counts[sym]/max(period_years,0.01):.0f}/yr)")

    # ── 3. Long vs Short ─────────────────────────────────────────────────
    long_mask = directions == 1
    short_mask = directions == -1

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    pnl_long = float(np.sum(dollar_pnl_1x[long_mask]))
    pnl_short = float(np.sum(dollar_pnl_1x[short_mask]))

    # Win rates per side
    long_wins = int(np.sum(dollar_pnl_1x[long_mask] > 0))
    short_wins = int(np.sum(dollar_pnl_1x[short_mask] > 0))

    print(f"\n{'─ 3. LONG vs SHORT ─':─<60}")
    print(f"  LONG:   {n_long:4d} trades ({n_long/n_trades*100:.1f}%)  PnL=${pnl_long:>10,.0f}  WR={long_wins/max(n_long,1)*100:.1f}%  Avg=${pnl_long/max(n_long,1):>+7,.0f}")
    print(f"  SHORT:  {n_short:4d} trades ({n_short/n_trades*100:.1f}%)  PnL=${pnl_short:>10,.0f}  WR={short_wins/max(n_short,1)*100:.1f}%  Avg=${pnl_short/max(n_short,1):>+7,.0f}")

    # Per symbol long/short
    for sym in sorted(sym_counts, key=sym_counts.get, reverse=True):
        sym_trades = [t for t in all_trades if t["symbol"] == sym]
        sym_pnl = np.array([t["dollar_pnl_1x"] for t in sym_trades])
        sym_dirs = np.array([t["direction"] for t in sym_trades])
        sl = sym_dirs == 1
        ss = sym_dirs == -1
        print(f"    {sym:15s}  L={int(sl.sum()):3d} ${np.sum(sym_pnl[sl]):>+9,.0f}   "
              f"S={int(ss.sum()):3d} ${np.sum(sym_pnl[ss]):>+9,.0f}")

    # ── 4. Buy & Hold Comparison ─────────────────────────────────────────
    print(f"\n{'─ 4. BUY & HOLD COMPARISON ─':─<60}")
    print(f"  {'Symbol':15s} {'Strategy%':>12s} {'B&H%':>10s} {'Alpha':>10s}")
    print(f"  {'─'*50}")

    total_strat_pct = total_return_1x / ACCOUNT * 100
    for sym, r in sorted(all_results.items()):
        sym_pnl_total = sum(t["dollar_pnl_1x"] for t in r["trades"])
        strat_pct = sym_pnl_total / ACCOUNT * 100
        bh_pct = r["bh_return_pct"]
        alpha = strat_pct - bh_pct
        print(f"  {sym:15s} {strat_pct:>+11.1f}% {bh_pct:>+9.1f}% {alpha:>+9.1f}%")

    # ── 5. Cost Sensitivity ──────────────────────────────────────────────
    print(f"\n{'─ 5. COST SENSITIVITY ─':─<60}")
    print(f"  {'':15s} {'1.0× costs':>12s} {'1.5× costs':>12s} {'Delta':>12s}")
    print(f"  {'─'*55}")
    print(f"  {'Total return':15s} ${total_return_1x:>10,.0f}   ${total_return_15x:>10,.0f}   ${total_return_15x-total_return_1x:>+10,.0f}")
    print(f"  {'CAGR':15s} {cagr_1x*100:>10.1f}%   {cagr_15x*100:>10.1f}%   {(cagr_15x-cagr_1x)*100:>+10.1f}%")

    # Per symbol cost breakdown
    for sym, r in sorted(all_results.items()):
        pnl_1 = sum(t["dollar_pnl_1x"] for t in r["trades"])
        pnl_15 = sum(t["dollar_pnl_15x"] for t in r["trades"])
        n = len(r["trades"])
        print(f"    {sym:15s}  1x=${pnl_1:>+9,.0f}  1.5x=${pnl_15:>+9,.0f}  "
              f"cost={r['cost_bps_1x']:.1f}bps  spread={r['spread_bps']:.1f}bps  "
              f"trades={n}")

    # ── 6. Equity Curve & Drawdown ───────────────────────────────────────
    equity_1x = np.cumsum(dollar_pnl_1x)
    equity_15x = np.cumsum(dollar_pnl_15x)

    peak_1x = np.maximum.accumulate(np.concatenate([[0], equity_1x]))
    dd_1x = peak_1x[1:] - equity_1x
    max_dd_1x = float(np.max(dd_1x)) if len(dd_1x) > 0 else 0

    peak_15x = np.maximum.accumulate(np.concatenate([[0], equity_15x]))
    dd_15x = peak_15x[1:] - equity_15x
    max_dd_15x = float(np.max(dd_15x)) if len(dd_15x) > 0 else 0

    # Drawdown duration (in trades and calendar days)
    in_dd = dd_1x > 0
    max_dd_duration_trades = 0
    current_dd_trades = 0
    for v in in_dd:
        if v:
            current_dd_trades += 1
            max_dd_duration_trades = max(max_dd_duration_trades, current_dd_trades)
        else:
            current_dd_trades = 0

    # Calendar-based drawdown duration
    max_dd_duration_days = 0
    dd_start_idx = None
    for i in range(len(in_dd)):
        if in_dd[i]:
            if dd_start_idx is None:
                dd_start_idx = i
        else:
            if dd_start_idx is not None:
                start_dt = timestamps[dd_start_idx]
                end_dt = timestamps[i]
                if hasattr(start_dt, 'to_pydatetime'):
                    start_dt = start_dt.to_pydatetime()
                if hasattr(end_dt, 'to_pydatetime'):
                    end_dt = end_dt.to_pydatetime()
                dd_days = (end_dt - start_dt).days
                max_dd_duration_days = max(max_dd_duration_days, dd_days)
                dd_start_idx = None

    # Sharpe (daily returns)
    daily_pnl = defaultdict(float)
    for ts, pnl in zip(timestamps, dollar_pnl_1x):
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        day = dt.date()
        daily_pnl[day] += pnl

    daily_returns = np.array(sorted(daily_pnl.values()))
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        daily_sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        daily_sharpe = 0.0

    print(f"\n{'─ 6. DRAWDOWN & RISK ─':─<60}")
    print(f"  Max drawdown (1x):  ${max_dd_1x:>10,.0f}  ({max_dd_1x/ACCOUNT*100:.1f}%)")
    print(f"  Max drawdown (1.5x):${max_dd_15x:>10,.0f}  ({max_dd_15x/ACCOUNT*100:.1f}%)")
    print(f"  Max DD duration:    {max_dd_duration_trades} trades / {max_dd_duration_days} calendar days")
    print(f"  Daily Sharpe:       {daily_sharpe:.2f}")
    print(f"  Calmar (1x):        {total_return_1x / max(max_dd_1x, 1):.2f}")

    # ── 7. Worst Rolling 12-Month ────────────────────────────────────────
    # Aggregate PnL by calendar month
    monthly_pnl = defaultdict(float)
    monthly_pnl_15x = defaultdict(float)
    for ts, p1, p15 in zip(timestamps, dollar_pnl_1x, dollar_pnl_15x):
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        month_key = dt.strftime("%Y-%m")
        monthly_pnl[month_key] += p1
        monthly_pnl_15x[month_key] += p15

    months_sorted = sorted(monthly_pnl.keys())
    month_pnl_arr = np.array([monthly_pnl[m] for m in months_sorted])
    month_pnl_15x_arr = np.array([monthly_pnl_15x[m] for m in months_sorted])

    print(f"\n{'─ 7. ROLLING 12-MONTH RETURNS ─':─<60}")
    if len(month_pnl_arr) >= 12:
        rolling_12m = np.array([
            np.sum(month_pnl_arr[i:i+12])
            for i in range(len(month_pnl_arr) - 11)
        ])
        rolling_12m_15x = np.array([
            np.sum(month_pnl_15x_arr[i:i+12])
            for i in range(len(month_pnl_15x_arr) - 11)
        ])
        worst_12m_idx = int(np.argmin(rolling_12m))
        worst_12m = float(rolling_12m[worst_12m_idx])
        best_12m = float(np.max(rolling_12m))
        worst_12m_start = months_sorted[worst_12m_idx]
        worst_12m_end = months_sorted[worst_12m_idx + 11]

        worst_12m_15x = float(np.min(rolling_12m_15x))

        print(f"  Best  12-month (1x):   ${best_12m:>+10,.0f}  ({best_12m/ACCOUNT*100:+.1f}%)")
        print(f"  Worst 12-month (1x):   ${worst_12m:>+10,.0f}  ({worst_12m/ACCOUNT*100:+.1f}%)  [{worst_12m_start} → {worst_12m_end}]")
        print(f"  Worst 12-month (1.5x): ${worst_12m_15x:>+10,.0f}  ({worst_12m_15x/ACCOUNT*100:+.1f}%)")

        # All rolling windows
        print(f"\n  Rolling 12-month windows:")
        rolling_starts = [months_sorted[i] for i in range(len(rolling_12m))]
        rolling_ends = [months_sorted[i+11] for i in range(len(rolling_12m))]
        for i in range(len(rolling_12m)):
            print(f"    {rolling_starts[i]} → {rolling_ends[i]}:  ${rolling_12m[i]:>+10,.0f}  ({rolling_12m[i]/ACCOUNT*100:+.1f}%)")
    else:
        print(f"  Only {len(month_pnl_arr)} months of data — insufficient for 12-month rolling window")

    # ── 8. Per-Year Regime Analysis ──────────────────────────────────────
    yearly_pnl = defaultdict(float)
    yearly_pnl_15x = defaultdict(float)
    yearly_trades = defaultdict(int)
    yearly_wins = defaultdict(int)
    yearly_long = defaultdict(int)
    yearly_short = defaultdict(int)
    yearly_long_pnl = defaultdict(float)
    yearly_short_pnl = defaultdict(float)

    for t, p1, p15, d in zip(all_trades, dollar_pnl_1x, dollar_pnl_15x, directions):
        ts = t["timestamp"]
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        if isinstance(dt, np.datetime64):
            dt = dt.astype('datetime64[s]').astype(datetime)
        elif isinstance(dt, (int, float)):
            dt = datetime.utcfromtimestamp(dt / 1e9 if dt > 1e12 else dt)
        year = dt.year
        yearly_pnl[year] += p1
        yearly_pnl_15x[year] += p15
        yearly_trades[year] += 1
        if p1 > 0:
            yearly_wins[year] += 1
        if d == 1:
            yearly_long[year] += 1
            yearly_long_pnl[year] += p1
        else:
            yearly_short[year] += 1
            yearly_short_pnl[year] += p1

    print(f"\n{'─ 8. PER-YEAR REGIME ANALYSIS ─':─<60}")
    print(f"  {'Year':6s} {'Trades':>7s} {'Return$':>12s} {'Return%':>9s} "
          f"{'WR%':>6s} {'Long':>5s} {'LongPnL':>10s} {'Short':>5s} {'ShortPnL':>10s} "
          f"{'1.5xRet':>10s}")
    print(f"  {'─'*88}")
    for year in sorted(yearly_pnl.keys()):
        nt = yearly_trades[year]
        wr = yearly_wins[year] / max(nt, 1) * 100
        ret = yearly_pnl[year]
        ret_pct = ret / ACCOUNT * 100
        ret_15x = yearly_pnl_15x[year]
        print(f"  {year:6d} {nt:7d} ${ret:>10,.0f}  {ret_pct:>+7.1f}% "
              f"{wr:5.1f}% {yearly_long[year]:5d} ${yearly_long_pnl[year]:>+9,.0f} "
              f"{yearly_short[year]:5d} ${yearly_short_pnl[year]:>+9,.0f} "
              f"${ret_15x:>+9,.0f}")

    # ── 9. Monthly PnL Table ─────────────────────────────────────────────
    print(f"\n{'─ 9. MONTHLY PnL ─':─<60}")
    print(f"  {'Month':10s} {'PnL$':>10s} {'PnL%':>8s} {'Cum$':>12s} {'Trades':>7s}")
    print(f"  {'─'*50}")
    cum = 0.0
    monthly_trade_counts = defaultdict(int)
    for t in all_trades:
        ts = t["timestamp"]
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        if isinstance(dt, np.datetime64):
            dt = dt.astype('datetime64[s]').astype(datetime)
        monthly_trade_counts[dt.strftime("%Y-%m")] += 1

    for m in months_sorted:
        p = monthly_pnl[m]
        cum += p
        mc = monthly_trade_counts.get(m, 0)
        print(f"  {m:10s} ${p:>+9,.0f} {p/ACCOUNT*100:>+6.1f}% ${cum:>11,.0f} {mc:7d}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*90}")
    print(f"  Period:             {first_date.strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')} ({period_years:.2f} years)")
    print(f"  Total trades:       {n_trades} ({trades_per_year:.0f}/yr)")
    print(f"  CAGR (1× costs):    {cagr_1x*100:.1f}%")
    print(f"  CAGR (1.5× costs):  {cagr_15x*100:.1f}%")
    print(f"  Max drawdown:       ${max_dd_1x:,.0f} ({max_dd_1x/ACCOUNT*100:.1f}%)")
    print(f"  Max DD duration:    {max_dd_duration_days} days")
    print(f"  Daily Sharpe:       {daily_sharpe:.2f}")
    print(f"  Long/Short split:   {n_long}/{n_short} ({n_long/n_trades*100:.0f}%/{n_short/n_trades*100:.0f}%)")
    print(f"  Long PnL:           ${pnl_long:>+10,.0f}")
    print(f"  Short PnL:          ${pnl_short:>+10,.0f}")
    print(f"  Disguised B&H?      {'UNLIKELY' if pnl_short > 0 else 'POSSIBLE'} — short side is {'profitable' if pnl_short > 0 else 'losing'}")
    if len(month_pnl_arr) >= 12:
        print(f"  Worst 12-month:     ${worst_12m:>+10,.0f} ({worst_12m/ACCOUNT*100:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
