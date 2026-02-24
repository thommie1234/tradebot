"""
Portfolio-level analysis of leak-free backtest results.

Computes:
  1. Portfolio max DD (combined equity curve)
  2. Portfolio Sharpe
  3. Portfolio PF
  4. Scaled risk (0.8×) impact
  5. Rolling correlation + worst cluster drawdown
  6. Stress scenario (-3σ equity shock)
  7. Worst rolling 3-month

Usage:
    python3 research/leak_free_portfolio_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import xgboost as xgb

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from research.integrated_pipeline import (
    purged_walk_forward_splits,
    set_polars_threads,
)
from research.train_ml_strategy import infer_spread_bps, make_time_bars
from research.optuna_orchestrator import (
    DATA_ROOTS,
    cluster_for_symbol,
    load_symbol_ticks_lf,
)
from research.exit_simulator import ExitParams, simulate_trades
from research.wfo_portfolio_backtest import (
    CONSERVATIVE_DEFAULTS,
    TF_WF_SIZES,
    compute_atr14,
    find_model_path,
    ftmo_cost_pct,
    optimize_exits_on_past_data,
)

ACCOUNT = 100_000.0
CONFIG_PATH = REPO_ROOT / "config" / "sovereign_configs.json"
EXIT_CSV = REPO_ROOT / "models" / "optuna_results" / "exit_all_20260215_203654" / "exit_active_8.csv"

# The 5-symbol portfolio (NZD_USD killed — PF 1.12, largest DD, 5% contribution)
PORTFOLIO_SYMBOLS = ["RACE", "NVDA", "PFE", "LVMH", "JP225.cash"]
WARMUP_FOLDS = 3
LEAK_FREE_TRIALS = 100


def reconstruct_trade_entries(
    entry_indices: np.ndarray,
    bars_held: np.ndarray,
    open_arr: np.ndarray,
    atr_arr: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    """Mirror simulate_trades skip logic to find which entries became trades."""
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

        result.append(idx)
        exit_bar = entry_bar + int(bars_held[trade_count])
        position_end = exit_bar
        trade_count += 1

    return np.array(result, dtype=np.int64)


def analyse_symbol_leak_free(symbol: str, cfg: dict, exit_row: dict) -> dict | None:
    """Leak-free WFO analysis with trade-level timestamps."""
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
        return None
    feat_clean = feat.drop_nulls(FEATURE_COLUMNS)
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
        except Exception:
            pass

    if len(entry_indices) < 10:
        return None

    # 3. OHLCV + ATR
    open_arr = feat_clean["open"].to_numpy().astype(np.float64)
    high_arr = feat_clean["high"].to_numpy().astype(np.float64)
    low_arr = feat_clean["low"].to_numpy().astype(np.float64)
    close_arr = feat_clean["close"].to_numpy().astype(np.float64)
    atr_arr = compute_atr14(high_arr, low_arr, close_arr)
    n_samples = len(feat_clean)

    # 4. Costs
    spread_bps = infer_spread_bps(ticks_df)
    cost_pct = ftmo_cost_pct(symbol, cluster, spread_bps)

    # 5. WF splits
    wf = TF_WF_SIZES.get(timeframe, TF_WF_SIZES["H1"])
    splits = purged_walk_forward_splits(
        n_samples=n_samples,
        train_size=wf["train_size"],
        test_size=wf["test_size"],
        purge=8, embargo=8,
    )
    if not splits:
        return None

    # 6. Leak-free rolling per-fold
    fold_entries_list = []
    fold_dirs_list = []
    for _, te_idx in splits:
        fm = np.isin(entry_indices, te_idx)
        fold_entries_list.append(entry_indices[fm])
        fold_dirs_list.append(directions[fm])

    trades = []

    for fold_idx, (_, te_idx) in enumerate(splits):
        fold_entries = fold_entries_list[fold_idx]
        fold_dirs = fold_dirs_list[fold_idx]
        if len(fold_entries) < 2:
            continue

        if fold_idx < WARMUP_FOLDS:
            fold_ep = CONSERVATIVE_DEFAULTS.get(cluster, CONSERVATIVE_DEFAULTS["index"])
        else:
            past_e, past_d, past_fids = [], [], []
            for prev_idx in range(fold_idx):
                pe = fold_entries_list[prev_idx]
                pd = fold_dirs_list[prev_idx]
                if len(pe) > 0:
                    past_e.append(pe)
                    past_d.append(pd)
                    past_fids.append(np.full(len(pe), prev_idx, dtype=np.int32))

            if not past_e or len(past_e) < WARMUP_FOLDS:
                fold_ep = CONSERVATIVE_DEFAULTS.get(cluster, CONSERVATIVE_DEFAULTS["index"])
            else:
                fold_ep = optimize_exits_on_past_data(
                    np.concatenate(past_e), np.concatenate(past_d),
                    np.concatenate(past_fids),
                    open_arr, high_arr, low_arr, close_arr, atr_arr,
                    cluster, cost_pct, n_trials=LEAK_FREE_TRIALS,
                )

        pnl, bars_held, exit_types = simulate_trades(
            fold_entries, fold_dirs,
            open_arr, high_arr, low_arr, close_arr, atr_arr,
            fold_ep, cost_pct,
        )
        if len(pnl) == 0:
            continue

        true_entries = reconstruct_trade_entries(
            fold_entries, bars_held, open_arr, atr_arr, n_samples
        )
        if len(true_entries) != len(pnl):
            continue

        true_dirs = primary_side[true_entries]
        sl_fracs = atr_arr[true_entries] * fold_ep.atr_sl_mult / close_arr[true_entries]
        sl_fracs = np.clip(sl_fracs, 1e-6, None)
        risk_amount = ACCOUNT * risk_pct

        for j in range(len(pnl)):
            entry_bar = true_entries[j] + 1
            dollar_pnl = risk_amount * pnl[j] / sl_fracs[j]
            trades.append({
                "symbol": symbol,
                "timestamp": time_arr[min(entry_bar, n_samples - 1)],
                "direction": int(true_dirs[j]),
                "pnl_pct": float(pnl[j]),
                "dollar_pnl": float(dollar_pnl),
                "dollar_pnl_08x": float(dollar_pnl * 0.8),
                "bars_held": int(bars_held[j]),
                "exit_type": int(exit_types[j]),
                "fold_idx": fold_idx,
                "is_warmup": fold_idx < WARMUP_FOLDS,
            })

    if not trades:
        return None

    total_pnl = sum(t["dollar_pnl"] for t in trades)
    print(f"  {symbol}: {len(trades)} trades, ${total_pnl:+,.0f}")
    return {"symbol": symbol, "trades": trades, "cluster": cluster, "risk_pct": risk_pct}


def main():
    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    exit_df = pl.read_csv(str(EXIT_CSV))
    exit_rows = {row["symbol"]: row for row in exit_df.iter_rows(named=True)}

    print(f"LEAK-FREE PORTFOLIO ANALYSIS")
    print(f"Symbols: {', '.join(PORTFOLIO_SYMBOLS)}")
    print(f"Account: ${ACCOUNT:,.0f}\n")

    all_results = {}
    for sym in PORTFOLIO_SYMBOLS:
        if sym not in exit_rows or sym not in configs:
            print(f"  {sym}: missing from configs/exit CSV")
            continue
        r = analyse_symbol_leak_free(sym, configs[sym], exit_rows[sym])
        if r is not None:
            all_results[sym] = r

    if not all_results:
        print("No results!")
        return

    # ── Combine all trades ──────────────────────────────────────────────
    all_trades = []
    for r in all_results.values():
        all_trades.extend(r["trades"])
    all_trades.sort(key=lambda t: t["timestamp"])

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

    dollar_pnl = np.array([t["dollar_pnl"] for t in all_trades])
    dollar_pnl_08x = np.array([t["dollar_pnl_08x"] for t in all_trades])
    directions = np.array([t["direction"] for t in all_trades])
    symbols_arr = np.array([t["symbol"] for t in all_trades])

    # Filter NaN
    valid = ~np.isnan(dollar_pnl)
    if not valid.all():
        print(f"\nFiltered {(~valid).sum()} NaN trades")
        dollar_pnl = dollar_pnl[valid]
        dollar_pnl_08x = dollar_pnl_08x[valid]
        directions = directions[valid]
        symbols_arr = symbols_arr[valid]
        timestamps = [ts for ts, v in zip(timestamps, valid) if v]
        all_trades = [t for t, v in zip(all_trades, valid) if v]

    n_trades = len(all_trades)
    first_date = min(timestamps)
    last_date = max(timestamps)
    if hasattr(first_date, 'to_pydatetime'):
        first_date = first_date.to_pydatetime()
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()
    period_days = (last_date - first_date).days
    period_years = period_days / 365.25

    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print(f"PORTFOLIO-LEVEL METRICS (LEAK-FREE) — {n_trades} trades, {len(all_results)} symbols")
    print(f"{'=' * 90}")

    # ── 1. Portfolio Equity Curve & Max DD ───────────────────────────────
    equity = np.cumsum(dollar_pnl)
    equity_08x = np.cumsum(dollar_pnl_08x)

    peak = np.maximum.accumulate(np.concatenate([[0], equity]))
    dd = peak[1:] - equity
    max_dd = float(np.max(dd))
    max_dd_pct = max_dd / ACCOUNT * 100

    peak_08x = np.maximum.accumulate(np.concatenate([[0], equity_08x]))
    dd_08x = peak_08x[1:] - equity_08x
    max_dd_08x = float(np.max(dd_08x))
    max_dd_08x_pct = max_dd_08x / ACCOUNT * 100

    total_return = float(equity[-1])
    total_return_08x = float(equity_08x[-1])

    # CAGR
    final_equity = ACCOUNT + total_return
    final_equity_08x = ACCOUNT + total_return_08x
    cagr = (final_equity / ACCOUNT) ** (1.0 / period_years) - 1.0 if period_years > 0 and final_equity > 0 else 0.0
    cagr_08x = (final_equity_08x / ACCOUNT) ** (1.0 / period_years) - 1.0 if period_years > 0 and final_equity_08x > 0 else 0.0

    # Calmar
    calmar = total_return / max(max_dd, 1.0)
    calmar_08x = total_return_08x / max(max_dd_08x, 1.0)

    print(f"\n{'─ 1. PORTFOLIO MAX DD ─':─<70}")
    print(f"  Total return (1.0×):    ${total_return:>12,.0f}  ({total_return/ACCOUNT*100:+.1f}%)")
    print(f"  Total return (0.8×):    ${total_return_08x:>12,.0f}  ({total_return_08x/ACCOUNT*100:+.1f}%)")
    print(f"  CAGR (1.0×):            {cagr*100:>8.1f}%")
    print(f"  CAGR (0.8×):            {cagr_08x*100:>8.1f}%")
    print(f"  Portfolio max DD (1.0×): ${max_dd:>10,.0f}  ({max_dd_pct:.1f}%)")
    print(f"  Portfolio max DD (0.8×): ${max_dd_08x:>10,.0f}  ({max_dd_08x_pct:.1f}%)")
    print(f"  Calmar (1.0×):          {calmar:.2f}")
    print(f"  Calmar (0.8×):          {calmar_08x:.2f}")
    print(f"  FTMO safe (1.0×):       {'YES' if max_dd_pct < 10 else 'NO'} ({max_dd_pct:.1f}% vs 10% limit)")
    print(f"  FTMO safe (0.8×):       {'YES' if max_dd_08x_pct < 10 else 'NO'} ({max_dd_08x_pct:.1f}% vs 10% limit)")

    # DD duration
    in_dd = dd > 0
    max_dd_duration_trades = 0
    current_dd_trades = 0
    for v in in_dd:
        if v:
            current_dd_trades += 1
            max_dd_duration_trades = max(max_dd_duration_trades, current_dd_trades)
        else:
            current_dd_trades = 0

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

    print(f"  Max DD duration:        {max_dd_duration_trades} trades / {max_dd_duration_days} calendar days")

    # ── 2. Portfolio Sharpe ─────────────────────────────────────────────
    daily_pnl = defaultdict(float)
    daily_pnl_08x = defaultdict(float)
    for ts, p, p08 in zip(timestamps, dollar_pnl, dollar_pnl_08x):
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        day = dt.date()
        daily_pnl[day] += p
        daily_pnl_08x[day] += p08

    daily_returns = np.array(sorted(daily_pnl.values()))
    daily_returns_08x = np.array(sorted(daily_pnl_08x.values()))

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    if len(daily_returns_08x) > 1 and np.std(daily_returns_08x) > 0:
        sharpe_08x = np.mean(daily_returns_08x) / np.std(daily_returns_08x) * np.sqrt(252)
    else:
        sharpe_08x = 0.0

    print(f"\n{'─ 2. PORTFOLIO SHARPE ─':─<70}")
    print(f"  Daily Sharpe (1.0×):    {sharpe:.2f}")
    print(f"  Daily Sharpe (0.8×):    {sharpe_08x:.2f}")
    print(f"  Trading days:           {len(daily_returns)}")

    # ── 3. Portfolio PF ─────────────────────────────────────────────────
    gross_profit = float(np.sum(dollar_pnl[dollar_pnl > 0]))
    gross_loss = float(np.abs(np.sum(dollar_pnl[dollar_pnl < 0])))
    pf = gross_profit / max(gross_loss, 1.0)

    gross_profit_08x = float(np.sum(dollar_pnl_08x[dollar_pnl_08x > 0]))
    gross_loss_08x = float(np.abs(np.sum(dollar_pnl_08x[dollar_pnl_08x < 0])))
    pf_08x = gross_profit_08x / max(gross_loss_08x, 1.0)

    wins = int(np.sum(dollar_pnl > 0))
    losses = int(np.sum(dollar_pnl < 0))

    print(f"\n{'─ 3. PORTFOLIO PF ─':─<70}")
    print(f"  Profit factor (1.0×):   {pf:.2f}")
    print(f"  Profit factor (0.8×):   {pf_08x:.2f}")
    print(f"  Gross profit:           ${gross_profit:>12,.0f}")
    print(f"  Gross loss:             ${gross_loss:>12,.0f}")
    print(f"  Win rate:               {wins/n_trades*100:.1f}% ({wins}/{n_trades})")
    print(f"  Avg win:                ${np.mean(dollar_pnl[dollar_pnl > 0]):>+8,.0f}")
    print(f"  Avg loss:               ${np.mean(dollar_pnl[dollar_pnl < 0]):>+8,.0f}")

    # ── 4. Per-Symbol Contribution ──────────────────────────────────────
    print(f"\n{'─ 4. PER-SYMBOL CONTRIBUTION ─':─<70}")
    print(f"  {'Symbol':15s} {'Trades':>7s} {'Return$':>10s} {'PF':>6s} {'WR%':>6s} {'AvgPnL$':>9s} {'MaxDD$':>10s}")
    print(f"  {'─'*65}")

    for sym in PORTFOLIO_SYMBOLS:
        if sym not in all_results:
            continue
        sym_trades = [t for t in all_trades if t["symbol"] == sym]
        sym_pnl = np.array([t["dollar_pnl"] for t in sym_trades])
        if len(sym_pnl) == 0:
            continue
        sym_eq = np.cumsum(sym_pnl)
        sym_peak = np.maximum.accumulate(np.concatenate([[0], sym_eq]))
        sym_dd = np.max(sym_peak[1:] - sym_eq)
        sym_wins = np.sum(sym_pnl > 0)
        sym_gp = np.sum(sym_pnl[sym_pnl > 0])
        sym_gl = np.abs(np.sum(sym_pnl[sym_pnl < 0]))
        sym_pf = sym_gp / max(sym_gl, 1.0)
        print(f"  {sym:15s} {len(sym_pnl):7d} ${np.sum(sym_pnl):>+9,.0f} {sym_pf:5.2f} "
              f"{sym_wins/len(sym_pnl)*100:5.1f}% ${np.mean(sym_pnl):>+8,.0f} ${sym_dd:>9,.0f}")

    # ── 5. Correlation Analysis ─────────────────────────────────────────
    print(f"\n{'─ 5. ROLLING CORRELATION & CLUSTER RISK ─':─<70}")

    # Build daily PnL per symbol
    all_days = sorted(set(
        (ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts).date()
        for ts in timestamps
    ))
    sym_daily = {}
    for sym in PORTFOLIO_SYMBOLS:
        if sym not in all_results:
            continue
        daily = defaultdict(float)
        for t in all_trades:
            if t["symbol"] != sym:
                continue
            ts = t["timestamp"]
            dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
            if isinstance(dt, np.datetime64):
                dt = dt.astype('datetime64[s]').astype(datetime)
            daily[dt.date()] += t["dollar_pnl"]
        sym_daily[sym] = np.array([daily.get(d, 0.0) for d in all_days])

    syms_with_data = [s for s in PORTFOLIO_SYMBOLS if s in sym_daily]
    n_syms = len(syms_with_data)

    if n_syms >= 2:
        # Full-period correlation matrix
        returns_matrix = np.column_stack([sym_daily[s] for s in syms_with_data])
        corr_matrix = np.corrcoef(returns_matrix.T)

        print(f"\n  Full-period correlation matrix:")
        print(f"  {'':15s}", end="")
        for s in syms_with_data:
            print(f" {s:>10s}", end="")
        print()
        for i, s1 in enumerate(syms_with_data):
            print(f"  {s1:15s}", end="")
            for j, s2 in enumerate(syms_with_data):
                print(f" {corr_matrix[i,j]:>10.3f}", end="")
            print()

        # Rolling 30-day correlation (equity cluster)
        equity_syms = [s for s in ["NVDA", "RACE", "LVMH", "PFE"] if s in sym_daily]
        if len(equity_syms) >= 2:
            print(f"\n  Equity cluster ({', '.join(equity_syms)}) — rolling 30d avg pairwise correlation:")
            window = 30
            rolling_corrs = []
            for start in range(0, len(all_days) - window, window // 2):
                end = start + window
                block = np.column_stack([sym_daily[s][start:end] for s in equity_syms])
                # Only compute if there's variance
                stds = block.std(axis=0)
                if np.all(stds > 0):
                    c = np.corrcoef(block.T)
                    # Average off-diagonal
                    n = len(equity_syms)
                    avg_corr = (c.sum() - n) / (n * (n - 1))
                    rolling_corrs.append((all_days[start], all_days[end - 1], avg_corr))

            if rolling_corrs:
                max_corr_window = max(rolling_corrs, key=lambda x: x[2])
                min_corr_window = min(rolling_corrs, key=lambda x: x[2])
                print(f"    Peak correlation:  {max_corr_window[2]:+.3f}  ({max_corr_window[0]} → {max_corr_window[1]})")
                print(f"    Min correlation:   {min_corr_window[2]:+.3f}  ({min_corr_window[0]} → {min_corr_window[1]})")
                avg_overall = np.mean([x[2] for x in rolling_corrs])
                print(f"    Average:           {avg_overall:+.3f}")

        # Correlation during worst DD period
        dd_peak_idx = int(np.argmax(dd))
        # Find DD start (last time equity was at peak before this)
        peak_val = peak[dd_peak_idx + 1]
        dd_start_search = dd_peak_idx
        for k in range(dd_peak_idx, -1, -1):
            if equity[k] >= peak_val - 0.01:
                dd_start_search = k
                break

        if dd_peak_idx > dd_start_search and n_syms >= 2:
            dd_start_day = (timestamps[dd_start_search].to_pydatetime()
                           if hasattr(timestamps[dd_start_search], 'to_pydatetime')
                           else timestamps[dd_start_search]).date()
            dd_end_day = (timestamps[dd_peak_idx].to_pydatetime()
                         if hasattr(timestamps[dd_peak_idx], 'to_pydatetime')
                         else timestamps[dd_peak_idx]).date()

            dd_day_mask = np.array([(d >= dd_start_day and d <= dd_end_day) for d in all_days])
            if dd_day_mask.sum() >= 5:
                dd_block = np.column_stack([sym_daily[s][dd_day_mask] for s in syms_with_data])
                stds = dd_block.std(axis=0)
                active = stds > 0
                if active.sum() >= 2:
                    active_syms = [s for s, a in zip(syms_with_data, active) if a]
                    dd_corr = np.corrcoef(dd_block[:, active].T)
                    n_active = len(active_syms)
                    avg_dd_corr = (dd_corr.sum() - n_active) / max(n_active * (n_active - 1), 1)

                    print(f"\n  Correlation during worst DD ({dd_start_day} → {dd_end_day}):")
                    print(f"    Avg pairwise: {avg_dd_corr:+.3f}")
                    print(f"    Per-symbol PnL during this DD:")
                    for s in syms_with_data:
                        dd_sym_pnl = sym_daily[s][dd_day_mask].sum()
                        print(f"      {s:15s}: ${dd_sym_pnl:>+8,.0f}")

    # ── 6. Stress Test: -3σ Equity Shock ────────────────────────────────
    print(f"\n{'─ 6. STRESS TEST: -3σ EQUITY SHOCK ─':─<70}")

    # Compute daily portfolio returns
    daily_rets = np.array(list(daily_pnl.values()))
    mean_daily = np.mean(daily_rets)
    std_daily = np.std(daily_rets)
    worst_daily = np.min(daily_rets)

    print(f"  Daily stats:")
    print(f"    Mean:                 ${mean_daily:>+8,.0f}")
    print(f"    Std:                  ${std_daily:>8,.0f}")
    print(f"    Worst day:            ${worst_daily:>+8,.0f}")
    print(f"    -3σ day:              ${mean_daily - 3*std_daily:>+8,.0f}")

    # 5-day shock (correlated drawdown)
    # Simulate 5 consecutive -2σ days
    shock_5d = 5 * (mean_daily - 2 * std_daily)
    shock_5d_pct = shock_5d / ACCOUNT * 100
    print(f"\n  5-day -2σ scenario:")
    print(f"    Impact:               ${shock_5d:>+8,.0f}  ({shock_5d_pct:+.1f}%)")
    print(f"    FTMO survival:        {'YES' if abs(shock_5d_pct) < 10 else 'NO'}")

    # Worst observed 5-day window
    sorted_days = sorted(daily_pnl.keys())
    daily_arr = np.array([daily_pnl[d] for d in sorted_days])
    if len(daily_arr) >= 5:
        rolling_5d = np.array([np.sum(daily_arr[i:i+5]) for i in range(len(daily_arr) - 4)])
        worst_5d = float(np.min(rolling_5d))
        worst_5d_idx = int(np.argmin(rolling_5d))
        worst_5d_start = sorted_days[worst_5d_idx]
        worst_5d_end = sorted_days[worst_5d_idx + 4]
        print(f"\n  Worst observed 5-day window:")
        print(f"    PnL:                  ${worst_5d:>+8,.0f}  ({worst_5d/ACCOUNT*100:+.1f}%)")
        print(f"    Period:               {worst_5d_start} → {worst_5d_end}")

    # ── 7. Worst Rolling 3-Month ────────────────────────────────────────
    monthly_pnl = defaultdict(float)
    monthly_pnl_08x = defaultdict(float)
    for ts, p, p08 in zip(timestamps, dollar_pnl, dollar_pnl_08x):
        dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        month_key = dt.strftime("%Y-%m")
        monthly_pnl[month_key] += p
        monthly_pnl_08x[month_key] += p08

    months_sorted = sorted(monthly_pnl.keys())
    month_arr = np.array([monthly_pnl[m] for m in months_sorted])
    month_arr_08x = np.array([monthly_pnl_08x[m] for m in months_sorted])

    print(f"\n{'─ 7. ROLLING PERIOD RETURNS ─':─<70}")
    if len(month_arr) >= 3:
        rolling_3m = np.array([np.sum(month_arr[i:i+3]) for i in range(len(month_arr) - 2)])
        rolling_3m_08x = np.array([np.sum(month_arr_08x[i:i+3]) for i in range(len(month_arr_08x) - 2)])
        worst_3m = float(np.min(rolling_3m))
        worst_3m_08x = float(np.min(rolling_3m_08x))
        best_3m = float(np.max(rolling_3m))
        worst_3m_idx = int(np.argmin(rolling_3m))

        print(f"  Best 3-month (1.0×):    ${best_3m:>+10,.0f}  ({best_3m/ACCOUNT*100:+.1f}%)")
        print(f"  Worst 3-month (1.0×):   ${worst_3m:>+10,.0f}  ({worst_3m/ACCOUNT*100:+.1f}%)  "
              f"[{months_sorted[worst_3m_idx]} → {months_sorted[worst_3m_idx+2]}]")
        print(f"  Worst 3-month (0.8×):   ${worst_3m_08x:>+10,.0f}  ({worst_3m_08x/ACCOUNT*100:+.1f}%)")

    if len(month_arr) >= 12:
        rolling_12m = np.array([np.sum(month_arr[i:i+12]) for i in range(len(month_arr) - 11)])
        worst_12m = float(np.min(rolling_12m))
        best_12m = float(np.max(rolling_12m))
        worst_12m_idx = int(np.argmin(rolling_12m))
        print(f"\n  Best 12-month (1.0×):   ${best_12m:>+10,.0f}  ({best_12m/ACCOUNT*100:+.1f}%)")
        print(f"  Worst 12-month (1.0×):  ${worst_12m:>+10,.0f}  ({worst_12m/ACCOUNT*100:+.1f}%)  "
              f"[{months_sorted[worst_12m_idx]} → {months_sorted[worst_12m_idx+11]}]")

    # ── 8. Monthly PnL ─────────────────────────────────────────────────
    print(f"\n{'─ 8. MONTHLY PnL (1.0×) ─':─<70}")
    print(f"  {'Month':10s} {'PnL$':>10s} {'PnL%':>8s} {'Cum$':>12s}")
    print(f"  {'─'*42}")
    cum = 0.0
    neg_months = 0
    for m in months_sorted:
        p = monthly_pnl[m]
        cum += p
        if p < 0:
            neg_months += 1
        print(f"  {m:10s} ${p:>+9,.0f} {p/ACCOUNT*100:>+6.1f}% ${cum:>11,.0f}")

    print(f"\n  Positive months: {len(months_sorted) - neg_months}/{len(months_sorted)} "
          f"({(len(months_sorted) - neg_months)/len(months_sorted)*100:.0f}%)")

    # ── 9. Long vs Short ────────────────────────────────────────────────
    long_mask = directions == 1
    short_mask = directions == -1
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    pnl_long = float(np.sum(dollar_pnl[long_mask]))
    pnl_short = float(np.sum(dollar_pnl[short_mask]))

    print(f"\n{'─ 9. LONG vs SHORT ─':─<70}")
    print(f"  LONG:   {n_long:4d} trades  PnL=${pnl_long:>+10,.0f}  WR={np.sum(dollar_pnl[long_mask]>0)/max(n_long,1)*100:.1f}%")
    print(f"  SHORT:  {n_short:4d} trades  PnL=${pnl_short:>+10,.0f}  WR={np.sum(dollar_pnl[short_mask]>0)/max(n_short,1)*100:.1f}%")

    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("EXECUTIVE SUMMARY — LEAK-FREE PORTFOLIO")
    print(f"{'=' * 90}")
    print(f"  Period:             {first_date.strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')} ({period_years:.2f} years)")
    print(f"  Symbols:            {', '.join(syms_with_data)}")
    print(f"  Total trades:       {n_trades}")
    print(f"  ")
    print(f"  {'Metric':25s} {'1.0× risk':>12s} {'0.8× risk':>12s}")
    print(f"  {'─'*50}")
    print(f"  {'Total return':25s} ${total_return:>11,.0f} ${total_return_08x:>11,.0f}")
    print(f"  {'CAGR':25s} {cagr*100:>11.1f}% {cagr_08x*100:>11.1f}%")
    print(f"  {'Portfolio max DD':25s} ${max_dd:>11,.0f} ${max_dd_08x:>11,.0f}")
    print(f"  {'Max DD %':25s} {max_dd_pct:>11.1f}% {max_dd_08x_pct:>11.1f}%")
    print(f"  {'Daily Sharpe':25s} {sharpe:>11.2f} {sharpe_08x:>11.2f}")
    print(f"  {'Calmar':25s} {calmar:>11.2f} {calmar_08x:>11.2f}")
    print(f"  {'Profit factor':25s} {pf:>11.2f} {pf_08x:>11.2f}")
    print(f"  {'FTMO safe (<10% DD)':25s} {'YES' if max_dd_pct < 10 else 'NO':>11s} {'YES' if max_dd_08x_pct < 10 else 'NO':>11s}")
    print(f"  ")
    ftmo_headroom = 10.0 - max_dd_pct
    ftmo_headroom_08x = 10.0 - max_dd_08x_pct
    print(f"  FTMO headroom (1.0×):   {ftmo_headroom:+.1f}% {'← TIGHT' if ftmo_headroom < 2 else ''}")
    print(f"  FTMO headroom (0.8×):   {ftmo_headroom_08x:+.1f}%")

    # ── 10. Risk Scaling Table ──────────────────────────────────────────
    print(f"\n{'─ 10. RISK SCALING TABLE ─':─<70}")
    print(f"  {'Scale':>7s} {'Return$':>12s} {'CAGR':>8s} {'MaxDD$':>10s} {'MaxDD%':>8s} {'Headroom':>10s} {'FTMO':>6s}")
    print(f"  {'─'*65}")

    best_scale = None
    for scale in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        sc_pnl = dollar_pnl * scale
        sc_eq = np.cumsum(sc_pnl)
        sc_peak = np.maximum.accumulate(np.concatenate([[0], sc_eq]))
        sc_dd = sc_peak[1:] - sc_eq
        sc_max_dd = float(np.max(sc_dd))
        sc_dd_pct = sc_max_dd / ACCOUNT * 100
        sc_return = float(sc_eq[-1])
        sc_final = ACCOUNT + sc_return
        sc_cagr = (sc_final / ACCOUNT) ** (1.0 / period_years) - 1.0 if period_years > 0 and sc_final > 0 else 0.0
        sc_headroom = 10.0 - sc_dd_pct
        sc_ftmo = "YES" if sc_dd_pct < 10 else "NO"
        marker = ""
        if sc_dd_pct <= 8.0 and best_scale is None:
            best_scale = scale
            marker = " ← TARGET"
        print(f"  {scale:>6.0%}   ${sc_return:>10,.0f} {sc_cagr*100:>7.1f}% ${sc_max_dd:>9,.0f} {sc_dd_pct:>7.1f}% {sc_headroom:>+9.1f}% {sc_ftmo:>6s}{marker}")

    verdict = ""
    if max_dd_08x_pct < 8 and sharpe_08x > 2:
        verdict = "FUNDABLE — passes professional risk committee"
    elif max_dd_08x_pct < 10 and sharpe_08x > 1.5:
        verdict = "BORDERLINE — proceed with caution, tight monitoring"
    else:
        verdict = "NOT READY — needs further risk reduction"
    print(f"\n  VERDICT: {verdict}")
    print()


if __name__ == "__main__":
    main()
