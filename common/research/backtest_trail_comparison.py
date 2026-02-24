"""
Backtest comparison: different breakeven / trailing stop settings.

Simulates bar-by-bar with actual OHLC to see how BE/trail affects outcomes.

Usage:
    python3 research/backtest_trail_comparison.py --symbols NVDA,LVMH,AAPL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from research.integrated_pipeline import (
    make_ev_custom_objective,
    purged_walk_forward_splits,
    set_polars_threads,
)
from research.train_ml_strategy import (
    infer_spread_bps,
    load_symbol_ticks,
    make_time_bars,
    sanitize_training_frame,
)
from research.optuna_orchestrator import broker_commission_bps, broker_slippage_bps

DATA_ROOTS = [
    "/home/tradebot/ssd_data_1/tick_data",
    "/home/tradebot/ssd_data_2/tick_data",
    "/home/tradebot/data_1/tick_data",
]

CONFIG_PATH = REPO_ROOT / "config" / "sovereign_configs.json"

# Trail configs to test: (label, breakeven_atr, trail_activation_atr, trail_distance_atr)
TRAIL_CONFIGS = [
    ("current_tight",  0.4, 1.0, 0.5),   # current settings
    ("medium",         0.8, 1.2, 0.6),   # medium
    ("relaxed",        1.2, 1.5, 0.7),   # proposed relaxed
    ("wide",           1.5, 2.0, 0.8),   # wide
    ("no_trail",       999, 999, 999),    # no BE/trail, pure SL/TP
]


def simulate_trade_with_trail(
    bars_ohlc: np.ndarray,  # shape (N, 4) = open, high, low, close
    atr: float,
    entry_price: float,
    direction: int,  # +1 = BUY, -1 = SELL
    sl_mult: float,
    tp_mult: float,
    be_atr: float,
    trail_act_atr: float,
    trail_dist_atr: float,
    horizon: int,
    cost_pct: float,
) -> float:
    """Simulate a single trade bar-by-bar with trailing stop mechanics.
    Returns net return (after costs)."""

    sl_dist = sl_mult * atr
    tp_dist = tp_mult * atr
    be_dist = be_atr * atr
    trail_act_dist = trail_act_atr * atr
    trail_trail_dist = trail_dist_atr * atr

    if direction == 1:  # BUY
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:  # SELL
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    be_triggered = False
    trailing_active = False
    best_price = entry_price

    for i in range(min(horizon, len(bars_ohlc))):
        o, h, l, c = bars_ohlc[i]

        if direction == 1:  # BUY
            # Check SL hit (use low)
            if l <= sl_price:
                ret = (sl_price - entry_price) / entry_price
                return ret - cost_pct

            # Check TP hit (use high)
            if h >= tp_price:
                ret = (tp_price - entry_price) / entry_price
                return ret - cost_pct

            # Update best price
            if h > best_price:
                best_price = h

            # Breakeven check
            profit = best_price - entry_price
            if not be_triggered and profit >= be_dist:
                sl_price = entry_price  # move SL to entry
                be_triggered = True

            # Trail activation
            if not trailing_active and profit >= trail_act_dist:
                trailing_active = True

            # Update trailing SL
            if trailing_active:
                new_sl = best_price - trail_trail_dist
                if new_sl > sl_price:
                    sl_price = new_sl

        else:  # SELL
            # Check SL hit (use high)
            if h >= sl_price:
                ret = (entry_price - sl_price) / entry_price
                return ret - cost_pct

            # Check TP hit (use low)
            if l <= tp_price:
                ret = (entry_price - tp_price) / entry_price
                return ret - cost_pct

            # Update best price (lowest for sell)
            if l < best_price:
                best_price = l

            profit = entry_price - best_price
            if not be_triggered and profit >= be_dist:
                sl_price = entry_price
                be_triggered = True

            if not trailing_active and profit >= trail_act_dist:
                trailing_active = True

            if trailing_active:
                new_sl = best_price + trail_trail_dist
                if new_sl < sl_price:
                    sl_price = new_sl

    # Horizon exit: close at last close
    if len(bars_ohlc) > 0:
        last_close = bars_ohlc[min(horizon - 1, len(bars_ohlc) - 1), 3]
        if direction == 1:
            ret = (last_close - entry_price) / entry_price
        else:
            ret = (entry_price - last_close) / entry_price
        return ret - cost_pct

    return -cost_pct


def run_trail_test(symbol: str, configs: dict) -> list[dict]:
    """Run WFO with bar-by-bar trail simulation for different settings."""
    set_polars_threads(4)

    sym_cfg = configs.get(symbol, {})
    sl_mult = sym_cfg.get("atr_sl_mult", 1.0)
    tp_mult = sym_cfg.get("atr_tp_mult", 2.5)

    ticks = load_symbol_ticks(symbol, DATA_ROOTS, years_filter=None)
    if ticks is None:
        return [{"symbol": symbol, "label": "no_data", "status": "no_tick_data"}]

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)
    cost_pct = (fee_bps + spread_bps + slippage_bps * 2) / 1e4

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat_base = build_bar_features(bars, z_threshold=1.0)

    # Get OHLC arrays for simulation
    ohlc = feat_base.select(["open", "high", "low", "close"]).to_numpy()
    # Compute ATR14 from bars (not exposed by feature_builder)
    h = feat_base["high"].to_numpy()
    l = feat_base["low"].to_numpy()
    c = feat_base["close"].to_numpy()
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    true_range = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_arr = np.convolve(true_range, np.ones(14)/14, mode='full')[:len(true_range)]
    atr_arr[:14] = atr_arr[14]  # fill warmup

    # Apply triple barrier for ML training labels (use no-trail as baseline)
    tb = apply_triple_barrier(
        close=feat_base["close"].to_numpy(),
        vol_proxy=feat_base["vol20"].to_numpy(),
        side=feat_base["primary_side"].to_numpy(),
        horizon=24, pt_mult=tp_mult, sl_mult=sl_mult,
    )

    feat = feat_base.with_columns([
        pl.Series("label", tb.label),
        pl.Series("target", tb.label),
        pl.Series("tb_ret", tb.tb_ret),
        pl.Series("avg_win", tb.upside),
        pl.Series("avg_loss", tb.downside),
        pl.Series("upside", tb.upside),
        pl.Series("downside", tb.downside),
        pl.lit(float(fee_bps)).alias("fee_bps"),
        pl.lit(float(spread_bps)).alias("spread_bps"),
        pl.lit(float(slippage_bps)).alias("slippage_bps"),
    ]).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    cols = FEATURE_COLUMNS + ["target", "tb_ret", "avg_win", "avg_loss",
                               "fee_bps", "spread_bps", "slippage_bps"]
    df = feat.select(cols).drop_nulls(cols)

    train_size, test_size = 1200, 300
    purge, embargo = 24, 24

    if df.height < train_size + test_size + 200:
        return [{"symbol": symbol, "label": c[0], "status": "insufficient_rows"}
                for c in TRAIL_CONFIGS]

    x = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy().astype(np.float32)
    avg_win = df["avg_win"].to_numpy().astype(np.float64)
    avg_loss = df["avg_loss"].to_numpy().astype(np.float64)
    costs = ((df["fee_bps"] + df["spread_bps"] + df["slippage_bps"] * 2.0).to_numpy() / 1e4).astype(np.float64)

    # Map sanitized rows back to original OHLC indices
    # Use close prices to find matching indices
    close_prices = df.select("close_raw" if "close_raw" in df.columns else FEATURE_COLUMNS[0]).to_numpy()

    splits = purged_walk_forward_splits(
        n_samples=len(df), train_size=train_size, test_size=test_size,
        purge=purge, embargo=embargo,
    )
    if not splits:
        return [{"symbol": symbol, "label": c[0], "status": "no_folds"}
                for c in TRAIL_CONFIGS]

    # First: get ML predictions for all test folds
    all_test_indices = []
    all_probas = []

    for tr_idx, te_idx in splits:
        dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
        dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

        obj = make_ev_custom_objective(
            float(np.mean(avg_win[tr_idx])),
            float(np.mean(avg_loss[tr_idx])),
            float(np.mean(costs[tr_idx])),
        )
        params = {
            "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
        }
        bst = xgb.train(
            params, dtrain, num_boost_round=500, obj=obj,
            evals=[(dtest, "valid")], early_stopping_rounds=50,
            verbose_eval=False,
        )
        proba = bst.predict(dtest)
        all_test_indices.append(te_idx)
        all_probas.append(proba)

    test_indices = np.concatenate(all_test_indices)
    test_probas = np.concatenate(all_probas)
    test_avg_win = avg_win[test_indices]
    test_avg_loss = avg_loss[test_indices]
    test_costs = costs[test_indices]

    # Determine which trades to take (same for all trail configs)
    ev = test_probas * test_avg_win - (1 - test_probas) * test_avg_loss - test_costs
    take_mask = (ev > 0) & (test_probas >= 0.5)

    # Get the side from features
    side_col = feat_base["primary_side"].to_numpy()

    results = []

    for label, be_atr, trail_act, trail_dist in TRAIL_CONFIGS:
        t0 = time.time()
        trade_returns = []

        for i, idx in enumerate(test_indices):
            if not take_mask[i]:
                continue

            # Get entry info
            entry_price = ohlc[idx, 3]  # close of signal bar
            atr = atr_arr[idx]
            direction = int(side_col[idx])

            if atr <= 0 or entry_price <= 0:
                continue

            # Get future bars for simulation
            future_start = idx + 1
            future_end = min(idx + 25, len(ohlc))
            if future_start >= len(ohlc):
                continue

            future_bars = ohlc[future_start:future_end]

            ret = simulate_trade_with_trail(
                future_bars, atr, entry_price, direction,
                sl_mult, tp_mult,
                be_atr, trail_act, trail_dist,
                horizon=24, cost_pct=cost_pct,
            )
            trade_returns.append(ret)

        trade_returns = np.array(trade_returns) if trade_returns else np.array([0.0])
        trades = trade_returns[trade_returns != 0]
        n = len(trades)

        if n > 0:
            wins = trades[trades > 0]
            losses = trades[trades < 0]
            mean_ret = np.mean(trades)
            std_ret = np.std(trades)
            sharpe = np.sqrt(252) * mean_ret / std_ret if std_ret > 0 else 0
            cumsum = np.cumsum(trades)
            running_max = np.maximum.accumulate(cumsum)
            max_dd = np.max(running_max - cumsum)
            pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999
            wr = len(wins) / n

            results.append({
                "symbol": symbol, "label": label, "status": "ok",
                "n_trades": n, "ev": float(mean_ret), "sharpe": float(sharpe),
                "win_rate": float(wr), "max_dd": float(max_dd),
                "pf": float(pf), "total_ret": float(np.sum(trades)),
                "be_atr": be_atr, "trail_act": trail_act, "trail_dist": trail_dist,
                "elapsed_s": round(time.time() - t0, 1),
            })
        else:
            results.append({
                "symbol": symbol, "label": label, "status": "ok",
                "n_trades": 0, "ev": 0, "sharpe": 0,
                "win_rate": 0, "max_dd": 0, "pf": 0, "total_ret": 0,
                "be_atr": be_atr, "trail_act": trail_act, "trail_dist": trail_dist,
                "elapsed_s": round(time.time() - t0, 1),
            })

    return results


def main():
    p = argparse.ArgumentParser(description="Trail settings comparison backtest")
    p.add_argument("--symbols", type=str, required=True)
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})

    print(f"[trail_comparison] {len(symbols)} symbols")
    print(f"  Testing: {[c[0] for c in TRAIL_CONFIGS]}")
    print()

    all_results = []
    for sym in symbols:
        sl = configs.get(sym, {}).get("atr_sl_mult", 1.0)
        tp = configs.get(sym, {}).get("atr_tp_mult", 2.5)
        print(f"  {sym} (SL={sl}×, TP={tp}×)...")
        results = run_trail_test(sym, configs)
        all_results.extend(results)
        for r in results:
            if r.get("status") == "ok":
                print(f"    {r['label']:<16} trades={r['n_trades']:>4}  "
                      f"EV={r['ev']:+.6f}  Sharpe={r['sharpe']:+.3f}  "
                      f"WR={r['win_rate']:.1%}  PF={r['pf']:.2f}")
        print()

    # Summary table
    print("=" * 120)
    header = (f"  {'Symbol':<12} │ {'Config':<16} │ {'BE':>4} │ {'Act':>4} │ {'Dist':>4} │ "
              f"{'Trades':>6} │ {'EV':>10} │ {'Sharpe':>7} │ {'WR':>6} │ {'PF':>6} │ {'TotRet':>9}")
    print(header)
    print("─" * 120)

    prev_sym = None
    for r in all_results:
        sym = r.get("symbol", "?")
        if prev_sym and sym != prev_sym:
            print("─" * 120)
        prev_sym = sym

        if r.get("status") != "ok":
            continue

        be = f"{r['be_atr']:.1f}" if r['be_atr'] < 100 else "off"
        act = f"{r['trail_act']:.1f}" if r['trail_act'] < 100 else "off"
        dist = f"{r['trail_dist']:.1f}" if r['trail_dist'] < 100 else "off"

        best_marker = ""
        sym_results = [x for x in all_results if x.get("symbol") == sym and x.get("status") == "ok"]
        if sym_results and r == max(sym_results, key=lambda x: x.get("ev", -999)):
            best_marker = " <<<"

        print(f"  {sym:<12} │ {r['label']:<16} │ {be:>4} │ {act:>4} │ {dist:>4} │ "
              f"{r['n_trades']:>6} │ {r['ev']:>+10.6f} │ {r['sharpe']:>+7.3f} │ "
              f"{r['win_rate']:>5.1%} │ {r['pf']:>6.2f} │ {r['total_ret']:>+9.4f}{best_marker}")

    print("─" * 120)

    # Winner summary
    print()
    print("  BEST CONFIG PER SYMBOL:")
    print("  " + "─" * 60)
    for sym in symbols:
        sym_results = [r for r in all_results if r["symbol"] == sym and r.get("status") == "ok" and r["n_trades"] > 0]
        if not sym_results:
            print(f"    {sym:<14} no trades")
            continue
        best = max(sym_results, key=lambda r: r["ev"])
        print(f"    {sym:<14} best = {best['label']:<16} "
              f"(EV={best['ev']:+.6f}, BE={best['be_atr']}, "
              f"Act={best['trail_act']}, Dist={best['trail_dist']})")


if __name__ == "__main__":
    main()
