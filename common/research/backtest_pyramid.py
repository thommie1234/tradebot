"""
Backtest: Pyramiding strategy impact analysis.

Compares baseline (1 position per symbol) vs pyramid (allow 2nd position when
existing trade reaches breakeven and a new same-direction ML signal fires with
confidence >= 0.85).

Usage:
    python3 research/backtest_pyramid.py --symbols NVDA,LVMH
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
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

# Per-symbol pyramid parameters (overridden from CLI / config)
PYRAMID_PARAMS: dict[str, dict] = {
    "NVDA": {"breakeven_atr": 1.5, "sl_mult": 0.501, "tp_mult": 2.0},
    "LVMH": {"breakeven_atr": 0.8, "sl_mult": 0.995, "tp_mult": 2.5},
}

PYRAMID_CONFIDENCE_MIN = 0.85  # min proba for 2nd position
MAX_PYRAMID_LAYERS = 1  # max 1 additional position (never 3+)


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    ret: float
    is_pyramid: bool
    breakeven_bar: int  # bar at which breakeven was reached (-1 if never)


def simulate_single_trade(
    ohlc: np.ndarray,        # full OHLC array
    atr_arr: np.ndarray,     # full ATR array
    entry_idx: int,          # bar index of the signal
    direction: int,          # +1 BUY, -1 SELL
    sl_mult: float,
    tp_mult: float,
    be_atr: float,           # breakeven distance in ATR multiples
    trail_act_atr: float,
    trail_dist_atr: float,
    horizon: int,
    cost_pct: float,
    is_pyramid: bool = False,
) -> TradeResult:
    """Simulate a single trade bar-by-bar. Returns TradeResult with detailed info."""

    entry_price = ohlc[entry_idx, 3]  # close of signal bar
    atr = atr_arr[entry_idx]

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
    breakeven_bar = -1

    future_start = entry_idx + 1
    future_end = min(entry_idx + 1 + horizon, len(ohlc))

    for bar_i in range(future_start, future_end):
        o, h, l, c = ohlc[bar_i]

        if direction == 1:  # BUY
            if l <= sl_price:
                ret = (sl_price - entry_price) / entry_price - cost_pct
                return TradeResult(entry_idx, bar_i, direction, entry_price,
                                   sl_price, ret, is_pyramid, breakeven_bar)
            if h >= tp_price:
                ret = (tp_price - entry_price) / entry_price - cost_pct
                return TradeResult(entry_idx, bar_i, direction, entry_price,
                                   tp_price, ret, is_pyramid, breakeven_bar)
            if h > best_price:
                best_price = h
            profit = best_price - entry_price
            if not be_triggered and profit >= be_dist:
                sl_price = entry_price
                be_triggered = True
                breakeven_bar = bar_i
            if not trailing_active and profit >= trail_act_dist:
                trailing_active = True
            if trailing_active:
                new_sl = best_price - trail_trail_dist
                if new_sl > sl_price:
                    sl_price = new_sl
        else:  # SELL
            if h >= sl_price:
                ret = (entry_price - sl_price) / entry_price - cost_pct
                return TradeResult(entry_idx, bar_i, direction, entry_price,
                                   sl_price, ret, is_pyramid, breakeven_bar)
            if l <= tp_price:
                ret = (entry_price - tp_price) / entry_price - cost_pct
                return TradeResult(entry_idx, bar_i, direction, entry_price,
                                   tp_price, ret, is_pyramid, breakeven_bar)
            if l < best_price:
                best_price = l
            profit = entry_price - best_price
            if not be_triggered and profit >= be_dist:
                sl_price = entry_price
                be_triggered = True
                breakeven_bar = bar_i
            if not trailing_active and profit >= trail_act_dist:
                trailing_active = True
            if trailing_active:
                new_sl = best_price + trail_trail_dist
                if new_sl < sl_price:
                    sl_price = new_sl

    # Horizon exit
    if future_end > future_start:
        last_close = ohlc[future_end - 1, 3]
        if direction == 1:
            ret = (last_close - entry_price) / entry_price - cost_pct
        else:
            ret = (entry_price - last_close) / entry_price - cost_pct
        return TradeResult(entry_idx, future_end - 1, direction, entry_price,
                           last_close, ret, is_pyramid, breakeven_bar)

    return TradeResult(entry_idx, entry_idx, direction, entry_price,
                       entry_price, -cost_pct, is_pyramid, breakeven_bar)


def check_breakeven_reached_at_bar(
    ohlc: np.ndarray,
    atr_arr: np.ndarray,
    entry_idx: int,
    direction: int,
    sl_mult: float,
    tp_mult: float,
    be_atr: float,
    check_bar: int,
) -> bool:
    """Check if a trade opened at entry_idx has reached breakeven by check_bar.

    Simulates bar-by-bar to see if breakeven threshold was hit and trade is
    still alive (not stopped or TP'd).
    """
    entry_price = ohlc[entry_idx, 3]
    atr = atr_arr[entry_idx]
    sl_dist = sl_mult * atr
    tp_dist = tp_mult * atr
    be_dist = be_atr * atr

    if direction == 1:
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    be_reached = False
    best_price = entry_price

    for bar_i in range(entry_idx + 1, min(check_bar + 1, len(ohlc))):
        o, h, l, c = ohlc[bar_i]

        if direction == 1:
            if l <= sl_price:
                return False  # stopped out
            if h >= tp_price:
                return False  # TP hit
            if h > best_price:
                best_price = h
            profit = best_price - entry_price
            if profit >= be_dist:
                be_reached = True
        else:
            if h >= sl_price:
                return False  # stopped out
            if l <= tp_price:
                return False  # TP hit
            if l < best_price:
                best_price = l
            profit = entry_price - best_price
            if profit >= be_dist:
                be_reached = True

    return be_reached


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

@dataclass
class SimMetrics:
    mode: str
    n_trades: int
    n_base_trades: int
    n_pyramid_trades: int
    total_ret: float
    mean_ret: float
    sharpe: float
    max_dd: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float


def compute_metrics(trades: list[TradeResult], mode: str) -> SimMetrics:
    if not trades:
        return SimMetrics(mode=mode, n_trades=0, n_base_trades=0,
                          n_pyramid_trades=0, total_ret=0, mean_ret=0,
                          sharpe=0, max_dd=0, win_rate=0, profit_factor=0,
                          avg_win=0, avg_loss=0)

    rets = np.array([t.ret for t in trades])
    n_base = sum(1 for t in trades if not t.is_pyramid)
    n_pyr = sum(1 for t in trades if t.is_pyramid)
    n = len(rets)

    wins = rets[rets > 0]
    losses = rets[rets < 0]

    total_ret = float(np.sum(rets))
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets))
    sharpe = np.sqrt(252) * mean_ret / std_ret if std_ret > 0 else 0.0

    cumsum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = float(np.max(running_max - cumsum)) if len(cumsum) > 0 else 0.0

    win_rate = float(len(wins) / n) if n > 0 else 0.0
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999.0
    avg_w = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_l = float(np.mean(losses)) if len(losses) > 0 else 0.0

    return SimMetrics(
        mode=mode, n_trades=n, n_base_trades=n_base,
        n_pyramid_trades=n_pyr, total_ret=total_ret,
        mean_ret=mean_ret, sharpe=sharpe, max_dd=max_dd,
        win_rate=win_rate, profit_factor=pf,
        avg_win=avg_w, avg_loss=avg_l,
    )


# ---------------------------------------------------------------------------
# Main backtest logic
# ---------------------------------------------------------------------------

def run_pyramid_test(symbol: str, configs: dict) -> dict:
    """Run WFO with pyramid comparison for a single symbol."""
    set_polars_threads(4)
    t0_total = time.time()

    sym_cfg = configs.get(symbol, {})
    params = PYRAMID_PARAMS.get(symbol, {})
    sl_mult = params.get("sl_mult", sym_cfg.get("atr_sl_mult", 1.0))
    tp_mult = params.get("tp_mult", sym_cfg.get("atr_tp_mult", 2.5))
    be_atr = params.get("breakeven_atr", sym_cfg.get("breakeven_atr", 1.0))
    trail_act_atr = sym_cfg.get("trail_activation_atr", 1.2)
    trail_dist_atr = sym_cfg.get("trail_distance_atr", 0.6)
    horizon = sym_cfg.get("exit_horizon", 24)

    print(f"  [{symbol}] Loading tick data...")
    ticks = load_symbol_ticks(symbol, DATA_ROOTS, years_filter=None)
    if ticks is None:
        return {"symbol": symbol, "status": "no_tick_data"}

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)
    cost_pct = (fee_bps + spread_bps + slippage_bps * 2) / 1e4

    print(f"  [{symbol}] Building bars and features...")
    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat_base = build_bar_features(bars, z_threshold=1.0)

    # OHLC for simulation
    ohlc = feat_base.select(["open", "high", "low", "close"]).to_numpy()

    # Compute ATR14
    h = feat_base["high"].to_numpy()
    l = feat_base["low"].to_numpy()
    c = feat_base["close"].to_numpy()
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    true_range = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_arr = np.convolve(true_range, np.ones(14)/14, mode='full')[:len(true_range)]
    atr_arr[:14] = atr_arr[14]

    # Triple barrier labels for ML training
    tb = apply_triple_barrier(
        close=feat_base["close"].to_numpy(),
        vol_proxy=feat_base["vol20"].to_numpy(),
        side=feat_base["primary_side"].to_numpy(),
        horizon=horizon, pt_mult=tp_mult, sl_mult=sl_mult,
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
        return {"symbol": symbol, "status": "insufficient_rows",
                "rows": df.height}

    x = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy().astype(np.float32)
    avg_win_arr = df["avg_win"].to_numpy().astype(np.float64)
    avg_loss_arr = df["avg_loss"].to_numpy().astype(np.float64)
    costs_arr = ((df["fee_bps"] + df["spread_bps"] + df["slippage_bps"] * 2.0).to_numpy() / 1e4).astype(np.float64)

    splits = purged_walk_forward_splits(
        n_samples=len(df), train_size=train_size, test_size=test_size,
        purge=purge, embargo=embargo,
    )
    if not splits:
        return {"symbol": symbol, "status": "no_folds"}

    print(f"  [{symbol}] WFO training: {len(splits)} folds, "
          f"SL={sl_mult}x, TP={tp_mult}x, BE_ATR={be_atr}, horizon={horizon}")

    # ----- ML Training (Walk-forward) -----
    all_test_indices = []
    all_probas = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
        dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

        obj = make_ev_custom_objective(
            float(np.mean(avg_win_arr[tr_idx])),
            float(np.mean(avg_loss_arr[tr_idx])),
            float(np.mean(costs_arr[tr_idx])),
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
    test_avg_win = avg_win_arr[test_indices]
    test_avg_loss = avg_loss_arr[test_indices]
    test_costs = costs_arr[test_indices]

    # Signal selection: positive EV and proba >= 0.5
    ev = test_probas * test_avg_win - (1 - test_probas) * test_avg_loss - test_costs
    take_mask = (ev > 0) & (test_probas >= 0.5)

    side_col = feat_base["primary_side"].to_numpy()

    # Build signal list: (bar_index_in_feat_base, proba, direction, df_row_index)
    # We need to map from df (sanitized) indices to feat_base (ohlc) indices.
    # The sanitized df dropped rows; we need the feat_base row mapping.
    # Use the close prices to reconstruct mapping.
    # Since df was built from feat filtered on target.is_finite() + drop_nulls,
    # and the indices in test_indices are into df, we need a mapping from
    # df row -> feat_base row.

    # Build the mapping by tracking which rows survived sanitization.
    # We recreate the filtering chain to get the original row indices.
    feat_filtered = feat_base.with_columns([
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
    ]).with_row_index("orig_idx").filter(pl.col("target").is_finite())
    feat_filtered = sanitize_training_frame(feat_filtered)

    # Now select same cols + orig_idx and drop_nulls
    mapping_cols = cols + ["orig_idx"]
    avail_cols = [c for c in mapping_cols if c in feat_filtered.columns]
    df_with_idx = feat_filtered.select(avail_cols).drop_nulls(cols)
    orig_indices = df_with_idx["orig_idx"].to_numpy()

    # Map test indices (into df) to original feat_base indices
    signal_list = []
    for i, df_row in enumerate(test_indices):
        if not take_mask[i]:
            continue
        orig_bar_idx = int(orig_indices[df_row])
        direction = int(side_col[orig_bar_idx])
        proba = float(test_probas[i])
        if direction == 0:
            continue
        if atr_arr[orig_bar_idx] <= 0 or ohlc[orig_bar_idx, 3] <= 0:
            continue
        if orig_bar_idx + 1 >= len(ohlc):
            continue
        signal_list.append((orig_bar_idx, proba, direction))

    # Sort signals by bar index (chronological)
    signal_list.sort(key=lambda x: x[0])
    print(f"  [{symbol}] {len(signal_list)} tradeable signals from {len(splits)} WFO folds")

    # ----- SIMULATION: Baseline vs Pyramid -----

    # ===== BASELINE: 1 position at a time =====
    baseline_trades: list[TradeResult] = []
    active_exit_bar = -1  # bar at which current trade exits

    for bar_idx, proba, direction in signal_list:
        # Skip if we still have an active position
        if bar_idx <= active_exit_bar:
            continue

        trade = simulate_single_trade(
            ohlc, atr_arr, bar_idx, direction,
            sl_mult, tp_mult, be_atr,
            trail_act_atr, trail_dist_atr,
            horizon, cost_pct, is_pyramid=False,
        )
        baseline_trades.append(trade)
        active_exit_bar = trade.exit_bar

    # ===== PYRAMID: allow 2nd position when existing is at breakeven =====
    pyramid_trades: list[TradeResult] = []
    active_positions: list[dict] = []  # list of {"entry_bar", "exit_bar", "direction", "trade"}

    for sig_i, (bar_idx, proba, direction) in enumerate(signal_list):
        # Clean up expired positions
        active_positions = [p for p in active_positions if p["exit_bar"] >= bar_idx]

        n_active = len(active_positions)

        if n_active == 0:
            # No position: open a new one (same as baseline, no proba filter beyond take_mask)
            trade = simulate_single_trade(
                ohlc, atr_arr, bar_idx, direction,
                sl_mult, tp_mult, be_atr,
                trail_act_atr, trail_dist_atr,
                horizon, cost_pct, is_pyramid=False,
            )
            pyramid_trades.append(trade)
            active_positions.append({
                "entry_bar": bar_idx,
                "exit_bar": trade.exit_bar,
                "direction": direction,
                "trade": trade,
            })

        elif n_active == 1:
            existing = active_positions[0]

            # Check if same direction
            if direction != existing["direction"]:
                continue  # opposite direction, skip

            # Check high confidence for pyramid
            if proba < PYRAMID_CONFIDENCE_MIN:
                continue

            # Check if existing position has reached breakeven
            be_reached = check_breakeven_reached_at_bar(
                ohlc, atr_arr,
                existing["entry_bar"], existing["direction"],
                sl_mult, tp_mult, be_atr,
                check_bar=bar_idx,
            )
            if not be_reached:
                continue

            # Open pyramid position
            trade = simulate_single_trade(
                ohlc, atr_arr, bar_idx, direction,
                sl_mult, tp_mult, be_atr,
                trail_act_atr, trail_dist_atr,
                horizon, cost_pct, is_pyramid=True,
            )
            pyramid_trades.append(trade)
            active_positions.append({
                "entry_bar": bar_idx,
                "exit_bar": trade.exit_bar,
                "direction": direction,
                "trade": trade,
            })

        else:
            # Already 2 positions (max pyramid layers reached), skip
            continue

    # Compute metrics
    baseline_metrics = compute_metrics(baseline_trades, "baseline")
    pyramid_metrics = compute_metrics(pyramid_trades, "pyramid")

    # Pyramid-only trade metrics (just the 2nd layer trades)
    pyr_only = [t for t in pyramid_trades if t.is_pyramid]
    pyr_only_metrics = compute_metrics(pyr_only, "pyramid_only")

    elapsed = time.time() - t0_total

    return {
        "symbol": symbol,
        "status": "ok",
        "params": {
            "sl_mult": sl_mult, "tp_mult": tp_mult,
            "breakeven_atr": be_atr, "trail_act_atr": trail_act_atr,
            "trail_dist_atr": trail_dist_atr, "horizon": horizon,
            "pyramid_confidence_min": PYRAMID_CONFIDENCE_MIN,
            "cost_pct": cost_pct,
        },
        "n_signals": len(signal_list),
        "baseline": baseline_metrics,
        "pyramid": pyramid_metrics,
        "pyramid_only": pyr_only_metrics,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_metrics_row(label: str, m: SimMetrics, indent: str = "    "):
    print(f"{indent}{label:<20} "
          f"trades={m.n_trades:>4} (base={m.n_base_trades}, pyr={m.n_pyramid_trades})  "
          f"TotRet={m.total_ret:>+9.4f}  EV={m.mean_ret:>+.6f}  "
          f"Sharpe={m.sharpe:>+7.3f}  WR={m.win_rate:>5.1%}  "
          f"PF={m.profit_factor:>6.2f}  MaxDD={m.max_dd:>.4f}")


def print_results(results: list[dict]):
    print()
    print("=" * 130)
    print("  PYRAMID BACKTEST RESULTS")
    print("=" * 130)

    for r in results:
        sym = r["symbol"]
        if r.get("status") != "ok":
            print(f"\n  {sym}: {r.get('status', 'error')}")
            continue

        p = r["params"]
        bl = r["baseline"]
        py = r["pyramid"]
        po = r["pyramid_only"]

        print(f"\n  {sym}  (SL={p['sl_mult']}x, TP={p['tp_mult']}x, "
              f"BE_ATR={p['breakeven_atr']}, pyramid_min_proba={p['pyramid_confidence_min']}, "
              f"cost={p['cost_pct']*1e4:.1f}bps)")
        print(f"  {'─' * 120}")
        print_metrics_row("Baseline", bl)
        print_metrics_row("Pyramid (all)", py)
        print_metrics_row("Pyramid-only (2nd)", po)

        # Delta analysis
        print(f"  {'─' * 120}")
        if bl.n_trades > 0:
            delta_ret = py.total_ret - bl.total_ret
            delta_sharpe = py.sharpe - bl.sharpe
            delta_dd = py.max_dd - bl.max_dd
            delta_trades = py.n_trades - bl.n_trades
            pct_improvement = (delta_ret / abs(bl.total_ret) * 100) if bl.total_ret != 0 else 0

            print(f"    DELTA (pyramid - baseline):")
            print(f"      Total Return: {delta_ret:>+.4f}  ({pct_improvement:>+.1f}% change)")
            print(f"      Sharpe:       {delta_sharpe:>+.3f}")
            print(f"      Max DD:       {delta_dd:>+.4f}  ({'worse' if delta_dd > 0 else 'better'})")
            print(f"      Extra trades: {delta_trades:>+d} (pyramid adds)")

            if po.n_trades > 0:
                print(f"\n    Pyramid-layer analysis:")
                print(f"      2nd-position trades:  {po.n_trades}")
                print(f"      2nd-pos win rate:     {po.win_rate:.1%}")
                print(f"      2nd-pos avg win:      {po.avg_win:+.6f}")
                print(f"      2nd-pos avg loss:     {po.avg_loss:+.6f}")
                print(f"      2nd-pos total return: {po.total_ret:+.4f}")
            else:
                print(f"\n    No pyramid trades triggered (breakeven + proba>=0.85 never coincided)")

        print(f"  [{sym}] elapsed: {r['elapsed_s']:.1f}s")

    # Summary comparison table
    print(f"\n{'=' * 130}")
    print("  SUMMARY COMPARISON TABLE")
    print(f"{'=' * 130}")
    header = (f"  {'Symbol':<10} │ {'Mode':<20} │ {'Trades':>6} │ {'TotRet':>10} │ "
              f"{'EV':>10} │ {'Sharpe':>8} │ {'WR':>6} │ {'PF':>7} │ {'MaxDD':>8}")
    print(header)
    print(f"  {'─' * 120}")

    for r in results:
        if r.get("status") != "ok":
            continue
        sym = r["symbol"]
        for mode, m in [("baseline", r["baseline"]), ("pyramid", r["pyramid"]),
                         ("pyramid_only(2nd)", r["pyramid_only"])]:
            marker = ""
            if mode == "pyramid" and r["baseline"].total_ret != 0:
                if m.total_ret > r["baseline"].total_ret:
                    marker = " <<<"
            print(f"  {sym:<10} │ {mode:<20} │ {m.n_trades:>6} │ {m.total_ret:>+10.4f} │ "
                  f"{m.mean_ret:>+10.6f} │ {m.sharpe:>+8.3f} │ {m.win_rate:>5.1%} │ "
                  f"{m.profit_factor:>7.2f} │ {m.max_dd:>8.4f}{marker}")
        print(f"  {'─' * 120}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Pyramid strategy backtest")
    p.add_argument("--symbols", type=str, required=True,
                   help="Comma-separated symbols (e.g. NVDA,LVMH)")
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})

    print(f"[pyramid_backtest] {len(symbols)} symbols: {symbols}")
    print(f"  Pyramid rules: max 1 pyramid layer, confidence >= {PYRAMID_CONFIDENCE_MIN}")
    print(f"  Per-symbol params: {PYRAMID_PARAMS}")
    print()

    all_results = []
    for sym in symbols:
        result = run_pyramid_test(sym, configs)
        all_results.append(result)

    print_results(all_results)


if __name__ == "__main__":
    main()
