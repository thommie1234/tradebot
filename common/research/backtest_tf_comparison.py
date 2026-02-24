"""
Backtest comparison: H1 vs M30 timeframe.

Tests same symbol on H1 and M30 to see which gives better EV.

Usage:
    python3 research/backtest_tf_comparison.py --symbols NVDA,PFE,RACE,LVMH
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

TIMEFRAMES = ["M30", "H1"]
CONFIG_PATH = REPO_ROOT / "config" / "sovereign_configs.json"


def calc_metrics(returns: np.ndarray) -> dict:
    trades = returns[returns != 0]
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "ev": 0, "sharpe": 0, "win_rate": 0,
                "max_dd": 0, "pf": 0, "total_ret": 0}

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    mean_ret = np.mean(trades)
    std_ret = np.std(trades)
    sharpe = np.sqrt(252) * mean_ret / std_ret if std_ret > 0 else 0
    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = np.max(running_max - cumsum) if len(cumsum) > 0 else 0
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999

    return {
        "n_trades": n,
        "ev": float(mean_ret),
        "sharpe": float(sharpe),
        "win_rate": float(len(wins) / n),
        "max_dd": float(max_dd),
        "pf": float(pf),
        "total_ret": float(np.sum(trades)),
    }


def run_wfo(feat_base: pl.DataFrame, sl_mult: float, tp_mult: float,
            fee_bps: float, spread_bps: float, slippage_bps: float,
            train_size: int = 1200, test_size: int = 300,
            purge: int = 24, embargo: int = 24) -> dict:
    """Run WFO backtest for given features."""

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
    if df.height < train_size + test_size + 200:
        return {"status": "insufficient_rows", "rows": df.height}

    x = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy().astype(np.float32)
    tb_ret = df["tb_ret"].to_numpy().astype(np.float64)
    avg_win = df["avg_win"].to_numpy().astype(np.float64)
    avg_loss = df["avg_loss"].to_numpy().astype(np.float64)
    costs = (
        (df["fee_bps"] + df["spread_bps"] + df["slippage_bps"] * 2.0).to_numpy() / 1e4
    ).astype(np.float64)

    splits = purged_walk_forward_splits(
        n_samples=len(df), train_size=train_size, test_size=test_size,
        purge=purge, embargo=embargo,
    )
    if not splits:
        return {"status": "no_folds", "rows": len(df)}

    all_returns = []
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

        test_costs = costs[te_idx]
        test_tb_ret = tb_ret[te_idx]
        test_avg_win = avg_win[te_idx]
        test_avg_loss = avg_loss[te_idx]

        ev = proba * test_avg_win - (1 - proba) * test_avg_loss - test_costs
        take = (ev > 0) & (proba >= 0.5)
        fold_returns = np.where(take, test_tb_ret - test_costs, 0.0)
        all_returns.append(fold_returns)

    all_returns = np.concatenate(all_returns)
    metrics = calc_metrics(all_returns)
    metrics["status"] = "ok"
    metrics["rows"] = len(df)
    metrics["folds"] = len(splits)
    return metrics


def compare_symbol(symbol: str, configs: dict) -> list[dict]:
    """Test H1 vs M30 for a single symbol."""
    set_polars_threads(4)

    sym_cfg = configs.get(symbol, {})
    sl_mult = sym_cfg.get("atr_sl_mult", 1.0)
    tp_mult = sym_cfg.get("atr_tp_mult", 2.5)

    ticks = load_symbol_ticks(symbol, DATA_ROOTS, years_filter=None)
    if ticks is None:
        return [{"symbol": symbol, "tf": tf, "status": "no_tick_data"}
                for tf in TIMEFRAMES]

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)

    results = []
    for tf in TIMEFRAMES:
        t0 = time.time()
        bars = make_time_bars(ticks.select(["time", "price", "size"]), tf)
        feat_base = build_bar_features(bars, z_threshold=1.0)

        # Adjust train/test sizes for timeframe
        if tf == "M30":
            train_size, test_size = 2400, 600
            purge, embargo = 48, 48
        else:
            train_size, test_size = 1200, 300
            purge, embargo = 24, 24

        try:
            m = run_wfo(feat_base, sl_mult, tp_mult,
                        fee_bps, spread_bps, slippage_bps,
                        train_size=train_size, test_size=test_size,
                        purge=purge, embargo=embargo)
        except Exception as e:
            m = {"status": f"error: {e}", "n_trades": 0}

        m["symbol"] = symbol
        m["tf"] = tf
        m["sl_mult"] = sl_mult
        m["tp_mult"] = tp_mult
        m["bars"] = feat_base.height
        m["elapsed_s"] = round(time.time() - t0, 1)
        results.append(m)
        print(f"    {tf}: {feat_base.height} bars, {m.get('n_trades', 0)} trades, "
              f"EV={m.get('ev', 0):+.6f}, Sharpe={m.get('sharpe', 0):+.3f} ({m['elapsed_s']:.0f}s)")

    return results


def main():
    p = argparse.ArgumentParser(description="H1 vs M30 timeframe comparison")
    p.add_argument("--symbols", type=str, required=True)
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})

    print(f"[tf_comparison] {len(symbols)} symbols")
    print(f"  Comparing: {TIMEFRAMES}")
    print(f"  TP/SL from sovereign_configs.json per symbol")
    print()

    all_results = []
    for sym in symbols:
        sl = configs.get(sym, {}).get("atr_sl_mult", 1.0)
        tp = configs.get(sym, {}).get("atr_tp_mult", 2.5)
        print(f"  {sym} (SL={sl}×, TP={tp}×):")
        results = compare_symbol(sym, configs)
        all_results.extend(results)
        print()

    # Summary table
    print("=" * 110)
    header = (f"  {'Symbol':<14} │ {'TF':<5} │ {'Bars':>7} │ {'Trades':>7} │ "
              f"{'EV':>10} │ {'Sharpe':>8} │ {'WinRate':>8} │ {'MaxDD':>8} │ "
              f"{'PF':>7} │ {'TotalRet':>10}")
    print(header)
    print("─" * 110)

    prev_sym = None
    for r in all_results:
        sym = r.get("symbol", "?")
        tf = r.get("tf", "?")
        if prev_sym and sym != prev_sym:
            print("─" * 110)
        prev_sym = sym

        if r.get("status") != "ok":
            print(f"  {sym:<14} │ {tf:<5} │ {r.get('bars', '?'):>7} │ "
                  f"{'—':>7} │ {'—':>10} │ {'—':>8} │ {'—':>8} │ "
                  f"{'—':>8} │ {'—':>7} │ {r.get('status', '?')}")
            continue

        ev_str = f"{r['ev']:+.6f}"
        print(f"  {sym:<14} │ {tf:<5} │ {r.get('bars', 0):>7} │ "
              f"{r['n_trades']:>7} │ {ev_str:>10} │ {r['sharpe']:>+8.3f} │ "
              f"{r['win_rate']:>7.1%} │ {r['max_dd']:>8.4f} │ "
              f"{r['pf']:>7.2f} │ {r['total_ret']:>+10.4f}")

    print("─" * 110)

    # Winner per symbol
    print()
    print("  WINNER PER SYMBOL:")
    print("  " + "─" * 50)
    for sym in symbols:
        sym_results = [r for r in all_results if r["symbol"] == sym and r.get("status") == "ok"]
        if not sym_results:
            print(f"    {sym:<14} no valid results")
            continue
        best = max(sym_results, key=lambda r: r.get("ev", -999))
        ev_diff = 0
        other = [r for r in sym_results if r["tf"] != best["tf"]]
        if other:
            ev_diff = best["ev"] - other[0].get("ev", 0)
        print(f"    {sym:<14} best = {best['tf']:<5} "
              f"(EV={best['ev']:+.6f}, delta={ev_diff:+.6f})")


if __name__ == "__main__":
    main()
