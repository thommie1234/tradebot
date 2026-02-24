"""
Backtest comparison: different TP multipliers.

Tests TP at 2.0×, 2.5×, 3.5×, 5.0×, 8.0× ATR with same SL and ML model.
Shows which TP level gives best EV, Sharpe, win rate.

Usage:
    python3 research/backtest_tp_comparison.py --active
    python3 research/backtest_tp_comparison.py --symbols NVDA,JP225.cash
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

# TP multipliers to test
TP_MULTS = [2.0, 2.5, 3.5, 5.0, 8.0]

# Load sovereign configs for SL multiplier per symbol
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


def run_tp_test(symbol: str, tp_mult: float, sl_mult: float,
                x: np.ndarray, y: np.ndarray, tb_ret_all: dict,
                avg_win_all: dict, avg_loss_all: dict, costs_arr: np.ndarray,
                feat_base: pl.DataFrame, fee_bps: float, spread_bps: float,
                slippage_bps: float,
                train_size: int = 1200, test_size: int = 300,
                purge: int = 24, embargo: int = 24) -> dict:
    """Run WFO backtest for a specific TP multiplier."""

    tb = apply_triple_barrier(
        close=feat_base["close"].to_numpy(),
        vol_proxy=feat_base["vol20"].to_numpy(),
        side=feat_base["primary_side"].to_numpy(),
        horizon=24, pt_mult=tp_mult, sl_mult=sl_mult,
    )

    # Build frame with this TP's labels
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


def compare_symbol(symbol: str, configs: dict) -> dict:
    """Test all TP multipliers for a single symbol."""
    set_polars_threads(4)
    t0 = time.time()

    sym_cfg = configs.get(symbol, {})
    sl_mult = sym_cfg.get("atr_sl_mult", 1.0)

    # Load data
    ticks = load_symbol_ticks(symbol, DATA_ROOTS, years_filter=None)
    if ticks is None:
        return {"symbol": symbol, "status": "no_tick_data"}

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat_base = build_bar_features(bars, z_threshold=1.0)

    results = {"symbol": symbol, "sl_mult": sl_mult}

    for tp_mult in TP_MULTS:
        key = f"tp_{tp_mult}"
        try:
            m = run_tp_test(
                symbol, tp_mult, sl_mult,
                None, None, {}, {}, {}, None,
                feat_base, fee_bps, spread_bps, slippage_bps,
            )
            results[key] = m
        except Exception as e:
            results[key] = {"status": f"error: {e}", "n_trades": 0}

    results["elapsed_s"] = round(time.time() - t0, 1)
    return results


def main():
    p = argparse.ArgumentParser(description="TP multiplier comparison backtest")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--symbols", type=str)
    grp.add_argument("--active", action="store_true")
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    else:
        symbols = sorted(configs.keys())

    print(f"[tp_comparison] {len(symbols)} symbols")
    print(f"  Testing TP multipliers: {TP_MULTS}")
    print(f"  SL from sovereign_configs.json per symbol")
    print()

    all_results = []
    for sym in symbols:
        sl = configs.get(sym, {}).get("atr_sl_mult", "?")
        print(f"  {sym} (SL={sl}×ATR)...", end=" ", flush=True)
        r = compare_symbol(sym, configs)
        all_results.append(r)
        if "elapsed_s" in r:
            print(f"done ({r['elapsed_s']:.0f}s)")
        else:
            print(r.get("status", "?"))

    # Print results table
    print()
    print("=" * 120)
    header = f"  {'Symbol':<14} │ {'TP mult':>7}"
    for label in ["Trades", "EV", "Sharpe", "WinRate", "MaxDD", "PF", "TotalRet"]:
        header += f" │ {label:>9}"
    print(header)
    print("─" * 120)

    # Track best TP per symbol
    best_tp = {}

    for r in sorted(all_results, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        if "status" in r and r.get("status") == "no_tick_data":
            print(f"  {sym:<14} │ no tick data")
            print("─" * 120)
            continue

        best_ev = -999
        best_mult = None

        for tp_mult in TP_MULTS:
            key = f"tp_{tp_mult}"
            m = r.get(key, {})
            if m.get("status") != "ok":
                print(f"  {sym:<14} │ {tp_mult:>5.1f}×  │ {m.get('status', '?')}")
                continue

            marker = ""
            if m["ev"] > best_ev:
                best_ev = m["ev"]
                best_mult = tp_mult

            print(f"  {sym:<14} │ {tp_mult:>5.1f}×  │ {m['n_trades']:>9} │ "
                  f"{m['ev']:>+9.6f} │ {m['sharpe']:>9.3f} │ "
                  f"{m['win_rate']:>8.1%} │ {m['max_dd']:>9.4f} │ "
                  f"{m['pf']:>9.2f} │ {m['total_ret']:>+9.4f}")

        if best_mult is not None:
            best_tp[sym] = (best_mult, best_ev)

        print("─" * 120)

    # Summary
    print()
    print("  BEST TP PER SYMBOL (by EV)")
    print("  " + "─" * 50)
    tp_votes = {}
    for sym, (tp, ev) in sorted(best_tp.items()):
        print(f"    {sym:<14}  best TP = {tp:.1f}× ATR  (EV={ev:+.6f})")
        tp_votes[tp] = tp_votes.get(tp, 0) + 1

    print()
    print("  TP VOTE TALLY:")
    for tp, count in sorted(tp_votes.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {tp:.1f}× ATR: {bar} ({count})")


if __name__ == "__main__":
    main()
