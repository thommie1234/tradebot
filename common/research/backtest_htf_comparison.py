"""
Backtest comparison: 39 features (base) vs 55 features (base + HTF H4/D1).

Walk-forward backtest on symbols that have both tick data AND bar data.
Outputs a comparison table with EV, Sharpe, win rate, max drawdown.

Usage:
    python3 research/backtest_htf_comparison.py --symbols NVDA,LVMH,AMZN
    python3 research/backtest_htf_comparison.py --active
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import xgboost as xgb

from engine.feature_builder import (
    FEATURE_COLUMNS,
    build_bar_features,
    build_htf_features,
    htf_feature_columns,
    merge_htf_features,
    normalize_bar_columns,
)
from engine.labeling import apply_triple_barrier
from research.integrated_pipeline import (
    expected_value_score,
    make_ev_custom_objective,
    purged_walk_forward_splits,
    set_polars_threads,
)
from research.train_ml_strategy import (
    infer_slippage_bps,
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
BAR_ROOTS = "/home/tradebot/ssd_data_2/bars"


def _load_htf_bars(symbol: str, timeframe: str) -> pl.DataFrame | None:
    """Load pre-downloaded bar data for a symbol from BAR_ROOTS."""
    variants = [symbol, symbol.replace("_", ""), symbol.replace("_", "/")]
    for variant in variants:
        bar_dir = Path(BAR_ROOTS) / timeframe / variant
        if not bar_dir.is_dir():
            continue
        pq_files = sorted(bar_dir.glob("*.parquet"))
        if not pq_files:
            continue
        try:
            df = pl.concat([pl.read_parquet(f) for f in pq_files]).sort("time")
            return normalize_bar_columns(df)
        except Exception:
            continue
    return None


def calc_metrics(returns: np.ndarray) -> dict:
    """Calculate trading metrics from a return series."""
    trades = returns[returns != 0]
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "ev": 0, "sharpe": 0, "sortino": 0,
                "win_rate": 0, "max_dd_pct": 0, "profit_factor": 0}

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    win_rate = len(wins) / n if n > 0 else 0

    mean_ret = np.mean(trades)
    std_ret = np.std(trades)
    sharpe = np.sqrt(252) * mean_ret / std_ret if std_ret > 0 else 0

    downside = np.std(losses) if len(losses) > 0 else 1e-8
    sortino = np.sqrt(252) * mean_ret / downside if downside > 0 else 0

    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999

    return {
        "n_trades": n,
        "ev": float(mean_ret),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(win_rate),
        "max_dd_pct": float(max_dd),
        "profit_factor": float(pf),
    }


def run_backtest(symbol: str, feature_cols: list[str], feat: pl.DataFrame,
                 fee_bps: float, spread_bps: float, slippage_bps: float,
                 train_size: int = 1200, test_size: int = 300,
                 purge: int = 24, embargo: int = 24,
                 horizon: int = 24, pt_mult: float = 2.0, sl_mult: float = 1.5,
                 ) -> dict:
    """Run WFO backtest on prepared features, return metrics."""
    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=horizon, pt_mult=pt_mult, sl_mult=sl_mult,
    )
    feat = feat.with_columns([
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

    cols = feature_cols + ["target", "tb_ret", "avg_win", "avg_loss",
                           "fee_bps", "spread_bps", "slippage_bps"]
    df = feat.select(cols).drop_nulls(cols)
    if df.height < train_size + test_size + 200:
        return {"status": "insufficient_rows", "rows": df.height}

    x = df.select(feature_cols).to_numpy()
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
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cpu",
        }
        bst = xgb.train(
            params, dtrain, num_boost_round=500, obj=obj,
            evals=[(dtest, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        proba = bst.predict(dtest)

        # EV-filtered returns
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


def compare_symbol(symbol: str, timeframe: str = "H1") -> dict | None:
    """Run 39-feat vs 55-feat backtest for a single symbol."""
    set_polars_threads(4)
    t0 = time.time()

    # Load tick data and build bars
    ticks = load_symbol_ticks(symbol, DATA_ROOTS, years_filter=None)
    if ticks is None:
        return {"symbol": symbol, "status": "no_tick_data"}

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)

    bars = make_time_bars(ticks.select(["time", "price", "size"]), timeframe)
    base_feat = build_bar_features(bars, z_threshold=1.0)

    # ── Run 1: Base (39 features) ──
    result_base = run_backtest(
        symbol, FEATURE_COLUMNS, base_feat,
        fee_bps, spread_bps, slippage_bps,
    )

    # ── Run 2: HTF (55 features) ──
    htf_feat = base_feat.clone()
    htf_ok = True
    for htf_tf, prefix in [("H4", "h4"), ("D1", "d1")]:
        htf_bars = _load_htf_bars(symbol, htf_tf)
        if htf_bars is not None and htf_bars.height >= 50:
            htf_f = build_htf_features(htf_bars, prefix)
            htf_feat = merge_htf_features(htf_feat, htf_f, prefix)
        else:
            htf_ok = False
            for col in [f"{prefix}_{n}" for n in
                        ["ret1", "ret3", "ma_cross", "vol20",
                         "atr_ratio", "z20", "adx_proxy", "regime"]]:
                htf_feat = htf_feat.with_columns(pl.lit(0.0).alias(col))

    all_feature_cols = FEATURE_COLUMNS + htf_feature_columns()
    result_htf = run_backtest(
        symbol, all_feature_cols, htf_feat,
        fee_bps, spread_bps, slippage_bps,
    )

    elapsed = time.time() - t0
    return {
        "symbol": symbol,
        "elapsed_s": round(elapsed, 1),
        "htf_data_available": htf_ok,
        "base_39": result_base,
        "htf_55": result_htf,
    }


def parse_args():
    p = argparse.ArgumentParser(description="39-feat vs 55-feat WFO backtest comparison")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--symbols", type=str, help="Comma-separated symbol list")
    grp.add_argument("--active", action="store_true", help="Use active symbols from config")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeframe", type=str, default="H1")
    return p.parse_args()


def main():
    args = parse_args()

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    else:
        config_path = REPO_ROOT / "config" / "sovereign_configs.json"
        with open(config_path) as f:
            symbols = sorted(json.load(f).keys())

    print(f"[htf_comparison] {len(symbols)} symbols | {args.timeframe}")
    print(f"  Comparing: 39 features (base) vs 55 features (base + H4 + D1)")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compare_symbol, sym, args.timeframe): sym
                   for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                print(f"  ERROR {sym}: {e}")
                results.append({"symbol": sym, "status": "error", "error": str(e)})

    # Print comparison table
    print()
    print("=" * 100)
    print(f"  {'Symbol':<14} │ {'Mode':<7} │ {'Trades':>6} │ {'EV':>9} │ {'Sharpe':>7} │ "
          f"{'WinRate':>7} │ {'MaxDD':>8} │ {'PF':>6} │ {'Status'}")
    print("─" * 100)

    ev_improvements = []
    for r in sorted(results, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        if "status" in r and r["status"] != None and "base_39" not in r:
            print(f"  {sym:<14} │ {'':7} │ {'':>6} │ {'':>9} │ {'':>7} │ "
                  f"{'':>7} │ {'':>8} │ {'':>6} │ {r.get('status', '?')}")
            continue

        for mode, key in [("Base39", "base_39"), ("HTF55", "htf_55")]:
            m = r[key]
            if m.get("status") != "ok":
                print(f"  {sym:<14} │ {mode:<7} │ {'':>6} │ {'':>9} │ {'':>7} │ "
                      f"{'':>7} │ {'':>8} │ {'':>6} │ {m.get('status', '?')}")
                continue
            print(f"  {sym:<14} │ {mode:<7} │ {m['n_trades']:>6} │ "
                  f"{m['ev']:>+9.6f} │ {m['sharpe']:>7.3f} │ "
                  f"{m['win_rate']:>6.1%} │ {m['max_dd_pct']:>8.4f} │ "
                  f"{m['profit_factor']:>6.2f} │ ok ({r['elapsed_s']:.0f}s)")

        # Track EV improvement
        b = r["base_39"]
        h = r["htf_55"]
        if b.get("status") == "ok" and h.get("status") == "ok":
            ev_diff = h["ev"] - b["ev"]
            ev_improvements.append((sym, b["ev"], h["ev"], ev_diff))

        print("─" * 100)

    # Summary
    if ev_improvements:
        print()
        print("  SUMMARY: EV improvement (HTF55 - Base39)")
        print("  " + "─" * 60)
        better = 0
        worse = 0
        for sym, base_ev, htf_ev, diff in sorted(ev_improvements, key=lambda x: -x[3]):
            arrow = "▲" if diff > 0 else "▼"
            print(f"    {sym:<14}  Base={base_ev:+.6f}  HTF={htf_ev:+.6f}  Δ={diff:+.6f} {arrow}")
            if diff > 0:
                better += 1
            else:
                worse += 1

        print(f"\n  Better: {better}/{len(ev_improvements)}, "
              f"Worse: {worse}/{len(ev_improvements)}")
        avg_diff = np.mean([d for _, _, _, d in ev_improvements])
        print(f"  Average EV change: {avg_diff:+.6f}")


if __name__ == "__main__":
    main()
