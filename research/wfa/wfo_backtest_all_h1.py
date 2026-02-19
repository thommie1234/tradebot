"""
Walk-Forward Out-of-Sample Backtest for ALL symbols from Optuna CSV.

Reads best params from Optuna summary, runs WFO per symbol, outputs
portfolio analysis with max DD constraint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from research.integrated_pipeline import (
    make_ev_custom_objective,
    purged_walk_forward_splits,
)
from engine.labeling import apply_triple_barrier
from research.train_ml_strategy import (
    infer_slippage_bps,
    infer_spread_bps,
    make_time_bars,
    sanitize_training_frame,
)


def load_ticks(symbol: str, data_root: str) -> pl.DataFrame:
    files = sorted(str(f) for f in Path(data_root, symbol).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data for {symbol}")
    lf = pl.concat([
        pl.scan_parquet(f).select(["time", "bid", "ask", "last", "volume", "volume_real"])
        for f in files
    ])
    return lf.with_columns([
        pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.when(pl.col("last") > 0)
        .then(pl.col("last"))
        .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
        .alias("price"),
        pl.when(pl.col("volume_real") > 0)
        .then(pl.col("volume_real"))
        .otherwise(pl.col("volume"))
        .alias("size"),
    ]).drop_nulls(["time", "price", "size"]).select(["time", "bid", "ask", "price", "size"]).collect()


def run_wfo_backtest(
    symbol: str,
    data_root: str,
    xgb_params: dict,
    num_boost_round: int,
    train_size: int = 400,
    test_size: int = 100,
    purge: int = 12,
    embargo: int = 12,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 6,
) -> dict:
    """Run walk-forward backtest and return results."""
    ticks = load_ticks(symbol, data_root)
    spread_bps = infer_spread_bps(ticks)
    slippage_bps = infer_slippage_bps(symbol)

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat = build_bar_features(bars, z_threshold=1.5)

    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=horizon,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
    )
    feat = feat.with_columns([
        pl.Series("label", tb.label),
        pl.Series("target", tb.label),
        pl.Series("tb_ret", tb.tb_ret),
        pl.Series("avg_win", tb.upside),
        pl.Series("avg_loss", tb.downside),
        pl.Series("upside", tb.upside),
        pl.Series("downside", tb.downside),
        pl.lit(3.0).alias("fee_bps"),
        pl.lit(float(spread_bps)).alias("spread_bps"),
        pl.lit(float(slippage_bps)).alias("slippage_bps"),
    ]).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    x = feat.select(FEATURE_COLUMNS).to_numpy()
    y = feat["target"].to_numpy().astype(np.float32)
    tb_ret = feat["tb_ret"].to_numpy().astype(np.float64)
    avg_win = feat["avg_win"].to_numpy().astype(np.float64)
    avg_loss = feat["avg_loss"].to_numpy().astype(np.float64)
    costs = (
        (feat["fee_bps"] + feat["spread_bps"] + feat["slippage_bps"] * 2.0).to_numpy() / 1e4
    ).astype(np.float64)
    times = feat["time"].to_list()
    sides = feat["primary_side"].to_numpy()

    splits = purged_walk_forward_splits(
        n_samples=len(feat),
        train_size=train_size,
        test_size=test_size,
        purge=purge,
        embargo=embargo,
    )
    if not splits:
        return None

    all_oos = []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
        dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

        fold_avg_win = float(np.mean(avg_win[tr_idx]))
        fold_avg_loss = float(np.mean(avg_loss[tr_idx]))
        fold_cost = float(np.mean(costs[tr_idx]))
        obj = make_ev_custom_objective(fold_avg_win, fold_avg_loss, fold_cost)

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=[(dtest, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        proba = bst.predict(dtest)

        for j, idx in enumerate(te_idx):
            all_oos.append({
                "fold": fold_i,
                "idx": int(idx),
                "time": times[idx],
                "proba": float(proba[j]),
                "label": float(y[idx]),
                "tb_ret": float(tb_ret[idx]),
                "side": int(sides[idx]),
                "cost": float(costs[idx]),
            })

    oos = pl.from_dicts(all_oos).sort("idx")

    results = {}
    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60]:
        filtered = oos.filter(pl.col("proba") >= threshold)
        if filtered.height == 0:
            results[threshold] = {"trades": 0}
            continue

        rets = filtered["tb_ret"].to_numpy()
        labels = filtered["label"].to_numpy()
        n_trades = len(rets)
        wins = int((labels == 1.0).sum())
        wr = wins / max(n_trades, 1)

        equity = np.cumprod(1.0 + rets)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        total_ret = float(equity[-1] - 1.0)

        gross_win = float(np.sum(rets[rets > 0])) if (rets > 0).any() else 0
        gross_loss = float(abs(np.sum(rets[rets < 0]))) if (rets < 0).any() else 1e-9
        pf = gross_win / gross_loss

        avg_trade = float(np.mean(rets))

        if np.std(rets) > 0:
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(min(n_trades, 250)))
        else:
            sharpe = 0.0

        results[threshold] = {
            "trades": n_trades,
            "wins": wins,
            "losses": n_trades - wins,
            "win_rate": wr,
            "total_return": total_ret,
            "max_dd": max_dd,
            "profit_factor": pf,
            "avg_trade": avg_trade,
            "sharpe": sharpe,
        }

    return {
        "symbol": symbol,
        "total_oos_bars": oos.height,
        "folds": len(splits),
        "results_by_threshold": results,
    }


def load_optuna_csv(csv_path: str) -> list[dict]:
    """Load Optuna summary CSV and return list of symbol configs."""
    df = pl.read_csv(csv_path)
    ok = df.filter(pl.col("status") == "ok")
    configs = []
    for row in ok.iter_rows(named=True):
        params = {
            "booster": "gbtree", "tree_method": "hist", "device": "cuda",
            "sampling_method": "gradient_based",
            "objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": int(row["best_max_depth"]),
            "eta": 0.03,
            "gamma": float(row["best_gamma"]),
            "subsample": float(row["best_subsample"]),
            "colsample_bytree": float(row["best_colsample_bytree"]),
            "colsample_bylevel": 0.75,
            "reg_alpha": float(row["best_reg_alpha"]),
            "reg_lambda": float(row["best_reg_lambda"]),
            "min_child_weight": float(row["best_min_child_weight"]),
            "max_bin": 512,
            "grow_policy": "lossguide",
            "verbosity": 0,
        }
        configs.append({
            "symbol": row["symbol"],
            "params": params,
            "rounds": int(row["best_num_boost_round"]),
            "optuna_ev": float(row["best_ev"]),
        })
    return configs


def print_portfolio(all_results: dict, account: float, max_dd_pct: float):
    """Portfolio analysis with max DD constraint."""
    dd_budget = account * max_dd_pct

    for threshold in [0.50, 0.52, 0.55]:
        viable = []
        for sym, res in all_results.items():
            r = res["results_by_threshold"].get(threshold, {})
            if r.get("trades", 0) >= 20 and r.get("profit_factor", 0) > 1.0:
                viable.append((sym, r, res["total_oos_bars"]))

        print(f"\n  === Threshold {threshold:.2f} — {len(viable)} viable symbols ===")
        if not viable:
            print("  No viable symbols")
            continue

        n = len(viable)
        dd_per = dd_budget / n

        print(f"  DD budget/symbol: ${dd_per:,.0f}")
        print(f"\n  {'Symbol':>14s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
              f"{'TotRet':>9s} {'MaxDD':>9s} {'Sharpe':>7s} "
              f"{'Alloc$':>10s} {'P&L$':>10s} {'$/mo':>8s}")
        print(f"  {'-'*104}")

        total_pnl = 0
        total_alloc = 0

        for sym, r, oos_bars in sorted(viable, key=lambda x: -x[1]["profit_factor"]):
            sym_dd = max(abs(r["max_dd"]), 0.01)
            alloc = dd_per / sym_dd
            pnl = alloc * r["total_return"]
            months = max(oos_bars / (6.5 * 21), 1)
            pnl_mo = pnl / months
            total_pnl += pnl
            total_alloc += alloc

            print(
                f"  {sym:>14s} {r['trades']:>7d} {r['win_rate']:>6.1%} "
                f"{r['profit_factor']:>7.3f} {r['total_return']:>+8.2%} "
                f"{r['max_dd']:>8.2%} {r['sharpe']:>+7.2f} "
                f"${alloc:>9,.0f} ${pnl:>+9,.0f} ${pnl_mo:>+7,.0f}"
            )

        print(f"  {'-'*104}")
        months_avg = np.mean([ob / (6.5 * 21) for _, _, ob in viable])
        print(f"  {'TOTAL':>14s} {'':>7s} {'':>7s} {'':>7s} {'':>9s} "
              f"{'':>9s} {'':>7s} "
              f"${total_alloc:>9,.0f} ${total_pnl:>+9,.0f} ${total_pnl/max(months_avg,1):>+7,.0f}")
        ret_pct = total_pnl / account * 100
        print(f"  Return on ${account/1000:.0f}k: {ret_pct:+.2f}% ({ret_pct/max(months_avg,1):+.2f}%/mo)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Optuna summary CSV path")
    parser.add_argument("--data-root", default="/home/tradebot/ssd_data_1/tick_data")
    parser.add_argument("--account", type=float, default=100_000)
    parser.add_argument("--max-dd", type=float, default=0.05, help="Max DD as fraction")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=100)
    args = parser.parse_args()

    configs = load_optuna_csv(args.csv)
    print(f"Loaded {len(configs)} symbols from {args.csv}")

    # Sort by optuna_ev descending (best first)
    configs.sort(key=lambda c: -c["optuna_ev"])

    all_results = {}
    for i, cfg in enumerate(configs):
        sym = cfg["symbol"]
        print(f"\n[{i+1}/{len(configs)}] {'='*60}")
        print(f"  {sym} (Optuna EV: {cfg['optuna_ev']:+.6f})")
        print(f"  {'='*60}")

        try:
            result = run_wfo_backtest(
                symbol=sym,
                data_root=args.data_root,
                xgb_params=cfg["params"],
                num_boost_round=cfg["rounds"],
                train_size=args.train_size,
                test_size=args.test_size,
                purge=12,
                embargo=12,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if result is None:
            print(f"  SKIP: not enough data for splits")
            continue

        all_results[sym] = result
        # Print threshold 0.50 summary
        r = result["results_by_threshold"].get(0.50, {})
        if r.get("trades", 0) > 0:
            print(f"  Folds: {result['folds']}, OOS: {result['total_oos_bars']} bars, "
                  f"Trades: {r['trades']}, WR: {r['win_rate']:.1%}, PF: {r['profit_factor']:.3f}, "
                  f"Ret: {r['total_return']:+.2%}, DD: {r['max_dd']:.2%}")
        else:
            print(f"  Folds: {result['folds']}, OOS: {result['total_oos_bars']} bars, 0 trades")

    # Full summary
    print(f"\n\n{'='*80}")
    print(f"  ALL SYMBOLS RESULTS (threshold 0.50)")
    print(f"{'='*80}")
    print(f"\n  {'Symbol':>14s} {'OptEV':>10s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
          f"{'TotRet':>9s} {'MaxDD':>9s} {'Sharpe':>7s}")
    print(f"  {'-'*78}")

    for cfg in configs:
        sym = cfg["symbol"]
        if sym not in all_results:
            print(f"  {sym:>14s} {cfg['optuna_ev']:>+10.6f}  SKIPPED")
            continue
        r = all_results[sym]["results_by_threshold"].get(0.50, {})
        if r.get("trades", 0) == 0:
            print(f"  {sym:>14s} {cfg['optuna_ev']:>+10.6f}  0 trades")
            continue
        marker = " ***" if r.get("profit_factor", 0) > 1.0 else ""
        print(
            f"  {sym:>14s} {cfg['optuna_ev']:>+10.6f} {r['trades']:>7d} {r['win_rate']:>6.1%} "
            f"{r['profit_factor']:>7.3f} {r['total_return']:>+8.2%} "
            f"{r['max_dd']:>8.2%} {r['sharpe']:>+7.2f}{marker}"
        )

    # Portfolio analysis
    print(f"\n\n{'='*80}")
    print(f"  PORTFOLIO — ${args.account:,.0f}, {args.max_dd:.0%} max DD (${args.account*args.max_dd:,.0f})")
    print(f"{'='*80}")
    print_portfolio(all_results, args.account, args.max_dd)
