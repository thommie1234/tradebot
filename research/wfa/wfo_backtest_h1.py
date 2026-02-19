"""
Walk-Forward Out-of-Sample Backtest with ML Filtering.

Trains XGBoost on each fold's training data using the best Optuna params,
predicts on out-of-sample test data, and only takes trades where the model
predicts probability > threshold. Builds a proper equity curve.

Portfolio-level analysis with max drawdown constraint for prop firm compliance.
"""
from __future__ import annotations

import json
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
    train_size: int = 1200,
    test_size: int = 300,
    purge: int = 24,
    embargo: int = 24,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 6,
    prob_threshold: float = 0.50,
) -> dict:
    """Run walk-forward backtest and return results."""
    ticks = load_ticks(symbol, data_root)
    spread_bps = infer_spread_bps(ticks)
    slippage_bps = infer_slippage_bps(symbol)

    # Fallback for zero-volume feeds (CFD data from forex brokers)
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

    # Extract arrays
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

    # Collect out-of-sample predictions
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

    # Apply ML filter: only take trades where model is confident
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
        losses = n_trades - wins
        wr = wins / max(n_trades, 1)

        # Equity curve
        equity = np.cumprod(1.0 + rets)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        total_ret = float(equity[-1] - 1.0)

        # Profit factor
        gross_win = float(np.sum(rets[rets > 0])) if (rets > 0).any() else 0
        gross_loss = float(abs(np.sum(rets[rets < 0]))) if (rets < 0).any() else 1e-9
        pf = gross_win / gross_loss

        # Avg trade
        avg_trade = float(np.mean(rets))

        # Sharpe (annualized, ~250 trading days, assume ~1 trade/day on H1)
        if np.std(rets) > 0:
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(min(n_trades, 250)))
        else:
            sharpe = 0.0

        results[threshold] = {
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
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
        "oos_trades": oos,  # raw OOS for portfolio analysis
    }


def _make_params(md, gamma, ss, csbt, ra, rl, mcw, rounds):
    """Helper to build XGBoost param dict from Optuna best values."""
    return {
        "params": {
            "booster": "gbtree", "tree_method": "hist", "device": "cuda",
            "sampling_method": "gradient_based",
            "objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": md, "eta": 0.03, "gamma": gamma,
            "subsample": ss, "colsample_bytree": csbt, "colsample_bylevel": 0.75,
            "reg_alpha": ra, "reg_lambda": rl,
            "min_child_weight": mcw, "max_bin": 512,
            "grow_policy": "lossguide", "verbosity": 0,
        },
        "rounds": rounds,
    }


def portfolio_analysis(all_results: dict, account: float = 100_000, max_dd_pct: float = 0.05):
    """
    Portfolio-level analysis with max DD constraint.

    For each symbol at each threshold, calculate position size such that
    the symbol's historical max DD maps to its share of the total DD budget.
    Equal DD budget allocation across all active symbols.
    """
    print(f"\n{'='*80}")
    print(f"  PORTFOLIO ANALYSIS — ${account:,.0f} account, {max_dd_pct:.0%} max DD (${account*max_dd_pct:,.0f})")
    print(f"{'='*80}")

    dd_budget = account * max_dd_pct

    for threshold in [0.50, 0.52, 0.55, 0.58, 0.60]:
        print(f"\n  --- Threshold {threshold:.2f} ---")

        # Collect symbols with positive PF AND enough trades
        viable = []
        for sym, res in all_results.items():
            r = res["results_by_threshold"].get(threshold, {})
            if r.get("trades", 0) >= 20 and r.get("profit_factor", 0) > 1.0:
                viable.append((sym, r))

        if not viable:
            print("  No viable symbols (PF > 1.0, trades >= 20)")
            continue

        n_symbols = len(viable)
        dd_per_symbol = dd_budget / n_symbols

        print(f"  Viable symbols: {n_symbols} | DD budget/symbol: ${dd_per_symbol:,.0f}")
        print()
        print(f"  {'Symbol':>12s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'TotRet':>9s} "
              f"{'MaxDD':>9s} {'Sharpe':>7s} {'Alloc$':>10s} {'P&L$':>10s} {'$/mo':>8s}")
        print(f"  {'-'*98}")

        total_pnl = 0
        total_alloc = 0
        portfolio_max_dd_dollar = 0

        for sym, r in sorted(viable, key=lambda x: -x[1]["profit_factor"]):
            sym_max_dd = abs(r["max_dd"])
            if sym_max_dd < 1e-6:
                sym_max_dd = 0.01  # floor at 1%

            # Position size: allocate so that sym_max_dd * alloc = dd_per_symbol
            alloc = dd_per_symbol / sym_max_dd
            pnl = alloc * r["total_return"]
            dd_dollar = alloc * sym_max_dd

            # Estimate months from OOS period (H1 bars / ~6.5 per day / ~21 per month)
            oos_bars = all_results[sym]["total_oos_bars"]
            months = max(oos_bars / (6.5 * 21), 1)
            pnl_per_month = pnl / months

            total_pnl += pnl
            total_alloc += alloc
            portfolio_max_dd_dollar += dd_dollar

            print(
                f"  {sym:>12s} {r['trades']:>7d} {r['win_rate']:>6.1%} "
                f"{r['profit_factor']:>7.3f} {r['total_return']:>+8.2%} "
                f"{r['max_dd']:>8.2%} {r['sharpe']:>+7.2f} "
                f"${alloc:>9,.0f} ${pnl:>+9,.0f} ${pnl_per_month:>+7,.0f}"
            )

        print(f"  {'-'*98}")
        months_est = np.mean([
            all_results[s]["total_oos_bars"] / (6.5 * 21) for s, _ in viable
        ])
        print(
            f"  {'TOTAL':>12s} {'':>7s} {'':>7s} {'':>7s} {'':>9s} "
            f"{'':>9s} {'':>7s} "
            f"${total_alloc:>9,.0f} ${total_pnl:>+9,.0f} ${total_pnl/max(months_est,1):>+7,.0f}"
        )
        pct_return_on_account = total_pnl / account * 100
        print(f"  Return on ${account/1000:.0f}k: {pct_return_on_account:+.2f}% "
              f"({pct_return_on_account/max(months_est,1):+.2f}%/mo)")

    # Also print ALL symbols summary regardless of viability
    print(f"\n\n{'='*80}")
    print(f"  ALL SYMBOLS SUMMARY (threshold 0.50)")
    print(f"{'='*80}")
    print(f"\n  {'Symbol':>12s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'TotRet':>9s} "
          f"{'MaxDD':>9s} {'Sharpe':>7s} {'AvgTrade':>10s}")
    print(f"  {'-'*78}")
    for sym in sorted(all_results.keys()):
        r = all_results[sym]["results_by_threshold"].get(0.50, {})
        if r.get("trades", 0) == 0:
            print(f"  {sym:>12s} {'0':>7s}")
            continue
        print(
            f"  {sym:>12s} {r['trades']:>7d} {r['win_rate']:>6.1%} "
            f"{r['profit_factor']:>7.3f} {r['total_return']:>+8.2%} "
            f"{r['max_dd']:>8.2%} {r['sharpe']:>+7.2f} "
            f"{r['avg_trade']:>+9.4%}"
        )


if __name__ == "__main__":
    DATA_ROOT = "/home/tradebot/ssd_data_1/tick_data"

    # All 22 symbols — best H1 params from Optuna (20260206_060227)
    symbols_params = {
        "AUD_JPY": _make_params(6, 0.03847, 0.6859, 0.5819, 0.01703, 0.12670, 25.440, 610),
        "CAD_JPY": _make_params(3, 0.01436, 0.8358, 0.8137, 0.00170, 0.00734, 18.682, 301),
        "CHF_JPY": _make_params(4, 0.99618, 0.8159, 0.6752, 0.00011, 0.00013, 1.341, 823),
        "GBP_CAD": _make_params(7, 0.02080, 0.7774, 0.6059, 0.25850, 0.00125, 16.283, 568),
        "GBP_NZD": _make_params(6, 0.02255, 0.6839, 0.7591, 0.35407, 0.00017, 3.224, 1056),
        "GOOG": _make_params(6, 0.48279, 0.7332, 0.6631, 0.10527, 0.08197, 7.991, 233),
        "LVMH": _make_params(4, 0.33944, 0.8782, 0.7589, 0.00174, 0.01467, 1.557, 556),
        "MSFT": _make_params(5, 0.01700, 0.7941, 0.5402, 0.00015, 0.00777, 0.588, 437),
        "NATGAS.cash": _make_params(7, 0.10044, 0.7319, 0.5122, 0.08019, 0.00020, 10.461, 372),
        "NVDA": _make_params(5, 0.01543, 0.7710, 0.8045, 0.01068, 0.14203, 1.635, 527),
        "T": _make_params(5, 0.07013, 0.6512, 0.7706, 0.00289, 0.01187, 5.115, 767),
        "TSLA": _make_params(7, 0.13146, 0.6674, 0.6535, 0.00054, 0.00022, 6.620, 806),
        "USD_CAD": _make_params(7, 0.77706, 0.6033, 0.5039, 0.00050, 0.00056, 0.879, 339),
        "USD_CHF": _make_params(4, 0.05699, 0.8852, 0.7356, 2.83359, 0.00065, 1.560, 535),
        "USD_JPY": _make_params(4, 0.11942, 0.6550, 0.6129, 0.03010, 0.63065, 12.080, 345),
        "USD_NOK": _make_params(7, 0.02568, 0.6573, 0.5477, 7.65445, 8.70465, 8.946, 214),
        "USD_ZAR": _make_params(5, 0.21577, 0.8239, 0.7583, 0.18123, 0.00463, 3.176, 1003),
        "USOIL.cash": _make_params(7, 0.50557, 0.7094, 0.5025, 0.00375, 0.00020, 2.385, 615),
        "V": _make_params(7, 0.12805, 0.7989, 0.5860, 7.02536, 0.02045, 7.366, 1060),
        "WHEAT.c": _make_params(6, 0.02215, 0.6792, 0.7594, 0.00060, 0.11052, 21.172, 301),
        "XAG_USD": _make_params(3, 0.36184, 0.8947, 0.5500, 0.04182, 0.05495, 1.910, 810),
        "XMRUSD": _make_params(6, 0.03207, 0.6087, 0.8301, 0.00362, 0.00501, 18.201, 296),
    }

    all_results = {}

    for sym, cfg in symbols_params.items():
        print(f"\n{'='*70}")
        print(f"  {sym} — Walk-Forward Backtest (H1, ML-Filtered)")
        print(f"{'='*70}")

        try:
            result = run_wfo_backtest(
                symbol=sym,
                data_root=DATA_ROOT,
                xgb_params=cfg["params"],
                num_boost_round=cfg["rounds"],
                train_size=400,
                test_size=100,
                purge=12,
                embargo=12,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        all_results[sym] = result

        print(f"  Folds: {result['folds']}, OOS bars: {result['total_oos_bars']}")
        print()
        print(f"  {'Threshold':>10s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'TotRet':>9s} {'MaxDD':>9s} {'Sharpe':>7s}")
        print(f"  {'-'*58}")

        for thr, r in result["results_by_threshold"].items():
            if r["trades"] == 0:
                print(f"  {thr:>10.2f} {'0':>7s}")
                continue
            print(
                f"  {thr:>10.2f} {r['trades']:>7d} {r['win_rate']:>6.1%} "
                f"{r['profit_factor']:>7.3f} {r['total_return']:>+8.2%} "
                f"{r['max_dd']:>8.2%} {r['sharpe']:>+7.2f}"
            )

    # Portfolio analysis with 5% max DD on $100k
    portfolio_analysis(all_results, account=100_000, max_dd_pct=0.05)
