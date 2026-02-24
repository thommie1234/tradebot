"""
Full backtest of optimized exit parameters on bar data.

Loads sovereign_configs.json, runs simulate_trades() per symbol on its
best exit_timeframe, and produces per-symbol + portfolio equity curves
with realistic costs.

Usage:
    python3 research/backtest_exits.py
    python3 research/backtest_exits.py --max-years 10 --account 100000
    python3 research/backtest_exits.py --symbols NVDA,AMZN
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from research.exit_optuna import (
    CONFIG_PATH,
    compute_atr14,
    find_model_path,
    load_bars_from_parquets,
)
from research.exit_simulator import ExitParams, simulate_trades
from trading_prop.production.optuna_orchestrator import (
    broker_commission_bps,
    broker_slippage_bps,
    cluster_for_symbol,
)

# Exit type labels
EXIT_LABELS = {0: "SL", 1: "TP", 2: "BE", 3: "TRAIL", 4: "HORIZON"}


def backtest_symbol(
    symbol: str,
    cfg: dict,
    max_years: int,
    account_size: float,
) -> dict | None:
    """Run full backtest for one symbol using its config params."""
    tf = cfg.get("exit_timeframe", "H4")
    cluster = cfg.get("asset_class", cluster_for_symbol(symbol))

    # Load bars
    bars = load_bars_from_parquets(symbol, tf)
    if bars is None or bars.height < 500:
        print(f"  {symbol:15s} {tf:4s}  SKIP — no bar data")
        return None

    # Trim to max_years
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_years * 365)
    time_dtype = bars["time"].dtype
    bars = bars.filter(pl.col("time") >= pl.lit(cutoff).cast(time_dtype))

    if bars.height < 500:
        print(f"  {symbol:15s} {tf:4s}  SKIP — only {bars.height} bars after cutoff")
        return None

    # Build features
    feat = build_bar_features(bars, z_threshold=1.0)
    feat_clean = feat.drop_nulls(FEATURE_COLUMNS)

    if feat_clean.height < 200:
        print(f"  {symbol:15s} {tf:4s}  SKIP — only {feat_clean.height} clean rows")
        return None

    # Entry signals from z-score
    primary_side = feat_clean["primary_side"].to_numpy().astype(np.int32)
    signal_mask = primary_side != 0
    entry_indices = np.where(signal_mask)[0]
    directions = primary_side[signal_mask]

    if len(entry_indices) < 10:
        print(f"  {symbol:15s} {tf:4s}  SKIP — only {len(entry_indices)} signals")
        return None

    # OHLCV + ATR
    open_arr = feat_clean["open"].to_numpy().astype(np.float64)
    high_arr = feat_clean["high"].to_numpy().astype(np.float64)
    low_arr = feat_clean["low"].to_numpy().astype(np.float64)
    close_arr = feat_clean["close"].to_numpy().astype(np.float64)
    atr_arr = compute_atr14(high_arr, low_arr, close_arr)

    # Transaction costs
    spread_bps = {"crypto": 15.0, "forex": 2.0, "equity": 3.0,
                  "index": 2.0, "commodity": 4.0}.get(cluster, 3.0)
    fee_bps = broker_commission_bps(symbol)
    slip_bps = broker_slippage_bps(symbol)
    cost_pct = (fee_bps + spread_bps + slip_bps * 2.0) / 1e4

    # Exit params from config
    ep = ExitParams(
        atr_sl_mult=cfg.get("atr_sl_mult", 1.5),
        atr_tp_mult=cfg.get("atr_tp_mult", 6.0),
        breakeven_atr=cfg.get("breakeven_atr", 1.0),
        trail_activation_atr=cfg.get("trail_activation_atr", 3.0),
        trail_distance_atr=cfg.get("trail_distance_atr", 1.5),
        horizon=cfg.get("exit_horizon", 24),
    )

    # Simulate
    pnl, bars_held, exit_types = simulate_trades(
        entry_indices, directions,
        open_arr, high_arr, low_arr, close_arr, atr_arr,
        ep, cost_pct,
    )

    if len(pnl) == 0:
        print(f"  {symbol:15s} {tf:4s}  SKIP — 0 trades executed")
        return None

    # Position sizing
    # pnl[i] = fractional price return (e.g. +0.02 = +2% of entry price)
    # risk_pct = fraction of equity risked per trade (e.g. 0.01 = 1%)
    # When SL hits, price moves ~atr_sl_mult * (avg ATR/price) against us
    # Leverage = risk_pct / sl_distance_as_fraction
    # Since pnl is already a price fraction, multiply by leverage to get equity return
    risk_pct = cfg.get("risk_per_trade", 0.01)
    trade_count = len(pnl)
    leverage = risk_pct / max(ep.atr_sl_mult * 0.01, 1e-6)

    equity = np.empty(trade_count + 1)
    equity[0] = account_size
    for i in range(trade_count):
        ret = pnl[i] * leverage
        ret = max(ret, -0.5)  # cap single-trade loss at 50% of equity
        equity[i + 1] = equity[i] * (1 + ret)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = float(np.max(drawdown)) * 100

    # Stats
    wins = int(np.sum(pnl > 0))
    losses = int(np.sum(pnl < 0))
    win_rate = wins / trade_count * 100
    total_return = (equity[-1] / equity[0] - 1) * 100
    avg_pnl = float(np.mean(pnl)) * 100
    median_pnl = float(np.median(pnl)) * 100
    avg_bars = float(np.mean(bars_held))

    # Profit factor
    gross_profit = float(np.sum(pnl[pnl > 0])) if wins > 0 else 0
    gross_loss = abs(float(np.sum(pnl[pnl < 0]))) if losses > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    # Exit type distribution
    exit_dist = {}
    for code, label in EXIT_LABELS.items():
        count = int(np.sum(exit_types == code))
        if count > 0:
            exit_dist[label] = count

    # Annualized return (rough)
    years = max_years
    ann_return = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe (on trade returns, annualized roughly)
    if trade_count > 1 and np.std(pnl) > 0:
        trades_per_year = trade_count / years
        sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Monte Carlo simulation
    mc = monte_carlo(pnl, account_size, risk_pct, ep.atr_sl_mult, n_sims=1000)

    result = {
        "symbol": symbol,
        "timeframe": tf,
        "cluster": cluster,
        "bars": len(feat_clean),
        "signals": len(entry_indices),
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_return_pct": round(total_return, 2),
        "ann_return_pct": round(ann_return, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(float(sharpe), 3),
        "profit_factor": round(profit_factor, 3),
        "avg_pnl_pct": round(avg_pnl, 4),
        "median_pnl_pct": round(median_pnl, 4),
        "avg_bars_held": round(avg_bars, 1),
        "cost_bps": round(fee_bps + spread_bps + slip_bps * 2, 1),
        "exit_dist": exit_dist,
        "final_equity": round(equity[-1], 2),
        "equity_curve": equity,
        "mc": mc,
        "params": {
            "SL": ep.atr_sl_mult, "TP": ep.atr_tp_mult,
            "BE": ep.breakeven_atr, "trail_act": ep.trail_activation_atr,
            "trail_dist": ep.trail_distance_atr, "horizon": ep.horizon,
        },
    }
    return result


def monte_carlo(
    pnl: np.ndarray,
    account_size: float,
    risk_pct: float,
    sl_mult: float,
    n_sims: int = 1000,
) -> dict:
    """Monte Carlo simulation — shuffle trade order N times.

    Returns percentile stats for final equity and max drawdown.
    """
    rng = np.random.default_rng(42)
    finals = np.empty(n_sims)
    max_dds = np.empty(n_sims)
    leverage = risk_pct / max(sl_mult * 0.01, 1e-6)

    for i in range(n_sims):
        shuffled = rng.permutation(pnl)
        equity = np.empty(len(shuffled) + 1)
        equity[0] = account_size
        for j in range(len(shuffled)):
            ret = shuffled[j] * leverage
            ret = max(ret, -0.5)
            equity[j + 1] = equity[j] * (1 + ret)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-9)
        finals[i] = equity[-1]
        max_dds[i] = np.max(dd) * 100

    return {
        "mc_dd_p5": round(float(np.percentile(max_dds, 5)), 1),
        "mc_dd_p25": round(float(np.percentile(max_dds, 25)), 1),
        "mc_dd_p50": round(float(np.percentile(max_dds, 50)), 1),
        "mc_dd_p75": round(float(np.percentile(max_dds, 75)), 1),
        "mc_dd_p95": round(float(np.percentile(max_dds, 95)), 1),
        "mc_ruin_pct": round(float(np.mean(finals < account_size * 0.5)) * 100, 1),
    }


def print_results(results: list[dict], account_size: float) -> None:
    """Pretty-print backtest results."""
    print("\n" + "=" * 110)
    print(f"{'Symbol':15s} {'TF':4s} {'Trades':>7s} {'WinR%':>6s} {'PF':>6s} "
          f"{'Return%':>9s} {'Ann%':>7s} {'MaxDD%':>7s} {'Sharpe':>7s} "
          f"{'AvgBars':>7s} {'Final$':>12s}  Exits")
    print("-" * 110)

    total_final = 0
    for r in sorted(results, key=lambda x: x["total_return_pct"], reverse=True):
        exits_str = " ".join(f"{k}:{v}" for k, v in sorted(r["exit_dist"].items()))
        print(
            f"  {r['symbol']:13s} {r['timeframe']:4s} {r['trades']:7d} "
            f"{r['win_rate']:5.1f}% {r['profit_factor']:6.2f} "
            f"{r['total_return_pct']:+8.1f}% {r['ann_return_pct']:+6.1f}% "
            f"{r['max_dd_pct']:6.1f}% {r['sharpe']:7.3f} "
            f"{r['avg_bars_held']:7.1f} {r['final_equity']:12,.0f}  {exits_str}"
        )
        total_final += r["final_equity"]

    print("-" * 110)

    # Portfolio summary
    n = len(results)
    if n == 0:
        return

    total_trades = sum(r["trades"] for r in results)
    avg_wr = np.mean([r["win_rate"] for r in results])
    avg_pf = np.mean([r["profit_factor"] for r in results])
    portfolio_return = (total_final / (account_size * n) - 1) * 100
    avg_sharpe = np.mean([r["sharpe"] for r in results])

    # Portfolio equity curve (equal-weight allocation)
    max_len = max(len(r["equity_curve"]) for r in results)
    portfolio_eq = np.zeros(max_len)
    for r in results:
        ec = r["equity_curve"]
        # Extend shorter curves with final value
        extended = np.full(max_len, ec[-1])
        extended[:len(ec)] = ec
        portfolio_eq += extended / n  # equal weight

    port_peak = np.maximum.accumulate(portfolio_eq)
    port_dd = (port_peak - portfolio_eq) / port_peak
    port_max_dd = float(np.max(port_dd)) * 100

    print(
        f"  {'PORTFOLIO':13s} {'':4s} {total_trades:7d} "
        f"{avg_wr:5.1f}% {avg_pf:6.2f} "
        f"{portfolio_return:+8.1f}% {'':7s} "
        f"{port_max_dd:6.1f}% {avg_sharpe:7.3f} "
        f"{'':7s} {total_final:12,.0f}"
    )
    print("=" * 110)

    # Monte Carlo summary
    print(f"\n  Monte Carlo (1000 trade-order shuffles per symbol):")
    print(f"  {'Symbol':15s} {'Return%':>9s} {'DD p5':>7s} {'DD p25':>7s} "
          f"{'DD p50':>7s} {'DD p75':>7s} {'DD p95':>7s} {'Ruin%':>6s}")
    print(f"  {'-'*70}")
    for r in sorted(results, key=lambda x: x["total_return_pct"], reverse=True):
        mc = r["mc"]
        print(
            f"  {r['symbol']:15s} {r['total_return_pct']:+8.1f}% "
            f"{mc['mc_dd_p5']:>6.1f}% {mc['mc_dd_p25']:>6.1f}% "
            f"{mc['mc_dd_p50']:>6.1f}% {mc['mc_dd_p75']:>6.1f}% "
            f"{mc['mc_dd_p95']:>6.1f}% {mc['mc_ruin_pct']:>5.1f}%"
        )


def main():
    p = argparse.ArgumentParser(description="Backtest optimized exit params on bar data")
    p.add_argument("--symbols", type=str, default="",
                   help="Comma-separated symbols (default: all from configs)")
    p.add_argument("--max-years", type=int, default=10)
    p.add_argument("--account", type=float, default=100_000,
                   help="Starting account size per symbol")
    p.add_argument("--optuna-csv", type=str, default="",
                   help="Path to exit_summary.csv — backtest ALL symbol×TF combos from optuna")
    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()

    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    # Build job list: either from optuna CSV or from configs
    jobs: list[tuple[str, dict]] = []  # (label, cfg_dict)

    if args.optuna_csv:
        opt_df = pl.read_csv(args.optuna_csv)
        opt_ok = opt_df.filter(pl.col("status") == "ok")
        for row in opt_ok.iter_rows(named=True):
            sym = row["symbol"]
            base_cfg = configs.get(sym, {})
            cfg = dict(base_cfg)  # copy
            cfg["exit_timeframe"] = row["timeframe"]
            cfg["atr_sl_mult"] = row["best_atr_sl_mult"]
            cfg["atr_tp_mult"] = row["best_atr_tp_mult"]
            cfg["breakeven_atr"] = row["best_breakeven_atr"]
            cfg["trail_activation_atr"] = row["best_trail_activation_atr"]
            cfg["trail_distance_atr"] = row["best_trail_distance_atr"]
            cfg["exit_horizon"] = int(row["best_horizon"])
            if not cfg.get("asset_class"):
                cfg["asset_class"] = row.get("cluster", "equity")
            label = f"{sym}_{row['timeframe']}"
            jobs.append((label, cfg))
    else:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",")]
        else:
            symbols = sorted(configs.keys())
        for sym in symbols:
            if sym in configs:
                jobs.append((sym, configs[sym]))

    print(f"[backtest] {len(jobs)} jobs | {args.max_years} years | "
          f"${args.account:,.0f} per symbol")
    print()

    results = []
    for label, cfg in jobs:
        sym = label.split("_")[0] if "_" in label and label.split("_")[-1] in ("M1","M5","M15","M30","H1","H4","D1") else label
        # Extract symbol name (handle e.g. "FRA40.cash_H4" or "NZD_USD_H4")
        tf = cfg.get("exit_timeframe", "H4")
        # Reconstruct symbol from label by removing _TF suffix
        if label.endswith(f"_{tf}"):
            sym = label[:-len(f"_{tf}")]
        else:
            sym = label
        r = backtest_symbol(sym, cfg, args.max_years, args.account)
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo symbols produced results.")
        return

    print_results(results, args.account)

    # Save CSV
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "models" / "optuna_results" / f"backtest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (without equity curves)
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("equity_curve", "exit_dist", "params", "mc")}
        row.update({f"exit_{k}": v for k, v in r["exit_dist"].items()})
        row.update(r["params"])
        row.update(r["mc"])
        rows.append(row)
    pl.from_dicts(rows).write_csv(str(out_dir / "backtest_summary.csv"))

    # Save equity curves
    for r in results:
        eq_df = pl.DataFrame({"trade_num": range(len(r["equity_curve"])),
                               "equity": r["equity_curve"]})
        eq_df.write_csv(str(out_dir / f"equity_{r['symbol']}_{r['timeframe']}.csv"))

    print(f"\n[backtest] Saved to {out_dir}")


if __name__ == "__main__":
    main()
