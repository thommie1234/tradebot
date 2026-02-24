#!/usr/bin/env python3
"""
Portfolio Monte Carlo — bootstrap from WFO backtest trade distributions.

Reconstructs per-symbol P&L distributions from backtest metrics,
then bootstraps portfolio equity curves to estimate:
  - Drawdown distribution & FTMO ruin probability
  - Annual return & Sharpe distribution
  - Probability of profit over various horizons
  - Worst-case consecutive loss streaks

Usage:
    python3 research/portfolio_monte_carlo.py [--sims 100000] [--no-gpu]
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_wfo_results(csv_path: str) -> list[dict]:
    """Load WFO results CSV into list of dicts."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def reconstruct_trade_pool(results: list[dict], equity: float = 100_000) -> np.ndarray:
    """
    Reconstruct realistic per-trade dollar P&L from WFO backtest metrics.

    For each symbol, we generate trades matching the observed:
      - win_rate
      - avg_win_pct / avg_loss_pct (as % of entry price, converted to $ via risk)
      - exit type distribution (SL, TP, BE, Trail, Horizon)
      - total number of trades

    Returns combined array of dollar P&L values across all symbols.
    """
    all_pnl = []

    for row in results:
        symbol = row["symbol"]
        n_trades = int(row["n_trades"])
        risk_pct = float(row["risk_pct"]) / 100  # CSV has 0.5 meaning 0.5%
        total_return = float(row["total_return"])
        win_rate = float(row["win_rate"]) / 100  # CSV has e.g. 32.8 meaning 32.8%

        # Exit type percentages
        pct_sl = float(row["pct_sl"]) / 100
        pct_tp = float(row["pct_tp"]) / 100
        pct_be = float(row["pct_be"]) / 100
        pct_trail = float(row["pct_trail"]) / 100
        pct_horizon = float(row["pct_horizon"]) / 100

        # Average win/loss as % of entry price
        avg_win_pct = float(row["avg_win_pct"]) / 100
        avg_loss_pct = float(row["avg_loss_pct"]) / 100

        # SL multiplier from CSV
        sl_mult = float(row["sl"])

        # Risk per trade in dollars
        risk_dollars = equity * risk_pct

        # Convert avg_win/loss from % of price to multiples of risk (R)
        # SL distance = sl_mult * ATR, so 1R = sl_mult * ATR in price terms
        # avg_win_pct and avg_loss_pct are in price % terms
        # We need to express them as multiples of the SL distance
        # A SL hit = -1R, a TP hit = +tp_mult/sl_mult R, etc.

        # avg_win in R-multiples: avg_win_pct / (sl_mult * ATR/price)
        # But we don't have ATR/price ratio directly. Use total_return to calibrate.

        # Simpler approach: use total_return to derive average $ per trade
        avg_dollar_per_trade = total_return / n_trades

        # Derive win/loss amounts from win_rate and avg dollar per trade
        # E(trade) = win_rate * avg_win - (1-win_rate) * avg_loss = avg_dollar_per_trade
        # Also: avg_win / avg_loss ≈ avg_win_pct / avg_loss_pct (ratio preserved)
        win_loss_ratio = avg_win_pct / max(avg_loss_pct, 0.001)

        # Solve: WR * W - (1-WR) * L = avg_dollar_per_trade
        #         W / L = win_loss_ratio
        # => WR * (win_loss_ratio * L) - (1-WR) * L = avg_dollar_per_trade
        # => L * (WR * win_loss_ratio - (1-WR)) = avg_dollar_per_trade
        denom = win_rate * win_loss_ratio - (1 - win_rate)
        if abs(denom) < 1e-6:
            # Edge case: near-zero expectancy, use risk as loss
            avg_loss_dollar = risk_dollars
            avg_win_dollar = risk_dollars * win_loss_ratio
        else:
            avg_loss_dollar = abs(avg_dollar_per_trade / denom)
            avg_win_dollar = avg_loss_dollar * win_loss_ratio

        # Generate trades per exit type with realistic P&L distributions
        rng = np.random.default_rng(hash(symbol) & 0xFFFFFFFF)

        n_sl = max(1, int(n_trades * pct_sl))
        n_tp = max(0, int(n_trades * pct_tp))
        n_be = max(0, int(n_trades * pct_be))
        n_trail = max(0, int(n_trades * pct_trail))
        n_horizon = max(0, n_trades - n_sl - n_tp - n_be - n_trail)

        trades = []

        # SL exits: always losses, concentrated around -1R
        sl_pnl = -avg_loss_dollar * rng.uniform(0.8, 1.2, size=n_sl)
        trades.extend(sl_pnl)

        # TP exits: always wins, concentrated around the TP target
        if n_tp > 0:
            tp_pnl = avg_win_dollar * rng.uniform(0.9, 1.3, size=n_tp)
            trades.extend(tp_pnl)

        # Breakeven exits: small wins/losses around 0
        if n_be > 0:
            # BE: moved SL to entry, some small wins, some tiny losses (slippage)
            be_pnl = avg_win_dollar * rng.uniform(-0.05, 0.3, size=n_be)
            trades.extend(be_pnl)

        # Trail exits: larger wins (trail catches trends)
        if n_trail > 0:
            trail_pnl = avg_win_dollar * rng.uniform(0.5, 2.0, size=n_trail)
            trades.extend(trail_pnl)

        # Horizon exits: mixed, typically small P&L
        if n_horizon > 0:
            horizon_pnl = avg_dollar_per_trade + risk_dollars * 0.3 * rng.standard_normal(n_horizon)
            trades.extend(horizon_pnl)

        # Calibrate: scale so sum matches total_return
        trades_arr = np.array(trades, dtype=np.float64)
        if abs(np.sum(trades_arr)) > 1e-6:
            trades_arr *= total_return / np.sum(trades_arr)

        all_pnl.extend(trades_arr)

    return np.array(all_pnl, dtype=np.float64)


def monte_carlo_portfolio(
    trade_pool: np.ndarray,
    n_sims: int = 100_000,
    trades_per_year: int = 2500,
    initial_equity: float = 100_000,
    use_gpu: bool = True,
) -> dict:
    """
    Bootstrap Monte Carlo on portfolio trade pool.

    Simulates 1-year equity paths by resampling trades.
    """
    xp = np
    gpu_used = False
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            gpu_used = True
        except ImportError:
            pass

    n_trades = trades_per_year
    pnl = xp.asarray(trade_pool)

    # Bootstrap: random sample with replacement
    indices = xp.random.randint(0, len(pnl), size=(n_sims, n_trades))
    pnl_matrix = pnl[indices]

    # Equity curves
    equity_curves = initial_equity + xp.cumsum(pnl_matrix, axis=1)

    # Max drawdown per simulation
    running_max = xp.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (running_max - equity_curves) / running_max
    max_dds = xp.max(drawdowns, axis=1)

    # Final equity & return
    final_equity = equity_curves[:, -1]
    annual_return = (final_equity - initial_equity) / initial_equity

    # Sharpe per simulation (annualized)
    per_trade_return = pnl_matrix / initial_equity
    mean_ret = xp.mean(per_trade_return, axis=1)
    std_ret = xp.std(per_trade_return, axis=1)
    std_ret = xp.where(std_ret > 1e-10, std_ret, 1e-10)
    sharpe = mean_ret / std_ret * xp.sqrt(xp.asarray(float(n_trades)))

    # Maximum consecutive losses per simulation
    is_loss = pnl_matrix < 0
    # Compute on CPU for consecutive loss calculation
    if gpu_used:
        is_loss_cpu = is_loss.get()
        max_dds_cpu = max_dds.get()
        final_eq_cpu = final_equity.get()
        annual_ret_cpu = annual_return.get()
        sharpe_cpu = sharpe.get()
    else:
        is_loss_cpu = is_loss
        max_dds_cpu = max_dds
        final_eq_cpu = final_equity
        annual_ret_cpu = annual_return
        sharpe_cpu = sharpe

    # Sample consecutive losses (too expensive for all sims, sample 10k)
    sample_size = min(10_000, n_sims)
    max_consec = np.zeros(sample_size)
    for i in range(sample_size):
        row = is_loss_cpu[i]
        cur = best = 0
        for v in row:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        max_consec[i] = best

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    return {
        "n_simulations": n_sims,
        "trades_per_year": n_trades,
        "trade_pool_size": len(trade_pool),
        "gpu_used": gpu_used,
        "pool_stats": {
            "mean_pnl": float(np.mean(trade_pool)),
            "median_pnl": float(np.median(trade_pool)),
            "std_pnl": float(np.std(trade_pool)),
            "win_rate": float(np.mean(trade_pool > 0)),
            "avg_win": float(np.mean(trade_pool[trade_pool > 0])) if np.any(trade_pool > 0) else 0,
            "avg_loss": float(np.mean(trade_pool[trade_pool < 0])) if np.any(trade_pool < 0) else 0,
            "profit_factor": float(
                np.sum(trade_pool[trade_pool > 0]) / abs(np.sum(trade_pool[trade_pool < 0]))
            ) if np.any(trade_pool < 0) else 999,
        },
        "max_drawdown": {
            p: float(np.percentile(max_dds_cpu, p)) for p in percentiles
        },
        "annual_return_pct": {
            p: float(np.percentile(annual_ret_cpu * 100, p)) for p in percentiles
        },
        "final_equity": {
            p: float(np.percentile(final_eq_cpu, p)) for p in percentiles
        },
        "sharpe_ratio": {
            p: float(np.percentile(sharpe_cpu, p)) for p in percentiles
        },
        "consecutive_losses": {
            p: int(np.percentile(max_consec, p)) for p in percentiles
        },
        "ruin_ftmo_10pct": float(np.mean(max_dds_cpu > 0.10)),
        "ruin_ftmo_5pct_daily": float(np.mean(max_dds_cpu > 0.05)),
        "prob_profit_1yr": float(np.mean(final_eq_cpu > initial_equity)),
        "prob_profit_20pct": float(np.mean(annual_ret_cpu > 0.20)),
        "prob_loss_10pct": float(np.mean(annual_ret_cpu < -0.10)),
        "mean_annual_return": float(np.mean(annual_ret_cpu * 100)),
        "median_annual_return": float(np.median(annual_ret_cpu * 100)),
        "mean_sharpe": float(np.mean(sharpe_cpu)),
        "median_sharpe": float(np.median(sharpe_cpu)),
    }


def print_results(results: dict):
    """Pretty-print Monte Carlo results."""
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO PORTFOLIO SIMULATION")
    print(f"{'='*70}")

    ps = results["pool_stats"]
    print(f"\n  Trade Pool: {results['trade_pool_size']} trades from WFO backtest")
    print(f"  Engine: {'GPU (CuPy)' if results['gpu_used'] else 'CPU (NumPy)'}")
    print(f"  Simulations: {results['n_simulations']:,} x {results['trades_per_year']} trades/year")
    print(f"\n  Pool Statistics:")
    print(f"    Mean P&L per trade:  ${ps['mean_pnl']:+.2f}")
    print(f"    Median P&L:          ${ps['median_pnl']:+.2f}")
    print(f"    Std Dev:             ${ps['std_pnl']:.2f}")
    print(f"    Win Rate:            {ps['win_rate']:.1%}")
    print(f"    Avg Win:             ${ps['avg_win']:+.2f}")
    print(f"    Avg Loss:            ${ps['avg_loss']:+.2f}")
    print(f"    Profit Factor:       {ps['profit_factor']:.2f}")

    print(f"\n  {'─'*66}")
    print(f"  Max Drawdown Distribution:")
    for p, v in results["max_drawdown"].items():
        bar = "█" * int(v * 100)
        flag = " ← FTMO LIMIT" if abs(v - 0.10) < 0.005 else ""
        print(f"    {p:3d}th pctl: {v:6.2%}  {bar}{flag}")

    print(f"\n  Annual Return Distribution:")
    for p, v in results["annual_return_pct"].items():
        print(f"    {p:3d}th pctl: {v:+7.1f}%")

    print(f"\n  Sharpe Ratio Distribution:")
    for p, v in results["sharpe_ratio"].items():
        print(f"    {p:3d}th pctl: {v:5.2f}")

    print(f"\n  Max Consecutive Losses:")
    for p, v in results["consecutive_losses"].items():
        print(f"    {p:3d}th pctl: {v:3d} trades")

    print(f"\n  {'─'*66}")
    print(f"  KEY METRICS:")
    print(f"    Mean Annual Return:     {results['mean_annual_return']:+.1f}%")
    print(f"    Median Annual Return:   {results['median_annual_return']:+.1f}%")
    print(f"    Mean Sharpe:            {results['mean_sharpe']:.2f}")
    print(f"    Median Sharpe:          {results['median_sharpe']:.2f}")
    print(f"\n  RISK METRICS:")
    print(f"    P(Max DD > 10%):        {results['ruin_ftmo_10pct']:.1%}  {'⚠ HIGH' if results['ruin_ftmo_10pct'] > 0.20 else '✓ OK'}")
    print(f"    P(Max DD > 5%):         {results['ruin_ftmo_5pct_daily']:.1%}")
    print(f"    P(Profit after 1yr):    {results['prob_profit_1yr']:.1%}")
    print(f"    P(Return > +20%/yr):    {results['prob_profit_20pct']:.1%}")
    print(f"    P(Loss > -10%/yr):      {results['prob_loss_10pct']:.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio Monte Carlo Simulation")
    parser.add_argument("--sims", type=int, default=100_000)
    parser.add_argument("--trades-per-year", type=int, default=0,
                        help="Trades per year (0 = derive from backtest)")
    parser.add_argument("--equity", type=float, default=100_000)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--results-csv", type=str, default=None,
                        help="Path to WFO results CSV")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Find latest WFO results
    if args.results_csv:
        csv_path = args.results_csv
    else:
        results_dir = REPO_ROOT / "models/optuna_results"
        candidates = sorted(results_dir.glob("wfo_leakfree_*/wfo_all_results.csv"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("ERROR: No WFO results found. Run wfo_portfolio_backtest.py first.")
            sys.exit(1)
        csv_path = str(candidates[0])

    print(f"Loading WFO results from: {csv_path}")
    wfo_results = load_wfo_results(csv_path)
    print(f"  {len(wfo_results)} symbols loaded")

    # Reconstruct trade pool
    trade_pool = reconstruct_trade_pool(wfo_results, equity=args.equity)
    print(f"  {len(trade_pool)} trades in pool")

    # Derive trades per year if not specified
    total_trades = sum(int(r["n_trades"]) for r in wfo_results)
    # Backtest spans ~3-4 years of OOS data typically
    # Use the actual backtest period: most symbols have ~35 folds * 4-week folds ≈ 2.7 years OOS
    # Better estimate: n_trades / years_of_data
    # Conservative: assume 2.5 years of OOS data → trades_per_year
    if args.trades_per_year > 0:
        tpy = args.trades_per_year
    else:
        # Estimate from fold count: avg folds * 20 trading days per fold / 252 days per year
        avg_folds = np.mean([int(r["n_folds"]) for r in wfo_results])
        years_oos = avg_folds * 20 / 252  # rough estimate
        tpy = int(total_trades / max(years_oos, 1))
        print(f"  Estimated {tpy} trades/year (over ~{years_oos:.1f} years OOS)")

    # Run Monte Carlo
    t0 = time.time()
    mc_results = monte_carlo_portfolio(
        trade_pool,
        n_sims=args.sims,
        trades_per_year=tpy,
        initial_equity=args.equity,
        use_gpu=not args.no_gpu,
    )
    elapsed = time.time() - t0
    print(f"  Monte Carlo completed in {elapsed:.1f}s")

    print_results(mc_results)

    # Save if requested
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(mc_results, f, indent=2)
        print(f"\nResults saved to: {args.save_json}")
