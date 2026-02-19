#!/usr/bin/env python3
"""
F17: Monte Carlo GPU Simulation — bootstrap equity curve analysis.

GPU-accelerated via CuPy (Tesla P40), fallback to NumPy.
Generates drawdown distributions, ruin probability, and Sharpe estimates.

Usage:
    python3 analysis/monte_carlo.py --trades-from-db --sims 50000
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _get_array_module(use_gpu: bool = True):
    """Return CuPy if available and requested, else NumPy."""
    if use_gpu:
        try:
            import cupy as cp
            return cp, True
        except ImportError:
            pass
    return np, False


def monte_carlo_equity_curves(
    trade_results: np.ndarray,
    n_simulations: int = 50_000,
    n_trades: int = 500,
    initial_equity: float = 100_000,
    use_gpu: bool = True,
    ruin_threshold: float = 0.10,
) -> dict:
    """
    Bootstrap resample trades -> equity curves -> statistics.

    Parameters
    ----------
    trade_results : array of P&L per trade (in currency)
    n_simulations : number of Monte Carlo paths
    n_trades : trades per simulated path
    initial_equity : starting equity
    use_gpu : attempt CuPy GPU acceleration
    ruin_threshold : drawdown level considered "ruin"

    Returns
    -------
    dict with keys:
        max_dd_percentiles: {5, 25, 50, 75, 95}
        final_equity_percentiles: {5, 25, 50, 75, 95}
        ruin_probability: float (P(max_dd > ruin_threshold))
        median_sharpe: float
        mean_final_equity: float
        gpu_used: bool
    """
    if len(trade_results) < 10:
        return {
            "max_dd_percentiles": {},
            "final_equity_percentiles": {},
            "ruin_probability": 0.0,
            "median_sharpe": 0.0,
            "mean_final_equity": initial_equity,
            "gpu_used": False,
            "error": "insufficient trades (need >= 10)",
        }

    xp, gpu_used = _get_array_module(use_gpu)

    # Move trade results to device
    pnl = xp.asarray(trade_results.astype(np.float64))

    # Bootstrap: random sample with replacement
    indices = xp.random.randint(0, len(pnl), size=(n_simulations, n_trades))
    pnl_matrix = pnl[indices]  # (n_sims, n_trades)

    # Build equity curves
    equity_curves = initial_equity + xp.cumsum(pnl_matrix, axis=1)

    # Max drawdown per simulation
    running_max = xp.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (running_max - equity_curves) / running_max
    max_dds = xp.max(drawdowns, axis=1)

    # Final equity
    final_equity = equity_curves[:, -1]

    # Sharpe ratio per simulation (annualized, assuming ~250 trading days, ~2 trades/day)
    daily_returns = pnl_matrix / initial_equity
    mean_ret = xp.mean(daily_returns, axis=1)
    std_ret = xp.std(daily_returns, axis=1)
    # Avoid division by zero
    std_ret = xp.where(std_ret > 1e-10, std_ret, 1e-10)
    sharpe = mean_ret / std_ret * xp.sqrt(xp.asarray(500.0))  # annualize over n_trades

    # Ruin probability
    ruin_prob = float(xp.mean((max_dds > ruin_threshold).astype(xp.float64)))

    # Percentiles
    percentiles = [5, 25, 50, 75, 95]

    def to_pct_dict(arr):
        if gpu_used:
            arr_cpu = arr.get()
        else:
            arr_cpu = arr
        return {p: float(np.percentile(arr_cpu, p)) for p in percentiles}

    if gpu_used:
        max_dds_cpu = max_dds.get()
        final_eq_cpu = final_equity.get()
        sharpe_cpu = sharpe.get()
    else:
        max_dds_cpu = max_dds
        final_eq_cpu = final_equity
        sharpe_cpu = sharpe

    return {
        "max_dd_percentiles": {p: float(np.percentile(max_dds_cpu, p)) for p in percentiles},
        "final_equity_percentiles": {p: float(np.percentile(final_eq_cpu, p)) for p in percentiles},
        "ruin_probability": ruin_prob,
        "median_sharpe": float(np.median(sharpe_cpu)),
        "mean_final_equity": float(np.mean(final_eq_cpu)),
        "gpu_used": gpu_used,
    }


def load_trades_from_db(db_path: str) -> np.ndarray:
    """Load P&L values from the live_trades table."""
    if not os.path.exists(db_path):
        return np.array([])

    conn = sqlite3.connect(db_path, timeout=10)
    rows = conn.execute("SELECT pnl FROM live_trades WHERE pnl IS NOT NULL ORDER BY id").fetchall()
    conn.close()

    if not rows:
        return np.array([])
    return np.array([r[0] for r in rows], dtype=np.float64)


def print_results(results: dict):
    """Pretty-print Monte Carlo results."""
    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULATION RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"\n  Error: {results['error']}")
        return

    gpu = "GPU (CuPy)" if results["gpu_used"] else "CPU (NumPy)"
    print(f"\n  Engine: {gpu}")

    print(f"\n  Max Drawdown Distribution:")
    for p, v in results["max_dd_percentiles"].items():
        print(f"    {p:3d}th percentile: {v:.2%}")

    print(f"\n  Final Equity Distribution:")
    for p, v in results["final_equity_percentiles"].items():
        print(f"    {p:3d}th percentile: ${v:,.0f}")

    print(f"\n  Mean Final Equity: ${results['mean_final_equity']:,.0f}")
    print(f"  Ruin Probability (DD>10%): {results['ruin_probability']:.2%}")
    print(f"  Median Sharpe Ratio: {results['median_sharpe']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo equity curve simulation")
    parser.add_argument("--trades-from-db", action="store_true",
                        help="Load trades from SQLite database")
    parser.add_argument("--db-path", default=None,
                        help="Path to SQLite database")
    parser.add_argument("--sims", type=int, default=50_000,
                        help="Number of simulations")
    parser.add_argument("--trades-per-sim", type=int, default=500,
                        help="Trades per simulation path")
    parser.add_argument("--equity", type=float, default=100_000,
                        help="Initial equity")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--ruin-threshold", type=float, default=0.10,
                        help="Drawdown threshold for ruin probability")
    args = parser.parse_args()

    from config.loader import cfg, load_config
    load_config()

    db_path = args.db_path or cfg.DB_PATH

    if args.trades_from_db:
        trades = load_trades_from_db(db_path)
        if len(trades) == 0:
            print("No trades found in database.")
            sys.exit(1)
        print(f"Loaded {len(trades)} trades from {db_path}")
        print(f"  Mean P&L: ${np.mean(trades):+.2f}")
        print(f"  Win rate: {np.mean(trades > 0):.1%}")
    else:
        # Demo: synthetic trades
        rng = np.random.default_rng(42)
        trades = rng.normal(5.0, 50.0, size=200)
        print(f"Using {len(trades)} synthetic trades for demo")

    results = monte_carlo_equity_curves(
        trades,
        n_simulations=args.sims,
        n_trades=args.trades_per_sim,
        initial_equity=args.equity,
        use_gpu=not args.no_gpu,
        ruin_threshold=args.ruin_threshold,
    )
    print_results(results)
