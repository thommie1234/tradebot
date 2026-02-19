#!/usr/bin/env python3
"""
Extreme Stress Test — CPU + GPU tot het uiterste.

5 phases:
  1. Monte Carlo Extreme     — GPU stress (P40 + GTX 1050)
  2. Synthetic Scenario Injection — CPU + logic stress
  3. Parallel Backtest Blast  — 28 workers, CPU 100% + GPU XGBoost
  4. FTMO Guardrail Torture   — risk guardrail edge-case validation
  5. System Monitor + Endurance — repeat Phase 1+3 for N minutes

Usage:
    python3 research/stress_test.py --endurance-minutes 120 --force
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.loader import cfg, load_config

load_config()

from analysis.monte_carlo import monte_carlo_equity_curves, load_trades_from_db
from research.exit_simulator import ExitParams, simulate_trades
from research.train_ml_strategy import make_time_bars, load_symbol_ticks
from engine.feature_builder import build_bar_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOTS = [
    "/home/tradebot/ssd_data_1/tick_data",
    "/home/tradebot/ssd_data_2/tick_data",
    "/home/tradebot/data_1/tick_data",
]

# Live symbolen
STRESS_SYMBOLS = ["AMZN", "TSLA", "JP225.cash", "LVMH", "NVDA", "PFE", "RACE"]

# Extra high-vol symbolen voor CPU blast
BLAST_SYMBOLS = STRESS_SYMBOLS + [
    "BTCUSD", "ETHUSD", "DASHUSD", "XAU_USD", "USOIL.cash",
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_JPY", "US100.cash",
    "GER40.cash", "AAPL", "META", "MSFT", "GOOG", "BABA",
    "XAG_USD", "UK100.cash", "HK50.cash", "EU50.cash", "US30.cash",
]

# Temperature limits
GPU_TEMP_LIMIT = 85   # C
CPU_TEMP_LIMIT = 92   # C

# Scenario definitions
SCENARIOS = {
    "flash_crash_8pct":   {"type": "crash",        "magnitude": -0.08, "duration_bars": 4},
    "flash_crash_15pct":  {"type": "crash",        "magnitude": -0.15, "duration_bars": 2},
    "gap_up_5pct":        {"type": "gap",          "magnitude": 0.05},
    "gap_down_5pct":      {"type": "gap",          "magnitude": -0.05},
    "vol_explosion_5x":   {"type": "vol_mult",     "factor": 5.0,  "duration_bars": 24},
    "vol_explosion_10x":  {"type": "vol_mult",     "factor": 10.0, "duration_bars": 12},
    "spread_blowout_10x": {"type": "spread_mult",  "factor": 10.0},
    "spread_blowout_50x": {"type": "spread_mult",  "factor": 50.0},
    "slippage_20x":       {"type": "slippage_mult","factor": 20.0},
    "dead_market":        {"type": "vol_mult",     "factor": 0.1,  "duration_bars": 48},
    "whipsaw":            {"type": "whipsaw",      "reversals": 10, "magnitude": 0.03},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubLogger:
    """Minimal logger for risk modules used outside live bot context."""
    def log(self, level, module, event, message=""):
        if level in ("WARNING", "ERROR"):
            print(f"  [{level}] {module}.{event}: {message}")


def save_json(data, path: str):
    """Save dict/list to JSON, converting numpy types."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    print(f"    Saved {path}")


def save_csv(rows: list[dict], path: str):
    """Save list of dicts as CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Saved {path} ({len(rows)} rows)")


def compute_atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute ATR(14) from OHLC arrays."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    # Simple rolling mean of 14
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(13, n):
        atr[i] = np.mean(tr[i - 13 : i + 1])
    # Fill leading NaNs with first valid value
    first_valid = atr[13] if n > 13 else tr[0]
    atr[:14] = first_valid
    return atr


def _max_consecutive(mask: np.ndarray) -> int:
    """Max consecutive True values in boolean array."""
    if len(mask) == 0:
        return 0
    max_run = cur = 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _load_ticks_eager(symbol: str) -> "pl.DataFrame | None":
    """Load tick data as eager Polars DataFrame."""
    return load_symbol_ticks(symbol, DATA_ROOTS, years_filter=set())


# ---------------------------------------------------------------------------
# Phase 1: Monte Carlo Extreme (GPU stress)
# ---------------------------------------------------------------------------

def phase1_monte_carlo_extreme(out_dir: str) -> dict:
    """Pin GPU's with massive MC simulations."""
    results = {}

    # Load real trades from audit DB
    trades = load_trades_from_db(cfg.DB_PATH)
    if len(trades) < 20:
        print("    Insufficient live trades, using synthetic P&L distribution")
        trades = np.random.default_rng(42).normal(15.0, 80.0, size=300)

    print(f"    Trade pool: {len(trades)} trades, mean=${np.mean(trades):+.2f}")

    # P40 (device 0): heavy simulations
    p40_runs = [
        (500_000,   1000, "extreme_500k"),
        (200_000,   2000, "deep_2k_trades"),
        (1_000_000, 500,  "million_sims"),
    ]
    for n_sims, n_trades, label in p40_runs:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"    P40: {label} ({n_sims:,} sims x {n_trades} trades)...")
        t0 = time.time()
        r = monte_carlo_equity_curves(
            trades, n_simulations=n_sims, n_trades=n_trades,
            initial_equity=100_000, use_gpu=True,
            ruin_threshold=0.10,
        )
        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 1)
        results[label] = r
        print(f"      Done in {elapsed:.1f}s — ruin={r['ruin_probability']:.2%} "
              f"gpu={'YES' if r.get('gpu_used') else 'NO'}")

    # GTX 1050 (device 1): smaller batches (2GB VRAM)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for n_sims, label in [(50_000, "gtx1050_50k"), (100_000, "gtx1050_100k")]:
        print(f"    GTX1050: {label} ({n_sims:,} sims)...")
        t0 = time.time()
        r = monte_carlo_equity_curves(
            trades, n_simulations=n_sims, n_trades=500,
            initial_equity=100_000, use_gpu=True,
            ruin_threshold=0.10,
        )
        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 1)
        results[label] = r
        print(f"      Done in {elapsed:.1f}s — ruin={r['ruin_probability']:.2%}")

    # Reset to default
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # GPU stress via XGBoost (CuPy not installed, so MC runs on CPU)
    # Train a massive XGBoost model on P40 to saturate GPU
    print("    GPU stress: XGBoost depth=14, 5000 rounds on P40...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        import xgboost as xgb
        rng = np.random.default_rng(42)
        n_rows, n_cols = 100_000, 50
        X_synth = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        y_synth = (rng.random(n_rows) > 0.5).astype(np.float32)
        dtrain = xgb.DMatrix(X_synth[:80_000], label=y_synth[:80_000])
        dtest = xgb.DMatrix(X_synth[80_000:], label=y_synth[80_000:])
        t0 = time.time()
        xgb.train(
            {"objective": "binary:logistic", "tree_method": "hist",
             "device": "cuda", "max_depth": 14, "eta": 0.003,
             "subsample": 0.95, "colsample_bytree": 0.9,
             "max_bin": 2048, "min_child_weight": 1},
            dtrain, num_boost_round=5000,
            evals=[(dtest, "test")], verbose_eval=False,
        )
        elapsed = time.time() - t0
        results["gpu_xgb_stress"] = {"elapsed_s": round(elapsed, 1), "rows": 100_000,
                                      "depth": 14, "rounds": 5000, "device": "P40"}
        print(f"      Done in {elapsed:.1f}s")
    except Exception as e:
        results["gpu_xgb_stress"] = {"error": str(e)[:200]}
        print(f"      Error: {e}")
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    save_json(results, f"{out_dir}/monte_carlo_results.json")
    return results


# ---------------------------------------------------------------------------
# Phase 2: Synthetic Scenario Injection
# ---------------------------------------------------------------------------

def inject_scenario(ohlc: np.ndarray, atr: np.ndarray, scenario: dict) -> tuple:
    """Modify OHLC/ATR data for a stress scenario.

    Returns (open, high, low, close, atr) — copies, originals untouched.
    """
    import polars as pl
    O = ohlc[:, 0].copy()
    H = ohlc[:, 1].copy()
    L = ohlc[:, 2].copy()
    C = ohlc[:, 3].copy()
    A = atr.copy()
    n = len(C)
    mid = n // 2  # inject around middle of the series

    stype = scenario["type"]

    if stype == "crash":
        mag = scenario["magnitude"]
        dur = scenario.get("duration_bars", 4)
        for i in range(mid, min(mid + dur, n)):
            frac = mag * (i - mid + 1) / dur
            C[i] = C[mid - 1] * (1 + frac)
            L[i] = min(L[i], C[i])
            H[i] = max(H[i], C[i])
            O[i] = C[i - 1] if i > 0 else O[i]

    elif stype == "gap":
        mag = scenario["magnitude"]
        # Gap at midpoint
        factor = 1 + mag
        O[mid:] *= factor
        H[mid:] *= factor
        L[mid:] *= factor
        C[mid:] *= factor

    elif stype == "vol_mult":
        factor = scenario["factor"]
        dur = scenario.get("duration_bars", 24)
        start = mid
        end = min(mid + dur, n)
        for i in range(start, end):
            center = (H[i] + L[i]) / 2
            half_range = (H[i] - L[i]) / 2 * factor
            H[i] = center + half_range
            L[i] = center - half_range
            A[i] = A[i] * factor

    elif stype == "spread_mult":
        # Spread blowout simulated as increased cost — modify ATR to reflect
        factor = scenario["factor"]
        A[:] = A * factor

    elif stype == "slippage_mult":
        # Simulate slippage by shifting entry prices against direction
        factor = scenario["factor"]
        avg_slip = np.mean(A) * 0.001 * factor
        O[:] = O + avg_slip  # unfavorable slippage on entries

    elif stype == "whipsaw":
        reversals = scenario.get("reversals", 10)
        mag = scenario.get("magnitude", 0.03)
        period = max(1, n // (reversals * 2))
        for i in range(n):
            cycle = (i // period) % 2
            shift = mag if cycle == 0 else -mag
            factor = 1 + shift
            O[i] *= factor
            H[i] *= factor
            L[i] *= factor
            C[i] *= factor

    return O, H, L, C, A


def phase2_scenario_injection(out_dir: str) -> list:
    """Run exit_simulator on every symbol x scenario combination."""
    import polars as pl
    results = []
    rng = np.random.default_rng(42)

    for sym in STRESS_SYMBOLS:
        print(f"    {sym}...")
        ticks = _load_ticks_eager(sym)
        if ticks is None or ticks.height < 1000:
            print(f"      Skipped (insufficient tick data)")
            continue

        bars = make_time_bars(ticks, "H1")
        if bars.height < 500:
            print(f"      Skipped (only {bars.height} H1 bars)")
            continue

        features = build_bar_features(bars, z_threshold=1.0)

        ohlc = features.select("open", "high", "low", "close").to_numpy().astype(np.float64)
        atr = compute_atr14(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3])
        n = len(ohlc)

        # Generate entry signals every 20 bars
        entries = np.arange(50, n - 100, 20, dtype=np.int64)
        directions = np.where(rng.random(len(entries)) > 0.5, 1, -1).astype(np.int64)

        # Baseline
        params = ExitParams(1.5, 4.5, 1.0, 2.0, 0.5, 24)
        pnl_base, bars_held_base, exits_base = simulate_trades(
            entries, directions,
            ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3],
            atr, params, cost_pct=0.001,
        )

        if len(pnl_base) == 0:
            print(f"      Skipped (no completed trades)")
            continue

        for scenario_name, scenario_cfg in SCENARIOS.items():
            O, H, L, C, A = inject_scenario(ohlc, atr, scenario_cfg)
            pnl, bh, ex = simulate_trades(
                entries, directions, O, H, L, C, A, params, cost_pct=0.001,
            )
            if len(pnl) == 0:
                continue

            base_mean = float(np.mean(pnl_base))
            stress_mean = float(np.mean(pnl))
            results.append({
                "symbol": sym,
                "scenario": scenario_name,
                "baseline_ev": round(base_mean, 6),
                "stress_ev": round(stress_mean, 6),
                "ev_change_pct": round(
                    (stress_mean - base_mean) / max(abs(base_mean), 1e-6) * 100, 2
                ),
                "max_single_loss": round(float(np.min(pnl)), 6),
                "max_consecutive_losses": _max_consecutive(pnl < 0),
                "trades": len(pnl),
                "sl_pct": round(float(np.mean(ex == 0)), 4),
                "tp_pct": round(float(np.mean(ex == 1)), 4),
                "trail_pct": round(float(np.mean(ex == 3)), 4),
                "horizon_pct": round(float(np.mean(ex == 4)), 4),
            })

        print(f"      {len(SCENARIOS)} scenarios x {len(pnl_base)} trades")

    save_csv(results, f"{out_dir}/scenario_results.csv")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Parallel Backtest Blast (CPU 100%)
# ---------------------------------------------------------------------------

def _blast_worker(args_tuple):
    """Heavy XGBoost training + backtest per symbol. Runs in subprocess."""
    sym, tf, data_roots, use_gpu = args_tuple

    # Each worker imports what it needs (subprocess isolation)
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))

    import numpy as np
    import os

    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # P40
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""    # CPU only

    try:
        from research.train_ml_strategy import make_time_bars, load_symbol_ticks
        from engine.feature_builder import build_bar_features

        ticks = load_symbol_ticks(sym, data_roots, years_filter=set())
        if ticks is None:
            return {"symbol": sym, "timeframe": tf, "status": "no_data", "device": "gpu" if use_gpu else "cpu"}

        bars = make_time_bars(ticks, tf)
        features = build_bar_features(bars, z_threshold=1.0)

        if len(features) < 500:
            return {"symbol": sym, "timeframe": tf, "status": "insufficient_data",
                    "n_bars": len(features), "device": "gpu" if use_gpu else "cpu"}

        # Feature/label prep
        exclude = {"time", "open", "high", "low", "close", "volume",
                    "primary_side", "atr14"}
        feat_cols = [c for c in features.columns if c not in exclude]
        X = features.select(feat_cols).to_numpy().astype(np.float32)
        # Use primary_side as label (1 if > 0, else 0)
        y_raw = features["primary_side"].fill_null(0).to_numpy()
        y = (y_raw > 0).astype(np.float32)

        # Replace NaN/Inf in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        split = int(len(X) * 0.8)
        if split < 200 or (len(X) - split) < 50:
            return {"symbol": sym, "timeframe": tf, "status": "insufficient_split",
                    "n_bars": len(features), "device": "gpu" if use_gpu else "cpu"}

        import xgboost as xgb

        device = "cuda" if use_gpu else "cpu"
        # Extreme training params for stress
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": device,
            "max_depth": 12,
            "eta": 0.005,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.001,
            "reg_lambda": 0.001,
            "min_child_weight": 1,
            "max_bin": 1024,
        }
        dtrain = xgb.DMatrix(X[:split], label=y[:split])
        dtest = xgb.DMatrix(X[split:], label=y[split:])

        t0 = time.time()
        model = xgb.train(
            params, dtrain, num_boost_round=3000,
            evals=[(dtest, "test")], verbose_eval=False,
        )
        elapsed = time.time() - t0

        preds = model.predict(dtest)
        acc = float(np.mean((preds > 0.5) == y[split:]))

        return {
            "symbol": sym, "timeframe": tf, "status": "ok",
            "train_time_s": round(elapsed, 1),
            "test_accuracy": round(acc, 4),
            "n_trees": 3000, "max_depth": 12,
            "n_bars": len(features),
            "n_features": X.shape[1],
            "device": device,
        }

    except Exception as e:
        return {"symbol": sym, "timeframe": tf, "status": "error",
                "error": str(e)[:200], "device": "gpu" if use_gpu else "cpu"}


def phase3_backtest_blast(out_dir: str) -> list:
    """28 workers simultaneously — 4 on GPU + 24 on CPU to avoid VRAM OOM."""
    # First 4 symbols get GPU, rest get CPU — avoids VRAM contention
    gpu_count = 4
    tasks = []
    for i, sym in enumerate(BLAST_SYMBOLS):
        use_gpu = i < gpu_count
        tasks.append((sym, "H1", DATA_ROOTS, use_gpu))

    results = []
    with ProcessPoolExecutor(max_workers=28) as executor:
        futures = {executor.submit(_blast_worker, t): t[0] for t in tasks}
        for f in as_completed(futures):
            sym = futures[f]
            try:
                r = f.result(timeout=600)
            except Exception as e:
                r = {"symbol": sym, "status": "exception", "error": str(e)[:200]}
            results.append(r)
            if r.get("status") == "ok":
                print(f"    BLAST {r['symbol']:12s} {r['train_time_s']:6.1f}s "
                      f"depth=12 trees=3000 acc={r['test_accuracy']:.3f} "
                      f"({r['n_bars']} bars, {r.get('device','?')})")
            else:
                print(f"    BLAST {r.get('symbol', '?'):12s} [{r['status']}] "
                      f"{r.get('error', '')[:80]}")

    save_csv(results, f"{out_dir}/backtest_blast_results.csv")
    return results


# ---------------------------------------------------------------------------
# Phase 4: FTMO Guardrail Torture
# ---------------------------------------------------------------------------

def phase4_ftmo_torture(out_dir: str) -> dict:
    """Test every guardrail under extreme conditions."""
    from risk.drawdown_guard import DrawdownGuard
    from risk.position_sizing import PositionSizingEngine, fractional_kelly

    logger = _StubLogger()
    results = {}

    # --- DrawdownGuard tests ---
    guard = DrawdownGuard(logger)

    # Test 1: Daily loss limit (-3.6% should be blocked)
    account = SimpleNamespace(balance=100_000, equity=96_400)
    allowed, reason = guard.check_daily_limits(account)
    results["daily_loss_limit_36pct"] = {
        "test": "equity -3.6% of balance",
        "expected": "BLOCKED",
        "actual": "BLOCKED" if not allowed else "ALLOWED",
        "pass": not allowed,
        "reason": reason,
    }

    # Test 2: Daily loss limit (-3.4% should be allowed)
    guard.reset_daily_flags()
    account2 = SimpleNamespace(balance=100_000, equity=96_600)
    allowed2, reason2 = guard.check_daily_limits(account2)
    results["daily_loss_limit_34pct"] = {
        "test": "equity -3.4% of balance (should pass)",
        "expected": "ALLOWED",
        "actual": "BLOCKED" if not allowed2 else "ALLOWED",
        "pass": allowed2,
        "reason": reason2,
    }

    # Test 3: Profit lock (+3.1% should be blocked)
    guard.reset_daily_flags()
    account3 = SimpleNamespace(balance=100_000, equity=103_100)
    allowed3, reason3 = guard.check_daily_limits(account3)
    results["profit_lock_31pct"] = {
        "test": "equity +3.1% — should lock profits",
        "expected": "BLOCKED",
        "actual": "BLOCKED" if not allowed3 else "ALLOWED",
        "pass": not allowed3,
        "reason": reason3,
    }

    # Test 4: DD recovery mode (DD 4.5%)
    guard2 = DrawdownGuard(logger)
    account4 = SimpleNamespace(balance=100_000, equity=95_500)
    recovery = guard2.check_dd_recovery(account4)
    results["dd_recovery_mode_45pct"] = {
        "test": "DD 4.5% — should enter recovery (halve lots)",
        "expected": True,
        "actual": recovery,
        "pass": recovery is True,
    }

    # Test 5: DD recovery exit (DD 0.5%)
    account5 = SimpleNamespace(balance=100_000, equity=99_500)
    recovery2 = guard2.check_dd_recovery(account5)
    results["dd_recovery_exit"] = {
        "test": "DD 0.5% after recovery — should exit recovery",
        "expected": False,
        "actual": recovery2,
        "pass": recovery2 is False,
    }

    # Test 6: None account (should not crash)
    guard3 = DrawdownGuard(logger)
    allowed6, reason6 = guard3.check_daily_limits(None)
    results["null_account"] = {
        "test": "None account_info — should allow",
        "expected": "ALLOWED",
        "actual": "ALLOWED" if allowed6 else "BLOCKED",
        "pass": allowed6,
    }

    # --- PositionSizingEngine tests ---
    sizer = PositionSizingEngine(logger)

    # Test 7: Lot calculation
    lot = sizer.calculate_lot_size(
        symbol="AMZN", account_equity=100_000,
        risk_pct=0.0025, sl_distance=3.0,
        symbol_info={"trade_contract_size": 1.0, "volume_min": 1.0,
                     "volume_max": 10000.0, "volume_step": 1.0},
    )
    expected_lot = round(100_000 * 0.0025 / 3.0)  # 83
    results["lot_calc_amzn"] = {
        "test": f"AMZN lot calc: equity=100k risk=0.25% SL=3.0",
        "expected_lot": expected_lot,
        "actual_lot": lot,
        "pass": abs(lot - expected_lot) <= 1,
    }

    # Test 8: Zero SL distance (edge case)
    lot_zero = sizer.calculate_lot_size(
        symbol="TEST", account_equity=100_000,
        risk_pct=0.003, sl_distance=0.0,
    )
    results["lot_calc_zero_sl"] = {
        "test": "SL distance = 0 — should return 0",
        "expected": 0.0,
        "actual": lot_zero,
        "pass": lot_zero == 0.0,
    }

    # Test 9: Kelly with no MT5 (should return fallback 0.003)
    kelly = sizer.kelly_risk_pct("NVDA")
    results["kelly_no_mt5"] = {
        "test": "Kelly without MT5 — should return fallback 0.3%",
        "expected": 0.003,
        "actual": kelly,
        "pass": kelly == 0.003,
    }

    # Test 10: Fractional Kelly edge cases
    for label, wr, pf in [("zero_wr", 0.0, 1.5), ("perfect_wr", 1.0, 2.0),
                           ("bad_pf", 0.5, 0.0), ("normal", 0.55, 1.8)]:
        fk = fractional_kelly(wr, pf, fraction=0.5)
        results[f"kelly_{label}"] = {
            "test": f"fractional_kelly(WR={wr}, PF={pf}, frac=0.5)",
            "result": fk,
            "pass": 0.0 <= fk <= 1.0,
        }

    # Test 11: RL adjust (should return base_risk when no RL sizer)
    adj_risk, arm = sizer.rl_adjust_risk(0.003, ml_confidence=0.7)
    results["rl_adjust_no_sizer"] = {
        "test": "RL adjust without RL sizer — should return base risk",
        "base_risk": 0.003,
        "adjusted_risk": adj_risk,
        "arm": arm,
        "pass": adj_risk == 0.003 and arm is None,
    }

    # Summary
    total = len(results)
    passed = sum(1 for v in results.values() if v.get("pass"))
    print(f"    Guardrail tests: {passed}/{total} passed")

    save_json(results, f"{out_dir}/ftmo_torture_results.json")
    return results


# ---------------------------------------------------------------------------
# Phase 5: System Monitor
# ---------------------------------------------------------------------------

def monitor_system(out_dir: str, stop_event: threading.Event):
    """Log CPU/GPU metrics every 10 seconds."""
    try:
        import psutil
    except ImportError:
        print("    WARNING: psutil not available, skipping system monitor")
        return

    metrics_file = f"{out_dir}/system_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "cpu_pct", "cpu_temp_c",
            "gpu0_util_pct", "gpu0_temp_c", "gpu0_mem_mb",
            "gpu1_util_pct", "gpu1_temp_c", "gpu1_mem_mb",
            "ram_used_gb", "ram_total_gb",
        ])

    while not stop_event.is_set():
        try:
            row = [datetime.now().isoformat()]

            # CPU
            cpu_pct = psutil.cpu_percent(interval=0)
            temps = psutil.sensors_temperatures()
            cpu_temps = temps.get("coretemp", [])
            cpu_temp = max((t.current for t in cpu_temps), default=0)
            row += [cpu_pct, cpu_temp]

            # GPUs
            for gpu_id in [0, 1]:
                try:
                    out = subprocess.check_output([
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        f"--id={gpu_id}",
                    ], text=True, timeout=5).strip()
                    parts = out.split(", ")
                    row += [int(parts[0]), int(parts[1]), int(parts[2])]
                except Exception:
                    row += [0, 0, 0]

            # RAM
            mem = psutil.virtual_memory()
            row += [round(mem.used / 1e9, 1), round(mem.total / 1e9, 1)]

            with open(metrics_file, "a", newline="") as f:
                csv.writer(f).writerow(row)

            # Safety warnings
            if cpu_temp > CPU_TEMP_LIMIT:
                print(f"\n    TEMP WARNING: CPU={cpu_temp}C > {CPU_TEMP_LIMIT}C!")
            for gpu_id in [0, 1]:
                try:
                    out = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits", f"--id={gpu_id}",
                    ], text=True, timeout=5).strip()
                    gpu_temp = int(out)
                    if gpu_temp > GPU_TEMP_LIMIT:
                        print(f"\n    TEMP WARNING: GPU{gpu_id}={gpu_temp}C > {GPU_TEMP_LIMIT}C!")
                except Exception:
                    pass

        except Exception as e:
            pass  # Don't crash the monitor thread

        stop_event.wait(10)


def check_temperatures() -> bool:
    """Quick temp check. Returns True if safe, False if too hot."""
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip()
        for line in out.split("\n"):
            if int(line.strip()) > GPU_TEMP_LIMIT:
                return False
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(out_dir: str, mc_results: dict, scenario_results: list,
                    blast_results: list, ftmo_results: dict):
    """Generate markdown summary report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Extreme Stress Test Report",
        f"",
        f"**Generated:** {ts}",
        f"",
        f"---",
        f"",
        f"## Phase 1: Monte Carlo Extreme",
        f"",
    ]

    for label, r in mc_results.items():
        ruin = r.get("ruin_probability", 0)
        gpu = "GPU" if r.get("gpu_used") else "CPU"
        elapsed = r.get("elapsed_s", "?")
        lines.append(f"- **{label}**: ruin={ruin:.2%}, "
                     f"median_sharpe={r.get('median_sharpe', 0):.2f}, "
                     f"engine={gpu}, time={elapsed}s")

    lines += [
        f"",
        f"## Phase 2: Scenario Injection",
        f"",
        f"| Symbol | Scenario | Baseline EV | Stress EV | Change% | Max Loss | Consec Losses |",
        f"|--------|----------|-------------|-----------|---------|----------|---------------|",
    ]
    for r in scenario_results[:30]:  # Top 30
        lines.append(
            f"| {r['symbol']} | {r['scenario']} | {r['baseline_ev']:.4f} | "
            f"{r['stress_ev']:.4f} | {r['ev_change_pct']:+.1f}% | "
            f"{r['max_single_loss']:.4f} | {r['max_consecutive_losses']} |"
        )

    lines += [
        f"",
        f"## Phase 3: Backtest Blast (28 workers)",
        f"",
    ]
    ok_results = [r for r in blast_results if r.get("status") == "ok"]
    failed = [r for r in blast_results if r.get("status") != "ok"]
    lines.append(f"- **Completed:** {len(ok_results)}/{len(blast_results)}")
    if ok_results:
        avg_time = np.mean([r["train_time_s"] for r in ok_results])
        avg_acc = np.mean([r["test_accuracy"] for r in ok_results])
        lines.append(f"- **Avg train time:** {avg_time:.1f}s")
        lines.append(f"- **Avg accuracy:** {avg_acc:.3f}")
    if failed:
        lines.append(f"- **Failed:** {', '.join(r.get('symbol','?') for r in failed)}")

    lines += [
        f"",
        f"## Phase 4: FTMO Guardrail Torture",
        f"",
    ]
    total = len(ftmo_results)
    passed = sum(1 for v in ftmo_results.values() if v.get("pass"))
    lines.append(f"**Result: {passed}/{total} tests passed**")
    lines.append(f"")
    for name, r in ftmo_results.items():
        status = "PASS" if r.get("pass") else "**FAIL**"
        lines.append(f"- [{status}] {name}: {r.get('test', '')}")

    lines += [
        f"",
        f"---",
        f"",
        f"## System Metrics",
        f"",
        f"See `system_metrics.csv` for per-10s CPU/GPU/RAM data.",
        f"",
    ]

    report_path = f"{out_dir}/stress_test_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extreme Stress Test")
    parser.add_argument("--force", action="store_true",
                        help="Run even during market hours")
    parser.add_argument("--endurance-minutes", type=int, default=0,
                        help="Minutes to repeat Phase 1+3 (0 = skip)")
    parser.add_argument("--phases", type=str, default="1,2,3,4,5",
                        help="Comma-separated phases to run (e.g. '1,3')")
    return parser.parse_args()


def main():
    args = parse_args()
    phases = set(int(p) for p in args.phases.split(","))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = str(REPO_ROOT / f"models/optuna_results/stress_test_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Start system monitor in background thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_system, args=(out_dir, stop_monitor), daemon=True,
    )
    monitor_thread.start()

    mc_results = {}
    scenario_results = []
    blast_results = []
    ftmo_results = {}

    try:
        print("=" * 70)
        print("  EXTREME STRESS TEST")
        print(f"  Output: {out_dir}")
        print(f"  Phases: {sorted(phases)}")
        print("=" * 70)

        # Phase 1: GPU stress
        if 1 in phases:
            print(f"\n[PHASE 1] Monte Carlo Extreme — GPU stress")
            t0 = time.time()
            mc_results = phase1_monte_carlo_extreme(out_dir)
            print(f"  Phase 1 done in {time.time() - t0:.0f}s")

        # Phase 2: Scenario injection
        if 2 in phases:
            print(f"\n[PHASE 2] Synthetic Scenario Injection")
            t0 = time.time()
            scenario_results = phase2_scenario_injection(out_dir)
            print(f"  Phase 2 done in {time.time() - t0:.0f}s "
                  f"({len(scenario_results)} results)")

        # Phase 3: CPU blast
        if 3 in phases:
            print(f"\n[PHASE 3] Parallel Backtest Blast — 28 workers")
            t0 = time.time()
            blast_results = phase3_backtest_blast(out_dir)
            print(f"  Phase 3 done in {time.time() - t0:.0f}s")

        # Phase 4: FTMO torture
        if 4 in phases:
            print(f"\n[PHASE 4] FTMO Guardrail Torture")
            t0 = time.time()
            ftmo_results = phase4_ftmo_torture(out_dir)
            print(f"  Phase 4 done in {time.time() - t0:.0f}s")

        # Phase 5: Endurance — GPU + CPU SIMULTANEOUSLY
        if 5 in phases and args.endurance_minutes > 0:
            from concurrent.futures import ThreadPoolExecutor

            print(f"\n[PHASE 5] Endurance Run — {args.endurance_minutes} min")
            print(f"  GPU stress (Phase 1) + CPU blast (Phase 3) run CONCURRENTLY")
            end_time = time.time() + args.endurance_minutes * 60
            cycle = 0
            while time.time() < end_time:
                cycle += 1
                remaining = (end_time - time.time()) / 60
                print(f"\n  Endurance cycle {cycle} "
                      f"({remaining:.0f} min remaining)...")

                if not check_temperatures():
                    print("  TEMP LIMIT — pausing 60s to cool down...")
                    time.sleep(60)
                    if not check_temperatures():
                        print("  Still too hot — ending endurance early")
                        break

                # Run GPU + CPU stress simultaneously via threads
                with ThreadPoolExecutor(max_workers=2) as tex:
                    gpu_future = tex.submit(phase1_monte_carlo_extreme, out_dir)
                    cpu_future = tex.submit(phase3_backtest_blast, out_dir)
                    # Wait for both
                    try:
                        mc_results = gpu_future.result(timeout=600)
                    except Exception as e:
                        print(f"    GPU phase error: {e}")
                    try:
                        blast_results = cpu_future.result(timeout=600)
                    except Exception as e:
                        print(f"    CPU phase error: {e}")

        # Generate report
        print(f"\n[REPORT] Generating summary...")
        generate_report(out_dir, mc_results, scenario_results,
                       blast_results, ftmo_results)

        print(f"\n{'=' * 70}")
        print(f"  STRESS TEST COMPLETE")
        print(f"  Results: {out_dir}/")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
        generate_report(out_dir, mc_results, scenario_results,
                       blast_results, ftmo_results)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=15)


if __name__ == "__main__":
    main()
