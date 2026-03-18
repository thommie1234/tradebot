#!/usr/bin/env python3
"""Bulk train XGBoost models for all BF broker symbols across all timeframes.

Uses BF's cost model (brightfunded.yaml) for spread/commission.
Saves production models to bf/models/ and bf/models/{TF}/.

Usage:
    python3 common/tools/bf_bulk_train.py
    python3 common/tools/bf_bulk_train.py --symbols EURUSD,GBPUSD
    python3 common/tools/bf_bulk_train.py --tf M30,H1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

# BF broker symbol -> tick data directory name
SYMBOL_MAP = {
    "AAVE/USD": "AAVE_USD", "ADA/USD": "ADA_USD", "ALGO/USD": "ALGO_USD",
    "BNB/USD": "BNB_USD", "BTC/USD": "BTC_USD", "DASH/USD": "DASH_USD",
    "DOT/USD": "DOT_USD", "ETH/USD": "ETH_USD", "LINK/USD": "LINK_USD",
    "LTC/USD": "LTC_USD", "NEO/USD": "NEO_USD", "SOL/USD": "SOL_USD",
    "UNI/USD": "UNI_USD", "XAG/USD": "XAG_USD", "XAU/USD": "XAU_USD",
    "XLM/USD": "XLM_USD", "XPD/USD": "XPD_USD", "XPT/USD": "XPT_USD",
    "XRP/USD": "XRP_USD",
    # Direct matches (forex, indices)
    "AUDCAD": "AUDCAD", "AUDCHF": "AUDCHF", "AUDJPY": "AUDJPY",
    "AUDNZD": "AUDNZD", "AUDUSD": "AUDUSD", "CADCHF": "CADCHF",
    "CADJPY": "CADJPY", "CHFJPY": "CHFJPY", "EURAUD": "EURAUD",
    "EURCAD": "EURCAD", "EURCHF": "EURCHF", "EURGBP": "EURGBP",
    "EURJPY": "EURJPY", "EURNOK": "EURNOK", "EURNZD": "EURNZD",
    "EURPLN": "EURPLN", "EURUSD": "EURUSD", "GBPAUD": "GBPAUD",
    "GBPCAD": "GBPCAD", "GBPCHF": "GBPCHF", "GBPJPY": "GBPJPY",
    "GBPNZD": "GBPNZD", "GBPUSD": "GBPUSD", "NZDCAD": "NZDCAD",
    "NZDCHF": "NZDCHF", "NZDJPY": "NZDJPY", "NZDUSD": "NZDUSD",
    "USDCAD": "USDCAD", "USDCHF": "USDCHF", "USDJPY": "USDJPY",
    "USDNOK": "USDNOK", "USDSEK": "USDSEK", "USDZAR": "USDZAR",
    # Indices
    "EU50.cash": "EU50.cash", "FRA40.cash": "FRA40.cash",
    "UK100.cash": "UK100.cash", "US100.cash": "US100.cash",
    "US30.cash": "US30.cash", "US500.cash": "US500.cash",
    # Crypto without slash
    "XDG/USD": "DOGEUSD",
}

DATA_DIRS = [
    r"C:\tick_data\ssd1",
    r"C:\tick_data\ssd2",
    r"C:\tick_data\nvme",
]

ALL_TFS = ["M5", "M15", "M30", "H1", "H4"]


def find_data_dir(data_sym: str) -> str | None:
    best = None
    best_count = 0
    for dd in DATA_DIRS:
        p = os.path.join(dd, data_sym)
        if os.path.isdir(p):
            n = sum(1 for _, _, files in os.walk(p) for f in files if f.endswith(".parquet"))
            if n > best_count:
                best = p
                best_count = n
    return best


def count_trading_days(data_dir: str) -> int:
    """Count trading days. Monthly parquets (~20 days each), daily = 1 each."""
    count = 0
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".parquet"):
                continue
            name = f.replace(".parquet", "")
            # Monthly: "2024-01" -> ~20 trading days
            if len(name) == 7 and name[4] == "-":
                count += 20
            # Daily: "2024-01-15" -> 1 trading day
            elif len(name) == 10 and name[4] == "-" and name[7] == "-":
                count += 1
            else:
                count += 1
    return count


def train_symbol(data_sym: str, timeframe: str, model_dir: str) -> bool:
    """Train one symbol on one timeframe. Returns True if successful."""
    data_dir = find_data_dir(data_sym)
    if not data_dir:
        return False

    tf_model_dir = model_dir if timeframe == "H1" else os.path.join(model_dir, timeframe)
    os.makedirs(tf_model_dir, exist_ok=True)

    # Check if model already exists
    model_path = os.path.join(tf_model_dir, f"{data_sym}.json")
    if os.path.exists(model_path):
        return True  # Already trained

    env = os.environ.copy()
    env["COST_MODEL"] = "brightfunded"
    env["SAVE_PRODUCTION_MODEL"] = "1"

    cmd = [
        sys.executable, "common/research/train_ml_strategy.py",
        "--symbols", data_sym,
        "--timeframes", timeframe,
        "--out-dir", tf_model_dir,
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0 and os.path.exists(model_path):
            return True
        # Check stderr for common issues
        if "no_folds" in result.stderr or "insufficient" in result.stderr.lower():
            print(f"    SKIP: insufficient data")
        elif result.returncode != 0:
            # Print last few lines of stderr
            err_lines = result.stderr.strip().split("\n")[-3:]
            for line in err_lines:
                print(f"    ERR: {line[:120]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (5min)")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", help="Comma-separated symbols (default: all)")
    parser.add_argument("--tf", help="Comma-separated timeframes (default: all)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    model_dir = str(REPO_ROOT / "bf" / "models")
    timeframes = args.tf.split(",") if args.tf else ALL_TFS

    # Build symbol list
    if args.symbols:
        requested = set(args.symbols.split(","))
        symbols = [(bf, data) for bf, data in SYMBOL_MAP.items()
                    if data in requested or bf in requested]
    else:
        symbols = list(SYMBOL_MAP.items())

    # Filter to symbols with data
    valid = []
    for bf_sym, data_sym in symbols:
        dd = find_data_dir(data_sym)
        if dd:
            days = count_trading_days(dd)
            if days >= 100:
                valid.append((bf_sym, data_sym, days))
            else:
                print(f"SKIP {data_sym}: only {days} days (need 100)")
        else:
            pass  # No data, silently skip

    print(f"Training {len(valid)} symbols × {len(timeframes)} TFs = {len(valid) * len(timeframes)} models")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Model dir: {model_dir}")
    print()

    trained = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, (bf_sym, data_sym, days) in enumerate(valid):
        for tf in timeframes:
            tf_dir = model_dir if tf == "H1" else os.path.join(model_dir, tf)
            model_path = os.path.join(tf_dir, f"{data_sym}.json")
            if os.path.exists(model_path):
                skipped += 1
                continue

            elapsed = time.time() - t0
            remaining = len(valid) * len(timeframes) - (i * len(timeframes) + timeframes.index(tf))
            rate = (trained + failed + 1) / max(elapsed, 1) * 60
            eta = remaining / max(rate, 0.1)

            print(f"[{i+1}/{len(valid)}] {data_sym} {tf} ({days}d) "
                  f"[done={trained} skip={skipped} fail={failed} ETA={eta:.0f}min]")

            ok = train_symbol(data_sym, tf, model_dir)
            if ok:
                trained += 1
            else:
                failed += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}min: {trained} trained, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
