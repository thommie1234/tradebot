#!/usr/bin/env python3
"""TTP Pipeline: monitor tick downloads and train models as symbols complete.

Watches Alpaca tick data directories. When a symbol has enough data,
runs: feature build → XGBoost train → exit optuna → save model.

No early cutoff: all symbols get trained regardless of EV.

Usage:
    python3 common/tools/ttp_pipeline.py --watch
    python3 common/tools/ttp_pipeline.py --train-all
    python3 common/tools/ttp_pipeline.py --train AAPL,MSFT,NVDA
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPACA_DIRS = [
    Path(r"C:\tick_data\ssd1\alpaca"),
    Path(r"C:\tick_data\ssd2\alpaca"),
    Path(r"C:\tick_data\nvme\alpaca"),
]
MODEL_DIR = REPO_ROOT / "ttp" / "models"
CONFIG_PATH = REPO_ROOT / "ttp" / "config.json"
STOCK_LIST = REPO_ROOT / "ttp" / "config" / "all_stocks.txt"
LOG_DIR = REPO_ROOT / "ttp" / "logs"
MIN_DAYS = 250  # minimum trading days (~1 year) before training
VENV_PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python3")


def find_symbol_dir(symbol: str) -> Path | None:
    """Find which disk has this symbol's tick data."""
    for d in ALPACA_DIRS:
        p = d / symbol
        if p.is_dir():
            return p
    return None


def count_days(symbol: str) -> int:
    """Count downloaded trading days for a symbol."""
    sym_dir = find_symbol_dir(symbol)
    if not sym_dir:
        return 0
    count = 0
    for year_dir in sym_dir.iterdir():
        if year_dir.is_dir():
            count += sum(1 for f in year_dir.iterdir() if f.suffix == ".parquet")
    return count


def is_trained(symbol: str) -> bool:
    """Check if model already exists."""
    return (MODEL_DIR / f"{symbol}.json").exists()


def load_symbols() -> list[str]:
    with open(STOCK_LIST) as f:
        return [line.strip() for line in f if line.strip()]


TRAIN_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]  # Train on all TFs for paper testing


def train_symbol(symbol: str, timeframe: str = "H1") -> dict:
    """Train a single symbol on one timeframe.

    Returns dict with results.
    """
    sym_dir = find_symbol_dir(symbol)
    if not sym_dir:
        return {"symbol": symbol, "status": "no_data"}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        VENV_PYTHON, str(REPO_ROOT / "common" / "research" / "train_ml_strategy.py"),
        "--data-roots", ",".join(str(d) for d in ALPACA_DIRS),
        "--symbols", symbol,
        "--timeframes", timeframe,
        "--out-dir", str(MODEL_DIR),
        "--device", "cuda",
        "--symbol-workers", "1",
        "--xgb-jobs", "4",
    ]
    env = {**os.environ, "COST_MODEL": "ttp", "SAVE_PRODUCTION_MODEL": "1"}

    log_file = LOG_DIR / f"train_{symbol}_{timeframe}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Training {symbol} on {timeframe}...", flush=True)
    result = subprocess.run(
        train_cmd, env=env,
        capture_output=True, text=True, timeout=600,
        cwd=str(REPO_ROOT),
    )

    with open(log_file, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}\n")

    if result.returncode != 0:
        print(f"  WARN: {symbol} {timeframe} training returned {result.returncode}", flush=True)
        last_lines = result.stderr.strip().split("\n")[-3:]
        for line in last_lines:
            print(f"    {line}", flush=True)

    trained = is_trained(symbol)

    # Exit optuna (if model was created)
    exit_result = None
    if trained:
        print(f"  Running exit optuna for {symbol} {timeframe}...", flush=True)
        exit_cmd = [
            VENV_PYTHON, str(REPO_ROOT / "common" / "research" / "exit_optuna.py"),
            "--account", "ttp_demo",
            "--symbols", symbol,
            "--trials", "200",
            "--workers", "4",
        ]
        exit_res = subprocess.run(
            exit_cmd,
            capture_output=True, text=True, timeout=600,
            cwd=str(REPO_ROOT),
        )
        exit_result = exit_res.returncode == 0

    return {
        "symbol": symbol,
        "status": "trained" if trained else "failed",
        "exit_optuna": exit_result,
        "days": count_days(symbol),
    }


def train_symbol_all_tfs(symbol: str) -> dict:
    """Train a symbol on all timeframes (H1, M30, M15)."""
    results = {}
    for tf in TRAIN_TIMEFRAMES:
        print(f"  [{tf}]", flush=True)
        results[tf] = train_symbol(symbol, timeframe=tf)
    # Return the H1 result as primary (for config update)
    return results.get("H1", results.get(TRAIN_TIMEFRAMES[0], {}))


def update_config(symbol: str, results: dict):
    """Add symbol to ttp/config.json with default params."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if symbol in config:
        return  # already configured

    # Default M5 exit params (will be overridden by exit optuna results if available)
    config[symbol] = {
        "asset_class": "equity",
        "atr_period": 14,
        "atr_timeframe": "H1",
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3.0,
        "breakeven_atr": 0.5,
        "exit_horizon": 12,
        "exit_timeframe": "H1",
        "magic_number": 4000 + hash(symbol) % 1000,
        "max_spread_pct": 0.002,
        "prob_threshold": 0.55,
        "risk_per_trade": 0.002,
        "sector": "us_equity",
        "trail_activation_atr": 1.0,
        "trail_distance_atr": 0.3,
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, sort_keys=False)


def _needs_tf_training(symbol: str) -> bool:
    """Check if symbol is missing any TF models."""
    for tf in TRAIN_TIMEFRAMES:
        if tf == "H1":
            continue
        tf_model = MODEL_DIR / tf / f"{symbol}.json"
        if not tf_model.exists():
            return True
    return False


def watch_and_train():
    """Watch download progress and train symbols as they complete."""
    symbols = load_symbols()
    print(f"Watching {len(symbols)} symbols for download completion (min {MIN_DAYS} days)...",
          flush=True)

    # First pass: retrain existing H1-only models on all TFs
    existing = [s for s in symbols if is_trained(s) and _needs_tf_training(s)]
    if existing:
        print(f"\n{len(existing)} existing symbols need multi-TF training...", flush=True)
        for i, sym in enumerate(existing):
            days = count_days(sym)
            if days < MIN_DAYS:
                continue
            missing = [tf for tf in TRAIN_TIMEFRAMES if tf != "H1"
                       and not (MODEL_DIR / tf / f"{sym}.json").exists()]
            print(f"\n[retrain {i+1}/{len(existing)}] {sym} ({days} days, missing: {','.join(missing)}):",
                  flush=True)
            for tf in missing:
                train_symbol(sym, timeframe=tf)
            print(f"  OK — all TFs trained", flush=True)

    trained = set()
    while True:
        ready = []
        downloading = 0
        for sym in symbols:
            if sym in trained:
                continue
            if is_trained(sym):
                trained.add(sym)
                continue
            days = count_days(sym)
            if days >= MIN_DAYS:
                ready.append((sym, days))
            elif days > 0:
                downloading += 1

        if ready:
            print(f"\n{len(ready)} symbols ready for training, "
                  f"{len(trained)} already trained, "
                  f"{downloading} still downloading", flush=True)

            for sym, days in ready:
                print(f"\n[{len(trained)+1}/{len(symbols)}] {sym} ({days} days):", flush=True)
                result = train_symbol_all_tfs(sym)
                if result["status"] == "trained":
                    update_config(sym, result)
                    trained.add(sym)
                    print(f"  OK — model saved on {len(TRAIN_TIMEFRAMES)} TFs, config updated", flush=True)
                else:
                    print(f"  SKIP — {result['status']}", flush=True)
                    trained.add(sym)  # don't retry failed ones

        # Check if all done
        if len(trained) >= len(symbols):
            print(f"\nAll {len(symbols)} symbols processed!", flush=True)
            break

        # Wait before next check
        remaining = len(symbols) - len(trained)
        print(f"\rWaiting... {remaining} remaining, {downloading} downloading",
              end="", flush=True)
        time.sleep(30)


def train_specific(symbol_list: list[str]):
    """Train specific symbols immediately."""
    for i, sym in enumerate(symbol_list):
        days = count_days(sym)
        print(f"\n[{i+1}/{len(symbol_list)}] {sym} ({days} days):", flush=True)
        if days < 10:
            print(f"  SKIP — only {days} days of data", flush=True)
            continue
        result = train_symbol(sym)
        if result["status"] == "trained":
            update_config(sym, result)
            print(f"  OK — model saved", flush=True)
        else:
            print(f"  {result['status']}", flush=True)


def train_all():
    """Train all symbols that have enough data."""
    symbols = load_symbols()
    ready = [(s, count_days(s)) for s in symbols if count_days(s) >= MIN_DAYS and not is_trained(s)]
    print(f"{len(ready)} symbols ready for training", flush=True)
    train_specific([s for s, _ in ready])


def main():
    parser = argparse.ArgumentParser(description="TTP download+train pipeline")
    parser.add_argument("--watch", action="store_true", help="Watch downloads and train as they complete")
    parser.add_argument("--train-all", action="store_true", help="Train all symbols with enough data")
    parser.add_argument("--train", help="Train specific symbols (comma-separated)")
    parser.add_argument("--status", action="store_true", help="Show download/training status")
    args = parser.parse_args()

    if args.status:
        symbols = load_symbols()
        total = len(symbols)
        with_data = sum(1 for s in symbols if count_days(s) > 0)
        ready = sum(1 for s in symbols if count_days(s) >= MIN_DAYS)
        trained = sum(1 for s in symbols if is_trained(s))
        print(f"Symbols: {total}")
        print(f"With data: {with_data}")
        print(f"Ready (>={MIN_DAYS} days): {ready}")
        print(f"Trained: {trained}")
        return

    if args.train:
        syms = [s.strip() for s in args.train.split(",")]
        train_specific(syms)
    elif args.train_all:
        train_all()
    elif args.watch:
        watch_and_train()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
