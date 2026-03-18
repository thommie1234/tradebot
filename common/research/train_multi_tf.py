#!/usr/bin/env python3
"""Train XGBoost models for all timeframes (M1-H4) and save to TF subdirectories.

Usage:
    python3 common/research/train_multi_tf.py --account ftmo_100k --timeframes M1,M5,M15,M30,H4
    python3 common/research/train_multi_tf.py --account bright_100k --timeframes M1,M5,M15,M30,H4

Models are saved to:
  {account_dir}/models/versions/{TF}/{SYMBOL}.json

H1 models are already in the root versions/ dir, so skip H1 unless --include-h1.
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "common"))

from config.loader import cfg, load_config


def get_symbols_for_account(account_id: str) -> list[str]:
    """Get all symbols with existing H1 models for this account."""
    import yaml
    accts_path = REPO_ROOT / "common" / "config" / "accounts.yaml"
    with open(accts_path) as f:
        accts = yaml.safe_load(f).get("accounts", {})
    acct = accts.get(account_id, {})
    model_dir = REPO_ROOT / acct.get("model_dir", "")
    if not model_dir.is_dir():
        print(f"Model dir not found: {model_dir}")
        return []
    models = [f.stem for f in model_dir.iterdir()
              if f.suffix == ".json" and f.stem != "rl_sizer"
              and not f.is_dir()]
    return sorted(set(models))


def train_symbol_tf(symbol: str, timeframe: str, model_dir: str, account_id: str) -> dict:
    """Train a single symbol for a single timeframe. Runs in subprocess."""
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT / "common"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Tesla P40

    from config.loader import cfg, load_config
    load_config()

    # Apply account-specific overrides (same as run_bot.py does)
    import yaml
    accts_path = REPO_ROOT / "common" / "config" / "accounts.yaml"
    with open(accts_path) as f:
        accts = yaml.safe_load(f).get("accounts", {})
    acct = accts.get(account_id, {})
    if acct.get("config_path"):
        cfg.CONFIG_PATH = str(REPO_ROOT / acct["config_path"])
    if acct.get("model_dir"):
        cfg.MODEL_DIR = str(REPO_ROOT / acct["model_dir"])

    from audit.audit_logger import BlackoutLogger
    from engine.inference import SovereignMLFilter

    import tempfile
    _tmp = tempfile.mktemp(suffix=".db")
    logger = BlackoutLogger(db_path=_tmp)

    # Override optuna config timeframe
    filt = SovereignMLFilter(symbol, logger, model_dir=model_dir)

    # Monkey-patch to use the right timeframe
    original_train = filt.train_model

    def patched_train():
        # Load optuna params and override timeframe
        cfg.load_optuna_params()
        # Cap boost rounds for sub-H1 TFs (more data = fewer rounds needed)
        max_rounds = {"M5": 300, "M15": 400, "M30": 500, "H4": 600}.get(timeframe, 800)
        # Normalize symbol for OPTUNA_PARAMS lookup (same as train_model does)
        for sym_key in [symbol, symbol.replace("_", "")]:
            if sym_key in cfg.OPTUNA_PARAMS:
                cfg.OPTUNA_PARAMS[sym_key]["timeframe"] = timeframe
                cfg.OPTUNA_PARAMS[sym_key].setdefault("xgb_params", {})["device"] = "cuda"
                orig_rounds = cfg.OPTUNA_PARAMS[sym_key].get("num_boost_round", 600)
                cfg.OPTUNA_PARAMS[sym_key]["num_boost_round"] = min(orig_rounds, max_rounds)
                break
        else:
            # No optuna params exist — train_model will use defaults,
            # which reads atr_timeframe from sym_cfg. Override that.
            sym_cfg = cfg.SYMBOLS.get(symbol, {})
            sym_cfg["atr_timeframe"] = timeframe
            cfg.SYMBOLS[symbol] = sym_cfg
        return original_train()

    t0 = time.time()
    try:
        ok = patched_train()
        elapsed = time.time() - t0
        return {"symbol": symbol, "tf": timeframe, "ok": ok, "time": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - t0
        return {"symbol": symbol, "tf": timeframe, "ok": False, "error": str(e)[:200], "time": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True, help="Account ID (ftmo_100k or bright_100k)")
    parser.add_argument("--timeframes", default="M1,M5,M15,M30,H4",
                        help="Comma-separated timeframes to train")
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbols (default: all with H1 models)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel workers (default: 6)")
    parser.add_argument("--include-h1", action="store_true",
                        help="Also retrain H1 models")
    args = parser.parse_args()

    load_config()

    import yaml
    accts_path = REPO_ROOT / "common" / "config" / "accounts.yaml"
    with open(accts_path) as f:
        accts = yaml.safe_load(f).get("accounts", {})
    acct = accts.get(args.account, {})
    base_model_dir = str(REPO_ROOT / acct.get("model_dir", ""))

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = get_symbols_for_account(args.account)

    timeframes = [t.strip().upper() for t in args.timeframes.split(",")]
    if not args.include_h1:
        timeframes = [t for t in timeframes if t != "H1"]

    print(f"Account: {args.account}")
    print(f"Symbols: {len(symbols)} ({', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''})")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Total jobs: {len(symbols) * len(timeframes)}")
    print(f"Workers: {args.workers}")
    print()

    # Create TF subdirectories
    for tf in timeframes:
        tf_dir = os.path.join(base_model_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)

    # Build job list (skip already-trained models)
    jobs = []
    skipped_existing = 0
    for tf in timeframes:
        tf_dir = os.path.join(base_model_dir, tf)
        for symbol in symbols:
            model_path = os.path.join(tf_dir, f"{symbol}.json")
            if os.path.exists(model_path):
                skipped_existing += 1
                continue
            jobs.append((symbol, tf, tf_dir, args.account))

    if skipped_existing:
        print(f"Skipped {skipped_existing} already-trained models")

    # Execute
    success = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(train_symbol_tf, *job): job for job in jobs}
        for fut in as_completed(futures):
            result = fut.result()
            sym, tf = result["symbol"], result["tf"]
            if result.get("ok"):
                success += 1
                print(f"  OK  {sym:20s} {tf:4s}  ({result['time']:.0f}s)")
            elif "error" in result:
                failed += 1
                print(f"  ERR {sym:20s} {tf:4s}  {result.get('error', '')[:80]}")
            else:
                skipped += 1
                print(f"  SKIP {sym:20s} {tf:4s}  (insufficient data, {result['time']:.0f}s)")

    print(f"\nDone: {success} trained, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
