#!/usr/bin/env python3
"""
Optuna ML Backtest - Geoptimaliseerd voor multi-GPU en 14 cores/28 threads.

Strategie:
- Symbols parallel over beide GPUs (RTX 2060 + GTX 1050)
- Grote train windows (4800 bars) voor betere GPU utilization
- ProcessPoolExecutor voor symbol-level parallelisme
- Polars met 28 threads voor data loading
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

# Silence Optuna info logs in workers
optuna.logging.set_verbosity(optuna.logging.WARNING)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.integrated_pipeline import make_optuna_objective, set_polars_threads
from engine.labeling import apply_triple_barrier
from research.train_ml_strategy import (
    infer_slippage_bps,
    infer_spread_bps,
    make_time_bars,
    sanitize_training_frame,
)
from engine.feature_builder import FEATURE_COLUMNS, build_bar_features


def parse_args():
    p = argparse.ArgumentParser(description="Parallel GPU Optuna tuning")
    p.add_argument("--een-md", default="")
    p.add_argument(
        "--data-roots",
        default="/home/tradebot/ssd_data_1/tick_data",
    )
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--symbols", default="", help="Comma-separated symbols override")
    p.add_argument("--z-threshold", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--trial-jobs", type=int, default=3, help="Parallel Optuna trials per symbol")
    p.add_argument("--symbol-workers", type=int, default=6, help="Parallel symbols (4 on RTX, 2 on GTX)")
    p.add_argument("--train-size", type=int, default=4800, help="Larger = more GPU work")
    p.add_argument("--test-size", type=int, default=1200)
    p.add_argument("--purge", type=int, default=96)
    p.add_argument("--embargo", type=int, default=96)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--horizon-bars", type=int, default=24)
    p.add_argument("--out-dir", default="trading_prop/production/optuna_parallel_gpu")
    return p.parse_args()


def extract_symbols_from_een(md_path: str) -> list[str]:
    txt = Path(md_path).read_text(encoding="utf-8")
    syms: list[str] = []
    in_symbol_table = False
    for line in txt.splitlines():
        if not line.startswith("|"):
            in_symbol_table = False
            continue
        low = line.lower()
        if "| symbol |" in low:
            in_symbol_table = True
            continue
        if "---" in line:
            continue
        if not in_symbol_table:
            continue
        m = re.match(r"^\|\s*([A-Za-z0-9_.]+)\s*\|", line)
        if m:
            syms.append(m.group(1))
    return sorted(set(syms))


def symbol_files(symbol: str, data_roots: list[str]) -> list[str]:
    out: list[str] = []
    for root in data_roots:
        d = os.path.join(root, symbol)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                out.append(os.path.join(d, f))
    return out


def load_symbol_ticks_lf(symbol: str, data_roots: list[str]) -> pl.LazyFrame | None:
    files = symbol_files(symbol, data_roots)
    if not files:
        return None
    lfs = [
        pl.scan_parquet(fp).select(["time", "bid", "ask", "last", "volume", "volume_real"])
        for fp in files
    ]
    lf = pl.concat(lfs, how="vertical").sort("time").with_columns(
        [
            pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
            pl.when(pl.col("last") > 0)
            .then(pl.col("last"))
            .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
            .alias("price"),
            pl.when(pl.col("volume_real") > 0).then(pl.col("volume_real")).otherwise(pl.col("volume")).alias("size"),
        ]
    ).drop_nulls(["time", "price", "size"])
    return lf.select(["time", "bid", "ask", "price", "size"])


def gpu_xgb_params(gpu_id: int = 0) -> dict:
    """XGBoost params optimized for GPU with larger batches."""
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": f"cuda:{gpu_id}",
        "sampling_method": "gradient_based",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "min_child_weight": 8.0,
        "eta": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.75,
        "reg_alpha": 0.08,
        "reg_lambda": 2.5,
        "gamma": 0.1,
        "max_bin": 512,
        "grow_policy": "lossguide",
        "verbosity": 0,
    }


def run_symbol_on_gpu(symbol: str, gpu_id: int, args_dict: dict) -> dict:
    """Run single symbol optimization on specified GPU."""
    # Set GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set Polars threads per worker (CPU-bound data loading)
    set_polars_threads(4)  # Meer workers = minder threads per worker

    roots = [x for x in args_dict["data_roots"].split(",") if x]
    ticks_lf = load_symbol_ticks_lf(symbol, roots)
    if ticks_lf is None:
        return {"symbol": symbol, "status": "no_data", "gpu": gpu_id}

    ticks_df = ticks_lf.collect()
    if ticks_df.height == 0:
        return {"symbol": symbol, "status": "no_data", "gpu": gpu_id}
    if ticks_df.select(pl.col("size").sum()).item() <= 0:
        ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks_df)
    slippage_bps = infer_slippage_bps(symbol)
    bars = make_time_bars(ticks_df.select(["time", "price", "size"]), args_dict["timeframe"])
    feat = build_bar_features(bars, z_threshold=args_dict["z_threshold"])
    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=args_dict["horizon_bars"],
        pt_mult=args_dict["pt_mult"],
        sl_mult=args_dict["sl_mult"],
    )
    feat = feat.with_columns(
        [
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
        ]
    ).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    min_rows = args_dict["train_size"] + args_dict["test_size"] + 200
    if feat.height < min_rows:
        return {"symbol": symbol, "status": "insufficient_rows", "rows": feat.height, "gpu": gpu_id}

    feat_lf = feat.lazy()
    base = gpu_xgb_params(gpu_id=0)  # Always 0 because CUDA_VISIBLE_DEVICES is set

    # Test GPU availability
    try:
        probe = xgb.DMatrix(np.array([[0.0], [1.0]]), label=np.array([0.0, 1.0]))
        xgb.train({"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"}, probe, num_boost_round=1)
    except Exception:
        base["device"] = "cpu"
        base["sampling_method"] = "uniform"

    objective = make_optuna_objective(
        features_lf=feat_lf,
        feature_cols=FEATURE_COLUMNS,
        target_col="target",
        avg_win_col="avg_win",
        avg_loss_col="avg_loss",
        base_params=base,
        train_size=args_dict["train_size"],
        test_size=args_dict["test_size"],
        purge=args_dict["purge"],
        embargo=args_dict["embargo"],
        slippage_multiplier=2.0,
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    # Parallel trials binnen één symbol voor betere GPU saturation
    study.optimize(
        objective,
        n_trials=args_dict["trials"],
        n_jobs=args_dict.get("trial_jobs", 3),  # Meerdere trials tegelijk
        show_progress_bar=False
    )
    best = study.best_trial

    return {
        "symbol": symbol,
        "status": "ok",
        "gpu": gpu_id,
        "timeframe": args_dict["timeframe"],
        "best_ev": float(best.value),
        "best_num_boost_round": int(best.params["num_boost_round"]),
        "best_max_depth": int(best.params["max_depth"]),
        "best_gamma": float(best.params["gamma"]),
        "best_subsample": float(best.params["subsample"]),
        "best_colsample_bytree": float(best.params["colsample_bytree"]),
        "best_reg_alpha": float(best.params["reg_alpha"]),
        "best_reg_lambda": float(best.params["reg_lambda"]),
        "best_min_child_weight": float(best.params["min_child_weight"]),
        "trials": int(args_dict["trials"]),
        "rows": feat.height,
        "device_used": base["device"],
    }


def discover_symbols_from_data(data_roots: str, min_parquets: int = 1) -> list[str]:
    """Discover all symbols that have parquet data in data roots."""
    syms = set()
    for root in data_roots.split(","):
        root = root.strip()
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            d = os.path.join(root, entry)
            if not os.path.isdir(d):
                continue
            pq_files = [f for f in os.listdir(d) if f.endswith(".parquet")]
            if len(pq_files) >= min_parquets:
                syms.add(entry)
    return sorted(syms)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    elif args.een_md and os.path.isfile(args.een_md):
        symbols = extract_symbols_from_een(args.een_md)
    else:
        symbols = discover_symbols_from_data(args.data_roots)
    if not symbols:
        raise SystemExit("No symbols found")

    # Convert args to dict for multiprocessing
    args_dict = {
        "data_roots": args.data_roots,
        "timeframe": args.timeframe,
        "z_threshold": args.z_threshold,
        "trials": args.trials,
        "trial_jobs": args.trial_jobs,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "purge": args.purge,
        "embargo": args.embargo,
        "pt_mult": args.pt_mult,
        "sl_mult": args.sl_mult,
        "horizon_bars": args.horizon_bars,
    }

    print(f"[optuna-parallel] Starting {len(symbols)} symbols on 2 GPUs")
    print(f"  - Symbol workers: {args.symbol_workers}")
    print(f"  - Parallel trials per symbol: {args.trial_jobs}")
    print(f"  - Total parallel GPU jobs: ~{args.symbol_workers * args.trial_jobs}")
    print(f"  - Trials per symbol: {args.trials}")
    print(f"  - Train size: {args.train_size}")
    print(f"  - Output: {args.out_dir}")
    print()

    rows = []
    # Distribute symbols across GPUs (weighted: 2/3 on RTX 2060, 1/3 on GTX 1050)
    with ProcessPoolExecutor(max_workers=args.symbol_workers) as executor:
        futures = {}
        for i, sym in enumerate(symbols):
            # RTX 2060 (6GB) gets 2x more work than GTX 1050 (2GB)
            gpu_id = 0 if (i % 3) < 2 else 1
            future = executor.submit(run_symbol_on_gpu, sym, gpu_id, args_dict)
            futures[future] = sym

        for future in as_completed(futures):
            sym = futures[future]
            try:
                r = future.result()
                status = r.get("status", "unknown")
                gpu = r.get("gpu", "?")
                if status == "ok":
                    print(f"[GPU {gpu}] {sym}: EV={r['best_ev']:.6f}, rounds={r['best_num_boost_round']}")
                else:
                    print(f"[GPU {gpu}] {sym}: {status}")
            except Exception as e:
                r = {"symbol": sym, "status": "error", "error": str(e)}
                print(f"[ERROR] {sym}: {e}")
            rows.append(r)

    out = pl.from_dicts(rows).sort("symbol")
    out.write_parquet(os.path.join(args.out_dir, "summary.parquet"), compression="zstd")
    out.write_csv(os.path.join(args.out_dir, "summary.csv"))
    print(f"\n[optuna-parallel] Done: {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
