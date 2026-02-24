#!/usr/bin/env python3
"""
Retrain Elite Fleet — Tier 1 + Tier 2 only
===========================================

Standalone script to retrain XGBoost models from the filtered Optuna CSV.
Runs entirely from parquet tick data on disk — no MT5 needed.

Usage:
    python3 retrain_elite.py --csv optuna_deep_H1_20260208_102205/summary_filtered.csv
    python3 retrain_elite.py --csv optuna_deep_H1_20260208_102205/summary_filtered.csv --symbol XMRUSD
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Root of the tradebots repo
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import importlib
import numpy as np
import polars as pl
import xgboost as xgb

# Import from engine.* / research.* (same pattern as sovereign_bot)
features_mod = importlib.import_module("engine.feature_builder")
build_bar_features = features_mod.build_bar_features
FEATURE_COLUMNS = features_mod.FEATURE_COLUMNS

labeling_mod = importlib.import_module("engine.labeling")
apply_triple_barrier = labeling_mod.apply_triple_barrier

train_mod = importlib.import_module("research.train_ml_strategy")
make_time_bars = train_mod.make_time_bars
sanitize_training_frame = train_mod.sanitize_training_frame
infer_spread_bps = train_mod.infer_spread_bps
infer_slippage_bps = train_mod.infer_slippage_bps

pipeline_mod = importlib.import_module("research.integrated_pipeline")
make_ev_custom_objective = pipeline_mod.make_ev_custom_objective

from config.loader import cfg, load_config
load_config()

MODEL_DIR = os.path.join(REPO_ROOT, "models", "sovereign_models")
DATA_ROOTS = cfg.DATA_ROOTS
TRAIN_SIZE = 1200  # Use more data for elite retrain


def load_ticks(symbol: str) -> pl.DataFrame | None:
    """Load tick data from parquet files."""
    frames = []
    for root in DATA_ROOTS:
        sym_dir = os.path.join(root, symbol)
        if not os.path.isdir(sym_dir):
            continue
        for f in sorted(os.listdir(sym_dir)):
            if f.endswith(".parquet"):
                df = pl.read_parquet(os.path.join(sym_dir, f)).select(
                    ["time", "bid", "ask", "last", "volume", "volume_real"]
                )
                if df.height > 0:
                    frames.append(df)
    if not frames:
        return None

    d = (
        pl.concat(frames, how="vertical")
        .sort("time")
        .with_columns([
            pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
            pl.when(pl.col("last") > 0)
            .then(pl.col("last"))
            .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
            .alias("price"),
            pl.when(pl.col("volume_real") > 0)
            .then(pl.col("volume_real"))
            .otherwise(pl.col("volume"))
            .alias("size"),
        ])
        .drop_nulls(["time", "price", "size"])
    )
    return d.select(["time", "bid", "ask", "price", "size"])


def train_symbol(symbol: str, params: dict) -> bool:
    """Train a single symbol with Optuna-tuned params."""
    t0 = time.time()

    # Load ticks
    ticks = load_ticks(symbol)
    if ticks is None:
        print(f"  SKIP {symbol}: no tick data")
        return False

    # Build H1 bars
    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))
    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")

    if bars.height < TRAIN_SIZE + 100:
        print(f"  SKIP {symbol}: only {bars.height} H1 bars (need {TRAIN_SIZE + 100})")
        return False

    # Build features
    spread_bps = infer_spread_bps(ticks)
    slippage_bps = infer_slippage_bps(symbol)
    feat = build_bar_features(bars, z_threshold=1.5)

    # Triple barrier
    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=6, pt_mult=2.0, sl_mult=1.5,
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

    if feat.height < TRAIN_SIZE:
        print(f"  SKIP {symbol}: {feat.height} samples after sanitization (need {TRAIN_SIZE})")
        return False

    # Use latest TRAIN_SIZE rows
    train_df = feat.tail(TRAIN_SIZE)
    x = train_df.select(FEATURE_COLUMNS).to_numpy()
    y = train_df["target"].to_numpy().astype(np.float32)
    avg_win = train_df["avg_win"].to_numpy().astype(np.float64)
    avg_loss = train_df["avg_loss"].to_numpy().astype(np.float64)
    costs = (
        (train_df["fee_bps"] + train_df["spread_bps"] + train_df["slippage_bps"] * 2.0).to_numpy() / 1e4
    ).astype(np.float64)

    # Build XGBoost params
    xgb_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cuda",
        "sampling_method": "gradient_based",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(params["best_max_depth"]),
        "eta": 0.03,
        "gamma": float(params["best_gamma"]),
        "subsample": float(params["best_subsample"]),
        "colsample_bytree": float(params["best_colsample_bytree"]),
        "colsample_bylevel": 0.75,
        "reg_alpha": float(params["best_reg_alpha"]),
        "reg_lambda": float(params["best_reg_lambda"]),
        "min_child_weight": float(params["best_min_child_weight"]),
        "max_bin": 512,
        "grow_policy": "lossguide",
        "verbosity": 0,
    }
    num_boost_round = int(params["best_num_boost_round"])

    obj = make_ev_custom_objective(
        float(np.mean(avg_win)),
        float(np.mean(avg_loss)),
        float(np.mean(costs)),
    )

    dtrain = xgb.DMatrix(x, label=y)
    try:
        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            verbose_eval=False,
        )
    except Exception as e:
        print(f"  GPU failed for {symbol} ({e}), trying CPU...")
        xgb_params["device"] = "cpu"
        xgb_params.pop("sampling_method", None)
        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            verbose_eval=False,
        )

    # Save + versioned snapshot
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{symbol}.json")
    bst.save_model(model_path)
    version_dir = os.path.join(MODEL_DIR, "versions")
    os.makedirs(version_dir, exist_ok=True)
    from datetime import datetime as _dt
    date_tag = _dt.now().strftime("%Y%m%d")
    bst.save_model(os.path.join(version_dir, f"{symbol}_{date_tag}.json"))
    elapsed = time.time() - t0
    print(f"  OK   {symbol}: {len(y)} samples, {num_boost_round} rounds, "
          f"depth={int(params['best_max_depth'])}, EV={float(params['best_ev']):+.6f} "
          f"({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Retrain Elite Fleet")
    parser.add_argument("--csv", required=True, help="Path to filtered Optuna summary CSV")
    parser.add_argument("--symbol", help="Train only this symbol (for testing)")
    args = parser.parse_args()

    csv_path = os.path.join(SCRIPT_DIR, args.csv) if not os.path.isabs(args.csv) else args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]

    if args.symbol:
        rows = [r for r in rows if r["symbol"] == args.symbol]

    print(f"=== RETRAIN ELITE FLEET ===")
    print(f"CSV: {csv_path}")
    print(f"Symbols: {len(rows)}")
    print(f"Train size: {TRAIN_SIZE}")
    print(f"GPU: cuda:0 (P40)")
    print(f"Model dir: {MODEL_DIR}")
    print()

    ok = 0
    fail = 0
    t_total = time.time()

    for row in sorted(rows, key=lambda x: -float(x["best_ev"])):
        symbol = row["symbol"]
        try:
            if train_symbol(symbol, row):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  FAIL {symbol}: {e}")
            fail += 1

    elapsed = time.time() - t_total
    print()
    print(f"=== DONE: {ok} trained, {fail} skipped/failed ({elapsed:.0f}s) ===")


if __name__ == "__main__":
    main()
