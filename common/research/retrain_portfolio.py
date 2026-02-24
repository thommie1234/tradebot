"""
Retrain XGBoost models for portfolio symbols using engine.feature_builder (39 features).

Reads Optuna CSV with best hyperparams, trains each symbol, saves to sovereign_models/.

Usage:
    python3 research/retrain_portfolio.py --csv models/optuna_results/combined_portfolio_optuna.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import polars as pl
import xgboost as xgb

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from engine.labeling import apply_triple_barrier
from trading_prop.ml.train_ml_strategy import (
    infer_spread_bps,
    infer_slippage_bps,
    make_time_bars,
    sanitize_training_frame,
)
from trading_prop.ml.integrated_pipeline import make_ev_custom_objective
from trading_prop.production.optuna_orchestrator import DATA_ROOTS

MODEL_DIR = REPO_ROOT / "models" / "sovereign_models"
TRAIN_SIZE = 1200


def load_ticks(symbol: str) -> pl.DataFrame | None:
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
    t0 = time.time()

    ticks = load_ticks(symbol)
    if ticks is None:
        print(f"  SKIP {symbol}: no tick data")
        return False

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    if bars.height < TRAIN_SIZE + 100:
        print(f"  SKIP {symbol}: only {bars.height} H1 bars")
        return False

    # engine.feature_builder → 39 features
    spread_bps = infer_spread_bps(ticks)
    slippage_bps = infer_slippage_bps(symbol)
    feat = build_bar_features(bars, z_threshold=0.0)

    # Triple barrier labeling
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
        print(f"  SKIP {symbol}: {feat.height} rows after sanitization")
        return False

    train_df = feat.tail(TRAIN_SIZE)
    x = train_df.select(FEATURE_COLUMNS).to_numpy()
    y = train_df["target"].to_numpy().astype(np.float32)
    avg_win = train_df["avg_win"].to_numpy().astype(np.float64)
    avg_loss = train_df["avg_loss"].to_numpy().astype(np.float64)
    costs = (
        (train_df["fee_bps"] + train_df["spread_bps"] + train_df["slippage_bps"] * 2.0).to_numpy() / 1e4
    ).astype(np.float64)

    # XGBoost params from Optuna
    xgb_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cuda",
        "sampling_method": "gradient_based",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(float(params["best_max_depth"])),
        "gamma": float(params["best_gamma"]),
        "subsample": float(params["best_subsample"]),
        "colsample_bytree": float(params["best_colsample_bytree"]),
        "reg_alpha": float(params["best_reg_alpha"]),
        "reg_lambda": float(params["best_reg_lambda"]),
        "min_child_weight": float(params["best_min_child_weight"]),
        "max_bin": 512,
        "grow_policy": "lossguide",
        "verbosity": 0,
    }
    # Optional params
    if "best_eta" in params and params["best_eta"]:
        xgb_params["eta"] = float(params["best_eta"])
    else:
        xgb_params["eta"] = 0.03
    if "best_colsample_bylevel" in params and params["best_colsample_bylevel"]:
        xgb_params["colsample_bylevel"] = float(params["best_colsample_bylevel"])

    num_boost_round = int(float(params["best_num_boost_round"]))

    obj = make_ev_custom_objective(
        float(np.mean(avg_win)),
        float(np.mean(avg_loss)),
        float(np.mean(costs)),
    )

    dtrain = xgb.DMatrix(x, label=y)
    try:
        bst = xgb.train(
            params=xgb_params, dtrain=dtrain,
            num_boost_round=num_boost_round, obj=obj, verbose_eval=False,
        )
    except Exception as e:
        print(f"  GPU fallback for {symbol}: {e}")
        xgb_params["device"] = "cpu"
        xgb_params.pop("sampling_method", None)
        bst = xgb.train(
            params=xgb_params, dtrain=dtrain,
            num_boost_round=num_boost_round, obj=obj, verbose_eval=False,
        )

    # Verify prediction quality
    preds = bst.predict(dtrain)
    pred_std = float(np.std(preds))

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{symbol}.json"
    bst.save_model(str(model_path))

    # Version
    version_dir = MODEL_DIR / "versions"
    version_dir.mkdir(exist_ok=True)
    from datetime import datetime
    date_tag = datetime.now().strftime("%Y%m%d")
    bst.save_model(str(version_dir / f"{symbol}_{date_tag}.json"))

    elapsed = time.time() - t0
    label_rate = float(np.mean(y))
    print(
        f"  OK {symbol:15s}  {len(y)} samples  {num_boost_round} rounds  "
        f"depth={xgb_params['max_depth']}  "
        f"pred_std={pred_std:.4f}  label_rate={label_rate:.2f}  "
        f"EV={float(params['best_ev']):+.6f}  ({elapsed:.1f}s)"
    )
    return pred_std > 0.01  # True only if model learned something


def main():
    parser = argparse.ArgumentParser(description="Retrain portfolio models (39 features)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", help="Train only this symbol")
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]

    if args.symbol:
        rows = [r for r in rows if r["symbol"] == args.symbol]

    print(f"=== RETRAIN PORTFOLIO (39 features, engine.feature_builder) ===")
    print(f"CSV: {args.csv}")
    print(f"Symbols: {len(rows)}")
    print(f"Train size: {TRAIN_SIZE}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Model dir: {MODEL_DIR}")
    print()

    ok = 0
    flat = 0
    fail = 0

    for row in sorted(rows, key=lambda x: -float(x["best_ev"])):
        symbol = row["symbol"]
        try:
            result = train_symbol(symbol, row)
            if result:
                ok += 1
            else:
                flat += 1
        except Exception as e:
            print(f"  FAIL {symbol}: {e}")
            import traceback
            traceback.print_exc()
            fail += 1

    print(f"\n=== DONE: {ok} trained (learned), {flat} trained (flat), {fail} failed ===")


if __name__ == "__main__":
    main()
