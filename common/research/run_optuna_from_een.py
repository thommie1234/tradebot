#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

# Ensure module imports work when executed as script.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.integrated_pipeline import make_optuna_objective, set_polars_threads, tesla_p40_xgb_params
from engine.labeling import apply_triple_barrier
from research.train_ml_strategy import (
    infer_slippage_bps,
    infer_spread_bps,
    make_time_bars,
    sanitize_training_frame,
)
from engine.feature_builder import FEATURE_COLUMNS, build_bar_features


def parse_args():
    p = argparse.ArgumentParser(description="Run Optuna tuning for symbols listed in production/een.md.")
    p.add_argument("--een-md", default="trading_prop/production/een.md")
    p.add_argument(
        "--data-roots",
        default="/home/tradebot/ssd_data_1/tick_data,/home/tradebot/ssd_data_2/tick_data,/home/tradebot/data_1/tick_data,/home/tradebot/data_2/tick_data,/home/tradebot/data_3/tick_data",
    )
    p.add_argument("--timeframe", default="M15", help="M1/M5/M15/M30/H1")
    p.add_argument("--symbols", default="", help="Optional comma-separated symbols override")
    p.add_argument("--z-threshold", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--threads", type=int, default=28)
    p.add_argument("--train-size", type=int, default=1200)
    p.add_argument("--test-size", type=int, default=400)
    p.add_argument("--purge", type=int, default=64)
    p.add_argument("--embargo", type=int, default=64)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--horizon-bars", type=int, default=24)
    p.add_argument("--out-dir", default="trading_prop/production/optuna_from_een")
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


def run_symbol(symbol: str, args) -> dict:
    roots = [x for x in args.data_roots.split(",") if x]
    ticks_lf = load_symbol_ticks_lf(symbol, roots)
    if ticks_lf is None:
        return {"symbol": symbol, "status": "no_data"}

    ticks_df = ticks_lf.collect(streaming=True)
    if ticks_df.height == 0:
        return {"symbol": symbol, "status": "no_data"}
    if ticks_df.select(pl.col("size").sum()).item() <= 0:
        ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks_df)
    slippage_bps = infer_slippage_bps(symbol)
    bars = make_time_bars(ticks_df.select(["time", "price", "size"]), args.timeframe)
    feat = build_bar_features(bars, z_threshold=args.z_threshold)
    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=args.horizon_bars,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
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
    if feat.height < args.train_size + args.test_size + 128:
        return {"symbol": symbol, "status": "insufficient_rows", "rows": feat.height}

    feat_lf = feat.lazy()
    base = tesla_p40_xgb_params()
    # Runtime fallback if this XGBoost build has no CUDA support.
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
        train_size=args.train_size,
        test_size=args.test_size,
        purge=args.purge,
        embargo=args.embargo,
        slippage_multiplier=2.0,
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    return {
        "symbol": symbol,
        "status": "ok",
        "timeframe": args.timeframe,
        "best_ev": float(best.value),
        "best_num_boost_round": int(best.params["num_boost_round"]),
        "best_reg_alpha": float(best.params["reg_alpha"]),
        "best_reg_lambda": float(best.params["reg_lambda"]),
        "best_min_child_weight": float(best.params["min_child_weight"]),
        "trials": int(args.trials),
        "tree_method_used": base["tree_method"],
    }


def main():
    args = parse_args()
    set_polars_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    symbols = extract_symbols_from_een(args.een_md)
    if args.symbols:
        req = {s.strip() for s in args.symbols.split(",") if s.strip()}
        symbols = [s for s in symbols if s in req]
    if not symbols:
        raise SystemExit(f"No symbols found in {args.een_md}")

    rows = []
    for s in symbols:
        print(f"[optuna] tuning {s} ...")
        try:
            r = run_symbol(s, args)
        except Exception as e:
            r = {"symbol": s, "status": "error", "error": str(e)}
        print(r)
        rows.append(r)

    out = pl.from_dicts(rows).sort("symbol")
    out.write_parquet(os.path.join(args.out_dir, "summary.parquet"), compression="zstd")
    out.write_csv(os.path.join(args.out_dir, "summary.csv"))
    print(f"[optuna] done: {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
