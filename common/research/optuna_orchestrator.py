"""
Cluster-based Optuna orchestrator with SQLite persistence, MedianPruner,
and Calmar-weighted EV objective.

Usage:
    python3 optuna_orchestrator.py --active --trials 80
    python3 optuna_orchestrator.py --symbols AAVUSD,EUR_PLN,WHEAT.c --trials 5
    python3 optuna_orchestrator.py --all --workers 4 --trial-jobs 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

from research.integrated_pipeline import (
    expected_value_score,
    make_ev_custom_objective,
    purged_walk_forward_splits,
    set_polars_threads,
    tesla_p40_xgb_params,
)
from engine.feature_builder import (
    FEATURE_COLUMNS, build_bar_features,
    build_htf_features, merge_htf_features, htf_feature_columns,
    normalize_bar_columns,
)
from engine.labeling import apply_triple_barrier
from research.train_ml_strategy import (
    infer_slippage_bps,
    infer_spread_bps,
    make_time_bars,
    sanitize_training_frame,
)
from risk.position_sizing import ASSET_CLASS, SECTOR_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = REPO_ROOT_DIR / "config" / "sovereign_configs.json"

DATA_ROOTS = [
    "/home/tradebot/ssd_data_1/tick_data",
    "/home/tradebot/ssd_data_2/tick_data",
    "/home/tradebot/data_1/tick_data",
]

INFO_DIR = REPO_ROOT_DIR / "data" / "instrument_specs"
INFO_CSVS = [
    "crypto.csv",
    "Forex.csv",
    "forex exotic.csv",
    "cash.csv",
    "metals.csv",
    "Equities.csv",
]

# ---------------------------------------------------------------------------
# Cluster search spaces — per-asset-class hyperparameter ranges
# ---------------------------------------------------------------------------

CLUSTER_SEARCH_SPACES = {
    "crypto": {  # High vol, noisy — aggressive regularisation
        "max_depth": (3, 10),
        "gamma": (0.01, 10.0),
        "min_child_weight": (1.0, 150.0),
        "subsample": (0.4, 0.95),
        "colsample_bytree": (0.3, 0.9),
        "colsample_bylevel": (0.3, 1.0),
        "reg_alpha": (1e-3, 100.0),
        "reg_lambda": (1e-3, 100.0),
        "eta": (0.005, 0.3),
        "num_boost_round": (50, 2000),
    },
    "forex": {  # Mean-reverting, small signals — shallow trees, tight reg
        "max_depth": (2, 8),
        "gamma": (0.001, 2.0),
        "min_child_weight": (0.5, 50.0),
        "subsample": (0.5, 0.95),
        "colsample_bytree": (0.4, 0.95),
        "colsample_bylevel": (0.3, 1.0),
        "reg_alpha": (1e-5, 20.0),
        "reg_lambda": (1e-5, 50.0),
        "eta": (0.005, 0.3),
        "num_boost_round": (100, 2000),
    },
    "commodity": {  # Trending — focus on subsample diversity
        "max_depth": (2, 9),
        "gamma": (0.01, 5.0),
        "min_child_weight": (1.0, 80.0),
        "subsample": (0.4, 0.95),
        "colsample_bytree": (0.3, 0.95),
        "colsample_bylevel": (0.3, 1.0),
        "reg_alpha": (1e-4, 50.0),
        "reg_lambda": (1e-4, 50.0),
        "eta": (0.005, 0.3),
        "num_boost_round": (100, 2000),
    },
    "index": {
        "max_depth": (2, 9),
        "gamma": (0.001, 3.0),
        "min_child_weight": (0.3, 60.0),
        "subsample": (0.4, 0.95),
        "colsample_bytree": (0.3, 0.95),
        "colsample_bylevel": (0.3, 1.0),
        "reg_alpha": (1e-5, 30.0),
        "reg_lambda": (1e-5, 50.0),
        "eta": (0.005, 0.3),
        "num_boost_round": (100, 2000),
    },
}
# equity mirrors index
CLUSTER_SEARCH_SPACES["equity"] = CLUSTER_SEARCH_SPACES["index"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_for_symbol(symbol: str) -> str:
    """Map *symbol* to an asset-class cluster via SECTOR_MAP → ASSET_CLASS."""
    sector = SECTOR_MAP.get(symbol, "unknown")
    return ASSET_CLASS.get(sector, "index")  # default to index ranges


# ---------------------------------------------------------------------------
# Broker symbol specs from information/ CSVs
# ---------------------------------------------------------------------------

def _normalise_symbol(csv_symbol: str) -> str:
    """Convert CSV symbol format (AUD/CAD, XAU/USD) to tick-data format (AUD_CAD, XAU_USD)."""
    return csv_symbol.strip().replace("/", "_")


def load_symbol_specs() -> dict[str, dict]:
    """Load all broker specs from information/ CSVs into a unified lookup.

    Returns dict keyed by normalised symbol name (e.g. AUD_CAD, BTCUSD,
    WHEAT.c) with fields: commission, commission_type, contract_size,
    margin_percent, leverage, asset_class.
    """
    import csv

    specs: dict[str, dict] = {}
    for csv_name in INFO_CSVS:
        csv_path = INFO_DIR / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            n_cols = len(headers)
            for fields in reader:
                if not fields or not fields[0].strip():
                    continue
                # Some descriptions contain a comma (e.g. "Bitcoin vs US Dollar, Spot CFD")
                # which creates an extra field — rejoin it
                if len(fields) == n_cols + 1:
                    fields = [fields[0], fields[1] + "," + fields[2]] + fields[3:]
                row = dict(zip(headers, fields))
                raw_sym = row.get("symbol", "").strip()
                if not raw_sym:
                    continue
                sym = _normalise_symbol(raw_sym)
                try:
                    commission = float(row.get("commission", "0") or "0")
                except ValueError:
                    commission = 0.0
                specs[sym] = {
                    "commission": commission,
                    "commission_type": (row.get("commission_type", "") or "").strip(),
                    "contract_size": (row.get("contract_size", "") or "").strip(),
                    "asset_class_broker": (row.get("asset_class", "") or "").strip(),
                    "margin_percent": row.get("margin_percent", ""),
                    "leverage": (row.get("leverage", "") or "").strip(),
                }
    return specs


# Module-level cache — loaded once per process
_SYMBOL_SPECS: dict[str, dict] | None = None


def _get_symbol_specs() -> dict[str, dict]:
    global _SYMBOL_SPECS
    if _SYMBOL_SPECS is None:
        _SYMBOL_SPECS = load_symbol_specs()
    return _SYMBOL_SPECS


def broker_commission_bps(symbol: str) -> float:
    """Return the real broker commission in basis points for *symbol*.

    Commission types from FTMO CSVs:
      PERCENT/VOLUME  — value is already in %, e.g. 0.065 → 6.5 bps
      USD/LOT         — $5 per 100k lot → 0.5 bps (at par)
      NO_COMMISSION   — 0 bps (spread-only instruments)
    """
    specs = _get_symbol_specs()
    spec = specs.get(symbol)
    if spec is None:
        # Fallback: 3.0 bps (old hardcoded default)
        return 3.0

    comm = spec["commission"]
    ctype = spec["commission_type"]

    if ctype == "NO_COMMISSION" or comm == 0.0:
        return 0.0
    if ctype == "PERCENT/VOLUME":
        # commission is in percent, convert to bps (× 100)
        # e.g. crypto 0.065% → 6.5 bps
        return comm * 100.0
    if ctype == "USD/LOT":
        # $X per standard lot (100,000 units face value)
        # At par (price~1.0): $5/100000 = 0.00005 = 0.5 bps
        # This is an approximation; exact bps depends on current price
        contract = spec.get("contract_size", "")
        try:
            lot_size = float(contract)
        except (ValueError, TypeError):
            lot_size = 100_000.0  # forex default
        return (comm / lot_size) * 1e4

    # Unknown type — conservative fallback
    return 3.0


def broker_slippage_bps(symbol: str) -> float:
    """Return realistic slippage estimate in bps based on broker specs.

    Uses leverage/margin as proxy for liquidity:
      High leverage (1:100, forex majors) → tight slippage
      Low leverage (1:3.33, crypto/equities) → wider slippage
    """
    specs = _get_symbol_specs()
    spec = specs.get(symbol)
    if spec is None:
        return infer_slippage_bps(symbol)  # fallback to heuristic

    ctype = spec["commission_type"]
    broker_class = spec.get("asset_class_broker", "")

    if "Crypto" in broker_class:
        return 8.0   # Wide spreads, volatile order books
    if broker_class == "Forex":
        return 1.5   # Deep liquidity, tight execution
    if "Exotic" in broker_class:
        return 3.0   # Less liquid than majors
    if "Metals" in broker_class:
        return 2.0   # Decent liquidity
    if "Equities" in broker_class:
        return 2.5   # Single-stock slippage
    if "Cash" in broker_class:
        # Indices and commodities
        if any(x in symbol for x in ("OIL", "NATGAS", "HEATOIL", "DXY")):
            return 3.0
        if any(x in symbol for x in ("COCOA", "COFFEE", "CORN", "SOYBEAN",
                                       "WHEAT", "COTTON", "SUGAR")):
            return 4.0
        return 1.5   # Index CFDs — tight

    return infer_slippage_bps(symbol)


def symbol_files(symbol: str, data_roots: list[str]) -> list[str]:
    """Collect all parquet tick files for *symbol* across data roots."""
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
    """Scan and concatenate tick parquets for *symbol*."""
    files = symbol_files(symbol, data_roots)
    if not files:
        return None
    lfs = [
        pl.scan_parquet(fp).select(
            ["time", "bid", "ask", "last", "volume", "volume_real"]
        )
        for fp in files
    ]
    lf = (
        pl.concat(lfs, how="vertical")
        .sort("time")
        .with_columns(
            [
                pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
                pl.when(pl.col("last") > 0)
                .then(pl.col("last"))
                .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
                .alias("price"),
                pl.when(pl.col("volume_real") > 0)
                .then(pl.col("volume_real"))
                .otherwise(pl.col("volume"))
                .alias("size"),
            ]
        )
        .drop_nulls(["time", "price", "size"])
    )
    return lf.select(["time", "bid", "ask", "price", "size"])


def calmar_weighted_score(ev_scores: list[float]) -> float:
    """Weight mean-EV by a Calmar-style drawdown penalty."""
    mean_ev = float(np.mean(ev_scores))
    if mean_ev <= 0:
        return mean_ev
    cumulative = np.cumsum(ev_scores)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    calmar = mean_ev / max(max_dd, abs(mean_ev) * 0.1)
    calmar_factor = min(2.0, max(0.1, calmar))
    return mean_ev * calmar_factor


# ---------------------------------------------------------------------------
# Clustered objective factory
# ---------------------------------------------------------------------------

def make_clustered_objective(
    x: np.ndarray,
    y: np.ndarray,
    avg_win: np.ndarray,
    avg_loss: np.ndarray,
    costs: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    base_params: dict,
    cluster: str,
):
    """Return an Optuna objective that uses cluster-specific search ranges
    and reports intermediate fold values for MedianPruner."""

    space = CLUSTER_SEARCH_SPACES.get(cluster, CLUSTER_SEARCH_SPACES["index"])

    def objective(trial) -> float:
        params = dict(base_params)
        params["max_depth"] = trial.suggest_int(
            "max_depth", space["max_depth"][0], space["max_depth"][1],
        )
        params["gamma"] = trial.suggest_float(
            "gamma", space["gamma"][0], space["gamma"][1], log=True,
        )
        params["min_child_weight"] = trial.suggest_float(
            "min_child_weight",
            space["min_child_weight"][0],
            space["min_child_weight"][1],
            log=True,
        )
        params["subsample"] = trial.suggest_float(
            "subsample", space["subsample"][0], space["subsample"][1],
        )
        params["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree",
            space["colsample_bytree"][0],
            space["colsample_bytree"][1],
        )
        params["reg_alpha"] = trial.suggest_float(
            "reg_alpha", space["reg_alpha"][0], space["reg_alpha"][1], log=True,
        )
        params["reg_lambda"] = trial.suggest_float(
            "reg_lambda", space["reg_lambda"][0], space["reg_lambda"][1], log=True,
        )
        params["eta"] = trial.suggest_float(
            "eta", space["eta"][0], space["eta"][1], log=True,
        )
        params["colsample_bylevel"] = trial.suggest_float(
            "colsample_bylevel",
            space["colsample_bylevel"][0],
            space["colsample_bylevel"][1],
        )
        max_rounds = trial.suggest_int(
            "num_boost_round",
            space["num_boost_round"][0],
            space["num_boost_round"][1],
        )

        ev_scores: list[float] = []
        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
            dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

            fold_avg_win = float(np.mean(avg_win[tr_idx]))
            fold_avg_loss = float(np.mean(avg_loss[tr_idx]))
            fold_cost = float(np.mean(costs[tr_idx]))
            obj = make_ev_custom_objective(fold_avg_win, fold_avg_loss, fold_cost)

            bst = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=max_rounds,
                obj=obj,
                evals=[(dtest, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            p = bst.predict(dtest)
            ev = expected_value_score(
                proba=p,
                avg_win=avg_win[te_idx],
                avg_loss=avg_loss[te_idx],
                costs=costs[te_idx],
            )
            ev_scores.append(ev)

            # Intermediate report for MedianPruner
            trial.report(float(np.mean(ev_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return calmar_weighted_score(ev_scores)

    return objective


# ---------------------------------------------------------------------------
# Per-symbol worker
# ---------------------------------------------------------------------------

def _load_htf_bars(symbol: str, timeframe: str, bar_roots: str) -> pl.DataFrame | None:
    """Load pre-downloaded bar data for a symbol/timeframe from bar_roots."""
    from pathlib import Path
    variants = [symbol, symbol.replace("_", ""), symbol.replace("_", "/")]
    for variant in variants:
        bar_dir = Path(bar_roots) / timeframe / variant
        if not bar_dir.is_dir():
            continue
        pq_files = sorted(bar_dir.glob("*.parquet"))
        if not pq_files:
            continue
        try:
            df = pl.concat([pl.read_parquet(f) for f in pq_files]).sort("time")
            # Normalize column names
            rename_map = {}
            if "tick_volume" in df.columns and "volume" not in df.columns:
                rename_map["tick_volume"] = "volume"
            if rename_map:
                df = df.rename(rename_map)
            if "volume" not in df.columns:
                df = df.with_columns(pl.lit(1.0).alias("volume"))
            # Ensure time is datetime for HTF merge
            if df["time"].dtype == pl.Int64:
                df = df.with_columns(
                    pl.from_epoch(pl.col("time"), time_unit="s").alias("time")
                )
            return df
        except Exception as e:
            print(f"  [bar-load] {variant}/{timeframe}: {type(e).__name__}: {e}")
            continue
    return None


def _aggregate_m5_to_target(m5_bars: pl.DataFrame, target_tf: str) -> pl.DataFrame:
    """Aggregate M5 bars to a target timeframe (e.g. H1, H4)."""
    tf_map = {"M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}
    every = tf_map.get(target_tf, "1h")
    agg = (
        m5_bars
        .sort("time")
        .group_by_dynamic("time", every=every)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
        .sort("time")
        .filter(pl.col("volume") > 0)
    )
    return agg


def run_symbol(symbol: str, args_dict: dict) -> dict:
    """Load data, build features, create/resume Optuna study, optimise."""
    # CUDA_VISIBLE_DEVICES inherited from parent — do NOT override
    set_polars_threads(4)

    cluster = cluster_for_symbol(symbol)
    roots = list(args_dict["data_roots"])
    use_htf = args_dict.get("htf", False)
    bar_roots = args_dict.get("bar_roots", "")

    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)

    # Try pre-downloaded bars first (much more history than tick data)
    bars = None
    spread_bps = None
    m5_agg = args_dict.get("m5_agg", False)

    # Always try tick data for spread (bar "spread" column is in MT5 points, not bps)
    _tick_lf_for_spread = load_symbol_ticks_lf(symbol, roots)
    if _tick_lf_for_spread is not None:
        _tick_df_spread = _tick_lf_for_spread.collect()
        if _tick_df_spread.height > 0:
            spread_bps = infer_spread_bps(_tick_df_spread)
        del _tick_df_spread
    del _tick_lf_for_spread

    if bar_roots:
        if m5_agg:
            # Load M5 bars and aggregate to target timeframe (better volume)
            m5_raw = _load_htf_bars(symbol, "M5", bar_roots)
            if m5_raw is not None and m5_raw.height >= 200:
                bars = _aggregate_m5_to_target(m5_raw, args_dict["timeframe"])

        if bars is None:
            # Load target timeframe bars directly
            preloaded = _load_htf_bars(symbol, args_dict["timeframe"], bar_roots)
            if preloaded is not None and preloaded.height >= 200:
                # Rename tick_volume -> volume if needed
                if "tick_volume" in preloaded.columns and "volume" not in preloaded.columns:
                    preloaded = preloaded.rename({"tick_volume": "volume"})
                if "volume" not in preloaded.columns:
                    preloaded = preloaded.with_columns(pl.lit(1.0).alias("volume"))
                bars = preloaded.select(["time", "open", "high", "low", "close", "volume"])

    # Fallback spread estimate if no tick data available
    if spread_bps is None or spread_bps <= 0:
        spread_bps = fee_bps + slippage_bps

    # Fallback to tick data for bars if no bar data found
    if bars is None:
        ticks_lf = load_symbol_ticks_lf(symbol, roots)
        if ticks_lf is None:
            return {"symbol": symbol, "status": "no_data", "cluster": cluster}
        ticks_df = ticks_lf.collect()
        if ticks_df.height == 0:
            return {"symbol": symbol, "status": "no_data", "cluster": cluster}
        if ticks_df.select(pl.col("size").sum()).item() <= 0:
            ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))
        if spread_bps is None or spread_bps <= 0:
            spread_bps = infer_spread_bps(ticks_df)
        bars = make_time_bars(
            ticks_df.select(["time", "price", "size"]), args_dict["timeframe"],
        )

    # Ensure time is datetime (build_bar_features requires it)
    if bars["time"].dtype == pl.Int64:
        bars = bars.with_columns(pl.from_epoch(pl.col("time"), time_unit="s").alias("time"))

    # Filter by min_date if specified (e.g. last 10 years)
    min_date_ts = args_dict.get("min_date_ts")
    if min_date_ts:
        min_dt = datetime.fromtimestamp(min_date_ts)
        bars = bars.filter(pl.col("time") >= min_dt)

    feat = build_bar_features(bars, z_threshold=args_dict["z_threshold"])

    # HTF feature enrichment (H4 + D1)
    if use_htf and bar_roots:
        # If m5-agg, aggregate M5 to HTF timeframes too
        m5_raw_for_htf = None
        if m5_agg:
            m5_raw_for_htf = _load_htf_bars(symbol, "M5", bar_roots)

        for htf_tf, prefix in [("H4", "h4"), ("D1", "d1")]:
            htf_bars = None
            if m5_raw_for_htf is not None and m5_raw_for_htf.height >= 200:
                htf_bars = _aggregate_m5_to_target(m5_raw_for_htf, htf_tf)
            if htf_bars is None or htf_bars.height < 50:
                htf_bars = _load_htf_bars(symbol, htf_tf, bar_roots)
            if htf_bars is not None and htf_bars.height >= 50:
                htf_feat = build_htf_features(htf_bars, prefix)
                feat = merge_htf_features(feat, htf_feat, prefix)
            else:
                # Fill with zeros if no HTF data
                for col in [f"{prefix}_{n}" for n in
                            ["ret1", "ret3", "ma_cross", "vol20",
                             "atr_ratio", "z20", "adx_proxy", "regime"]]:
                    feat = feat.with_columns(pl.lit(0.0).alias(col))

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
            pl.lit(float(fee_bps)).alias("fee_bps"),
            pl.lit(float(spread_bps)).alias("spread_bps"),
            pl.lit(float(slippage_bps)).alias("slippage_bps"),
        ]
    ).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    # Dynamic train/test sizing — scale to available data, never reject
    MIN_TRAIN = 300
    MIN_TEST = 80
    requested_train = args_dict["train_size"]
    requested_test = args_dict["test_size"]
    purge_embargo = args_dict.get("purge", 24) + args_dict.get("embargo", 24)
    available = feat.height - purge_embargo  # reserve for purge + embargo
    if available < MIN_TRAIN + MIN_TEST:
        return {
            "symbol": symbol,
            "status": "insufficient_rows",
            "rows": feat.height,
            "cluster": cluster,
        }
    if available < requested_train + requested_test:
        # Scale proportionally: keep ~80% train, ~20% test
        ratio = requested_train / (requested_train + requested_test)
        effective_train = max(MIN_TRAIN, int(available * ratio))
        effective_test = max(MIN_TEST, available - effective_train)
        args_dict = dict(args_dict)  # copy to avoid mutating caller
        args_dict["train_size"] = effective_train
        args_dict["test_size"] = effective_test

    # Prepare numpy arrays — use expanded features if HTF enabled
    feature_cols = FEATURE_COLUMNS + (htf_feature_columns() if use_htf else [])
    cols = feature_cols + [
        "target", "avg_win", "avg_loss", "fee_bps", "spread_bps", "slippage_bps",
    ]
    df = feat.select(cols).drop_nulls(cols)
    x = df.select(feature_cols).to_numpy()
    y = df["target"].to_numpy().astype(np.float32)
    avg_win = df["avg_win"].to_numpy().astype(np.float64)
    avg_loss = df["avg_loss"].to_numpy().astype(np.float64)
    costs = (
        (df["fee_bps"] + df["spread_bps"] + df["slippage_bps"] * 2.0).to_numpy()
        / 1e4
    ).astype(np.float64)

    splits = purged_walk_forward_splits(
        n_samples=len(df),
        train_size=args_dict["train_size"],
        test_size=args_dict["test_size"],
        purge=args_dict["purge"],
        embargo=args_dict["embargo"],
    )
    if not splits:
        return {
            "symbol": symbol,
            "status": "no_folds",
            "rows": len(df),
            "cluster": cluster,
        }

    # Base XGB params — probe GPU availability
    base = tesla_p40_xgb_params()
    try:
        probe = xgb.DMatrix(
            np.array([[0.0], [1.0]]), label=np.array([0.0, 1.0]),
        )
        xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"},
            probe,
            num_boost_round=1,
        )
    except Exception:
        base["device"] = "cpu"
        base["sampling_method"] = "uniform"

    device_used = base["device"]

    # Build clustered objective
    objective = make_clustered_objective(
        x=x,
        y=y,
        avg_win=avg_win,
        avg_loss=avg_loss,
        costs=costs,
        splits=splits,
        base_params=base,
        cluster=cluster,
    )

    # SQLite-backed study — resume if exists
    out_dir = args_dict["out_dir"]
    storage_url = f"sqlite:///{out_dir}/optuna_studies.db"
    study_name = f"{symbol}_{args_dict['timeframe']}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42 + int(datetime.now().strftime("%Y%m%d")) % 10000,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=20, n_warmup_steps=5,
        ),
    )

    target_trials = args_dict["trials"]
    discovery_cutoff = args_dict.get("discovery_cutoff", 500)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Phase 1: Discovery (up to discovery_cutoff trials) ──
    current = len(study.trials)
    if current < discovery_cutoff:
        phase1_remaining = discovery_cutoff - current
        study.optimize(
            objective,
            n_trials=phase1_remaining,
            n_jobs=args_dict["trial_jobs"],
            show_progress_bar=False,
        )

    # ── Kill Switch: if best EV < 0 after discovery, stop here ──
    if study.best_trial is None:
        return {
            "symbol": symbol,
            "status": "no_completed_trials",
            "rows": len(df),
            "cluster": cluster,
        }

    discovery_ev = study.best_trial.value
    no_kill = args_dict.get("no_kill", False)
    if not no_kill and discovery_ev < 0 and len(study.trials) <= discovery_cutoff + 50:
        return {
            "symbol": symbol,
            "status": "killed_negative_ev",
            "gpu": 0,
            "timeframe": args_dict["timeframe"],
            "best_ev": float(discovery_ev),
            "trials": len(study.trials),
            "rows": len(df),
            "device_used": device_used,
            "cluster": cluster,
            "calmar_score": float(discovery_ev),
            "fee_bps": fee_bps,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
        }

    # ── Phase 2: Deep dive (all symbols if --no-kill, else only EV >= 0) ──
    remaining = max(0, target_trials - len(study.trials))
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=args_dict["trial_jobs"],
            show_progress_bar=False,
        )

    # Free VRAM and memory between symbols in the same worker process
    gc.collect()

    best = study.best_trial
    return {
        "symbol": symbol,
        "status": "ok",
        "gpu": 0,
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
        "best_eta": float(best.params.get("eta", 0.03)),
        "best_colsample_bylevel": float(best.params.get("colsample_bylevel", 1.0)),
        "trials": len(study.trials),
        "rows": len(df),
        "device_used": device_used,
        "cluster": cluster,
        "calmar_score": float(best.value),
        "fee_bps": fee_bps,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "n_features": len(feature_cols),
        "htf_enabled": use_htf,
    }


# ---------------------------------------------------------------------------
# Symbol discovery
# ---------------------------------------------------------------------------

def discover_symbols_from_data(data_roots: list[str], min_parquets: int = 1) -> list[str]:
    """Discover all symbols that have parquet data in data roots."""
    syms: set[str] = set()
    for root in data_roots:
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


def load_active_symbols() -> list[str]:
    """Load active symbols from sovereign_configs.json."""
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"sovereign_configs.json not found at {CONFIG_PATH}\n"
            "Run with --symbols or --all instead."
        )
    with open(CONFIG_PATH) as f:
        configs = json.load(f)
    # Skip non-symbol keys like margin_leverage
    skip_keys = {"margin_leverage"}
    return sorted(k for k in configs.keys() if k not in skip_keys)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster-based Optuna optimisation with SQLite persistence",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--active", action="store_true",
        help="Use symbols from sovereign_configs.json",
    )
    grp.add_argument(
        "--all", action="store_true",
        help="Auto-discover all symbols with tick data",
    )
    grp.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbol list",
    )

    p.add_argument("--trials", type=int, default=80, help="Trials per symbol")
    p.add_argument("--timeframe", type=str, default="H1")
    p.add_argument("--workers", type=int, default=4, help="Parallel symbol workers")
    p.add_argument("--trial-jobs", type=int, default=2, help="Parallel trials per symbol")
    p.add_argument("--train-size", type=int, default=4800)
    p.add_argument("--test-size", type=int, default=1200)
    p.add_argument("--purge", type=int, default=24)
    p.add_argument("--embargo", type=int, default=24)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--horizon-bars", type=int, default=24)
    p.add_argument("--z-threshold", type=float, default=1.0)
    p.add_argument("--discovery-cutoff", type=int, default=500,
                   help="Trials for discovery phase; kill if EV<0 after this")
    p.add_argument("--no-kill", action="store_true",
                   help="Never kill symbols with negative EV — always run full trials")
    p.add_argument(
        "--out-dir", type=str, default="",
        help="Output directory (default: timestamped)",
    )
    p.add_argument(
        "--htf", action="store_true",
        help="Add H4+D1 higher-timeframe features (55 features instead of 39)",
    )
    p.add_argument(
        "--m5-agg", action="store_true",
        help="Aggregate M5 bars to target timeframe (better volume, more data)",
    )
    p.add_argument(
        "--bar-roots", type=str,
        default="/home/tradebot/ssd_data_2/bars",
        help="Root directory for pre-downloaded bar data",
    )
    p.add_argument(
        "--min-years", type=int, default=0,
        help="Only use the last N years of data (0 = all)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve symbols
    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    elif args.active:
        symbols = load_active_symbols()
    else:
        symbols = discover_symbols_from_data(DATA_ROOTS)

    if not symbols:
        raise SystemExit("No symbols found")

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"optuna_clustered_{args.timeframe}_{ts}"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    args_dict = {
        "data_roots": DATA_ROOTS,
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
        "discovery_cutoff": args.discovery_cutoff,
        "no_kill": args.no_kill,
        "out_dir": out_dir,
        "htf": args.htf,
        "m5_agg": args.m5_agg,
        "bar_roots": args.bar_roots,
        "min_date_ts": int((datetime.now() - timedelta(days=args.min_years * 365)).timestamp()) if args.min_years > 0 else None,
    }

    m5_tag = " | M5→agg" if args.m5_agg else ""
    print(f"[orchestrator] {len(symbols)} symbols | {args.timeframe}{m5_tag} | "
          f"z={args.z_threshold} | "
          f"discovery={args.discovery_cutoff} → deep={args.trials} trials | "
          f"{args.workers} workers | {args.trial_jobs} trial-jobs/sym")
    print(f"  Output: {out_dir}")
    for sym in symbols:
        fee = broker_commission_bps(sym)
        slip = broker_slippage_bps(sym)
        print(f"  {sym:20s} -> {cluster_for_symbol(sym):9s}  "
              f"fee={fee:.1f}bps  slip={slip:.1f}bps")
    print()

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_symbol, sym, args_dict): sym for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                r = future.result()
                status = r.get("status", "unknown")
                cluster = r.get("cluster", "?")
                if status == "ok":
                    print(
                        f"  ✓ {sym:20s} [{cluster:9s}] EV={r['best_ev']:.6f}  "
                        f"calmar={r['calmar_score']:.6f}  "
                        f"rounds={r['best_num_boost_round']}  "
                        f"trials={r['trials']}"
                    )
                elif status == "killed_negative_ev":
                    print(
                        f"  ✗ {sym:20s} [{cluster:9s}] KILLED @ {r.get('trials',0)} trials  "
                        f"EV={r.get('best_ev',0):.6f}"
                    )
                else:
                    print(f"  - {sym:20s} [{cluster:9s}] {status}")
            except Exception as e:
                r = {"symbol": sym, "status": "error", "error": str(e)}
                print(f"  {sym:20s} ERROR: {e}")
            rows.append(r)

    out_df = pl.from_dicts(rows).sort("symbol")
    out_df.write_parquet(os.path.join(out_dir, "summary.parquet"), compression="zstd")
    out_df.write_csv(os.path.join(out_dir, "summary.csv"))
    print(f"\n[orchestrator] Done — {os.path.join(out_dir, 'summary.csv')}")
    print(f"[orchestrator] SQLite DB — {os.path.join(out_dir, 'optuna_studies.db')}")


if __name__ == "__main__":
    main()
