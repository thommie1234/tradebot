#!/usr/bin/env python3
"""
ML-Only Backtest for Sovereign Sniper v0.5

This script isolates the ML model's performance by:
1. Running the same Walk-Forward Optimization (WFO) as the full simulator.
2. Instead of a complex portfolio simulation, it calculates a theoretical P/L
   based on the triple-barrier-return (`tb_ret`) for each signal that
   crosses the `ML_THRESHOLD`.
3. It aggregates these returns and calculates weekly performance metrics.
4. Uses a new CostEngine to apply realistic commissions from CSV files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

# --- Boilerplate to access project modules ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from research.integrated_pipeline import (
    make_ev_custom_objective,
    purged_walk_forward_splits,
)
from engine.labeling import apply_triple_barrier
from research.train_ml_strategy import (
    infer_spread_bps,
    make_time_bars,
    sanitize_training_frame,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --- Core Parameters from your analysis ---
ML_THRESHOLD = 0.65
Z_THRESHOLD = 0.0
RISK_PER_TRADE = 0.01 # Using a fixed 1% risk for this theoretical backtest

# --- New Cost Engine ---
class CostEngine:
    """Loads and provides trading costs from CSV files."""
    def __init__(self, info_dir: str | Path):
        self.info_dir = Path(info_dir)
        self.cost_data = self._load_all_costs()

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().replace("/", "")

    def _load_all_costs(self) -> dict:
        """Loads all relevant CSVs and stores cost data in a dict."""
        cost_dict = {}
        files_to_load = [
            "Forex.csv", "forex exotic.csv", "metals.csv", 
            "crypto.csv", "cash.csv", "Equities.csv"
        ]
        
        for filename in files_to_load:
            file_path = self.info_dir / filename
            if not file_path.exists():
                print(f"[CostEngine] WARNING: {filename} not found.")
                continue
            
            df = pl.read_csv(file_path)
            for row in df.iter_rows(named=True):
                symbol = self._normalize_symbol(row['symbol'])
                commission_usd = row.get('commission', 0.0)
                contract_size = row.get('contract_size', 100000.0)
                
                # Convert commission to bps (basis points)
                # Formula: (Commission_USD_per_Lot / Contract_Size) * 10,000
                # This is a simplification; a more precise calculation would need the current price.
                # For a 100k contract and $5 commission, this is (5/100000)*10000 = 0.5 bps
                # We'll use this as the base commission, and add spread.
                if contract_size > 0:
                    commission_bps = (float(commission_usd) / float(contract_size)) * 10000
                else:
                    commission_bps = 0.0

                cost_dict[symbol] = {"commission_bps": commission_bps}
        
        print(f"[CostEngine] Loaded cost data for {len(cost_dict)} symbols.")
        return cost_dict

    def get_total_cost_bps(self, symbol: str, spread_bps: float) -> float:
        """Returns total cost (commission + spread) in bps."""
        normalized_symbol = self._normalize_symbol(symbol)
        costs = self.cost_data.get(normalized_symbol)
        
        commission_bps = 0.0
        if costs:
            commission_bps = costs.get("commission_bps", 0.0)
        
        # Total cost is commission (round trip) + spread
        return (commission_bps * 2) + spread_bps

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="ML-Only Signal Backtest with Realistic Costs")
    parser.add_argument("--csv", required=True, help="Path to the Optuna summary CSV")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--purge", type=int, default=8)
    parser.add_argument("--embargo", type=int, default=8)
    return parser.parse_args()

def load_ticks(symbol: str) -> pl.DataFrame | None:
    """Loads all tick data for a symbol from standard data roots."""
    DATA_ROOTS = [
        "/home/tradebot/ssd_data_1/tick_data",
        "/home/tradebot/tick_data",
    ]
    # ... (rest of the function is unchanged)
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
            .then(pl.col("volume_real")),
            .otherwise(pl.col("volume"))
            .alias("size"),
        ])
        .drop_nulls(["time", "price", "size"])
    )
    return d.select(["time", "bid", "ask", "price", "size"])


def run_wfo_for_symbol(symbol: str, xgb_params: dict, num_boost_round: int, args: argparse.Namespace, cost_engine: CostEngine):
    """
    Runs WFO logic for a symbol, now using the CostEngine.
    """
    ticks = load_ticks(symbol)
    if ticks is None:
        return None, 0

    # Infer spread from data, but commission will come from CostEngine
    spread_bps = infer_spread_bps(ticks)
    total_cost_bps = cost_engine.get_total_cost_bps(symbol, spread_bps)
    
    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat = build_bar_features(bars, z_threshold=Z_THRESHOLD)

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
        # Keep the costs separate for clarity
        pl.lit(total_cost_bps).alias("total_cost_bps"),
    ]).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    x = feat.select(FEATURE_COLUMNS).to_numpy()
    y = feat["target"].to_numpy().astype(np.float32)

    splits = purged_walk_forward_splits(
        n_samples=len(feat), train_size=args.train_size, test_size=args.test_size,
        purge=args.purge, embargo=args.embargo,
    )
    if not splits:
        return None, 0

    oos_proba = np.full(len(feat), np.nan)
    for tr_idx, te_idx in splits:
        dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
        dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

        bst = xgb.train(
            params=xgb_params, dtrain=dtrain, num_boost_round=num_boost_round,
            evals=[(dtest, "valid")], early_stopping_rounds=50, verbose_eval=False,
        )
        oos_proba[te_idx] = bst.predict(dtest)

    bar_data = feat.with_columns([
        pl.Series("proba", oos_proba),
    ]).filter(pl.col("proba").is_not_nan())

    return bar_data, len(splits)

def main():
    args = parse_args()
    
    # Initialize the new CostEngine
    info_dir = Path(SCRIPT_DIR).parent / "information"
    cost_engine = CostEngine(info_dir)

    csv_path = str(Path(SCRIPT_DIR) / args.csv) if not os.path.isabs(args.csv) else args.csv
    df = pl.read_csv(csv_path)
    ok_symbols = df.filter(pl.col("status") == "ok")

    configs = []
    for row in ok_symbols.iter_rows(named=True):
        params = {
            "booster": "gbtree", "tree_method": "hist", "device": "cuda",
            "objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": int(row["best_max_depth"]), "eta": 0.03,
            "gamma": float(row["best_gamma"]), "subsample": float(row["best_subsample"]),
            "colsample_bytree": float(row["best_colsample_bytree"]),
            "reg_alpha": float(row["best_reg_alpha"]), "reg_lambda": float(row["best_reg_lambda"]),
            "min_child_weight": float(row["best_min_child_weight"]),
            "verbosity": 0,
        }
        configs.append({ "symbol": row["symbol"], "params": params, "rounds": int(row["best_num_boost_round"]) })

    print(f"Loaded {len(configs)} elite symbols for ML-only backtest.")
    print(f"ML Threshold: {ML_THRESHOLD}, Z-Threshold: {Z_THRESHOLD}")
    print("-" * 50)

    all_trades = []
    total_oos_bars = 0
    
    for cfg in configs:
        sym = cfg["symbol"]
        t0 = time.time()
        print(f"Processing {sym}...")
        
        oos_data, num_folds = run_wfo_for_symbol(sym, cfg["params"], cfg["rounds"], args, cost_engine)
        
        if oos_data is None or oos_data.height == 0:
            print(f"  Skipping {sym} (no OOS data).")
            continue
            
        total_oos_bars += oos_data.height
        print(f"  {sym}: {num_folds} folds, {oos_data.height} OOS bars ({time.time() - t0:.1f}s)")
        
        signals = oos_data.filter(pl.col("proba") >= ML_THRESHOLD)
        
        # Calculate P/L, now subtracting the realistic costs
        trade_returns = signals.with_columns([
            (pl.col("tb_ret") - (pl.col("total_cost_bps") / 10000)).alias("net_ret"),
        ]).with_columns([
            (pl.col("net_ret") * RISK_PER_TRADE).alias("pnl_pct")
        ])
        
        all_trades.append(trade_returns)

    if not all_trades:
        print("No trades were generated in the backtest. Exiting.")
        return

    all_trades_df = pl.concat(all_trades)
    
    num_trades = all_trades_df.height
    wins = all_trades_df.filter(pl.col("pnl_pct") > 0).height
    win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0
    
    total_pnl_pct = all_trades_df.select(pl.col("pnl_pct").sum()).item()
    
    total_duration_hours = total_oos_bars
    total_duration_weeks = total_duration_hours / (24 * 7)
    
    avg_weekly_pnl_pct = (total_pnl_pct / total_duration_weeks) if total_duration_weeks > 0 else 0
    
    print("\n" + "=" * 50)
    print("  ML-ONLY BACKTEST RESULTS (with Realistic Costs)")
    print("=" * 50)
    print(f"  Total Trades Generated: {num_trades}")
    print(f"  Win Rate:               {win_rate:.2f}%")
    print(f"  Total Net P/L (%):      {total_pnl_pct:+.4f}%")
    print("-" * 50)
    print(f"  Total OOS Bars (Hours): {total_oos_bars}")
    print(f"  Total Simulated Weeks:  {total_duration_weeks:.1f}")
    print(f"  Avg. Weekly P/L (%):    {avg_weekly_pnl_pct:+.4f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()