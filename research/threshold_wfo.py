"""
Per-symbol ML threshold optimization via WFO grid sweep.

Tests thresholds from 0.50 to 0.75 for each symbol using its optimized
exit params and realistic FTMO costs. Finds the Calmar-optimal threshold.

Usage:
    python3 research/threshold_wfo.py --active --workers 6
    python3 research/threshold_wfo.py --symbols NVDA,AMZN --workers 4
    python3 research/threshold_wfo.py --active --workers 6 --update-configs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from trading_prop.ml.integrated_pipeline import (
    purged_walk_forward_splits,
    set_polars_threads,
)
from trading_prop.ml.train_ml_strategy import (
    infer_spread_bps,
    make_time_bars,
)
from trading_prop.production.optuna_orchestrator import (
    DATA_ROOTS,
    cluster_for_symbol,
    load_symbol_ticks_lf,
)
from research.exit_simulator import ExitParams, simulate_trades

# ---------------------------------------------------------------------------
CONFIG_PATH = REPO_ROOT / "config" / "sovereign_configs.json"
MODEL_DIR = REPO_ROOT / "models" / "sovereign_models"

TF_WF_SIZES = {
    "M15": {"train_size": 4800, "test_size": 1200},
    "M30": {"train_size": 3000, "test_size": 800},
    "H1":  {"train_size": 1000, "test_size": 300},
    "H4":  {"train_size": 250,  "test_size": 80},
}

# FTMO costs (from trading_prop/information/)
FTMO_COMMISSION_BPS = {
    "crypto": 6.5, "equity": 0.4, "forex": 0.5,
    "index": 0.0, "commodity": 0.0, "metals": 0.14,
}
FTMO_SLIPPAGE_BPS = {
    "crypto": 8.0, "equity": 2.5, "forex": 1.5,
    "index": 1.5, "commodity": 3.0, "metals": 2.0,
}

# Threshold grid
THRESHOLDS = np.round(np.arange(0.50, 0.76, 0.01), 2)


# ---------------------------------------------------------------------------
def compute_atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(13, n):
        atr[i] = np.mean(tr[i - 13 : i + 1])
    return atr


def ftmo_cost_pct(symbol: str, cluster: str, spread_bps: float) -> float:
    if cluster == "crypto":
        fee = 6.5
    elif cluster == "equity":
        fee = 0.4
    elif cluster == "forex":
        fee = 0.5
    elif cluster == "commodity":
        if any(x in symbol for x in ("XAG", "XAU", "XPT", "XPD", "XCU")):
            fee = 0.14
        else:
            fee = 0.0
    elif cluster == "index":
        fee = 0.0
    else:
        fee = 3.0
    slip = FTMO_SLIPPAGE_BPS.get(cluster, 3.0)
    if cluster == "commodity" and any(x in symbol for x in ("XAG", "XAU", "XCU")):
        slip = 2.0
    return (fee + spread_bps + slip * 2.0) / 1e4


def find_model_path(symbol: str) -> Path | None:
    for p in [MODEL_DIR / f"{symbol}.json",
              MODEL_DIR / f"{symbol.replace('_', '')}.json"]:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
def sweep_symbol(symbol: str, sym_cfg: dict, args_dict: dict) -> list[dict]:
    """Sweep thresholds for one symbol. Returns list of result dicts (one per threshold)."""
    import xgboost as xgb
    set_polars_threads(2)

    cluster = cluster_for_symbol(symbol)
    timeframe = sym_cfg.get("exit_timeframe", sym_cfg.get("atr_timeframe", "H1"))
    roots = list(args_dict["data_roots"])
    account_size = args_dict["account_size"]

    # 1. Load data (ONCE)
    ticks_lf = load_symbol_ticks_lf(symbol, roots)
    if ticks_lf is None:
        return [{"symbol": symbol, "threshold": 0.0, "status": "no_data"}]

    ticks_df = ticks_lf.collect()
    if ticks_df.height == 0:
        return [{"symbol": symbol, "threshold": 0.0, "status": "no_data"}]
    if ticks_df.select(pl.col("size").sum()).item() <= 0:
        ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks_df.select(["time", "price", "size"]), timeframe)
    feat = build_bar_features(bars, z_threshold=args_dict.get("z_threshold", 1.0))
    if feat.height < 500:
        return [{"symbol": symbol, "threshold": 0.0, "status": "insufficient_rows",
                 "rows": feat.height}]

    feat_clean = feat.drop_nulls(FEATURE_COLUMNS)

    # 2. Primary side entries (z-score)
    primary_side = feat_clean["primary_side"].to_numpy().astype(np.int32)
    signal_mask = primary_side != 0
    all_entry_indices = np.where(signal_mask)[0]
    all_directions = primary_side[signal_mask]

    # 3. Model probas (compute ONCE)
    model_path = find_model_path(symbol)
    if model_path is None:
        return [{"symbol": symbol, "threshold": 0.0, "status": "no_model"}]

    bst = xgb.Booster()
    bst.load_model(str(model_path))
    x = feat_clean.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    dmat = xgb.DMatrix(x)
    probas = bst.predict(dmat)

    if probas.std() < 0.01:
        return [{"symbol": symbol, "threshold": 0.0, "status": "flat_probas"}]

    # Probas at signal points only
    signal_probas = probas[signal_mask]

    # 4. OHLCV + ATR (compute ONCE)
    open_arr = feat_clean["open"].to_numpy().astype(np.float64)
    high_arr = feat_clean["high"].to_numpy().astype(np.float64)
    low_arr = feat_clean["low"].to_numpy().astype(np.float64)
    close_arr = feat_clean["close"].to_numpy().astype(np.float64)
    atr_arr = compute_atr14(high_arr, low_arr, close_arr)

    # 5. Costs
    spread_bps = infer_spread_bps(ticks_df)
    cost_pct = ftmo_cost_pct(symbol, cluster, spread_bps)

    # 6. Exit params from sovereign_configs
    ep = ExitParams(
        atr_sl_mult=sym_cfg["atr_sl_mult"],
        atr_tp_mult=sym_cfg["atr_tp_mult"],
        breakeven_atr=sym_cfg["breakeven_atr"],
        trail_activation_atr=sym_cfg["trail_activation_atr"],
        trail_distance_atr=sym_cfg["trail_distance_atr"],
        horizon=int(sym_cfg.get("exit_horizon", 24)),
    )
    risk_pct = sym_cfg.get("risk_per_trade", 0.02)

    # 7. WF splits (compute ONCE)
    wf = TF_WF_SIZES.get(timeframe, TF_WF_SIZES["H1"])
    n_samples = len(feat_clean)
    splits = purged_walk_forward_splits(
        n_samples=n_samples,
        train_size=wf["train_size"],
        test_size=wf["test_size"],
        purge=8, embargo=8,
    )
    if not splits:
        return [{"symbol": symbol, "threshold": 0.0, "status": "no_folds",
                 "rows": n_samples}]

    # 8. Sweep thresholds
    results = []
    tf_bars_per_year = {"M15": 35040, "M30": 17520, "H1": 8760, "H4": 2190}
    bpy = tf_bars_per_year.get(timeframe, 8760)

    for thr in THRESHOLDS:
        # Filter entries by threshold (BUY: proba >= thr, SELL: proba <= 1-thr)
        thr_mask = (signal_probas >= thr) | (signal_probas <= (1.0 - thr))
        entry_indices = all_entry_indices[thr_mask]
        directions = all_directions[thr_mask]

        if len(entry_indices) < 10:
            results.append({
                "symbol": symbol, "threshold": float(thr),
                "status": "too_few_signals",
                "n_signals": int(len(entry_indices)),
            })
            continue

        # Run on OOS folds
        all_pnl = []
        all_bars = []
        all_entry_bars = []

        for _, te_idx in splits:
            fold_mask = np.isin(entry_indices, te_idx)
            fold_entries = entry_indices[fold_mask]
            fold_dirs = directions[fold_mask]
            if len(fold_entries) < 2:
                continue
            pnl, bars_h, _ = simulate_trades(
                fold_entries, fold_dirs,
                open_arr, high_arr, low_arr, close_arr, atr_arr,
                ep, cost_pct,
            )
            if len(pnl) == 0:
                continue
            all_pnl.append(pnl)
            all_bars.append(bars_h)
            all_entry_bars.append(fold_entries[:len(pnl)])

        if not all_pnl:
            results.append({
                "symbol": symbol, "threshold": float(thr),
                "status": "no_trades", "n_signals": int(len(entry_indices)),
            })
            continue

        pnl = np.concatenate(all_pnl)
        bars_held = np.concatenate(all_bars)
        entry_bar_indices = np.concatenate(all_entry_bars)
        n_trades = len(pnl)

        # Position sizing → equity curve
        sl_fracs = atr_arr[entry_bar_indices] * ep.atr_sl_mult / close_arr[entry_bar_indices]
        sl_fracs = np.clip(sl_fracs, 1e-6, None)
        risk_amount = account_size * risk_pct
        dollar_pnl = risk_amount * pnl / sl_fracs[:n_trades]

        equity = np.cumsum(dollar_pnl)
        peak = np.maximum.accumulate(np.concatenate([[0], equity]))
        drawdown = peak[1:] - equity
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        total_return = float(equity[-1]) if len(equity) > 0 else 0.0

        wins = int(np.sum(pnl > 0))
        win_rate = wins / n_trades if n_trades > 0 else 0.0
        gross_profit = float(np.sum(dollar_pnl[dollar_pnl > 0]))
        gross_loss = float(np.abs(np.sum(dollar_pnl[dollar_pnl < 0])))
        profit_factor = gross_profit / max(gross_loss, 1.0)
        expectancy = float(np.mean(pnl))

        avg_bars_held = float(np.mean(bars_held))
        trades_per_year = bpy / max(avg_bars_held, 1.0)
        if np.std(dollar_pnl) > 0:
            sharpe = (np.mean(dollar_pnl) / np.std(dollar_pnl)) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        calmar = total_return / max(max_dd, 1.0)
        dd_pct = max_dd / account_size * 100

        results.append({
            "symbol": symbol,
            "threshold": float(thr),
            "status": "ok",
            "cluster": cluster,
            "timeframe": timeframe,
            "n_trades": n_trades,
            "total_return": round(total_return, 2),
            "max_dd": round(max_dd, 2),
            "dd_pct": round(dd_pct, 2),
            "sharpe": round(float(sharpe), 2),
            "calmar": round(calmar, 2),
            "profit_factor": round(profit_factor, 2),
            "win_rate": round(win_rate * 100, 1),
            "expectancy_pct": round(expectancy * 100, 4),
            "avg_bars_held": round(avg_bars_held, 1),
            "n_folds": len(splits),
        })

    return results


def _worker(task: tuple) -> list[dict]:
    symbol, sym_cfg, args_dict = task
    return sweep_symbol(symbol, sym_cfg, args_dict)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Per-symbol ML threshold WFO sweep")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--active", action="store_true",
                     help="Use symbols from sovereign_configs.json")
    grp.add_argument("--symbols", type=str, default="",
                     help="Comma-separated symbol list")
    p.add_argument("--account-size", type=float, default=100_000.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--update-configs", action="store_true",
                   help="Write best threshold per symbol to sovereign_configs.json")
    p.add_argument("--out-dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    with open(CONFIG_PATH) as f:
        configs = json.load(f)

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    else:
        symbols = sorted(configs.keys())

    # Filter to symbols that have a model
    symbols = [s for s in symbols if s in configs and find_model_path(s) is not None]

    if not symbols:
        raise SystemExit("No symbols with models found")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"models/optuna_results/threshold_wfo_{ts}"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    args_dict = {
        "data_roots": DATA_ROOTS,
        "account_size": args.account_size,
        "z_threshold": 1.0,
    }

    print(f"[threshold-wfo] {len(symbols)} symbols × {len(THRESHOLDS)} thresholds "
          f"= {len(symbols) * len(THRESHOLDS)} combos | {args.workers} workers")
    print(f"  Thresholds: {THRESHOLDS[0]:.2f} → {THRESHOLDS[-1]:.2f} (step 0.01)")
    print(f"  Output: {out_dir}")
    print()

    tasks = [(sym, configs[sym], args_dict) for sym in symbols]
    all_results: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                rows = future.result()
                ok_rows = [r for r in rows if r.get("status") == "ok"]
                if ok_rows:
                    best = max(ok_rows, key=lambda r: r.get("calmar", -999))
                    print(
                        f"  [{done}/{len(tasks)}] {sym:15s}  "
                        f"best_thr={best['threshold']:.2f}  "
                        f"Calmar={best['calmar']:7.1f}  "
                        f"Sharpe={best['sharpe']:6.2f}  "
                        f"PF={best['profit_factor']:5.2f}  "
                        f"WR={best['win_rate']:4.1f}%  "
                        f"trades={best['n_trades']:4d}  "
                        f"${best['total_return']:>9,.0f}  "
                        f"DD={best['dd_pct']:5.1f}%"
                    )
                else:
                    status = rows[0].get("status", "?") if rows else "empty"
                    print(f"  [{done}/{len(tasks)}] {sym:15s}  {status}")
                all_results.extend(rows)
            except Exception as e:
                print(f"  [{done}/{len(tasks)}] {sym:15s}  ERROR: {e}")
                all_results.append({"symbol": sym, "threshold": 0.0,
                                    "status": "error", "error": str(e)})

    # Save all results
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    if ok_results:
        df = pl.from_dicts(ok_results).sort(["symbol", "threshold"])
        df.write_csv(os.path.join(out_dir, "threshold_sweep.csv"))

    # Find best threshold per symbol
    best_per_sym: dict[str, dict] = {}
    for r in ok_results:
        sym = r["symbol"]
        if sym not in best_per_sym or r["calmar"] > best_per_sym[sym]["calmar"]:
            best_per_sym[sym] = r

    # Also find result at default 0.55 for comparison
    default_per_sym: dict[str, dict] = {}
    for r in ok_results:
        if abs(r["threshold"] - 0.55) < 0.001:
            default_per_sym[r["symbol"]] = r

    # Print comparison table
    print("\n" + "=" * 130)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 130)
    print(f"{'Symbol':15s} {'Best':>5s} {'Calmar':>8s} {'Sharpe':>7s} {'PF':>6s} "
          f"{'WR%':>5s} {'Trades':>6s} {'Return':>10s}  │  "
          f"{'@0.55':>8s} {'Sharpe':>7s} {'PF':>6s} {'Trades':>6s}  │  {'Delta':>7s}")
    print("-" * 130)

    improved = 0
    for sym in sorted(best_per_sym):
        b = best_per_sym[sym]
        d = default_per_sym.get(sym)

        d_cal = f"{d['calmar']:7.1f}" if d else "  N/A  "
        d_sha = f"{d['sharpe']:6.2f}" if d else "  N/A "
        d_pf  = f"{d['profit_factor']:5.2f}" if d else " N/A "
        d_tr  = f"{d['n_trades']:5d}" if d else "  N/A"

        if d and d["calmar"] > 0:
            delta = (b["calmar"] - d["calmar"]) / d["calmar"] * 100
            delta_str = f"{delta:+6.1f}%"
            if delta > 0:
                improved += 1
        else:
            delta_str = "   N/A"

        print(
            f"{sym:15s} {b['threshold']:5.2f} {b['calmar']:7.1f} "
            f"{b['sharpe']:6.2f} {b['profit_factor']:5.2f} "
            f"{b['win_rate']:4.1f}% {b['n_trades']:5d} "
            f"${b['total_return']:>9,.0f}  │  "
            f"{d_cal} {d_sha} {d_pf} {d_tr}  │  {delta_str}"
        )

    print("-" * 130)
    print(f"  {improved}/{len(best_per_sym)} symbols improved vs default 0.55")

    # Save best per symbol
    if best_per_sym:
        best_df = pl.from_dicts(sorted(best_per_sym.values(),
                                       key=lambda x: x["symbol"]))
        best_df.write_csv(os.path.join(out_dir, "best_thresholds.csv"))

    # Update configs
    if args.update_configs and best_per_sym:
        with open(CONFIG_PATH) as f:
            live_configs = json.load(f)

        updated = 0
        for sym, r in best_per_sym.items():
            if sym in live_configs:
                live_configs[sym]["prob_threshold"] = r["threshold"]
                updated += 1

        with open(CONFIG_PATH, "w") as f:
            json.dump(live_configs, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\n[threshold-wfo] Updated {updated} symbols with prob_threshold in {CONFIG_PATH}")

    print(f"\n[threshold-wfo] Done — {len(ok_results)} results, "
          f"{len(best_per_sym)} symbols optimized")
    print(f"  Saved: {out_dir}/threshold_sweep.csv")
    print(f"  Saved: {out_dir}/best_thresholds.csv")


if __name__ == "__main__":
    main()
