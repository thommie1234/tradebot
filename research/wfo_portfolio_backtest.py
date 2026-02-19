"""
Walk-Forward Portfolio Backtest with realistic FTMO costs.

Runs OOS equity simulation per symbol using optimized entry + exit params,
then constructs a diversified 24-hour portfolio.

Usage:
    python3 research/wfo_portfolio_backtest.py --results-dir models/optuna_results/exit_sizing_combined
    python3 research/wfo_portfolio_backtest.py --symbols TSLA,AAPL --trials-csv exit_best_per_symbol.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
    broker_commission_bps,
    broker_slippage_bps,
    calmar_weighted_score,
    cluster_for_symbol,
    load_symbol_ticks_lf,
)
from research.exit_simulator import ExitParams, simulate_trades

# ---------------------------------------------------------------------------
# Leak-free exit optimization constants
# ---------------------------------------------------------------------------

CONSERVATIVE_DEFAULTS = {
    "equity": ExitParams(1.5, 4.5, 1.0, 2.0, 0.5, 24),
    "crypto": ExitParams(1.5, 6.0, 1.0, 3.0, 1.0, 30),
    "index":  ExitParams(1.5, 6.0, 0.8, 2.5, 1.0, 30),
    "forex":  ExitParams(1.5, 6.0, 0.8, 2.5, 1.0, 30),
    "commodity": ExitParams(1.5, 6.0, 0.8, 2.5, 1.0, 30),
}

EXIT_SEARCH_SPACES = {
    "equity": {
        "atr_sl_mult": (0.5, 2.5), "atr_tp_mult": (2.0, 8.0),
        "breakeven_atr": (0.3, 1.5), "trail_activation_atr": (1.0, 4.0),
        "trail_distance_atr": (0.3, 2.0), "horizon": (6, 36),
    },
    "crypto": {
        "atr_sl_mult": (0.8, 3.0), "atr_tp_mult": (3.0, 10.0),
        "breakeven_atr": (0.3, 2.0), "trail_activation_atr": (1.5, 5.0),
        "trail_distance_atr": (0.5, 2.5), "horizon": (6, 48),
    },
    "index": {
        "atr_sl_mult": (0.5, 2.5), "atr_tp_mult": (2.0, 8.0),
        "breakeven_atr": (0.3, 1.5), "trail_activation_atr": (1.0, 4.0),
        "trail_distance_atr": (0.3, 2.0), "horizon": (6, 36),
    },
}
EXIT_SEARCH_SPACES["forex"] = EXIT_SEARCH_SPACES["index"]
EXIT_SEARCH_SPACES["commodity"] = EXIT_SEARCH_SPACES["index"]

# ---------------------------------------------------------------------------
# FTMO realistic costs (from trading_prop/information/ CSVs)
# ---------------------------------------------------------------------------

FTMO_COMMISSION_BPS = {
    "crypto":    6.5,    # 0.065% per volume
    "equity":    0.4,    # 0.004% per volume
    "forex":     0.5,    # $5/lot on 100k
    "index":     0.0,    # NO_COMMISSION
    "commodity": 0.0,    # NO_COMMISSION (oils, agri)
    "metals":    0.14,   # 0.0014% per volume
}

FTMO_SLIPPAGE_BPS = {
    "crypto":    8.0,
    "equity":    2.5,
    "forex":     1.5,
    "index":     1.5,
    "commodity": 3.0,
    "metals":    2.0,
}

# Trading hours GMT+2 per asset class (for session coverage analysis)
TRADING_SESSIONS = {
    "crypto":    {"open": 0, "close": 24, "days": "Mon-Sun"},
    "forex":     {"open": 0, "close": 24, "days": "Mon-Fri"},
    "metals":    {"open": 1, "close": 24, "days": "Mon-Fri"},
    # US equities
    "us_equity": {"open": 16, "close": 23, "days": "Mon-Fri"},
    # EU equities
    "eu_equity": {"open": 10, "close": 18, "days": "Mon-Fri"},
    # US indices
    "us_idx":    {"open": 1, "close": 24, "days": "Mon-Fri"},
    # EU indices
    "eu_idx":    {"open": 9, "close": 23, "days": "Mon-Fri"},
    # APAC indices
    "apac_idx":  {"open": 1, "close": 23, "days": "Mon-Fri"},
    # Commodities
    "oil":       {"open": 1, "close": 24, "days": "Mon-Fri"},
    "agri":      {"open": 12, "close": 20, "days": "Mon-Fri"},
}

# Symbol → session mapping
SYMBOL_SESSION = {
    # Crypto (24/7)
    "AAVUSD": "crypto", "ALGUSD": "crypto", "AVAUSD": "crypto",
    "BARUSD": "crypto", "BNBUSD": "crypto", "DASHUSD": "crypto",
    "ETCUSD": "crypto", "FETUSD": "crypto", "GALUSD": "crypto",
    "GRTUSD": "crypto", "ICPUSD": "crypto", "IMXUSD": "crypto",
    "LNKUSD": "crypto", "MANUSD": "crypto", "NERUSD": "crypto",
    "SANUSD": "crypto", "SOLUSD": "crypto", "UNIUSD": "crypto",
    "VECUSD": "crypto", "XLMUSD": "crypto",
    # Forex (nearly 24h Mon-Fri)
    "AUD_CHF": "forex", "AUD_JPY": "forex", "AUD_USD": "forex",
    "CAD_JPY": "forex", "EUR_JPY": "forex", "EUR_USD": "forex",
    "GBP_JPY": "forex", "GBP_USD": "forex", "NZD_JPY": "forex",
    "NZD_USD": "forex", "USD_CHF": "forex", "USD_JPY": "forex",
    # Metals
    "XAG_AUD": "metals", "XAG_EUR": "metals", "XAG_USD": "metals",
    "XAU_AUD": "metals", "XAU_EUR": "metals", "XAU_USD": "metals",
    "XCU_USD": "metals",
    # US equities
    "AAPL": "us_equity", "AMZN": "us_equity", "BAC": "us_equity",
    "GOOG": "us_equity", "META": "us_equity", "MSFT": "us_equity",
    "NVDA": "us_equity", "PFE": "us_equity", "TSLA": "us_equity",
    "V": "us_equity",
    # EU equities
    "ALVG": "eu_equity", "BAYGn": "eu_equity", "DBKGn": "eu_equity",
    "LVMH": "eu_equity", "RACE": "eu_equity",
    # BABA trades US hours
    "BABA": "us_equity",
    # US indices
    "US100.cash": "us_idx", "US30.cash": "us_idx",
    "US500.cash": "us_idx", "US2000.cash": "us_idx",
    # EU indices
    "EU50.cash": "eu_idx", "FRA40.cash": "eu_idx",
    "GER40.cash": "eu_idx", "SPN35.cash": "eu_idx",
    "UK100.cash": "eu_idx",
    # APAC indices
    "AUS200.cash": "apac_idx", "HK50.cash": "apac_idx",
    "JP225.cash": "apac_idx",
    # Commodities
    "UKOIL.cash": "oil", "USOIL.cash": "oil",
    "COCOA.c": "agri", "COFFEE.c": "agri",
}

# Correlation groups (max same-direction positions per group)
CORRELATION_GROUPS = {
    "us_tech":   ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"],
    "eu_equity": ["ALVG", "BAYGn", "DBKGn", "LVMH", "RACE"],
    "us_idx":    ["US100.cash", "US30.cash", "US500.cash", "US2000.cash"],
    "eu_idx":    ["EU50.cash", "FRA40.cash", "GER40.cash", "SPN35.cash", "UK100.cash"],
    "apac_idx":  ["AUS200.cash", "HK50.cash", "JP225.cash"],
    "jpy_fx":    ["AUD_JPY", "CAD_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "USD_JPY"],
    "usd_fx":    ["AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD"],
    "gold":      ["XAU_AUD", "XAU_EUR", "XAU_USD"],
    "silver":    ["XAG_AUD", "XAG_EUR", "XAG_USD"],
    "oil":       ["UKOIL.cash", "USOIL.cash"],
    "defi":      ["AAVUSD", "UNIUSD", "LNKUSD"],
    "alt_crypto":["ALGUSD", "DASHUSD", "ETCUSD", "FETUSD", "GALUSD",
                  "GRTUSD", "ICPUSD", "IMXUSD", "MANUSD", "NERUSD",
                  "SOLUSD", "VECUSD", "XLMUSD", "BARUSD", "SANUSD",
                  "AVAUSD", "BNBUSD"],
}

# WF sizes per TF
TF_WF_SIZES = {
    "M15": {"train_size": 4800, "test_size": 1200},
    "M30": {"train_size": 3000, "test_size": 800},
    "H1":  {"train_size": 1000, "test_size": 300},
    "H4":  {"train_size": 250,  "test_size": 80},
}

MODEL_DIR = REPO_ROOT / "models" / "sovereign_models"


# ---------------------------------------------------------------------------
# Helpers
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
    """Total round-trip cost in fraction (not bps)."""
    # Commission from FTMO specs
    if cluster in ("crypto",):
        fee = 6.5
    elif cluster in ("equity",):
        fee = 0.4
    elif cluster in ("forex",):
        fee = 0.5
    elif cluster in ("commodity",):
        # metals vs cash
        if any(x in symbol for x in ("XAG", "XAU", "XPT", "XPD", "XCU")):
            fee = 0.14
        else:
            fee = 0.0
    elif cluster in ("index",):
        fee = 0.0
    else:
        fee = 3.0

    # Slippage
    slip = FTMO_SLIPPAGE_BPS.get(cluster, 3.0)
    if cluster == "commodity" and any(x in symbol for x in ("XAG", "XAU", "XCU")):
        slip = 2.0

    return (fee + spread_bps + slip * 2.0) / 1e4


def find_model_path(symbol: str) -> Path | None:
    candidates = [MODEL_DIR / f"{symbol}.json", MODEL_DIR / f"{symbol.replace('_', '')}.json"]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Leak-free per-fold exit optimization
# ---------------------------------------------------------------------------

def optimize_exits_on_past_data(
    past_entries: np.ndarray,
    past_dirs: np.ndarray,
    past_fold_ids: np.ndarray,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    cluster: str,
    cost_pct: float,
    n_trials: int = 100,
) -> ExitParams:
    """Run mini-Optuna on past OOS fold data to find exit params for next fold.

    Parameters
    ----------
    past_entries : entry bar indices from previous OOS folds
    past_dirs : trade directions for those entries
    past_fold_ids : which fold each entry came from (for per-fold scoring)
    open_arr..atr_arr : full OHLC+ATR arrays
    cluster : asset class for search space selection
    cost_pct : round-trip transaction cost as fraction
    n_trials : Optuna trials to run

    Returns
    -------
    ExitParams optimized on past fold data only
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = EXIT_SEARCH_SPACES.get(cluster, EXIT_SEARCH_SPACES["index"])
    unique_folds = np.unique(past_fold_ids)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial) -> float:
        sl = trial.suggest_float("atr_sl_mult", *space["atr_sl_mult"])
        tp = trial.suggest_float("atr_tp_mult", *space["atr_tp_mult"])
        be = trial.suggest_float("breakeven_atr", *space["breakeven_atr"])
        ta = trial.suggest_float("trail_activation_atr", *space["trail_activation_atr"])
        td = trial.suggest_float("trail_distance_atr", *space["trail_distance_atr"])
        hz = trial.suggest_int("horizon", *space["horizon"])

        if ta <= be or tp <= sl:
            raise optuna.TrialPruned()

        ep = ExitParams(
            atr_sl_mult=sl, atr_tp_mult=tp,
            breakeven_atr=be, trail_activation_atr=ta,
            trail_distance_atr=td, horizon=hz,
        )

        ev_scores: list[float] = []
        for fid in unique_folds:
            fold_mask = past_fold_ids == fid
            fold_entries = past_entries[fold_mask]
            fold_dirs = past_dirs[fold_mask]
            if len(fold_entries) < 3:
                continue

            pnl, _, _ = simulate_trades(
                fold_entries, fold_dirs,
                open_arr, high_arr, low_arr, close_arr, atr_arr,
                ep, cost_pct,
            )
            if len(pnl) == 0:
                continue
            ev_scores.append(float(np.mean(pnl)))

        if not ev_scores:
            return -999.0
        return calmar_weighted_score(ev_scores)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    if study.best_trial is None:
        return CONSERVATIVE_DEFAULTS.get(cluster, CONSERVATIVE_DEFAULTS["index"])

    bp = study.best_trial.params
    return ExitParams(
        atr_sl_mult=bp["atr_sl_mult"],
        atr_tp_mult=bp["atr_tp_mult"],
        breakeven_atr=bp["breakeven_atr"],
        trail_activation_atr=bp["trail_activation_atr"],
        trail_distance_atr=bp["trail_distance_atr"],
        horizon=int(bp["horizon"]),
    )


# ---------------------------------------------------------------------------
# Per-symbol WFO backtest
# ---------------------------------------------------------------------------

def backtest_symbol(symbol: str, exit_row: dict, args_dict: dict) -> dict:
    """Run full WFO backtest for one symbol with its optimized exit params."""
    import xgboost as xgb
    set_polars_threads(2)

    leak_free = args_dict.get("leak_free", False)
    leak_free_trials = args_dict.get("leak_free_trials", 100)
    warmup_folds = args_dict.get("warmup_folds", 3)
    cluster = cluster_for_symbol(symbol)
    timeframe = exit_row["timeframe"]
    roots = list(args_dict["data_roots"])
    account_size = args_dict["account_size"]

    # 1. Load data
    ticks_lf = load_symbol_ticks_lf(symbol, roots)
    if ticks_lf is None:
        return {"symbol": symbol, "status": "no_data", "cluster": cluster}

    ticks_df = ticks_lf.collect()
    if ticks_df.height == 0:
        return {"symbol": symbol, "status": "no_data", "cluster": cluster}
    if ticks_df.select(pl.col("size").sum()).item() <= 0:
        ticks_df = ticks_df.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks_df.select(["time", "price", "size"]), timeframe)
    feat = build_bar_features(bars, z_threshold=args_dict.get("z_threshold", 1.0))

    if feat.height < 500:
        return {"symbol": symbol, "status": "insufficient_rows", "rows": feat.height, "cluster": cluster}

    feat_clean = feat.drop_nulls(FEATURE_COLUMNS)

    # 2. Entry signals
    primary_side = feat_clean["primary_side"].to_numpy().astype(np.int32)
    signal_mask = primary_side != 0
    entry_indices = np.where(signal_mask)[0]
    directions = primary_side[signal_mask]

    # 3. ML model filter (if available)
    model_path = find_model_path(symbol)
    if model_path is not None:
        try:
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            x = feat_clean.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
            dmat = xgb.DMatrix(x)
            probas = bst.predict(dmat)
            if probas.std() > 0.01:
                threshold = args_dict.get("sym_thresholds", {}).get(
                    symbol, args_dict.get("prob_threshold", 0.55))
                model_ok = (probas >= threshold) | (probas <= (1.0 - threshold))
                model_filter = model_ok[signal_mask]
                entry_indices = entry_indices[model_filter]
                directions = directions[model_filter]
        except Exception:
            pass

    if len(entry_indices) < 10:
        return {"symbol": symbol, "status": "too_few_signals", "n_signals": len(entry_indices), "cluster": cluster}

    # 4. OHLCV + ATR
    open_arr = feat_clean["open"].to_numpy().astype(np.float64)
    high_arr = feat_clean["high"].to_numpy().astype(np.float64)
    low_arr = feat_clean["low"].to_numpy().astype(np.float64)
    close_arr = feat_clean["close"].to_numpy().astype(np.float64)
    atr_arr = compute_atr14(high_arr, low_arr, close_arr)

    # 5. Realistic FTMO costs
    spread_bps = infer_spread_bps(ticks_df)
    cost_pct = ftmo_cost_pct(symbol, cluster, spread_bps)

    # 6. Exit params (used in non-leak-free mode, or as reference)
    ep = ExitParams(
        atr_sl_mult=exit_row["best_atr_sl_mult"],
        atr_tp_mult=exit_row["best_atr_tp_mult"],
        breakeven_atr=exit_row["best_breakeven_atr"],
        trail_activation_atr=exit_row["best_trail_activation_atr"],
        trail_distance_atr=exit_row["best_trail_distance_atr"],
        horizon=int(exit_row["best_horizon"]),
    )

    risk_pct = (exit_row.get("best_risk_pct")
                or args_dict.get("sym_risk", {}).get(symbol)
                or 0.01)

    # 7. Walk-forward OOS backtest
    wf = TF_WF_SIZES.get(timeframe, TF_WF_SIZES["H1"])
    n_samples = len(feat_clean)
    splits = purged_walk_forward_splits(
        n_samples=n_samples,
        train_size=wf["train_size"],
        test_size=wf["test_size"],
        purge=8,
        embargo=8,
    )
    if not splits:
        return {"symbol": symbol, "status": "no_folds", "rows": n_samples, "cluster": cluster}

    # Collect ALL OOS trades across folds (no overlap between folds)
    all_pnl = []
    all_bars = []
    all_exits = []
    all_entry_bars = []  # for time-based analysis
    all_sl_mults = []    # per-trade SL mult (varies in leak-free mode)
    n_warmup_trades = 0

    if leak_free:
        # --- Leak-free rolling per-fold optimization ---
        # Pre-compute entries per fold for accumulation
        fold_entries_list = []
        fold_dirs_list = []
        for _, te_idx in splits:
            fm = np.isin(entry_indices, te_idx)
            fold_entries_list.append(entry_indices[fm])
            fold_dirs_list.append(directions[fm])

        for fold_idx, (_, te_idx) in enumerate(splits):
            fold_entries = fold_entries_list[fold_idx]
            fold_dirs = fold_dirs_list[fold_idx]

            if len(fold_entries) < 2:
                continue

            if fold_idx < warmup_folds:
                # Warmup: use conservative defaults
                fold_ep = CONSERVATIVE_DEFAULTS.get(cluster, CONSERVATIVE_DEFAULTS["index"])
            else:
                # Accumulate entries from all past OOS folds 0..fold_idx-1
                past_e = []
                past_d = []
                past_fids = []
                for prev_idx in range(fold_idx):
                    pe = fold_entries_list[prev_idx]
                    pd = fold_dirs_list[prev_idx]
                    if len(pe) > 0:
                        past_e.append(pe)
                        past_d.append(pd)
                        past_fids.append(np.full(len(pe), prev_idx, dtype=np.int32))

                if not past_e or len(past_e) < warmup_folds:
                    fold_ep = CONSERVATIVE_DEFAULTS.get(cluster, CONSERVATIVE_DEFAULTS["index"])
                else:
                    past_entries = np.concatenate(past_e)
                    past_dirs = np.concatenate(past_d)
                    past_fold_ids = np.concatenate(past_fids)
                    fold_ep = optimize_exits_on_past_data(
                        past_entries, past_dirs, past_fold_ids,
                        open_arr, high_arr, low_arr, close_arr, atr_arr,
                        cluster, cost_pct, n_trials=leak_free_trials,
                    )

            pnl, bars_h, exits = simulate_trades(
                fold_entries, fold_dirs,
                open_arr, high_arr, low_arr, close_arr, atr_arr,
                fold_ep, cost_pct,
            )
            if len(pnl) == 0:
                continue

            if fold_idx < warmup_folds:
                n_warmup_trades += len(pnl)

            all_pnl.append(pnl)
            all_bars.append(bars_h)
            all_exits.append(exits)
            all_entry_bars.append(fold_entries[:len(pnl)])
            all_sl_mults.append(np.full(len(pnl), fold_ep.atr_sl_mult))
    else:
        # --- Original mode: single exit params for all folds ---
        for fold_idx, (_, te_idx) in enumerate(splits):
            fold_mask = np.isin(entry_indices, te_idx)
            fold_entries = entry_indices[fold_mask]
            fold_dirs = directions[fold_mask]

            if len(fold_entries) < 2:
                continue

            pnl, bars_h, exits = simulate_trades(
                fold_entries, fold_dirs,
                open_arr, high_arr, low_arr, close_arr, atr_arr,
                ep, cost_pct,
            )
            if len(pnl) == 0:
                continue

            all_pnl.append(pnl)
            all_bars.append(bars_h)
            all_exits.append(exits)
            all_entry_bars.append(fold_entries[:len(pnl)])
            all_sl_mults.append(np.full(len(pnl), ep.atr_sl_mult))

    if not all_pnl:
        return {"symbol": symbol, "status": "no_trades", "cluster": cluster}

    pnl = np.concatenate(all_pnl)
    bars_held = np.concatenate(all_bars)
    exit_types = np.concatenate(all_exits)
    entry_bar_indices = np.concatenate(all_entry_bars)
    sl_mults = np.concatenate(all_sl_mults)

    # 8. Position sizing → dollar equity curve
    # Use per-trade sl_mult (varies per fold in leak-free mode)
    sl_fracs = atr_arr[entry_bar_indices] * sl_mults / close_arr[entry_bar_indices]
    sl_fracs = np.clip(sl_fracs, 1e-6, None)
    risk_amount = account_size * risk_pct
    dollar_pnl = risk_amount * pnl / sl_fracs[:len(pnl)]

    equity = np.cumsum(dollar_pnl)
    peak = np.maximum.accumulate(np.concatenate([[0], equity]))
    drawdown = peak[1:] - equity
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    total_return = float(equity[-1]) if len(equity) > 0 else 0.0

    # 9. Performance metrics
    n_trades = len(pnl)
    wins = np.sum(pnl > 0)
    losses = np.sum(pnl < 0)
    win_rate = float(wins / n_trades) if n_trades > 0 else 0.0

    gross_profit = float(np.sum(dollar_pnl[dollar_pnl > 0]))
    gross_loss = float(np.abs(np.sum(dollar_pnl[dollar_pnl < 0])))
    profit_factor = gross_profit / max(gross_loss, 1.0)

    avg_win = float(np.mean(pnl[pnl > 0])) if wins > 0 else 0.0
    avg_loss = float(np.mean(np.abs(pnl[pnl < 0]))) if losses > 0 else 0.0
    expectancy = float(np.mean(pnl))

    # Sharpe (annualize based on TF)
    tf_bars_per_year = {"M15": 35040, "M30": 17520, "H1": 8760, "H4": 2190}
    bpy = tf_bars_per_year.get(timeframe, 8760)
    avg_bars_held = float(np.mean(bars_held)) if len(bars_held) > 0 else 1.0
    trades_per_year = bpy / max(avg_bars_held, 1.0)
    if np.std(dollar_pnl) > 0:
        sharpe = (np.mean(dollar_pnl) / np.std(dollar_pnl)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Calmar
    calmar = total_return / max(max_dd, 1.0)

    # Max DD as % of account
    dd_pct = max_dd / account_size * 100

    # FTMO compliance
    ftmo_ok = dd_pct < 10.0  # total max loss

    # Exit type distribution
    n_sl = int(np.sum(exit_types == 0))
    n_tp = int(np.sum(exit_types == 1))
    n_be = int(np.sum(exit_types == 2))
    n_trail = int(np.sum(exit_types == 3))
    n_horizon = int(np.sum(exit_types == 4))

    # Trading frequency
    total_bars = n_samples
    n_folds = len(splits)

    session = SYMBOL_SESSION.get(symbol, "unknown")
    rrr = ep.atr_tp_mult / ep.atr_sl_mult

    result = {
        "symbol": symbol,
        "status": "ok",
        "cluster": cluster,
        "session": session,
        "timeframe": timeframe,
        # Exit params (from CSV — reference only in leak-free mode)
        "sl": round(ep.atr_sl_mult, 2),
        "tp": round(ep.atr_tp_mult, 2),
        "rrr": round(rrr, 1),
        "be": round(ep.breakeven_atr, 2),
        "trail_act": round(ep.trail_activation_atr, 2),
        "trail_dist": round(ep.trail_distance_atr, 2),
        "horizon": ep.horizon,
        "risk_pct": round(risk_pct * 100, 2),
        # Costs
        "cost_bps": round(cost_pct * 1e4, 1),
        "spread_bps": round(spread_bps, 1),
        # Performance
        "total_return": round(total_return, 2),
        "max_dd": round(max_dd, 2),
        "dd_pct": round(dd_pct, 2),
        "sharpe": round(float(sharpe), 2),
        "calmar": round(calmar, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_pct": round(expectancy * 100, 4),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": n_trades,
        "avg_bars_held": round(avg_bars_held, 1),
        "n_folds": n_folds,
        "avg_win_pct": round(avg_win * 100, 4),
        "avg_loss_pct": round(avg_loss * 100, 4),
        # Exit distribution
        "pct_sl": round(n_sl / n_trades * 100, 1) if n_trades > 0 else 0,
        "pct_tp": round(n_tp / n_trades * 100, 1) if n_trades > 0 else 0,
        "pct_be": round(n_be / n_trades * 100, 1) if n_trades > 0 else 0,
        "pct_trail": round(n_trail / n_trades * 100, 1) if n_trades > 0 else 0,
        "pct_horizon": round(n_horizon / n_trades * 100, 1) if n_trades > 0 else 0,
        # FTMO
        "ftmo_ok": ftmo_ok,
    }

    if leak_free:
        result["leak_free"] = True
        result["warmup_trades"] = n_warmup_trades
        result["optimized_trades"] = n_trades - n_warmup_trades

    return result


def _worker(task: tuple) -> dict:
    symbol, exit_row, args_dict = task
    return backtest_symbol(symbol, exit_row, args_dict)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(results: list[dict], max_corr_per_group: int = 2) -> dict:
    """Select diversified portfolio with 24h coverage."""
    ok = [r for r in results if r.get("status") == "ok" and r.get("total_return", 0) > 0
          and r.get("ftmo_ok", False)]

    if not ok:
        return {"portfolio": [], "sessions": {}}

    # Sort by Calmar (risk-adjusted)
    ok.sort(key=lambda x: x.get("calmar", 0), reverse=True)

    selected = []
    group_counts: dict[str, int] = {}

    for r in ok:
        sym = r["symbol"]

        # Check correlation group limits
        blocked = False
        for grp_name, grp_syms in CORRELATION_GROUPS.items():
            if sym in grp_syms:
                cnt = group_counts.get(grp_name, 0)
                if cnt >= max_corr_per_group:
                    blocked = True
                    break

        if blocked:
            continue

        selected.append(r)

        # Update group counts
        for grp_name, grp_syms in CORRELATION_GROUPS.items():
            if sym in grp_syms:
                group_counts[grp_name] = group_counts.get(grp_name, 0) + 1

    # Session coverage analysis
    sessions_covered: dict[str, list[str]] = {
        "asia_night (00-08)": [],
        "europe (08-16)": [],
        "us (16-23)": [],
        "weekend": [],
    }
    for r in selected:
        session = r.get("session", "unknown")
        sym = r["symbol"]
        sess_info = TRADING_SESSIONS.get(session, {})
        o = sess_info.get("open", 0)
        c = sess_info.get("close", 24)
        days = sess_info.get("days", "Mon-Fri")

        if o <= 8:
            sessions_covered["asia_night (00-08)"].append(sym)
        if (o <= 16 and c >= 8):
            sessions_covered["europe (08-16)"].append(sym)
        if c >= 16:
            sessions_covered["us (16-23)"].append(sym)
        if "Sun" in days or "Sat" in days:
            sessions_covered["weekend"].append(sym)

    return {"portfolio": selected, "sessions": sessions_covered}


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WFO Portfolio Backtest with FTMO costs")
    p.add_argument("--results-csv", type=str, required=True,
                   help="Path to exit_best_per_symbol.csv from exit Optuna")
    p.add_argument("--symbols", type=str, default="",
                   help="Comma-separated filter (default: all)")
    p.add_argument("--account-size", type=float, default=100_000.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-corr", type=int, default=2,
                   help="Max symbols per correlation group in portfolio")
    p.add_argument("--leak-free", action="store_true",
                   help="Rolling per-fold exit optimization (no leakage)")
    p.add_argument("--leak-free-trials", type=int, default=100,
                   help="Optuna trials per fold in leak-free mode")
    p.add_argument("--warmup-folds", type=int, default=3,
                   help="Folds to skip before optimizing (use defaults)")
    p.add_argument("--out-dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    # Load exit optimization results
    exit_df = pl.read_csv(args.results_csv)
    exit_rows = {row["symbol"]: row for row in exit_df.iter_rows(named=True)}

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
        exit_rows = {s: exit_rows[s] for s in symbols if s in exit_rows}
    else:
        symbols = sorted(exit_rows.keys())

    if not symbols:
        raise SystemExit("No symbols found")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "wfo_leak_free" if args.leak_free else "wfo_portfolio"
        out_dir = f"models/optuna_results/{prefix}_{ts}"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load per-symbol thresholds and risk from sovereign_configs
    config_path = REPO_ROOT / "config" / "sovereign_configs.json"
    sym_thresholds = {}
    sym_risk = {}
    if config_path.exists():
        with open(config_path) as f:
            _cfgs = json.load(f)
        for s, c in _cfgs.items():
            if "prob_threshold" in c:
                sym_thresholds[s] = c["prob_threshold"]
            if "risk_per_trade" in c:
                sym_risk[s] = c["risk_per_trade"]

    args_dict = {
        "data_roots": DATA_ROOTS,
        "account_size": args.account_size,
        "z_threshold": 1.0,
        "prob_threshold": 0.55,
        "sym_thresholds": sym_thresholds,
        "sym_risk": sym_risk,
        "leak_free": args.leak_free,
        "leak_free_trials": args.leak_free_trials,
        "warmup_folds": args.warmup_folds,
    }

    mode = "LEAK-FREE" if args.leak_free else "STANDARD"
    print(f"[WFO] {len(symbols)} symbols | account=${args.account_size:,.0f} | {args.workers} workers | {mode}")
    if args.leak_free:
        print(f"  Leak-free: {args.leak_free_trials} trials/fold, {args.warmup_folds} warmup folds (conservative defaults)")
    print(f"  Output: {out_dir}")
    print()

    # Run backtests
    tasks = [(sym, exit_rows[sym], args_dict) for sym in symbols if sym in exit_rows]
    results: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                r = future.result()
                if r.get("status") == "ok":
                    ftmo = "FTMO-OK" if r["ftmo_ok"] else "FTMO-FAIL"
                    lf_info = ""
                    if r.get("leak_free"):
                        lf_info = f"  warmup={r['warmup_trades']}/opt={r['optimized_trades']}"
                    print(
                        f"  [{done}/{len(tasks)}] {sym:20s} "
                        f"${r['total_return']:>10,.0f}  DD={r['dd_pct']:5.1f}%  "
                        f"Sharpe={r['sharpe']:5.2f}  Calmar={r['calmar']:7.1f}  "
                        f"PF={r['profit_factor']:5.2f}  WR={r['win_rate']:4.1f}%  "
                        f"trades={r['n_trades']:4d}  {ftmo}  "
                        f"[{r['session']}]{lf_info}"
                    )
                else:
                    print(f"  [{done}/{len(tasks)}] {sym:20s} {r.get('status', '?')}")
            except Exception as e:
                r = {"symbol": sym, "status": "error", "error": str(e)}
                print(f"  [{done}/{len(tasks)}] {sym:20s} ERROR: {e}")
            results.append(r)

    # Save raw results
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        raw_df = pl.from_dicts(ok_results).sort("calmar", descending=True)
        raw_csv = os.path.join(out_dir, "wfo_all_results.csv")
        raw_df.write_csv(raw_csv)

    # Build portfolio
    portfolio = build_portfolio(results, max_corr_per_group=args.max_corr)
    selected = portfolio["portfolio"]
    sessions = portfolio["sessions"]

    # Print portfolio
    print("\n" + "=" * 130)
    print("SELECTED PORTFOLIO (diversified, FTMO-compliant, 24h coverage)")
    print("=" * 130)

    total_return = 0
    total_trades = 0

    print(f"{'#':>3}  {'Symbol':20s} {'Session':15s} {'TF':4s} "
          f"{'Return':>10s} {'MaxDD%':>7s} {'Sharpe':>7s} {'Calmar':>8s} "
          f"{'PF':>6s} {'WR%':>5s} {'Trades':>6s} {'Risk%':>6s} {'RRR':>5s} "
          f"{'SL%':>5s} {'TP%':>5s} {'Trail%':>6s}")
    print("-" * 130)

    for i, r in enumerate(selected, 1):
        total_return += r["total_return"]
        total_trades += r["n_trades"]
        print(
            f"{i:3d}  {r['symbol']:20s} {r['session']:15s} {r['timeframe']:4s} "
            f"${r['total_return']:>9,.0f} {r['dd_pct']:6.1f}% {r['sharpe']:6.2f} "
            f"{r['calmar']:7.1f} {r['profit_factor']:5.2f} {r['win_rate']:4.1f}% "
            f"{r['n_trades']:5d} {r['risk_pct']:5.1f}% {r['rrr']:4.1f} "
            f"{r['pct_sl']:4.1f} {r['pct_tp']:4.1f} {r['pct_trail']:5.1f}"
        )

    print("-" * 130)
    print(f"     TOTAAL: ${total_return:>9,.0f}  |  {total_trades} trades  |  {len(selected)} symbols")

    # Session coverage
    print(f"\n{'SESSION COVERAGE':=^80}")
    for sess, syms in sessions.items():
        print(f"  {sess:25s}: {len(syms):2d} symbols — {', '.join(syms[:8])}" +
              (f" +{len(syms)-8} more" if len(syms) > 8 else ""))

    # Correlation group usage
    print(f"\n{'CORRELATION GROUPS':=^80}")
    for grp_name, grp_syms in CORRELATION_GROUPS.items():
        in_portfolio = [s for s in grp_syms if any(r["symbol"] == s for r in selected)]
        if in_portfolio:
            print(f"  {grp_name:15s}: {len(in_portfolio)}/{args.max_corr} — {', '.join(in_portfolio)}")

    # Save portfolio
    if selected:
        port_df = pl.from_dicts(selected)
        port_csv = os.path.join(out_dir, "portfolio_selected.csv")
        port_df.write_csv(port_csv)

        # Also save as JSON config for sovereign_configs
        port_config = {}
        for r in selected:
            sym = r["symbol"]
            port_config[sym] = {
                "asset_class": r["cluster"],
                "sector": r["session"],
                "atr_period": 14,
                "atr_timeframe": "H1",
                "atr_sl_mult": r["sl"],
                "atr_tp_mult": r["tp"],
                "breakeven_atr": r["be"],
                "trail_activation_atr": r["trail_act"],
                "trail_distance_atr": r["trail_dist"],
                "exit_horizon": r["horizon"],
                "exit_timeframe": r["timeframe"],
                "risk_per_trade": round(r["risk_pct"] / 100, 4),
                "max_spread_pct": 0.005 if r["cluster"] == "crypto" else 0.001 if r["cluster"] == "equity" else 0.0005,
                "magic_number": 2000,
                "wfo_sharpe": r["sharpe"],
                "wfo_calmar": r["calmar"],
                "wfo_profit_factor": r["profit_factor"],
                "wfo_total_return": r["total_return"],
                "wfo_max_dd_pct": r["dd_pct"],
                "wfo_n_trades": r["n_trades"],
            }

        config_path = os.path.join(out_dir, "portfolio_configs.json")
        with open(config_path, "w") as f:
            json.dump(port_config, f, indent=2, sort_keys=True)
            f.write("\n")

        print(f"\nSaved: {port_csv}")
        print(f"Saved: {config_path}")

    positive = sum(1 for r in ok_results if r.get("total_return", 0) > 0)
    negative = sum(1 for r in ok_results if r.get("total_return", 0) <= 0)
    print(f"\n[WFO] Done — {len(ok_results)} backtested, {positive} profitable, "
          f"{negative} unprofitable, {len(selected)} in portfolio")


if __name__ == "__main__":
    main()
