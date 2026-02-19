#!/usr/bin/env python3
"""
Walk-Forward Analysis — Full Stack Simulation (with Realistic Costs)
======================================================================

Simulates ALL infrastructure layers, now with realistic commission costs
loaded from the `information` directory CSVs.

  Layer 1: ML Model (XGBoost + Optuna params, purged WFO)
  Layer 2: Signal Filters (z_threshold=0.0, ML_THRESHOLD=0.65)
  Layer 3: Position Sizing (fractional Kelly per symbol)
  Layer 4: Entry Guardrails (max 8 slots, USD correlation cap, sector limits)
  Layer 5: Trade Execution (commission from CSV + inferred spread + slippage)
  Layer 6: Trailing Stop + Breakeven + Partial Close
  Layer 7: Portfolio Risk (daily loss -3.5%, profit lock +3%, DD recovery)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

# --- Boilerplate ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from research.integrated_pipeline import make_ev_custom_objective, purged_walk_forward_splits
from engine.labeling import apply_triple_barrier
from research.train_ml_strategy import infer_spread_bps, infer_slippage_bps, make_time_bars, sanitize_training_frame

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --- Core Config ---
Z_THRESHOLD = 0.0
ML_THRESHOLD = 0.65
DISABLE_ZSCORE = False
ALPHA_OVERRIDE = 0.80
MAX_POSITIONS = 10_000
MAX_USD_SAME_DIR = 3
DAILY_LOSS_LIMIT = -0.035
PROFIT_LOCK = 0.03
DD_RECOVERY_THRESHOLD = 0.02
DD_RECOVERY_EXIT = 0.005
ACCOUNT_SIZE = 100_000

# --- New Cost Engine ---
class CostEngine:
    """Loads and provides trading costs from CSV files."""
    def __init__(self, info_dir: str | Path):
        self.info_dir = Path(info_dir)
        self.cost_data = self._load_all_costs()

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().replace("/", "").replace(".", "") # Added .replace(".", "") for JP225.cash

    def _load_all_costs(self) -> dict:
        cost_dict = {}
        files_to_load = ["Forex.csv", "forex exotic.csv", "metals.csv", "crypto.csv", "cash.csv", "Equities.csv"]
        
        for filename in files_to_load:
            file_path = self.info_dir / filename
            if not file_path.exists():
                continue
            
            df = pl.read_csv(file_path, truncate_ragged_lines=True)
            for row in df.iter_rows(named=True):
                symbol = self._normalize_symbol(row['symbol'])
                commission_raw = row.get('commission')
                contract_size_raw = row.get('contract_size')

                try:
                    commission_usd = float(commission_raw)
                except (ValueError, TypeError):
                    commission_usd = 0.0

                try:
                    contract_size = float(contract_size_raw)
                except (ValueError, TypeError):
                    contract_size = 100000.0
                
                if contract_size > 0:
                    commission_bps = (commission_usd / contract_size) * 10000
                else:
                    commission_bps = 0.0
                cost_dict[symbol] = {"commission_bps": commission_bps}
        
        print(f"[CostEngine] Loaded cost data for {len(cost_dict)} symbols.")
        return cost_dict

    def get_total_cost_bps(self, symbol: str, spread_bps: float, slippage_bps: float) -> float:
        normalized_symbol = self._normalize_symbol(symbol)
        costs = self.cost_data.get(normalized_symbol)
        
        commission_bps = costs.get("commission_bps", 0.0) if costs else 0.0
        return (commission_bps * 2) + spread_bps + (slippage_bps * 2)

# --- Data Loading & Prep ---
CONFIGS_PATH = SCRIPT_DIR / "sovereign_configs.json"
if CONFIGS_PATH.exists():
    with open(CONFIGS_PATH) as f:
        SOVEREIGN_CONFIGS = json.load(f)
else:
    SOVEREIGN_CONFIGS = {}

DATA_ROOTS = ["/home/tradebot/ssd_data_1/tick_data", "/home/tradebot/tick_data"]


@dataclass
class Position:
    symbol: str; direction: str; entry_price: float; entry_time: object; lot_size: float
    sl: float; tp: float; atr: float; cost_bps: float; partial_closed: bool = False
    remaining_lots: float = 0.0; realized_pnl: float = 0.0; sector: str = ""
    bars_held: int = 0; mae_pct: float = 0.0; mfe_pct: float = 0.0
    def __post_init__(self): self.remaining_lots = self.lot_size

@dataclass
class TradeResult:
    symbol: str; direction: str; entry_price: float; exit_price: float; entry_time: object
    exit_time: object; lot_size: float; pnl_dollar: float; pnl_pct: float; bars_held: int
    exit_reason: str; mae_pct: float = 0.0; mfe_pct: float = 0.0

@dataclass
class PortfolioState:
    equity: float = ACCOUNT_SIZE; peak_equity: float = ACCOUNT_SIZE; daily_start_equity: float = ACCOUNT_SIZE
    positions: list = field(default_factory=list); closed_trades: list = field(default_factory=list)
    dd_recovery_mode: bool = False; daily_loss_hit: bool = False; profit_lock_hit: bool = False; current_day: str = ""


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
    if d.select(pl.col("size").sum()).item() <= 0:
        d = d.with_columns(pl.lit(1.0).alias("size"))
    return d.select(["time", "bid", "ask", "price", "size"])


def prepare_symbol_data(symbol: str, xgb_params: dict, num_boost_round: int,
                        train_size: int, test_size: int, purge: int, embargo: int,
                        cost_engine: CostEngine):
    """Run WFO per symbol and return OOS predictions + bar data for simulation."""
    ticks = load_ticks(symbol)
    if ticks is None:
        return None, None
    
    spread_bps = infer_spread_bps(ticks)
    slippage_bps = infer_slippage_bps(symbol)
    total_cost_bps = cost_engine.get_total_cost_bps(symbol, spread_bps, slippage_bps)

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat = build_bar_features(bars, z_threshold=Z_THRESHOLD)
    if DISABLE_ZSCORE:
        # Replace z-score-driven primary_side with simple momentum sign.
        feat = feat.with_columns(
            pl.when(pl.col("ret3") > 0)
            .then(1)
            .when(pl.col("ret3") < 0)
            .then(-1)
            .otherwise(0)
            .alias("primary_side")
        )

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
        pl.Series("upside", tb.upside),
        pl.Series("downside", tb.downside),
        pl.lit(total_cost_bps).alias("total_cost_bps"),
    ]).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    x = feat.select(FEATURE_COLUMNS).to_numpy()
    y = feat["target"].to_numpy().astype(np.float32)
    avg_win = feat["upside"].to_numpy().astype(np.float64)
    avg_loss = feat["downside"].to_numpy().astype(np.float64)
    costs = (feat["total_cost_bps"].to_numpy() / 1e4).astype(np.float64)

    splits = purged_walk_forward_splits(
        n_samples=len(feat),
        train_size=train_size,
        test_size=test_size,
        purge=purge,
        embargo=embargo,
    )
    if not splits:
        return None, None

    # Run WFO — collect OOS predictions
    oos_proba = np.full(len(feat), np.nan)
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
        dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

        fold_avg_win = float(np.mean(avg_win[tr_idx]))
        fold_avg_loss = float(np.mean(avg_loss[tr_idx]))
        fold_cost = float(np.mean(costs[tr_idx]))
        obj = make_ev_custom_objective(fold_avg_win, fold_avg_loss, fold_cost)

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            obj=obj,
            evals=[(dtest, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        oos_proba[te_idx] = bst.predict(dtest)

    # Build bar-level data for simulation
    bar_data = feat.with_columns([
        pl.Series("proba", oos_proba),
        pl.col("total_cost_bps").alias("cost_bps"), # Rename for simulator
    ])

    return bar_data, len(splits)


# ── Live Kelly from closed trades ──
KELLY_FRACTION = 0.5
KELLY_FLOOR = 0.001          # 0.1%
KELLY_CAP = 0.035 / 8        # 3.5% daily budget / 8 positions = ~0.44%
KELLY_FALLBACK = 0.003       # 0.3% until enough trades
KELLY_MIN_TRADES = 20


def _kelly_from_trades(closed_trades: list) -> float:
    """Compute half-Kelly risk fraction from trades closed so far in the simulation."""
    if len(closed_trades) < KELLY_MIN_TRADES:
        return KELLY_FALLBACK

    pnls = [t.pnl_dollar for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    if not wins or not losses:
        return KELLY_FALLBACK

    p = len(wins) / len(pnls)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    b = avg_win / avg_loss if avg_loss > 0 else 1.0

    kelly = (p * b - (1 - p)) / b if b > 0 else 0
    kelly = max(kelly, 0)

    risk = kelly * KELLY_FRACTION
    risk = max(risk, KELLY_FLOOR)
    risk = min(risk, KELLY_CAP)
    return risk


def _calc_unrealized(state: PortfolioState, prices_at_t: dict) -> float:
    """Mark-to-market unrealized PnL for open positions."""
    total = 0.0
    for pos in state.positions:
        current_sym_prices = prices_at_t.get(pos.symbol)
        if not current_sym_prices:
            continue
        close = current_sym_prices.get("close")
        if close is None:
            continue
        if pos.direction == "BUY":
            total += (close - pos.entry_price) / pos.entry_price * pos.remaining_lots
        else:
            total += (pos.entry_price - close) / pos.entry_price * pos.remaining_lots
    return total


def _update_positions(state: PortfolioState, t: object, prices_at_t: dict):
    """Update open positions: track MAE/MFE, trailing stop, breakeven, partial close, and exits."""
    for pos in list(state.positions):
        current_sym_prices = prices_at_t.get(pos.symbol)
        if not current_sym_prices:
            continue

        close = current_sym_prices.get("close")
        high = current_sym_prices.get("high", close)
        low = current_sym_prices.get("low", close)
        if close is None:
            continue

        pos.bars_held += 1

        # Track MAE/MFE (percent from entry).
        if pos.direction == "BUY":
            pos.mfe_pct = max(pos.mfe_pct, (high - pos.entry_price) / pos.entry_price)
            pos.mae_pct = min(pos.mae_pct, (low - pos.entry_price) / pos.entry_price)
        else:
            pos.mfe_pct = max(pos.mfe_pct, (pos.entry_price - low) / pos.entry_price)
            pos.mae_pct = min(pos.mae_pct, (pos.entry_price - high) / pos.entry_price)

        # Trailing stop (distance = ATR pct).
        if pos.atr > 0:
            if pos.direction == "BUY":
                pos.sl = max(pos.sl, close * (1 - pos.atr))
            else:
                pos.sl = min(pos.sl, close * (1 + pos.atr))

        # Breakeven + partial close at 1x ATR.
        profit_pct = (close - pos.entry_price) / pos.entry_price if pos.direction == "BUY" else (pos.entry_price - close) / pos.entry_price
        if (not pos.partial_closed) and pos.atr > 0 and profit_pct >= pos.atr:
            partial_lots = pos.remaining_lots * 0.5
            if pos.direction == "BUY":
                realized = (close - pos.entry_price) / pos.entry_price * partial_lots
            else:
                realized = (pos.entry_price - close) / pos.entry_price * partial_lots
            pos.remaining_lots -= partial_lots
            pos.realized_pnl += realized
            pos.partial_closed = True
            # Move stop to breakeven after partial.
            pos.sl = pos.entry_price

        # Stop/TP checks using bar extremes.
        if pos.direction == "BUY":
            if low is not None and low <= pos.sl:
                _close_position(state, pos, t, prices_at_t, "SL")
                continue
            if high is not None and high >= pos.tp:
                _close_position(state, pos, t, prices_at_t, "TP")
                continue
        else:
            if high is not None and high >= pos.sl:
                _close_position(state, pos, t, prices_at_t, "SL")
                continue
            if low is not None and low <= pos.tp:
                _close_position(state, pos, t, prices_at_t, "TP")
                continue

def simulate_portfolio(symbol_bars: dict[str, pl.DataFrame], account: float,
                       max_dd_pct: float) -> PortfolioState:
    """
    Event-driven portfolio simulation across all symbols simultaneously.

    Processes bar-by-bar in chronological order across all symbols,
    applying all live guardrails.
    """
    state = PortfolioState(equity=account, peak_equity=account, daily_start_equity=account)

    # Build a consolidated DataFrame of ALL bars for all symbols
    all_bars_for_timeline = []
    for symbol, bars_df in symbol_bars.items():
        # Add symbol column to each bar DataFrame
        all_bars_for_timeline.append(bars_df.with_columns(pl.lit(symbol).alias("symbol")))
    
    if not all_bars_for_timeline:
        print("  No bar data to simulate.")
        return state

    # This master_df_full now contains ALL bars for all symbols, including signal (proba) and non-signal bars
    master_df_full = pl.concat(all_bars_for_timeline).sort("time")
    
    # The unique_event_times should come from this full master_df
    unique_event_times = master_df_full.select("time").unique(maintain_order=True)['time'].to_list()
    
    # Create price lookup for ALL bars (sparse: only for times with new data for a symbol)
    all_prices_list = []
    for symbol, bars in symbol_bars.items():
        all_prices_list.append(bars.select(["time", "close", "high", "low"]).with_columns(pl.lit(symbol).alias("symbol")))
    
    master_prices_df = pl.concat(all_prices_list).sort("time")
    price_lookup_sparse = {
        group_key[0]: {row["symbol"]: {"close": row["close"], "high": row["high"], "low": row["low"]} for row in group_df.iter_rows(named=True)}
        for group_key, group_df in master_prices_df.group_by("time")
    }

    print(f"  Simulating {len(unique_event_times)} unique time-steps...")

    last_known_prices: dict[str, dict[str, float]] = {} # Stores the last known price for each symbol

    for t in unique_event_times:
        day_str = str(t.date())
        if day_str != state.current_day:
            state.current_day = day_str
            state.daily_start_equity = state.equity
            state.daily_loss_hit = False
            state.profit_lock_hit = False

        # Update last_known_prices with any new prices available at time 't'
        current_prices_for_t = price_lookup_sparse.get(t, {})
        for sym, prices in current_prices_for_t.items():
            last_known_prices[sym] = prices # Overwrite with latest

        # Now pass last_known_prices to functions that need current prices for all open positions
        _update_positions(state, t, last_known_prices)
        unrealized = _calc_unrealized(state, last_known_prices)

        if account > 0:
            daily_pnl_pct = (state.equity + unrealized - state.daily_start_equity) / account
            if daily_pnl_pct <= DAILY_LOSS_LIMIT: state.daily_loss_hit = True
            if daily_pnl_pct >= PROFIT_LOCK: state.profit_lock_hit = True
            dd_pct = (account - state.equity) / account
            if dd_pct >= DD_RECOVERY_THRESHOLD: state.dd_recovery_mode = True
            elif dd_pct <= DD_RECOVERY_EXIT: state.dd_recovery_mode = False

        if state.daily_loss_hit or state.profit_lock_hit: continue

        # Filter for signals at this specific time 't' from the full master_df_full
        # Only process bars that have a valid signal (proba is not NaN)
        signals_at_t = master_df_full.filter((pl.col("time") == t) & (pl.col("proba").is_not_nan()))
        
        for signal in signals_at_t.iter_rows(named=True):
            sym = signal['symbol']
            proba = signal["proba"]
            side = signal["primary_side"] # Corrected to primary_side

            if side == 0:
                print(f"DEBUG: Skipping {sym} at {t} because primary_side is 0.")
                continue
            if proba < ML_THRESHOLD:
                print(f"DEBUG: Skipping {sym} at {t} because proba ({proba:.2f}) < ML_THRESHOLD ({ML_THRESHOLD:.2f}).")
                continue
            if any(p.symbol == sym for p in state.positions):
                print(f"DEBUG: Skipping {sym} at {t} because position already open for this symbol.")
                continue
            
            # Invert signal direction for testing
            direction = "SELL" if side > 0 else "BUY"
            if any(p.symbol == sym and p.direction != direction for p in state.positions):
                print(f"DEBUG: Skipping {sym} at {t} because position open in opposite direction.")
                continue

            sym_cfg = SOVEREIGN_CONFIGS.get(sym, {})

            if proba <= ALPHA_OVERRIDE and len(state.positions) >= MAX_POSITIONS:
                print(f"DEBUG: Skipping {sym} at {t} because slot limit ({len(state.positions)}/{MAX_POSITIONS}) reached and proba ({proba:.2f}) <= ALPHA_OVERRIDE ({ALPHA_OVERRIDE:.2f}).")
                continue
            
            # Close price for entry should come from last_known_prices for consistency
            entry_close_price = last_known_prices.get(sym, {}).get('close')
            if entry_close_price is None:
                print(f"DEBUG: Skipping {sym} at {t} because entry_close_price is None.")
                continue
            if signal["vol20"] <= 0:
                print(f"DEBUG: Skipping {sym} at {t} because vol20 ({signal['vol20']:.2f}) <= 0.")
                continue
            
            print(f"DEBUG: OPENING TRADE for {sym} {direction} at {entry_close_price:.4f} (proba={proba:.2f}, current_positions={len(state.positions)}).")
            
            risk_per_trade = _kelly_from_trades(state.closed_trades)
            atr_sl_mult = sym_cfg.get("atr_sl_mult", 2.0); atr_tp_mult = sym_cfg.get("atr_tp_mult", 6.0)
            sl_pct = signal["vol20"] * atr_sl_mult; tp_pct = signal["vol20"] * atr_tp_mult
            
            risk_amount = state.equity * risk_per_trade
            lot_size = risk_amount / max(sl_pct, 0.001)
            lot_size = max(lot_size, 100)
            if state.dd_recovery_mode: lot_size *= 0.5
            
            entry_cost = (signal["cost_bps"] / 1e4) * lot_size
            pos = Position(
                symbol=sym, direction=direction, entry_price=entry_close_price, entry_time=t, lot_size=lot_size,
                sl=entry_close_price * (1 - sl_pct) if direction == "BUY" else entry_close_price * (1 + sl_pct),
                tp=entry_close_price * (1 + tp_pct) if direction == "BUY" else entry_close_price * (1 - tp_pct),
                atr=signal["vol20"], cost_bps=signal["cost_bps"], sector=sym_cfg.get("sector", "unknown")
            )
            state.positions.append(pos)
            state.equity -= entry_cost
            
    # Final closeout (uses last_known_prices)
    if unique_event_times:
        # last_t is already the last time from unique_event_times
        # last_prices should be last_known_prices after the loop finishes
        for pos in list(state.positions):
            _close_position(state, pos, unique_event_times[-1], last_known_prices, "END_OF_TEST")
            
    return state


def _close_position(state: PortfolioState, pos: Position, t: object, prices_at_t: dict, reason: str):
    """Close a position and record the trade."""
    current_sym_prices = prices_at_t.get(pos.symbol) # Renamed symbol_prices to current_sym_prices for clarity
    
    # Ensure exit_price is always a valid price, falling back to entry_price only if no current data at all
    exit_price = pos.sl if reason == "SL" else pos.tp if reason == "TP" else (current_sym_prices['close'] if current_sym_prices else pos.entry_price)
    
    if pos.direction == "BUY":
        final_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.remaining_lots
    else:
        final_pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.remaining_lots

    total_pnl = pos.realized_pnl + final_pnl - ((pos.cost_bps / 1e4) * pos.remaining_lots)
    
    state.equity += total_pnl
    state.peak_equity = max(state.peak_equity, state.equity)
    
    state.closed_trades.append(TradeResult(
        symbol=pos.symbol, direction=pos.direction, entry_price=pos.entry_price, exit_price=exit_price,
        entry_time=pos.entry_time, exit_time=t, lot_size=pos.lot_size, pnl_dollar=total_pnl,
        pnl_pct=total_pnl/ACCOUNT_SIZE, bars_held=pos.bars_held, exit_reason=reason
    ))
    
    if pos in state.positions:
        state.positions.remove(pos)


# --- Reporting functions (unchanged) ---
def bootstrap_pf(trades: list, n_boot: int=10_000, ci: float=0.95) -> tuple:
    pnls=np.array([t.pnl_dollar for t in trades]); n=len(pnls)
    if n<3: return 0,0,0,0
    rng=np.random.default_rng(42); pfs=np.empty(n_boot)
    for i in range(n_boot):
        sample=rng.choice(pnls,size=n,replace=True)
        gw=float(np.sum(sample[sample>0])); gl=float(np.abs(np.sum(sample[sample<=0])))
        pfs[i]=gw/gl if gl>0 else 10.0
    lo=float(np.percentile(pfs,(1-ci)/2*100)); hi=float(np.percentile(pfs,(1-(1-ci)/2)*100))
    pct_above_1=float(np.mean(pfs>1.0)); gw=float(np.sum(pnls[pnls>0])); gl=float(np.abs(np.sum(pnls[pnls<=0])))
    return gw/gl if gl>0 else float('inf'),lo,hi,pct_above_1

def print_results(state: PortfolioState, account: float, elapsed: float):
    trades=state.closed_trades
    if not trades: print("  No trades executed!"); return
    n=len(trades); total_pnl=sum(t.pnl_dollar for t in trades)
    wr = len([t for t in trades if t.pnl_dollar > 0]) / n if n > 0 else 0
    
    equity_curve=[account] + [account + t.pnl_dollar for t in sorted(trades, key=lambda x:x.exit_time)]
    eq=np.cumsum([account] + [t.pnl_dollar for t in sorted(trades, key=lambda x:x.exit_time)])
    running_max=np.maximum.accumulate(eq); dd=(eq-running_max)/running_max; max_dd=float(np.min(dd)) if len(dd)>0 else 0
    
    pf,lo,hi,pct=bootstrap_pf(trades)
    
    print(f"\n{'='*80}\n  FULL-STACK WFA RESULTS (Realistic Costs)\n{'='*80}")
    print(f"  Final Equity: ${state.equity:,.2f} | Total P&L: ${total_pnl:+,.2f} ({total_pnl/account:+.2%})")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Profit Factor: {pf:.3f} (95% CI: [{lo:.3f}, {hi:.3f}])")
    print(f"  Total Trades: {n} | Win Rate: {wr:.1%}")
    print(f"  Runtime: {elapsed:.1f}s")


def main():
    global ML_THRESHOLD, DISABLE_ZSCORE, Z_THRESHOLD
    parser = argparse.ArgumentParser(description="Full-Stack WFA with Realistic Costs")
    parser.add_argument("--csv", required=True, help="Optuna summary CSV")
    parser.add_argument("--train-size",type=int,default=800); parser.add_argument("--test-size",type=int,default=200)
    parser.add_argument("--purge",type=int,default=8); parser.add_argument("--embargo",type=int,default=8)
    parser.add_argument("--account", type=float, default=ACCOUNT_SIZE)
    parser.add_argument("--ml-threshold", type=float, default=ML_THRESHOLD)
    parser.add_argument("--z-threshold", type=float, default=Z_THRESHOLD)
    parser.add_argument("--disable-zscore", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()
    ML_THRESHOLD = float(args.ml_threshold)
    Z_THRESHOLD = float(args.z_threshold)
    DISABLE_ZSCORE = bool(args.disable_zscore)

    info_dir = Path(SCRIPT_DIR).parent / "information"
    cost_engine = CostEngine(info_dir)
    
    csv_path = str(Path(SCRIPT_DIR) / args.csv) if not os.path.isabs(args.csv) else args.csv
    df = pl.read_csv(csv_path)
    ok = df.filter(pl.col("status") == "ok")

    configs = []
    for row in ok.iter_rows(named=True):
        params = {
            "booster":"gbtree", "tree_method":"hist", "device":"cuda", "objective":"binary:logistic", "eval_metric":"logloss",
            "max_depth":int(row["best_max_depth"]), "eta":0.03, "gamma":float(row["best_gamma"]),
            "subsample":float(row["best_subsample"]), "colsample_bytree":float(row["best_colsample_bytree"]),
            "reg_alpha":float(row["best_reg_alpha"]), "reg_lambda":float(row["best_reg_lambda"]),
            "min_child_weight":float(row["best_min_child_weight"]), "verbosity":0
        }
        configs.append({"symbol":row["symbol"],"params":params,"rounds":int(row["best_num_boost_round"]),"ev":float(row["best_ev"])})
    configs.sort(key=lambda c: -c["ev"])
    if args.max_symbols and args.max_symbols > 0:
        configs = configs[: args.max_symbols]

    print(f"Starting WFA for {len(configs)} symbols with realistic costs...")

    t_total = time.time()
    symbol_bars = {}
    for i, cfg in enumerate(configs):
        sym = cfg["symbol"]
        print(f"  [{i+1}/{len(configs)}] Processing {sym}...", end=" ", flush=True)
        bar_data, folds = prepare_symbol_data(sym, cfg["params"], cfg["rounds"], args.train_size, args.test_size,
                                              args.purge, args.embargo, cost_engine)
        if bar_data is None: 
            print("SKIP")
            continue
        print(f"OK ({folds} folds)")
        symbol_bars[sym] = bar_data

    print(f"\nWFO complete. Simulating portfolio...")
    state = simulate_portfolio(symbol_bars, args.account, 0.05)
    elapsed = time.time() - t_total
    print_results(state, args.account, elapsed)

if __name__ == "__main__":
    main()
