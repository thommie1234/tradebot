#!/usr/bin/env python3
"""
Pipeline Simulation Test — simulates 1 month of trading using real models,
real features, and real order routing logic with mocked MT5 data.

Usage:
    python3 tests/test_pipeline_sim.py
    python3 tests/test_pipeline_sim.py --days 7
    python3 tests/test_pipeline_sim.py --symbols DOGEUSD,TSLA
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
INITIAL_BALANCE = 95_000.0
MAX_CONCURRENT = 12
RISK_PER_TRADE = 0.005  # 0.5%
COMMISSION_PER_LOT = {
    "crypto": 3.0,
    "equity": 1.5,
    "index": 0.0,
    "commodity": 0.0,
    "forex": 0.0,
}
SPREAD_BPS = {
    "crypto": 10.0,
    "equity": 5.0,
    "index": 3.0,
    "commodity": 3.0,
    "forex": 8.0,
}

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SimPosition:
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float = 0.0
    swap: float = 0.0
    magic: int = 2000
    time: int = 0
    comment: str = ""

    def update_pnl(self, bid: float, ask: float, contract_size: float):
        if self.type == 0:  # BUY
            self.price_current = bid
            self.profit = (bid - self.price_open) * self.volume * contract_size
        else:  # SELL
            self.price_current = ask
            self.profit = (self.price_open - ask) * self.volume * contract_size


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    direction: str
    entry: float
    exit_price: float = 0.0
    volume: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    reason: str = ""
    bars_held: int = 0


@dataclass
class SimAccount:
    balance: float = INITIAL_BALANCE
    equity: float = INITIAL_BALANCE
    margin: float = 0.0
    margin_free: float = INITIAL_BALANCE
    margin_level: float = 0.0
    profit: float = 0.0


# ── Instrument Specs ────────────────────────────────────────────────────────

def load_instrument_specs() -> dict:
    """Load instrument specs from data/instrument_specs.json or use defaults."""
    specs_path = REPO / "data" / "instrument_specs.json"
    specs = {}
    if specs_path.exists():
        with open(specs_path) as f:
            raw = json.load(f)
        for sym, s in raw.items():
            specs[sym] = s
    return specs


def get_contract_size(symbol: str, specs: dict, asset_class: str) -> float:
    if symbol in specs:
        return specs[symbol].get("trade_contract_size", 1.0)
    if asset_class == "crypto":
        if "BTC" in symbol:
            return 1.0
        if "ETH" in symbol:
            return 1.0
        return 100_000.0  # altcoins
    if asset_class == "index":
        return 1.0
    if asset_class == "commodity":
        return 1.0
    if asset_class == "forex":
        return 100_000.0
    return 1.0  # equity


def get_volume_limits(symbol: str, specs: dict, asset_class: str):
    if symbol in specs:
        return (
            specs[symbol].get("volume_min", 0.01),
            specs[symbol].get("volume_max", 100.0),
            specs[symbol].get("volume_step", 0.01),
        )
    if asset_class == "equity":
        return (1.0, 10000.0, 1.0)
    if asset_class == "crypto":
        return (0.01, 1.0, 0.01)
    if asset_class == "index":
        return (0.01, 1000.0, 0.01)
    return (0.01, 100.0, 0.01)


# ── Load Data ───────────────────────────────────────────────────────────────

def load_bar_data(symbols: list[str], days: int) -> dict[str, list[dict]]:
    """Load H1 bar data for all symbols via MT5 bridge."""
    from tools.mt5_bridge import MT5BridgeClient

    mt5 = MT5BridgeClient(port=5056)
    bars_needed = days * 24 + 200  # extra for feature warmup
    TF_H1 = 16385  # MT5 TIMEFRAME_H1 constant

    all_bars = {}
    for sym in symbols:
        try:
            result = mt5.copy_rates_from_pos(sym, TF_H1, 0, bars_needed)
            if result and len(result) > 100:
                all_bars[sym] = result
                print(f"  {sym:<14} {len(result):>5} bars loaded")
            else:
                print(f"  {sym:<14} SKIP — insufficient data ({len(result) if result else 0} bars)")
        except Exception as e:
            print(f"  {sym:<14} ERROR — {e}")

    return all_bars


def get_active_symbols() -> list[str]:
    """Get all active symbols from sovereign_configs.json."""
    cfg_path = REPO / "config" / "sovereign_configs.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    return [s for s in cfg if s != "margin_leverage" and not cfg[s].get("disabled")]


def load_symbol_configs() -> dict:
    cfg_path = REPO / "config" / "sovereign_configs.json"
    with open(cfg_path) as f:
        return json.load(f)


# ── Feature & Inference ─────────────────────────────────────────────────────

def build_features_for_bar(bars_up_to: list[dict], z_threshold: float = 1.0):
    """Build features from bars using the real feature builder."""
    import polars as pl
    from engine.feature_builder import FEATURE_COLUMNS, build_bar_features

    if len(bars_up_to) < 100:
        return None

    df = pl.DataFrame({
        "time": [datetime.fromtimestamp(b["time"], tz=timezone.utc) for b in bars_up_to],
        "open": [float(b["open"]) for b in bars_up_to],
        "high": [float(b["high"]) for b in bars_up_to],
        "low": [float(b["low"]) for b in bars_up_to],
        "close": [float(b["close"]) for b in bars_up_to],
        "volume": [float(b.get("tick_volume", b.get("volume", 0))) for b in bars_up_to],
    })

    try:
        feat_df = build_bar_features(df, z_threshold=z_threshold)
        if feat_df is None or len(feat_df) == 0:
            return None

        last_row = feat_df.tail(1)
        available = [c for c in FEATURE_COLUMNS if c in last_row.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in last_row.columns]

        features = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
        for i, col in enumerate(FEATURE_COLUMNS):
            if col in last_row.columns:
                val = last_row[col][0]
                features[i] = float(val) if val is not None else 0.0

        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)

        return features.reshape(1, -1)
    except Exception as e:
        return None


def load_models(symbols: list[str]) -> dict:
    """Load real ML models for all symbols."""
    from engine.inference import SovereignMLFilter

    class QuietLogger:
        def log(self, *a, **kw): pass

    models = {}
    logger = QuietLogger()
    for sym in symbols:
        try:
            filt = SovereignMLFilter(sym, logger)
            if filt.load_model():
                models[sym] = filt
                print(f"  {sym:<14} model loaded")
            else:
                print(f"  {sym:<14} no model")
        except Exception as e:
            print(f"  {sym:<14} error: {e}")
    return models


# ── ATR Calculation ─────────────────────────────────────────────────────────

def calculate_atr(bars: list[dict], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return float(np.mean(trs[-period:]))


# ── Position Sizing ─────────────────────────────────────────────────────────

def calculate_lots(equity: float, risk_pct: float, sl_dist: float,
                   contract_size: float, vol_min: float, vol_max: float,
                   vol_step: float) -> float:
    if sl_dist <= 0 or contract_size <= 0:
        return 0.0
    raw = (equity * risk_pct) / (sl_dist * contract_size)
    if vol_step > 0:
        raw = math.floor(raw / vol_step) * vol_step
    return max(vol_min, min(vol_max, raw))


# ── Simulation Engine ───────────────────────────────────────────────────────

class PipelineSimulator:
    def __init__(self, symbols: list[str], bars: dict, configs: dict,
                 models: dict, specs: dict, days: int):
        self.symbols = symbols
        self.bars = bars
        self.configs = configs
        self.models = models
        self.specs = specs
        self.days = days

        self.account = SimAccount()
        self.positions: list[SimPosition] = []
        self.trades: list[TradeRecord] = []
        self.rejects: list[dict] = []
        self.ticket_counter = 100000
        self.current_time = ""

        # Tracking
        self.equity_curve: list[tuple[str, float]] = []
        self.max_equity = INITIAL_BALANCE
        self.max_drawdown = 0.0
        self.daily_pnl: dict[str, float] = defaultdict(float)

        # Correlation tracking
        self.crypto_positions: dict[str, str] = {}  # sym -> direction

    def run(self):
        """Main simulation loop."""
        # Find common time range
        all_times = set()
        for sym in self.symbols:
            if sym in self.bars:
                for b in self.bars[sym]:
                    all_times.add(b["time"])

        sorted_times = sorted(all_times)
        if not sorted_times:
            print("No data to simulate!")
            return

        # Only simulate last N days
        cutoff = sorted_times[-1] - (self.days * 86400)
        sim_times = [t for t in sorted_times if t >= cutoff]

        print(f"\nSimulating {len(sim_times)} bars over {self.days} days...")
        print(f"From {datetime.fromtimestamp(sim_times[0], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
        print(f"  To {datetime.fromtimestamp(sim_times[-1], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
        print()

        bar_count = 0
        for t in sim_times:
            self.current_time = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            day_str = self.current_time[:10]

            # 1. Update prices & check SL/TP
            self._update_positions(t)

            # 2. Trailing stop & breakeven
            self._manage_positions(t)

            # 3. For each symbol, check signals
            for sym in self.symbols:
                if sym not in self.bars or sym not in self.models:
                    continue
                sym_bars = [b for b in self.bars[sym] if b["time"] <= t]
                if len(sym_bars) < 100:
                    continue

                # Only check on H4 boundaries (every 4 hours) for H4 symbols
                bar_hour = datetime.fromtimestamp(t, tz=timezone.utc).hour
                sym_cfg = self.configs.get(sym, {})
                exit_tf = sym_cfg.get("exit_timeframe", "H4")
                if exit_tf == "H4" and bar_hour % 4 != 0:
                    continue

                # Build features
                features = build_features_for_bar(sym_bars)
                if features is None:
                    continue

                # Inference
                try:
                    proba = self.models[sym].predict(features)
                except Exception:
                    continue

                threshold = sym_cfg.get("prob_threshold", 0.55)
                if proba < threshold:
                    continue

                # Determine direction from primary_side
                from engine.feature_builder import FEATURE_COLUMNS
                ps_idx = FEATURE_COLUMNS.index("primary_side") if "primary_side" in FEATURE_COLUMNS else -1
                if ps_idx >= 0:
                    primary_side = features[0, ps_idx]
                    direction = "BUY" if primary_side > 0 else "SELL"
                else:
                    direction = "BUY"

                # Execute trade
                self._try_open(sym, direction, proba, sym_bars, t)

            # Track equity
            floating = sum(p.profit for p in self.positions)
            self.account.equity = self.account.balance + floating
            self.equity_curve.append((self.current_time, self.account.equity))

            if self.account.equity > self.max_equity:
                self.max_equity = self.account.equity
            dd = (self.max_equity - self.account.equity) / self.max_equity
            if dd > self.max_drawdown:
                self.max_drawdown = dd

            bar_count += 1
            if bar_count % 200 == 0:
                print(f"  Bar {bar_count}/{len(sim_times)}  "
                      f"Equity: ${self.account.equity:,.0f}  "
                      f"Positions: {len(self.positions)}  "
                      f"Trades: {len(self.trades)}")

        # Close remaining positions at last price
        for p in list(self.positions):
            self._close_position(p, "END_OF_SIM", sim_times[-1])

        print(f"\nSimulation complete: {bar_count} bars processed")

    def _update_positions(self, current_time: int):
        """Update P&L and check SL/TP for all positions."""
        for pos in list(self.positions):
            sym_bars = [b for b in self.bars.get(pos.symbol, []) if b["time"] <= current_time]
            if not sym_bars:
                continue

            last_bar = sym_bars[-1]
            bid = last_bar["close"]
            ask = bid * 1.0001  # tiny spread approximation

            asset_class = self.configs.get(pos.symbol, {}).get("asset_class", "equity")
            cs = get_contract_size(pos.symbol, self.specs, asset_class)
            pos.update_pnl(bid, ask, cs)

            # Check SL
            if pos.type == 0 and pos.sl > 0 and bid <= pos.sl:
                self._close_position(pos, "SL_HIT", current_time)
            elif pos.type == 1 and pos.sl > 0 and ask >= pos.sl:
                self._close_position(pos, "SL_HIT", current_time)
            # Check TP
            elif pos.type == 0 and pos.tp > 0 and bid >= pos.tp:
                self._close_position(pos, "TP_HIT", current_time)
            elif pos.type == 1 and pos.tp > 0 and ask <= pos.tp:
                self._close_position(pos, "TP_HIT", current_time)

    def _manage_positions(self, current_time: int):
        """Breakeven and trailing stop management."""
        for pos in list(self.positions):
            sym_cfg = self.configs.get(pos.symbol, {})
            sym_bars = [b for b in self.bars.get(pos.symbol, []) if b["time"] <= current_time]
            if len(sym_bars) < 15:
                continue

            atr = calculate_atr(sym_bars)
            if not atr or atr <= 0:
                continue

            be_atr = sym_cfg.get("breakeven_atr", 0.5)
            trail_act = sym_cfg.get("trail_activation_atr", 3.0)
            trail_dist = sym_cfg.get("trail_distance_atr", 1.0)

            last_bar = sym_bars[-1]
            current_price = last_bar["close"]

            if pos.type == 0:  # BUY
                profit_distance = current_price - pos.price_open

                # Breakeven
                if profit_distance >= be_atr * atr and pos.sl < pos.price_open:
                    pos.sl = pos.price_open + atr * 0.05  # small buffer

                # Trailing stop
                if profit_distance >= trail_act * atr:
                    new_sl = current_price - trail_dist * atr
                    if new_sl > pos.sl:
                        pos.sl = new_sl

            else:  # SELL
                profit_distance = pos.price_open - current_price

                # Breakeven
                if profit_distance >= be_atr * atr and (pos.sl > pos.price_open or pos.sl == 0):
                    pos.sl = pos.price_open - atr * 0.05

                # Trailing stop
                if profit_distance >= trail_act * atr:
                    new_sl = current_price + trail_dist * atr
                    if pos.sl == 0 or new_sl < pos.sl:
                        pos.sl = new_sl

    def _try_open(self, symbol: str, direction: str, proba: float,
                  sym_bars: list[dict], current_time: int):
        """Try to open a new position with guardrails."""
        sym_cfg = self.configs.get(symbol, {})
        asset_class = sym_cfg.get("asset_class", "equity")

        # Check if already in market
        existing = [p for p in self.positions if p.symbol == symbol]
        if existing:
            return  # skip, already have position

        # Max concurrent
        if len(self.positions) >= MAX_CONCURRENT:
            self.rejects.append({"time": self.current_time, "symbol": symbol,
                                 "reason": f"max {MAX_CONCURRENT} positions"})
            return

        # Correlation check (simplified)
        same_class_same_dir = sum(
            1 for p in self.positions
            if self.configs.get(p.symbol, {}).get("asset_class") == asset_class
            and ("BUY" if p.type == 0 else "SELL") == direction
        )
        max_same = 3 if asset_class == "crypto" else 4 if asset_class == "equity" else 2
        if same_class_same_dir >= max_same:
            self.rejects.append({"time": self.current_time, "symbol": symbol,
                                 "reason": f"correlation cap ({asset_class} {direction})"})
            return

        # Calculate ATR
        atr = calculate_atr(sym_bars)
        if not atr or atr <= 0:
            return

        # SL/TP distances
        sl_mult = sym_cfg.get("atr_sl_mult", 1.5)
        tp_mult = sym_cfg.get("atr_tp_mult", 7.0)
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        # Confidence scaling for TP
        conf_scale = max(1.0, min(2.0, proba / 0.55))
        tp_dist *= conf_scale

        # Position sizing
        cs = get_contract_size(symbol, self.specs, asset_class)
        vol_min, vol_max, vol_step = get_volume_limits(symbol, self.specs, asset_class)
        lots = calculate_lots(self.account.equity, RISK_PER_TRADE, sl_dist,
                              cs, vol_min, vol_max, vol_step)
        if lots <= 0:
            return

        # Entry price
        last_bar = sym_bars[-1]
        entry = last_bar["close"]
        spread = entry * SPREAD_BPS.get(asset_class, 5.0) / 10000
        if direction == "BUY":
            entry += spread / 2
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry -= spread / 2
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Commission
        comm = COMMISSION_PER_LOT.get(asset_class, 1.0) * lots

        # Open position
        self.ticket_counter += 1
        pos = SimPosition(
            ticket=self.ticket_counter,
            symbol=symbol,
            type=0 if direction == "BUY" else 1,
            volume=lots,
            price_open=entry,
            price_current=entry,
            sl=sl,
            tp=tp,
            magic=2000,
            time=current_time,
        )
        self.positions.append(pos)
        self.account.balance -= comm  # commission on entry

        self.trades.append(TradeRecord(
            timestamp=self.current_time,
            symbol=symbol,
            direction=direction,
            entry=entry,
            volume=lots,
            commission=comm,
        ))

    def _close_position(self, pos: SimPosition, reason: str, current_time: int):
        """Close a position and record the trade."""
        if pos not in self.positions:
            return

        self.positions.remove(pos)

        asset_class = self.configs.get(pos.symbol, {}).get("asset_class", "equity")
        comm = COMMISSION_PER_LOT.get(asset_class, 1.0) * pos.volume

        self.account.balance += pos.profit - comm
        day_str = self.current_time[:10]
        self.daily_pnl[day_str] += pos.profit - comm

        # Find matching trade record and update
        for tr in reversed(self.trades):
            if tr.symbol == pos.symbol and tr.exit_price == 0:
                tr.exit_price = pos.price_current
                tr.pnl = pos.profit - comm - tr.commission
                tr.reason = reason
                tr.bars_held = (current_time - pos.time) // 3600 if pos.time else 0
                break

    def report(self):
        """Print comprehensive simulation report."""
        closed = [t for t in self.trades if t.exit_price > 0]
        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in closed)
        total_comm = sum(t.commission for t in closed)

        print("\n" + "=" * 80)
        print("  PIPELINE SIMULATION REPORT")
        print("=" * 80)

        print(f"\n  Initial Balance:  ${INITIAL_BALANCE:>12,.2f}")
        print(f"  Final Balance:    ${self.account.balance:>12,.2f}")
        print(f"  Total P&L:        ${total_pnl:>+12,.2f}")
        print(f"  Total Commission: ${total_comm:>12,.2f}")
        print(f"  Max Drawdown:     {self.max_drawdown:>12.1%}")
        print(f"  Total Trades:     {len(closed):>12}")
        print(f"  Winners:          {len(winners):>12} ({len(winners)/max(1,len(closed))*100:.0f}%)")
        print(f"  Losers:           {len(losers):>12}")

        if winners:
            print(f"  Avg Win:          ${np.mean([t.pnl for t in winners]):>+12,.2f}")
        if losers:
            print(f"  Avg Loss:         ${np.mean([t.pnl for t in losers]):>+12,.2f}")

        # Profit Factor
        gross_win = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        print(f"  Profit Factor:    {pf:>12.2f}")

        # Sharpe (daily)
        if self.daily_pnl:
            daily_returns = list(self.daily_pnl.values())
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                print(f"  Sharpe (daily):   {sharpe:>12.2f}")

        # Calmar
        if self.max_drawdown > 0 and self.daily_pnl:
            annual_return = total_pnl / INITIAL_BALANCE * (365 / max(1, self.days))
            calmar = annual_return / self.max_drawdown
            print(f"  Calmar:           {calmar:>12.2f}")

        # FTMO check
        ftmo_daily_ok = all(v > -INITIAL_BALANCE * 0.05 for v in self.daily_pnl.values())
        ftmo_total_ok = self.max_drawdown < 0.10
        print(f"\n  FTMO Daily DD:    {'PASS' if ftmo_daily_ok else 'FAIL':>12}")
        print(f"  FTMO Total DD:    {'PASS' if ftmo_total_ok else 'FAIL':>12}")

        # Per-symbol breakdown
        print(f"\n{'─' * 80}")
        print(f"  {'Symbol':<14} {'Trades':>6} {'WR%':>5} {'P&L':>10} {'AvgW':>8} {'AvgL':>8} {'PF':>6} {'AvgBars':>7}")
        print(f"{'─' * 80}")

        sym_stats = defaultdict(lambda: {"trades": [], "wins": 0, "losses": 0})
        for t in closed:
            sym_stats[t.symbol]["trades"].append(t)
            if t.pnl > 0:
                sym_stats[t.symbol]["wins"] += 1
            else:
                sym_stats[t.symbol]["losses"] += 1

        for sym in sorted(sym_stats, key=lambda s: sum(t.pnl for t in sym_stats[s]["trades"]), reverse=True):
            s = sym_stats[sym]
            trades = s["trades"]
            total = sum(t.pnl for t in trades)
            wr = s["wins"] / len(trades) * 100 if trades else 0
            w = [t.pnl for t in trades if t.pnl > 0]
            l = [t.pnl for t in trades if t.pnl <= 0]
            avg_w = np.mean(w) if w else 0
            avg_l = np.mean(l) if l else 0
            gw = sum(w)
            gl = abs(sum(l))
            pf = gw / gl if gl > 0 else float("inf")
            avg_bars = np.mean([t.bars_held for t in trades])
            print(f"  {sym:<14} {len(trades):>6} {wr:>4.0f}% ${total:>+9,.0f} ${avg_w:>+7,.0f} ${avg_l:>+7,.0f} {pf:>5.2f} {avg_bars:>6.0f}h")

        # Recent rejects
        if self.rejects:
            print(f"\n{'─' * 80}")
            print(f"  REJECTED TRADES ({len(self.rejects)} total)")
            reject_reasons = defaultdict(int)
            for r in self.rejects:
                reject_reasons[r["reason"]] += 1
            for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason:<40} {count:>5}x")

        # Last 10 trades
        if closed:
            print(f"\n{'─' * 80}")
            print(f"  LAST 10 TRADES")
            print(f"  {'Time':<16} {'Symbol':<12} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'Reason':<12}")
            for t in closed[-10:]:
                print(f"  {t.timestamp:<16} {t.symbol:<12} {t.direction:<5} "
                      f"{t.entry:>10.4f} {t.exit_price:>10.4f} ${t.pnl:>+9,.2f} {t.reason:<12}")

        print(f"\n{'=' * 80}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Simulation Test")
    parser.add_argument("--days", type=int, default=30, help="Days to simulate")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    args = parser.parse_args()

    print("=" * 60)
    print("  SOVEREIGN BOT — PIPELINE SIMULATION")
    print("=" * 60)

    # Load configs
    print("\n[1/4] Loading configs...")
    configs = load_symbol_configs()
    specs = load_instrument_specs()

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = get_active_symbols()
    print(f"  {len(symbols)} symbols: {', '.join(symbols)}")

    # Load models
    print("\n[2/4] Loading ML models...")
    models = load_models(symbols)
    symbols = [s for s in symbols if s in models]
    print(f"  {len(models)} models loaded")

    # Load bar data
    print("\n[3/4] Loading bar data...")
    bars = load_bar_data(symbols, args.days)
    symbols = [s for s in symbols if s in bars]
    print(f"  {len(bars)} symbols with data")

    if not symbols:
        print("\nNo symbols with both models and data. Exiting.")
        return

    # Run simulation
    print("\n[4/4] Running simulation...")
    sim = PipelineSimulator(symbols, bars, configs, models, specs, args.days)
    sim.run()
    sim.report()


if __name__ == "__main__":
    main()
