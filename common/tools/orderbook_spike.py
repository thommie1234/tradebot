#!/usr/bin/env python3
"""Orderbook Spike Detector — paper trades based on crypto orderbook imbalance.

Connects to Alpaca crypto websocket for real-time orderbook data.
Detects large bid/ask imbalances that precede price spikes.
Executes paper trades via MT5 bridge with separate logging.

This is STANDALONE — no sector limits, correlation guards, or position caps.
All trades logged to spike_trades.db (separate from normal paper_trades.db).

Usage:
    python3 common/tools/orderbook_spike.py --account-id ftmo_100k
    python3 common/tools/orderbook_spike.py --account-id bright_100k

Systemd:
    systemctl --user start spike-detector
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "common"))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SPIKE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spike")

# ── Config ────────────────────────────────────────────────────────────

# Alpaca crypto symbols to monitor (must exist on broker too)
# Format: alpaca_symbol -> broker_symbol
SYMBOLS = {
    "BTC/USD": "BTCUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
    "DOGE/USD": "DOGEUSD",
    "XRP/USD": "XRPUSD",
    "LINK/USD": "LINKUSD",
    "LTC/USD": "LTCUSD",
    "ADA/USD": "ADAUSD",
}

# Imbalance thresholds
IMBALANCE_WINDOW = 30          # seconds of orderbook snapshots to average
IMBALANCE_TRIGGER = 2.5        # bid/ask ratio to trigger (2.5 = 2.5x more on bid)
IMBALANCE_CONFIRM_SECS = 5    # imbalance must persist this many seconds
PRICE_CONFIRM_PCT = 0.05       # price must move 0.05% in imbalance direction to confirm
INSTANT_TRIGGER = 20.0         # ratio above this = instant entry, no confirmation wait
MAX_IMBALANCE = 50.0           # skip if ratio > this — too volatile, slippage kills profits
MAX_SLIPPAGE_PCT = 0.15        # skip trade if price already moved >0.15% past trigger

# Trade parameters
RISK_USD = 500                 # was $2000 — reduced because 3x slippage common on spikes
SL_PCT = 0.5                   # 0.5% stop loss
TP_PCT = 0.5                   # 0.5% take profit (realistic for 2-min spike)
SPIKE_TIMEOUT_SECS = 90        # close trade after 90s if spike didn't resolve
TRAIL_ACTIVATE_PCT = 0.15      # activate trailing SL after 0.15% profit
TRAIL_OFFSET_PCT = 0.10        # trail SL 0.10% behind best price
MAX_OPEN_PER_SYMBOL = 1        # max 1 spike trade per symbol
MAX_OPEN_TOTAL = 10            # max 10 spike trades total
COOLDOWN_SECS = 300            # 5 min cooldown per symbol after trade

# ── Data structures ───────────────────────────────────────────────────

@dataclass
class OrderbookState:
    """Rolling orderbook state for one symbol."""
    bid_volumes: deque = field(default_factory=lambda: deque(maxlen=60))
    ask_volumes: deque = field(default_factory=lambda: deque(maxlen=60))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    last_bid: float = 0.0
    last_ask: float = 0.0
    imbalance_start: float = 0.0  # when imbalance first detected
    imbalance_direction: str = ""  # "BUY" or "SELL"
    price_at_trigger: float = 0.0
    last_trade_time: float = 0.0  # cooldown tracking


@dataclass
class SpikePosition:
    id: int = 0
    symbol: str = ""
    broker_symbol: str = ""
    direction: str = ""
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    risk_usd: float = 0.0
    imbalance_ratio: float = 0.0
    entry_time: str = ""
    status: str = "OPEN"
    best_price: float = 0.0      # best price seen (for trailing SL)


# ── Database ──────────────────────────────────────────────────────────

class SpikeDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS spike_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            broker_symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            sl_price REAL NOT NULL,
            tp_price REAL NOT NULL,
            risk_usd REAL NOT NULL,
            imbalance_ratio REAL NOT NULL,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_timestamp TEXT,
            pnl REAL,
            exit_reason TEXT,
            account_id TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS spike_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT
        )""")
        conn.commit()
        conn.close()

    def open_trade(self, pos: SpikePosition, account_id: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            """INSERT INTO spike_trades
               (timestamp, symbol, broker_symbol, direction, entry_price,
                sl_price, tp_price, risk_usd, imbalance_ratio, status, account_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
            (pos.entry_time, pos.symbol, pos.broker_symbol, pos.direction,
             pos.entry_price, pos.sl_price, pos.tp_price, pos.risk_usd,
             pos.imbalance_ratio, account_id),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id

    def close_trade(self, trade_id: int, exit_price: float, pnl: float, reason: str):
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """UPDATE spike_trades SET
               status = 'CLOSED', exit_price = ?, exit_timestamp = ?,
               pnl = ?, exit_reason = ?
               WHERE id = ?""",
            (exit_price, now, pnl, reason, trade_id),
        )
        conn.commit()
        conn.close()

    def log_event(self, symbol: str, event_type: str, details: str = ""):
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO spike_events (timestamp, symbol, event_type, details) VALUES (?, ?, ?, ?)",
            (now, symbol, event_type, details),
        )
        conn.commit()
        conn.close()

    def get_open_trades(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM spike_trades WHERE status = 'OPEN'"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def count_open(self, symbol: str = None) -> int:
        conn = sqlite3.connect(self.db_path)
        if symbol:
            row = conn.execute(
                "SELECT COUNT(*) FROM spike_trades WHERE status = 'OPEN' AND symbol = ?",
                (symbol,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM spike_trades WHERE status = 'OPEN'"
            ).fetchone()
        conn.close()
        return row[0]


# ── Spike Detector ────────────────────────────────────────────────────

class SpikeDetector:
    def __init__(self, account_id: str, bridge_port: int, db: SpikeDB):
        self.account_id = account_id
        self.bridge_port = bridge_port
        self.db = db
        self.states: dict[str, OrderbookState] = {
            sym: OrderbookState() for sym in SYMBOLS
        }
        self.positions: list[SpikePosition] = []
        self._mt5 = None
        self._monitor_running = False

        # Load commission from cost model (FTMO = per-instrument, BF = 0)
        try:
            from research.cost_model import get_cost_model
            cm = get_cost_model(account_id)
            self._commission_bps = cm.commission_bps("ETHUSD")  # crypto rate
        except Exception:
            self._commission_bps = 6.5  # FTMO default

    def _get_mt5(self):
        if self._mt5 is None:
            import MetaTrader5 as _mt5
            if _mt5.terminal_info() is not None or _mt5.initialize():
                self._mt5 = _mt5
        return self._mt5

    def on_orderbook(self, data):
        """Called on each orderbook update from Alpaca websocket."""
        symbol = data.symbol
        if symbol not in self.states:
            return

        state = self.states[symbol]
        now = time.time()

        # Sum bid/ask volumes from top levels
        bid_vol = sum(float(b.size) for b in data.bids[:10]) if data.bids else 0
        ask_vol = sum(float(a.size) for a in data.asks[:10]) if data.asks else 0

        state.bid_volumes.append(bid_vol)
        state.ask_volumes.append(ask_vol)
        state.timestamps.append(now)

        if data.bids:
            state.last_bid = float(data.bids[0].price)
        if data.asks:
            state.last_ask = float(data.asks[0].price)

        # Check for imbalance
        self._check_imbalance(symbol, state, now)

    def _check_imbalance(self, symbol: str, state: OrderbookState, now: float):
        """Check if orderbook imbalance triggers a spike signal."""
        if len(state.bid_volumes) < 5:
            return

        # Only use recent snapshots (within IMBALANCE_WINDOW seconds)
        cutoff = now - IMBALANCE_WINDOW
        recent_bids = []
        recent_asks = []
        for i, ts in enumerate(state.timestamps):
            if ts >= cutoff:
                recent_bids.append(state.bid_volumes[i])
                recent_asks.append(state.ask_volumes[i])

        if not recent_bids:
            return

        avg_bid = sum(recent_bids) / len(recent_bids)
        avg_ask = sum(recent_asks) / len(recent_asks)

        if avg_ask <= 0 or avg_bid <= 0:
            return

        bid_ask_ratio = avg_bid / avg_ask
        ask_bid_ratio = avg_ask / avg_bid

        direction = ""
        ratio = 0.0
        if bid_ask_ratio >= IMBALANCE_TRIGGER:
            direction = "BUY"
            ratio = bid_ask_ratio
        elif ask_bid_ratio >= IMBALANCE_TRIGGER:
            direction = "SELL"
            ratio = ask_bid_ratio

        if not direction:
            # Reset if no imbalance
            state.imbalance_start = 0
            state.imbalance_direction = ""
            return

        # First detection — record start
        if state.imbalance_direction != direction:
            state.imbalance_start = now
            state.imbalance_direction = direction
            state.price_at_trigger = (state.last_bid + state.last_ask) / 2
            self.db.log_event(symbol, "IMBALANCE_START",
                              f"{direction} ratio={ratio:.2f} price={state.price_at_trigger:.4f}")
            log.info(f"{symbol} imbalance detected: {direction} ratio={ratio:.2f}")

            # Extreme imbalance = instant entry at trigger price (no wait)
            if ratio >= INSTANT_TRIGGER:
                log.info(f"{symbol} EXTREME imbalance {ratio:.1f}x >= {INSTANT_TRIGGER}x — instant entry")
                mid = state.price_at_trigger  # use trigger price, not delayed price
            else:
                return

        else:
            # Check if imbalance persisted long enough
            elapsed = now - state.imbalance_start
            if elapsed < IMBALANCE_CONFIRM_SECS:
                return

            # Check price confirmation
            mid = (state.last_bid + state.last_ask) / 2
            price_change_pct = (mid - state.price_at_trigger) / state.price_at_trigger * 100

            confirmed = False
            if direction == "BUY" and price_change_pct >= PRICE_CONFIRM_PCT:
                confirmed = True
            elif direction == "SELL" and price_change_pct <= -PRICE_CONFIRM_PCT:
                confirmed = True

            if not confirmed:
                return

            # Slippage guard: skip if price already moved too far past trigger
            slippage_pct = abs(price_change_pct)
            if slippage_pct > MAX_SLIPPAGE_PCT:
                log.warning(f"{symbol} skipping — price moved {slippage_pct:.2f}% past trigger (max {MAX_SLIPPAGE_PCT}%)")
                state.imbalance_start = 0
                state.imbalance_direction = ""
                return

        # Extreme imbalance guard — too volatile, slippage will destroy us
        if ratio > MAX_IMBALANCE:
            log.warning(f"{symbol} skipping — imbalance {ratio:.1f}x > {MAX_IMBALANCE}x (too volatile)")
            state.imbalance_start = 0
            state.imbalance_direction = ""
            return

        # Cooldown check
        if now - state.last_trade_time < COOLDOWN_SECS:
            return

        # Position limit check
        if self.db.count_open(symbol) >= MAX_OPEN_PER_SYMBOL:
            return
        if self.db.count_open() >= MAX_OPEN_TOTAL:
            return

        # FIRE! Execute paper trade — use MT5 price for realistic fill
        # Alpaca orderbook detects the spike, but we fill at MT5 broker price
        broker_sym = SYMBOLS[symbol]
        mt5 = self._get_mt5()
        if mt5:
            tick = mt5.symbol_info_tick(broker_sym)
            if tick and tick.bid > 0:
                fill_price = tick.ask if direction == "BUY" else tick.bid
                spread = tick.ask - tick.bid
                # Verify MT5 price isn't already far from Alpaca signal price
                state = self.states[symbol]
                alpaca_mid = (state.last_bid + state.last_ask) / 2
                if alpaca_mid > 0:
                    drift_pct = abs(fill_price - alpaca_mid) / alpaca_mid * 100
                    if drift_pct > MAX_SLIPPAGE_PCT:
                        log.warning(f"{symbol} skipping — MT5 price {fill_price:.4f} drifted "
                                    f"{drift_pct:.2f}% from Alpaca {alpaca_mid:.4f}")
                        state.imbalance_start = 0
                        state.imbalance_direction = ""
                        return
            else:
                state = self.states[symbol]
                fill_price = state.last_ask if direction == "BUY" else state.last_bid
                spread = abs(state.last_ask - state.last_bid) if state.last_ask > 0 else 0
        else:
            state = self.states[symbol]
            fill_price = state.last_ask if direction == "BUY" else state.last_bid
            spread = abs(state.last_ask - state.last_bid) if state.last_ask > 0 else 0

        self._execute_spike_trade(symbol, direction, fill_price, ratio, spread)

        # Reset state
        state.imbalance_start = 0
        state.imbalance_direction = ""
        state.last_trade_time = now

    def _execute_spike_trade(self, symbol: str, direction: str, price: float,
                             ratio: float, spread: float = 0):
        """Open a paper spike trade with realistic costs."""
        broker_sym = SYMBOLS[symbol]

        if direction == "BUY":
            sl = price * (1 - SL_PCT / 100)
            tp = price * (1 + TP_PCT / 100)
        else:
            sl = price * (1 + SL_PCT / 100)
            tp = price * (1 - TP_PCT / 100)

        # Calculate commission from cost model
        position_value = RISK_USD / (SL_PCT / 100)
        comm_bps = self._commission_bps
        # Round-trip commission in USD
        commission_usd = (comm_bps / 10000) * position_value * 2

        now = datetime.now(timezone.utc).isoformat()
        pos = SpikePosition(
            symbol=symbol,
            broker_symbol=broker_sym,
            direction=direction,
            entry_price=price,
            sl_price=sl,
            tp_price=tp,
            risk_usd=RISK_USD,
            imbalance_ratio=ratio,
            entry_time=now,
            best_price=price,
        )

        pos.id = self.db.open_trade(pos, self.account_id)
        self.positions.append(pos)

        self.db.log_event(symbol, "SPIKE_ENTRY",
                          f"{direction} @ {price:.4f} SL={sl:.4f} TP={tp:.4f} "
                          f"imbalance={ratio:.2f} spread={spread:.4f} comm=${commission_usd:.2f}")
        log.info(f"SPIKE TRADE: {symbol} {direction} @ {price:.4f} "
                 f"SL={sl:.4f} TP={tp:.4f} ratio={ratio:.2f} "
                 f"spread={spread:.4f} comm=${commission_usd:.2f}")

    def start_monitor(self):
        """Start background thread to monitor open spike positions via MT5 ticks."""
        self._monitor_running = True

        # Restore open positions from DB
        open_trades = self.db.get_open_trades()
        for t in open_trades:
            self.positions.append(SpikePosition(
                id=t["id"], symbol=t["symbol"], broker_symbol=t["broker_symbol"],
                direction=t["direction"], entry_price=t["entry_price"],
                sl_price=t["sl_price"], tp_price=t["tp_price"],
                risk_usd=t["risk_usd"], imbalance_ratio=t["imbalance_ratio"],
                entry_time=t["timestamp"],
            ))
        if self.positions:
            log.info(f"Restored {len(self.positions)} open spike positions")

        t = threading.Thread(target=self._monitor_loop, daemon=True, name="SpikeMonitor")
        t.start()

    def _monitor_loop(self):
        """Check open positions against live prices."""
        while self._monitor_running:
            if not self.positions:
                time.sleep(2)
                continue

            mt5 = self._get_mt5()
            if mt5 is None:
                time.sleep(5)
                continue

            # Quick probe
            try:
                test = mt5.symbol_info_tick(self.positions[0].broker_symbol)
                if test is None:
                    time.sleep(5)
                    continue
            except Exception:
                time.sleep(5)
                continue

            closed = []
            for pos in self.positions:
                try:
                    tick = mt5.symbol_info_tick(pos.broker_symbol)
                    if not tick or tick.bid <= 0:
                        continue

                    current = tick.bid if pos.direction == "BUY" else tick.ask

                    # Check SL
                    sl_hit = False
                    if pos.direction == "BUY" and current <= pos.sl_price:
                        sl_hit = True
                    elif pos.direction == "SELL" and current >= pos.sl_price:
                        sl_hit = True

                    if sl_hit:
                        pnl = self._calc_pnl(pos, current)
                        self.db.close_trade(pos.id, current, pnl, "SL")
                        closed.append(pos)
                        log.info(f"SPIKE CLOSED SL: {pos.symbol} {pos.direction} "
                                 f"PnL=${pnl:+.2f}")
                        continue

                    # Check TP
                    tp_hit = False
                    if pos.direction == "BUY" and current >= pos.tp_price:
                        tp_hit = True
                    elif pos.direction == "SELL" and current <= pos.tp_price:
                        tp_hit = True

                    if tp_hit:
                        pnl = self._calc_pnl(pos, current)
                        self.db.close_trade(pos.id, current, pnl, "TP")
                        closed.append(pos)
                        log.info(f"SPIKE CLOSED TP: {pos.symbol} {pos.direction} "
                                 f"PnL=${pnl:+.2f}")
                        continue

                    # Trailing SL — lock in profit once we're up TRAIL_ACTIVATE_PCT
                    if pos.direction == "BUY":
                        if current > pos.best_price:
                            pos.best_price = current
                        profit_pct = (pos.best_price - pos.entry_price) / pos.entry_price * 100
                        if profit_pct >= TRAIL_ACTIVATE_PCT:
                            trail_sl = pos.best_price * (1 - TRAIL_OFFSET_PCT / 100)
                            if trail_sl > pos.sl_price:
                                pos.sl_price = trail_sl
                    else:
                        if current < pos.best_price or pos.best_price <= 0:
                            pos.best_price = current
                        profit_pct = (pos.entry_price - pos.best_price) / pos.entry_price * 100
                        if profit_pct >= TRAIL_ACTIVATE_PCT:
                            trail_sl = pos.best_price * (1 + TRAIL_OFFSET_PCT / 100)
                            if trail_sl < pos.sl_price:
                                pos.sl_price = trail_sl

                    # Check timeout — spike should resolve fast
                    try:
                        entry_dt = datetime.fromisoformat(pos.entry_time)
                        age_secs = (datetime.now(timezone.utc) - entry_dt).total_seconds()
                        if age_secs >= SPIKE_TIMEOUT_SECS:
                            pnl = self._calc_pnl(pos, current)
                            self.db.close_trade(pos.id, current, pnl, "TIMEOUT")
                            closed.append(pos)
                            log.info(f"SPIKE CLOSED TIMEOUT ({age_secs:.0f}s): {pos.symbol} {pos.direction} "
                                     f"PnL=${pnl:+.2f}")
                    except Exception:
                        pass

                except Exception:
                    pass

            for c in closed:
                self.positions = [p for p in self.positions if p.id != c.id]

            time.sleep(1)

    def _calc_pnl(self, pos: SpikePosition, exit_price: float) -> float:
        """Calculate PnL based on fixed risk sizing, including commission."""
        if pos.direction == "BUY":
            move_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            move_pct = (pos.entry_price - exit_price) / pos.entry_price
        position_value = pos.risk_usd / (SL_PCT / 100)
        raw_pnl = move_pct * position_value
        # Deduct round-trip commission
        commission_usd = (self._commission_bps / 10000) * position_value * 2
        return round(raw_pnl - commission_usd, 2)


# ── Main ──────────────────────────────────────────────────────────────

async def run(detector: SpikeDetector):
    """Run the Alpaca websocket and spike detector."""
    from alpaca.data.live import CryptoDataStream

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    stream = CryptoDataStream(api_key, api_secret)

    async def on_orderbook(data):
        detector.on_orderbook(data)

    stream.subscribe_orderbooks(on_orderbook, *list(SYMBOLS.keys()))

    log.info(f"Subscribing to orderbooks for {len(SYMBOLS)} crypto symbols")
    log.info(f"Imbalance trigger: {IMBALANCE_TRIGGER}x, confirm: {IMBALANCE_CONFIRM_SECS}s, "
             f"price confirm: {PRICE_CONFIRM_PCT}%")
    log.info(f"Trade params: SL={SL_PCT}%, TP={TP_PCT}%, risk=${RISK_USD}")

    await stream._run_forever()


def main():
    parser = argparse.ArgumentParser(description="Orderbook Spike Detector (paper)")
    parser.add_argument("--account-id", default="ftmo_100k",
                        help="Account ID for bridge port (default: ftmo_100k)")
    args = parser.parse_args()

    # Load account config for bridge port
    import yaml
    accts_path = REPO_ROOT / "common" / "config" / "accounts.yaml"
    with open(accts_path) as f:
        accts = yaml.safe_load(f).get("accounts", {})
    acct_cfg = accts.get(args.account_id, {})
    bridge_port = acct_cfg.get("bridge_port", 5056)

    # DB in account-specific audit dir
    prefix = args.account_id.split("_")[0]
    dir_map = {"bright": "bf"}
    dir_prefix = dir_map.get(prefix, prefix)
    db_path = str(REPO_ROOT / dir_prefix / "audit" / "spike_trades.db")

    db = SpikeDB(db_path)
    detector = SpikeDetector(args.account_id, bridge_port, db)

    # Start position monitor
    detector.start_monitor()

    log.info(f"Spike detector started for {args.account_id} (bridge port {bridge_port})")
    log.info(f"DB: {db_path}")

    # Run websocket
    asyncio.run(run(detector))


if __name__ == "__main__":
    main()
