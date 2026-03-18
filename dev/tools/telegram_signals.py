"""Telegram Signal Scraper — monitors channels for trade signals and executes them.

Parses BUY/SELL signals from Telegram channels, ignores their SL/TP,
and opens trades using the bot's own ATR-based risk management.

Usage:
    python telegram_signals.py --channels "@thefxculture,@another_channel"
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient, events


def _safe_print(msg: str):
    """Print with emoji-safe encoding fallback."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())

# Ensure project root on path
_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

# ── Config ──────────────────────────────────────────────────────

API_ID = 33475248
API_HASH = "361bef592b485c2d489fa2ff2d9d6193"
SESSION_FILE = str(REPO_ROOT / "tools" / "telegram_session")
SIGNAL_DB = str(REPO_ROOT / "bf" / "audit" / "telegram_signals.db")

# Default channels to monitor — top 3 from backtest
DEFAULT_CHANNELS = [
    2505604296,         # FXNL - Official
    1848239554,         # Hobie's Trading Room
    2399120063,         # FX Culture | Free Signal Group
]

# Symbol name mapping: Telegram format → MT5 broker format
# Includes abbreviations, Dutch names, slash variants, and common nicknames
SYMBOL_MAP = {
    # Forex — full names + slash + abbreviations
    "EURUSD": "EURUSD", "EUR/USD": "EURUSD", "EU": "EURUSD", "FIBER": "EURUSD",
    "GBPUSD": "GBPUSD", "GBP/USD": "GBPUSD", "GU": "GBPUSD", "CABLE": "GBPUSD",
    "USDJPY": "USDJPY", "USD/JPY": "USDJPY", "UJ": "USDJPY", "GOPHER": "USDJPY",
    "GBPCAD": "GBPCAD", "GBP/CAD": "GBPCAD", "GC": "GBPCAD",
    "GBPAUD": "GBPAUD", "GBP/AUD": "GBPAUD", "GA": "GBPAUD",
    "NZDUSD": "NZDUSD", "NZD/USD": "NZDUSD", "NU": "NZDUSD", "KIWI": "NZDUSD",
    "AUDUSD": "AUDUSD", "AUD/USD": "AUDUSD", "AU": "AUDUSD", "AUSSIE": "AUDUSD",
    "USDCHF": "USDCHF", "USD/CHF": "USDCHF", "UCHF": "USDCHF", "SWISSY": "USDCHF",
    "USDCAD": "USDCAD", "USD/CAD": "USDCAD", "UC": "USDCAD", "LOONIE": "USDCAD",
    "EURCHF": "EURCHF", "EUR/CHF": "EURCHF", "ECHF": "EURCHF",
    "EURJPY": "EURJPY", "EUR/JPY": "EURJPY", "EJ": "EURJPY",
    "GBPJPY": "GBPJPY", "GBP/JPY": "GBPJPY", "GJ": "GBPJPY", "GUPPY": "GBPJPY", "GEPPY": "GBPJPY",
    "EURGBP": "EURGBP", "EUR/GBP": "EURGBP", "EG": "EURGBP",
    "AUDCAD": "AUDCAD", "AUD/CAD": "AUDCAD", "AC": "AUDCAD",
    "AUDNZD": "AUDNZD", "AUD/NZD": "AUDNZD", "AN": "AUDNZD",
    "AUDJPY": "AUDJPY", "AUD/JPY": "AUDJPY", "AJ": "AUDJPY",
    "NZDJPY": "NZDJPY", "NZD/JPY": "NZDJPY", "NJ": "NZDJPY",
    "NZDCAD": "NZDCAD", "NZD/CAD": "NZDCAD", "NC": "NZDCAD",
    "CADJPY": "CADJPY", "CAD/JPY": "CADJPY", "CJ": "CADJPY",
    "CHFJPY": "CHFJPY", "CHF/JPY": "CHFJPY",
    "CADCHF": "CADCHF", "CAD/CHF": "CADCHF",
    "GBPNZD": "GBPNZD", "GBP/NZD": "GBPNZD", "GN": "GBPNZD",
    "GBPCHF": "GBPCHF", "GBP/CHF": "GBPCHF",
    "EURAUD": "EURAUD", "EUR/AUD": "EURAUD", "EA": "EURAUD",
    "EURCAD": "EURCAD", "EUR/CAD": "EURCAD", "EC": "EURCAD",
    "EURNZD": "EURNZD", "EUR/NZD": "EURNZD",
    "AUDCHF": "AUDCHF", "AUD/CHF": "AUDCHF",
    "NZDCHF": "NZDCHF", "NZD/CHF": "NZDCHF",
    # Commodities / metals
    "XAUUSD": "XAUUSD", "XAU/USD": "XAUUSD", "GOLD": "XAUUSD", "GOUD": "XAUUSD",
    "XAGUSD": "XAGUSD", "XAG/USD": "XAGUSD", "SILVER": "XAGUSD", "ZILVER": "XAGUSD",
    "XTIUSD": "USOIL.cash", "WTI": "USOIL.cash", "CRUDE": "USOIL.cash", "OLIE": "USOIL.cash",
    "USOIL": "USOIL.cash", "OIL": "USOIL.cash",
    # Indices
    "US100": "US100.cash", "NAS100": "US100.cash", "NASDAQ": "US100.cash", "NDX": "US100.cash", "NQ": "US100.cash", "USTEC": "US100.cash",
    "US30": "US30.cash", "DJ30": "US30.cash", "DOW": "US30.cash", "DJIA": "US30.cash", "DJI": "US30.cash",
    "US500": "US500.cash", "SPX500": "US500.cash", "SP500": "US500.cash", "SPX": "US500.cash",
    "FRA40": "FRA40.cash", "FR40": "FRA40.cash", "CAC40": "FRA40.cash", "CAC": "FRA40.cash",
    "UK100": "UK100.cash", "FTSE": "UK100.cash", "FTSE100": "UK100.cash",
    "GER40": "GER40.cash", "DAX": "GER40.cash", "DE40": "GER40.cash", "GER30": "GER40.cash", "DE30": "GER40.cash",
    "JP225": "JP225.cash", "NIKKEI": "JP225.cash", "NI225": "JP225.cash",
    # Crypto
    "BTCUSD": "BTC_USD", "BTC/USD": "BTC_USD", "BITCOIN": "BTC_USD", "BTC": "BTC_USD",
    "ETHUSD": "ETH_USD", "ETH/USD": "ETH_USD", "ETHEREUM": "ETH_USD", "ETH": "ETH_USD",
}


@dataclass
class Signal:
    """Parsed trading signal."""
    symbol: str           # MT5 symbol name
    direction: str        # BUY or SELL
    entry_price: float    # Entry price (midpoint if zone)
    entry_low: float      # Entry zone low
    entry_high: float     # Entry zone high
    raw_text: str         # Original message text
    channel: str          # Source channel
    timestamp: str        # ISO timestamp
    sl_price: float | None = None
    tp_price: float | None = None   # TP1 — breakeven trigger
    tp2_price: float | None = None  # TP2 — trail activation


# ── Signal Parser ───────────────────────────────────────────────

def parse_signal(text: str, channel: str = "") -> Signal | None:
    """Parse a trading signal from message text.

    Handles many formats used by Dutch and international signal channels:
      BUY EURUSD 1.0850-1.0860 SL: 1.0800 TP: 1.0950
      🟢 BUY TRADE XAU/USD 2350-2355 SL: 2330 Target: open
      SELL GBP/USD @ 1.2650 SL 1.2700 TP 1.2550
      📈 LONG GJ 191.50 SL 190.80
      🔴 SHORT EU @ 1.0850
      KOOP GOUD 2350-2355
      GU SELL 1.2650
    """
    if not text:
        return None

    text_upper = text.upper().strip()

    # Skip non-signal messages (analysis, news, chat, summaries, promos)
    skip_patterns = [
        r'\bANALYSIS\b', r'\bANALYSE\b', r'\bUPDATE\b', r'\bNIEUWS\b',
        r'\bRESULT\b', r'\bCLOSED\b', r'\bGESLOTEN\b', r'\bHIT TP\b',
        r'\bHIT SL\b', r'\bBREAKEVEN\b', r'\bBE HIT\b',
        r'\bDON\'T\b', r'\bDONT\b', r'\bAVOID\b', r'\bSKIP\b',
        r'\bNIET\b.*\b(?:BUY|SELL|KOOP|VERKOOP)\b',
        r'\bWEEKLY SUMMARY\b', r'\bOVERVIEW\b', r'\bSUMMARY\b',
        r'\bGOOD MORNING\b', r'\bGOEDEMORGEN\b',
        r'\bCLAIM\b', r'\bSIGN UP\b', r'\bJOIN\b.*\bVIP\b',
        r'\bMARKET\s+OVERVIEW\b', r'\bKEY\s+INSIGHTS\b',
        r'\bCOULD\s+DROP\b', r'\bMIGHT\b', r'\bCOULD\b.*\bIF\b',
        r'\bWANT\s+MY\b', r'\bNEW\s+SIGNAL!!\b',
        r'\bLIVE\s+TRADING\s+STREAM\b', r'\bLIVE\s+STREAM\b',
        r'\bTODAY\s+WE\s+WILL\b', r'\bTHESE\s+SIGNALS\b',
        r'\bJUST\s+HOW\s+GOOD\b', r'\bFULL\s+TP\b',
        r'\bGET\s+ACC\b', r'\bLOOSE\s+THEIR\b', r'\bLOSE\s+THEIR\b',
        r'\bFORGETTING\b', r'\bZOMBIES\b', r'\bDEPOSITS?\b',
        r'\bPUBLIC\s+SIGNAL\b', r'\bVIP\s+SIGNAL\b',
        r'\bWE\'RE\s+NOT\s+DONE\b', r'\bA\+\s+SETUP\s+IS\s+LOADING\b',
        r'\bTHOUSANDS\s+OF\s+TRADERS\b',
        r'\bENTER\s+IN\s+LINK\b', r'\bSHORT\s+\d+X\b', r'\bLONG\s+\d+X\b',
        r'\bTP\d?\s*\+\s*\d+\s*PIPS\b', r'\bTP\d?\s+HIT\b',
        r'\bLETSSS\b', r'\bLETS\s+GO\b', r'\bPRECISION\b',
        r'\bWHAT\s+COULD\s+A\b', r'\bMAKE\s+THIS\s+WEEK\b',
        r'\bWIN\s+RATE\b', r'\bI\'LL\s+TAKE\b',
        r'\bADDING\s+\d+\s+TRADERS\b', r'\bFREE\s+ACCESS\b',
        r'\bMADE\s+\+\d+\s+POINTS\b', r'\bLAST\s+WEEK\b',
        r'\bYESTERDAY\s+WE\s+HIT\b', r'\bPULLED\s+OFF\b',
        r'\bHIGHLY\s+RECOMMEND\b', r'\bLET\'S\s+HEAR\b',
        r'\bUS.IRAN\b', r'\bCONFLICT\b',
        r'\bLET\'S\s+GET\s+STARTED\b', r'\bCHART\'S\s+TELLING\b',
        r'\bVERDIENDE\b', r'\bPREMIUM\b', r'\bBERICHT\b',
        r'\bNEW\s+SIGNAL\s+UPLOADED\b',
    ]
    for pat in skip_patterns:
        if re.search(pat, text_upper):
            return None

    # Find direction — support many languages/styles
    direction = None
    buy_patterns = [r'\bBUY\b', r'\bLONG\b', r'\bKOOP\b', r'\bBUYS?\b', r'🟢', r'📈', r'🔵']
    sell_patterns = [r'\bSELL\b', r'\bSHORT\b', r'\bVERKOOP\b', r'\bSELLS?\b', r'🔴', r'📉', r'🔻']

    for pat in buy_patterns:
        if re.search(pat, text_upper if '\\b' in pat else text):
            direction = "BUY"
            break
    if not direction:
        for pat in sell_patterns:
            if re.search(pat, text_upper if '\\b' in pat else text):
                direction = "SELL"
                break
    if not direction:
        return None

    # Find symbol — try longest match first to avoid "EU" matching before "EURUSD"
    symbol = None
    sorted_names = sorted(SYMBOL_MAP.keys(), key=len, reverse=True)
    for name in sorted_names:
        # For 2-letter abbreviations, require word boundary
        if len(name) <= 2:
            pattern = r'\b' + re.escape(name) + r'\b'
        else:
            pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_upper):
            symbol = SYMBOL_MAP[name]
            break

    if not symbol:
        return None

    # Find entry price(s) — multiple strategies

    # Strategy 1: Look for explicit ENTRY keyword with price
    entry_match = re.search(r'(?:ENTRY|ENTRE|INSTAP)[:\s]*(\d+\.?\d*)', text_upper)
    if entry_match:
        entry_price = float(entry_match.group(1))
        # Check for entry range: "ENTRY: 5182.5" or "ENTRY 5090-5095"
        range_match = re.search(r'(?:ENTRY|ENTRE|INSTAP)[:\s]*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', text_upper)
        if range_match:
            entry_low = float(range_match.group(1))
            entry_high = float(range_match.group(2))
            entry_price = (entry_low + entry_high) / 2
        else:
            entry_low = entry_high = entry_price
    else:
        # Strategy 2: Look for zone pattern after direction+symbol
        # e.g. "BUY XAUUSD 5170-5173" or "SELL EURUSD\n1.0850-1.0860"
        zone_match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', text)
        if zone_match:
            p1 = float(zone_match.group(1))
            p2 = float(zone_match.group(2))
            # Verify both are plausible prices (within 5% of each other)
            if p1 > 0 and p2 > 0:
                ratio = max(p1, p2) / min(p1, p2)
                if ratio < 1.05:
                    entry_low = min(p1, p2)
                    entry_high = max(p1, p2)
                    entry_price = (entry_low + entry_high) / 2
                else:
                    entry_low = entry_high = entry_price = p1
            else:
                entry_low = entry_high = entry_price = p1
        else:
            # Strategy 3: Look for @ price
            at_match = re.search(r'@\s*(\d+\.?\d*)', text)
            if at_match:
                entry_price = float(at_match.group(1))
                entry_low = entry_high = entry_price
            else:
                # Strategy 4: First price after symbol name (skip SL/TP labeled prices)
                # Remove SL/TP sections first
                clean = re.sub(r'(?:SL|STOP\s*LOSS|TP\d?|TAKE\s*PROFIT|TARGET|DOEL)[:\s]*\d+\.?\d*', '', text_upper)
                prices = re.findall(r'(\d+\.?\d*)', clean)
                prices = [float(p) for p in prices if 0.5 < float(p) < 1000000]
                if not prices:
                    return None
                entry_low = entry_high = entry_price = prices[0]

    if entry_price <= 0:
        return None

    # Sanity check: entry price must be realistic for the symbol
    price_ranges = {
        "XAUUSD": (1000, 10000), "XAGUSD": (10, 100),
        "BTC_USD": (10000, 200000), "ETH_USD": (500, 10000),
        "EURUSD": (0.8, 1.5), "GBPUSD": (1.0, 1.8), "USDJPY": (80, 200),
        "GBPCAD": (1.4, 2.2), "GBPAUD": (1.5, 2.5), "NZDUSD": (0.4, 0.9),
        "AUDUSD": (0.5, 0.9), "USDCHF": (0.7, 1.2), "USDCAD": (1.1, 1.6),
        "EURCHF": (0.8, 1.2), "EURJPY": (100, 200), "GBPJPY": (130, 250),
        "EURGBP": (0.7, 1.0), "AUDNZD": (1.0, 1.3), "AUDCAD": (0.8, 1.1),
        "AUDJPY": (70, 120), "NZDJPY": (70, 110), "CADJPY": (80, 130),
        "CHFJPY": (120, 200), "NZDCAD": (0.7, 1.0),
        "GBPNZD": (1.8, 2.5), "GBPCHF": (1.0, 1.4),
        "EURAUD": (1.4, 2.0), "EURCAD": (1.3, 1.7), "EURNZD": (1.6, 2.1),
        "US100.cash": (10000, 25000),
        "US30.cash": (25000, 50000), "FRA40.cash": (5000, 10000),
        "USOIL.cash": (40, 120),
    }
    if symbol in price_ranges:
        lo, hi = price_ranges[symbol]
        if entry_price < lo or entry_price > hi:
            return None

    # Try to find SL (optional, for logging)
    sl_price = None
    sl_match = re.search(r'(?:SL|STOP\s*LOSS|STOPLOSS)[:\s]*(\d+\.?\d*)', text_upper)
    if sl_match:
        sl_price = float(sl_match.group(1))

    # Find TP1, TP2 from signal (used for BE and trail activation)
    tp_price = None
    tp2_price = None
    tp_matches = re.findall(r'(?:TP\s*\d?|TAKE\s*PROFIT|TARGET|DOEL)[:\s]*(\d+\.?\d*)', text_upper)
    if tp_matches:
        tp_prices = [float(p) for p in tp_matches]
        tp_price = tp_prices[0] if len(tp_prices) >= 1 else None
        tp2_price = tp_prices[1] if len(tp_prices) >= 2 else None

    return Signal(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        entry_low=entry_low,
        entry_high=entry_high,
        raw_text=text[:500],
        channel=channel,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sl_price=sl_price,
        tp_price=tp_price,
        tp2_price=tp2_price,
    )


# ── Signal Database ─────────────────────────────────────────────

def _init_signal_db(db_path: str):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            channel TEXT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            entry_low REAL,
            entry_high REAL,
            sl_price REAL,
            tp_price REAL,
            raw_text TEXT,
            executed INTEGER DEFAULT 0,
            execution_result TEXT,
            mt5_ticket INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _save_signal(db_path: str, signal: Signal, executed: bool = False,
                 result: str = "", ticket: int = 0):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("""
        INSERT INTO signals (timestamp, channel, symbol, direction, entry_price,
                           entry_low, entry_high, sl_price, tp_price, raw_text,
                           executed, execution_result, mt5_ticket)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (signal.timestamp, signal.channel, signal.symbol, signal.direction,
          signal.entry_price, signal.entry_low, signal.entry_high,
          signal.sl_price, signal.tp_price, signal.raw_text,
          1 if executed else 0, result, ticket))
    conn.commit()
    conn.close()


# ── MT5 Trade Execution ────────────────────────────────────────

# Track recent signals to prevent duplicates
_recent_signals: dict[str, float] = {}  # "SYMBOL_DIR" -> timestamp
_SIGNAL_COOLDOWN = 1800  # 30 minutes between same symbol+direction


def execute_signal(signal: Signal, account_id: str = "bright_100k") -> tuple[bool, str]:
    """Execute a signal using the bot's existing infrastructure.

    Only uses the direction and symbol from the signal.
    SL, TP, position sizing, trailing — all handled by the bot's own logic.

    Safeguards:
    - No duplicate: skip if same symbol+direction within 30 min
    - No stack: skip if already have open position on this symbol
    - No hedge: skip if open position in opposite direction
    """
    try:
        from tools.mt5_bridge import mt5
        from config.loader import cfg, load_config
        from execution.broker_api import get_symbol_info

        # Safeguard 1: Duplicate cooldown — per account
        sig_key = f"{account_id}_{signal.symbol}_{signal.direction}"
        now = time.time()
        last_time = _recent_signals.get(sig_key, 0)
        if now - last_time < _SIGNAL_COOLDOWN:
            mins_ago = (now - last_time) / 60
            return False, f"Duplicate: {signal.symbol} {signal.direction} already traded {mins_ago:.0f}min ago"

        # Safeguard 2: Check for existing positions on this symbol
        mt5.symbol_select(signal.symbol, True)
        positions = mt5.positions_get(symbol=signal.symbol)
        if positions and len(positions) > 0:
            for p in positions:
                p_dir = "BUY" if p.type == 0 else "SELL"
                # Check if existing position is at breakeven or better
                if p_dir == signal.direction:
                    if signal.direction == "BUY":
                        at_be = p.sl >= p.price_open
                    else:
                        at_be = p.sl <= p.price_open and p.sl > 0
                    if not at_be:
                        return False, f"Already have {signal.direction} on {signal.symbol} (not yet at BE)"
                    # At BE — allow new trade
                else:
                    return False, f"Opposing position open on {signal.symbol} ({p_dir}), skipping hedge"

        # Default params per asset class — used for any symbol not in config
        # Signal trades: tighter TP + faster breakeven than ML trades
        # These are scalp signals — need to take profit quicker
        _defaults = {
            "forex": {"atr_timeframe": "M30", "atr_period": 14, "atr_sl_mult": 1.0,
                      "atr_tp_mult": 1.5, "breakeven_atr": 0.15, "trail_activation_atr": 0.4,
                      "trail_distance_atr": 0.25, "risk_per_trade": 0.003, "magic_number": 5000},
            "gold": {"atr_timeframe": "M15", "atr_period": 14, "atr_sl_mult": 0.8,
                     "atr_tp_mult": 1.5, "breakeven_atr": 0.15, "trail_activation_atr": 0.35,
                     "trail_distance_atr": 0.20, "risk_per_trade": 0.002, "magic_number": 5001},
            "index": {"atr_timeframe": "M30", "atr_period": 14, "atr_sl_mult": 1.0,
                      "atr_tp_mult": 1.5, "breakeven_atr": 0.15, "trail_activation_atr": 0.4,
                      "trail_distance_atr": 0.25, "risk_per_trade": 0.002, "magic_number": 5002},
            "crypto": {"atr_timeframe": "H1", "atr_period": 14, "atr_sl_mult": 0.8,
                       "atr_tp_mult": 2.0, "breakeven_atr": 0.3, "trail_activation_atr": 0.8,
                       "trail_distance_atr": 0.5, "risk_per_trade": 0.002, "magic_number": 5003},
        }

        # Detect asset class from symbol
        def _asset_class(sym):
            if sym in ("XAUUSD", "XAGUSD"):
                return "gold"
            if sym.endswith(".cash"):
                return "index"
            if sym.endswith("_USD") and sym[:3] not in ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF"):
                return "crypto"
            return "forex"

        # Try config first, fallback to defaults
        try:
            if not hasattr(cfg, 'SYMBOLS') or not cfg.SYMBOLS:
                load_config()
        except Exception:
            pass

        sym_cfg = None
        if hasattr(cfg, 'SYMBOLS'):
            sym_cfg = cfg.SYMBOLS.get(signal.symbol)
        if not sym_cfg:
            sym_cfg = _defaults[_asset_class(signal.symbol)]

        # Check if market is open
        tick = mt5.symbol_info_tick(signal.symbol)
        if not tick:
            mt5.symbol_select(signal.symbol, True)
            tick = mt5.symbol_info_tick(signal.symbol)
        if not tick:
            return False, f"No tick data for {signal.symbol}"

        # Get current price
        if signal.direction == "BUY":
            entry_price = tick.ask
        else:
            entry_price = tick.bid

        # Check if current price is within entry zone (allow 0.5% tolerance)
        if signal.entry_low > 0 and signal.entry_high > 0:
            tolerance = entry_price * 0.005
            if entry_price < signal.entry_low - tolerance or entry_price > signal.entry_high + tolerance:
                return False, f"Price {entry_price:.5f} outside entry zone {signal.entry_low:.5f}-{signal.entry_high:.5f}"

        # Calculate ATR-based SL
        sym_info = get_symbol_info(signal.symbol)
        if not sym_info:
            return False, f"No symbol info for {signal.symbol}"

        # Get ATR
        tf_map = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                  "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
                  "H4": mt5.TIMEFRAME_H4}
        atr_tf = sym_cfg.get("atr_timeframe", "H1")
        mt5_tf = tf_map.get(atr_tf, mt5.TIMEFRAME_H1)
        atr_period = sym_cfg.get("atr_period", 14)
        bars = mt5.copy_rates_from_pos(signal.symbol, mt5_tf, 0, atr_period + 5)

        if bars is None or len(bars) < atr_period:
            return False, f"Not enough bars for ATR calculation"

        import numpy as np
        highs = np.array([b[2] for b in bars[-atr_period:]])
        lows = np.array([b[3] for b in bars[-atr_period:]])
        closes = np.array([b[4] for b in bars[-atr_period:]])
        tr = np.maximum(highs - lows, np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1))
        ))[1:]
        atr = float(np.mean(tr))

        sl_mult = sym_cfg.get("atr_sl_mult", 1.0)
        sl_distance = atr * sl_mult

        if signal.direction == "BUY":
            sl_price = entry_price - sl_distance
        else:
            sl_price = entry_price + sl_distance

        # Position sizing
        account_info = mt5.account_info()
        if not account_info:
            return False, "No account info"

        risk_pct = sym_cfg.get("risk_per_trade", 0.003)
        tick_value = sym_info.get("trade_tick_value", 0)
        tick_size = sym_info.get("trade_tick_size", 0)

        if tick_value > 0 and tick_size > 0:
            risk_per_lot = (sl_distance / tick_size) * tick_value
        else:
            return False, "tick_value is 0"

        if risk_per_lot <= 0:
            return False, "risk_per_lot <= 0"

        lot_size = (account_info.equity * risk_pct) / risk_per_lot
        vol_step = sym_info.get("volume_step", 0.01)
        vol_min = sym_info.get("volume_min", 0.01)
        vol_max = sym_info.get("volume_max", 100)
        if vol_step > 0:
            lot_size = round(lot_size / vol_step) * vol_step
        lot_size = max(vol_min, min(vol_max, lot_size))

        # TP — set very far (10x ATR), trail will exit long before TP is hit
        tp_mult = 10.0

        # If signal has TP1, use it to set a smarter SL
        # SL = 2x the TP1 distance (so risk:reward starts at 1:0.5, but BE kicks in fast)
        if signal.tp_price:
            if signal.direction == "BUY" and signal.tp_price > entry_price:
                tp1_dist = signal.tp_price - entry_price
                sl_distance = min(sl_distance, tp1_dist * 2.5)
                sl_price = entry_price - sl_distance
            elif signal.direction == "SELL" and signal.tp_price < entry_price:
                tp1_dist = entry_price - signal.tp_price
                sl_distance = min(sl_distance, tp1_dist * 2.5)
                sl_price = entry_price + sl_distance

        if signal.direction == "BUY":
            tp_price = entry_price + atr * tp_mult
        else:
            tp_price = entry_price - atr * tp_mult

        # Place order
        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
        magic = sym_cfg.get("magic_number", 9000)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": round(lot_size, 8),
            "type": order_type,
            "price": entry_price,
            "sl": round(sl_price, sym_info.get("digits", 5)),
            "tp": round(tp_price, sym_info.get("digits", 5)),
            "deviation": 20,
            "magic": magic,
            "comment": f"Sovereign_{signal.symbol}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            _recent_signals[sig_key] = time.time()
            return True, f"Executed: ticket={result.order}, lots={lot_size:.2f}, SL={sl_price:.5f}, TP={tp_price:.5f}"
        else:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "No result"
            return False, f"Order failed: {retcode} — {comment}"

    except Exception as e:
        return False, f"Error: {e}"


# ── Telegram Client ─────────────────────────────────────────────

async def run_scraper(channels: list = None, account_id: str = "bright_100k",
                      dry_run: bool = False):
    """Main scraper loop — monitors Telegram channels for signals."""
    if channels is None:
        channels = DEFAULT_CHANNELS

    _init_signal_db(SIGNAL_DB)

    print(f"[TELEGRAM] Starting signal scraper")
    print(f"[TELEGRAM] Channels: {channels}")
    print(f"[TELEGRAM] Account: {account_id}")
    print(f"[TELEGRAM] Dry run: {dry_run}")
    print()

    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    await client.start()

    me = await client.get_me()
    print(f"[TELEGRAM] Logged in as: {me.first_name} ({me.phone})")

    # Resolve channel entities — build lookup from user's dialogs first
    dialog_map = {}  # id -> entity
    async for dialog in client.iter_dialogs():
        dialog_map[dialog.entity.id] = dialog.entity

    monitored = []
    for ch in channels:
        found = False
        if isinstance(ch, int):
            # Try matching against dialog IDs
            for did, entity in dialog_map.items():
                if did == ch or did == abs(ch):
                    monitored.append(did)
                    title = getattr(entity, 'title', None) or str(did)
                    _safe_print(f"[TELEGRAM] Monitoring: {title} (id={did})")
                    found = True
                    break
            if not found:
                # Fallback: try get_entity with variations
                for try_id in [ch, -ch, int(f"-100{abs(ch)}"), abs(ch)]:
                    try:
                        entity = await client.get_entity(try_id)
                        monitored.append(entity.id)
                        title = getattr(entity, 'title', None) or str(try_id)
                        _safe_print(f"[TELEGRAM] Monitoring: {title} (id={entity.id})")
                        found = True
                        break
                    except Exception:
                        continue
            if not found:
                _safe_print(f"[TELEGRAM] WARNING: Cannot find channel id={ch}")
        else:
            try:
                entity = await client.get_entity(ch)
                monitored.append(entity.id)
                title = getattr(entity, 'title', None) or ch
                _safe_print(f"[TELEGRAM] Monitoring: {title} (id={entity.id})")
            except Exception as e:
                _safe_print(f"[TELEGRAM] WARNING: Cannot find channel '{ch}': {e}")

    if not monitored:
        print("[TELEGRAM] ERROR: No channels found to monitor")
        return

    # Initialize MT5 if not dry run
    if not dry_run:
        from tools.mt5_bridge import mt5, initialize_mt5
        ok, err, mode = initialize_mt5()
        if ok:
            info = mt5.account_info()
            print(f"[TELEGRAM] MT5 connected: {info.login} | Balance: ${info.balance:,.2f}")
        else:
            print(f"[TELEGRAM] WARNING: MT5 not connected: {err}")
            print("[TELEGRAM] Signals will be logged but NOT executed")
            dry_run = True

    print()
    print("[TELEGRAM] Listening for signals... (Press Ctrl+C to stop)")
    print("=" * 60)

    @client.on(events.NewMessage(chats=monitored))
    async def handler(event):
        text = event.raw_text
        channel_name = ""
        try:
            chat = await event.get_chat()
            channel_name = getattr(chat, 'title', '') or getattr(chat, 'username', '') or str(chat.id)
        except Exception:
            channel_name = str(event.chat_id)

        signal = parse_signal(text, channel_name)

        if signal:
            _safe_print(f"\n[SIGNAL] {signal.direction} {signal.symbol}")
            _safe_print(f"  Entry: {signal.entry_low:.5f} - {signal.entry_high:.5f}")
            _safe_print(f"  Channel: {channel_name}")
            if signal.sl_price:
                _safe_print(f"  SL (ignored): {signal.sl_price}")
            if signal.tp_price:
                _safe_print(f"  TP (ignored): {signal.tp_price}")

            if dry_run:
                _safe_print(f"  [DRY RUN] Would execute {signal.direction} {signal.symbol}")
                _save_signal(SIGNAL_DB, signal, executed=False, result="DRY_RUN")
            else:
                # Random delay 5-30 seconds to avoid pattern detection
                import random
                delay = random.uniform(5, 30)
                _safe_print(f"  [DELAY] Waiting {delay:.0f}s before execution...")
                await asyncio.sleep(delay)

                # Execute on all configured accounts
                for acct in (account_id if isinstance(account_id, list) else [account_id]):
                    ok, result = execute_signal(signal, acct)
                    acct_label = "BF" if "bright" in acct else "FTMO"
                    _safe_print(f"  [{acct_label}] [{'OK' if ok else 'FAIL'}] {result}")
                    _save_signal(SIGNAL_DB, signal, executed=ok,
                                result=f"[{acct_label}] {result}")

    await client.run_until_disconnected()


# ── CLI ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Telegram Signal Scraper")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated channel names/IDs (default: preconfigured channels)")
    parser.add_argument("--account", type=str, default="bright_100k,ftmo_100k",
                        help="Comma-separated account IDs (default: bright_100k,ftmo_100k)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse signals but don't execute trades")
    parser.add_argument("--db", type=str, default=None,
                        help="Signal database path")
    args = parser.parse_args()

    if args.db:
        global SIGNAL_DB
        SIGNAL_DB = args.db

    if args.channels:
        channels = []
        for c in args.channels.split(","):
            c = c.strip()
            if c:
                try:
                    channels.append(int(c))
                except ValueError:
                    channels.append(c)
    else:
        channels = None  # Use DEFAULT_CHANNELS
    accounts = [a.strip() for a in args.account.split(",") if a.strip()]
    account_arg = accounts if len(accounts) > 1 else accounts[0]
    asyncio.run(run_scraper(channels, account_arg, args.dry_run))


if __name__ == "__main__":
    main()
