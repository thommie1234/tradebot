"""Session Guard — proactive session close using MQL5 exports + Finnhub holidays.

Data sources (in priority order):
1. MQL5 EA exports: data/sessions/{broker}_sessions.json (real broker session times)
2. Finnhub API: /stock/market-holiday (early closes, holidays)
3. Fallback CSV: data/instrument_specs/*.csv (static schedule)

The MQL5 EA (SessionExporter.mq5) runs in each MT5 terminal and writes
the actual trading session times daily. This module reads those files
and provides accurate is_trading_open() with minutes_left.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Optional: Finnhub for holiday/early close detection
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")

# Exchange mapping for Finnhub
SYMBOL_EXCHANGE = {
    "EURUSD": "forex", "GBPUSD": "forex", "USDJPY": "forex",
    "GBPCAD": "forex", "GBPAUD": "forex", "NZDUSD": "forex",
    "AUDUSD": "forex", "EURCHF": "forex", "GBPJPY": "forex",
    "FRA40.cash": "PA",    # Euronext Paris
    "EU50.cash": "PA",
    "US30.cash": "US",
    "US100.cash": "US",
    "US500.cash": "US",
    "UK100.cash": "L",     # London
    "NVDA": "US", "AAPL": "US", "MSFT": "US", "TSLA": "US",
    "AMZN": "US", "GOOG": "US", "META": "US",
    "GER40.cash": "XETR",  # Xetra
    "JP225.cash": "T",     # Tokyo
    "HK50.cash": "HK",     # Hong Kong
    "XAUUSD": "forex", "XAGUSD": "forex",
}


class SessionGuard:
    """Proactive session close guard using real broker data + Finnhub."""

    CLOSE_BUFFER_MIN = 10  # Close positions X minutes before session end

    def __init__(self, broker_tag: str = "bf", logger=None):
        self.broker_tag = broker_tag
        self.logger = logger
        self._sessions = {}       # symbol -> {day: [(open_min, close_min), ...]}
        self._gmt_offset = 0      # Server GMT offset in seconds
        self._holidays = {}       # exchange -> [{date, event, trading_hour}]
        self._holiday_cache_date = None
        self._last_load = 0

        self._load_mql5_sessions()
        self._load_finnhub_holidays()

    def _log(self, level, msg):
        if self.logger:
            self.logger.log(level, 'SessionGuard', 'SESSION_INFO', msg)
        else:
            print(f"[SessionGuard] {msg}")

    # ── MQL5 Session Loading ─────────────────────────────────────

    def _load_mql5_sessions(self):
        """Load session times from MQL5 EA export."""
        json_path = SESSIONS_DIR / f"{self.broker_tag}_sessions.json"

        # Also check MT5 common data folder
        common_paths = [
            Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files" / f"{self.broker_tag}_sessions.json",
            json_path,
        ]

        data = None
        for p in common_paths:
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    self._log('INFO', f'Loaded sessions from {p} ({len(data.get("symbols", {}))} symbols)')
                    break
                except Exception as e:
                    self._log('WARNING', f'Failed to load {p}: {e}')

        if not data:
            self._log('WARNING', f'No MQL5 session data found for {self.broker_tag}')
            return

        self._gmt_offset = data.get("gmt_offset", 0)

        day_map = {"sunday": 6, "monday": 0, "tuesday": 1, "wednesday": 2,
                   "thursday": 3, "friday": 4, "saturday": 5}

        for sym, sym_data in data.get("symbols", {}).items():
            sessions = sym_data.get("sessions", {})
            parsed = {}
            for day_name, day_sessions in sessions.items():
                day_idx = day_map.get(day_name)
                if day_idx is None:
                    continue
                parsed_sessions = []
                for session in day_sessions:
                    if len(session) == 2:
                        open_parts = session[0].split(":")
                        close_parts = session[1].split(":")
                        if len(open_parts) == 2 and len(close_parts) == 2:
                            open_min = int(open_parts[0]) * 60 + int(open_parts[1])
                            close_min = int(close_parts[0]) * 60 + int(close_parts[1])
                            parsed_sessions.append((open_min, close_min))
                if parsed_sessions:
                    parsed[day_idx] = parsed_sessions

            if parsed:
                self._sessions[sym] = parsed

        self._last_load = time.time()
        self._log('INFO', f'Parsed {len(self._sessions)} symbols from MQL5 data')

    # ── Finnhub Holiday Loading ──────────────────────────────────

    def _load_finnhub_holidays(self):
        """Load today's holidays/early closes from Finnhub."""
        if not FINNHUB_KEY or not HAS_REQUESTS:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._holiday_cache_date == today:
            return  # Already loaded today

        exchanges = set(SYMBOL_EXCHANGE.values()) - {"forex"}

        for exchange in exchanges:
            try:
                url = f"https://finnhub.io/api/v1/stock/market-holiday?exchange={exchange}&token={FINNHUB_KEY}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    holidays = []
                    for h in data.get("data", []):
                        if h.get("atDate", "") == today:
                            holidays.append({
                                "event": h.get("eventName", ""),
                                "trading_hour": h.get("tradingHour", ""),
                            })
                    if holidays:
                        self._holidays[exchange] = holidays
                        for h in holidays:
                            self._log('WARNING', f'Holiday today on {exchange}: {h["event"]} '
                                                 f'(hours: {h["trading_hour"] or "CLOSED"})')
            except Exception as e:
                self._log('WARNING', f'Finnhub error for {exchange}: {e}')

        self._holiday_cache_date = today
        if self._holidays:
            self._log('INFO', f'Loaded holidays for {len(self._holidays)} exchanges')

    def _get_holiday_close(self, symbol: str) -> int | None:
        """Get early close time in server minutes for today, or None."""
        exchange = SYMBOL_EXCHANGE.get(symbol)
        if not exchange or exchange == "forex":
            return None

        holidays = self._holidays.get(exchange, [])
        for h in holidays:
            trading_hour = h.get("trading_hour", "")
            if not trading_hour:
                return 0  # Market fully closed
            # Parse "09:30-13:00" format
            if "-" in trading_hour:
                parts = trading_hour.split("-")
                close_str = parts[-1].strip()
                close_parts = close_str.split(":")
                if len(close_parts) == 2:
                    # This is in exchange local time, need to convert to server time
                    # For now, return as-is (approximation)
                    return int(close_parts[0]) * 60 + int(close_parts[1])
        return None

    # ── Public API ───────────────────────────────────────────────

    def reload_if_needed(self):
        """Reload MQL5 data if it's been >1 hour since last load."""
        if time.time() - self._last_load > 3600:
            self._load_mql5_sessions()
            self._load_finnhub_holidays()

    def is_trading_open(self, symbol: str) -> tuple[bool, int | None]:
        """Check if symbol is currently tradeable.

        Returns:
            (is_open, minutes_left) — minutes_left is None if unknown
        """
        # Reload daily
        self.reload_if_needed()

        # Get current server time (apply GMT offset)
        now_utc = datetime.now(timezone.utc)
        server_offset = timedelta(seconds=self._gmt_offset)
        now_server = now_utc + server_offset

        day_idx = now_server.weekday()  # 0=Monday
        current_min = now_server.hour * 60 + now_server.minute

        # Check Finnhub holiday/early close first
        holiday_close = self._get_holiday_close(symbol)
        if holiday_close is not None:
            if holiday_close == 0:
                return False, None  # Market fully closed today
            if current_min >= holiday_close:
                return False, None
            # Market open but closes early
            mins_left = holiday_close - current_min
            return True, mins_left

        # Check MQL5 sessions
        sched = self._sessions.get(symbol)
        if sched:
            sessions = sched.get(day_idx)
            if not sessions:
                return False, None  # No sessions today

            for open_min, close_min in sessions:
                if open_min <= current_min < close_min:
                    return True, close_min - current_min

            return False, None  # Outside all sessions

        # No data at all — return unknown (don't block trading)
        # Weekend check for non-crypto
        if day_idx >= 5:
            return False, None

        return True, None

    def _is_daily_close(self, symbol: str, day_idx: int, current_min: int) -> bool:
        """Check if the current close is the DAILY close, not a short break.

        A short break = next session on the same day starts within 60 minutes.
        Daily close = no more sessions today, or next session is >60 min away.
        """
        sched = self._sessions.get(symbol)
        if not sched:
            return True  # No data, assume daily close

        sessions = sched.get(day_idx, [])

        # Find the current/closing session
        for i, (open_min, close_min) in enumerate(sessions):
            if current_min <= close_min + 5:  # We're in or near this session
                # Check if there's another session today after this one
                if i + 1 < len(sessions):
                    next_open = sessions[i + 1][0]
                    gap = next_open - close_min
                    if gap <= 60:
                        return False  # Short break, not daily close
                return True  # Last session of the day, or gap > 60 min

        return True  # Outside all sessions

    def should_close_position(self, symbol: str) -> tuple[bool, str]:
        """Check if a position should be closed proactively.

        Ignores short intraday breaks (< 60 min). Only triggers on
        daily close or holiday close.

        Returns:
            (should_close, reason)
        """
        is_open, mins_left = self.is_trading_open(symbol)

        now_utc = datetime.now(timezone.utc)
        server_offset = timedelta(seconds=self._gmt_offset)
        now_server = now_utc + server_offset
        day_idx = now_server.weekday()
        current_min = now_server.hour * 60 + now_server.minute

        # Market already closed — check if it's a short break or daily close
        if not is_open:
            if self._is_daily_close(symbol, day_idx, current_min):
                return True, f"{symbol} daily close — market closed"
            return False, ""  # Short break, ignore

        # Approaching close — only close if it's the daily close
        if mins_left is not None and mins_left <= self.CLOSE_BUFFER_MIN:
            if self._is_daily_close(symbol, day_idx, current_min + mins_left):
                return True, f"{symbol} daily close in {mins_left} min"
            return False, ""  # Approaching short break, ignore

        return False, ""

    def should_block_new_trade(self, symbol: str) -> tuple[bool, str]:
        """Check if new trades should be blocked (close approaching).

        Does NOT block for short intraday breaks.

        Returns:
            (should_block, reason)
        """
        is_open, mins_left = self.is_trading_open(symbol)

        now_utc = datetime.now(timezone.utc)
        server_offset = timedelta(seconds=self._gmt_offset)
        now_server = now_utc + server_offset
        day_idx = now_server.weekday()
        current_min = now_server.hour * 60 + now_server.minute

        if not is_open:
            if self._is_daily_close(symbol, day_idx, current_min):
                return True, f"{symbol} market closed (daily)"
            return False, ""  # Short break

        if mins_left is not None and mins_left <= self.CLOSE_BUFFER_MIN:
            if self._is_daily_close(symbol, day_idx, current_min + mins_left):
                return True, f"{symbol} daily close in {mins_left} min"
            return False, ""  # Short break approaching

        # Check Finnhub: is today a holiday?
        holiday_close = self._get_holiday_close(symbol)
        if holiday_close == 0:
            return True, f"{symbol} market closed (holiday)"

        return False, ""

    def get_session_info(self, symbol: str) -> dict:
        """Get full session info for a symbol (for debugging)."""
        sched = self._sessions.get(symbol, {})
        holiday = self._get_holiday_close(symbol)
        is_open, mins_left = self.is_trading_open(symbol)

        return {
            "symbol": symbol,
            "has_mql5_data": symbol in self._sessions,
            "is_open": is_open,
            "minutes_left": mins_left,
            "holiday_close_min": holiday,
            "sessions": {str(k): v for k, v in sched.items()},
        }


def print_all_sessions(broker_tag: str = "bf"):
    """Debug: print all loaded sessions."""
    sg = SessionGuard(broker_tag)
    for sym in sorted(sg._sessions.keys()):
        info = sg.get_session_info(sym)
        print(f"\n{sym}: open={info['is_open']}, mins_left={info['minutes_left']}")
        for day, sessions in sorted(info['sessions'].items()):
            day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            for o, c in sessions:
                print(f"  {day_names.get(int(day), day)}: {o//60:02d}:{o%60:02d} - {c//60:02d}:{c%60:02d}")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "bf"
    print_all_sessions(tag)
