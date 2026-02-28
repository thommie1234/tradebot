"""
FTMO compliance guard — market hours, Friday close, daily loss, total DD.

Extracted from TradingSchedule + FTMO checks in sovereign_bot.py.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone

from config.loader import cfg


class TradingSchedule:
    """Per-instrument trading hours loaded from FTMO CSV files."""

    GMT2_OFFSET = timedelta(hours=2)
    FRIDAY_CLOSE_BUFFER_MIN = 10

    def __init__(self, info_dir=None, override_csv=None):
        self.info_dir = info_dir or cfg.INFO_DIR
        self.schedule = {}
        self.crypto_symbols = set()
        self._load_all()
        if override_csv:
            self._load_override_csv(override_csv)

    @staticmethod
    def _normalize_symbol(csv_sym):
        return csv_sym.strip().replace("/", "_")

    @staticmethod
    def _parse_time_parts(t_str):
        result = []
        for p in t_str.strip().split(";"):
            p = p.strip()
            if ":" in p:
                h, m = p.split(":")
                result.append(int(h) * 60 + int(m))
        return result

    def _load_csv(self, path, is_crypto=False):
        if not os.path.exists(path):
            return

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if not row:
                    continue
                sym = self._normalize_symbol(row[0])
                if not sym:
                    continue

                if is_crypto:
                    self.crypto_symbols.add(sym)

                tz_idx = None
                for i, val in enumerate(row):
                    if val.strip() == "GMT+2":
                        tz_idx = i
                        break
                if tz_idx is None:
                    continue

                tv = [v.strip() for v in row[tz_idx + 1:]]

                day_map = {}
                for day_idx in range(5):
                    oi = day_idx * 2
                    ci = oi + 1
                    if ci >= len(tv):
                        break
                    open_val, close_val = tv[oi], tv[ci]
                    if not open_val or not close_val:
                        continue
                    if "closed" in open_val.lower() or "closed" in close_val.lower():
                        continue
                    opens = self._parse_time_parts(open_val)
                    closes = self._parse_time_parts(close_val)
                    sessions = list(zip(opens, closes))
                    if sessions:
                        day_map[day_idx] = sessions

                if is_crypto:
                    for day_idx, base in [(5, 10), (6, 12)]:
                        if base + 1 >= len(tv):
                            break
                        open_val, close_val = tv[base], tv[base + 1]
                        if not open_val or not close_val:
                            continue
                        if "closed" in open_val.lower() or "closed" in close_val.lower():
                            continue
                        opens = self._parse_time_parts(open_val)
                        closes = self._parse_time_parts(close_val)
                        sessions = list(zip(opens, closes))
                        if sessions:
                            day_map[day_idx] = sessions

                if day_map:
                    self.schedule[sym] = day_map

    def _load_all(self):
        csv_files = [
            ("Forex.csv", False),
            ("forex exotic.csv", False),
            ("metals.csv", False),
            ("cash.csv", False),
            ("Equities.csv", False),
            ("crypto.csv", True),
        ]
        for filename, is_crypto in csv_files:
            self._load_csv(os.path.join(self.info_dir, filename), is_crypto)
        print(f"[SCHEDULE] Loaded trading hours for {len(self.schedule)} instruments "
              f"({len(self.crypto_symbols)} crypto)")

    def _load_override_csv(self, path):
        """Load a simple session CSV (bf_sessions.csv format) and override schedule."""
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                sym = row.get("symbol", "").strip()
                if not sym:
                    continue
                day_map = {}
                day_cols = [
                    ("mon_open", "mon_close", 0),
                    ("tue_open", "tue_close", 1),
                    ("wed_open", "wed_close", 2),
                    ("thu_open", "thu_close", 3),
                    ("fri_open", "fri_close", 4),
                ]
                for open_col, close_col, day_idx in day_cols:
                    o_str = row.get(open_col, "").strip()
                    c_str = row.get(close_col, "").strip()
                    if not o_str or not c_str or "closed" in o_str.lower():
                        continue
                    o_parts = o_str.split(":")
                    c_parts = c_str.split(":")
                    if len(o_parts) == 2 and len(c_parts) == 2:
                        o_min = int(o_parts[0]) * 60 + int(o_parts[1])
                        c_min = int(c_parts[0]) * 60 + int(c_parts[1])
                        day_map[day_idx] = [(o_min, c_min)]
                if day_map:
                    self.schedule[sym] = day_map
                    count += 1
        print(f"[SCHEDULE] Override: {count} symbols from {os.path.basename(path)}")

    def _now_gmt2(self):
        return datetime.now(timezone.utc) + self.GMT2_OFFSET

    def is_crypto(self, symbol):
        return symbol in self.crypto_symbols

    def is_trading_open(self, symbol):
        now = self._now_gmt2()
        day_idx = now.weekday()
        current_min = now.hour * 60 + now.minute

        sched = self.schedule.get(symbol)
        if sched is None:
            return True, None

        sessions = sched.get(day_idx)
        if not sessions:
            return False, None

        for open_min, close_min in sessions:
            if open_min <= current_min < close_min:
                return True, close_min - current_min

        return False, None

    def get_friday_close_min(self, symbol):
        sched = self.schedule.get(symbol)
        if sched is None:
            return None
        friday = sched.get(4)
        if not friday:
            return None
        return friday[-1][1]

    def should_friday_close(self, symbol):
        if self.is_crypto(symbol):
            return False

        now = self._now_gmt2()
        if now.weekday() != 4:
            return False

        close_min = self.get_friday_close_min(symbol)
        if close_min is None:
            close_min = 23 * 60 + 50

        current_min = now.hour * 60 + now.minute
        return current_min >= (close_min - self.FRIDAY_CLOSE_BUFFER_MIN)
