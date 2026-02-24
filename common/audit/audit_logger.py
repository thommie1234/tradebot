"""
Audit logger — SQLite WAL + hash chaining for tamper-proof event logging.

Extracted from BlackoutLogger in sovereign_bot.py (lines 241-400).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime

from config.loader import cfg


class BlackoutLogger:
    """Production event logger — tracks everything with hash chain.

    Every connection is opened and closed per-call using try/finally
    to guarantee no file descriptor leaks.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or cfg.DB_PATH
        self._db_dir = os.path.dirname(self.db_path)
        self._last_event_hash = "GENESIS"
        self._last_trade_hash = "GENESIS"
        self._lock = threading.Lock()
        self.init_database()

    @staticmethod
    def _sha256(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _chain_hash(self, prev_hash: str, *fields) -> str:
        payload = prev_hash + "|" + "|".join(str(f) for f in fields)
        return self._sha256(payload)

    def _open(self):
        if self._db_dir:
            os.makedirs(self._db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init_database(self):
        conn = self._open()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    data TEXT,
                    hash TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    tp_price REAL,
                    sl_price REAL,
                    lot_size REAL,
                    spread_pct REAL,
                    ml_confidence REAL,
                    ticket INTEGER,
                    status TEXT,
                    exit_timestamp TEXT,
                    exit_price REAL,
                    pnl REAL,
                    hash TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS heartbeats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mt5_connected BOOLEAN,
                    account_balance REAL,
                    account_equity REAL,
                    open_positions INTEGER,
                    daily_pnl REAL
                )
            ''')

            for table, col in [("events", "hash"), ("trades", "hash")]:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass

            row = cursor.execute(
                "SELECT hash FROM events WHERE hash IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                self._last_event_hash = row[0]

            row = cursor.execute(
                "SELECT hash FROM trades WHERE hash IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                self._last_trade_hash = row[0]

            conn.commit()
        finally:
            conn.close()

    def log(self, level, component, event_type, message, data=None):
        ts = datetime.now().isoformat()
        data_str = json.dumps(data) if data else None
        h = self._chain_hash(self._last_event_hash, ts, level, component, event_type, message, data_str)
        self._last_event_hash = h
        try:
            with self._lock:
                conn = self._open()
                try:
                    conn.execute('''
                        INSERT INTO events (timestamp, level, component, event_type, message, data, hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (ts, level, component, event_type, message, data_str, h))
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Logger.DB_ERROR: {e}")
        print(f"[{level}] {component}.{event_type}: {message}")

    def log_trade(self, symbol, direction, entry_price, tp_price, sl_price,
                  lot_size, spread_pct, ml_confidence, ticket=None, status='PENDING'):
        ts = datetime.now().isoformat()
        h = self._chain_hash(self._last_trade_hash, ts, symbol, direction,
                             entry_price, tp_price, sl_price, lot_size,
                             spread_pct, ml_confidence, ticket, status)
        self._last_trade_hash = h
        try:
            with self._lock:
                conn = self._open()
                try:
                    conn.execute('''
                        INSERT INTO trades (timestamp, symbol, direction, entry_price, tp_price,
                                            sl_price, lot_size, spread_pct, ml_confidence, ticket, status, hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (ts, symbol, direction, entry_price, tp_price,
                          sl_price, lot_size, spread_pct, ml_confidence, ticket, status, h))
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Logger.DB_ERROR: {e}")

    def log_heartbeat(self, mt5_connected, account_balance, account_equity,
                      open_positions, daily_pnl):
        try:
            with self._lock:
                conn = self._open()
                try:
                    conn.execute('''
                        INSERT INTO heartbeats (timestamp, mt5_connected, account_balance,
                                                account_equity, open_positions, daily_pnl)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(), mt5_connected, account_balance,
                        account_equity, open_positions, daily_pnl
                    ))
                    conn.commit()
                finally:
                    conn.close()
                    del conn
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Logger.DB_ERROR: {e}")

    @staticmethod
    def _fd_count():
        import os
        try:
            return len(os.listdir(f'/proc/{os.getpid()}/fd'))
        except Exception:
            return -1
