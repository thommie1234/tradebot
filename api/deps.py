"""Dependency injection — shared resources for the API."""
from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager

from api.config import BRIDGE_HOST, BRIDGE_PORT

# Async lock to rate-limit MT5 bridge access (max ~2 req/5s)
bridge_lock = asyncio.Lock()

# Simple TTL cache for bridge responses
_cache: dict[str, tuple[float, object]] = {}


def cache_get(key: str, ttl: float = 5.0):
    import time
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def cache_set(key: str, value: object):
    import time
    _cache[key] = (time.time(), value)


def get_bridge():
    from tools.mt5_bridge import MT5BridgeClient
    return MT5BridgeClient(host=BRIDGE_HOST, port=BRIDGE_PORT, timeout=5)


@contextmanager
def get_db():
    from config.loader import cfg
    conn = sqlite3.connect(cfg.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
    finally:
        conn.close()
