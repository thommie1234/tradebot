"""Dependency injection — shared resources for the API (multi-account)."""
from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import yaml

from api.config import BRIDGE_HOST, BRIDGE_PORT

# Async lock to rate-limit MT5 bridge access (max ~2 req/5s)
bridge_lock = asyncio.Lock()

# Simple TTL cache for bridge responses
_cache: dict[str, tuple[float, object]] = {}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Account config ───────────────────────────────────────────────────

def _load_accounts() -> dict[str, dict]:
    """Load accounts from config/accounts.yaml."""
    acct_file = REPO_ROOT / "config" / "accounts.yaml"
    if not acct_file.exists():
        return {}
    with open(acct_file) as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.get("accounts", {}).items() if v.get("enabled", True)}


ACCOUNTS = _load_accounts()


def get_account_config(account_id: str) -> dict:
    """Get account config by ID. Raises KeyError if not found."""
    if account_id not in ACCOUNTS:
        raise KeyError(f"Unknown account: {account_id}")
    return ACCOUNTS[account_id]


def get_account_list() -> list[dict]:
    """Return list of accounts with id and name."""
    return [{"id": k, "name": v.get("name", k)} for k, v in ACCOUNTS.items()]


# ── Cache ────────────────────────────────────────────────────────────

def cache_get(key: str, ttl: float = 5.0):
    import time
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def cache_set(key: str, value: object):
    import time
    _cache[key] = (time.time(), value)


# ── Bridge (multi-account) ──────────────────────────────────────────

def get_bridge(account_id: str | None = None):
    """Get MT5BridgeClient for account. Falls back to default port."""
    from tools.mt5_bridge import MT5BridgeClient
    port = BRIDGE_PORT
    if account_id and account_id in ACCOUNTS:
        port = ACCOUNTS[account_id].get("bridge_port", BRIDGE_PORT)
    return MT5BridgeClient(host=BRIDGE_HOST, port=port, timeout=5)


# ── Database (multi-account) ────────────────────────────────────────

def _resolve_db_path(account_id: str | None = None) -> str:
    """Resolve audit DB path for account."""
    if account_id and account_id in ACCOUNTS:
        rel = ACCOUNTS[account_id].get("audit_db", "")
        if rel:
            return str(REPO_ROOT / rel)
    # Fallback to cfg.DB_PATH
    from config.loader import cfg
    return cfg.DB_PATH


@contextmanager
def get_db(account_id: str | None = None):
    db_path = _resolve_db_path(account_id)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
