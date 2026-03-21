"""Dependency injection — shared resources for the API (multi-account)."""
from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import yaml

# Async lock to rate-limit MT5 access
bridge_lock = asyncio.Lock()

# Simple TTL cache
_cache: dict[str, tuple[float, object]] = {}

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            import sys
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

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


# ── MT5 (native singleton) ──────────────────────────────────────────

_mt5_initialized = False


def get_bridge(account_id: str | None = None):
    """Get MT5 module (native). Returns None if MT5 is not available."""
    global _mt5_initialized
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    if not _mt5_initialized:
        if mt5.terminal_info() is None:
            if not mt5.initialize():
                return None
        _mt5_initialized = True
    return mt5


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


def _resolve_paper_db_path(account_id: str) -> str | None:
    """Resolve paper_trades.db path for account."""
    prefix = account_id.split("_")[0]
    for folder in [prefix, "bf" if prefix == "bright" else prefix]:
        path = REPO_ROOT / folder / "audit" / "paper_trades.db"
        if path.exists():
            return str(path)
    return None


@contextmanager
def get_paper_db(account_id: str):
    db_path = _resolve_paper_db_path(account_id)
    if db_path is None:
        yield None
        return
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
