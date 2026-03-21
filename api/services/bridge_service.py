"""Async MT5 wrapper with lock to rate-limit access (multi-account)."""
from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import datetime, timezone

from api.deps import cache_get, cache_set, get_bridge

logger = logging.getLogger("api.bridge")


async def _run_bridge(account_id: str, fn, *args, cache_key: str | None = None, ttl: float = 5.0):
    """Run a bridge call in a thread with the async lock."""
    full_key = f"{account_id}:{cache_key}" if cache_key else None
    if full_key:
        cached = cache_get(full_key, ttl)
        if cached is not None:
            return cached

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, fn, *args), timeout=3
        )
        if full_key and result is not None:
            cache_set(full_key, result)
        return result
    except (TimeoutError, Exception) as e:
        logger.warning(f"Bridge call failed for {account_id}:{cache_key}: {e}")
        return None


async def ping(account_id: str) -> bool:
    mt5 = get_bridge(account_id)
    if mt5 is None:
        return False
    try:
        return await _run_bridge(account_id, lambda: mt5.terminal_info() is not None)
    except Exception:
        return False


async def get_account_info(account_id: str) -> dict | None:
    mt5 = get_bridge(account_id)
    if mt5 is None:
        return None

    def _fetch():
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": getattr(info, "margin", 0),
            "margin_free": getattr(info, "margin_free", 0),
        }

    return await _run_bridge(account_id, _fetch, cache_key="account_info", ttl=5)


async def get_positions(account_id: str) -> list[dict]:
    mt5 = get_bridge(account_id)
    if mt5 is None:
        return []

    def _fetch():
        positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for pos in positions:
            direction = "BUY" if pos.type == 0 else "SELL"
            current_price = getattr(pos, "price_current", 0)
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "direction": direction,
                "volume": pos.volume,
                "entry_price": pos.price_open,
                "current_price": current_price,
                "pnl": pos.profit,
                "swap": getattr(pos, "swap", 0),
                "sl": getattr(pos, "sl", None),
                "tp": getattr(pos, "tp", None),
                "magic": getattr(pos, "magic", 0),
                "open_time": datetime.fromtimestamp(
                    pos.time, tz=timezone.utc
                ).isoformat() if hasattr(pos, "time") else None,
                "type": pos.type,
                "time": getattr(pos, "time", 0),
            })
        return result

    return await _run_bridge(account_id, _fetch, cache_key="positions", ttl=3)


async def ping_latency(account_id: str) -> float | None:
    """Return round-trip ping latency in ms, or None if unreachable."""
    mt5 = get_bridge(account_id)
    if mt5 is None:
        return None
    try:
        t0 = _time.monotonic()
        ok = await _run_bridge(account_id, lambda: mt5.terminal_info() is not None)
        elapsed = (_time.monotonic() - t0) * 1000
        return round(elapsed, 1) if ok else None
    except Exception:
        return None


async def get_sector_exposure(account_id: str) -> list[dict]:
    """Compute sector exposure from open positions using SECTOR_MAP."""
    try:
        from risk.position_sizing import MAX_SECTOR_EXPOSURE, SECTOR_MAP
    except ImportError:
        return []

    cache_key = f"{account_id}:sector_exposure"
    cached = cache_get(cache_key, 5.0)
    if cached is not None:
        return cached

    positions = await get_positions(account_id)
    account_info = await get_account_info(account_id)
    equity = account_info["equity"] if account_info else 100_000.0

    sector_abs: dict[str, float] = {}
    for pos in positions:
        sector = SECTOR_MAP.get(pos["symbol"], "other")
        sector_abs[sector] = sector_abs.get(sector, 0.0) + abs(pos["pnl"])

    result = []
    for sector, used_abs in sorted(sector_abs.items()):
        limit_pct = MAX_SECTOR_EXPOSURE.get(sector, 0.02)
        used_pct = used_abs / equity if equity > 0 else 0.0
        result.append({
            "sector": sector,
            "used_pct": round(used_pct, 4),
            "limit_pct": round(limit_pct, 4),
            "used_of_limit": round(used_pct / limit_pct, 3) if limit_pct > 0 else 0.0,
        })

    cache_set(cache_key, result)
    return result


def group_split_orders(positions: list[dict]) -> list[dict]:
    """Group split orders into logical positions (same symbol, direction, 5-min bucket)."""
    groups: dict[tuple, list[dict]] = {}
    for p in positions:
        key = (p["symbol"], p["type"], p.get("time", 0) // 300)
        groups.setdefault(key, []).append(p)

    result = []
    for key, chunks in groups.items():
        total_volume = sum(c["volume"] for c in chunks)
        total_pnl = sum(c["pnl"] for c in chunks)
        total_swap = sum(c["swap"] for c in chunks)
        rep = chunks[0]  # representative
        result.append({
            "symbol": rep["symbol"],
            "direction": rep["direction"],
            "volume": round(total_volume, 2),
            "entry_price": rep["entry_price"],
            "current_price": rep["current_price"],
            "pnl": round(total_pnl, 2),
            "swap": round(total_swap, 2),
            "sl": rep["sl"],
            "tp": rep["tp"],
            "magic": rep["magic"],
            "open_time": rep["open_time"],
            "order_count": len(chunks),
            "tickets": [c["ticket"] for c in chunks],
        })
    return result
