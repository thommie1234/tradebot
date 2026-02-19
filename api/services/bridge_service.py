"""Async MT5 bridge wrapper with lock to rate-limit bridge access."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from api.deps import bridge_lock, cache_get, cache_set, get_bridge

logger = logging.getLogger("api.bridge")


async def _run_bridge(fn, *args, cache_key: str | None = None, ttl: float = 5.0):
    """Run a bridge call in a thread with the async lock."""
    if cache_key:
        cached = cache_get(cache_key, ttl)
        if cached is not None:
            return cached

    async with bridge_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, fn, *args)
        if cache_key and result is not None:
            cache_set(cache_key, result)
        return result


async def ping() -> bool:
    bridge = get_bridge()
    try:
        return await _run_bridge(bridge.ping)
    except Exception:
        return False


async def get_account_info() -> dict | None:
    bridge = get_bridge()

    def _fetch():
        info = bridge.account_info()
        if info is None:
            return None
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": getattr(info, "margin", 0),
            "margin_free": getattr(info, "margin_free", 0),
        }

    return await _run_bridge(_fetch, cache_key="account_info", ttl=5)


async def get_positions() -> list[dict]:
    bridge = get_bridge()

    def _fetch():
        positions = bridge.positions_get()
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
            })
        return result

    return await _run_bridge(_fetch, cache_key="positions", ttl=3)


async def emergency_close_all() -> bool:
    """Close all positions via emergency kill. No caching, no rate limit bypass."""
    bridge = get_bridge()

    def _execute():
        from live.emergency_kill import emergency_close_all as _kill

        class _SimpleLogger:
            def log(self, level, component, event_type, message):
                logger.info(f"[{level}] {component}/{event_type}: {message}")

        _kill(_SimpleLogger(), bridge)
        return True

    async with bridge_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _execute)
