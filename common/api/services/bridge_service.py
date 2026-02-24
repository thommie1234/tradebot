"""Async MT5 bridge wrapper with lock to rate-limit bridge access (multi-account)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from api.deps import bridge_lock, cache_get, cache_set, get_bridge

logger = logging.getLogger("api.bridge")


async def _run_bridge(account_id: str, fn, *args, cache_key: str | None = None, ttl: float = 5.0):
    """Run a bridge call in a thread with the async lock."""
    full_key = f"{account_id}:{cache_key}" if cache_key else None
    if full_key:
        cached = cache_get(full_key, ttl)
        if cached is not None:
            return cached

    async with bridge_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, fn, *args)
        if full_key and result is not None:
            cache_set(full_key, result)
        return result


async def ping(account_id: str) -> bool:
    bridge = get_bridge(account_id)
    try:
        return await _run_bridge(account_id, bridge.ping)
    except Exception:
        return False


async def get_account_info(account_id: str) -> dict | None:
    bridge = get_bridge(account_id)

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

    return await _run_bridge(account_id, _fetch, cache_key="account_info", ttl=5)


async def get_positions(account_id: str) -> list[dict]:
    bridge = get_bridge(account_id)

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
                "type": pos.type,
                "time": getattr(pos, "time", 0),
            })
        return result

    return await _run_bridge(account_id, _fetch, cache_key="positions", ttl=3)


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
