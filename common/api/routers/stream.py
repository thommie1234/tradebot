"""SSE stream — single persistent connection pushes all dashboard data."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.deps import get_account_config, get_account_list, ACCOUNTS
from api.services import bridge_service, db_service
from api.services.bot_control import is_bot_running, get_bot_uptime

logger = logging.getLogger("api.stream")

router = APIRouter(prefix="/api", tags=["stream"])

FAST_INTERVAL = 2      # seconds between fast updates (dashboard + positions)
SLOW_EVERY = 15        # every Nth fast tick, include slow data (charts, events, temps)


async def _get_dashboard(account_id: str) -> dict:
    """Collect dashboard data for one account."""
    try:
        acct = get_account_config(account_id)
    except KeyError:
        return {}

    account_size = acct.get("account_size", 100000)

    account_info = await bridge_service.get_account_info(account_id)
    hb = db_service.get_latest_heartbeat(account_id)
    start_balance = db_service.get_daily_start_balance(account_id)
    trades_today = db_service.get_trades_today(account_id)

    if account_info:
        balance = account_info["balance"]
        equity = account_info["equity"]
        profit = account_info["profit"]
        margin_free = account_info["margin_free"]
    elif hb:
        balance = hb.get("account_balance", 0)
        equity = hb.get("account_equity", 0)
        profit = hb.get("daily_pnl", 0)
        margin_free = 0
    else:
        balance = equity = profit = margin_free = 0

    base = start_balance or balance or account_size
    daily_pnl = (equity - base) if base > 0 else 0
    # Percentages relative to account_size (matches prop firm limit definitions)
    daily_pnl_pct = daily_pnl / account_size if account_size > 0 else 0

    daily_dd_pct = abs(min(0, daily_pnl_pct))

    # Total DD: trailing (from HWM) or static (from account_size)
    dd_type = acct.get("dd_type", "static")
    if dd_type == "trailing":
        hwm = db_service.get_high_water_mark(account_id) or account_size
        dd_base = max(hwm, account_size)
    else:
        dd_base = account_size
    total_dd_pct = max(0, (dd_base - equity) / dd_base) if equity < dd_base else 0

    filled = [t for t in trades_today if t.get("status") == "FILLED"]
    wins = [t for t in filled if (t.get("pnl") or 0) > 0]

    return {
        "account_name": acct.get("name", account_id),
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "profit": round(profit, 2),
        "margin_free": round(margin_free, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl_pct, 4),
        "daily_dd_pct": round(daily_dd_pct, 4),
        "daily_dd_limit": acct.get("max_daily_loss_pct", 0.05),
        "total_dd_pct": round(total_dd_pct, 4),
        "total_dd_limit": acct.get("max_total_dd_pct", 0.10),
        "total_dd_warning": acct.get("total_dd_warning_pct", 0.08),
        "profit_target_step1": acct.get("profit_target_step1_pct", 0),
        "profit_target_step2": acct.get("profit_target_step2_pct", 0),
        "total_profit_pct": round((balance - account_size) / account_size, 4) if account_size > 0 else 0,
        "profit_gate_active": daily_pnl_pct >= acct.get("profit_gate_pct", 0.015),
        "trades_today": len(filled),
        "win_rate_today": round(len(wins) / len(filled), 2) if filled else None,
        "open_positions": hb.get("open_positions", 0) if hb else 0,
        "bridge_connected": account_info is not None,
        "bot_running": hb is not None,
        "last_heartbeat": hb.get("timestamp") if hb else None,
    }


async def _get_positions(account_id: str) -> dict:
    raw = await bridge_service.get_positions(account_id)
    grouped = bridge_service.group_split_orders(raw)
    return {
        "positions": grouped,
        "total_pnl": round(sum(p["pnl"] for p in grouped), 2),
        "logical_count": len(grouped),
        "order_count": len(raw),
    }


async def _get_slow_data() -> dict:
    """Charts, events, temps, bot status — called less frequently."""
    account_ids = list(ACCOUNTS.keys())

    results = {}
    for aid in account_ids:
        results[aid] = {
            "equity": db_service.get_equity_curve(aid, hours=168, max_points=300),
            "daily_pnl": db_service.get_daily_pnl(aid, days=30),
            "events": db_service.get_recent_events(aid, limit=12),
            "trades": db_service.get_recent_trades(aid, limit=8),
            "stats": db_service.get_trade_stats(aid, days=7),
        }

    # Temps
    try:
        from live.healthcheck import HeartbeatMonitor
        temps = HeartbeatMonitor.read_temperatures()
    except Exception:
        temps = {}

    # Bot status
    running = await is_bot_running()
    uptime = await get_bot_uptime() if running else None
    bridges = {}
    for aid in account_ids:
        bridges[aid] = await bridge_service.ping(aid)

    results["_system"] = {
        "temps": temps,
        "bot_running": running,
        "bot_uptime": uptime,
        "bridges": bridges,
    }

    return results


@router.get("/stream")
async def sse_stream(request: Request):
    """Server-Sent Events endpoint — streams all dashboard data."""

    async def event_generator():
        tick = 0
        account_ids = list(ACCOUNTS.keys())

        while True:
            # Check client disconnect
            if await request.is_disconnected():
                break

            try:
                # Fast data: dashboard + positions for all accounts
                fast = {}
                for aid in account_ids:
                    dash, pos = await asyncio.gather(
                        _get_dashboard(aid),
                        _get_positions(aid),
                    )
                    fast[aid] = {"dashboard": dash, "positions": pos}

                payload = {"type": "fast", "accounts": fast, "ts": time.time()}

                # Slow data: include on first tick and every SLOW_EVERY ticks
                if tick == 0 or tick % SLOW_EVERY == 0:
                    slow = await _get_slow_data()
                    payload["type"] = "full"
                    payload["slow"] = slow

                yield f"data: {json.dumps(payload)}\n\n"
                tick += 1

            except Exception as e:
                logger.error(f"SSE error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            await asyncio.sleep(FAST_INTERVAL)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
