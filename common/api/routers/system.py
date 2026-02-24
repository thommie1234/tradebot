"""System router — temps, bot status, events, bot control."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_auth
from api.deps import get_account_config
from api.services import bridge_service, db_service
from api.services.bot_control import get_bot_uptime, is_bot_running, restart_bot

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/temps")
async def temperatures():
    from live.healthcheck import HeartbeatMonitor
    try:
        temps = HeartbeatMonitor.read_temperatures()
    except Exception:
        temps = {}
    return temps


@router.get("/events/{account_id}")
async def recent_events(
    account_id: str,
    limit: int = Query(20, ge=1, le=100),
):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    return db_service.get_recent_events(account_id, limit=limit)


@router.get("/bot/status")
async def bot_status():
    running = await is_bot_running()
    uptime = await get_bot_uptime() if running else None

    # Check both bridges
    ftmo_ok = await bridge_service.ping("ftmo_100k")
    bf_ok = await bridge_service.ping("bright_100k")

    return {
        "bot_running": running,
        "bot_uptime": uptime,
        "bridges": {
            "ftmo_100k": ftmo_ok,
            "bright_100k": bf_ok,
        },
    }


@router.post("/bot/restart")
async def bot_restart(_=Depends(require_auth)):
    ok, msg = await restart_bot()
    return {"success": ok, "message": msg}
