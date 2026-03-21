"""System router — temps, bot status, events, bot control."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_auth
from api.deps import get_account_config
from api.services import bridge_service, db_service
from api.services.bot_control import get_bot_uptime, is_bot_running, restart_bot, scan_bot

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
    accounts = {}
    for aid in ("ftmo_100k", "bright_100k"):
        r = await is_bot_running(aid)
        accounts[aid] = {
            "running": r,
            "uptime": await get_bot_uptime(aid) if r else None,
            "bridge": await bridge_service.ping(aid),
        }

    return {
        "bot_running": any(a["running"] for a in accounts.values()),
        "bot_uptime": next((a["uptime"] for a in accounts.values() if a.get("uptime")), None),
        "accounts": accounts,
    }


@router.post("/bot/restart")
async def bot_restart(_=Depends(require_auth)):
    ok, msg = await restart_bot()
    return {"success": ok, "message": msg}


@router.post("/bot/scan/{account_id}")
async def bot_scan(account_id: str):
    """Force a full scan (H1 + multi-TF) for the given account."""
    ok, msg = await scan_bot(account_id)
    if not ok:
        raise HTTPException(500, msg)
    return {"success": ok, "message": msg}
