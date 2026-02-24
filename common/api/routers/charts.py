"""Charts router — equity curve and daily P&L data."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.deps import get_account_config
from api.services import db_service

router = APIRouter(prefix="/api", tags=["charts"])


@router.get("/equity/{account_id}")
async def equity_curve(
    account_id: str,
    hours: int = Query(168, ge=1, le=720),
    max_points: int = Query(300, ge=50, le=1000),
):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    return db_service.get_equity_curve(account_id, hours=hours, max_points=max_points)


@router.get("/daily-pnl/{account_id}")
async def daily_pnl(
    account_id: str,
    days: int = Query(30, ge=1, le=90),
):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    return db_service.get_daily_pnl(account_id, days=days)
