"""History router — trade history and stats per account."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.deps import get_account_config
from api.services import db_service

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history/{account_id}")
async def trade_history(
    account_id: str,
    days: int | None = Query(None),
    symbol: str | None = Query(None),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    trades, total = db_service.get_trade_history(
        account_id, days=days, symbol=symbol, status=status,
        page=page, page_size=page_size,
    )
    return {"trades": trades, "total": total, "page": page, "page_size": page_size}


@router.get("/stats/{account_id}")
async def trade_stats(
    account_id: str,
    days: int | None = Query(7),
):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    return db_service.get_trade_stats(account_id, days=days)


@router.get("/trades/recent/{account_id}")
async def recent_trades(
    account_id: str,
    limit: int = Query(10, ge=1, le=100),
):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    return db_service.get_recent_trades(account_id, limit=limit)
