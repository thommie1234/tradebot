"""Positions router — open positions per account (split orders grouped)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.deps import get_account_config
from api.services.bridge_service import get_positions, group_split_orders

router = APIRouter(prefix="/api", tags=["positions"])


@router.get("/positions/{account_id}")
async def list_positions(account_id: str):
    try:
        get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    raw = await get_positions(account_id)
    grouped = group_split_orders(raw)
    total_pnl = sum(p["pnl"] for p in grouped)

    return {
        "positions": grouped,
        "total_pnl": round(total_pnl, 2),
        "logical_count": len(grouped),
        "order_count": len(raw),
    }
