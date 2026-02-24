"""Dashboard router — main overview data per account."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.deps import get_account_config, get_account_list
from api.services import bridge_service, db_service

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/accounts")
async def list_accounts():
    return get_account_list()


@router.get("/dashboard/{account_id}")
async def get_dashboard(account_id: str):
    try:
        acct = get_account_config(account_id)
    except KeyError:
        raise HTTPException(404, f"Unknown account: {account_id}")

    account_size = acct.get("account_size", 100000)
    max_daily_loss_pct = acct.get("max_daily_loss_pct", 0.05)
    max_total_dd_pct = acct.get("max_total_dd_pct", 0.10)

    # Fetch MT5 data
    account_info = await bridge_service.get_account_info(account_id)
    bridge_ok = account_info is not None

    # DB data
    hb = db_service.get_latest_heartbeat(account_id)
    start_balance = db_service.get_daily_start_balance(account_id)
    closed_pnl = db_service.get_daily_closed_pnl(account_id)
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

    # Compliance calcs
    daily_dd_pct = abs(min(0, daily_pnl_pct))

    # Total DD: trailing (from HWM) or static (from account_size)
    dd_type = acct.get("dd_type", "static")
    if dd_type == "trailing":
        hwm = db_service.get_high_water_mark(account_id) or account_size
        dd_base = max(hwm, account_size)
    else:
        dd_base = account_size
    total_dd_pct = max(0, (dd_base - equity) / dd_base) if equity < dd_base else 0

    # Profit gate status
    profit_gate_pct = acct.get("profit_gate_pct", 0.015)
    profit_gate_active = daily_pnl_pct >= profit_gate_pct

    # Trade counts
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
        "daily_dd_limit": max_daily_loss_pct,
        "total_dd_pct": round(total_dd_pct, 4),
        "total_dd_limit": max_total_dd_pct,
        "total_dd_warning": acct.get("total_dd_warning_pct", 0.08),
        "profit_target_step1": acct.get("profit_target_step1_pct", 0),
        "profit_target_step2": acct.get("profit_target_step2_pct", 0),
        "total_profit_pct": round((balance - account_size) / account_size, 4) if account_size > 0 else 0,
        "profit_gate_active": profit_gate_active,
        "trades_today": len(filled),
        "win_rate_today": round(len(wins) / len(filled), 2) if filled else None,
        "open_positions": hb.get("open_positions", 0) if hb else 0,
        "bridge_connected": bridge_ok,
        "bot_running": hb is not None,
        "last_heartbeat": hb.get("timestamp") if hb else None,
    }
