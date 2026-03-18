"""SSE stream — single persistent connection pushes all dashboard data."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.deps import get_account_config, get_account_list, get_bridge, ACCOUNTS
from api.services import bridge_service, db_service
from api.services.bot_control import is_bot_running, get_bot_uptime

logger = logging.getLogger("api.stream")

router = APIRouter(prefix="/api", tags=["stream"])

FAST_INTERVAL = 2      # seconds between fast updates (dashboard + positions)
SLOW_EVERY = 15        # every Nth fast tick, include slow data (charts, events, temps)

# Repo root: find by marker directories
_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        break
_REPO_ROOT = _p


def _load_wfo_metrics(account_id: str) -> dict:
    """Load per-symbol WFO metrics from ftmo/config.json or bf/config.json."""
    folder = "ftmo" if "ftmo" in account_id else "bf"
    cfg_path = _REPO_ROOT / folder / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        import json as _json
        with open(cfg_path) as f:
            data = _json.load(f)
        return {
            sym: {k: v for k, v in sym_cfg.items() if k.startswith("wfo_")}
            for sym, sym_cfg in data.items()
        }
    except Exception:
        return {}


def _get_next_candle_countdowns() -> dict:
    """Seconds until next candle close for M15, M30, H1, H4."""
    now = datetime.now(timezone.utc)
    result = {}
    for tf, minutes in [("m15", 15), ("m30", 30), ("h1", 60), ("h4", 240)]:
        if minutes <= 60:
            secs_into = (now.minute % minutes) * 60 + now.second
        else:
            total_min = now.hour * 60 + now.minute
            secs_into = (total_min % minutes) * 60 + now.second
        result[f"{tf}_secs"] = minutes * 60 - secs_into + 3
    # Keep backwards compat keys
    result["h1_secs"] = result["h1_secs"]
    result["h4_secs"] = result["h4_secs"]
    return result


async def _get_dashboard(account_id: str) -> dict:
    """Collect dashboard data for one account."""
    try:
        acct = get_account_config(account_id)
    except KeyError:
        return {}

    account_size = acct.get("account_size", 100000)

    try:
        account_info = await asyncio.wait_for(
            bridge_service.get_account_info(account_id), timeout=3
        )
    except (TimeoutError, Exception):
        account_info = None
    try:
        hb = db_service.get_latest_heartbeat(account_id)
    except Exception:
        hb = None
    try:
        start_balance = db_service.get_daily_start_balance(account_id)
    except Exception:
        start_balance = None
    try:
        trades_today = db_service.get_trades_today(account_id)
    except Exception:
        trades_today = []

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
    try:
        raw = await asyncio.wait_for(
            bridge_service.get_positions(account_id), timeout=3
        )
    except (TimeoutError, Exception):
        return {"positions": [], "sector_exposure": []}
    grouped = bridge_service.group_split_orders(raw)
    try:
        sector_exposure = await asyncio.wait_for(
            bridge_service.get_sector_exposure(account_id), timeout=3
        )
    except (TimeoutError, Exception):
        sector_exposure = []

    now = datetime.now(timezone.utc)
    for pos in grouped:
        # Time in trade
        ot = pos.get("open_time")
        if ot:
            try:
                dt = datetime.fromisoformat(ot.replace("Z", "+00:00"))
                secs = int((now - dt).total_seconds())
                h, rem = divmod(secs, 3600)
                m = rem // 60
                pos["time_in_trade"] = f"{h}h {m:02d}m" if h > 0 else f"{m}m"
            except Exception:
                pos["time_in_trade"] = "—"
        else:
            pos["time_in_trade"] = "—"

        # Breakeven status: SL has moved past the entry price
        sl = pos.get("sl") or 0
        entry = pos.get("entry_price") or 0
        if entry > 0 and sl > 0:
            if pos["direction"] == "BUY":
                pos["breakeven"] = sl >= entry
            else:
                pos["breakeven"] = sl <= entry
        else:
            pos["breakeven"] = False

    return {
        "positions": grouped,
        "total_pnl": round(sum(p["pnl"] for p in grouped), 2),
        "logical_count": len(grouped),
        "order_count": len(raw),
        "sector_exposure": sector_exposure,
    }


async def _get_slow_data() -> dict:
    """Charts, events, temps, bot status — called less frequently."""
    account_ids = [k for k, v in ACCOUNTS.items() if not v.get("paper_only")]

    results = {}
    for aid in account_ids:
        try:
            results[aid] = {
                # Existing
                "equity": db_service.get_equity_curve(aid, hours=168, max_points=300),
                "equity_30d": db_service.get_equity_curve(aid, hours=720, max_points=300),
                "daily_pnl": db_service.get_daily_pnl(aid, days=30),
                "events": db_service.get_recent_events(aid, limit=12),
                "trades": db_service.get_recent_trades(aid, limit=8),
                "stats": db_service.get_trade_stats(aid, days=7),
                # New analytics
                "extended_stats": db_service.get_extended_stats(aid),
                "symbol_stats": db_service.get_symbol_stats(aid),
                "autoclose_errors": db_service.get_autoclose_errors_today(aid),
                "last_signals": db_service.get_last_signal_per_symbol(aid),
                "decay_events": db_service.get_decay_events(aid, days=7),
                "calmar": db_service.get_calmar_ratio(aid, days=90),
                "sharpe": db_service.get_sharpe_ratio(aid, days=30),
                "wfo_metrics": _load_wfo_metrics(aid),
                # Paper trading
                "paper_summary": db_service.get_paper_summary(aid),
                "paper_positions": db_service.get_paper_open_positions(aid),
                "paper_recent": db_service.get_paper_recent_trades(aid, limit=8),
                "paper_symbols": db_service.get_paper_symbol_stats(aid),
            }
        except Exception as e:
            logger.warning(f"Slow data failed for {aid}: {e}")
            results[aid] = {}

    # Paper-only accounts (e.g. TTP demo) — only paper data, no live trading data
    paper_only_ids = [k for k, v in ACCOUNTS.items() if v.get("paper_only")]
    for aid in paper_only_ids:
        try:
            results[aid] = {
                "paper_summary": db_service.get_paper_summary(aid),
                "paper_positions": db_service.get_paper_open_positions(aid),
                "paper_recent": db_service.get_paper_recent_trades(aid, limit=8),
                "paper_symbols": db_service.get_paper_symbol_stats(aid),
            }
        except Exception as e:
            logger.warning(f"Paper-only data failed for {aid}: {e}")
            results[aid] = {}

    # Enrich paper positions with unrealized P&L from live bridge prices
    all_paper_ids = account_ids + paper_only_ids
    for aid in all_paper_ids:
        paper_pos = results[aid].get("paper_positions") or []
        paper_summary = results[aid].get("paper_summary")
        if not paper_pos or not paper_summary:
            continue

        def _fetch_paper_pnl(positions, account_id):
            bridge = get_bridge(account_id)
            if bridge is None:
                return 0.0
            total = 0.0
            for pp in positions:
                sym = pp.get("symbol", "")
                # Try symbol name variants (XAU_USD → XAUUSD, etc.)
                tick = None
                variants = [sym]
                if "_" in sym:
                    variants.append(sym.replace("_", ""))
                for variant in variants:
                    try:
                        tick = bridge.symbol_info_tick(variant)
                        if tick and hasattr(tick, "bid") and tick.bid > 0:
                            break
                        tick = None
                    except Exception:
                        tick = None
                if tick and hasattr(tick, "bid"):
                    current = tick.bid if pp["direction"] == "BUY" else tick.ask
                    entry = pp["entry_price"]
                    pnl_price = (current - entry) if pp["direction"] == "BUY" else (entry - current)
                    contract = pp.get("contract_size") or 1.0
                    lots = pp["lot_size"]
                    pp["current_price"] = current
                    pp["unrealized_pnl"] = round(pnl_price * lots * contract, 2)
                    total += pp["unrealized_pnl"]

                    # SL locked profit in USD
                    sl = pp.get("sl_price") or 0
                    if sl > 0:
                        if pp["direction"] == "BUY":
                            sl_pnl_price = sl - entry
                        else:
                            sl_pnl_price = entry - sl
                        pp["sl_locked_pnl"] = round(sl_pnl_price * lots * contract, 2)
                    else:
                        pp["sl_locked_pnl"] = None
                else:
                    pp["current_price"] = None
                    pp["unrealized_pnl"] = None
                    pp["sl_locked_pnl"] = None
            return total

        try:
            unrealized = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_paper_pnl, paper_pos, aid)
            paper_summary["unrealized_pnl"] = round(unrealized, 2)
        except Exception as e:
            logger.warning(f"Paper unrealized P&L failed for {aid}: {e}")
            paper_summary["unrealized_pnl"] = 0.0

    # Temps
    try:
        from live.healthcheck import HeartbeatMonitor
        temps = HeartbeatMonitor.read_temperatures()
    except Exception:
        temps = {}

    # Bot status (check each account separately)
    bot_status = {}
    for aid in account_ids:
        r = await is_bot_running(aid)
        bot_status[aid] = {
            "running": r,
            "uptime": await get_bot_uptime(aid) if r else None,
        }
    running = any(b["running"] for b in bot_status.values())
    uptime = next((b["uptime"] for b in bot_status.values() if b["uptime"]), None)
    bridges = {}
    bridge_latency = {}
    for aid in account_ids:
        bridges[aid] = await bridge_service.ping(aid)
        bridge_latency[aid] = await bridge_service.ping_latency(aid)

    results["_system"] = {
        "temps": temps,
        "bot_running": running,
        "bot_uptime": uptime,
        "bot_accounts": bot_status,
        "bridges": bridges,
        "bridge_latency": bridge_latency,
        "next_candles": _get_next_candle_countdowns(),
    }

    # Spike detector data (separate from normal paper)
    try:
        results["_spike"] = {
            "summary": db_service.get_spike_summary(),
            "positions": db_service.get_spike_open_positions(),
            "recent": db_service.get_spike_recent_trades(limit=8),
            "events": db_service.get_spike_recent_events(limit=10),
        }
    except Exception:
        results["_spike"] = None

    return results


@router.get("/stream")
async def sse_stream(request: Request):
    """Server-Sent Events endpoint — streams all dashboard data."""

    async def event_generator():
        tick = 0
        account_ids = [k for k, v in ACCOUNTS.items() if not v.get("paper_only")]

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
