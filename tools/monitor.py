#!/usr/bin/env python3
"""
SOVEREIGN BOT MONITOR — Live CLI Dashboard
============================================
Real-time terminal dashboard showing account status, FTMO compliance,
open positions, hardware temps, and recent activity.

Usage:
    python3 tools/monitor.py              # Live dashboard, 10s refresh
    python3 tools/monitor.py --once       # Single snapshot, then exit
    python3 tools/monitor.py --interval 5 # Custom refresh interval
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.mt5_bridge import MT5BridgeClient
from live.healthcheck import HeartbeatMonitor

# ── Constants ────────────────────────────────────────────────────────

INITIAL_BALANCE = 100_000.0
MAX_DAILY_LOSS_PCT = 0.05   # 5%
MAX_TOTAL_DD_PCT = 0.10     # 10%
DB_PATH = REPO_ROOT / "audit" / "sovereign_log.db"
WIDTH = 70                  # inner width between ║ ║


# ── ANSI Colors ──────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    CYAN   = "\033[36m"
    WHITE  = "\033[37m"
    BG_RED = "\033[41m"

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def _parse_ts(raw: str) -> datetime:
    """Parse ISO timestamp, treating naive as UTC."""
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def vlen(s: str) -> int:
    """Visible length of string (excluding ANSI escapes)."""
    return len(_ANSI_RE.sub('', s))


def pad(s: str, width: int) -> str:
    """Pad string to visible width."""
    return s + ' ' * max(0, width - vlen(s))


def line(text: str = "") -> str:
    """Render a box line:  ║  text...padding  ║"""
    inner = f"  {text}"
    return f"║{pad(inner, WIDTH)}║"


def pnl_color(value: float) -> str:
    if value > 0:
        return C.GREEN
    elif value < 0:
        return C.RED
    return C.WHITE


def pnl_str(value: float) -> str:
    if value >= 0:
        return f"+${abs(value):,.2f}"
    return f"-${abs(value):,.2f}"


def progress_bar(value: float, maximum: float, width: int = 20) -> str:
    ratio = min(value / maximum, 1.0) if maximum > 0 else 0
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


# ── Data Fetching ────────────────────────────────────────────────────

def get_account_data(mt5: MT5BridgeClient) -> dict | None:
    try:
        acc = mt5.account_info()
        if acc is None:
            return None
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin_free": getattr(acc, "margin_free", 0),
            "profit": getattr(acc, "profit", 0),
        }
    except Exception:
        return None


def get_positions(mt5: MT5BridgeClient) -> list[dict] | None:
    try:
        positions = mt5.positions_get()
        if positions is None:
            return None
        result = []
        for p in positions:
            direction = "BUY" if getattr(p, "type", 0) == 0 else "SELL"
            price_open = getattr(p, "price_open", 0)
            sl = getattr(p, "sl", 0)

            # Trailing = SL moved past entry (breakeven or better)
            if direction == "BUY":
                trail = "ACTIVE" if sl > 0 and sl >= price_open else "pending"
            else:
                trail = "ACTIVE" if sl > 0 and sl <= price_open else "pending"

            result.append({
                "symbol": getattr(p, "symbol", "???"),
                "direction": direction,
                "volume": getattr(p, "volume", 0),
                "profit": getattr(p, "profit", 0),
                "trail": trail,
            })
        return result
    except Exception:
        return None


def _db_connect(db_path: str):
    """Open audit DB read-only."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def get_daily_start_balance(db_path: str) -> float | None:
    try:
        conn = _db_connect(db_path)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT account_balance FROM heartbeats "
            "WHERE timestamp >= ? ORDER BY id ASC LIMIT 1",
            (today,),
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def get_recent_trades(db_path: str, limit: int = 5) -> list[dict]:
    try:
        conn = _db_connect(db_path)
        rows = conn.execute(
            "SELECT timestamp, symbol, direction, pnl, ml_confidence, status "
            "FROM trades ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"timestamp": r[0], "symbol": r[1], "direction": r[2],
             "pnl": r[3], "confidence": r[4], "status": r[5]}
            for r in rows
        ]
    except Exception:
        return []


def get_recent_events(db_path: str, limit: int = 5) -> list[dict]:
    try:
        conn = _db_connect(db_path)
        rows = conn.execute(
            "SELECT timestamp, level, component, event_type, message "
            "FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"timestamp": r[0], "level": r[1], "component": r[2],
             "event_type": r[3], "message": r[4]}
            for r in rows
        ]
    except Exception:
        return []


def get_bot_status(db_path: str) -> dict:
    status = {"bot": "UNKNOWN", "mt5": "UNKNOWN", "last_scan": "N/A",
              "trades_today": 0, "inactivity_days": 0, "sl_mods_today": 0}
    try:
        conn = _db_connect(db_path)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Latest heartbeat → bot alive? MT5 connected?
        row = conn.execute(
            "SELECT timestamp, mt5_connected FROM heartbeats "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            try:
                hb_time = _parse_ts(row[0])
                age = (_utcnow() - hb_time).total_seconds()
                status["last_scan"] = hb_time.strftime("%H:%M")
                status["bot"] = "RUNNING" if age < 120 else "STALE"
            except Exception:
                status["last_scan"] = "?"
                status["bot"] = "RUNNING"
            status["mt5"] = "CONNECTED" if row[1] else "DISCONNECTED"

        # Trades today
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (today,)
        ).fetchone()
        status["trades_today"] = row[0] if row else 0

        # Inactivity (days since last trade)
        row = conn.execute(
            "SELECT timestamp FROM trades ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            try:
                last = _parse_ts(row[0])
                status["inactivity_days"] = max(0, (_utcnow() - last).days)
            except Exception:
                pass

        # SL modification events today
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE timestamp >= ? "
            "AND (event_type LIKE '%SL%' OR event_type LIKE '%TRAIL%' "
            "OR event_type LIKE '%BREAKEVEN%')",
            (today,),
        ).fetchone()
        status["sl_mods_today"] = row[0] if row else 0

        conn.close()
    except Exception:
        pass
    return status


def get_temperatures() -> dict:
    try:
        return HeartbeatMonitor.read_temperatures()
    except Exception:
        return {}


def get_ftmo_status(equity: float, daily_start: float | None) -> dict:
    if daily_start is None:
        daily_start = equity

    daily_loss = max(0, daily_start - equity)
    daily_dd_pct = (daily_loss / daily_start * 100) if daily_start > 0 else 0

    total_loss = max(0, INITIAL_BALANCE - equity)
    total_dd_pct = (total_loss / INITIAL_BALANCE * 100)

    return {
        "daily_dd_pct": daily_dd_pct,
        "daily_dd_limit": MAX_DAILY_LOSS_PCT * 100,
        "daily_ok": daily_dd_pct < MAX_DAILY_LOSS_PCT * 100,
        "total_dd_pct": total_dd_pct,
        "total_dd_limit": MAX_TOTAL_DD_PCT * 100,
        "total_ok": total_dd_pct < MAX_TOTAL_DD_PCT * 100,
    }


# ── Rendering ────────────────────────────────────────────────────────

def render(data: dict) -> str:
    out: list[str] = []
    now = datetime.now().strftime("%d %b %Y %H:%M")

    # ── Header ──
    out.append(f"╔{'═' * WIDTH}╗")
    title = "SOVEREIGN BOT MONITOR"
    hdr = f"{C.BOLD}{C.CYAN}{title}{C.RESET}"
    hdr_pad = WIDTH - 4 - len(title) - len(now)
    out.append(f"║  {hdr}{' ' * hdr_pad}{now}  ║")
    out.append(f"╠{'═' * WIDTH}╣")

    # ── Account ──
    acc = data.get("account")
    out.append(line(f"{C.BOLD}ACCOUNT{C.RESET}"))
    if acc:
        out.append(line(
            f"Balance: ${acc['balance']:,.0f}    "
            f"Equity: ${acc['equity']:,.0f}    "
            f"Margin Free: ${acc['margin_free']:,.0f}"
        ))
        dpnl = data.get("daily_pnl", 0)
        ds = data.get("daily_start") or acc["balance"]
        dpct = (dpnl / ds * 100) if ds else 0
        tpnl = acc["balance"] - INITIAL_BALANCE
        tpct = tpnl / INITIAL_BALANCE * 100
        dc, tc = pnl_color(dpnl), pnl_color(tpnl)
        out.append(line(
            f"Daily P&L: {dc}{pnl_str(dpnl)} "
            f"({'+' if dpct >= 0 else ''}{dpct:.2f}%){C.RESET}    "
            f"Total P&L: {tc}{pnl_str(tpnl)} "
            f"({'+' if tpct >= 0 else ''}{tpct:.2f}%){C.RESET}"
        ))
    else:
        out.append(line(f"{C.RED}MT5 bridge unavailable — no live data{C.RESET}"))
    out.append(line())

    # ── FTMO Compliance ──
    ftmo = data.get("ftmo", {})
    bs = data.get("bot_status", {})
    out.append(line(f"{C.BOLD}FTMO COMPLIANCE{C.RESET}"))
    if ftmo:
        dd = ftmo["daily_dd_pct"]
        dl = ftmo["daily_dd_limit"]
        bar = progress_bar(dd, dl)
        st = f"{C.GREEN}OK{C.RESET}" if ftmo["daily_ok"] else f"{C.BG_RED}{C.WHITE}BREACH{C.RESET}"
        out.append(line(f"Daily DD:   {dd:>5.2f}% / {dl:.2f}%  {bar}  {st}"))

        td = ftmo["total_dd_pct"]
        tl = ftmo["total_dd_limit"]
        bar2 = progress_bar(td, tl)
        st2 = f"{C.GREEN}OK{C.RESET}" if ftmo["total_ok"] else f"{C.BG_RED}{C.WHITE}BREACH{C.RESET}"
        out.append(line(f"Total DD:   {td:>5.2f}% / {tl:.1f}%  {bar2}  {st2}"))

    out.append(line(
        f"Trades:     {bs.get('trades_today', 0)} / 2000       "
        f"Inactivity: {bs.get('inactivity_days', 0)} days"
    ))
    out.append(line(f"SL Mods:    {bs.get('sl_mods_today', 0)} today"))
    out.append(line())

    # ── Positions ──
    positions = data.get("positions")
    if positions is not None:
        total_p = sum(p["profit"] for p in positions)
        pc = pnl_color(total_p)
        out.append(line(
            f"{C.BOLD}POSITIONS{C.RESET} "
            f"({len(positions)} open, P&L: {pc}{pnl_str(total_p)}{C.RESET})"
        ))
        if positions:
            for p in positions:
                sym = p["symbol"][:8].ljust(8)
                d = p["direction"].ljust(4)
                vol = f"{p['volume']:.2f} lots"
                pc2 = pnl_color(p["profit"])
                ps = pnl_str(p["profit"])
                trail = p["trail"]
                trc = C.GREEN if trail == "ACTIVE" else C.DIM
                out.append(line(
                    f"{sym}  {d}  {vol}  {pc2}{ps}{C.RESET}"
                    f"  Trail: {trc}{trail}{C.RESET}"
                ))
        else:
            out.append(line(f"{C.DIM}No open positions{C.RESET}"))
    else:
        out.append(line(f"{C.BOLD}POSITIONS{C.RESET}"))
        out.append(line(f"{C.DIM}Unavailable (MT5 bridge down){C.RESET}"))
    out.append(line())

    # ── Hardware ──
    temps = data.get("temps", {})
    out.append(line(f"{C.BOLD}HARDWARE{C.RESET}"))

    def tc(t):
        if t >= 75: return C.RED
        if t >= 65: return C.YELLOW
        return C.GREEN

    cpu = temps.get("cpu", 0)
    gpu = temps.get("gpu", 0)
    gname = temps.get("gpu_name", "GPU")
    nvme = temps.get("nvme", 0)
    trd = f"{C.GREEN}OK{C.RESET}" if not HeartbeatMonitor.GPU_TRADING_PAUSE \
        else f"{C.RED}PAUSED{C.RESET}"

    if any([cpu, gpu, nvme]):
        out.append(line(
            f"CPU: {tc(cpu)}{cpu:.0f}°C{C.RESET}    "
            f"GPU: {tc(gpu)}{gpu:.0f}°C{C.RESET} ({gname[:3]})    "
            f"NVMe: {tc(nvme)}{nvme:.0f}°C{C.RESET}    "
            f"Trading: {trd}"
        ))
    else:
        out.append(line(f"{C.DIM}Temperature data unavailable{C.RESET}    Trading: {trd}"))
    out.append(line())

    # ── Recent Trades ──
    trades = data.get("recent_trades", [])
    out.append(line(f"{C.BOLD}RECENT TRADES{C.RESET} (last {len(trades)})"))
    if trades:
        for t in trades:
            try:
                dt = _parse_ts(t["timestamp"])
                ts = dt.strftime("%d %b %H:%M")
            except Exception:
                ts = (t["timestamp"] or "")[:12]
            sym = (t["symbol"] or "???")[:7].ljust(7)
            d = (t["direction"] or "???").ljust(4)
            pnl = t.get("pnl")
            if pnl is not None:
                pc2 = pnl_color(pnl)
                ps = f"{'+' if pnl >= 0 else ''}{pnl:>8.2f}"
            else:
                pc2, ps = C.DIM, "    open"
            conf = t.get("confidence")
            cs = f" {conf:.3f}" if conf else ""
            status = t.get("status") or ""
            if status and status not in ("CLOSED",):
                ss = f" ({status[:8].lower()})"
            else:
                ss = ""
            out.append(line(f"{ts}  {sym} {d}  {pc2}{ps}{C.RESET}{cs}{ss}"))
    else:
        out.append(line(f"{C.DIM}No recent trades{C.RESET}"))
    out.append(line())

    # ── Recent Events ──
    events = data.get("recent_events", [])
    out.append(line(f"{C.BOLD}RECENT EVENTS{C.RESET} (last {len(events)})"))
    if events:
        for e in events:
            try:
                dt = _parse_ts(e["timestamp"])
                ts = dt.strftime("%H:%M")
            except Exception:
                ts = (e["timestamp"] or "")[:5]
            lvl = (e.get("level") or "INFO")
            lc = {"INFO": C.GREEN, "WARNING": C.YELLOW, "CRITICAL": C.RED,
                  "ERROR": C.RED, "DEBUG": C.DIM}.get(lvl, C.WHITE)
            comp = (e.get("component") or "")[:15].ljust(15)
            etype = (e.get("event_type") or "")[:8]
            # Remaining space: 68 - 5(ts) - 2 - 7(lvl) - 2 - 15 - 2 - 8 - 2 = 25
            msg = (e.get("message") or "")[:25]
            out.append(line(
                f"{ts}  {lc}{lvl:<7}{C.RESET}  {comp}  {etype}  {msg}"
            ))
    else:
        out.append(line(f"{C.DIM}No recent events{C.RESET}"))
    out.append(line())

    # ── Status Bar ──
    bot = bs.get("bot", "UNKNOWN")
    m5 = bs.get("mt5", "UNKNOWN")
    ls = bs.get("last_scan", "N/A")
    bc = C.GREEN if bot == "RUNNING" else C.RED if bot == "STALE" else C.YELLOW
    mc = C.GREEN if m5 == "CONNECTED" else C.RED
    out.append(line(
        f"Bot: {bc}{bot}{C.RESET}    "
        f"MT5: {mc}{m5}{C.RESET}    "
        f"Last scan: {ls}"
    ))

    out.append(f"╚{'═' * WIDTH}╝")
    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────

def collect_data(mt5: MT5BridgeClient, db_path: str) -> dict:
    data: dict = {}

    data["account"] = get_account_data(mt5)
    data["positions"] = get_positions(mt5)

    daily_start = get_daily_start_balance(db_path)
    data["daily_start"] = daily_start

    if data["account"] and daily_start:
        data["daily_pnl"] = data["account"]["equity"] - daily_start
    elif data["account"]:
        data["daily_pnl"] = data["account"]["profit"]
    else:
        data["daily_pnl"] = 0

    equity = data["account"]["equity"] if data["account"] else INITIAL_BALANCE
    data["ftmo"] = get_ftmo_status(equity, daily_start)
    data["temps"] = get_temperatures()
    data["recent_trades"] = get_recent_trades(db_path)
    data["recent_events"] = get_recent_events(db_path)
    data["bot_status"] = get_bot_status(db_path)

    return data


def main():
    parser = argparse.ArgumentParser(description="Sovereign Bot CLI Monitor")
    parser.add_argument("--once", action="store_true",
                        help="Single snapshot, then exit")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--port", type=int, default=5056,
                        help="MT5 bridge port (default: 5056)")
    args = parser.parse_args()

    mt5 = MT5BridgeClient(host="127.0.0.1", port=args.port, timeout=5)
    db_path = str(DB_PATH)

    if args.once:
        data = collect_data(mt5, db_path)
        print(render(data))
        return

    try:
        while True:
            data = collect_data(mt5, db_path)
            os.system("clear")
            print(render(data))
            print(f"  Auto-refresh: {args.interval}s | Press Ctrl+C to exit")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
