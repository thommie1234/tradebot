#!/usr/bin/env python3
"""
SOVEREIGN BOT MONITOR — Live CLI Dashboard (Multi-Account)
============================================================
Real-time terminal dashboard showing all accounts' status,
compliance, open positions, hardware temps, and recent activity.

Reads account config from config/accounts.yaml automatically.

Usage:
    python3 tools/monitor.py              # Live dashboard, 10s refresh
    python3 tools/monitor.py --once       # Single snapshot, then exit
    python3 tools/monitor.py --interval 5 # Custom refresh interval
    python3 tools/monitor.py --account ftmo_100k  # Single account only
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
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.mt5_bridge import MT5BridgeClient
from live.healthcheck import HeartbeatMonitor

# ── Constants ────────────────────────────────────────────────────────

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


# ── Account Config Loading ──────────────────────────────────────────

def load_account_configs() -> list[dict]:
    """Load account configs from accounts.yaml. Returns list of account dicts."""
    import yaml

    accounts_path = REPO_ROOT / "config" / "accounts.yaml"
    if not accounts_path.exists():
        # Fallback: single default account
        return [{
            "id": "default",
            "name": "FTMO (default)",
            "bridge_port": int(os.getenv("MT5_BRIDGE_PORT", "5056")),
            "account_size": 100_000,
            "max_daily_loss_pct": 0.05,
            "max_total_dd_pct": 0.10,
            "audit_db": "audit/sovereign_log.db",
            "enabled": True,
        }]

    with open(accounts_path) as f:
        data = yaml.safe_load(f)

    accounts = []
    for acct_id, cfg in data.get("accounts", {}).items():
        if not cfg.get("enabled", False):
            continue
        accounts.append({
            "id": acct_id,
            "name": cfg.get("name", acct_id),
            "bridge_port": cfg.get("bridge_port", 5056),
            "account_size": cfg.get("account_size", 100_000),
            "max_daily_loss_pct": cfg.get("max_daily_loss_pct", 0.05),
            "max_total_dd_pct": cfg.get("max_total_dd_pct", 0.10),
            "audit_db": cfg.get("audit_db", f"audit/{acct_id}.db"),
        })
    return accounts


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


def get_daily_closed_pnl(db_path: str) -> float:
    """Sum of P&L from trades closed today."""
    try:
        conn = _db_connect(db_path)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades "
            "WHERE timestamp >= ? AND pnl IS NOT NULL",
            (today,),
        ).fetchone()
        conn.close()
        return row[0] if row else 0.0
    except Exception:
        return 0.0


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


def get_ftmo_status(equity: float, daily_start: float | None,
                    initial_balance: float = 100_000.0,
                    max_daily_loss_pct: float = 0.05,
                    max_total_dd_pct: float = 0.10) -> dict:
    if daily_start is None:
        daily_start = equity

    daily_loss = max(0, daily_start - equity)
    daily_dd_pct = (daily_loss / daily_start * 100) if daily_start > 0 else 0

    total_loss = max(0, initial_balance - equity)
    total_dd_pct = (total_loss / initial_balance * 100)

    return {
        "daily_dd_pct": daily_dd_pct,
        "daily_dd_limit": max_daily_loss_pct * 100,
        "daily_ok": daily_dd_pct < max_daily_loss_pct * 100,
        "total_dd_pct": total_dd_pct,
        "total_dd_limit": max_total_dd_pct * 100,
        "total_ok": total_dd_pct < max_total_dd_pct * 100,
    }


# ── Rendering ────────────────────────────────────────────────────────

def render_account(acct_cfg: dict, data: dict) -> list[str]:
    """Render a single account section."""
    out: list[str] = []
    name = acct_cfg["name"]
    port = acct_cfg["bridge_port"]
    initial_balance = acct_cfg["account_size"]

    # ── Account header ──
    out.append(line(f"{C.BOLD}{C.CYAN}{name}{C.RESET}  (port {port})"))

    acc = data.get("account")
    if acc:
        out.append(line(
            f"Balance: ${acc['balance']:,.0f}    "
            f"Equity: ${acc['equity']:,.0f}    "
            f"Margin Free: ${acc['margin_free']:,.0f}"
        ))
        dpnl = data.get("daily_pnl", 0)
        ds = acc["equity"] - dpnl if dpnl else acc["equity"]
        dpct = (dpnl / ds * 100) if ds else 0
        tpnl = acc["equity"] - initial_balance
        tpct = tpnl / initial_balance * 100
        dc, tc = pnl_color(dpnl), pnl_color(tpnl)
        out.append(line(
            f"Daily P&L: {dc}{pnl_str(dpnl)} "
            f"({'+' if dpct >= 0 else ''}{dpct:.2f}%){C.RESET}    "
            f"Total P&L: {tc}{pnl_str(tpnl)} "
            f"({'+' if tpct >= 0 else ''}{tpct:.2f}%){C.RESET}"
        ))
    else:
        out.append(line(f"{C.RED}MT5 bridge unavailable — no live data{C.RESET}"))

    # ── Compliance ──
    ftmo = data.get("ftmo", {})
    bs = data.get("bot_status", {})
    if ftmo:
        dd = ftmo["daily_dd_pct"]
        dl = ftmo["daily_dd_limit"]
        bar = progress_bar(dd, dl)
        st = f"{C.GREEN}OK{C.RESET}" if ftmo["daily_ok"] else f"{C.BG_RED}{C.WHITE}BREACH{C.RESET}"
        out.append(line(f"Daily DD:  {dd:>5.2f}% / {dl:.2f}%  {bar}  {st}"))

        td = ftmo["total_dd_pct"]
        tl = ftmo["total_dd_limit"]
        bar2 = progress_bar(td, tl)
        st2 = f"{C.GREEN}OK{C.RESET}" if ftmo["total_ok"] else f"{C.BG_RED}{C.WHITE}BREACH{C.RESET}"
        out.append(line(f"Total DD:  {td:>5.2f}% / {tl:.1f}%  {bar2}  {st2}"))

    # ── Positions (group split orders by symbol+direction) ──
    positions = data.get("positions")
    if positions is not None:
        total_p = sum(p["profit"] for p in positions)
        pc = pnl_color(total_p)

        # Group by (symbol, direction) to merge split orders
        grouped: dict[tuple[str, str], dict] = {}
        for p in positions:
            key = (p["symbol"], p["direction"])
            if key not in grouped:
                grouped[key] = {"symbol": p["symbol"], "direction": p["direction"],
                                "volume": 0.0, "profit": 0.0, "count": 0,
                                "any_active": False}
            g = grouped[key]
            g["volume"] += p["volume"]
            g["profit"] += p["profit"]
            g["count"] += 1
            if p["trail"] == "ACTIVE":
                g["any_active"] = True

        n_logical = len(grouped)
        n_raw = len(positions)
        count_label = f"{n_logical} open" if n_logical == n_raw else f"{n_logical} open ({n_raw} orders)"
        out.append(line(
            f"Positions: {count_label}, "
            f"P&L: {pc}{pnl_str(total_p)}{C.RESET}    "
            f"Trades today: {bs.get('trades_today', 0)}    "
            f"SL mods: {bs.get('sl_mods_today', 0)}"
        ))
        for g in grouped.values():
            sym = g["symbol"][:10].ljust(10)
            d = g["direction"].ljust(4)
            vol = f"{g['volume']:.2f}"
            pc2 = pnl_color(g["profit"])
            ps = pnl_str(g["profit"])
            trail = "ACTIVE" if g["any_active"] else "pending"
            trc = C.GREEN if trail == "ACTIVE" else C.DIM
            split = f"  {C.DIM}({g['count']}×){C.RESET}" if g["count"] > 1 else ""
            out.append(line(
                f"  {sym} {d} {vol} lots  {pc2}{ps}{C.RESET}"
                f"  Trail: {trc}{trail}{C.RESET}{split}"
            ))
    else:
        out.append(line(f"Positions: {C.DIM}unavailable{C.RESET}"))

    # ── Status bar ──
    bot = bs.get("bot", "UNKNOWN")
    m5 = bs.get("mt5", "UNKNOWN")
    ls = bs.get("last_scan", "N/A")
    bc = C.GREEN if bot == "RUNNING" else C.RED if bot == "STALE" else C.YELLOW
    mc = C.GREEN if m5 == "CONNECTED" else C.RED
    inact = bs.get("inactivity_days", 0)
    inact_s = f"{C.YELLOW}{inact}d{C.RESET}" if inact > 3 else f"{inact}d"
    out.append(line(
        f"Bot: {bc}{bot}{C.RESET}  "
        f"MT5: {mc}{m5}{C.RESET}  "
        f"Scan: {ls}  "
        f"Inactivity: {inact_s}"
    ))

    return out


def render_recent(all_data: dict[str, dict], account_cfgs: list[dict]) -> list[str]:
    """Render combined recent trades and events from all accounts."""
    out: list[str] = []

    # ── Recent Trades (merged, sorted by time) ──
    all_trades = []
    for acfg in account_cfgs:
        aid = acfg["id"]
        label = acfg["name"][:6]
        for t in all_data.get(aid, {}).get("recent_trades", []):
            all_trades.append({**t, "account": label})

    # Sort by timestamp descending
    all_trades.sort(key=lambda t: t.get("timestamp") or "", reverse=True)
    trades = all_trades[:5]

    out.append(line(f"{C.BOLD}RECENT TRADES{C.RESET} (last {len(trades)})"))
    if trades:
        for t in trades:
            try:
                dt = _parse_ts(t["timestamp"])
                ts = dt.strftime("%d %b %H:%M")
            except Exception:
                ts = (t["timestamp"] or "")[:12]
            acct = f"[{t['account']}]".ljust(8)
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
            out.append(line(f"{ts} {acct} {sym} {d} {pc2}{ps}{C.RESET}{cs}"))
    else:
        out.append(line(f"{C.DIM}No recent trades{C.RESET}"))

    # ── Recent Events (merged) ──
    all_events = []
    for acfg in account_cfgs:
        aid = acfg["id"]
        label = acfg["name"][:6]
        for e in all_data.get(aid, {}).get("recent_events", []):
            all_events.append({**e, "account": label})

    all_events.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
    events = all_events[:5]

    out.append(line())
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
            comp = (e.get("component") or "")[:12].ljust(12)
            etype = (e.get("event_type") or "")[:8]
            msg = (e.get("message") or "")[:20]
            out.append(line(
                f"{ts}  {lc}{lvl:<7}{C.RESET}  {comp}  {etype}  {msg}"
            ))
    else:
        out.append(line(f"{C.DIM}No recent events{C.RESET}"))

    return out


def render(account_cfgs: list[dict], all_data: dict[str, dict]) -> str:
    out: list[str] = []
    now = datetime.now().strftime("%d %b %Y %H:%M")

    # ── Header ──
    out.append(f"╔{'═' * WIDTH}╗")
    title = "SOVEREIGN BOT MONITOR"
    hdr = f"{C.BOLD}{C.CYAN}{title}{C.RESET}"
    n_accounts = len(account_cfgs)
    subtitle = f"{n_accounts} account{'s' if n_accounts > 1 else ''}"
    hdr_pad = WIDTH - 4 - len(title) - len(now) - len(subtitle) - 3
    out.append(f"║  {hdr}  {subtitle}{' ' * max(1, hdr_pad)}{now}  ║")
    out.append(f"╠{'═' * WIDTH}╣")

    # ── Per-account sections ──
    for i, acfg in enumerate(account_cfgs):
        aid = acfg["id"]
        data = all_data.get(aid, {})
        out.extend(render_account(acfg, data))
        if i < len(account_cfgs) - 1:
            out.append(f"╠{'─' * WIDTH}╣")

    out.append(f"╠{'═' * WIDTH}╣")

    # ── Hardware (shared) ──
    temps = all_data.get("_temps", {})
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

    # ── Recent trades & events (merged from all accounts) ──
    out.extend(render_recent(all_data, account_cfgs))
    out.append(line())

    out.append(f"╚{'═' * WIDTH}╝")
    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────

def collect_data(acct_cfg: dict) -> dict:
    port = acct_cfg["bridge_port"]
    db_path = str(REPO_ROOT / acct_cfg["audit_db"])
    initial_balance = acct_cfg["account_size"]

    mt5 = MT5BridgeClient(host="127.0.0.1", port=port, timeout=5)

    data: dict = {}
    data["account"] = get_account_data(mt5)
    data["positions"] = get_positions(mt5)

    # Daily P&L = closed trades P&L today + floating P&L
    closed_pnl = get_daily_closed_pnl(db_path)
    floating_pnl = data["account"]["profit"] if data["account"] else 0
    data["daily_pnl"] = closed_pnl + floating_pnl
    data["daily_start"] = None  # Not used anymore

    equity = data["account"]["equity"] if data["account"] else initial_balance
    # For DD calculation, derive daily_start from daily_pnl
    daily_start_calc = equity - data["daily_pnl"] if data["daily_pnl"] else equity
    data["ftmo"] = get_ftmo_status(
        equity, daily_start_calc,
        initial_balance=initial_balance,
        max_daily_loss_pct=acct_cfg["max_daily_loss_pct"],
        max_total_dd_pct=acct_cfg["max_total_dd_pct"],
    )
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
    parser.add_argument("--account", type=str, default=None,
                        help="Show only this account (by id)")
    args = parser.parse_args()

    account_cfgs = load_account_configs()

    if args.account:
        account_cfgs = [a for a in account_cfgs if a["id"] == args.account]
        if not account_cfgs:
            print(f"Account '{args.account}' not found in accounts.yaml")
            sys.exit(1)

    def refresh():
        all_data: dict[str, dict] = {}
        for acfg in account_cfgs:
            all_data[acfg["id"]] = collect_data(acfg)
        all_data["_temps"] = get_temperatures()
        return all_data

    if args.once:
        all_data = refresh()
        print(render(account_cfgs, all_data))
        return

    try:
        while True:
            all_data = refresh()
            os.system("clear")
            print(render(account_cfgs, all_data))
            print(f"  Auto-refresh: {args.interval}s | Press Ctrl+C to exit")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
