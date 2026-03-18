"""SQLite read service — queries heartbeats, trades, events (multi-account)."""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from api.deps import get_db, get_paper_db

_REPO_ROOT = Path(__file__).resolve().parents[3]


def get_latest_heartbeat(account_id: str) -> dict | None:
    with get_db(account_id) as conn:
        row = conn.execute(
            "SELECT * FROM heartbeats ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def get_heartbeats_since(account_id: str, hours: int = 24, max_points: int = 200) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db(account_id) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM heartbeats WHERE timestamp >= ?", (cutoff,)
        ).fetchone()[0]

        step = max(1, total // max_points)
        rows = conn.execute(
            "SELECT * FROM heartbeats WHERE timestamp >= ? AND id % ? = 0 ORDER BY timestamp",
            (cutoff, step),
        ).fetchall()
        return [dict(r) for r in rows]


def get_trades_today(account_id: str) -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_db(account_id) as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY id DESC",
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_trades(account_id: str, limit: int = 10) -> list[dict]:
    with get_db(account_id) as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_trade_history(
    account_id: str,
    days: int | None = None,
    symbol: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    conditions = []
    params: list = []

    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conditions.append("timestamp >= ?")
        params.append(cutoff)
    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = " AND ".join(conditions) if conditions else "1=1"

    with get_db(account_id) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE {where}", params
        ).fetchone()[0]

        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()

        return [dict(r) for r in rows], total


def get_trade_stats(account_id: str, days: int | None = None) -> dict:
    conditions = []
    params: list = []

    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conditions.append("timestamp >= ?")
        params.append(cutoff)

    conditions.append("pnl IS NOT NULL")
    where = " AND ".join(conditions)

    with get_db(account_id) as conn:
        rows = conn.execute(
            f"SELECT pnl FROM trades WHERE {where}", params
        ).fetchall()

    pnls = [r["pnl"] for r in rows]
    if not pnls:
        return {
            "total_pnl": 0, "trade_count": 0, "win_count": 0,
            "loss_count": 0, "win_rate": None, "avg_win": None,
            "avg_loss": None, "best_trade": None, "worst_trade": None,
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        "total_pnl": round(sum(pnls), 2),
        "trade_count": len(pnls),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / len(pnls), 3) if pnls else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
    }


def get_equity_curve(account_id: str, hours: int = 168, max_points: int = 300) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db(account_id) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM heartbeats WHERE timestamp >= ? AND account_equity > 0",
            (cutoff,),
        ).fetchone()[0]

        step = max(1, total // max_points)
        rows = conn.execute(
            "SELECT timestamp, account_equity FROM heartbeats "
            "WHERE timestamp >= ? AND account_equity > 0 AND id % ? = 0 ORDER BY timestamp",
            (cutoff, step),
        ).fetchall()
        return [{"timestamp": r["timestamp"], "equity": r["account_equity"]} for r in rows]


def get_daily_pnl(account_id: str, days: int = 30) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_db(account_id) as conn:
        rows = conn.execute(
            """
            SELECT DATE(timestamp) as date,
                   MAX(daily_pnl) as pnl
            FROM heartbeats
            WHERE timestamp >= ? AND daily_pnl IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            (cutoff,),
        ).fetchall()
        return [{"date": r["date"], "pnl": round(r["pnl"], 2) if r["pnl"] else 0} for r in rows]


def get_recent_events(account_id: str, limit: int = 20) -> list[dict]:
    with get_db(account_id) as conn:
        rows = conn.execute(
            "SELECT timestamp, level, component, event_type, message "
            "FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_daily_start_balance(account_id: str) -> float | None:
    """Get first heartbeat balance of today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_db(account_id) as conn:
        row = conn.execute(
            "SELECT account_balance FROM heartbeats WHERE timestamp >= ? "
            "AND account_balance > 0 ORDER BY id ASC LIMIT 1",
            (today,),
        ).fetchone()
        return row["account_balance"] if row else None


def get_high_water_mark(account_id: str) -> float | None:
    """Get highest balance ever recorded (for trailing DD calc)."""
    with get_db(account_id) as conn:
        row = conn.execute(
            "SELECT MAX(account_balance) as hwm FROM heartbeats WHERE account_balance > 0"
        ).fetchone()
        return row["hwm"] if row and row["hwm"] else None


def get_daily_closed_pnl(account_id: str) -> float:
    """Sum of closed trade P&L today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_db(account_id) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total "
            "FROM trades WHERE exit_timestamp >= ? AND pnl IS NOT NULL",
            (today,),
        ).fetchone()
        return round(row["total"], 2) if row else 0.0


# ── New analytics functions ────────────────────────────────────────────────────

def _window_stats(pnls: list[float]) -> dict:
    """Compute trade stats for a list of closed P&Ls."""
    if not pnls:
        return {
            "win_rate": None, "trade_count": 0, "avg_win": None,
            "avg_loss": None, "expectancy": None,
            "best_trade": None, "worst_trade": None,
        }
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    expectancy = round(aw * wr + al * (1 - wr), 2)
    return {
        "win_rate": round(wr, 3),
        "trade_count": len(pnls),
        "avg_win": round(aw, 2) if wins else None,
        "avg_loss": round(al, 2) if losses else None,
        "expectancy": expectancy,
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
    }


def get_extended_stats(account_id: str) -> dict:
    """Win rate / expectancy for three time windows: today, week, all-time."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    with get_db(account_id) as conn:
        def _fetch(cond: str, params: list) -> list[float]:
            rows = conn.execute(
                f"SELECT pnl FROM trades WHERE pnl IS NOT NULL AND {cond}", params
            ).fetchall()
            return [r["pnl"] for r in rows]

        today_pnls = _fetch("timestamp >= ?", [today])
        week_pnls = _fetch("timestamp >= ?", [week_cutoff])
        all_pnls = _fetch("1=1", [])

    return {
        "today": _window_stats(today_pnls),
        "week": _window_stats(week_pnls),
        "alltime": _window_stats(all_pnls),
    }


def get_symbol_stats(account_id: str) -> list[dict]:
    """Per-symbol trade count, win rate, total P&L and average P&L."""
    with get_db(account_id) as conn:
        rows = conn.execute(
            """
            SELECT symbol,
                   COUNT(*) as trade_count,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY symbol
            ORDER BY trade_count DESC
            """
        ).fetchall()
        return [
            {
                "symbol": r["symbol"],
                "trade_count": r["trade_count"],
                "wins": r["wins"] or 0,
                "total_pnl": round(r["total_pnl"], 2) if r["total_pnl"] else 0.0,
                "avg_pnl": round(r["avg_pnl"], 2) if r["avg_pnl"] else 0.0,
            }
            for r in rows
        ]


def get_autoclose_errors_today(account_id: str) -> int:
    """Count AutoClose CLOSE_FAILED events today."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_db(account_id) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE component='AutoClose' AND event_type='CLOSE_FAILED' "
            "AND timestamp >= ?",
            (today,),
        ).fetchone()
        return row[0] if row else 0


def get_last_signal_per_symbol(account_id: str) -> dict[str, str]:
    """Return {symbol: last_signal_timestamp} for all symbols that fired signals."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with get_db(account_id) as conn:
        rows = conn.execute(
            "SELECT message, MAX(timestamp) as last_ts "
            "FROM events "
            "WHERE (component='Signal' OR component='MultiTF') "
            "AND event_type='SIGNAL' "
            "AND timestamp >= ? "
            "GROUP BY SUBSTR(message, 1, INSTR(message, ' ') - 1)",
            (cutoff,),
        ).fetchall()
    result: dict[str, str] = {}
    for r in rows:
        msg = r["message"] or ""
        sym = msg.split(" ")[0] if msg else None
        if sym and r["last_ts"]:
            result[sym] = r["last_ts"]
    return result


def get_decay_events(account_id: str, days: int = 7) -> list[dict]:
    """Recent ModelDecay DECAY_DETECTED events."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_db(account_id) as conn:
        rows = conn.execute(
            "SELECT timestamp, message FROM events "
            "WHERE component='ModelDecay' AND event_type='DECAY_DETECTED' "
            "AND timestamp >= ? ORDER BY id DESC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_calmar_ratio(account_id: str, days: int = 90) -> float | None:
    """Calmar ratio: annualized return / max drawdown from equity curve."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_db(account_id) as conn:
        rows = conn.execute(
            "SELECT account_equity FROM heartbeats "
            "WHERE timestamp >= ? AND account_equity > 0 ORDER BY timestamp",
            (cutoff,),
        ).fetchall()
    if len(rows) < 10:
        return None
    equities = [r["account_equity"] for r in rows]
    start = equities[0]
    if start <= 0:
        return None
    total_return = (equities[-1] - start) / start
    annual_return = total_return * (365 / days)
    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    if max_dd < 0.0001:
        return None
    return round(annual_return / max_dd, 2)


def get_sharpe_ratio(account_id: str, days: int = 30) -> float | None:
    """Annualized Sharpe ratio from daily P&L series (risk-free = 0)."""
    daily = get_daily_pnl(account_id, days=days)
    pnls = [d["pnl"] for d in daily if d["pnl"] is not None]
    if len(pnls) < 5:
        return None
    n = len(pnls)
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / max(n - 1, 1)
    std = math.sqrt(variance)
    if std < 0.01:
        return None
    return round((mean / std) * math.sqrt(252), 2)


# ── Paper trading queries ─────────────────────────────────────────────────────

def get_paper_summary(account_id: str) -> dict | None:
    """Paper account overview: equity, open/closed stats, today's P&L."""
    with get_paper_db(account_id) as conn:
        if conn is None:
            return None

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Today's closed trades
        closed_today = conn.execute(
            "SELECT pnl, commission_usd FROM paper_trades "
            "WHERE account_id = ? AND status != 'OPEN' AND exit_timestamp LIKE ?",
            (account_id, f"{today}%"),
        ).fetchall()

        daily_pnl = sum(r["pnl"] or 0 for r in closed_today)
        daily_comm = sum(r["commission_usd"] or 0 for r in closed_today)
        daily_wins = sum(1 for r in closed_today if (r["pnl"] or 0) > 0)
        daily_losses = len(closed_today) - daily_wins

        # Open positions
        open_rows = conn.execute(
            "SELECT COUNT(*) as n FROM paper_trades "
            "WHERE account_id = ? AND status = 'OPEN'",
            (account_id,),
        ).fetchone()

        # All-time
        alltime = conn.execute(
            "SELECT COUNT(*) as n, COALESCE(SUM(pnl), 0) as total, "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins "
            "FROM paper_trades WHERE account_id = ? AND status != 'OPEN'",
            (account_id,),
        ).fetchone()

        # Latest equity
        eq_row = conn.execute(
            "SELECT equity FROM paper_equity "
            "WHERE account_id = ? ORDER BY id DESC LIMIT 1",
            (account_id,),
        ).fetchone()

        return {
            "equity": round(eq_row["equity"], 2) if eq_row else 100000.0,
            "daily_pnl": round(daily_pnl, 2),
            "daily_trades": len(closed_today),
            "daily_wins": daily_wins,
            "daily_losses": daily_losses,
            "daily_commission": round(daily_comm, 2),
            "win_rate_today": round(daily_wins / len(closed_today), 3) if closed_today else None,
            "open_positions": open_rows["n"] if open_rows else 0,
            "alltime_trades": alltime["n"] or 0,
            "alltime_pnl": round(alltime["total"] or 0, 2),
            "alltime_winrate": round((alltime["wins"] or 0) / max(alltime["n"] or 1, 1), 3),
        }


def get_paper_open_positions(account_id: str) -> list[dict]:
    """Get all currently open paper positions."""
    with get_paper_db(account_id) as conn:
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT id, symbol, direction, entry_price, sl_price, tp_price, "
            "lot_size, confidence, bars_held, timestamp, atr, contract_size, "
            "commission_usd, timeframe, high_water "
            "FROM paper_trades WHERE account_id = ? AND status = 'OPEN' "
            "ORDER BY timestamp DESC",
            (account_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_paper_recent_trades(account_id: str, limit: int = 10) -> list[dict]:
    """Get recently closed paper trades."""
    with get_paper_db(account_id) as conn:
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT symbol, direction, entry_price, exit_price, pnl, "
            "confidence, status, bars_held, exit_timestamp, commission_usd, timeframe "
            "FROM paper_trades WHERE account_id = ? AND status != 'OPEN' "
            "ORDER BY id DESC LIMIT ?",
            (account_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_paper_symbol_stats(account_id: str) -> list[dict]:
    """Per-symbol paper trade stats."""
    with get_paper_db(account_id) as conn:
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT symbol, timeframe, COUNT(*) as trade_count, "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "COALESCE(SUM(pnl), 0) as total_pnl, "
            "AVG(pnl) as avg_pnl "
            "FROM paper_trades WHERE account_id = ? AND status != 'OPEN' "
            "GROUP BY symbol, timeframe ORDER BY total_pnl DESC",
            (account_id,),
        ).fetchall()
        return [
            {
                "symbol": r["symbol"],
                "timeframe": r["timeframe"] or "H1",
                "trade_count": r["trade_count"],
                "wins": r["wins"] or 0,
                "total_pnl": round(r["total_pnl"], 2) if r["total_pnl"] else 0.0,
                "avg_pnl": round(r["avg_pnl"], 2) if r["avg_pnl"] else 0.0,
            }
            for r in rows
        ]


def get_paper_equity_curve(account_id: str, max_points: int = 200) -> list[dict]:
    """Paper equity curve from paper_equity table."""
    with get_paper_db(account_id) as conn:
        if conn is None:
            return []
        total = conn.execute(
            "SELECT COUNT(*) FROM paper_equity WHERE account_id = ?",
            (account_id,),
        ).fetchone()[0]
        step = max(1, total // max_points)
        rows = conn.execute(
            "SELECT timestamp, equity FROM paper_equity "
            "WHERE account_id = ? AND id % ? = 0 ORDER BY timestamp",
            (account_id, step),
        ).fetchall()
        return [{"timestamp": r["timestamp"], "equity": r["equity"]} for r in rows]


# ── Spike trades (orderbook spike detector) ──────────────────────────

def _get_spike_db(account_id: str):
    """Get spike_trades.db path for account."""
    prefix = account_id.split("_")[0]
    dir_map = {"bright": "bf"}
    folder = dir_map.get(prefix, prefix)
    path = _REPO_ROOT / folder / "audit" / "spike_trades.db"
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def get_spike_summary() -> dict | None:
    """Get spike trading summary across all accounts."""
    results = {}
    for folder in ["ftmo", "bf"]:
        path = _REPO_ROOT / folder / "audit" / "spike_trades.db"
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            open_count = conn.execute(
                "SELECT COUNT(*) FROM spike_trades WHERE status = 'OPEN'"
            ).fetchone()[0]
            closed_today = conn.execute(
                "SELECT COUNT(*) FROM spike_trades WHERE status = 'CLOSED' AND exit_timestamp >= ?",
                (today,),
            ).fetchone()[0]
            pnl_today = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM spike_trades WHERE status = 'CLOSED' AND exit_timestamp >= ?",
                (today,),
            ).fetchone()[0]
            total_closed = conn.execute(
                "SELECT COUNT(*) FROM spike_trades WHERE status = 'CLOSED'"
            ).fetchone()[0]
            total_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM spike_trades WHERE status = 'CLOSED'"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM spike_trades WHERE status = 'CLOSED' AND pnl > 0"
            ).fetchone()[0]
            events_1h = conn.execute(
                "SELECT COUNT(*) FROM spike_events WHERE timestamp >= ?",
                ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),),
            ).fetchone()[0]
            conn.close()

            results[folder] = {
                "open": open_count,
                "closed_today": closed_today,
                "pnl_today": round(pnl_today, 2),
                "total_closed": total_closed,
                "total_pnl": round(total_pnl, 2),
                "win_rate": round(wins / total_closed, 3) if total_closed > 0 else None,
                "events_1h": events_1h,
            }
        except Exception:
            continue
    return results if results else None


def get_spike_open_positions() -> list[dict]:
    """Get all open spike positions."""
    positions = []
    for folder in ["ftmo", "bf"]:
        path = _REPO_ROOT / folder / "audit" / "spike_trades.db"
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM spike_trades WHERE status = 'OPEN' ORDER BY timestamp DESC"
            ).fetchall()
            conn.close()
            for r in rows:
                positions.append({
                    "id": r["id"], "symbol": r["symbol"],
                    "direction": r["direction"], "entry_price": r["entry_price"],
                    "sl_price": r["sl_price"], "tp_price": r["tp_price"],
                    "risk_usd": r["risk_usd"], "imbalance_ratio": r["imbalance_ratio"],
                    "timestamp": r["timestamp"], "account": folder,
                })
        except Exception:
            continue
    return positions


def get_spike_recent_trades(limit: int = 10) -> list[dict]:
    """Get recent closed spike trades."""
    trades = []
    for folder in ["ftmo", "bf"]:
        path = _REPO_ROOT / folder / "audit" / "spike_trades.db"
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM spike_trades WHERE status = 'CLOSED' ORDER BY exit_timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            for r in rows:
                trades.append({
                    "id": r["id"], "symbol": r["symbol"],
                    "direction": r["direction"], "entry_price": r["entry_price"],
                    "exit_price": r["exit_price"], "pnl": r["pnl"],
                    "exit_reason": r["exit_reason"],
                    "imbalance_ratio": r["imbalance_ratio"],
                    "timestamp": r["timestamp"],
                    "exit_timestamp": r["exit_timestamp"],
                    "account": folder,
                })
        except Exception:
            continue
    return sorted(trades, key=lambda x: x.get("exit_timestamp", ""), reverse=True)[:limit]


def get_spike_recent_events(limit: int = 15) -> list[dict]:
    """Get recent spike detection events."""
    events = []
    for folder in ["ftmo", "bf"]:
        path = _REPO_ROOT / folder / "audit" / "spike_trades.db"
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM spike_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            for r in rows:
                events.append({
                    "timestamp": r["timestamp"], "symbol": r["symbol"],
                    "event_type": r["event_type"], "details": r["details"],
                })
        except Exception:
            continue
    return sorted(events, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
