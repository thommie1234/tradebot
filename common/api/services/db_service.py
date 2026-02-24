"""SQLite read service — queries heartbeats, trades, events (multi-account)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from api.deps import get_db


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
