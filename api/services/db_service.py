"""SQLite read service — queries heartbeats, trades, events."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from api.deps import get_db


def get_latest_heartbeat() -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM heartbeats ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def get_heartbeats_since(hours: int = 24, max_points: int = 200) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM heartbeats WHERE timestamp >= ?", (cutoff,)
        ).fetchone()[0]

        # Downsample if too many points
        step = max(1, total // max_points)
        rows = conn.execute(
            "SELECT * FROM heartbeats WHERE timestamp >= ? AND id % ? = 0 ORDER BY timestamp",
            (cutoff, step),
        ).fetchall()
        return [dict(r) for r in rows]


def get_trades_today() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY id DESC",
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_trades(limit: int = 3) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_trade_history(
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

    with get_db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE {where}", params
        ).fetchone()[0]

        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, page_size, offset],
        ).fetchall()

        return [dict(r) for r in rows], total


def get_trade_stats(days: int | None = None) -> dict:
    conditions = []
    params: list = []

    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conditions.append("timestamp >= ?")
        params.append(cutoff)

    # Only count closed trades with a pnl value
    conditions.append("pnl IS NOT NULL")
    where = " AND ".join(conditions)

    with get_db() as conn:
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
        "total_pnl": sum(pnls),
        "trade_count": len(pnls),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": len(wins) / len(pnls) if pnls else None,
        "avg_win": sum(wins) / len(wins) if wins else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
    }


def get_equity_curve(hours: int = 168, max_points: int = 300) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
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


def get_daily_pnl(days: int = 30) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_db() as conn:
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
        return [{"date": r["date"], "pnl": r["pnl"]} for r in rows]


def get_last_trade_id() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT MAX(id) as max_id FROM trades").fetchone()
        return row["max_id"] or 0


def get_trades_since_id(trade_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE id > ? ORDER BY id", (trade_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def ensure_push_tokens_table():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS push_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)
        conn.commit()


def register_push_token(token: str) -> bool:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO push_tokens (token, created_at, active) VALUES (?, ?, 1)",
            (token, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    return True


def get_active_push_tokens() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT token FROM push_tokens WHERE active = 1"
        ).fetchall()
        return [r["token"] for r in rows]
