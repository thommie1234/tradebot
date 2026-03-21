"""
F16: LLM Trade Journal — automatic post-trade analysis via Ollama.

Generates LLM-powered analysis after each closed trade and weekly summaries.
Stores entries in SQLite for review.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta

from config.loader import cfg


class TradeJournal:
    """Automatic LLM trade journal — analyses each closed trade."""

    def __init__(self, db_path: str | None = None,
                 ollama_host: str | None = None,
                 ollama_model: str | None = None):
        self.db_path = db_path or cfg.DB_PATH
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self._init_table()

    def _connect(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_table(self):
        try:
            conn = self._connect()
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trade_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT,
                        pnl REAL,
                        hold_hours REAL,
                        features_json TEXT,
                        llm_analysis TEXT
                    )
                ''')
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            pass

    def _query_llm(self, prompt: str, system: str) -> str | None:
        """Query Ollama LLM. Returns response text or None on failure."""
        try:
            import requests
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass
        return None

    def journal_trade(self, trade_data: dict) -> str | None:
        """Generate LLM analysis of a closed trade and store in DB.

        trade_data keys: symbol, direction, pnl, hold_hours, entry_price,
                         exit_price, ml_confidence, features (dict)
        """
        symbol = trade_data.get("symbol", "?")
        direction = trade_data.get("direction", "?")
        pnl = trade_data.get("pnl", 0.0)
        hold_hours = trade_data.get("hold_hours", 0.0)
        features = trade_data.get("features", {})
        ml_conf = trade_data.get("ml_confidence", 0.0)

        prompt = (
            f"Trade gesloten:\n"
            f"  Symbol: {symbol}\n"
            f"  Richting: {direction}\n"
            f"  P&L: ${pnl:+.2f}\n"
            f"  Hold time: {hold_hours:.1f} uur\n"
            f"  ML confidence: {ml_conf:.3f}\n"
            f"  Entry: {trade_data.get('entry_price', '?')}\n"
            f"  Exit: {trade_data.get('exit_price', '?')}\n"
        )
        if features:
            prompt += "  Features bij entry:\n"
            for k, v in list(features.items())[:10]:
                prompt += f"    {k}: {v:.4f}\n"

        system = (
            "Je bent een trading-analist voor een algoritmische FTMO prop trading bot. "
            "Analyseer deze gesloten trade in 3-5 zinnen in het Nederlands. "
            "Focus op: was de entry goed? Was de hold time optimaal? "
            "Wat zeggen de features over marktcontext? Wat kan beter?"
        )

        analysis = self._query_llm(prompt, system)

        # Store in DB (even without LLM analysis)
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO trade_journal (timestamp, symbol, direction, pnl, "
                    "hold_hours, features_json, llm_analysis) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(),
                        symbol,
                        direction,
                        pnl,
                        hold_hours,
                        json.dumps(features) if features else "{}",
                        analysis or "",
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            pass

        return analysis

    def weekly_summary(self) -> str | None:
        """Generate weekly summary of all trades."""
        try:
            conn = self._connect()
            try:
                cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                rows = conn.execute(
                    "SELECT symbol, direction, pnl, hold_hours, llm_analysis "
                    "FROM trade_journal WHERE timestamp > ? ORDER BY timestamp",
                    (cutoff,),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return None

        if not rows:
            return None

        total_pnl = sum(r[2] or 0 for r in rows)
        wins = sum(1 for r in rows if (r[2] or 0) > 0)
        losses = len(rows) - wins

        lines = [f"Week overzicht: {len(rows)} trades, P&L=${total_pnl:+.2f}, W/L={wins}/{losses}"]
        for r in rows:
            lines.append(f"  {r[0]} {r[1]} P&L=${r[2]:+.2f} ({r[3]:.1f}h)")

        prompt = "\n".join(lines)
        system = (
            "Je bent een trading-analist. Analyseer deze week aan trades in 5-8 zinnen. "
            "Zoek patronen: welke symbols waren winstgevend? Welke verliesgevend? "
            "Zijn er timing-problemen? Suggesties voor verbetering? Nederlands."
        )
        return self._query_llm(prompt, system)

    def get_journal_entries(self, symbol: str | None = None, days: int = 7) -> list[dict]:
        """Retrieve journal entries for review."""
        try:
            conn = self._connect()
            try:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                if symbol:
                    rows = conn.execute(
                        "SELECT id, timestamp, symbol, direction, pnl, hold_hours, "
                        "features_json, llm_analysis FROM trade_journal "
                        "WHERE timestamp > ? AND symbol = ? ORDER BY timestamp DESC",
                        (cutoff, symbol),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, timestamp, symbol, direction, pnl, hold_hours, "
                        "features_json, llm_analysis FROM trade_journal "
                        "WHERE timestamp > ? ORDER BY timestamp DESC",
                        (cutoff,),
                    ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "id": r[0], "timestamp": r[1], "symbol": r[2],
                "direction": r[3], "pnl": r[4], "hold_hours": r[5],
                "features": json.loads(r[6]) if r[6] else {},
                "llm_analysis": r[7],
            }
            for r in rows
        ]
