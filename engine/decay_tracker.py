"""
Model decay tracker — Z-score audit of live vs backtest performance.

Extracted from ModelDecayTracker in sovereign_bot.py (lines 870-1031).
F14: Auto-retrain trigger when decay detected (instead of just disabling).
"""
from __future__ import annotations

import os
import sqlite3
import time as _time
from datetime import datetime

from config.loader import cfg


class ModelDecayTracker:
    """
    Compares live trading performance vs backtest expectations.
    If live results deviate > 2 sigma, attempts auto-retrain first.
    Only disables if retrain fails.

    Tracks per-symbol:
    - Rolling win rate (last 50 trades)
    - Rolling profit factor
    - Z-score vs backtest baseline
    """

    DECAY_THRESHOLD = 2.0   # Standard deviations
    MIN_TRADES = 20         # Minimum trades before checking
    RETRAIN_COOLDOWN = 86400  # 24 hours between retrains per symbol

    def __init__(self, logger, db_path=None):
        self.logger = logger
        self.db_path = db_path or cfg.DB_PATH
        self._db_dir = os.path.dirname(self.db_path)
        self._init_table()
        self.baselines = {}   # {symbol: {win_rate, profit_factor, avg_trade}}
        self.disabled = set()  # symbols disabled due to decay
        self._retrain_timestamps = {}  # {symbol: last_retrain_epoch}

    def _connect(self):
        if self._db_dir:
            os.makedirs(self._db_dir, exist_ok=True)
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
                    CREATE TABLE IF NOT EXISTS live_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT,
                        pnl REAL,
                        win INTEGER,
                        ml_confidence REAL
                    )
                ''')
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            self.logger.log('WARNING', 'ModelDecay', 'DB_ERROR', str(e))

    def set_baseline(self, symbol: str, win_rate: float, profit_factor: float):
        self.baselines[symbol] = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def load_baselines_from_config(self):
        for sym, sym_cfg in cfg.SYMBOLS.items():
            pf = sym_cfg.get("profit_factor", 1.0)
            est_wr = pf / (1.0 + pf) if pf > 0 else 0.5
            self.baselines[sym] = {
                "win_rate": est_wr,
                "profit_factor": pf,
            }

    def record_trade(self, symbol: str, pnl: float, direction: str = "",
                     ml_confidence: float = 0.0):
        win = 1 if pnl > 0 else 0
        try:
            conn = self._connect()
            try:
                conn.execute('''
                    INSERT INTO live_trades (timestamp, symbol, direction, pnl, win, ml_confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), symbol, direction, pnl, win, ml_confidence))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            self.logger.log('WARNING', 'ModelDecay', 'DB_ERROR', str(e))

    def check_decay(self, symbol: str) -> tuple[bool, float]:
        baseline = self.baselines.get(symbol)
        if not baseline:
            return False, 0.0

        try:
            conn = self._connect()
            try:
                cursor = conn.execute('''
                    SELECT win, pnl FROM live_trades
                    WHERE symbol = ?
                    ORDER BY id DESC LIMIT 50
                ''', (symbol,))
                rows = cursor.fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            self.logger.log('WARNING', 'ModelDecay', 'DB_ERROR', str(e))
            return False, 0.0

        n = len(rows)
        if n < self.MIN_TRADES:
            return False, 0.0

        wins = sum(r[0] for r in rows)
        live_wr = wins / n

        expected_wr = baseline["win_rate"]
        se = (expected_wr * (1 - expected_wr) / n) ** 0.5 if 0 < expected_wr < 1 else 0.01
        z_score = (live_wr - expected_wr) / max(se, 1e-6)

        is_decayed = z_score < -self.DECAY_THRESHOLD
        return is_decayed, z_score

    def audit_all(self, discord=None) -> list[str]:
        if cfg.DISABLE_ZSCORE:
            return []
        newly_disabled = []
        for symbol in list(self.baselines.keys()):
            if symbol in self.disabled:
                continue
            try:
                is_decayed, z_score = self.check_decay(symbol)
            except Exception as e:
                self.logger.log('WARNING', 'ModelDecay', 'AUDIT_ERROR', str(e))
                continue
            if is_decayed:
                baseline = self.baselines[symbol]
                self.logger.log('CRITICAL', 'ModelDecay', 'DECAY_DETECTED',
                                f'{symbol}: Z={z_score:.2f} — attempting auto-retrain')

                # F14: Try auto-retrain before disabling
                retrained = self._auto_retrain(symbol, discord)
                if retrained:
                    self.disabled.discard(symbol)
                    # Clear trade history for this symbol so fresh evaluation starts
                    self._clear_trades(symbol)
                    if discord:
                        discord.send(
                            f"AUTO-RETRAIN: {symbol}",
                            f"Z-Score: {z_score:+.2f} (threshold: -2.0)\n"
                            f"Baseline WR: {baseline['win_rate']:.1%}\n"
                            f"Action: Model RETRAINED successfully",
                            "orange"
                        )
                else:
                    self.disabled.add(symbol)
                    newly_disabled.append(symbol)
                    if discord:
                        discord.send(
                            f"MODEL DECAY: {symbol}",
                            f"Z-Score: {z_score:+.2f} (threshold: -2.0)\n"
                            f"Baseline WR: {baseline['win_rate']:.1%}\n"
                            f"Action: Symbol DISABLED (retrain failed)",
                            "red"
                        )
        return newly_disabled

    def _auto_retrain(self, symbol: str, discord=None) -> bool:
        """Attempt auto-retrain for a decayed symbol. Returns True if successful."""
        # Cooldown: max 1 retrain per symbol per 24h
        now = _time.time()
        last_retrain = self._retrain_timestamps.get(symbol, 0)
        if now - last_retrain < self.RETRAIN_COOLDOWN:
            hours_ago = (now - last_retrain) / 3600
            self.logger.log('WARNING', 'ModelDecay', 'RETRAIN_COOLDOWN',
                            f'{symbol}: retrained {hours_ago:.1f}h ago, cooldown active')
            return False

        try:
            # Lazy import to avoid circular dependency
            from engine.inference import SovereignMLFilter
            filt = SovereignMLFilter(symbol, self.logger)
            success = filt.train_model()
            self._retrain_timestamps[symbol] = now

            if success:
                self.logger.log('INFO', 'ModelDecay', 'AUTO_RETRAIN_OK',
                                f'{symbol}: auto-retrain successful')
            else:
                self.logger.log('WARNING', 'ModelDecay', 'AUTO_RETRAIN_FAILED',
                                f'{symbol}: auto-retrain failed (no data/params)')
            return success

        except Exception as e:
            self.logger.log('ERROR', 'ModelDecay', 'AUTO_RETRAIN_ERROR',
                            f'{symbol}: {e}')
            return False

    def _clear_trades(self, symbol: str):
        """Clear trade history for a symbol after successful retrain."""
        try:
            conn = self._connect()
            try:
                conn.execute('DELETE FROM live_trades WHERE symbol = ?', (symbol,))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            self.logger.log('WARNING', 'ModelDecay', 'DB_ERROR', str(e))

    def is_disabled(self, symbol: str) -> bool:
        if cfg.DISABLE_ZSCORE:
            return False
        return symbol in self.disabled
