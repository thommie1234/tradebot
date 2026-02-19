"""
Feature logger — saves feature snapshots at trade time to parquet.

Extracted from FeatureLogger in sovereign_bot.py (lines 1037-1092).
"""
from __future__ import annotations

import os
from datetime import datetime

from config.loader import cfg


class FeatureLogger:
    """Saves all 28 feature values at the moment of each trade to parquet."""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir or cfg.TRADE_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self._buffer = []

    def log_trade_features(self, symbol: str, direction: str, confidence: float,
                           features_dict: dict, entry_price: float = 0.0,
                           sl_price: float = 0.0, tp_price: float = 0.0,
                           lot_size: float = 0.0, status: str = "SIGNAL"):
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "lot_size": lot_size,
            "status": status,
        }
        record.update(features_dict)
        self._buffer.append(record)

        if len(self._buffer) >= 10:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        import polars as pl

        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.log_dir, f"trade_features_{today}.parquet")

        df = pl.DataFrame(self._buffer)
        if os.path.exists(path):
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df], how="diagonal")
        df.write_parquet(path)
        self._buffer.clear()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass
