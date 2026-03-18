#!/usr/bin/env python3
"""
TICK RECORDER — Continuous real-time tick storage
==================================================

Records live ticks from MT5 bridge for all portfolio symbols,
stores as daily parquet files compatible with existing data pipeline.

Runs as a standalone systemd service.

Usage:
    python3 tick_recorder.py                # Record from all account bridges
    python3 tick_recorder.py --port 5057    # Record from specific bridge only
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT / "common") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "common"))

import MetaTrader5 as _mt5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where to store recorded ticks (first SSD with space)
DATA_ROOTS = [
    Path(r"C:\tick_data\ssd1"),
    Path(r"C:\tick_data\ssd2"),
    Path(r"C:\tick_data\nvme"),
]

BATCH_SIZE = 5000           # Ticks per MT5 request
POLL_INTERVAL = 10          # Seconds between fetch cycles
FLUSH_INTERVAL = 300        # Flush parquet to disk every 5 min
MIN_FREE_GB = 5             # Skip disk if less than this free

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TickRecorder")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_data_root() -> Path | None:
    """Find first data root with enough free space."""
    for root in DATA_ROOTS:
        root.mkdir(parents=True, exist_ok=True)
        stat = os.statvfs(root)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if free_gb >= MIN_FREE_GB:
            return root
    return None


ACCOUNT_PORTS = {"ftmo": 5056, "bf": 5057}


def _load_portfolio_symbols() -> dict[str, list[tuple[str, int]]]:
    """Load symbols from account configs.

    Returns {internal_name: [(broker_symbol, port), ...]}.
    A symbol may appear in multiple accounts with different broker names.
    """
    symbols: dict[str, list[tuple[str, int]]] = {}
    for acct_dir, port in ACCOUNT_PORTS.items():
        cfg_path = REPO_ROOT / acct_dir / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    data = json.load(f)
                for sym, val in data.items():
                    if isinstance(val, dict) and sym != "margin_leverage":
                        broker_sym = val.get("broker_symbol", sym)
                        symbols.setdefault(sym, []).append((broker_sym, port))
            except Exception:
                pass
    return symbols


# ---------------------------------------------------------------------------
# TickRecorder
# ---------------------------------------------------------------------------

class TickRecorder:
    """Fetches ticks from MT5 and saves daily parquets."""

    def __init__(self, ports: list[int] | None = None):
        self.mt5 = _mt5
        self._portfolio_mtime: dict[str, float] = {}

        # Per-symbol state: last fetched timestamp (epoch ms)
        self._cursors: dict[str, int] = {}
        self._last_flush = time.time()

        # Connect to MT5 (singleton)
        if not self.mt5.terminal_info():
            if not self.mt5.initialize():
                log.error("MT5 init failed: %s", self.mt5.last_error())
        acct = self.mt5.account_info()
        if acct:
            log.info("Connected to MT5 (account %s)", acct.login)

        # Load portfolio and assign each symbol to broker name
        raw_symbols = _load_portfolio_symbols()
        self._sym_targets: dict[str, str] = {}  # internal_name → broker_symbol
        for sym, entries in raw_symbols.items():
            for broker_sym, _port in entries:
                self._sym_targets[sym] = broker_sym
                break

        self._buffers: dict[str, list] = {sym: [] for sym in self._sym_targets}

        log.info("Recording %d symbols", len(self._sym_targets))
        for sym, bsym in self._sym_targets.items():
            log.info("  %s -> %s", sym, bsym)

    def _refresh_portfolio(self):
        """Reload portfolio if configs changed."""
        changed = False
        for acct_dir in ACCOUNT_PORTS:
            cfg_path = REPO_ROOT / acct_dir / "config.json"
            if cfg_path.exists():
                mtime = cfg_path.stat().st_mtime
                if mtime != self._portfolio_mtime.get(acct_dir, 0):
                    changed = True
                    self._portfolio_mtime[acct_dir] = mtime

        if not changed:
            return

        raw_symbols = _load_portfolio_symbols()
        for sym, entries in raw_symbols.items():
            if sym not in self._sym_targets:
                for broker_sym, _port in entries:
                    self._sym_targets[sym] = broker_sym
                    self._buffers[sym] = []
                    log.info("Added symbol: %s -> %s", sym, broker_sym)
                    break

    def _fetch_ticks(self, symbol: str):
        """Fetch new ticks for a symbol since last cursor."""
        broker_sym = self._sym_targets.get(symbol)
        if not broker_sym:
            return

        # Start from last cursor or 1 hour ago
        if symbol not in self._cursors:
            start = datetime.now(timezone.utc) - timedelta(hours=1)
        else:
            start = datetime.fromtimestamp(self._cursors[symbol] / 1000.0, tz=timezone.utc)

        try:
            ticks = self.mt5.copy_ticks_from(broker_sym, start, BATCH_SIZE)
        except Exception as e:
            log.debug("Fetch failed %s: %s", symbol, e)
            return

        if not ticks or len(ticks) == 0:
            return

        # Filter to only new ticks (after cursor)
        cursor_msc = self._cursors.get(symbol, 0)
        new_ticks = []
        for t in ticks:
            msc = t.get("time_msc", 0)
            if msc > cursor_msc:
                new_ticks.append(t)

        if new_ticks:
            self._buffers[symbol].extend(new_ticks)
            self._cursors[symbol] = new_ticks[-1]["time_msc"]

    def _flush_to_disk(self):
        """Write buffered ticks to daily parquet files."""
        try:
            import polars as pl
        except ImportError:
            import pandas as pd
            pl = None

        data_root = _get_data_root()
        if not data_root:
            log.error("No data root with sufficient free space!")
            return

        flushed_total = 0
        for symbol, ticks in self._buffers.items():
            if not ticks:
                continue

            sym_dir = data_root / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)

            # Group by YYYY-MM for monthly files (compatible with existing pipeline)
            months: dict[str, list] = {}
            for t in ticks:
                ts = t.get("time_msc", t.get("time", 0) * 1000)
                dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
                key = dt.strftime("%Y-%m")
                months.setdefault(key, []).append(t)

            for month_key, month_ticks in months.items():
                parquet_path = sym_dir / f"{month_key}.parquet"

                # Build dataframe
                if pl:
                    df_new = pl.DataFrame({
                        "time": [datetime.fromtimestamp(t["time"], tz=timezone.utc) for t in month_ticks],
                        "bid": [float(t.get("bid", 0)) for t in month_ticks],
                        "ask": [float(t.get("ask", 0)) for t in month_ticks],
                        "last": [float(t.get("last", 0)) for t in month_ticks],
                        "volume": [int(t.get("volume", 0)) for t in month_ticks],
                        "time_msc": [int(t.get("time_msc", 0)) for t in month_ticks],
                        "flags": [int(t.get("flags", 0)) for t in month_ticks],
                        "volume_real": [float(t.get("volume_real", 0)) for t in month_ticks],
                    }).with_columns(
                        pl.col("time").cast(pl.Datetime("ms", "UTC"))
                    )

                    # Append to existing file
                    if parquet_path.exists():
                        df_existing = pl.read_parquet(parquet_path)
                        df_combined = pl.concat([df_existing, df_new])
                        # Deduplicate on time_msc + bid + ask
                        df_combined = df_combined.unique(
                            subset=["time_msc", "bid", "ask", "last", "volume", "flags"],
                            keep="first",
                        ).sort("time_msc")
                    else:
                        df_combined = df_new.sort("time_msc")

                    df_combined.write_parquet(parquet_path, compression="snappy")
                    flushed_total += len(month_ticks)

            # Clear buffer
            self._buffers[symbol] = []

        if flushed_total > 0:
            log.info("Flushed %d ticks to disk", flushed_total)

        self._last_flush = time.time()

    def run(self):
        """Main loop: fetch ticks → buffer → flush to parquet."""
        if not self.mt5.terminal_info():
            log.error("MT5 not connected. Exiting.")
            return

        log.info("Starting tick recorder: %d symbols, poll=%ds, flush=%ds",
                 len(self._sym_targets), POLL_INTERVAL, FLUSH_INTERVAL)

        while True:
            try:
                # Refresh portfolio if configs changed
                self._refresh_portfolio()

                # Fetch new ticks for each symbol
                for symbol in list(self._sym_targets.keys()):
                    self._fetch_ticks(symbol)

                # Flush to disk periodically
                if time.time() - self._last_flush >= FLUSH_INTERVAL:
                    self._flush_to_disk()

                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                log.info("Shutting down — flushing remaining ticks...")
                self._flush_to_disk()
                break
            except Exception as e:
                log.error("Recorder error: %s", e)
                time.sleep(30)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time tick recorder")
    parser.add_argument("--port", type=int, nargs="+", help="MT5 bridge port(s)")
    args = parser.parse_args()

    ports = args.port if args.port else None
    recorder = TickRecorder(ports=ports)
    recorder.run()
