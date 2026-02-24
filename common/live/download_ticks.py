#!/usr/bin/env python3
"""
Tick Data Downloader (Wine-side)
================================
Standalone tick data downloader for all configured symbols.
Downloads monthly parquet files to data drives.

Usage (via Wine launcher):
    run_wine.sh download_ticks.py
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYMBOLS_FILE = os.path.join(REPO_ROOT, "data", "symbols.xlsx")
CONFIGS_FILE = os.path.join(REPO_ROOT, "config", "sovereign_configs.json")

DRIVE_PRIORITY = [
    "/home/tradebot/ssd_data_1",
    "/home/tradebot/ssd_data_2",
    "/home/tradebot/data_1",
    "/home/tradebot/data_2",
    "/home/tradebot/data_3",
]
DATA_DIR_NAME = "tick_data"
MIN_FREE_GB = 10
YEARS_BACK = 10


def log(level, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [TICK-DL] [{level.upper()}] {msg}")


def load_symbols():
    """Load symbols from sovereign_configs.json + symbols.xlsx."""
    symbols = set()

    # From sovereign_configs.json (these are the actively traded symbols)
    try:
        with open(CONFIGS_FILE) as f:
            cfg = json.load(f)
        symbols.update(cfg.keys())
        log("info", f"Loaded {len(cfg)} symbols from sovereign_configs.json")
    except Exception as e:
        log("warning", f"Could not read sovereign_configs.json: {e}")

    # From symbols.xlsx (full universe)
    try:
        sheets = pd.read_excel(SYMBOLS_FILE, sheet_name=None, header=None)
        for _, df in sheets.items():
            if not df.empty:
                symbols.update(df.iloc[:, 0].dropna().astype(str).tolist())
        log("info", f"Total {len(symbols)} unique symbols after merging with symbols.xlsx")
    except Exception as e:
        log("warning", f"Could not read symbols.xlsx: {e}")

    return sorted(symbols)


def free_gb(path):
    try:
        total, used, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    except FileNotFoundError:
        return 0.0


def pick_drive(start_idx=0):
    for i in range(start_idx, len(DRIVE_PRIORITY)):
        space = free_gb(DRIVE_PRIORITY[i])
        if space > MIN_FREE_GB:
            log("info", f"Using drive {DRIVE_PRIORITY[i]} ({space:.1f} GB free)")
            return i, DRIVE_PRIORITY[i]
    return -1, None


def resolve_symbol(symbol):
    for variant in [symbol, symbol.replace("/", ""), symbol.replace("/", "."), symbol.replace(" ", "")]:
        if variant and mt5.symbol_select(variant, True):
            return variant
    return None


def month_range(year, month):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    nm = month + 1 if month < 12 else 1
    ny = year if month < 12 else year + 1
    end = datetime(ny, nm, 1, tzinfo=timezone.utc)
    return start, end


def has_ticks(mt5_sym, year, month):
    start, end = month_range(year, month)
    ticks = mt5.copy_ticks_from(mt5_sym, start, 1, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return False
    t = pd.to_datetime(int(ticks[0]["time"]), unit="s", utc=True).to_pydatetime()
    return start <= t < end


def fetch_ticks(mt5_sym, start, end, batch=5000):
    chunks = []
    cursor = start
    while cursor < end:
        ticks = mt5.copy_ticks_from(mt5_sym, cursor, batch, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            break
        df = pd.DataFrame(ticks)
        if df.empty:
            break
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[df["time"] < end]
        if df.empty:
            break
        chunks.append(df)
        last_msc = int(df["time_msc"].iloc[-1])
        cursor = pd.to_datetime(last_msc + 1, unit="ms", utc=True).to_pydatetime()
        if len(df) < batch:
            break
    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(subset=["time_msc", "bid", "ask", "last", "volume", "flags"])


def main():
    log("info", "Starting tick data downloader")
    time.sleep(5)

    if not mt5.initialize():
        log("critical", f"MT5 init failed: {mt5.last_error()}")
        return
    log("info", f"MT5 connected: build {mt5.version()}")

    symbols = load_symbols()
    if not symbols:
        log("critical", "No symbols found")
        mt5.shutdown()
        return

    drv_idx, drive = pick_drive()
    if drive is None:
        mt5.shutdown()
        return

    now = datetime.now(timezone.utc)
    start_year = now.year - YEARS_BACK + 1

    # Build month lists
    months_desc = []
    months_asc = []
    for y in range(now.year, start_year - 1, -1):
        max_m = now.month if y == now.year else 12
        months_desc.extend([(y, m) for m in range(max_m, 0, -1)])
        months_asc.extend([(y, m) for m in range(1, max_m + 1)])

    # Resolve symbols and detect bounds
    meta = []
    for sym in symbols:
        mt5_sym = resolve_symbol(sym)
        if not mt5_sym:
            log("warning", f"Symbol '{sym}' not available in MT5, skipping")
            continue

        latest = earliest = None
        for y, m in months_desc:
            if has_ticks(mt5_sym, y, m):
                latest = y * 100 + m
                break
        if latest is None:
            log("warning", f"No tick data for '{sym}', skipping")
            continue
        for y, m in months_asc:
            if has_ticks(mt5_sym, y, m):
                earliest = y * 100 + m
                break

        meta.append({"sym": sym, "mt5": mt5_sym, "dir": sym.replace("/", "_"), "lo": earliest, "hi": latest})
        log("info", f"{sym}: data range {earliest}..{latest}")

    log("info", f"Downloading {len(meta)} symbols")

    # Download newest months first
    for y, m in months_desc:
        key = y * 100 + m
        for s in meta:
            if key < s["lo"] or key > s["hi"]:
                continue

            # Check drive space
            if free_gb(drive) < MIN_FREE_GB:
                drv_idx, drive = pick_drive(drv_idx + 1)
                if drive is None:
                    log("critical", "No drives with space left")
                    mt5.shutdown()
                    return

            target_dir = os.path.join(drive, DATA_DIR_NAME, s["dir"])
            os.makedirs(target_dir, exist_ok=True)
            fpath = os.path.join(target_dir, f"{y}-{m:02d}.parquet")

            # Always refresh the current month (it grows daily)
            is_current_month = (y == now.year and m == now.month)
            if os.path.exists(fpath) and not is_current_month:
                continue

            start, end = month_range(y, m)
            try:
                df = fetch_ticks(s["mt5"], start, end)
                if df is None or len(df) == 0:
                    log("warning", f"No ticks for {s['sym']} {y}-{m:02d}")
                    continue
                df.to_parquet(fpath, index=False, compression="snappy")
                log("ok", f"{s['sym']} {y}-{m:02d}: {len(df)} ticks -> {fpath}")
            except Exception as e:
                log("error", f"{s['sym']} {y}-{m:02d}: {e}")

    log("info", "Download complete")
    mt5.shutdown()


if __name__ == "__main__":
    main()
