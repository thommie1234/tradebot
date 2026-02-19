#!/usr/bin/env python3
"""
MT5 Bars Downloader (M1)
========================

Downloads M1 bar data for symbols listed in the /information folder.
Stores data month-by-month in Parquet to enable resuming.

Usage:
  python mt5_bars_downloader.py

Environment overrides:
  INFO_DIR                 Path to info folder with CSV/XLSX symbols
  DATA_ROOT_DIR_NAME       Base folder name on data drives (default: bars_m1)
  YEARS_TO_DOWNLOAD        How many years back to scan (default: 20)
  DRIVE_FREE_SPACE_GB      Min free space before switching drive (default: 10)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd

import MetaTrader5 as mt5_module


INFO_DIR = os.getenv("INFO_DIR", "/home/tradebot/tradebots/data/instrument_specs")
DATA_ROOT_DIR_NAME = os.getenv("DATA_ROOT_DIR_NAME", "bars_m1")
YEARS_TO_DOWNLOAD = int(os.getenv("YEARS_TO_DOWNLOAD", "20"))
DRIVE_FREE_SPACE_GB = float(os.getenv("DRIVE_FREE_SPACE_GB", "10"))

DRIVE_PRIORITY = [
    "/home/tradebot/ssd_data_1",
    "/home/tradebot/ssd_data_2",
    "/home/tradebot/data_1",
    "/home/tradebot/data_2",
    "/home/tradebot/data_3",
]


def print_log(level: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [M1-BARS] [{level.upper()}] {message}")


def get_available_space_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024**3)
    except FileNotFoundError:
        return 0.0


def select_active_drive(current_index: int) -> tuple[int, str | None]:
    for i in range(current_index, len(DRIVE_PRIORITY)):
        drive = DRIVE_PRIORITY[i]
        free_gb = get_available_space_gb(drive)
        print_log("info", f"Checking drive '{drive}': {free_gb:.2f} GB free.")
        if free_gb > DRIVE_FREE_SPACE_GB:
            print_log("info", f"Selected active drive: '{drive}'.")
            return i, drive
    print_log("critical", "No drives with enough free space found.")
    return -1, None


def sanitize_symbol_for_path(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(" ", "")


def load_symbols_from_information(info_dir: str) -> list[str]:
    symbols: set[str] = set()
    for name in os.listdir(info_dir):
        path = os.path.join(info_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        try:
            if lower.endswith(".csv"):
                df = pd.read_csv(path)
                if "symbol" in df.columns:
                    symbols.update(df["symbol"].dropna().astype(str).tolist())
                else:
                    symbols.update(df.iloc[:, 0].dropna().astype(str).tolist())
            elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                sheets = pd.read_excel(path, sheet_name=None, header=None)
                for _, df in sheets.items():
                    if not df.empty:
                        symbols.update(df.iloc[:, 0].dropna().astype(str).tolist())
        except Exception as exc:
            print_log("warning", f"Failed to read symbols from '{path}': {exc}")

    out = sorted({s.strip() for s in symbols if s and str(s).strip()})
    print_log("info", f"Loaded {len(out)} symbols from information folder.")
    return out


def resolve_mt5_symbol(mt5, requested_symbol: str) -> str | None:
    candidates = []
    for s in [
        requested_symbol,
        requested_symbol.replace("/", ""),
        requested_symbol.replace("/", "."),
        requested_symbol.replace(" ", ""),
    ]:
        if s and s not in candidates:
            candidates.append(s)
    for candidate in candidates:
        if mt5.symbol_select(candidate, True):
            return candidate
    return None


def month_start_end(year: int, month: int):
    from_date = datetime(year, month, 1, tzinfo=timezone.utc)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    to_date = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
    return from_date, to_date


def month_has_bars(mt5, symbol: str, year: int, month: int) -> bool:
    from_date, to_date = month_start_end(year, month)
    bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, from_date, to_date)
    return bars is not None and len(bars) > 0


def detect_symbol_month_bounds(mt5, symbol: str, months_desc, months_asc):
    latest_key = None
    earliest_key = None

    for y, m in months_desc:
        if month_has_bars(mt5, symbol, y, m):
            latest_key = y * 100 + m
            break
    if latest_key is None:
        return None, None

    for y, m in months_asc:
        if month_has_bars(mt5, symbol, y, m):
            earliest_key = y * 100 + m
            break
    return latest_key, earliest_key


def main() -> None:
    print_log("info", "Starting M1 bars downloader.")
    time.sleep(5)

    mt5 = mt5_module
    if not mt5.initialize():
        print_log("critical", f"MT5 initialize() failed: {mt5.last_error()}")
        return
    print_log("info", f"MT5 connected: build {mt5.version()}")

    symbols = load_symbols_from_information(INFO_DIR)
    if not symbols:
        print_log("critical", "No symbols found. Exiting.")
        mt5.shutdown()
        return

    current_drive_idx, active_drive = select_active_drive(0)
    if active_drive is None:
        mt5.shutdown()
        return

    now = datetime.now(timezone.utc)
    start_year = now.year - YEARS_TO_DOWNLOAD + 1
    years_desc = list(range(now.year, start_year - 1, -1))
    months_desc_all = []
    months_asc_all = []
    for y in years_desc:
        if y == now.year:
            md = list(range(now.month, 0, -1))
            ma = list(range(1, now.month + 1))
        else:
            md = list(range(12, 0, -1))
            ma = list(range(1, 13))
        months_desc_all.extend([(y, m) for m in md])
        months_asc_all.extend([(y, m) for m in ma])

    symbol_meta = []
    for symbol in symbols:
        mt5_symbol = resolve_mt5_symbol(mt5, symbol)
        if not mt5_symbol:
            print_log("warning", f"Could not select symbol '{symbol}' in MT5. Skipping.")
            continue
        latest_key, earliest_key = detect_symbol_month_bounds(mt5, mt5_symbol, months_desc_all, months_asc_all)
        if latest_key is None:
            print_log("warning", f"No M1 data found for '{symbol}'. Skipping.")
            continue
        symbol_meta.append({
            "symbol": symbol,
            "mt5_symbol": mt5_symbol,
            "symbol_dir": sanitize_symbol_for_path(symbol),
            "latest_key": latest_key,
            "earliest_key": earliest_key,
        })
        print_log("info", f"Range {symbol}: {earliest_key}..{latest_key}")

    for year in years_desc:
        if year == now.year:
            months_desc = list(range(now.month, 0, -1))
        else:
            months_desc = list(range(12, 0, -1))

        for month in months_desc:
            print_log("info", f"=== Processing batch {year}-{month:02d} across all symbols ===")
            month_key = year * 100 + month

            for meta in symbol_meta:
                symbol = meta["symbol"]
                mt5_symbol = meta["mt5_symbol"]
                if month_key > meta["latest_key"] or month_key < meta["earliest_key"]:
                    continue

                free_gb = get_available_space_gb(active_drive)
                if free_gb < DRIVE_FREE_SPACE_GB:
                    print_log("warning", f"Drive '{active_drive}' is low on space ({free_gb:.2f} GB left).")
                    new_idx, new_drive = select_active_drive(current_drive_idx + 1)
                    if new_drive is None:
                        print_log("critical", "No more drives with space. Stopping download.")
                        mt5.shutdown()
                        return
                    current_drive_idx = new_idx
                    active_drive = new_drive

                target_dir = os.path.join(active_drive, DATA_ROOT_DIR_NAME, meta["symbol_dir"])
                os.makedirs(target_dir, exist_ok=True)
                file_name = f"{year}-{month:02d}.parquet"
                target_path = os.path.join(target_dir, file_name)
                if os.path.exists(target_path):
                    print_log("info", f"Skipping '{target_path}' (already exists).")
                    continue

                from_date, to_date = month_start_end(year, month)
                if mt5_symbol != symbol:
                    print_log("info", f"Downloading {symbol} via MT5 symbol '{mt5_symbol}' for {year}-{month:02d}...")
                else:
                    print_log("info", f"Downloading {symbol} for {year}-{month:02d}...")

                try:
                    bars = mt5.copy_rates_range(mt5_symbol, mt5.TIMEFRAME_M1, from_date, to_date)
                    if bars is None or len(bars) == 0:
                        print_log("warning", f"No M1 bars for {symbol} in {year}-{month:02d}. MT5 error: {mt5.last_error()}")
                        continue

                    df = pd.DataFrame(bars)
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    df.to_parquet(target_path, index=False, compression="snappy")
                    print_log("success", f"Saved {len(df)} bars to '{target_path}'.")
                except Exception as exc:
                    print_log("error", f"Failed {symbol} {year}-{month:02d}: {exc}")

    print_log("info", "All symbols processed. Downloader finished.")
    mt5.shutdown()


if __name__ == "__main__":
    main()
