#!/usr/bin/env python3
"""
Tick Data Downloader (Wine-side) — account-agnostic version.

Downloads monthly parquet files for specified symbols.
Run via Wine launcher with the appropriate account env:

    source mt5/brightfunded/env.sh
    mt5/common/run_wine.sh tools/download_ticks_bridge.py BTC/USD ETH/USD SOL/USD

Or for all BF symbols:
    source mt5/brightfunded/env.sh
    mt5/common/run_wine.sh tools/download_ticks_bridge.py --all-bf

Saves to: /home/tradebot/ssd_data_1/tick_data/{SYMBOL}/{YYYY-MM}.parquet
"""
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd


DRIVE_PRIORITY = [
    "/home/tradebot/ssd_data_1",
    "/home/tradebot/ssd_data_2",
    "/home/tradebot/data_1",
]
DATA_DIR_NAME = "tick_data"
MIN_FREE_GB = 10
YEARS_BACK = 5
BATCH_SIZE = 50000

# All BrightFunded symbols
BF_ALL_SYMBOLS = [
    # Crypto
    "BTC/USD", "ETH/USD", "SOL/USD", "UNI/USD", "AAVE/USD", "ADA/USD",
    "DOT/USD", "XRP/USD", "LTC/USD", "BNB/USD", "LINK/USD", "DASH/USD",
    "ALGO/USD", "XLM/USD", "NEO/USD",
    # Indices
    "US30.cash", "US100.cash", "US500.cash", "EU50.cash", "UK100.cash", "FRA40.cash",
    # Metals
    "XAU/USD", "XAG/USD", "XPD/USD", "XPT/USD",
    # Forex
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDJPY", "NZDCAD", "NZDCHF",
    "CADJPY", "CADCHF", "CHFJPY", "USDSEK",
]


def log(level, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [TICK-DL] [{level.upper()}] {msg}", flush=True)


def free_gb(path):
    try:
        _, _, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    except FileNotFoundError:
        return 0.0


def pick_drive():
    for drv in DRIVE_PRIORITY:
        space = free_gb(drv)
        if space > MIN_FREE_GB:
            return drv
    return None


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


def fetch_ticks(mt5_sym, start, end):
    chunks = []
    cursor = start
    total = 0
    while cursor < end:
        ticks = mt5.copy_ticks_from(mt5_sym, cursor, BATCH_SIZE, mt5.COPY_TICKS_ALL)
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
        total += len(df)
        last_msc = int(df["time_msc"].iloc[-1])
        cursor = pd.to_datetime(last_msc + 1, unit="ms", utc=True).to_pydatetime()
        if len(df) < BATCH_SIZE:
            break
    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(subset=["time_msc", "bid", "ask", "last", "volume", "flags"])


def main():
    # Parse args
    args = sys.argv[1:]
    if "--all-bf" in args:
        symbols = BF_ALL_SYMBOLS
        args.remove("--all-bf")
    elif args:
        symbols = args
    else:
        print("Usage: download_ticks_bridge.py [--all-bf] [SYMBOL1 SYMBOL2 ...]")
        print("Example: download_ticks_bridge.py BTC/USD ETH/USD SOL/USD")
        return

    years = YEARS_BACK
    for i, a in enumerate(args):
        if a == "--years" and i + 1 < len(args):
            years = int(args[i + 1])

    log("info", f"Starting tick data downloader — {len(symbols)} symbols, {years} years")

    if not mt5.initialize():
        log("critical", f"MT5 init failed: {mt5.last_error()}")
        return
    log("info", f"MT5 connected: build {mt5.version()}")

    acct = mt5.account_info()
    if acct:
        log("info", f"Account: {acct.login} | Server: {acct.server}")

    drive = pick_drive()
    if not drive:
        log("critical", "No drives with space")
        mt5.shutdown()
        return
    log("info", f"Using drive: {drive} ({free_gb(drive):.1f} GB free)")

    now = datetime.now(timezone.utc)

    # Build descending month list (newest first for priority)
    months = []
    for y in range(now.year, now.year - years, -1):
        max_m = now.month if y == now.year else 12
        for m in range(max_m, 0, -1):
            months.append((y, m))

    # Resolve symbols
    resolved = []
    for sym in symbols:
        mt5_sym = resolve_symbol(sym)
        if mt5_sym:
            dir_name = sym.replace("/", "_")
            resolved.append({"sym": sym, "mt5": mt5_sym, "dir": dir_name})
            log("ok", f"Resolved: {sym} -> {mt5_sym}")
        else:
            log("warning", f"Not found in MT5: {sym}")

    log("info", f"Resolved {len(resolved)}/{len(symbols)} symbols")
    log("info", f"Months: {len(months)} | Total combos: {len(resolved) * len(months)}")

    # Download
    total_files = 0
    total_ticks = 0
    skipped = 0
    empty = 0

    for si, s in enumerate(resolved):
        sym_files = 0
        sym_ticks = 0

        for y, m in months:
            target_dir = os.path.join(drive, DATA_DIR_NAME, s["dir"])
            os.makedirs(target_dir, exist_ok=True)
            fpath = os.path.join(target_dir, f"{y}-{m:02d}.parquet")

            # Skip existing (refresh current month)
            is_current = (y == now.year and m == now.month)
            if os.path.exists(fpath) and not is_current:
                skipped += 1
                continue

            start, end = month_range(y, m)
            try:
                df = fetch_ticks(s["mt5"], start, end)
                if df is None or len(df) == 0:
                    empty += 1
                    continue
                df.to_parquet(fpath, index=False, compression="snappy")
                sym_files += 1
                sym_ticks += len(df)
                total_files += 1
                total_ticks += len(df)
            except Exception as e:
                log("error", f"{s['sym']} {y}-{m:02d}: {e}")

        pct = (si + 1) / len(resolved) * 100
        log("ok", f"[{pct:5.1f}%] {s['sym']:15s} | {sym_files} new files, {sym_ticks:>10,} ticks")

    log("info", "=" * 50)
    log("info", f"COMPLETE: {total_files} files, {total_ticks:,} ticks, {skipped} skipped, {empty} empty")

    # Show disk usage
    tick_dir = os.path.join(drive, DATA_DIR_NAME)
    for s in resolved:
        d = os.path.join(tick_dir, s["dir"])
        if os.path.exists(d):
            sz = sum(f.stat().st_size for f in __import__("pathlib").Path(d).glob("*.parquet"))
            log("info", f"  {s['sym']:15s}: {sz / 1024 / 1024:.1f} MB")

    mt5.shutdown()
    log("info", "Done.")


if __name__ == "__main__":
    main()
