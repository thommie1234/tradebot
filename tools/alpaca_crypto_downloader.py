#!/usr/bin/env python3
"""Download crypto tick data from Alpaca for symbols missing from FTMO.

Uses Alpaca's CryptoHistoricalDataClient (free, no SIP restriction).
Saves daily parquet files compatible with train_ml_strategy.py.

Usage:
    python3 common/tools/alpaca_crypto_downloader.py
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p
os.chdir(REPO_ROOT)
load_dotenv()

# BF crypto symbols that have no tick data yet
# Alpaca format: "APT/USD", save as: "APTUSD" directory
SYMBOLS = [
    "APT/USD", "ARB/USD", "CRV/USD", "DYDX/USD", "FIL/USD",
    "INJ/USD", "LDO/USD", "NEAR/USD", "OP/USD", "RNDR/USD",
    "RUNE/USD", "SEI/USD", "STX/USD", "SUI/USD", "TRX/USD",
    "WOO/USD", "MATIC/USD", "XDG/USD",
]

DATA_DIRS = [
    r"C:\tick_data\ssd1",
    r"C:\tick_data\ssd2",
    r"C:\tick_data\nvme",
]

YEARS = 3


def dir_name(alpaca_sym: str) -> str:
    """APT/USD -> APTUSD"""
    return alpaca_sym.replace("/", "")


def week_ranges(start: datetime, end: datetime):
    ranges = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        week_end = min(current + timedelta(days=7), end)
        ranges.append((current, week_end))
        current = week_end
    return ranges


def week_is_cached(out_dir: str, week_start: datetime, week_end: datetime) -> bool:
    """Check if a week's data is already cached (>= 3 daily parquet files)."""
    year_dir = os.path.join(out_dir, str(week_start.year))
    if not os.path.isdir(year_dir):
        return False
    count = 0
    d = week_start
    while d < week_end:
        fp = os.path.join(year_dir, f"{d.strftime('%Y-%m-%d')}.parquet")
        if os.path.exists(fp):
            count += 1
        d += timedelta(days=1)
    return count >= 3  # Crypto trades 7 days, 3 is reasonable minimum


def main():
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoTradesRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    client = CryptoHistoricalDataClient(api_key, api_secret)

    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * YEARS)
    weeks = week_ranges(start, end)

    print(f"Symbols: {len(SYMBOLS)}, Weeks: {len(weeks)}, Range: {start.date()} — {end.date()}")

    for si, alpaca_sym in enumerate(SYMBOLS):
        sym_dir_name = dir_name(alpaca_sym)
        out_base = DATA_DIRS[si % len(DATA_DIRS)]
        out_dir = os.path.join(out_base, sym_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        new_weeks = 0
        total_ticks = 0
        t0 = time.time()

        for week_start, week_end in weeks:
            if week_is_cached(out_dir, week_start, week_end):
                continue

            try:
                request = CryptoTradesRequest(
                    symbol_or_symbols=alpaca_sym,
                    start=week_start,
                    end=week_end,
                )
                trades = client.get_crypto_trades(request)

                # Extract trades for this symbol
                trade_list = trades.data.get(alpaca_sym, [])

                if not trade_list:
                    new_weeks += 1
                    continue

                # Convert to records
                records = []
                for t in trade_list:
                    records.append({
                        "timestamp": int(t.timestamp.timestamp() * 1_000_000),  # microseconds
                        "price": float(t.price),
                        "size": float(t.size),
                    })

                df = pl.DataFrame(records)
                total_ticks += len(df)

                # Split by day and save
                df = df.with_columns(
                    pl.from_epoch(pl.col("timestamp"), time_unit="us")
                    .alias("_dt")
                )
                df = df.with_columns(
                    pl.col("_dt").dt.strftime("%Y-%m-%d").alias("_date"),
                    pl.col("_dt").dt.year().alias("_year"),
                )

                for (date_str, year), group in df.group_by(["_date", "_year"]):
                    year_dir = os.path.join(out_dir, str(year))
                    os.makedirs(year_dir, exist_ok=True)
                    fp = os.path.join(year_dir, f"{date_str}.parquet")
                    group.select(["timestamp", "price", "size"]).write_parquet(fp)

                new_weeks += 1

            except Exception as e:
                err = str(e)[:100]
                print(f"  ERROR {alpaca_sym} {week_start.date()}: {err}")
                time.sleep(1)

        elapsed = time.time() - t0
        days_cached = sum(
            len([f for f in os.listdir(os.path.join(out_dir, d)) if f.endswith(".parquet")])
            for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))
        ) if os.path.isdir(out_dir) else 0

        done = si + 1
        rate = done / max(time.time() - t0, 1) if si == 0 else None
        print(f"[{done}/{len(SYMBOLS)}] {alpaca_sym}: +{new_weeks} weeks, "
              f"{total_ticks:,} ticks, {days_cached} days on disk ({elapsed:.0f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
