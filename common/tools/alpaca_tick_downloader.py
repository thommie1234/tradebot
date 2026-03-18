#!/usr/bin/env python3
"""Bulk tick data downloader from Alpaca for US stocks.

Downloads trade ticks per symbol in weekly chunks, stores as daily parquet files.
Resumes where it left off — safe to restart.

v3: Concurrent downloads (10 threads), direct REST API (no SDK RAM blow-up),
    rate-limit aware with 429 backoff.

Usage:
    python3 common/tools/alpaca_tick_downloader.py --symbols ttp/config/all_stocks.txt --years 3
    python3 common/tools/alpaca_tick_downloader.py --symbols AAPL,MSFT,NVDA --days 30
"""
from __future__ import annotations

import argparse
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests as req
from dotenv import load_dotenv


# ── Globals ──
_rate_lock = threading.Lock()
_request_times: list[float] = []
RATE_LIMIT = 180  # stay under 200/min with margin
API_BASE = "https://data.alpaca.markets/v2/stocks"
_headers: dict[str, str] = {}
_stats_lock = threading.Lock()
_stats = {"symbols_done": 0, "total_ticks": 0, "errors": 0, "start_time": 0.0}


def init_client():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    _headers["APCA-API-KEY-ID"] = os.environ["ALPACA_API_KEY"]
    _headers["APCA-API-SECRET-KEY"] = os.environ["ALPACA_SECRET_KEY"]


def rate_wait():
    """Block until we're under the rate limit."""
    with _rate_lock:
        now = time.time()
        # Prune old entries (older than 60s)
        cutoff = now - 60
        while _request_times and _request_times[0] < cutoff:
            _request_times.pop(0)
        if len(_request_times) >= RATE_LIMIT:
            sleep_until = _request_times[0] + 60.01
            wait = sleep_until - now
            if wait > 0:
                time.sleep(wait)
        _request_times.append(time.time())


def fetch_trades_page(symbol: str, start: str, end: str,
                      page_token: str | None = None) -> tuple[list[dict], str | None]:
    """Fetch one page of trades from Alpaca REST API."""
    rate_wait()
    params = {
        "start": start,
        "end": end,
        "limit": "10000",
    }
    if page_token:
        params["page_token"] = page_token

    for attempt in range(5):
        try:
            r = req.get(f"{API_BASE}/{symbol}/trades", headers=_headers,
                        params=params, timeout=30)
        except req.exceptions.RequestException as e:
            print(f"  NETWORK {symbol}: {e}", flush=True)
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", 5))
            print(f"  RATE LIMITED, sleeping {retry_after}s", flush=True)
            time.sleep(retry_after)
            continue
        if r.status_code == 422:
            # SIP data restriction for current month
            return [], None
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} {symbol}: {r.text[:200]}", flush=True)
            time.sleep(2 ** attempt)
            continue

        data = r.json()
        trades = data.get("trades", [])
        next_token = data.get("next_page_token")
        return trades, next_token

    return [], None


def week_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Generate (week_start, week_end) pairs."""
    ranges = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        week_end = min(current + timedelta(days=7), end)
        ranges.append((current, week_end))
        current = week_end
    return ranges


def week_is_cached(symbol: str, week_start: datetime, out_dir: Path) -> bool:
    """Check if we already have data for this week (at least 3 daily files)."""
    year_dir = out_dir / symbol / str(week_start.year)
    if not year_dir.exists():
        return False
    week_end = week_start + timedelta(days=7)
    count = 0
    for f in year_dir.iterdir():
        if f.suffix != ".parquet":
            continue
        try:
            fdate = datetime.strptime(f.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if week_start <= fdate < week_end:
                count += 1
        except ValueError:
            continue
    return count >= 3


def download_week(symbol: str, week_start: datetime, week_end: datetime,
                  out_dir: Path) -> int:
    """Download all trades for one symbol for one week via paginated REST API."""
    start_iso = week_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = week_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    all_records = []
    page_token = None

    while True:
        trades, next_token = fetch_trades_page(symbol, start_iso, end_iso, page_token)
        if not trades:
            break

        for t in trades:
            ts_str = t.get("t", "")
            try:
                # Parse ISO timestamp to microseconds
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_us = int(ts.timestamp() * 1_000_000)
            except (ValueError, AttributeError):
                continue
            all_records.append({
                "timestamp": ts_us,
                "price": float(t.get("p", 0)),
                "size": float(t.get("s", 0)),
                "exchange": str(t.get("x", "")),
                "conditions": ",".join(t.get("c", []) or []),
            })

        if not next_token:
            break
        page_token = next_token

    if not all_records:
        return 0

    df = pl.DataFrame(all_records)
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="us")
          .dt.date()
          .alias("date")
    )

    total = 0
    for date_val, day_df in df.group_by("date"):
        date_obj = date_val[0]
        year_dir = out_dir / symbol / str(date_obj.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = year_dir / f"{date_obj}.parquet"
        if parquet_path.exists():
            continue
        day_df.drop("date").write_parquet(parquet_path, compression="zstd")
        total += len(day_df)

    return total


def process_symbol(symbol: str, weeks: list[tuple[datetime, datetime]],
                   out_dir: Path, si: int, total_symbols: int) -> int:
    """Download all weeks for one symbol. Returns total new ticks."""
    sym_ticks = 0
    sym_new = 0

    for week_start, week_end in weeks:
        if week_is_cached(symbol, week_start, out_dir):
            continue
        count = download_week(symbol, week_start, week_end, out_dir)
        if count > 0:
            sym_ticks += count
            sym_new += 1

    with _stats_lock:
        _stats["symbols_done"] += 1
        _stats["total_ticks"] += sym_ticks
        done = _stats["symbols_done"]
        elapsed = time.time() - _stats["start_time"]
        rate = done / (elapsed / 3600) if elapsed > 0 else 0
        eta = (total_symbols - done) / rate if rate > 0 else 0

    status = f"[{done}/{total_symbols}] {symbol}: "
    if sym_new > 0:
        status += f"+{sym_new} weeks, {sym_ticks:,} ticks"
    else:
        status += "all cached"
    status += f" (ETA: {eta:.1f}h)"
    print(status, flush=True)

    return sym_ticks


def load_symbols(arg: str) -> list[str]:
    if os.path.isfile(arg):
        with open(arg) as f:
            return [line.strip() for line in f if line.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Download tick data from Alpaca (v3 — concurrent)")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--days", type=int)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out-dir", default=r"C:\tick_data\ssd1\alpaca",
                        help="Output dir(s), comma-separated for round-robin")
    parser.add_argument("--resume-from", help="Resume from this symbol")
    parser.add_argument("--years", type=int)
    parser.add_argument("--workers", type=int, default=10, help="Concurrent download threads")
    args = parser.parse_args()

    symbols = load_symbols(args.symbols)
    out_dirs = [Path(d.strip()) for d in args.out_dir.split(",")]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.years:
        start = now - timedelta(days=args.years * 365)
    elif args.days:
        start = now - timedelta(days=args.days)
    else:
        start = now - timedelta(days=365)

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        # Free tier: no SIP data for current month
        end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    weeks = week_ranges(start, end)

    # Skip already-done symbols if --resume-from
    if args.resume_from:
        try:
            idx = symbols.index(args.resume_from)
            symbols = symbols[idx:]
            print(f"Resuming from {args.resume_from} ({len(symbols)} remaining)", flush=True)
        except ValueError:
            print(f"WARNING: {args.resume_from} not found, starting from beginning", flush=True)

    print(f"Symbols: {len(symbols)}, Weeks: {len(weeks)}, "
          f"Range: {weeks[0][0].date()} — {weeks[-1][1].date()}", flush=True)
    print(f"Output: {[str(d) for d in out_dirs]}", flush=True)
    print(f"Workers: {args.workers}, Rate limit: {RATE_LIMIT}/min", flush=True)
    print(f"Requests: ~{len(symbols) * len(weeks):,}", flush=True)

    init_client()
    _stats["start_time"] = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for si, symbol in enumerate(symbols):
            out_dir = out_dirs[si % len(out_dirs)]
            f = pool.submit(process_symbol, symbol, weeks, out_dir, si, len(symbols))
            futures[f] = symbol

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                sym = futures[f]
                print(f"  FAILED {sym}: {e}", flush=True)
                with _stats_lock:
                    _stats["errors"] += 1

    elapsed = time.time() - _stats["start_time"]
    print(f"\nDone! Ticks: {_stats['total_ticks']:,}, Errors: {_stats['errors']}, "
          f"Time: {elapsed / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
