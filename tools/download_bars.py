#!/usr/bin/env python3
"""Download OHLCV bars from MT5 for all available symbols and timeframes.

Usage:
    python3 tools/download_bars.py                       # all symbols, H1+M30+M15+H4+D1
    python3 tools/download_bars.py --timeframes H1 M30   # specific timeframes
    python3 tools/download_bars.py --symbols NVDA AMZN   # specific symbols
    python3 tools/download_bars.py --years 5             # last 5 years only
    python3 tools/download_bars.py --dry-run             # show plan, no download

Data stored on SSD:
    /home/tradebot/ssd_data_2/bars/{TF}/{SYMBOL}/{YYYY-MM}.parquet

Resumable: skips existing files. Re-run safely.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.mt5_bridge import MT5BridgeClient, initialize_mt5


# ── MT5 timeframe constants ──────────────────────────────────────────
TF_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H2":  16386,
    "H4":  16388,
    "D1":  16408,
    "W1":  32769,
}

DEFAULT_TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]
DEFAULT_YEARS = 20
BAR_ROOT = "/home/tradebot/ssd_data_2/bars"
SPECS_DIR = "/home/tradebot/tradebots/data/instrument_specs"


# ── Symbol loading ───────────────────────────────────────────────────
def load_all_symbols() -> list[dict]:
    """Load symbols from all instrument spec CSVs."""
    symbols = []
    files = ["Forex.csv", "crypto.csv", "Equities.csv", "cash.csv", "metals.csv"]

    for fname in files:
        fpath = os.path.join(SPECS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] {fname} not found, skipping")
            continue
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("symbol", "").strip()
                asset_class = row.get("asset_class", "").strip()
                if sym:
                    symbols.append({
                        "symbol": sym,
                        "asset_class": asset_class,
                        "source": fname,
                    })
    return symbols


# ── Download logic ───────────────────────────────────────────────────
def download_symbol_tf(
    mt5: MT5BridgeClient,
    symbol: str,
    tf_name: str,
    tf_const: int,
    years: int,
    out_root: str,
) -> dict:
    """Download bars for one symbol/timeframe, month by month."""
    out_dir = os.path.join(out_root, tf_name, symbol.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=years * 365)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "rows_total": 0, "empty": 0}

    # Generate month list
    months = []
    current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while current < now:
        months.append(current)
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)

    for month_start in months:
        fname = f"{month_start.strftime('%Y-%m')}.parquet"
        fpath = os.path.join(out_dir, fname)

        # Skip existing (re-download current month for freshness)
        is_current_month = (month_start.year == now.year and month_start.month == now.month)
        if os.path.exists(fpath) and not is_current_month:
            stats["skipped"] += 1
            continue

        # Month end
        if month_start.month == 12:
            month_end = datetime(month_start.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            month_end = datetime(month_start.year, month_start.month + 1, 1, tzinfo=timezone.utc)
        if month_end > now:
            month_end = now

        # Download via bridge with retry + backoff
        bars = None
        for attempt in range(3):
            try:
                bars = mt5.copy_rates_range(symbol, tf_const, month_start, month_end)
                if bars is not None:
                    break
            except Exception:
                pass
            time.sleep(2 + attempt * 3)  # 2s, 5s, 8s backoff

        if bars is None:
            stats["failed"] += 1
            continue

        # Throttle: give MT5 breathing room between requests
        time.sleep(0.5)

        if not bars or len(bars) == 0:
            stats["empty"] += 1
            continue

        # Convert to polars and save
        try:
            df = pl.DataFrame(bars)
            if "time" not in df.columns:
                stats["failed"] += 1
                continue
            df.write_parquet(fpath, compression="snappy")
            stats["downloaded"] += 1
            stats["rows_total"] += len(df)
        except Exception:
            stats["failed"] += 1

    return stats


def try_symbol_variants(mt5: MT5BridgeClient, symbol: str) -> str | None:
    """Try different symbol name variants until one works in MT5."""
    variants = [
        symbol,
        symbol.replace("/", ""),
        symbol.replace("/", "_"),
        symbol.replace("_", "/"),
    ]
    for variant in variants:
        if mt5.symbol_select(variant, True):
            info = mt5.symbol_info(variant)
            if info is not None:
                return variant
    return None


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download OHLCV bars from MT5")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"Timeframes (default: {DEFAULT_TIMEFRAMES})")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Specific symbols (default: all)")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"Years of history (default: {DEFAULT_YEARS})")
    parser.add_argument("--out-root", default=BAR_ROOT,
                        help=f"Output root (default: {BAR_ROOT})")
    parser.add_argument("--port", type=int, default=None,
                        help="MT5 bridge port (default: env MT5_BRIDGE_PORT)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    port = args.port or int(os.getenv("MT5_BRIDGE_PORT", "5055"))
    mt5 = MT5BridgeClient(port=port, timeout=30)

    if not args.dry_run:
        print(f"[download_bars] Connecting to MT5 bridge on port {port}...")
        ok, err, mode = initialize_mt5(mt5)
        if not ok:
            print(f"[FATAL] MT5 init failed: {err}")
            sys.exit(1)
        print(f"[download_bars] MT5 connected ({mode})")
        acct = mt5.account_info()
        if acct:
            print(f"[download_bars] Account: {getattr(acct, 'login', '?')}")

    # Resolve symbols
    if args.symbols:
        all_symbols = [{"symbol": s, "asset_class": "?", "source": "cli"} for s in args.symbols]
    else:
        all_symbols = load_all_symbols()

    print(f"\n[download_bars] {len(all_symbols)} symbols | TFs: {args.timeframes} | {args.years} years")
    print(f"[download_bars] Output: {args.out_root}/{{TF}}/{{SYMBOL}}/\n")

    if args.dry_run:
        for s in all_symbols:
            print(f"  {s['symbol']:20s}  ({s['asset_class']})")
        print(f"\n  ~{args.years * 12 * len(args.timeframes) * len(all_symbols)} parquet files")
        return

    # Resolve MT5 symbol names
    print("[download_bars] Resolving symbols in MT5...")
    resolved = []
    failed_symbols = []
    for sym_info in all_symbols:
        mt5_name = try_symbol_variants(mt5, sym_info["symbol"])
        if mt5_name:
            resolved.append({**sym_info, "mt5_name": mt5_name})
        else:
            failed_symbols.append(sym_info["symbol"])

    print(f"[download_bars] Resolved: {len(resolved)} | Not found: {len(failed_symbols)}")
    if failed_symbols:
        print(f"  Not found: {', '.join(failed_symbols[:30])}")

    # Download
    total = {"downloaded": 0, "skipped": 0, "failed": 0, "rows_total": 0, "empty": 0}
    combos = len(resolved) * len(args.timeframes)
    done = 0

    for tf_name in args.timeframes:
        tf_const = TF_MAP.get(tf_name)
        if tf_const is None:
            print(f"[WARN] Unknown timeframe: {tf_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  TIMEFRAME: {tf_name}")
        print(f"{'='*60}")

        for sym_info in resolved:
            done += 1
            mt5_name = sym_info["mt5_name"]
            pct = done / combos * 100

            stats = download_symbol_tf(mt5, mt5_name, tf_name, tf_const, args.years, args.out_root)

            status = "OK" if stats["failed"] == 0 else "PARTIAL"
            if stats["downloaded"] == 0 and stats["rows_total"] == 0:
                status = "NO_DATA" if stats["empty"] > 0 else "SKIP"

            print(
                f"  [{pct:5.1f}%] {mt5_name:20s} {tf_name:4s} | "
                f"new={stats['downloaded']:3d} skip={stats['skipped']:3d} "
                f"empty={stats['empty']:3d} fail={stats['failed']:3d} "
                f"rows={stats['rows_total']:>8,d}  [{status}]"
            )

            for k in total:
                total[k] += stats[k]

            time.sleep(2)  # 2s between symbols to not overload MT5 terminal

    # Summary
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Resolved:    {len(resolved)} symbols")
    print(f"  Not found:   {len(failed_symbols)}")
    print(f"  Downloaded:  {total['downloaded']:,} files")
    print(f"  Skipped:     {total['skipped']:,} files")
    print(f"  Empty:       {total['empty']:,}")
    print(f"  Failed:      {total['failed']:,}")
    print(f"  Total rows:  {total['rows_total']:,}")

    # Disk usage
    for tf_name in args.timeframes:
        bars_dir = os.path.join(args.out_root, tf_name)
        if os.path.exists(bars_dir):
            sz = sum(f.stat().st_size for f in Path(bars_dir).rglob("*.parquet"))
            print(f"  {tf_name}: {sz / 1024 / 1024:.1f} MB")

    mt5.shutdown()
    print("\n[download_bars] Done.")


if __name__ == "__main__":
    main()
