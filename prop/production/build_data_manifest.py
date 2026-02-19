#!/usr/bin/env python3
"""Step 1/2: integrity manifest + universe filter for tick parquet data."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


MONTH_RE = re.compile(r"^(20\d{2})-(0[1-9]|1[0-2])\.parquet$")


@dataclass
class FileCheck:
    ok: bool
    rows: int
    error: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--roots",
        default="/home/tradebot/ssd_data_1/tick_data,/home/tradebot/ssd_data_2/tick_data,/home/tradebot/data_1/tick_data,/home/tradebot/data_2/tick_data,/home/tradebot/data_3/tick_data",
    )
    p.add_argument("--min-months", type=int, default=12)
    p.add_argument("--max-gap-ratio", type=float, default=0.35)
    p.add_argument("--out-dir", default="/home/tradebot/tradebots/prop/production/data_manifest")
    return p.parse_args()


def month_range_count(start: str, end: str) -> int:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    return (ey - sy) * 12 + (em - sm) + 1


def iter_expected_months(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def check_parquet(path: Path) -> FileCheck:
    try:
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows if pf.metadata is not None else 0
        return FileCheck(ok=True, rows=int(rows), error="")
    except Exception as e:  # noqa: BLE001
        return FileCheck(ok=False, rows=0, error=str(e))


def build_manifest(roots: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    file_rows = []

    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for sym_dir in sorted([p for p in rp.iterdir() if p.is_dir()]):
            month_files = []
            corrupt = 0
            empty = 0
            total_rows = 0
            total_bytes = 0

            for fp in sorted(sym_dir.glob("*.parquet")):
                m = MONTH_RE.match(fp.name)
                if not m:
                    continue
                month = f"{m.group(1)}-{m.group(2)}"
                stat = fp.stat()
                chk = check_parquet(fp)
                if not chk.ok:
                    corrupt += 1
                if chk.ok and chk.rows == 0:
                    empty += 1
                if chk.ok:
                    total_rows += chk.rows
                total_bytes += stat.st_size
                month_files.append(month)
                file_rows.append(
                    {
                        "symbol": sym_dir.name,
                        "month": month,
                        "path": str(fp),
                        "size_bytes": stat.st_size,
                        "rows": chk.rows,
                        "ok": chk.ok,
                        "error": chk.error,
                    }
                )

            if not month_files:
                continue
            months_sorted = sorted(set(month_files))
            min_month = months_sorted[0]
            max_month = months_sorted[-1]
            expected = iter_expected_months(min_month, max_month)
            missing = sorted(set(expected) - set(months_sorted))
            span_months = month_range_count(min_month, max_month)
            gap_ratio = len(missing) / span_months if span_months else 0.0

            rows.append(
                {
                    "symbol": sym_dir.name,
                    "root": str(rp),
                    "file_count": len(months_sorted),
                    "min_month": min_month,
                    "max_month": max_month,
                    "span_months": span_months,
                    "missing_months": len(missing),
                    "gap_ratio": gap_ratio,
                    "corrupt_files": corrupt,
                    "empty_files": empty,
                    "rows_total": total_rows,
                    "size_bytes_total": total_bytes,
                    "missing_month_list": ",".join(missing),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(file_rows)


def main() -> None:
    args = parse_args()
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, files = build_manifest(roots)
    if manifest.empty:
        raise SystemExit("No symbol parquet data found in given roots.")

    manifest = manifest.sort_values(["file_count", "rows_total"], ascending=[False, False]).reset_index(drop=True)

    shortlist = manifest[
        (manifest["file_count"] >= args.min_months)
        & (manifest["gap_ratio"] <= args.max_gap_ratio)
        & (manifest["corrupt_files"] == 0)
    ].copy()

    manifest_pq = out_dir / "symbol_manifest.parquet"
    manifest_csv = out_dir / "symbol_manifest.csv"
    files_pq = out_dir / "file_manifest.parquet"
    shortlist_pq = out_dir / "universe_shortlist.parquet"
    shortlist_csv = out_dir / "universe_shortlist.csv"
    shortlist_txt = out_dir / "universe_shortlist.txt"

    manifest.to_parquet(manifest_pq, index=False, compression="zstd")
    manifest.to_csv(manifest_csv, index=False)
    files.to_parquet(files_pq, index=False, compression="zstd")
    shortlist.to_parquet(shortlist_pq, index=False, compression="zstd")
    shortlist.to_csv(shortlist_csv, index=False)
    shortlist_txt.write_text("\n".join(shortlist["symbol"].tolist()) + "\n", encoding="utf-8")

    print(f"symbols_total={len(manifest)}")
    print(f"symbols_shortlist={len(shortlist)}")
    print(f"manifest={manifest_pq}")
    print(f"shortlist={shortlist_pq}")


if __name__ == "__main__":
    main()
