#!/usr/bin/env python3
"""Split symbols.xlsx into multiple non-overlapping XLSX files."""

import argparse
import os
from pathlib import Path

import pandas as pd


def read_symbols(path):
    all_sheets = pd.read_excel(path, sheet_name=None, header=None)
    syms = set()
    for _, df in all_sheets.items():
        if df.empty:
            continue
        for v in df.iloc[:, 0].dropna().tolist():
            s = str(v).strip()
            if s:
                syms.add(s)
    return sorted(syms)


def chunked(items, n):
    out = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/home/tradebot/tradebots/data/symbols.xlsx")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", default="/home/tradebot/tradebots/data/symbol_splits")
    args = p.parse_args()

    symbols = read_symbols(args.input)
    os.makedirs(args.out_dir, exist_ok=True)
    parts = chunked(symbols, args.workers)
    for i, part in enumerate(parts, start=1):
        out = Path(args.out_dir) / f"symbols_part_{i:02d}.xlsx"
        pd.DataFrame(part).to_excel(out, index=False, header=False)
        print(f"{out} -> {len(part)} symbols")


if __name__ == "__main__":
    main()
