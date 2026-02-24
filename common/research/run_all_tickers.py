#!/usr/bin/env python3
"""Run Optuna + WFO on ALL tickers across ALL timeframes.

Designed to finish within 12 hours using full GPU capacity.
Phase 1: Optuna on all GPUs in parallel (P40 + GTX)
Phase 2: WFO validation on GTX (CPU-heavy, doesn't need P40)

Usage:
    python3 research/run_all_tickers.py
    python3 research/run_all_tickers.py --timeframes M30,H1
    python3 research/run_all_tickers.py --workers 9
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.rolling_optuna import (
    CET,
    PIPELINE_ARGS,
    ROLLING_DIR,
    build_symbol_queue,
    run_wfo_validation,
    update_rolling_summary,
)

GPU_P40 = 0
GPU_GTX = 1


def _run_optuna(symbol: str, args_dict: dict, gpu_id: int) -> dict:
    """Run Optuna optimization for one symbol on specified GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        from research.optuna_orchestrator import run_symbol
        return run_symbol(symbol, args_dict)
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e)}
    finally:
        gc.collect()


def _run_wfo(symbol: str, timeframe: str, ev: float, optuna_result: dict) -> dict:
    """Run WFO validation (CPU-heavy, GPU optional)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GTX for WFO
    try:
        result = run_wfo_validation(symbol, timeframe, ev, optuna_result=optuna_result)
        return {"symbol": symbol, "wfo": result}
    except Exception as e:
        print(f"  WFO {symbol}: ERROR {e}", flush=True)
        return {"symbol": symbol, "wfo": None, "error": str(e)}


def run_timeframe(timeframe: str, queue: list[str], workers: int) -> list[dict]:
    args_dict = dict(PIPELINE_ARGS)
    args_dict["timeframe"] = timeframe
    args_dict["out_dir"] = str(ROLLING_DIR / timeframe)
    os.makedirs(args_dict["out_dir"], exist_ok=True)

    print(f"\n{'='*70}")
    print(f" {timeframe} — {len(queue)} symbols, {workers} workers")
    print(f"{'='*70}", flush=True)

    # ── Phase 1: Optuna on all GPUs ──────────────────────────────────
    print(f"\n[{timeframe}] Phase 1: Optuna optimization", flush=True)
    results = []
    wfo_queue = []  # (symbol, ev, optuna_result) for WFO
    processed = 0
    remaining = list(queue)

    while remaining:
        batch = remaining[:workers]
        remaining = remaining[workers:]

        # Assign GPUs: first 3 get GTX, rest get P40
        gpus = []
        for i in range(len(batch)):
            gpus.append(GPU_GTX if i < 3 else GPU_P40)

        print(f"\n[{timeframe}] Batch {processed+1}-{processed+len(batch)} "
              f"of {len(queue)} | {len(remaining)} left", flush=True)

        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for sym, gpu in zip(batch, gpus):
                f = executor.submit(_run_optuna, sym, args_dict, gpu)
                futures[f] = sym

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    r = future.result(timeout=600)
                    status = r.get("status", "?")
                    ev = r.get("best_ev", 0)
                    rows = r.get("rows", "?")
                    if status == "ok":
                        print(f"  + {sym:20s} EV={ev:+.6f} rows={rows}", flush=True)
                    elif status == "killed_negative_ev":
                        print(f"  x {sym:20s} EV={ev:+.6f} rows={rows}", flush=True)
                    else:
                        print(f"  - {sym:20s} {status} rows={rows}", flush=True)
                    # Queue for WFO
                    if status in ("ok", "killed_negative_ev"):
                        wfo_queue.append((sym, ev, r))
                    results.append(r)
                except Exception as e:
                    print(f"  ! {sym:20s} ERROR: {e}", flush=True)
                    results.append({"symbol": sym, "status": "error", "error": str(e)})
                processed += 1

    # ── Phase 2: WFO validation in parallel ──────────────────────────
    if wfo_queue:
        print(f"\n[{timeframe}] Phase 2: WFO validation — {len(wfo_queue)} symbols", flush=True)
        wfo_workers = min(4, len(wfo_queue))  # WFO is CPU-bound, 4 is enough
        with ProcessPoolExecutor(max_workers=wfo_workers) as executor:
            futures = {}
            for sym, ev, optuna_r in wfo_queue:
                f = executor.submit(_run_wfo, sym, timeframe, ev, optuna_r)
                futures[f] = sym

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    future.result(timeout=300)
                except Exception as e:
                    print(f"  WFO {sym}: TIMEOUT/ERROR {e}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all tickers, all timeframes")
    parser.add_argument("--timeframes", type=str, default="M15,M30,H1,H4")
    parser.add_argument("--workers", type=int, default=9,
                        help="Total parallel workers (default: 9 = 6 P40 + 3 GTX)")
    parser.add_argument("--trials", type=int, default=0,
                        help="Override trials per symbol (0 = use default from PIPELINE_ARGS)")
    args = parser.parse_args()

    timeframes = [t.strip() for t in args.timeframes.split(",")]
    start = datetime.now(CET)
    print(f"[run-all] Start: {start.strftime('%Y-%m-%d %H:%M CET')}")
    print(f"[run-all] Timeframes: {timeframes}")
    print(f"[run-all] Workers: {args.workers}")

    # Override trials if requested
    if args.trials > 0:
        PIPELINE_ARGS["trials"] = args.trials
        print(f"[run-all] Trials override: {args.trials}")

    all_results = {}
    for tf in timeframes:
        queue = build_symbol_queue(tf)
        print(f"[run-all] {tf}: {len(queue)} symbols queued")
        tf_start = datetime.now(CET)
        results = run_timeframe(tf, queue, args.workers)
        tf_elapsed = datetime.now(CET) - tf_start
        all_results[tf] = results
        update_rolling_summary(results, tf)

        ok = sum(1 for r in results if r.get("status") == "ok")
        killed = sum(1 for r in results if r.get("status") == "killed_negative_ev")
        print(f"\n[run-all] {tf} done in {tf_elapsed} — OK={ok} Killed={killed}", flush=True)

    # Final summary
    elapsed = datetime.now(CET) - start
    print(f"\n{'='*70}")
    print(f" FINAL SUMMARY — {elapsed}")
    print(f"{'='*70}")
    for tf in timeframes:
        results = all_results.get(tf, [])
        ok = sum(1 for r in results if r.get("status") == "ok")
        killed = sum(1 for r in results if r.get("status") == "killed_negative_ev")
        nodata = sum(1 for r in results if r.get("status") in ("no_data", "insufficient_rows"))
        errors = sum(1 for r in results if r.get("status") == "error")
        print(f"  {tf:6s} | OK={ok:3d} Killed={killed:3d} NoData={nodata:3d} Error={errors:3d} "
              f"| Total={len(results)}")
    print(f"\n[run-all] Done in {elapsed}", flush=True)


if __name__ == "__main__":
    main()
