"""
Rolling Daily Optuna Pipeline — 24-hour continuous hyperparameter optimization.

Runs daily 12:00 CET → 09:00 CET with phased GPU allocation:
  Day   (12:00-22:00): P40 = 1 worker, GTX 1050 = 3 workers  (bot on P40)
  Night (22:00-09:00): P40 = 6 workers, GTX 1050 = 3 workers  (full gas)

Warm-starts from persistent per-timeframe SQLite DBs, adds ~30 trials/symbol/day.
Rotates through timeframes: M15 → M30 → H1 → H4 (one per day).

Usage:
    python3 research/rolling_optuna.py              # normal run
    python3 research/rolling_optuna.py --dry-run    # show plan, don't execute
"""
from __future__ import annotations

import csv
import fcntl
import gc
import os
import signal
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.optuna_orchestrator import (
    DATA_ROOTS,
    discover_symbols_from_data,
    load_active_symbols,
    run_symbol,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CET = timezone(timedelta(hours=1))  # CET = UTC+1 (winter), close enough
LOCKFILE = "/tmp/rolling_optuna.lock"
ROLLING_DIR = REPO_ROOT / "models" / "optuna_results" / "rolling"
AUDIT_DB = REPO_ROOT / "audit" / "sovereign_log.db"

TIMEFRAME_ROTATION = ["M15", "M30", "H1", "H4"]

# GPU IDs — P40 = 0, GTX 1050 = 1
GPU_P40 = 0
GPU_GTX = 1

# Phase configs: (p40_workers, gtx_workers)
PHASE_DAY = (1, 3)      # 12:00-22:00 — bot runs on P40
PHASE_NIGHT = (6, 3)     # 22:00-09:00 — full gas
PHASE_DONE = (0, 0)      # 09:00-12:00 — rest

# Per-symbol trial budget
TRIALS_PER_SYMBOL = 30
TRIAL_JOBS = 1  # sequential trials within symbol (avoids GPU contention)

# Skip hopeless symbols: >500 trials and best EV < 0
HOPELESS_TRIAL_THRESHOLD = 500

# Shared XGB/pipeline settings (match sunday_ritual.sh)
PIPELINE_ARGS = {
    "data_roots": DATA_ROOTS,
    "z_threshold": 1.0,
    "trials": TRIALS_PER_SYMBOL,
    "trial_jobs": TRIAL_JOBS,
    "train_size": 2400,
    "test_size": 600,
    "purge": 24,
    "embargo": 24,
    "pt_mult": 2.0,
    "sl_mult": 1.5,
    "horizon_bars": 24,
    "discovery_cutoff": TRIALS_PER_SYMBOL,  # no separate discovery phase for 30 trials
    "htf": False,
    "bar_roots": "/home/tradebot/ssd_data_2/bars",
}

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"\n[rolling] Received signal {signum} — shutting down gracefully...")


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_phase() -> str:
    """Return current phase based on CET time."""
    hour = datetime.now(CET).hour
    if 12 <= hour < 22:
        return "day"
    elif hour >= 22 or hour < 9:
        return "night"
    else:
        return "done"  # 09:00-12:00 = rest


def get_phase_workers(phase: str) -> tuple[int, int]:
    """Return (p40_workers, gtx_workers) for given phase."""
    if phase == "day":
        return PHASE_DAY
    elif phase == "night":
        return PHASE_NIGHT
    return PHASE_DONE


def past_deadline() -> bool:
    """True if past 09:00 CET (next morning)."""
    now = datetime.now(CET)
    # If it's the next day between 09:00 and 12:00, we're done
    return now.hour >= 9 and now.hour < 12


def todays_timeframe() -> str:
    """Rotate through timeframes based on day-of-year."""
    doy = datetime.now(CET).timetuple().tm_yday
    return TIMEFRAME_ROTATION[doy % len(TIMEFRAME_ROTATION)]


def check_open_positions() -> int:
    """Query latest heartbeat from audit DB for open position count."""
    if not AUDIT_DB.exists():
        return 0
    try:
        conn = sqlite3.connect(str(AUDIT_DB), timeout=5)
        row = conn.execute(
            "SELECT open_positions FROM heartbeats "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def is_sunday_ritual_active() -> bool:
    """Check if sunday-ritual.service is currently running."""
    try:
        ret = os.system("systemctl --user is-active --quiet sunday-ritual.service")
        return ret == 0
    except Exception:
        return False


def is_hopeless(symbol: str, timeframe: str, db_path: str) -> bool:
    """Check if a symbol has >500 trials with best EV < 0 in the study DB."""
    if not os.path.exists(db_path):
        return False
    try:
        import optuna
        storage_url = f"sqlite:///{db_path}"
        study_name = f"{symbol}_{timeframe}"
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
        )
        n_trials = len(study.trials)
        if n_trials < HOPELESS_TRIAL_THRESHOLD:
            return False
        best_val = study.best_trial.value if study.best_trial else -1.0
        return best_val < 0
    except Exception:
        return False


def get_live_tickers() -> list[str]:
    """Load live tickers from sovereign_configs.json (dynamic, not hardcoded)."""
    try:
        return load_active_symbols()
    except SystemExit:
        return []


def build_symbol_queue(timeframe: str) -> list[str]:
    """Build prioritized symbol queue: live tickers first, then rest alphabetically.
    Skip hopeless symbols."""
    all_symbols = discover_symbols_from_data(DATA_ROOTS)
    if not all_symbols:
        return []

    live_tickers = get_live_tickers()
    db_path = str(ROLLING_DIR / timeframe / "optuna_studies.db")

    # Separate live from rest
    live_set = set(live_tickers)
    live = [s for s in live_tickers if s in set(all_symbols)]
    rest = [s for s in all_symbols if s not in live_set]

    # Filter out hopeless symbols
    queue = []
    skipped = 0
    for sym in live + rest:
        if is_hopeless(sym, timeframe, db_path):
            skipped += 1
        else:
            queue.append(sym)

    if skipped:
        print(f"[rolling] Skipped {skipped} hopeless symbols (>{HOPELESS_TRIAL_THRESHOLD} trials, EV<0)")

    return queue


def _run_symbol_with_gpu(symbol: str, args_dict: dict, gpu_id: int) -> dict:
    """Worker wrapper: set CUDA_VISIBLE_DEVICES before running."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        return run_symbol(symbol, args_dict)
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e)}
    finally:
        gc.collect()


def update_rolling_summary(results: list[dict], timeframe: str):
    """Merge new results into rolling_summary.csv — keep best EV per (symbol, TF)."""
    summary_path = ROLLING_DIR / "rolling_summary.csv"

    # Load existing
    existing: dict[tuple[str, str], dict] = {}
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["symbol"], row["timeframe"])
                existing[key] = row

    # Merge new results
    for r in results:
        if r.get("status") != "ok":
            continue
        sym = r["symbol"]
        key = (sym, timeframe)
        ev = float(r["best_ev"])
        old_ev = float(existing[key]["best_ev"]) if key in existing else -999.0
        if ev > old_ev:
            existing[key] = {
                "symbol": sym,
                "timeframe": timeframe,
                "best_ev": f"{ev:.6f}",
                "calmar_score": f"{r.get('calmar_score', ev):.6f}",
                "trials": str(r.get("trials", 0)),
                "cluster": r.get("cluster", ""),
                "fee_bps": f"{r.get('fee_bps', 0):.1f}",
                "spread_bps": f"{r.get('spread_bps', 0):.1f}",
                "slippage_bps": f"{r.get('slippage_bps', 0):.1f}",
                "updated": datetime.now(CET).strftime("%Y-%m-%d %H:%M"),
            }

    # Write back
    if not existing:
        return

    fieldnames = ["symbol", "timeframe", "best_ev", "calmar_score", "trials",
                  "cluster", "fee_bps", "spread_bps", "slippage_bps", "updated"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing.keys()):
            writer.writerow(existing[key])

    print(f"[rolling] Updated {summary_path} ({len(existing)} entries)")


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def run_rolling_pipeline(dry_run: bool = False):
    """Main entry point for the rolling daily Optuna pipeline."""
    start_time = datetime.now(CET)
    timeframe = todays_timeframe()
    out_dir = str(ROLLING_DIR / timeframe)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[rolling] ═══════════════════════════════════════════════════════")
    print(f"[rolling] Rolling Daily Optuna Pipeline")
    print(f"[rolling] Start:     {start_time.strftime('%Y-%m-%d %H:%M CET')}")
    print(f"[rolling] Deadline:  09:00 CET tomorrow")
    print(f"[rolling] Timeframe: {timeframe}")
    print(f"[rolling] Trials:    {TRIALS_PER_SYMBOL} per symbol (warm-start)")
    print(f"[rolling] DB:        {out_dir}/optuna_studies.db")
    print(f"[rolling] ═══════════════════════════════════════════════════════")

    # Build queue
    queue = build_symbol_queue(timeframe)
    if not queue:
        print("[rolling] No symbols to process — exiting")
        return

    live_tickers = get_live_tickers()
    print(f"[rolling] Queue: {len(queue)} symbols "
          f"(live first: {[s for s in queue if s in live_tickers]})")

    if dry_run:
        print("[rolling] DRY RUN — would process:")
        for i, sym in enumerate(queue):
            print(f"  {i+1:3d}. {sym}")
        return

    # Prepare args
    args_dict = dict(PIPELINE_ARGS)
    args_dict["timeframe"] = timeframe
    args_dict["out_dir"] = out_dir

    results: list[dict] = []
    processed = 0
    remaining_queue = list(queue)

    while remaining_queue and not _shutdown:
        # Check deadline
        if past_deadline():
            print(f"[rolling] Deadline reached (09:00 CET) — stopping with "
                  f"{len(remaining_queue)} symbols remaining")
            break

        # Check phase and workers
        phase = get_current_phase()
        if phase == "done":
            print("[rolling] Phase=done (09:00-12:00) — stopping")
            break

        p40_workers, gtx_workers = get_phase_workers(phase)

        # Safety: if many open positions, reduce P40 load
        open_pos = check_open_positions()
        if open_pos > 10:
            print(f"[rolling] WARNING: {open_pos} open positions — P40 workers → 0")
            p40_workers = 0

        total_workers = p40_workers + gtx_workers
        if total_workers == 0:
            print("[rolling] No workers available — waiting 60s...")
            time.sleep(60)
            continue

        # Build GPU assignment for this batch
        batch_size = min(total_workers, len(remaining_queue))
        batch = remaining_queue[:batch_size]
        remaining_queue = remaining_queue[batch_size:]

        # Assign GPUs: first gtx_workers get GTX, rest get P40
        gpu_assignments = []
        for i in range(batch_size):
            if i < gtx_workers:
                gpu_assignments.append(GPU_GTX)
            else:
                gpu_assignments.append(GPU_P40)

        phase_label = f"[{phase}] P40={p40_workers} GTX={gtx_workers}"
        print(f"\n[rolling] {phase_label} — Batch {processed+1}-{processed+batch_size} "
              f"of {len(queue)} | {len(remaining_queue)} remaining")

        # Submit batch
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            futures = {}
            for sym, gpu_id in zip(batch, gpu_assignments):
                f = executor.submit(_run_symbol_with_gpu, sym, args_dict, gpu_id)
                futures[f] = (sym, gpu_id)

            for future in as_completed(futures):
                if _shutdown:
                    print("[rolling] Shutdown requested — cancelling pending futures")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                sym, gpu_id = futures[future]
                try:
                    r = future.result(timeout=600)  # 10 min max per symbol
                    status = r.get("status", "unknown")
                    if status == "ok":
                        ev = r.get("best_ev", 0)
                        trials = r.get("trials", 0)
                        print(f"  ✓ {sym:20s} EV={ev:.6f} trials={trials} gpu={gpu_id}")
                    elif status == "killed_negative_ev":
                        print(f"  ✗ {sym:20s} KILLED EV={r.get('best_ev',0):.6f} "
                              f"trials={r.get('trials',0)}")
                    else:
                        print(f"  - {sym:20s} {status}")
                    results.append(r)
                except Exception as e:
                    print(f"  ! {sym:20s} ERROR: {e}")
                    results.append({"symbol": sym, "status": "error", "error": str(e)})

                processed += 1

    # Write results
    elapsed = datetime.now(CET) - start_time
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    killed_count = sum(1 for r in results if r.get("status") == "killed_negative_ev")
    error_count = sum(1 for r in results if r.get("status") == "error")

    print(f"\n[rolling] ═══════════════════════════════════════════════════════")
    print(f"[rolling] DONE — {processed}/{len(queue)} symbols in {elapsed}")
    print(f"[rolling] OK={ok_count} Killed={killed_count} Error={error_count}")
    print(f"[rolling] ═══════════════════════════════════════════════════════")

    # Update rolling summary
    if results:
        update_rolling_summary(results, timeframe)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Rolling Daily Optuna Pipeline")
    p.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = p.parse_args()

    # Check if sunday-ritual is active
    if is_sunday_ritual_active():
        print("[rolling] Sunday ritual is active — skipping")
        sys.exit(0)

    # Lockfile
    try:
        lock_fd = open(LOCKFILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
    except (IOError, OSError):
        print(f"[rolling] Another instance is running (lockfile: {LOCKFILE}) — exiting")
        sys.exit(0)

    try:
        run_rolling_pipeline(dry_run=args.dry_run)
    finally:
        # Release lockfile
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            os.unlink(LOCKFILE)
        except Exception:
            pass


if __name__ == "__main__":
    main()
