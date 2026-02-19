#!/usr/bin/env python3
"""
Brute-force system burn test — CPU 100% + GPU 100% for 1 hour.

No complex logic, just raw computational load on every core and GPU.

Usage:
    python3 research/burn_test.py [duration_minutes]
    python3 research/burn_test.py 60
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DURATION_MINUTES = 60
N_CPU_WORKERS = 27   # Leave 1 thread for monitor


# ---------------------------------------------------------------------------
# CPU burn: heavy XGBoost training in loop
# ---------------------------------------------------------------------------

def cpu_burn_worker(worker_id: int, end_time: float):
    """Burn a single CPU core with continuous XGBoost + numpy work."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No GPU for CPU workers

    import xgboost as xgb

    rng = np.random.default_rng(worker_id)
    cycle = 0

    while time.time() < end_time:
        cycle += 1
        n_rows = rng.integers(5000, 20000)
        n_cols = rng.integers(20, 50)

        X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        y = (rng.random(n_rows) > 0.5).astype(np.float32)

        split = int(n_rows * 0.8)
        dtrain = xgb.DMatrix(X[:split], label=y[:split])
        dtest = xgb.DMatrix(X[split:], label=y[split:])

        depth = rng.integers(8, 14)
        rounds = rng.integers(500, 2000)

        xgb.train(
            {"objective": "binary:logistic", "tree_method": "hist",
             "device": "cpu", "max_depth": int(depth), "eta": 0.01,
             "subsample": 0.9, "colsample_bytree": 0.9,
             "nthread": 1, "verbosity": 0},
            dtrain, num_boost_round=int(rounds),
            evals=[(dtest, "test")], verbose_eval=False,
        )

        # Extra: heavy numpy work (SVD decomposition)
        A = rng.standard_normal((2000, 2000))
        _ = np.linalg.svd(A, full_matrices=False)

        if cycle % 5 == 0:
            sys.stdout.write(f"  CPU-{worker_id:02d}: cycle {cycle}\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# GPU burn: continuous XGBoost on P40
# ---------------------------------------------------------------------------

def gpu_burn_p40(end_time: float):
    """Burn P40 (GPU 0) with continuous heavy XGBoost training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import xgboost as xgb

    rng = np.random.default_rng(999)
    cycle = 0

    while time.time() < end_time:
        cycle += 1
        n_rows = 200_000
        n_cols = 60
        X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        y = (rng.random(n_rows) > 0.5).astype(np.float32)

        dtrain = xgb.DMatrix(X[:160_000], label=y[:160_000])
        dtest = xgb.DMatrix(X[160_000:], label=y[160_000:])

        t0 = time.time()
        xgb.train(
            {"objective": "binary:logistic", "tree_method": "hist",
             "device": "cuda", "max_depth": 14, "eta": 0.003,
             "subsample": 0.95, "colsample_bytree": 0.9,
             "max_bin": 2048, "min_child_weight": 1, "verbosity": 0},
            dtrain, num_boost_round=5000,
            evals=[(dtest, "test")], verbose_eval=False,
        )
        elapsed = time.time() - t0
        sys.stdout.write(f"  GPU-P40: cycle {cycle}, 200k x d14 x 5000r = {elapsed:.0f}s\n")
        sys.stdout.flush()


def gpu_burn_1050(end_time: float):
    """Burn GTX 1050 (GPU 1) with smaller XGBoost training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import xgboost as xgb

    rng = np.random.default_rng(888)
    cycle = 0

    while time.time() < end_time:
        cycle += 1
        n_rows = 30_000
        n_cols = 40
        X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
        y = (rng.random(n_rows) > 0.5).astype(np.float32)

        dtrain = xgb.DMatrix(X[:24_000], label=y[:24_000])
        dtest = xgb.DMatrix(X[24_000:], label=y[24_000:])

        t0 = time.time()
        xgb.train(
            {"objective": "binary:logistic", "tree_method": "hist",
             "device": "cuda", "max_depth": 10, "eta": 0.005,
             "subsample": 0.9, "colsample_bytree": 0.9,
             "max_bin": 512, "verbosity": 0},
            dtrain, num_boost_round=2000,
            evals=[(dtest, "test")], verbose_eval=False,
        )
        elapsed = time.time() - t0
        sys.stdout.write(f"  GPU-1050: cycle {cycle}, 30k x d10 x 2000r = {elapsed:.0f}s\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

def monitor_loop(end_time: float):
    """Print CPU/GPU status every 30 seconds."""
    while time.time() < end_time:
        try:
            remaining = (end_time - time.time()) / 60

            gpu_info = ""
            for gid in [0, 1]:
                try:
                    out = subprocess.check_output([
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits", f"--id={gid}",
                    ], text=True, timeout=5).strip()
                    parts = [p.strip() for p in out.split(",")]
                    gpu_info += f"GPU{gid}={parts[0]}%/{parts[1]}C/{parts[2]}MB "
                except Exception:
                    gpu_info += f"GPU{gid}=? "

            try:
                import psutil
                cpu_pct = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                ram_gb = mem.used / 1e9
            except ImportError:
                cpu_pct = 0
                ram_gb = 0

            msg = (f"\n  === [{remaining:.0f}min left] CPU={cpu_pct:.0f}% "
                   f"RAM={ram_gb:.1f}GB {gpu_info}===\n")
            sys.stdout.write(msg)
            sys.stdout.flush()
        except Exception:
            pass

        time.sleep(30)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else DURATION_MINUTES
    end_time = time.time() + duration * 60

    print("=" * 70)
    print(f"  BURN TEST — {duration} minutes")
    print(f"  {N_CPU_WORKERS} CPU workers + P40 GPU + GTX 1050 GPU")
    print(f"  End time: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print("=" * 70, flush=True)

    # Start monitor in main process thread
    monitor = threading.Thread(target=monitor_loop, args=(end_time,), daemon=True)
    monitor.start()

    # Start GPU processes (pass end_time as arg)
    gpu_targets = [(gpu_burn_p40, "P40"), (gpu_burn_1050, "GTX1050")]
    gpu_procs = []
    for target, name in gpu_targets:
        p = mp.Process(target=target, args=(end_time,), daemon=True)
        p.start()
        gpu_procs.append(p)
        print(f"  Started GPU worker: {name} (PID {p.pid})", flush=True)

    # Start CPU workers (pass end_time as arg)
    cpu_procs = []
    for i in range(N_CPU_WORKERS):
        p = mp.Process(target=cpu_burn_worker, args=(i, end_time), daemon=True)
        p.start()
        cpu_procs.append(p)

    print(f"  Started {N_CPU_WORKERS} CPU workers", flush=True)
    print(f"  Running until {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}...\n",
          flush=True)

    # Wait and restart dead GPU procs
    try:
        while time.time() < end_time:
            time.sleep(10)
            for i, (p, (target, name)) in enumerate(zip(gpu_procs, gpu_targets)):
                if not p.is_alive() and time.time() < end_time:
                    print(f"  Restarting dead GPU worker: {name}", flush=True)
                    p = mp.Process(target=target, args=(end_time,), daemon=True)
                    p.start()
                    gpu_procs[i] = p
    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    print(f"\n{'=' * 70}")
    print("  BURN TEST COMPLETE")
    print(f"{'=' * 70}", flush=True)

    for p in gpu_procs + cpu_procs:
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
