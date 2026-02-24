#!/bin/bash
# =============================================================================
# Full Entry + Exit Optuna pipeline — all symbols × all timeframes
# Walk-forward validation throughout.
#
# Phase 1: Entry Optuna (XGBoost hyperparams) on both GPUs in parallel
#   P40  (cuda:0) → M1 → M15 → M30  (14 workers)
#   1050 (cuda:1) → H1 → H4          (14 workers)
# Phase 2: Exit Optuna (SL/TP/BE/trail params) on all 28 CPUs
# Phase 3: Cross-validation summary — entry × exit viability check
#
# Usage:
#   bash research/run_full_optuna.sh
#   bash research/run_full_optuna.sh --trials-entry 40 --trials-exit 200
# =============================================================================

set -euo pipefail

cd /home/tradebot/tradebots
VENV=".venv/bin/python3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="models/optuna_results/full_${TIMESTAMP}"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"

# Tunable defaults
TRIALS_ENTRY=${TRIALS_ENTRY:-80}
TRIALS_EXIT=${TRIALS_EXIT:-200}
DISCOVERY_CUTOFF=${DISCOVERY_CUTOFF:-40}
WORKERS_P40=${WORKERS_P40:-8}     # P40 23GB — 8 workers to avoid RAM OOM (each loads full dataset)
WORKERS_1050=${WORKERS_1050:-4}   # 1050 2GB — 4 GPU workers fit in VRAM
WORKERS_EXIT=${WORKERS_EXIT:-28}  # Exit is CPU-only (numpy simulation)

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials-entry) TRIALS_ENTRY="$2"; shift 2 ;;
        --trials-exit)  TRIALS_EXIT="$2"; shift 2 ;;
        --discovery)    DISCOVERY_CUTOFF="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "================================================================="
echo "  FULL OPTUNA PIPELINE — $(date)"
echo "  Entry trials: ${TRIALS_ENTRY}  |  Exit trials: ${TRIALS_EXIT}"
echo "  Discovery cutoff: ${DISCOVERY_CUTOFF}"
echo "  Output: ${OUT_BASE}"
echo "================================================================="
echo ""

# ─────────────────────────────────────────────────────────────────────
# PHASE 1: Entry Optuna — XGBoost hyperparameters (WFO)
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 1: Entry Optuna (all TFs, all symbols, WFO) ==="
PHASE1_START=$(date +%s)

# Helper: get train/test size for timeframe
# ~2 years tick data → M1: 210k bars, M15: 14k, M30: 7k, H1: 3.5k, H4: 900
tf_train_size() {
    case "$1" in
        M1)  echo 10000 ;;
        M15) echo 4800 ;;
        M30) echo 3000 ;;
        H1)  echo 1000 ;;
        H4)  echo 250 ;;
        *)   echo 4800 ;;
    esac
}
tf_test_size() {
    case "$1" in
        M1)  echo 2500 ;;
        M15) echo 1200 ;;
        M30) echo 800 ;;
        H1)  echo 300 ;;
        H4)  echo 80 ;;
        *)   echo 1200 ;;
    esac
}

# P40 stream: M1 → M15 → M30 (larger datasets, faster GPU)
(
    export CUDA_VISIBLE_DEVICES=0
    for tf in M1 M15 M30; do
        tr=$(tf_train_size "$tf")
        te=$(tf_test_size "$tf")
        echo "[P40] $(date +%H:%M:%S) Starting entry Optuna: ${tf} (train=${tr} test=${te})..."
        $VENV -u common/research/optuna_orchestrator.py \
            --all \
            --trials "$TRIALS_ENTRY" \
            --timeframe "$tf" \
            --workers "$WORKERS_P40" \
            --trial-jobs 2 \
            --train-size "$tr" \
            --test-size "$te" \
            --discovery-cutoff "$DISCOVERY_CUTOFF" \
            --out-dir "${OUT_BASE}/entry_${tf}" \
            2>&1 | tee "${LOG_DIR}/entry_${tf}_p40.log"
        echo "[P40] $(date +%H:%M:%S) Done: ${tf}"
    done
    echo "[P40] ALL DONE"
) &
P40_PID=$!

# 1050 stream: H1 → H4 (2GB VRAM — fewer workers, smaller data)
(
    export CUDA_VISIBLE_DEVICES=1
    for tf in H1 H4; do
        tr=$(tf_train_size "$tf")
        te=$(tf_test_size "$tf")
        echo "[1050] $(date +%H:%M:%S) Starting entry Optuna: ${tf} (train=${tr} test=${te})..."
        $VENV -u common/research/optuna_orchestrator.py \
            --all \
            --trials "$TRIALS_ENTRY" \
            --timeframe "$tf" \
            --workers "$WORKERS_1050" \
            --trial-jobs 1 \
            --train-size "$tr" \
            --test-size "$te" \
            --discovery-cutoff "$DISCOVERY_CUTOFF" \
            --out-dir "${OUT_BASE}/entry_${tf}" \
            2>&1 | tee "${LOG_DIR}/entry_${tf}_1050.log"
        echo "[1050] $(date +%H:%M:%S) Done: ${tf}"
    done
    echo "[1050] ALL DONE"
) &
GTX_PID=$!

echo "  P40 PID=$P40_PID  (M1, M15, M30)"
echo "  1050 PID=$GTX_PID (H1, H4)"
echo "  Waiting for both GPU streams..."
wait $P40_PID $GTX_PID
PHASE1_END=$(date +%s)
echo ""
echo "=== PHASE 1 COMPLETE in $(( (PHASE1_END - PHASE1_START) / 60 )) min ==="
echo ""

# ─────────────────────────────────────────────────────────────────────
# PHASE 2: Exit Optuna — SL/TP/breakeven/trailing params (WFO)
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 2: Exit Optuna (all TFs, all symbols, WFO) ==="
PHASE2_START=$(date +%s)

$VENV -u common/research/exit_optuna.py \
    --all \
    --trials "$TRIALS_EXIT" \
    --timeframes M1,M15,M30,H1,H4 \
    --workers "$WORKERS_EXIT" \
    --out-dir "${OUT_BASE}/exit" \
    --update-configs \
    2>&1 | tee "${LOG_DIR}/exit_all.log"

PHASE2_END=$(date +%s)
echo ""
echo "=== PHASE 2 COMPLETE in $(( (PHASE2_END - PHASE2_START) / 60 )) min ==="
echo ""

# ─────────────────────────────────────────────────────────────────────
# PHASE 3: Cross-validation — entry × exit merged summary
# ─────────────────────────────────────────────────────────────────────
echo "=== PHASE 3: Cross-validation summary ==="

$VENV -u - << 'PYEOF'
import os, sys, glob
import polars as pl
from pathlib import Path

out_base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_BASE", "")
if not out_base:
    # Find most recent full_* directory
    candidates = sorted(glob.glob("models/optuna_results/full_*"))
    out_base = candidates[-1] if candidates else "."

print(f"\n  Loading results from: {out_base}\n")

# Collect entry results
entry_rows = []
for tf in ["M1", "M15", "M30", "H1", "H4"]:
    csv = os.path.join(out_base, f"entry_{tf}", "summary.csv")
    if os.path.exists(csv):
        df = pl.read_csv(csv)
        if "timeframe" not in df.columns:
            df = df.with_columns(pl.lit(tf).alias("timeframe"))
        entry_rows.append(df)

if not entry_rows:
    print("  No entry results found!")
    sys.exit(0)

entry_df = pl.concat(entry_rows, how="diagonal_relaxed")
entry_ok = entry_df.filter(pl.col("status") == "ok")

# Collect exit results
exit_csv = os.path.join(out_base, "exit", "exit_summary.csv")
exit_ok = None
if os.path.exists(exit_csv):
    exit_df = pl.read_csv(exit_csv)
    exit_ok = exit_df.filter(pl.col("status") == "ok")

# Best entry TF per symbol (by calmar_score or best_ev)
ev_col = "calmar_score" if "calmar_score" in entry_ok.columns else "best_ev"
best_entry = (
    entry_ok
    .sort(ev_col, descending=True)
    .group_by("symbol")
    .first()
    .select(["symbol", "timeframe", ev_col])
    .rename({"timeframe": "entry_tf", ev_col: "entry_ev"})
    .sort("symbol")
)

print(f"  Entry: {entry_ok.height} successful runs, {best_entry.height} unique symbols")

if exit_ok is not None and exit_ok.height > 0:
    best_exit = (
        exit_ok
        .sort("best_ev", descending=True)
        .group_by("symbol")
        .first()
        .select([
            "symbol", "timeframe", "best_ev",
            "best_atr_sl_mult", "best_atr_tp_mult",
            "best_breakeven_atr", "best_trail_activation_atr",
            "best_trail_distance_atr", "best_horizon",
        ])
        .rename({"timeframe": "exit_tf", "best_ev": "exit_ev"})
        .sort("symbol")
    )
    print(f"  Exit:  {exit_ok.height} successful runs, {best_exit.height} unique symbols")

    # Cross-join: symbols with positive EV on BOTH entry and exit
    merged = best_entry.join(best_exit, on="symbol", how="inner")
    viable = merged.filter(
        (pl.col("entry_ev") > 0) & (pl.col("exit_ev") > 0)
    ).sort("exit_ev", descending=True)

    print(f"\n  CROSS-VALIDATED VIABLE SYMBOLS: {viable.height}")
    print(f"  {'Symbol':20s} {'Entry TF':8s} {'Entry EV':>10s} {'Exit TF':8s} {'Exit EV':>10s} "
          f"{'SL':>6s} {'TP':>6s} {'BE':>6s} {'Trail':>12s} {'Horizon':>7s}")
    print("  " + "-" * 110)

    for row in viable.iter_rows(named=True):
        trail_str = f"{row['best_trail_activation_atr']:.2f}/{row['best_trail_distance_atr']:.2f}"
        print(f"  {row['symbol']:20s} {row['entry_tf']:8s} {row['entry_ev']:>+10.6f} "
              f"{row['exit_tf']:8s} {row['exit_ev']:>+10.6f} "
              f"{row['best_atr_sl_mult']:>6.2f} {row['best_atr_tp_mult']:>6.2f} "
              f"{row['best_breakeven_atr']:>6.2f} {trail_str:>12s} {row['best_horizon']:>7d}")

    # Save merged results
    merged_path = os.path.join(out_base, "cross_validated.csv")
    merged.write_csv(merged_path)
    viable_path = os.path.join(out_base, "viable_symbols.csv")
    viable.write_csv(viable_path)
    print(f"\n  Saved: {merged_path}")
    print(f"  Saved: {viable_path}")
else:
    print("  No exit results found — skipping cross-validation")

# Also save full entry summary
entry_all_path = os.path.join(out_base, "entry_all_tfs.csv")
entry_ok.write_csv(entry_all_path)
print(f"  Saved: {entry_all_path}")
PYEOF

TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - PHASE1_START) / 60 ))

echo ""
echo "================================================================="
echo "  PIPELINE COMPLETE — Total: ${TOTAL_MIN} min"
echo "  Entry:  $(( (PHASE1_END - PHASE1_START) / 60 )) min"
echo "  Exit:   $(( (PHASE2_END - PHASE2_START) / 60 )) min"
echo "  Output: ${OUT_BASE}"
echo "================================================================="
