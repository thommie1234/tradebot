#!/bin/bash
# Follow-up script: waits for Entry Optuna, then runs steps 3-6
# Merge → Exit Optuna (+ sizing) → Threshold WFO → WFO Validate
#
# Started: 2026-02-24 ~21:00
# Deadline: 05:00

set -euo pipefail
cd /home/tradebot/tradebots

VENV=".venv/bin/python3"
LOG_DIR="audit/logs"
TIMESTAMP="20260224_203637"
LOG_FILE="$LOG_DIR/followup_${TIMESTAMP}.log"

FTMO_PID=1834851
BF_PID=1981217

FTMO_H4_DIR="ftmo/optuna/ritual_H4_${TIMESTAMP}"
BF_H4_DIR="bf/optuna/ritual_H4_${TIMESTAMP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FOLLOWUP: $1" | tee -a "$LOG_FILE"; }

log "========================================="
log "FOLLOW-UP SCRIPT START"
log "Waiting for Entry Optuna to finish..."
log "  FTMO PID=$FTMO_PID  BF PID=$BF_PID"
log "========================================="

# ── Wait for both optuna runs ──
while kill -0 $FTMO_PID 2>/dev/null || kill -0 $BF_PID 2>/dev/null; do
    ftmo_done=$(grep -cE "✓|✗|no_data|insufficient" "$LOG_DIR/optuna_ftmo_H4_${TIMESTAMP}.log" 2>/dev/null || echo 0)
    bf_done=$(grep -cE "✓|✗|no_data|insufficient" "$LOG_DIR/optuna_bf_H4_${TIMESTAMP}.log" 2>/dev/null || echo 0)
    log "  Progress: FTMO=$ftmo_done/177  BF=$bf_done/177"
    sleep 300  # check every 5 min
done
log "Both Entry Optuna runs DONE"

# ── Step 3: Merge — best TF per symbol (H4-only, so just copy) ──
log "STEP 3: Creating combined CSVs..."

FTMO_COMBINED="ftmo/optuna/ritual_combined_${TIMESTAMP}.csv"
BF_COMBINED="bf/optuna/ritual_combined_${TIMESTAMP}.csv"

# FTMO
if [ -f "$FTMO_H4_DIR/summary.csv" ]; then
    cp "$FTMO_H4_DIR/summary.csv" "$FTMO_COMBINED"
    FTMO_COUNT=$("$VENV" -c "import polars as pl; print(len(pl.read_csv('$FTMO_COMBINED').filter(pl.col('status')=='ok')))")
    log "FTMO combined: $FTMO_COUNT symbols"
else
    log "ERROR: No FTMO summary.csv found"
    FTMO_COUNT=0
fi

# BF
if [ -f "$BF_H4_DIR/summary.csv" ]; then
    cp "$BF_H4_DIR/summary.csv" "$BF_COMBINED"
    BF_COUNT=$("$VENV" -c "import polars as pl; print(len(pl.read_csv('$BF_COMBINED').filter(pl.col('status')=='ok')))")
    log "BF combined: $BF_COUNT symbols"
else
    log "ERROR: No BF summary.csv found"
    BF_COUNT=0
fi

# ── Step 4a: Exit Optuna + sizing (FTMO on P40, BF on GTX 1050 — parallel) ──
log "STEP 4a: Exit Optuna + position sizing..."

FTMO_EXIT_DIR="ftmo/optuna/exit_${TIMESTAMP}"
BF_EXIT_DIR="bf/optuna/exit_${TIMESTAMP}"

if [ "$FTMO_COUNT" -gt 0 ]; then
    (
        log "FTMO Exit Optuna starting (P40)..."
        CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/exit_optuna.py" \
            --account ftmo_100k \
            --active --trials 400 --workers 12 --use-sizing \
            --update-configs \
            --out-dir "$FTMO_EXIT_DIR" \
            2>&1 | tee -a "$LOG_DIR/exit_ftmo_${TIMESTAMP}.log"
        log "FTMO Exit Optuna DONE"
    ) &
    EXIT_FTMO_PID=$!
fi

if [ "$BF_COUNT" -gt 0 ]; then
    (
        log "BF Exit Optuna starting (GTX 1050)..."
        CUDA_VISIBLE_DEVICES=1 "$VENV" -u "common/research/exit_optuna.py" \
            --account bright_100k \
            --active --trials 400 --workers 10 --use-sizing \
            --update-configs \
            --out-dir "$BF_EXIT_DIR" \
            2>&1 | tee -a "$LOG_DIR/exit_bf_${TIMESTAMP}.log"
        log "BF Exit Optuna DONE"
    ) &
    EXIT_BF_PID=$!
fi

# Wait for exit optuna
wait ${EXIT_FTMO_PID:-} 2>/dev/null || true
wait ${EXIT_BF_PID:-} 2>/dev/null || true
log "All Exit Optuna runs DONE"

# ── Step 4b: Threshold WFO (both accounts — parallel) ──
log "STEP 4b: Threshold WFO optimization..."

FTMO_THRESH_DIR="ftmo/optuna/threshold_wfo_${TIMESTAMP}"
BF_THRESH_DIR="bf/optuna/threshold_wfo_${TIMESTAMP}"

if [ "$FTMO_COUNT" -gt 0 ]; then
    (
        log "FTMO Threshold WFO starting..."
        CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/threshold_wfo.py" \
            --active --workers 12 --update-configs \
            --out-dir "$FTMO_THRESH_DIR" \
            2>&1 | tee -a "$LOG_DIR/threshold_ftmo_${TIMESTAMP}.log"
        log "FTMO Threshold WFO DONE"
    ) &
    THRESH_FTMO_PID=$!
fi

if [ "$BF_COUNT" -gt 0 ]; then
    (
        log "BF Threshold WFO starting..."
        CUDA_VISIBLE_DEVICES=1 "$VENV" -u "common/research/threshold_wfo.py" \
            --active --workers 10 --update-configs \
            --out-dir "$BF_THRESH_DIR" \
            2>&1 | tee -a "$LOG_DIR/threshold_bf_${TIMESTAMP}.log"
        log "BF Threshold WFO DONE"
    ) &
    THRESH_BF_PID=$!
fi

wait ${THRESH_FTMO_PID:-} 2>/dev/null || true
wait ${THRESH_BF_PID:-} 2>/dev/null || true
log "All Threshold WFO runs DONE"

# ── Step 5: WFO Validate (both accounts — parallel) ──
log "STEP 5: WFO Validation..."

# Find exit CSVs
FTMO_EXIT_CSV=$(ls -t "$FTMO_EXIT_DIR"/exit_best_per_symbol.csv 2>/dev/null | head -1)
if [ -z "$FTMO_EXIT_CSV" ]; then
    FTMO_EXIT_CSV="ftmo/optuna/exit_sizing_combined/exit_best_per_symbol.csv"
fi
BF_EXIT_CSV=$(ls -t "$BF_EXIT_DIR"/exit_best_per_symbol.csv 2>/dev/null | head -1)

FTMO_WFO_DIR="ftmo/optuna/wfo_validated_${TIMESTAMP}"
BF_WFO_DIR="bf/optuna/wfo_validated_${TIMESTAMP}"

if [ "$FTMO_COUNT" -gt 0 ] && [ -f "$FTMO_EXIT_CSV" ]; then
    (
        log "FTMO WFO Validate starting..."
        CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/wfo_validate.py" \
            --account ftmo_100k \
            --entry-csv "$FTMO_COMBINED" \
            --exit-csv "$FTMO_EXIT_CSV" \
            --out-dir "$FTMO_WFO_DIR" \
            --workers 6 \
            2>&1 | tee -a "$LOG_DIR/wfo_ftmo_${TIMESTAMP}.log"
        log "FTMO WFO Validate DONE"
    ) &
    WFO_FTMO_PID=$!
fi

if [ "$BF_COUNT" -gt 0 ] && [ -n "$BF_EXIT_CSV" ] && [ -f "$BF_EXIT_CSV" ]; then
    (
        log "BF WFO Validate starting..."
        CUDA_VISIBLE_DEVICES=1 "$VENV" -u "common/research/wfo_validate.py" \
            --account bright_100k \
            --entry-csv "$BF_COMBINED" \
            --exit-csv "$BF_EXIT_CSV" \
            --out-dir "$BF_WFO_DIR" \
            --workers 4 \
            2>&1 | tee -a "$LOG_DIR/wfo_bf_${TIMESTAMP}.log"
        log "BF WFO Validate DONE"
    ) &
    WFO_BF_PID=$!
fi

wait ${WFO_FTMO_PID:-} 2>/dev/null || true
wait ${WFO_BF_PID:-} 2>/dev/null || true
log "All WFO Validate runs DONE"

# ── Summary ──
log "========================================="
log "FOLLOW-UP COMPLETE — $(date)"
log ""
log "Output:"
[ -f "$FTMO_COMBINED" ] && log "  FTMO entry:     $FTMO_COMBINED"
[ -f "$FTMO_EXIT_CSV" ] && log "  FTMO exit:      $FTMO_EXIT_CSV"
[ -d "$FTMO_THRESH_DIR" ] && log "  FTMO threshold: $FTMO_THRESH_DIR"
[ -d "$FTMO_WFO_DIR" ] && log "  FTMO WFO:       $FTMO_WFO_DIR"
[ -f "$BF_COMBINED" ] && log "  BF entry:       $BF_COMBINED"
[ -n "${BF_EXIT_CSV:-}" ] && [ -f "${BF_EXIT_CSV:-}" ] && log "  BF exit:        $BF_EXIT_CSV"
[ -d "$BF_THRESH_DIR" ] && log "  BF threshold:   $BF_THRESH_DIR"
[ -d "$BF_WFO_DIR" ] && log "  BF WFO:         $BF_WFO_DIR"
log "========================================="
