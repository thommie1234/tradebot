#!/bin/bash
# Follow-up script: per-account independent follow-up
# Each account's follow-up starts as soon as its entry optuna finishes
#
# Started: 2026-02-24 ~21:35
# Deadline: 05:00
# Entry trials: 150 (discovery=150, deep=150)

set -euo pipefail
cd /home/tradebot/tradebots

VENV=".venv/bin/python3"
LOG_DIR="audit/logs"
TIMESTAMP="20260224_213500"
LOG_FILE="$LOG_DIR/followup_${TIMESTAMP}.log"

FTMO_PID=2281098
BF_PID=2281615

FTMO_H4_DIR="ftmo/optuna/ritual_H4_${TIMESTAMP}"
BF_H4_DIR="bf/optuna/ritual_H4_${TIMESTAMP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FOLLOWUP: $1" | tee -a "$LOG_FILE"; }

log "========================================="
log "FOLLOW-UP SCRIPT START (independent per-account)"
log "  FTMO PID=$FTMO_PID  BF PID=$BF_PID"
log "  Entry trials: 150  Exit trials: 200"
log "========================================="

# ── Per-account follow-up function ──
run_followup() {
    local ACCT="$1"
    local ENTRY_PID="$2"
    local H4_DIR="$3"
    local GPU="$4"
    local WORKERS="$5"
    local ACCT_ID="$6"
    local PREFIX="$7"

    log "[$ACCT] Waiting for entry optuna PID=$ENTRY_PID..."

    while kill -0 $ENTRY_PID 2>/dev/null; do
        local done_count=$(grep -cE "✓|✗|no_data|insufficient|ERROR" "$LOG_DIR/optuna_${PREFIX}_H4_${TIMESTAMP}.log" 2>/dev/null || echo 0)
        log "[$ACCT] Progress: $done_count/177"
        sleep 300
    done
    log "[$ACCT] Entry optuna DONE"

    # Step 3: Merge (H4-only = just copy summary.csv)
    local COMBINED="${PREFIX}/optuna/ritual_combined_${TIMESTAMP}.csv"
    if [ -f "$H4_DIR/summary.csv" ]; then
        cp "$H4_DIR/summary.csv" "$COMBINED"
        local COUNT=$("$VENV" -c "import polars as pl; print(len(pl.read_csv('$COMBINED').filter(pl.col('status')=='ok')))")
        log "[$ACCT] Combined: $COUNT symbols with status=ok"
    else
        log "[$ACCT] ERROR: No summary.csv found in $H4_DIR"
        return 1
    fi

    if [ "$COUNT" -eq 0 ]; then
        log "[$ACCT] No symbols passed — skipping follow-up"
        return 0
    fi

    # Step 4a: Exit Optuna + sizing (200 trials)
    local EXIT_DIR="${PREFIX}/optuna/exit_${TIMESTAMP}"
    log "[$ACCT] Exit Optuna starting (GPU=$GPU, workers=$WORKERS, trials=200)..."
    CUDA_VISIBLE_DEVICES=$GPU "$VENV" -u "common/research/exit_optuna.py" \
        --account "$ACCT_ID" \
        --active --trials 200 --workers "$WORKERS" --use-sizing \
        --update-configs \
        --out-dir "$EXIT_DIR" \
        2>&1 | tee -a "$LOG_DIR/exit_${PREFIX}_${TIMESTAMP}.log"
    log "[$ACCT] Exit Optuna DONE"

    # Step 4b: Threshold WFO
    local THRESH_DIR="${PREFIX}/optuna/threshold_wfo_${TIMESTAMP}"
    log "[$ACCT] Threshold WFO starting..."
    CUDA_VISIBLE_DEVICES=$GPU "$VENV" -u "common/research/threshold_wfo.py" \
        --active --workers "$WORKERS" --update-configs \
        --out-dir "$THRESH_DIR" \
        2>&1 | tee -a "$LOG_DIR/threshold_${PREFIX}_${TIMESTAMP}.log"
    log "[$ACCT] Threshold WFO DONE"

    # Step 5: WFO Validate
    local EXIT_CSV=$(ls -t "$EXIT_DIR"/exit_best_per_symbol.csv 2>/dev/null | head -1)
    if [ -z "$EXIT_CSV" ]; then
        EXIT_CSV="${PREFIX}/optuna/exit_sizing_combined/exit_best_per_symbol.csv"
    fi

    local WFO_DIR="${PREFIX}/optuna/wfo_validated_${TIMESTAMP}"
    if [ -f "$EXIT_CSV" ]; then
        log "[$ACCT] WFO Validate starting..."
        CUDA_VISIBLE_DEVICES=$GPU "$VENV" -u "common/research/wfo_validate.py" \
            --account "$ACCT_ID" \
            --entry-csv "$COMBINED" \
            --exit-csv "$EXIT_CSV" \
            --out-dir "$WFO_DIR" \
            --workers "$WORKERS" \
            2>&1 | tee -a "$LOG_DIR/wfo_${PREFIX}_${TIMESTAMP}.log"
        log "[$ACCT] WFO Validate DONE"
    else
        log "[$ACCT] WARNING: No exit CSV found, skipping WFO validate"
    fi

    log "[$ACCT] ALL FOLLOW-UP STEPS COMPLETE"
}

# ── Launch both account follow-ups in parallel ──
run_followup "FTMO" $FTMO_PID "$FTMO_H4_DIR" 0 12 "ftmo_100k" "ftmo" &
FTMO_FOLLOWUP_PID=$!

run_followup "BF" $BF_PID "$BF_H4_DIR" 1 10 "bright_100k" "bf" &
BF_FOLLOWUP_PID=$!

log "Follow-up PIDs: FTMO=$FTMO_FOLLOWUP_PID  BF=$BF_FOLLOWUP_PID"

wait $FTMO_FOLLOWUP_PID 2>/dev/null || log "FTMO follow-up exited with error"
wait $BF_FOLLOWUP_PID 2>/dev/null || log "BF follow-up exited with error"

log "========================================="
log "ALL FOLLOW-UP COMPLETE — $(date)"
log "========================================="
