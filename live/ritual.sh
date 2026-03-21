#!/bin/bash
# Unified Ritual — Account-agnostic reoptimization pipeline
# Replaces sunday_ritual.sh + daily_ritual.sh
#
# Pipeline (7 steps):
#   1. Download tick data (--skip-download to skip)
#   2. Entry Optuna (multi-TF, multi-GPU, --no-kill)
#   3. Merge results → best TF per symbol
#   4. Exit Optuna (400 trials)
#   5. WFO Validate → portfolio selection
#   6. Retrain + update config.json
#   7. Restart bot
#
# Usage:
#   bash common/live/ritual.sh --account ftmo_100k
#   bash common/live/ritual.sh --account bright_100k --skip-download
#   bash common/live/ritual.sh --account ftmo_100k --step 5  (start from step 5)
#   bash common/live/ritual.sh --account ftmo_100k --dry-run

set -euo pipefail
cd /home/tradebot/tradebots

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ACCOUNT=""
SKIP_DOWNLOAD=false
START_STEP=1
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)   ACCOUNT="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --step)      START_STEP="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$ACCOUNT" ]; then
    echo "ERROR: --account is required (e.g. ftmo_100k, bright_100k)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Account-specific paths (derived from --account)
# ---------------------------------------------------------------------------
case "$ACCOUNT" in
    ftmo_100k)
        ACCT_PREFIX="ftmo"
        SERVICE="sovereign-bot-ftmo"
        CONFIG_JSON="ftmo/config.json"
        OPTUNA_DIR="ftmo/optuna"
        MODEL_DIR="ftmo/models"
        ;;
    bright_100k)
        ACCT_PREFIX="bf"
        SERVICE="sovereign-bot-bf"
        CONFIG_JSON="bf/config.json"
        OPTUNA_DIR="bf/optuna"
        MODEL_DIR="bf/models"
        ;;
    *)
        echo "ERROR: Unknown account '$ACCOUNT' (expected ftmo_100k or bright_100k)"
        exit 1
        ;;
esac

VENV=".venv/bin/python3"
LOG_DIR="audit/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ritual_${ACCT_PREFIX}_${TIMESTAMP}.log"
START_EPOCH=$(date +%s)

# Optuna settings
TRIALS=600
DISCOVERY=500
TRAIN_SIZE=2400
TEST_SIZE=600
PURGE=24
EMBARGO=24

# Budget: 20 hours main + 4 hours overflow
BUDGET_SECS=$((20 * 3600))
OVERFLOW_SECS=$((24 * 3600))

mkdir -p "$LOG_DIR" "$OPTUNA_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] RITUAL($ACCT_PREFIX): $1" | tee -a "$LOG_FILE"; }

past_budget() {
    local elapsed=$(( $(date +%s) - START_EPOCH ))
    [ "$elapsed" -ge "$BUDGET_SECS" ]
}

past_overflow() {
    local elapsed=$(( $(date +%s) - START_EPOCH ))
    [ "$elapsed" -ge "$OVERFLOW_SECS" ]
}

elapsed_hours() {
    echo $(( ($(date +%s) - START_EPOCH) / 3600 ))
}

run_or_dry() {
    if [ "$DRY_RUN" = true ]; then
        log "[DRY-RUN] Would run: $*"
    else
        "$@"
    fi
}

log "========================================="
log "RITUAL START — account=$ACCOUNT — $(date)"
log "Budget: 20h | Overflow: 24h"
log "Skip download: $SKIP_DOWNLOAD | Start step: $START_STEP | Dry run: $DRY_RUN"
log "========================================="

# ── Step 1: Download tick data ──
if [ "$START_STEP" -le 1 ]; then
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log "STEP 1: Skipping tick data download (--skip-download)"
    else
        DOW=$(date +%u)  # 1=Mon, 7=Sun
        if [ "$DOW" -eq 6 ] || [ "$DOW" -eq 7 ]; then
            log "STEP 1: Weekend — downloading fresh tick data..."
            log "Stopping $SERVICE and mt5-bridge-proxy for download..."
            run_or_dry systemctl --user stop "$SERVICE" 2>/dev/null || true
            run_or_dry systemctl --user stop mt5-bridge-proxy.service 2>/dev/null || true
            sleep 5

            WINE_LAUNCHER="common/live/run_wine.sh"
            WORKERS=6
            SPLIT_DIR="common/tools/symbol_splits"
            mkdir -p "$SPLIT_DIR"
            run_or_dry "$VENV" "common/tools/make_symbol_splits.py" \
                --workers "$WORKERS" --out-dir "$SPLIT_DIR" 2>&1 | tee -a "$LOG_FILE"

            DOWNLOAD_PIDS=()
            for f in "$SPLIT_DIR"/symbols_part_*.xlsx; do
                [ -f "$f" ] || continue
                part="$(basename "$f" .xlsx)"
                dl_log="$LOG_DIR/ritual_download_${part}_${TIMESTAMP}.log"
                log "Starting download: $part"
                if [ "$DRY_RUN" = false ]; then
                    SYMBOLS_FILE_PATH="$f" bash "$WINE_LAUNCHER" strategy_bot.py --download-data > "$dl_log" 2>&1 &
                    DOWNLOAD_PIDS+=($!)
                fi
            done

            if [ "$DRY_RUN" = false ] && [ ${#DOWNLOAD_PIDS[@]} -gt 0 ]; then
                log "Waiting for ${#DOWNLOAD_PIDS[@]} download workers (max 4h)..."
                DL_START=$(date +%s)
                DL_TIMEOUT=14400
                for pid in "${DOWNLOAD_PIDS[@]}"; do
                    elapsed=$(( $(date +%s) - DL_START ))
                    remaining=$(( DL_TIMEOUT - elapsed ))
                    if [ "$remaining" -le 0 ]; then
                        log "Download timeout — killing remaining workers"
                        kill "${DOWNLOAD_PIDS[@]}" 2>/dev/null || true
                        break
                    fi
                    timeout "$remaining" tail --pid=$pid -f /dev/null 2>/dev/null || true
                done
            fi
            log "Tick data download complete"

            log "Restarting mt5-bridge-proxy..."
            run_or_dry systemctl --user start mt5-bridge-proxy.service
            sleep 10

            log "Restarting $SERVICE..."
            run_or_dry systemctl --user start "$SERVICE"
            sleep 5
        else
            log "STEP 1: Weekday — skipping tick data download"
        fi
    fi
fi

# ── Step 2: Entry Optuna (multi-TF, multi-GPU) ──
H4_DIR="$OPTUNA_DIR/ritual_H4_${TIMESTAMP}"
M30_DIR="$OPTUNA_DIR/ritual_M30_${TIMESTAMP}"
M15_DIR="$OPTUNA_DIR/ritual_M15_${TIMESTAMP}"
H1_DIR="$OPTUNA_DIR/ritual_H1_${TIMESTAMP}"

OPTUNA_COMMON="--account $ACCOUNT --all --trials $TRIALS --discovery-cutoff $DISCOVERY --no-kill \
    --train-size $TRAIN_SIZE --test-size $TEST_SIZE \
    --purge $PURGE --embargo $EMBARGO \
    --pt-mult 2.0 --sl-mult 1.5 --horizon-bars 24 --z-threshold 1.0"

if [ "$START_STEP" -le 2 ]; then
    log "STEP 2: Entry Optuna optimization (account=$ACCOUNT, --no-kill)..."
    log "  P40 (CUDA:0): H4 → M30 → M15 (sequential, 4 workers)"
    log "  GTX (CUDA:1): H1 (2 workers, parallel)"

    # P40 pipeline: H4 (production TF) → M30 → M15
    (
        log "Starting H4 on P40... ($(elapsed_hours)h elapsed)"
        run_or_dry env CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
            $OPTUNA_COMMON --workers 4 --trial-jobs 2 \
            --timeframe H4 \
            --out-dir "$H4_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_H4_${ACCT_PREFIX}_${TIMESTAMP}.log"
        log "H4 complete. ($(elapsed_hours)h elapsed)"

        if past_budget; then
            log "BUDGET REACHED after H4 — skipping M30+M15"
        else
            log "Starting M30 on P40... ($(elapsed_hours)h elapsed)"
            run_or_dry env CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
                $OPTUNA_COMMON --workers 4 --trial-jobs 2 \
                --timeframe M30 \
                --out-dir "$M30_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_M30_${ACCT_PREFIX}_${TIMESTAMP}.log"
            log "M30 complete. ($(elapsed_hours)h elapsed)"

            if past_budget; then
                log "BUDGET REACHED after M30 — skipping M15"
            else
                log "Starting M15 on P40... ($(elapsed_hours)h elapsed)"
                run_or_dry env CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
                    $OPTUNA_COMMON --workers 4 --trial-jobs 2 \
                    --timeframe M15 \
                    --out-dir "$M15_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_M15_${ACCT_PREFIX}_${TIMESTAMP}.log"
                log "M15 complete. ($(elapsed_hours)h elapsed)"
            fi
        fi
    ) &
    P40_PID=$!

    # GTX 1050: H1
    (
        log "Starting H1 on GTX 1050... ($(elapsed_hours)h elapsed)"
        run_or_dry env CUDA_VISIBLE_DEVICES=1 "$VENV" -u "common/research/optuna_orchestrator.py" \
            $OPTUNA_COMMON --workers 2 --trial-jobs 1 \
            --timeframe H1 \
            --out-dir "$H1_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_H1_${ACCT_PREFIX}_${TIMESTAMP}.log"
        log "H1 complete. ($(elapsed_hours)h elapsed)"
    ) &
    GTX_PID=$!

    # Wait for both GPU pipelines
    log "Waiting for Optuna runs (PIDs: P40=$P40_PID, GTX=$GTX_PID)..."
    while kill -0 $P40_PID 2>/dev/null || kill -0 $GTX_PID 2>/dev/null; do
        if past_overflow; then
            log "OVERFLOW DEADLINE — killing remaining Optuna runs ($(elapsed_hours)h elapsed)"
            kill $P40_PID 2>/dev/null || true
            kill $GTX_PID 2>/dev/null || true
            sleep 5
            pkill -f "optuna_orchestrator.*ritual_" 2>/dev/null || true
            break
        fi
        sleep 60
    done
    wait $P40_PID 2>/dev/null || true
    wait $GTX_PID 2>/dev/null || true
    log "All Optuna runs finished ($(elapsed_hours)h elapsed)"
fi

# ── Step 3: Merge results — best TF per symbol ──
COMBINED_CSV="$OPTUNA_DIR/ritual_combined_${TIMESTAMP}.csv"
if [ "$START_STEP" -le 3 ]; then
    log "STEP 3: Merging multi-TF results..."

    # If starting from step 3+, find latest ritual dirs
    if [ "$START_STEP" -ge 3 ]; then
        H4_DIR=$(ls -dt "$OPTUNA_DIR"/ritual_H4_* 2>/dev/null | head -1)
        M30_DIR=$(ls -dt "$OPTUNA_DIR"/ritual_M30_* 2>/dev/null | head -1)
        M15_DIR=$(ls -dt "$OPTUNA_DIR"/ritual_M15_* 2>/dev/null | head -1)
        H1_DIR=$(ls -dt "$OPTUNA_DIR"/ritual_H1_* 2>/dev/null | head -1)
    fi

    "$VENV" -c "
import polars as pl, os, json, yaml
from datetime import datetime

results = {}
for tf, d in [('H4', '$H4_DIR'), ('M30', '$M30_DIR'), ('M15', '$M15_DIR'), ('H1', '$H1_DIR')]:
    csv = os.path.join(d, 'summary.csv') if d else ''
    if not csv or not os.path.exists(csv):
        print(f'  {tf}: no summary.csv found (dir={d})')
        continue
    df = pl.read_csv(csv)
    ok = df.filter(pl.col('status') == 'ok')
    print(f'  {tf}: {len(ok)} completed out of {len(df)}')
    for row in ok.iter_rows(named=True):
        sym = row['symbol']
        ev = float(row['best_ev'])
        if sym not in results or ev > results[sym]['ev']:
            results[sym] = {'tf': tf, 'ev': ev, 'csv': csv}

print(f'\nBest TF per symbol: {len(results)} symbols')
for sym in sorted(results):
    info = results[sym]
    marker = ' *' if info['ev'] < 0 else ''
    print(f'  {sym}: {info[\"tf\"]} (EV={info[\"ev\"]:.6f}){marker}')

combined_rows = []
for tf, d in [('H4', '$H4_DIR'), ('M30', '$M30_DIR'), ('M15', '$M15_DIR'), ('H1', '$H1_DIR')]:
    csv = os.path.join(d, 'summary.csv') if d else ''
    if not csv or not os.path.exists(csv):
        continue
    df = pl.read_csv(csv).filter(pl.col('status') == 'ok')
    for row in df.iter_rows(named=True):
        sym = row['symbol']
        if sym in results and results[sym]['tf'] == tf:
            combined_rows.append(row)

if combined_rows:
    combined = pl.DataFrame(combined_rows)
    combined.write_csv('$COMBINED_CSV')
    print(f'\nSaved: $COMBINED_CSV ({len(combined)} rows)')

    mtf = {'symbols': {}}
    today = datetime.now().strftime('%Y-%m-%d')
    for sym, info in sorted(results.items()):
        if info['tf'] != 'H1':
            mtf['symbols'][sym] = {
                'timeframe': info['tf'],
                'optuna_ev': round(info['ev'], 6),
                'source': f'ritual_{today} ({info[\"tf\"]})',
            }
    if mtf['symbols']:
        with open('config/multi_tf.yaml', 'w') as f:
            f.write('# Multi-timeframe symbol configuration\n')
            f.write(f'# Auto-generated by ritual {today} (account=$ACCOUNT)\n\n')
            yaml.dump(mtf, f, default_flow_style=False)
        print(f'Updated multi_tf.yaml: {len(mtf[\"symbols\"])} non-H1 symbols')
else:
    print('WARNING: No completed results')
" 2>&1 | tee -a "$LOG_FILE"

    # Count results
    RESULTS_COUNT=0
    if [ -f "$COMBINED_CSV" ]; then
        RESULTS_COUNT=$("$VENV" -c "
import polars as pl
df = pl.read_csv('$COMBINED_CSV')
print(len(df))
" 2>/dev/null || echo "0")
    fi
    log "Total symbols: $RESULTS_COUNT"
fi

if [ ! -f "$COMBINED_CSV" ]; then
    # Try to find latest combined CSV
    COMBINED_CSV=$(ls -t "$OPTUNA_DIR"/ritual_combined_*.csv 2>/dev/null | head -1)
    if [ -z "$COMBINED_CSV" ]; then
        log "ERROR: No combined CSV found — cannot continue"
        exit 1
    fi
    log "Using existing combined CSV: $COMBINED_CSV"
fi

# ── Step 4: Exit Optuna ──
EXIT_DIR="$OPTUNA_DIR/exit_${TIMESTAMP}"
EXIT_CSV="$EXIT_DIR/exit_best_per_symbol.csv"
if [ "$START_STEP" -le 4 ]; then
    log "STEP 4: Exit-parameter Optuna (400 trials, account=$ACCOUNT)..."
    run_or_dry "$VENV" -u "common/research/exit_optuna.py" \
        --account "$ACCOUNT" \
        --active --trials 400 --workers 12 --update-configs \
        --out-dir "$EXIT_DIR" \
        2>&1 | tee -a "$LOG_DIR/ritual_exit_${ACCT_PREFIX}_${TIMESTAMP}.log"
    log "Exit-parameter optimization complete"
fi

# Find exit CSV if starting from step 5+
if [ ! -f "$EXIT_CSV" ]; then
    EXIT_CSV=$(ls -t "$OPTUNA_DIR"/exit_*/exit_best_per_symbol.csv 2>/dev/null | head -1)
    if [ -z "$EXIT_CSV" ]; then
        # Fallback to exit_sizing_combined
        EXIT_CSV="$OPTUNA_DIR/exit_sizing_combined/exit_best_per_symbol.csv"
    fi
fi

# ── Step 5: WFO Validate ──
WFO_DIR="$OPTUNA_DIR/wfo_validated_${TIMESTAMP}"
if [ "$START_STEP" -le 5 ]; then
    log "STEP 5: WFO Validation (account=$ACCOUNT)..."
    log "  Entry CSV: $COMBINED_CSV"
    log "  Exit CSV:  $EXIT_CSV"
    run_or_dry "$VENV" -u "common/research/wfo_validate.py" \
        --account "$ACCOUNT" \
        --entry-csv "$COMBINED_CSV" \
        --exit-csv "$EXIT_CSV" \
        --out-dir "$WFO_DIR" \
        --workers 6 \
        2>&1 | tee -a "$LOG_DIR/ritual_wfo_${ACCT_PREFIX}_${TIMESTAMP}.log"
    log "WFO Validation complete"
fi

# ── Step 6: Retrain + update config.json ──
if [ "$START_STEP" -le 6 ]; then
    log "STEP 6: Retraining models (account=$ACCOUNT)..."
    run_or_dry env CUDA_VISIBLE_DEVICES=0 \
        OPTUNA_CSV_OVERRIDE="$(realpath "$COMBINED_CSV")" \
        "$VENV" -u "common/live/run_bot.py" \
        --retrain --account-id "$ACCOUNT" \
        2>&1 | tail -30 | tee -a "$LOG_FILE"

    # Update paths.yaml
    sed -i "s|^optuna_csv:.*|optuna_csv: ${COMBINED_CSV}|" config/paths.yaml
    log "Retrain complete. Updated paths.yaml → $COMBINED_CSV"

    # Copy WFO portfolio configs if available
    WFO_CONFIG="$WFO_DIR/portfolio_configs.json"
    if [ -f "$WFO_CONFIG" ]; then
        cp "$WFO_CONFIG" "$CONFIG_JSON.wfo_backup_${TIMESTAMP}"
        log "WFO portfolio config backed up to $CONFIG_JSON.wfo_backup_${TIMESTAMP}"
        # Merge WFO configs into main config
        "$VENV" -c "
import json
with open('$CONFIG_JSON') as f:
    main = json.load(f)
with open('$WFO_CONFIG') as f:
    wfo = json.load(f)
for sym, params in wfo.items():
    if sym in main:
        main[sym].update(params)
    else:
        main[sym] = params
with open('$CONFIG_JSON', 'w') as f:
    json.dump(main, f, indent=2, sort_keys=True)
    f.write('\n')
print(f'Merged {len(wfo)} symbols from WFO into $CONFIG_JSON')
" 2>&1 | tee -a "$LOG_FILE"
    fi

    # Decay check
    log "Running decay check..."
    "$VENV" -c "
from config.loader import cfg, load_config
from audit.audit_logger import BlackoutLogger
from engine.decay_tracker import ModelDecayTracker
load_config(); cfg.load(); cfg.DISABLE_ZSCORE = False
logger = BlackoutLogger()
tracker = ModelDecayTracker(logger)
tracker.load_baselines_from_config()
disabled = tracker.audit_all()
print(f'Decay check: {len(disabled)} symbols disabled')
" 2>&1 | tee -a "$LOG_FILE" || log "WARNING: Decay check failed"
fi

# ── Step 7: Restart bot ──
if [ "$START_STEP" -le 7 ]; then
    log "STEP 7: Restarting $SERVICE..."
    run_or_dry systemctl --user restart "$SERVICE"
    sleep 5
    if systemctl --user is-active "$SERVICE" &>/dev/null; then
        log "Bot restarted successfully"
    else
        log "ERROR: Bot failed to start — attempting restart"
        run_or_dry systemctl --user restart "$SERVICE"
        sleep 5
        systemctl --user is-active "$SERVICE" &>/dev/null && \
            log "Bot restarted OK" || log "ERROR: Bot still not running!"
    fi
fi

# ── Maintenance ──
log "Maintenance..."
sqlite3 "${ACCT_PREFIX}/audit/sovereign_log.db" "VACUUM;" 2>/dev/null && log "SQLite VACUUM done" || true
find "${ACCT_PREFIX}/models" -name "*.json" -path "*/versions/*" -mtime +28 -delete 2>/dev/null && log "Old model versions pruned (>28d)" || true
# Prune old ritual dirs (keep last 7)
ls -dt "$OPTUNA_DIR"/ritual_H4_* 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null || true
ls -dt "$OPTUNA_DIR"/ritual_H1_* 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null || true
ls -dt "$OPTUNA_DIR"/ritual_M30_* 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null || true
ls -dt "$OPTUNA_DIR"/ritual_M15_* 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null || true
log "Old ritual dirs pruned (keep 7)"

TOTAL_HOURS=$(elapsed_hours)
log "========================================="
log "RITUAL COMPLETE — account=$ACCOUNT — $(date)"
log "Total runtime: ${TOTAL_HOURS}h"
log "========================================="
