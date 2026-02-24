#!/bin/bash
# Weekend Ritual — Weekly reoptimization pipeline
# Runs Saturday 00:00 CET via systemd timer, must finish by Sunday 12:00 CET
#
# Pipeline:
#   1. Download fresh tick data from MT5
#   2. Multi-TF Optuna: M15 + M30 on P40, H1 on GTX 1050 (parallel)
#   3. Merge results, select best TF per symbol
#   4. Retrain models with new params
#   5. Update multi_tf.yaml
#   6. LLM audit + Monte Carlo
#   7. Restart bot
#
# Deadline: Sunday 12:00 CET (36 hours budget)
#
# Usage: bash sunday_ritual.sh
#        Or via systemd: systemctl --user start sunday-ritual.service

set -euo pipefail
cd /home/tradebot/tradebots

VENV=".venv/bin/python3"
LOG_DIR="audit/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ritual_${TIMESTAMP}.log"
DEADLINE="Sunday 12:00"

# Optuna shared settings
TRIALS=600
DISCOVERY=200
TRAIN_SIZE=2400
TEST_SIZE=600
PURGE=24
EMBARGO=24

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] RITUAL: $1" | tee -a "$LOG_FILE"; }

past_deadline() {
  # Returns 0 (true) if we're past Sunday 12:00
  local dow=$(date +%u)  # 1=Mon, 7=Sun
  local hour=$(date +%H)
  if [ "$dow" -eq 7 ] && [ "$hour" -ge 12 ]; then
    return 0
  fi
  # If it's Monday or later, we're way past
  if [ "$dow" -ge 1 ] && [ "$dow" -le 5 ]; then
    return 0
  fi
  return 1
}

log "========================================="
log "WEEKEND RITUAL START — $(date)"
log "Deadline: $DEADLINE"
log "========================================="

# ── Step 1: Download fresh tick data ──
log "STEP 1: Downloading fresh tick data..."
# Stop services that hold the MT5 Wine connection
log "Stopping mt5-bridge-proxy and sovereign-bot for download..."
systemctl --user stop sovereign-bot.service 2>/dev/null || true
systemctl --user stop mt5-bridge-proxy.service 2>/dev/null || true
sleep 5

# Run parallel tick data download via Wine/MT5 (6 workers)
WINE_LAUNCHER="common/live/run_wine.sh"
WORKERS=6
SPLIT_DIR="common/tools/symbol_splits"
mkdir -p "$SPLIT_DIR"
"$VENV" "common/tools/make_symbol_splits.py" --workers "$WORKERS" --out-dir "$SPLIT_DIR" 2>&1 | tee -a "$LOG_FILE"

DOWNLOAD_PIDS=()
for f in "$SPLIT_DIR"/symbols_part_*.xlsx; do
    [ -f "$f" ] || continue
    part="$(basename "$f" .xlsx)"
    dl_log="$LOG_DIR/ritual_download_${part}_${TIMESTAMP}.log"
    log "Starting download: $part"
    SYMBOLS_FILE_PATH="$f" bash "$WINE_LAUNCHER" strategy_bot.py --download-data > "$dl_log" 2>&1 &
    DOWNLOAD_PIDS+=($!)
done

# Wait for all download workers (max 4 hours)
log "Waiting for ${#DOWNLOAD_PIDS[@]} download workers (max 4h)..."
DL_START=$(date +%s)
DL_TIMEOUT=14400  # 4 hours
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

log "Tick data download complete"

# Restart MT5 bridge (needed for Optuna MT5 calls)
log "Restarting mt5-bridge-proxy..."
systemctl --user start mt5-bridge-proxy.service
sleep 10

# ── Step 2: Multi-TF Optuna (parallel on 2 GPUs) ──
log "STEP 2: Multi-TF Optuna optimization..."
log "  P40:      M15 (4 workers) + M30 (4 workers, sequential)"
log "  GTX 1050: H1 (6 workers)"

M15_DIR="models/optuna_results/ritual_M15_${TIMESTAMP}"
M30_DIR="models/optuna_results/ritual_M30_${TIMESTAMP}"
H1_DIR="models/optuna_results/ritual_H1_${TIMESTAMP}"

# Start M15 on P40 (foreground on P40, M30 follows after)
(
  log "Starting M15 on P40..."
  CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
    --all --trials $TRIALS --discovery-cutoff $DISCOVERY \
    --train-size $TRAIN_SIZE --test-size $TEST_SIZE \
    --purge $PURGE --embargo $EMBARGO \
    --pt-mult 2.0 --sl-mult 1.5 --horizon-bars 24 --z-threshold 1.0 \
    --workers 4 --trial-jobs 2 \
    --timeframe M15 \
    --out-dir "$M15_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_M15_${TIMESTAMP}.log"

  log "M15 complete. Starting M30 on P40..."

  if past_deadline; then
    log "DEADLINE REACHED — skipping M30"
  else
    CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
      --all --trials $TRIALS --discovery-cutoff $DISCOVERY \
      --train-size $TRAIN_SIZE --test-size $TEST_SIZE \
      --purge $PURGE --embargo $EMBARGO \
      --pt-mult 2.0 --sl-mult 1.5 --horizon-bars 24 --z-threshold 1.0 \
      --workers 4 --trial-jobs 2 \
      --timeframe M30 \
      --out-dir "$M30_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_M30_${TIMESTAMP}.log"
    log "M30 complete."
  fi
) &
P40_PID=$!

# Start H1 on GTX 1050 (parallel)
(
  log "Starting H1 on GTX 1050..."
  CUDA_VISIBLE_DEVICES=1 "$VENV" -u "common/research/optuna_orchestrator.py" \
    --all --trials $TRIALS --discovery-cutoff $DISCOVERY \
    --train-size $TRAIN_SIZE --test-size $TEST_SIZE \
    --purge $PURGE --embargo $EMBARGO \
    --pt-mult 2.0 --sl-mult 1.5 --horizon-bars 24 --z-threshold 1.0 \
    --workers 6 --trial-jobs 2 \
    --timeframe H1 \
    --out-dir "$H1_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_H1_${TIMESTAMP}.log"
  log "H1 complete."
) &
GTX_PID=$!

# Wait for both GPU pipelines, but respect deadline
log "Waiting for Optuna runs (PIDs: P40=$P40_PID, GTX=$GTX_PID)..."
while kill -0 $P40_PID 2>/dev/null || kill -0 $GTX_PID 2>/dev/null; do
  if past_deadline; then
    log "DEADLINE REACHED — killing remaining Optuna runs"
    kill $P40_PID 2>/dev/null || true
    kill $GTX_PID 2>/dev/null || true
    sleep 5
    # Kill any leftover workers
    pkill -f "optuna_orchestrator.*ritual_" 2>/dev/null || true
    break
  fi
  sleep 60
done

wait $P40_PID 2>/dev/null || true
wait $GTX_PID 2>/dev/null || true
log "All Optuna runs finished (or deadline reached)"

# ── Step 3: Merge results — best TF per symbol ──
log "STEP 3: Merging multi-TF results..."
"$VENV" -c "
import polars as pl, os, json

results = {}  # symbol → {tf, ev, csv_path}

for tf, d in [('M15', '$M15_DIR'), ('M30', '$M30_DIR'), ('H1', '$H1_DIR')]:
    csv = os.path.join(d, 'summary.csv')
    if not os.path.exists(csv):
        print(f'  {tf}: no summary.csv found')
        continue
    df = pl.read_csv(csv)
    ok = df.filter(pl.col('status') == 'ok')
    print(f'  {tf}: {len(ok)} survivors out of {len(df)}')
    for row in ok.iter_rows(named=True):
        sym = row['symbol']
        ev = float(row['best_ev'])
        if sym not in results or ev > results[sym]['ev']:
            results[sym] = {'tf': tf, 'ev': ev, 'csv': csv}

# Write combined CSV (union of best TF per symbol)
print(f'\nBest TF per symbol: {len(results)} total survivors')
combined_rows = []
for tf, d in [('M15', '$M15_DIR'), ('M30', '$M30_DIR'), ('H1', '$H1_DIR')]:
    csv = os.path.join(d, 'summary.csv')
    if not os.path.exists(csv):
        continue
    df = pl.read_csv(csv).filter(pl.col('status') == 'ok')
    for row in df.iter_rows(named=True):
        sym = row['symbol']
        if sym in results and results[sym]['tf'] == tf:
            combined_rows.append(row)

if combined_rows:
    combined = pl.DataFrame(combined_rows)
    out = 'models/optuna_results/ritual_combined_${TIMESTAMP}.csv'
    combined.write_csv(out)
    print(f'Saved combined CSV: {out} ({len(combined)} rows)')

    # Update multi_tf.yaml for non-H1 winners
    import yaml
    mtf = {'symbols': {}}
    for sym, info in sorted(results.items()):
        if info['tf'] != 'H1':
            mtf['symbols'][sym] = {
                'timeframe': info['tf'],
                'optuna_ev': round(info['ev'], 6),
                'source': f'ritual_{info[\"tf\"]}_{os.path.basename(d)}',
            }
    if mtf['symbols']:
        with open('config/multi_tf.yaml', 'w') as f:
            f.write('# Multi-timeframe symbol configuration\\n')
            f.write(f'# Auto-generated by weekend ritual {os.environ.get(\"TIMESTAMP\", \"\")}\\n')
            f.write('# Best TF per symbol from Optuna real-cost optimization\\n\\n')
            yaml.dump(mtf, f, default_flow_style=False)
        print(f'Updated multi_tf.yaml: {len(mtf[\"symbols\"])} non-H1 symbols')
else:
    print('WARNING: No survivors from any timeframe')
" 2>&1 | tee -a "$LOG_FILE"

# Count total survivors
COMBINED_CSV="models/optuna_results/ritual_combined_${TIMESTAMP}.csv"
SURVIVORS=0
if [ -f "$COMBINED_CSV" ]; then
  SURVIVORS=$("$VENV" -c "
import polars as pl
df = pl.read_csv('$COMBINED_CSV')
print(len(df))
" 2>/dev/null || echo "0")
fi
log "Total survivors across all timeframes: $SURVIVORS"

if [ "$SURVIVORS" -eq 0 ]; then
  log "WARNING: No survivors — keeping existing models"
else
  # ── Step 4: Retrain models ──
  log "STEP 4: Retraining models with new Optuna params..."
  CUDA_VISIBLE_DEVICES=0 OPTUNA_CSV_OVERRIDE="$(realpath "$COMBINED_CSV")" \
    "$VENV" -u "common/live/run_bot.py" --retrain 2>&1 | tail -30 | \
    tee -a "$LOG_FILE"

  # Update paths.yaml to point to new combined CSV
  sed -i "s|^optuna_csv:.*|optuna_csv: ${COMBINED_CSV}|" config/paths.yaml
  log "Retrain complete. Updated paths.yaml → $COMBINED_CSV"

  # ── Step 4.5: Exit-parameter optimization ──
  log "STEP 4.5: Exit-parameter Optuna (400 trials, ~3 min)..."
  "$VENV" -u "common/research/exit_optuna.py" \
    --active --trials 400 --workers 12 --update-configs \
    --out-dir "models/optuna_results/exit_${TIMESTAMP}" \
    2>&1 | tee -a "$LOG_DIR/ritual_exit_${TIMESTAMP}.log"
  log "Exit-parameter optimization complete"
fi

# ── Step 5: Decay check + auto-retrain (F14) ──
log "STEP 5: Running decay check..."
"$VENV" -c "
from config.loader import cfg, load_config
from audit.audit_logger import BlackoutLogger
from engine.decay_tracker import ModelDecayTracker
load_config(); cfg.load(); cfg.DISABLE_ZSCORE = False
logger = BlackoutLogger()
tracker = ModelDecayTracker(logger)
tracker.load_baselines_from_config()
disabled = tracker.audit_all()
print(f'Decay check: {len(disabled)} symbols disabled after retrain attempt')
" 2>&1 | tee -a "$LOG_FILE" || log "WARNING: Decay check failed"
log "Decay check complete"

# ── Step 6: LLM Audit ──
log "STEP 6: LLM audit..."
CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/tools/llm_trade_advisor.py" --once 2>&1 | \
  tee -a "$LOG_FILE" || log "WARNING: LLM audit failed"
log "LLM audit complete"

# ── Step 6b: Monte Carlo simulation (F17) ──
log "STEP 6b: Running Monte Carlo simulation..."
CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/analysis/monte_carlo.py" \
  --trades-from-db --sims 50000 2>&1 | \
  tee -a "$LOG_FILE" || log "WARNING: Monte Carlo failed"
log "Monte Carlo complete"

# ── Step 7: Start bot ──
log "STEP 7: Starting sovereign bot..."
systemctl --user start sovereign-bot.service
sleep 5
if systemctl --user is-active sovereign-bot.service &>/dev/null; then
  log "Bot started successfully"
else
  log "ERROR: Bot failed to start — attempting restart"
  systemctl --user restart sovereign-bot.service
  sleep 5
  systemctl --user is-active sovereign-bot.service &>/dev/null && \
    log "Bot restarted OK" || log "ERROR: Bot still not running!"
fi

# ── Step 8: Maintenance ──
log "STEP 8: Maintenance..."
sqlite3 "audit/sovereign_log.db" "VACUUM;" 2>/dev/null && log "SQLite VACUUM done" || log "WARNING: SQLite VACUUM failed"
find "models/sovereign_models/versions" -name "*.json" -mtime +28 -delete 2>/dev/null && log "Old model versions pruned (>28d)" || true
# Prune old ritual dirs (keep last 4)
ls -dt models/optuna_results/ritual_*/ 2>/dev/null | tail -n +5 | xargs rm -rf 2>/dev/null && log "Old ritual dirs pruned" || true
if timedatectl show --property=NTPSynchronized 2>/dev/null | grep -q "yes"; then
  log "NTP: synchronized"
else
  log "WARNING: NTP not synchronized!"
fi

log "========================================="
log "WEEKEND RITUAL COMPLETE — $(date)"
log "Survivors: $SURVIVORS"
log "========================================="
