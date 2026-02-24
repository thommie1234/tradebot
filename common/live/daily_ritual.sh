#!/bin/bash
# Daily Ritual — Daily reoptimization pipeline (all 4 timeframes, no kill switch)
# Runs every day at 00:00 CET via systemd timer
#
# Pipeline:
#   0. (Saturday only) Download fresh tick data from MT5
#   1. Multi-TF Optuna: H4+M30+M15 on P40, H1 on GTX 1050 (parallel), --no-kill
#   2. Merge results, select best TF per symbol
#   3. Retrain models with new params
#   4. Exit-parameter optimization
#   5. Threshold WFO
#   6. Decay check
#   7. Restart bot
#   8. Maintenance
#
# Budget: 20 hours (00:00 → 20:00), 4 hours overflow (→ 24:00)
# All symbols run full WFO even with negative EV (--no-kill)
#
# Usage: bash daily_ritual.sh
#        Or via systemd: systemctl --user start daily-ritual.service

set -euo pipefail
cd /home/tradebot/tradebots

VENV=".venv/bin/python3"
LOG_DIR="audit/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ritual_${TIMESTAMP}.log"
START_EPOCH=$(date +%s)

# 20-hour budget + 4-hour overflow
BUDGET_SECS=$((20 * 3600))
OVERFLOW_SECS=$((24 * 3600))

# Optuna shared settings
TRIALS=600
DISCOVERY=500
TRAIN_SIZE=2400
TEST_SIZE=600
PURGE=24
EMBARGO=24

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] RITUAL: $1" | tee -a "$LOG_FILE"; }

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

log "========================================="
log "DAILY RITUAL START — $(date)"
log "Budget: 20h | Overflow: 24h"
log "========================================="

# ── Step 0: Tick data download (Saturday only) ──
DOW=$(date +%u)  # 1=Mon, 7=Sun
if [ "$DOW" -eq 6 ]; then
  log "STEP 0: Saturday — downloading fresh tick data..."
  log "Stopping mt5-bridge-proxy and sovereign-bot for download..."
  systemctl --user stop sovereign-bot.service 2>/dev/null || true
  systemctl --user stop mt5-bridge-proxy.service 2>/dev/null || true
  sleep 5

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
  log "Tick data download complete"

  # Restart MT5 bridge
  log "Restarting mt5-bridge-proxy..."
  systemctl --user start mt5-bridge-proxy.service
  sleep 10

  # Restart bot (was stopped for download)
  log "Restarting sovereign-bot..."
  systemctl --user start sovereign-bot.service
  sleep 5
else
  log "STEP 0: Weekday — skipping tick data download (using existing data)"
fi

# ── Step 1: Multi-TF Optuna (all 4 timeframes, --no-kill) ──
log "STEP 1: Multi-TF Optuna optimization (ALL timeframes, --no-kill)..."
log "  P40 (CUDA:0): H4 → M30 → M15 (sequential, 4 workers)"
log "  GTX (CUDA:1): H1 (2 workers, parallel)"

H4_DIR="models/optuna_results/ritual_H4_${TIMESTAMP}"
M30_DIR="models/optuna_results/ritual_M30_${TIMESTAMP}"
M15_DIR="models/optuna_results/ritual_M15_${TIMESTAMP}"
H1_DIR="models/optuna_results/ritual_H1_${TIMESTAMP}"

OPTUNA_COMMON="--all --trials $TRIALS --discovery-cutoff $DISCOVERY --no-kill \
  --train-size $TRAIN_SIZE --test-size $TEST_SIZE \
  --purge $PURGE --embargo $EMBARGO \
  --pt-mult 2.0 --sl-mult 1.5 --horizon-bars 24 --z-threshold 1.0"

# P40 pipeline: H4 (production TF, highest priority) → M30 → M15
(
  log "Starting H4 on P40... ($(elapsed_hours)h elapsed)"
  CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
    $OPTUNA_COMMON --workers 4 --trial-jobs 2 \
    --timeframe H4 \
    --out-dir "$H4_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_H4_${TIMESTAMP}.log"
  log "H4 complete. ($(elapsed_hours)h elapsed)"

  if past_budget; then
    log "BUDGET REACHED after H4 — skipping M30+M15 on P40"
  else
    log "Starting M30 on P40... ($(elapsed_hours)h elapsed)"
    CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
      $OPTUNA_COMMON --workers 4 --trial-jobs 2 \
      --timeframe M30 \
      --out-dir "$M30_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_M30_${TIMESTAMP}.log"
    log "M30 complete. ($(elapsed_hours)h elapsed)"

    if past_budget; then
      log "BUDGET REACHED after M30 — skipping M15"
    else
      log "Starting M15 on P40... ($(elapsed_hours)h elapsed)"
      CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/research/optuna_orchestrator.py" \
        $OPTUNA_COMMON --workers 4 --trial-jobs 2 \
        --timeframe M15 \
        --out-dir "$M15_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_M15_${TIMESTAMP}.log"
      log "M15 complete. ($(elapsed_hours)h elapsed)"
    fi
  fi
) &
P40_PID=$!

# GTX 1050 pipeline: H1 (smaller GPU, 2 workers)
(
  log "Starting H1 on GTX 1050... ($(elapsed_hours)h elapsed)"
  CUDA_VISIBLE_DEVICES=1 "$VENV" -u "common/research/optuna_orchestrator.py" \
    $OPTUNA_COMMON --workers 2 --trial-jobs 1 \
    --timeframe H1 \
    --out-dir "$H1_DIR" 2>&1 | tee -a "$LOG_DIR/ritual_H1_${TIMESTAMP}.log"
  log "H1 complete. ($(elapsed_hours)h elapsed)"
) &
GTX_PID=$!

# Wait for both GPU pipelines, respect overflow deadline
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

# ── Step 2: Merge results — best TF per symbol ──
log "STEP 2: Merging multi-TF results..."
"$VENV" -c "
import polars as pl, os, json, yaml

results = {}  # symbol → {tf, ev, csv_path}

for tf, d in [('H4', '$H4_DIR'), ('M30', '$M30_DIR'), ('M15', '$M15_DIR'), ('H1', '$H1_DIR')]:
    csv = os.path.join(d, 'summary.csv')
    if not os.path.exists(csv):
        print(f'  {tf}: no summary.csv found')
        continue
    df = pl.read_csv(csv)
    # With --no-kill, all symbols have status='ok' — include all
    ok = df.filter(pl.col('status') == 'ok')
    total = len(df)
    print(f'  {tf}: {len(ok)} completed out of {total}')
    for row in ok.iter_rows(named=True):
        sym = row['symbol']
        ev = float(row['best_ev'])
        if sym not in results or ev > results[sym]['ev']:
            results[sym] = {'tf': tf, 'ev': ev, 'csv': csv}

# Write combined CSV (union of best TF per symbol)
print(f'\nBest TF per symbol: {len(results)} symbols')
for sym in sorted(results):
    info = results[sym]
    marker = ' *' if info['ev'] < 0 else ''
    print(f'  {sym}: {info[\"tf\"]} (EV={info[\"ev\"]:.6f}){marker}')

combined_rows = []
for tf, d in [('H4', '$H4_DIR'), ('M30', '$M30_DIR'), ('M15', '$M15_DIR'), ('H1', '$H1_DIR')]:
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
    print(f'\nSaved combined CSV: {out} ({len(combined)} rows)')

    # Update multi_tf.yaml for non-H1 winners
    mtf = {'symbols': {}}
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    for sym, info in sorted(results.items()):
        if info['tf'] != 'H1':
            mtf['symbols'][sym] = {
                'timeframe': info['tf'],
                'optuna_ev': round(info['ev'], 6),
                'source': f'daily_ritual_{today} ({info[\"tf\"]})',
            }
    if mtf['symbols']:
        with open('config/multi_tf.yaml', 'w') as f:
            f.write('# Multi-timeframe symbol configuration\n')
            f.write(f'# Auto-generated by daily ritual {today}\n')
            f.write('# Best TF per symbol from Optuna real-cost optimization\n\n')
            yaml.dump(mtf, f, default_flow_style=False)
        print(f'Updated multi_tf.yaml: {len(mtf[\"symbols\"])} non-H1 symbols')
else:
    print('WARNING: No completed results from any timeframe')
" 2>&1 | tee -a "$LOG_FILE"

# Count total results
COMBINED_CSV="models/optuna_results/ritual_combined_${TIMESTAMP}.csv"
RESULTS_COUNT=0
if [ -f "$COMBINED_CSV" ]; then
  RESULTS_COUNT=$("$VENV" -c "
import polars as pl
df = pl.read_csv('$COMBINED_CSV')
print(len(df))
" 2>/dev/null || echo "0")
fi
log "Total symbols with results across all timeframes: $RESULTS_COUNT"

if [ "$RESULTS_COUNT" -eq 0 ]; then
  log "WARNING: No results — keeping existing models"
else
  # ── Step 3: Retrain models ──
  log "STEP 3: Retraining models with new Optuna params..."
  CUDA_VISIBLE_DEVICES=0 OPTUNA_CSV_OVERRIDE="$(realpath "$COMBINED_CSV")" \
    "$VENV" -u "common/live/run_bot.py" --retrain 2>&1 | tail -30 | \
    tee -a "$LOG_FILE"

  # Update paths.yaml to point to new combined CSV
  sed -i "s|^optuna_csv:.*|optuna_csv: ${COMBINED_CSV}|" config/paths.yaml
  log "Retrain complete. Updated paths.yaml → $COMBINED_CSV"

  # ── Step 4: Exit-parameter optimization ──
  log "STEP 4: Exit-parameter Optuna (400 trials)..."
  "$VENV" -u "common/research/exit_optuna.py" \
    --active --trials 400 --workers 12 --update-configs \
    --out-dir "models/optuna_results/exit_${TIMESTAMP}" \
    2>&1 | tee -a "$LOG_DIR/ritual_exit_${TIMESTAMP}.log"
  log "Exit-parameter optimization complete"

  # ── Step 4.5: Threshold WFO ──
  log "STEP 4.5: Threshold WFO optimization..."
  PYTHONPATH="/home/tradebot/tradebots/archive:$PYTHONPATH" \
    "$VENV" -u "common/research/threshold_wfo.py" --active 2>&1 | \
    tee -a "$LOG_DIR/ritual_threshold_${TIMESTAMP}.log" || \
    log "WARNING: Threshold WFO failed"
  log "Threshold WFO complete"
fi

# ── Step 5: Decay check ──
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

# ── Step 6b: Monte Carlo simulation ──
log "STEP 6b: Running Monte Carlo simulation..."
CUDA_VISIBLE_DEVICES=0 "$VENV" -u "common/analysis/monte_carlo.py" \
  --trades-from-db --sims 50000 2>&1 | \
  tee -a "$LOG_FILE" || log "WARNING: Monte Carlo failed"
log "Monte Carlo complete"

# ── Step 7: Restart bot (pick up new models) ──
log "STEP 7: Restarting sovereign bot with new models..."
systemctl --user restart sovereign-bot.service
sleep 5
if systemctl --user is-active sovereign-bot.service &>/dev/null; then
  log "Bot restarted successfully"
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
# Prune old ritual dirs (keep last 7 days)
ls -dt models/optuna_results/ritual_*/ 2>/dev/null | tail -n +8 | xargs rm -rf 2>/dev/null && log "Old ritual dirs pruned (keep 7)" || true
if timedatectl show --property=NTPSynchronized 2>/dev/null | grep -q "yes"; then
  log "NTP: synchronized"
else
  log "WARNING: NTP not synchronized!"
fi

TOTAL_HOURS=$(elapsed_hours)
log "========================================="
log "DAILY RITUAL COMPLETE — $(date)"
log "Total runtime: ${TOTAL_HOURS}h"
log "Symbols processed: $RESULTS_COUNT"
log "========================================="
