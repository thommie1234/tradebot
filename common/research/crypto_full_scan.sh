#!/bin/bash
# Full crypto Optuna scan — all symbols × all timeframes
# P40 (GPU 0): H4 + M30 (heavier, sequential)
# GTX 1050 (GPU 1): H1 + M15 (lighter, sequential)
#
# 25-month cryptos: all 4 TFs
# 11-month cryptos: M15/M30/H1 only (H4 insufficient data)

set -euo pipefail
cd /home/tradebot/tradebots
VENV=".venv/bin/python3"
PYTHONPATH="/home/tradebot/tradebots/common"
export PYTHONPATH

TRIALS=200
DISCOVERY=80
TRAIN_SIZE=2400
TEST_SIZE=600
PURGE=24
EMBARGO=24

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="audit/logs"
mkdir -p "$LOG_DIR"

# 25-month cryptos (H4 viable)
CRYPTO_25M="BTCUSD,DASHUSD,ETHUSD,LTCUSD,XRPUSD,XMRUSD,NEOUSD,ADAUSD,DOTUSD,DOGEUSD,BCHUSD"

# 11-month cryptos (no H4)
CRYPTO_11M="SOLUSD,AVAUSD,ETCUSD,BNBUSD,SANUSD,LNKUSD,NERUSD,ALGUSD,ICPUSD,AAVUSD,BARUSD,GALUSD,GRTUSD,IMXUSD,MANUSD,VECUSD,XLMUSD,UNIUSD,FETUSD,XTZUSD"

# All cryptos (for M15/M30/H1)
ALL_CRYPTO="$CRYPTO_25M,$CRYPTO_11M"

COMMON="--trials $TRIALS --discovery-cutoff $DISCOVERY --no-kill \
  --train-size $TRAIN_SIZE --test-size $TEST_SIZE \
  --purge $PURGE --embargo $EMBARGO"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ═══════════════════════════════════════════════════════════════
# GPU 0 (P40): H4 → M30 → M15 → H1 (sequential, 4 workers)
# ═══════════════════════════════════════════════════════════════
run_p40() {
  export CUDA_VISIBLE_DEVICES=0

  # H4: only 25-month cryptos
  log "P40: Starting H4 (11 symbols)"
  $VENV common/research/optuna_orchestrator.py \
    --symbols "$CRYPTO_25M" --timeframe H4 $COMMON \
    --workers 4 --trial-jobs 2 \
    --out-dir "models/optuna_results/crypto_scan_H4_${TIMESTAMP}" \
    2>&1 | tee "$LOG_DIR/crypto_scan_H4_${TIMESTAMP}.log"

  # M30: all cryptos
  log "P40: Starting M30 (31 symbols)"
  $VENV common/research/optuna_orchestrator.py \
    --symbols "$ALL_CRYPTO" --timeframe M30 $COMMON \
    --workers 4 --trial-jobs 2 \
    --out-dir "models/optuna_results/crypto_scan_M30_${TIMESTAMP}" \
    2>&1 | tee "$LOG_DIR/crypto_scan_M30_${TIMESTAMP}.log"

  log "P40: DONE"
}

# ═══════════════════════════════════════════════════════════════
# GPU 1 (GTX 1050): H1 → M15 (sequential, 2 workers)
# ═══════════════════════════════════════════════════════════════
run_gtx() {
  export CUDA_VISIBLE_DEVICES=1

  # H1: all cryptos
  log "GTX: Starting H1 (31 symbols)"
  $VENV common/research/optuna_orchestrator.py \
    --symbols "$ALL_CRYPTO" --timeframe H1 $COMMON \
    --workers 2 --trial-jobs 1 \
    --out-dir "models/optuna_results/crypto_scan_H1_${TIMESTAMP}" \
    2>&1 | tee "$LOG_DIR/crypto_scan_H1_${TIMESTAMP}.log"

  # M15: all cryptos
  log "GTX: Starting M15 (31 symbols)"
  $VENV common/research/optuna_orchestrator.py \
    --symbols "$ALL_CRYPTO" --timeframe M15 $COMMON \
    --workers 2 --trial-jobs 1 \
    --out-dir "models/optuna_results/crypto_scan_M15_${TIMESTAMP}" \
    2>&1 | tee "$LOG_DIR/crypto_scan_M15_${TIMESTAMP}.log"

  log "GTX: DONE"
}

# Launch both GPU pipelines in parallel
log "Starting full crypto scan: $TRIALS trials, $DISCOVERY discovery cutoff"
log "104 symbol/TF combos across 2 GPUs"

run_p40 &
PID_P40=$!

run_gtx &
PID_GTX=$!

log "P40 PID=$PID_P40 | GTX PID=$PID_GTX"
log "Waiting for both to finish..."

wait $PID_P40
EXIT_P40=$?
log "P40 finished (exit=$EXIT_P40)"

wait $PID_GTX
EXIT_GTX=$?
log "GTX finished (exit=$EXIT_GTX)"

log "═══════════════════════════════════════"
log "FULL CRYPTO SCAN COMPLETE"
log "═══════════════════════════════════════"

# Combine results
log "Combining CSV results..."
$VENV -c "
import polars as pl, glob, os

csvs = sorted(glob.glob('models/optuna_results/crypto_scan_*_${TIMESTAMP}/summary.csv'))
if not csvs:
    print('No summary CSVs found')
    exit(0)

frames = []
for c in csvs:
    tf = os.path.basename(os.path.dirname(c)).split('_')[2]  # H4, M30, H1, M15
    df = pl.read_csv(c)
    if 'timeframe' not in df.columns:
        df = df.with_columns(pl.lit(tf).alias('timeframe'))
    frames.append(df)
    print(f'  {c}: {len(df)} symbols')

combined = pl.concat(frames)
out = f'models/optuna_results/crypto_scan_combined_{TIMESTAMP}.csv'
combined.write_csv(out)
print(f'Combined: {len(combined)} rows -> {out}')

# Show best per symbol (highest EV)
if 'best_ev' in combined.columns:
    best = combined.sort('best_ev', descending=True)
    print()
    print('=== Top 20 by EV ===')
    for row in best.head(20).iter_rows(named=True):
        sym = row.get('symbol','?')
        tf = row.get('timeframe','?')
        ev = row.get('best_ev',0)
        spread = row.get('spread_bps',0)
        status = row.get('status','?')
        print(f'  {sym:12s} {tf:4s}  EV={ev:+.4f}  spread={spread:.1f}bps  [{status}]')
"

log "All done."
