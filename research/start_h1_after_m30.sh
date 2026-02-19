#!/usr/bin/env bash
# Wait for M30 to finish, then start H1 on GTX 1050
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

M30_DB="models/optuna_results/optuna_realcosts_M30_20260211_192247/optuna_studies.db"

echo "[$(date)] Waiting for M30 to finish (144 symbols)..."

while true; do
    count=$(python3 -c "
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
studies = optuna.study.get_all_study_summaries('sqlite:///${M30_DB}')
done = sum(1 for s in studies if s.n_trials >= 150)
print(done)
" 2>/dev/null)

    if [ "$count" -ge 144 ]; then
        echo "[$(date)] M30 done! All 144 symbols complete."
        break
    fi
    echo "[$(date)] M30: ${count}/144 done, waiting..."
    sleep 120
done

# Start H1 on GTX 1050
echo "[$(date)] Starting H1 on GTX 1050..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_H1="models/optuna_results/optuna_realcosts_H1_${TIMESTAMP}"

CUDA_VISIBLE_DEVICES=1 \
MT5_BRIDGE_PORT=5056 \
OMP_NUM_THREADS=4 \
python3 research/optuna_orchestrator.py \
    --all \
    --workers 6 \
    --trial-jobs 2 \
    --train-size 2400 \
    --test-size 600 \
    --purge 24 \
    --embargo 24 \
    --pt-mult 2.0 \
    --sl-mult 1.5 \
    --horizon-bars 24 \
    --z-threshold 1.0 \
    --timeframe H1 \
    --trials 600 \
    --discovery-cutoff 150 \
    --out-dir "$OUT_H1" \
    2>&1 | tee "audit/logs/optuna_H1_${TIMESTAMP}.log"

echo "[$(date)] H1 done! Results: ${OUT_H1}"
