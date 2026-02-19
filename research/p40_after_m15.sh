#!/usr/bin/env bash
# After M15 finishes on P40: kill the parent multi-tf script (prevent
# duplicate M30/H1) and start H1 on P40 if the GTX 1050 hasn't finished it yet.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

M15_DB="models/optuna_results/optuna_realcosts_M15_20260211_155211/optuna_studies.db"
PARENT_PID=308805  # run_optuna_multi_tf.sh

echo "[$(date)] Watching M15 on P40 (144 symbols)..."

while true; do
    count=$(python3 -c "
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
studies = optuna.study.get_all_study_summaries('sqlite:///${M15_DB}')
done = sum(1 for s in studies if s.n_trials >= 200)
print(done)
" 2>/dev/null)

    if [ "$count" -ge 144 ]; then
        echo "[$(date)] M15 done! All 144 symbols complete."
        break
    fi
    echo "[$(date)] M15: ${count}/144 done, waiting..."
    sleep 120
done

# Kill the parent script so it doesn't start duplicate M30/H1
echo "[$(date)] Killing parent multi-tf script (PID ${PARENT_PID})..."
kill ${PARENT_PID} 2>/dev/null || true
sleep 5

# Check if H1 is still running on the GTX 1050
H1_RUNNING=$(pgrep -f "timeframe H1.*optuna_orchestrator" || echo "")
if [ -n "$H1_RUNNING" ]; then
    echo "[$(date)] H1 is still running on GTX 1050 (PID: ${H1_RUNNING})."
    echo "[$(date)] Starting H1 helper on P40 to assist..."
    # H1 on 1050 is already handling all 144 symbols via the same orchestrator.
    # We can't easily split — just let the 1050 finish.
    echo "[$(date)] P40 is now free. Nothing more to do."
else
    # H1 hasn't started yet or already finished — check
    H1_WATCHER=$(pgrep -f "start_h1_after_m30" || echo "")
    if [ -n "$H1_WATCHER" ]; then
        echo "[$(date)] H1 watcher is waiting for M30. H1 will run on GTX 1050."
        echo "[$(date)] P40 is now free."
    else
        # H1 already done or never started — run it on P40
        echo "[$(date)] No H1 running. Starting H1 on P40..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUT_H1="models/optuna_results/optuna_realcosts_H1_P40_${TIMESTAMP}"

        CUDA_VISIBLE_DEVICES=0 \
        MT5_BRIDGE_PORT=5056 \
        OMP_NUM_THREADS=4 \
        python3 research/optuna_orchestrator.py \
            --all \
            --workers 4 \
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
            2>&1 | tee "audit/logs/optuna_H1_P40_${TIMESTAMP}.log"
        echo "[$(date)] H1 on P40 done!"
    fi
fi

echo "[$(date)] P40 watcher finished."
