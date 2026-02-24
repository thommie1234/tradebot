#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WINE="$SCRIPT_DIR/run_wine.sh"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SYMBOLS=(BTCUSD ETHUSD SOLUSD MSFT NVDA AAPL TSLA AMZN GOOGL META)
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[backtest-all-cores] max jobs: $MAX_JOBS"
echo "[backtest-all-cores] symbols: ${SYMBOLS[*]}"

running=0
for sym in "${SYMBOLS[@]}"; do
    log="$LOG_DIR/backtest_${sym}_${TS}.log"
    echo "[backtest-all-cores] start $sym -> $log"
    nohup "$RUN_WINE" backtest_engine.py "$sym" > "$log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait
echo "[backtest-all-cores] done. Logs in $LOG_DIR"
