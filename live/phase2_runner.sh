#!/bin/bash
set -euo pipefail

# Phase-2 flow:
# 1) Backtest first
# 2) Only then start live strategy bot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WINE="$SCRIPT_DIR/run_wine.sh"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

usage() {
    echo "Usage:"
    echo "  $0 backtest"
    echo "  $0 live <SYMBOL...>"
    echo "  $0 phase2 <SYMBOL...>   # runs backtest first, then live"
}

run_backtest() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/backtest_${ts}.log"
    echo "[phase2] Running backtest first..."
    "$RUN_WINE" backtest_engine.py | tee "$log_file"
    echo "[phase2] Backtest log: $log_file"
}

run_live() {
    if [ "$#" -lt 1 ]; then
        echo "[phase2] Provide at least one symbol for live mode."
        usage
        exit 1
    fi
    if [ "${ENABLE_LIVE_TRADING:-0}" != "1" ]; then
        echo "[phase2] Live trading is disabled. Export ENABLE_LIVE_TRADING=1 to allow live orders."
        exit 1
    fi
    echo "[phase2] Starting live bot for symbols: $*"
    exec "$RUN_WINE" strategy_bot.py "$@"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

cmd="$1"
shift

case "$cmd" in
    backtest)
        run_backtest
        ;;
    live)
        run_live "$@"
        ;;
    phase2)
        run_backtest
        run_live "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac
