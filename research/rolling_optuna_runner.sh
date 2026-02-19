#!/bin/bash
# Rolling Daily Optuna — bash wrapper for systemd
#
# Runs with low priority (nice/ionice) to not interfere with live trading.
# Lockfile prevents double-runs. Skips if sunday-ritual is active.
#
# Usage: bash research/rolling_optuna_runner.sh
#        Or via systemd: systemctl --user start rolling-optuna.service

set -euo pipefail
cd /home/tradebot/tradebots

VENV=".venv/bin/python3"
LOG_DIR="audit/logs"
TIMESTAMP=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/rolling_optuna_${TIMESTAMP}.log"
LOCKFILE="/tmp/rolling_optuna.lock"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ROLLING: $1" | tee -a "$LOG_FILE"; }

# Skip if sunday-ritual is active
if systemctl --user is-active --quiet sunday-ritual.service 2>/dev/null; then
    log "Sunday ritual is active — skipping"
    exit 0
fi

# Lockfile check (belt-and-suspenders, Python also checks)
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        log "Another instance running (PID $PID) — skipping"
        exit 0
    else
        log "Stale lockfile found — removing"
        rm -f "$LOCKFILE"
    fi
fi

log "Starting rolling Optuna pipeline"

# Run with low priority
nice -n 15 ionice -c 3 "$VENV" -u "research/rolling_optuna.py" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

log "Rolling Optuna finished (exit code: $EXIT_CODE)"

# Prune logs older than 14 days
find "$LOG_DIR" -name "rolling_optuna_*.log" -mtime +14 -delete 2>/dev/null || true

exit $EXIT_CODE
