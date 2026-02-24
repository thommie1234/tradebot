#!/usr/bin/env bash
# Daily data sync — downloads tick data + M1 bars for all symbols
# Runs via systemd timer: daily-data-sync.timer at 02:00 CET
#
# IMPORTANT: Temporarily stops mt5-bridge-proxy because Wine Python
# can only have one MT5 connection at a time.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_ROOT/audit/logs/daily_data_sync.log"
WINE_LAUNCHER="$REPO_ROOT/live/run_wine.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Daily data sync started ==="

# Stop services that hold the MT5 connection
log "Stopping mt5-bridge-proxy and sovereign-bot..."
systemctl --user stop sovereign-bot.service 2>/dev/null || true
systemctl --user stop mt5-bridge-proxy.service 2>/dev/null || true
sleep 5

# Step 1: Download tick data (for ML training)
log "--- Step 1: Tick data download ---"
if "$WINE_LAUNCHER" download_ticks.py >> "$LOG" 2>&1; then
    log "Tick data download completed OK"
else
    log "WARNING: Tick data download exited with errors"
fi

# Step 2: Download M1 bars (for backtesting/research)
log "--- Step 2: M1 bars download ---"
if "$WINE_LAUNCHER" data_downloader.py >> "$LOG" 2>&1; then
    log "M1 bars download completed OK"
else
    log "WARNING: M1 bars download exited with errors"
fi

# Restart services
log "Restarting mt5-bridge-proxy and sovereign-bot..."
systemctl --user start mt5-bridge-proxy.service
sleep 10
systemctl --user start sovereign-bot.service

log "=== Daily data sync finished ==="
