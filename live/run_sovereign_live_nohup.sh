#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine64}"
export DISPLAY="${DISPLAY:-:1}"
export MT5_BRIDGE_HOST="${MT5_BRIDGE_HOST:-127.0.0.1}"
export MT5_BRIDGE_PORT="${MT5_BRIDGE_PORT:-5055}"
export ENABLE_LIVE_TRADING="${ENABLE_LIVE_TRADING:-1}"

PROXY_LOG="$LOG_DIR/mt5_bridge_proxy.nohup.log"
BOT_LOG="$LOG_DIR/sovereign_live.nohup.log"
PROXY_PID_FILE="$LOG_DIR/mt5_bridge_proxy.pid"
BOT_PID_FILE="$LOG_DIR/sovereign_live.pid"

# Start MT5 proxy (Wine) in background via nohup
nohup "$SCRIPT_DIR/run_wine.sh" mt5_bridge_proxy.py > "$PROXY_LOG" 2>&1 &
echo $! > "$PROXY_PID_FILE"

# Give proxy a moment to bind
sleep 2

# Start Sovereign bot (Linux) in background via nohup
cd "$REPO_ROOT"
nohup .venv/bin/python3 live/run_bot.py --live > "$BOT_LOG" 2>&1 &
echo $! > "$BOT_PID_FILE"

echo "Started MT5 proxy (pid $(cat "$PROXY_PID_FILE")) -> $PROXY_LOG"
echo "Started Sovereign bot (pid $(cat "$BOT_PID_FILE")) -> $BOT_LOG"
