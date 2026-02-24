#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine64}"
export DISPLAY="${DISPLAY:-:1}"
export MT5_BRIDGE_HOST="${MT5_BRIDGE_HOST:-127.0.0.1}"
export MT5_BRIDGE_PORT="${MT5_BRIDGE_PORT:-5055}"
export ENABLE_LIVE_TRADING="${ENABLE_LIVE_TRADING:-1}"

# Start Wine proxy (background)
"$SCRIPT_DIR/run_wine.sh" mt5_bridge_proxy.py &
PROXY_PID=$!

# Small wait to let the proxy bind
sleep 2

# Start Sovereign bot (foreground)
cd "$REPO_ROOT"
exec .venv/bin/python3 live/run_bot.py --live
