#!/bin/bash
# Wine Python wrapper — shared across all accounts.
#
# Required env vars (set by account-specific env.sh):
#   WINEPREFIX   — Wine prefix for this account
#   DISPLAY      — VNC display for this account
#   MT5_BRIDGE_PORT — Bridge proxy port
#
# Usage: source ../ftmo/env.sh && ./run_wine.sh mt5_bridge_proxy.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (overridden by env.sh per account)
WINEPREFIX="${WINEPREFIX:-$HOME/.wine64}"
DISPLAY="${DISPLAY:-:1}"

export WINEDEBUG=-all
export WINEPREFIX
export DISPLAY
export ENABLE_LIVE_TRADING="${ENABLE_LIVE_TRADING:-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

# Find Wine Python in the prefix
WINE_PYTHON="$WINEPREFIX/drive_c/Python311/python.exe"

if [ -z "$1" ]; then
    echo "Usage: $0 <script.py> [arguments...]"
    exit 1
fi

if [ ! -f "$WINE_PYTHON" ]; then
    echo "Wine Python not found: $WINE_PYTHON"
    echo "Make sure WINEPREFIX is set and MT5 + Python 3.11 are installed."
    exit 1
fi

cd "$SCRIPT_DIR"
wine "$WINE_PYTHON" "Z:${SCRIPT_DIR}/launcher.py" "$@"
