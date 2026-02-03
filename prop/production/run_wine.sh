#!/bin/bash
# Wine Python wrapper for trading scripts
# Usage: ./run_wine.sh strategy_bot.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINE_PYTHON="$HOME/.wine/drive_c/Python312/python.exe"

export WINEDEBUG=-all

if [ -z "$1" ]; then
    echo "Usage: $0 <script.py> [arguments...]"
    exit 1
fi

cd "$SCRIPT_DIR"
wine "$WINE_PYTHON" "Z:${SCRIPT_DIR}/launcher.py" "$@"
