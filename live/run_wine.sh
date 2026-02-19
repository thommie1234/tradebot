#!/bin/bash
# Wine Python wrapper for trading scripts
# Usage: ./run_wine.sh strategy_bot.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINEPREFIX="${WINEPREFIX:-$HOME/.wine64}"

# Prefer MT5 prefix Python311, keep a couple of fallbacks for portability.
WINE_PYTHON_CANDIDATES=(
    "$WINEPREFIX/drive_c/Python311/python.exe"
)

export WINEDEBUG=-all
export WINEPREFIX
export DISPLAY="${DISPLAY:-:1}"
export ENABLE_LIVE_TRADING="${ENABLE_LIVE_TRADING:-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

WINE_PYTHON=""
for candidate in "${WINE_PYTHON_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        WINE_PYTHON="$candidate"
        break
    fi
done

if [ -z "$1" ]; then
    echo "Usage: $0 <script.py> [arguments...]"
    exit 1
fi

if [ -z "$WINE_PYTHON" ]; then
    echo "Wine Python not found in expected locations:"
    for candidate in "${WINE_PYTHON_CANDIDATES[@]}"; do
        echo "  - $candidate"
    done
    exit 1
fi

cd "$SCRIPT_DIR"
wine "$WINE_PYTHON" "Z:${SCRIPT_DIR}/launcher.py" "$@"
