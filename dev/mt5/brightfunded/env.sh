#!/bin/bash
# BrightFunded account environment — source this before starting services.
#
# Usage: source env.sh

export ACCOUNT_NAME="brightfunded"
export WINEPREFIX="/home/tradebot/tradebots/mt5/brightfunded/wineprefix"
export DISPLAY=":2"
export VNC_PORT="5902"
export MT5_BRIDGE_HOST="127.0.0.1"
export MT5_BRIDGE_PORT="5057"
export XDG_RUNTIME_DIR="/tmp/xdg-$(id -u)"
export WINEDEBUG="-all"
export TMPDIR="/tmp"
