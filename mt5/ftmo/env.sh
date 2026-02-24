#!/bin/bash
# FTMO account environment — source this before starting services.
#
# Usage: source env.sh

export ACCOUNT_NAME="ftmo"
export WINEPREFIX="/home/tradebot/.wine64"
export DISPLAY=":1"
export VNC_PORT="5901"
export MT5_BRIDGE_HOST="127.0.0.1"
export MT5_BRIDGE_PORT="5056"
export XDG_RUNTIME_DIR="/tmp/xdg-$(id -u)"
export WINEDEBUG="-all"
export TMPDIR="/tmp"
