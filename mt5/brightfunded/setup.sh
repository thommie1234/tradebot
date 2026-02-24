#!/bin/bash
# Setup script for BrightFunded account.
#
# Run this ONCE after buying the BrightFunded account.
# It creates the Wine prefix, installs MT5, and sets up systemd services.
#
# Usage: bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=========================================="
echo "  BrightFunded MT5 Setup"
echo "=========================================="
echo "  Wine prefix:  $WINEPREFIX"
echo "  VNC display:  $DISPLAY (port $VNC_PORT)"
echo "  Bridge port:  $MT5_BRIDGE_PORT"
echo "=========================================="

# ── Step 1: Create Wine prefix ──
echo ""
echo "[1/5] Creating Wine prefix..."
if [ -d "$WINEPREFIX" ]; then
    echo "  Wine prefix already exists, skipping."
else
    WINEDEBUG=-all WINEPREFIX="$WINEPREFIX" wineboot --init
    echo "  Wine prefix created."
fi

# ── Step 2: Copy Python 3.11 from FTMO prefix ──
echo ""
echo "[2/5] Copying Python 3.11 from FTMO prefix..."
FTMO_PYTHON="/home/tradebot/.wine64/drive_c/Python311"
BF_PYTHON="$WINEPREFIX/drive_c/Python311"
if [ -d "$BF_PYTHON" ]; then
    echo "  Python 3.11 already present, skipping."
else
    if [ -d "$FTMO_PYTHON" ]; then
        cp -r "$FTMO_PYTHON" "$BF_PYTHON"
        echo "  Python 3.11 copied."
    else
        echo "  ERROR: FTMO Python not found at $FTMO_PYTHON"
        echo "  You'll need to install Python 3.11 manually in this prefix."
    fi
fi

# ── Step 3: Install systemd services ──
echo ""
echo "[3/5] Installing systemd services..."
SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

cp "$SCRIPT_DIR/vncserver-bf.service" "$SYSTEMD_DIR/"
cp "$SCRIPT_DIR/mt5-bridge-bf.service" "$SYSTEMD_DIR/"
systemctl --user daemon-reload
echo "  Services installed."

# ── Step 4: Start VNC ──
echo ""
echo "[4/5] Starting VNC on display $DISPLAY..."
systemctl --user enable vncserver-bf.service
systemctl --user start vncserver-bf.service
echo "  VNC started. Connect to port $VNC_PORT to see the desktop."

# ── Step 5: Instructions ──
echo ""
echo "[5/5] Manual steps needed:"
echo ""
echo "  1. Connect via VNC to port $VNC_PORT"
echo "  2. Install MT5 in the Wine prefix:"
echo "     WINEPREFIX=$WINEPREFIX wine mt5setup.exe"
echo "  3. Log in with your BrightFunded credentials"
echo "  4. Enable algo trading in MT5 settings"
echo "  5. Start the bridge proxy:"
echo "     systemctl --user enable mt5-bridge-bf.service"
echo "     systemctl --user start mt5-bridge-bf.service"
echo "  6. Enable in accounts.yaml:"
echo "     bright_50k.enabled: true"
echo "  7. Restart the bot:"
echo "     systemctl --user restart sovereign-bot"
echo ""
echo "Done! The BrightFunded account is ready for setup."
