#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
SPLIT_DIR="$SCRIPT_DIR/symbol_splits"
RUN_WINE="$SCRIPT_DIR/run_wine.sh"
PY_SPLIT="${PY_SPLIT:-python3}"
WORKERS="${WORKERS:-6}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$SPLIT_DIR"

echo "[max-download] creating $WORKERS symbol splits..."
"$PY_SPLIT" "$SCRIPT_DIR/make_symbol_splits.py" --workers "$WORKERS" --out-dir "$SPLIT_DIR"

echo "[max-download] stopping old single downloader if running..."
pkill -f "strategy_bot.py --download-data" || true
sleep 2

for f in "$SPLIT_DIR"/symbols_part_*.xlsx; do
    [ -f "$f" ] || continue
    part="$(basename "$f" .xlsx)"
    log="$LOG_DIR/downloader_${part}_${TS}.log"
    echo "[max-download] start $part -> $log"
    SYMBOLS_FILE_PATH="$f" nohup "$RUN_WINE" strategy_bot.py --download-data > "$log" 2>&1 &
done

echo "[max-download] started."
