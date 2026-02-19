#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="${VENV_PY:-/home/tradebot/backup_prop_20260202/venv/bin/python}"

if [ ! -x "$VENV_PY" ]; then
  echo "[native-ml] Python not found: $VENV_PY"
  exit 1
fi

DEVICE="${DEVICE:-auto}"              # auto = GPU first, CPU fallback
SYMBOL_WORKERS="${SYMBOL_WORKERS:-4}" # outer parallel symbols
XGB_JOBS="${XGB_JOBS:-1}"             # inner per-model threads
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/ml_boxes_native}"

mkdir -p "$OUT_DIR"

echo "[native-ml] repo: $REPO_ROOT"
echo "[native-ml] python: $VENV_PY"
echo "[native-ml] device: $DEVICE (gpu-first if available)"
echo "[native-ml] symbol_workers: $SYMBOL_WORKERS, xgb_jobs: $XGB_JOBS"
echo "[native-ml] out_dir: $OUT_DIR"

exec "$VENV_PY" -m trading_prop.ml.train_ml_strategy \
  --device "$DEVICE" \
  --symbol-workers "$SYMBOL_WORKERS" \
  --xgb-jobs "$XGB_JOBS" \
  --out-dir "$OUT_DIR" \
  "$@"
