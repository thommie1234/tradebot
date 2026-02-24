#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

FORCE="${1:-}"
LOG="audit/logs/stress_test_$(date +%Y%m%d_%H%M).log"
mkdir -p audit/logs

# Check market status (skip with --force)
if [[ "$FORCE" != "--force" ]]; then
    HOUR=$(TZ="Europe/Amsterdam" date +%H)
    if (( HOUR >= 8 && HOUR < 22 )); then
        echo "Market is open (CET hour=$HOUR). Use --force to override."
        exit 1
    fi
fi

# Stop bot temporarily
echo "Stopping sovereign-bot for stress test..."
systemctl --user stop sovereign-bot || true

# Wait for bot to fully stop
sleep 5

# Preload data into page cache
echo "Warming up page cache..."
find /home/tradebot/ssd_data_1 -name "*.parquet" -print0 2>/dev/null | \
    xargs -0 -P 8 cat > /dev/null 2>&1 || true
find /home/tradebot/ssd_data_2 -name "*.parquet" -print0 2>/dev/null | \
    xargs -0 -P 4 cat > /dev/null 2>&1 || true

# Run stress test
echo "Starting extreme stress test..."
echo "Log: $LOG"
nice -n 0 python3 common/research/stress_test.py \
    --endurance-minutes "${ENDURANCE:-120}" \
    ${FORCE:+--force} \
    2>&1 | tee "$LOG"

# Restart bot
echo "Restarting sovereign-bot..."
systemctl --user start sovereign-bot

# Wait and verify
sleep 8
if systemctl --user is-active --quiet sovereign-bot; then
    echo "sovereign-bot is running again."
else
    echo "WARNING: sovereign-bot failed to restart!"
    systemctl --user status sovereign-bot || true
fi

echo ""
echo "Stress test complete."
echo "Results: models/optuna_results/stress_test_*/"
echo "Log:     $LOG"
