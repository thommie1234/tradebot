#!/bin/bash
# Daily backup of audit trail DB + logs to secondary disk
# Run via cron: 0 3 * * * /home/tradebot/tradebots/tools/backup_audit.sh

BACKUP_ROOT="/home/tradebot/data_1/backups"
LOGS_PERM="/home/tradebot/data_1/audit_logs"
DATE=$(date +%Y-%m-%d)
BACKUP_DIR="$BACKUP_ROOT/$DATE"
SRC_DB="/home/tradebot/tradebots/audit/sovereign_log.db"
SRC_LOGS="/home/tradebot/tradebots/audit/logs"

mkdir -p "$BACKUP_DIR"
mkdir -p "$LOGS_PERM"

# SQLite safe backup (handles WAL mode)
sqlite3 "$SRC_DB" ".backup '$BACKUP_DIR/sovereign_log.db'"
echo "[backup] DB: $(du -sh "$BACKUP_DIR/sovereign_log.db" | cut -f1)"

# Sync logs to permanent location (kept forever)
cp -a "$SRC_LOGS"/*.log "$LOGS_PERM/" 2>/dev/null
echo "[backup] Logs synced: $(du -sh "$LOGS_PERM" | cut -f1)"

echo "[backup] Done: DB=$BACKUP_DIR ($(du -sh "$BACKUP_DIR" | cut -f1)) | Logs=$LOGS_PERM"
