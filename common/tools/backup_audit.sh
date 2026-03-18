#!/bin/bash
# Daily backup of ALL audit trail DBs + logs to secondary disk
# Triggered by: systemctl --user start audit-backup (via audit-backup.timer at 03:00)

REPO="/home/tradebot/tradebots"
BACKUP_ROOT="/home/tradebot/data_1/backups"
LOGS_PERM="/home/tradebot/data_1/audit_logs"
DATE=$(date +%Y-%m-%d)
BACKUP_DIR="$BACKUP_ROOT/$DATE"

mkdir -p "$BACKUP_DIR"
mkdir -p "$LOGS_PERM"

# Backup all active audit DBs (SQLite safe backup handles WAL mode)
for ACCT in ftmo bf; do
    DB="$REPO/$ACCT/audit/sovereign_log.db"
    if [ -f "$DB" ]; then
        sqlite3 "$DB" ".backup '$BACKUP_DIR/${ACCT}_sovereign_log.db'"
        echo "[backup] $ACCT audit: $(du -sh "$BACKUP_DIR/${ACCT}_sovereign_log.db" | cut -f1)"
    fi

    PAPER_DB="$REPO/$ACCT/audit/paper_trades.db"
    if [ -f "$PAPER_DB" ] && [ -s "$PAPER_DB" ]; then
        sqlite3 "$PAPER_DB" ".backup '$BACKUP_DIR/${ACCT}_paper_trades.db'"
        echo "[backup] $ACCT paper: $(du -sh "$BACKUP_DIR/${ACCT}_paper_trades.db" | cut -f1)"
    fi
done

# Sync logs to permanent location
for ACCT in ftmo bf; do
    LOGDIR="$REPO/$ACCT/logs"
    if [ -d "$LOGDIR" ]; then
        mkdir -p "$LOGS_PERM/$ACCT"
        cp -a "$LOGDIR"/*.log "$LOGS_PERM/$ACCT/" 2>/dev/null
    fi
done
echo "[backup] Logs synced: $(du -sh "$LOGS_PERM" | cut -f1)"

echo "[backup] Done: $BACKUP_DIR ($(du -sh "$BACKUP_DIR" | cut -f1))"
