"""Quick log viewer — usage: python logs.py [N] [filter]

Examples:
    python logs.py              # last 30 entries
    python logs.py 50           # last 50 entries
    python logs.py 20 ERROR     # last 20 errors
    python logs.py 20 MultiTF   # last 20 MultiTF entries
    python logs.py 20 SCAN      # last 20 scan-related entries
"""
import sqlite3
import sys
import os
from pathlib import Path

# Fix Windows cp1252 encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Search all possible log locations
DB_PATHS = [
    Path(r"C:\tradebots\bf\audit\sovereign_log.db"),
    Path(r"C:\tradebots\ftmo\audit\sovereign_log.db"),
    Path(r"C:\tradebots\common\audit\sovereign_log.db"),
    Path(r"C:\tradebots\common\audit\logs\sovereign_log.db"),
]

# Pick most recently modified
candidates = [(p, p.stat().st_mtime) for p in DB_PATHS if p.exists() and p.stat().st_size > 0]
if not candidates:
    print("No log database found!")
    sys.exit(1)

candidates.sort(key=lambda x: x[1], reverse=True)
db = candidates[0][0]

n = 30
filt = None
for arg in sys.argv[1:]:
    if arg.isdigit():
        n = int(arg)
    else:
        filt = arg

conn = sqlite3.connect(str(db), timeout=10)
conn.row_factory = sqlite3.Row

# Auto-detect table name
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
table = "events" if "events" in tables else "logs" if "logs" in tables else None
if not table:
    print(f"No events/logs table found in {db}")
    print(f"Tables: {tables}")
    sys.exit(1)

# Auto-detect column names
cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
evt_col = "event_type" if "event_type" in cols else "event" if "event" in cols else None

sel = f"timestamp, level, component, {evt_col}, message" if evt_col else "timestamp, level, component, message"

if filt:
    where_parts = ["level LIKE ?", "component LIKE ?", "message LIKE ?"]
    params = [f"%{filt}%", f"%{filt}%", f"%{filt}%"]
    if evt_col:
        where_parts.append(f"{evt_col} LIKE ?")
        params.append(f"%{filt}%")
    rows = conn.execute(
        f"SELECT {sel} FROM {table} WHERE {' OR '.join(where_parts)} ORDER BY rowid DESC LIMIT ?",
        params + [n],
    ).fetchall()
else:
    rows = conn.execute(
        f"SELECT {sel} FROM {table} ORDER BY rowid DESC LIMIT ?", (n,),
    ).fetchall()

conn.close()

if not rows:
    print("No log entries found.")
    sys.exit(0)

print(f"--- {db.name} ({db.parent.parent.name}/{db.parent.name}) --- last {len(rows)} entries ---\n")

# Print newest last (reverse)
for r in reversed(rows):
    ts = r["timestamp"] or ""
    # Shorten to HH:MM:SS
    if "T" in ts:
        ts = ts.split("T", 1)[1][:8]
    elif " " in ts:
        ts = ts.split(" ", 1)[1][:8]
    lvl = (r["level"] or "")[:5]
    comp = (r["component"] or "")[:12]
    evt = (r[evt_col] or "")[:20] if evt_col else ""
    msg = r["message"] or ""
    print(f"{ts} {lvl:<5} {comp:<12} {evt:<20} {msg}")
