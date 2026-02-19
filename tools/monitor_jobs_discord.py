#!/usr/bin/env python3
"""Periodic progress monitor for downloader/backtests with Discord notifications."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from glob import glob

DRIVES = [
    "/home/tradebot/ssd_data_1",
    "/home/tradebot/ssd_data_2",
    "/home/tradebot/data_1",
    "/home/tradebot/data_2",
    "/home/tradebot/data_3",
]


def load_webhook(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("enabled", False):
        return None
    return cfg.get("webhook_url")


def send_discord_embed(webhook_url, title, description, fields, color=0x0099FF):
    payload = {
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.now(UTC).isoformat(),
                "fields": fields,
                "footer": {"text": "Sovereign Progress Monitor"},
            }
        ]
    }
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as _:
        return


def hbytes(num):
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(num)
    for u in units:
        if n < 1024.0:
            return f"{n:.1f}{u}"
        n /= 1024.0
    return f"{n:.1f}PB"


def drive_stats():
    rows = []
    for d in DRIVES:
        td = os.path.join(d, "tick_data")
        files = glob(os.path.join(td, "*", "*.parquet")) if os.path.isdir(td) else []
        total_size = 0
        for fp in files:
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                pass
        try:
            st = os.statvfs(d)
            free = st.f_bavail * st.f_frsize
        except OSError:
            free = 0
        rows.append((d, len(files), total_size, free))
    return rows


def months_coverage(root="/home/tradebot/ssd_data_1/tick_data"):
    if not os.path.isdir(root):
        return 0, 0, "n/a"
    symbols = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    if not symbols:
        return 0, 0, "n/a"
    month_sets = []
    for s in symbols:
        files = glob(os.path.join(root, s, "*.parquet"))
        months = {os.path.basename(f).replace(".parquet", "") for f in files}
        month_sets.append(months)
    common = set.intersection(*month_sets) if month_sets else set()
    latest = max(common) if common else "n/a"
    return len(symbols), len(common), latest


def proc_count(pattern):
    cmd = ["bash", "-lc", f"ps -ef | rg -n \"{pattern}\" | rg -v \"rg -n\" | wc -l"]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out or "0")


def one_report(webhook_url):
    ds = drive_stats()
    total_files = sum(x[1] for x in ds)
    total_size = sum(x[2] for x in ds)
    symbols, common_months, latest_common = months_coverage()

    downloader = proc_count("strategy_bot.py --download-data")
    backtests = proc_count("launcher.py backtest_engine.py")

    fields = [
        {"name": "Downloader", "value": f"{downloader} proc", "inline": True},
        {"name": "Backtests", "value": f"{backtests} proc", "inline": True},
        {"name": "Total Files", "value": str(total_files), "inline": True},
        {"name": "Total Tick Size", "value": hbytes(total_size), "inline": True},
        {"name": "Symbols (ssd_data_1)", "value": str(symbols), "inline": True},
        {"name": "Common Months", "value": f"{common_months} (latest {latest_common})", "inline": False},
    ]
    for d, files, size, free in ds:
        fields.append(
            {"name": os.path.basename(d), "value": f"{files} files | {hbytes(size)} | free {hbytes(free)}", "inline": False}
        )

    send_discord_embed(
        webhook_url=webhook_url,
        title="📡 TRADES Data/Backtest Progress",
        description=f"Snapshot {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fields=fields,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=600, help="Seconds between reports")
    p.add_argument("--config", default="/home/tradebot/tradebots/config/discord_config.json")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    webhook = load_webhook(args.config)
    if not webhook:
        raise SystemExit("Discord webhook disabled/not found.")

    try:
        one_report(webhook)
    except Exception as e:
        print(f"[monitor] first send failed: {e}")
    if args.once:
        return
    while True:
        time.sleep(args.interval)
        try:
            one_report(webhook)
        except Exception as e:
            print(f"[monitor] send failed: {e}")


if __name__ == "__main__":
    main()
