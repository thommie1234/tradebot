"""Watchdog — monitors trading bot processes and restarts on crash.

Checks every 60s if the live bot, paper bot, and MT5 terminal are running.
Sends Discord alerts on crash detection and restart.

Usage:
    python watchdog.py --account bright_100k
"""
import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request

ACCOUNT_CONFIGS = {
    "bright_100k": {
        "name": "BrightFunded 100k",
        "bot_script": "run_bot.py",
        "paper_script": "paper_bot.py",
        "live_bat": r"C:\tradebots\bf\run_live.bat",
        "paper_bat": r"C:\tradebots\bf\run_paper.bat",
        "mt5_path": r"C:\Program Files\BrightFunded MT5 Terminal\terminal64.exe",
        "log_dir": r"C:\tradebots\bf\logs",
    },
    "ftmo_100k": {
        "name": "FTMO 100k",
        "bot_script": "run_bot.py",
        "paper_script": "paper_bot.py",
        "live_bat": r"C:\tradebots\ftmo\run_live.bat",
        "paper_bat": r"C:\tradebots\ftmo\run_paper.bat",
        "mt5_path": r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
        "log_dir": r"C:\tradebots\ftmo\logs",
    },
    "ttp_100k": {
        "name": "The Trading Pit 100k",
        "bot_script": "run_bot.py",
        "paper_script": "paper_bot.py",
        "live_bat": r"C:\tradebots\ttp\run_live.bat",
        "paper_bat": r"C:\tradebots\ttp\run_paper.bat",
        "mt5_path": r"C:\MT5\TTP\terminal64.exe",
        "log_dir": r"C:\tradebots\ttp\logs",
    },
}

LOG_PATH = Path(r"C:\tradebots\logs\watchdog.log")


def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def send_discord(webhook_url: str, title: str, message: str, color: str = "red"):
    if not webhook_url:
        return
    colors = {"red": 0xFF0000, "green": 0x00FF00, "orange": 0xFFA500}
    payload = json.dumps({
        "embeds": [{
            "title": title,
            "description": message,
            "color": colors.get(color, 0xFF0000),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }).encode()
    req = urllib.request.Request(
        webhook_url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logging.warning(f"Discord send failed: {e}")


def is_process_running(name_fragment: str) -> bool:
    """Check if a process with the given command-line fragment is running."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             f"CommandLine like '%{name_fragment}%'",
             "get", "ProcessId", "/value"],
            capture_output=True, text=True, timeout=10,
        )
        return "ProcessId=" in result.stdout
    except Exception:
        return False


def is_mt5_running(mt5_path: str) -> bool:
    """Check if MT5 terminal is running."""
    exe_name = Path(mt5_path).name
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}"],
            capture_output=True, text=True, timeout=10,
        )
        return exe_name.lower() in result.stdout.lower()
    except Exception:
        return False


def restart_process(bat_path: str, title: str):
    """Start a bat file in a new window."""
    if not Path(bat_path).exists():
        logging.error(f"Bat file not found: {bat_path}")
        return False
    try:
        subprocess.Popen(
            f'start "{title}" cmd /c "{bat_path}"',
            shell=True,
        )
        logging.info(f"Restarted: {title} via {bat_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to restart {title}: {e}")
        return False


def restart_mt5(mt5_path: str):
    """Start MT5 terminal."""
    try:
        subprocess.Popen([mt5_path])
        logging.info(f"Restarted MT5: {mt5_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to restart MT5: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Trading bot watchdog")
    parser.add_argument("--account", default="bright_100k")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()

    setup_logging()

    acct = ACCOUNT_CONFIGS.get(args.account)
    if not acct:
        logging.error(f"Unknown account: {args.account}")
        return

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    # Try to load webhook from discord notifier config
    if not webhook_url:
        try:
            from config.loader import cfg
            webhook_url = getattr(cfg, 'DISCORD_WEBHOOK', '')
        except Exception:
            pass

    logging.info(f"Watchdog started for {args.account} (interval={args.interval}s)")
    send_discord(webhook_url, "WATCHDOG STARTED",
                 f"Monitoring {args.account} every {args.interval}s", "green")

    while True:
        try:
            # Check MT5
            if not is_mt5_running(acct["mt5_path"]):
                logging.warning("MT5 not running — restarting")
                send_discord(webhook_url, "MT5 CRASHED",
                             f"MT5 terminal not running. Restarting...", "red")
                restart_mt5(acct["mt5_path"])
                time.sleep(30)  # Wait for MT5 to connect

            # Check live bot
            if not is_process_running(acct["bot_script"]):
                logging.warning("Live bot not running — restarting")
                send_discord(webhook_url, "BOT CRASHED",
                             f"Live bot ({args.account}) not running. Restarting...", "red")
                restart_process(acct["live_bat"], f"sovereign-bot-{args.account}")
                time.sleep(10)

            # Check paper bot
            if not is_process_running(acct["paper_script"]):
                logging.warning("Paper bot not running — restarting")
                restart_process(acct["paper_bat"], f"paper-bot-{args.account}")

        except Exception as e:
            logging.error(f"Watchdog error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
