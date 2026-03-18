"""Dynamic multi-account startup — reads accounts.yaml and starts all enabled accounts.

For each enabled account:
  1. Start its MT5 terminal
  2. Wait for MT5 IPC to be ready (max 3 min)
  3. Start the live bot
  4. Start the paper bot

Usage:
    python start_trading.py              # Start all enabled accounts
    python start_trading.py --paper-only # Only paper bots, no live trading
    python start_trading.py --account bright_100k  # Start one specific account
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
ACCOUNTS_YAML = REPO_ROOT / "config" / "accounts.yaml"


def load_accounts(only_account: str | None = None) -> list[dict]:
    """Load enabled accounts from accounts.yaml."""
    with open(ACCOUNTS_YAML) as f:
        data = yaml.safe_load(f)

    accounts = []
    for acct_id, acfg in data.get("accounts", {}).items():
        if only_account and acct_id != only_account:
            continue
        if not acfg.get("enabled", False) and not only_account:
            continue
        accounts.append({"id": acct_id, **acfg})
    return accounts


def start_mt5(terminal_path: str, name: str) -> bool:
    """Start MT5 terminal and wait for IPC to be ready."""
    if not Path(terminal_path).exists():
        print(f"  [!] MT5 niet gevonden: {terminal_path}")
        return False

    print(f"  MT5 starten: {name}...")
    subprocess.Popen([terminal_path])

    # Wacht tot MT5 IPC werkt (max 3 minuten)
    for attempt in range(1, 19):
        time.sleep(10)
        try:
            result = subprocess.run(
                [str(VENV_PYTHON), "-c",
                 f"import MetaTrader5 as mt5; "
                 f"mt5.initialize(r'{terminal_path}'); "
                 f"ok=mt5.terminal_info() is not None; "
                 f"mt5.shutdown(); "
                 f"exit(0 if ok else 1)"],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                print(f"  MT5 IPC ready na {attempt * 10}s")
                return True
        except subprocess.TimeoutExpired:
            pass
        print(f"  Poging {attempt}/18 — MT5 nog niet klaar...")

    print(f"  [!] MT5 IPC timeout na 3 minuten voor {name}")
    return False


def start_bot(bat_path: str, log_path: str, title: str):
    """Start a bot via its bat file in a new window."""
    if not Path(bat_path).exists():
        print(f"  [!] Bat niet gevonden: {bat_path}")
        return
    # Zorg dat logs dir bestaat
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        f'start "{title}" cmd /c "{bat_path} >{log_path} 2>&1"',
        shell=True,
    )
    print(f"  Gestart: {title} -> {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Start alle enabled trading accounts")
    parser.add_argument("--paper-only", action="store_true",
                        help="Alleen paper bots starten, geen live trading")
    parser.add_argument("--account", type=str, default=None,
                        help="Start alleen dit specifieke account")
    parser.add_argument("--no-mt5", action="store_true",
                        help="MT5 terminals niet starten (al open)")
    args = parser.parse_args()

    if not ACCOUNTS_YAML.exists():
        print(f"[!] accounts.yaml niet gevonden: {ACCOUNTS_YAML}")
        sys.exit(1)

    accounts = load_accounts(args.account)
    if not accounts:
        print("[!] Geen enabled accounts gevonden in accounts.yaml")
        sys.exit(1)

    print("=" * 60)
    print(f"  SOVEREIGN TRADING — {len(accounts)} account(s)")
    print("=" * 60)
    for acct in accounts:
        print(f"  {acct['name']} ({acct['id']}) — {'LIVE' if not args.paper_only else 'PAPER ONLY'}")
    print("=" * 60)
    print()

    # Groepeer per MT5 terminal (meerdere accounts kunnen dezelfde terminal delen)
    terminals_started = set()

    for acct in accounts:
        acct_id = acct["id"]
        name = acct["name"]
        terminal_path = acct.get("terminal_path", "")
        # Account directory prefix (bright_100k -> bf, ftmo_100k -> ftmo, ttp_100k -> ttp)
        prefix_map = {
            "bright_100k": "bf",
            "ftmo_100k": "ftmo",
            "ttp_100k": "ttp",
        }
        prefix = prefix_map.get(acct_id, acct_id.split("_")[0])

        print(f"[{name}]")

        # 1. Start MT5 terminal (als nog niet gestart)
        if not args.no_mt5 and terminal_path and terminal_path not in terminals_started:
            ok = start_mt5(terminal_path, name)
            if not ok:
                print(f"  [!] Overslaan — MT5 niet bereikbaar\n")
                continue
            terminals_started.add(terminal_path)
        elif terminal_path in terminals_started:
            print(f"  MT5 al gestart (gedeelde terminal)")

        # 2. Start live bot
        if not args.paper_only and not acct.get("paper_only", False):
            live_bat = str(REPO_ROOT / prefix / "run_live.bat")
            live_log = str(REPO_ROOT / prefix / "logs" / "service.log")
            start_bot(live_bat, live_log, f"sovereign-bot-{prefix}")
            time.sleep(5)  # Even ruimte tussen starts

        # 3. Start paper bot
        paper_bat = str(REPO_ROOT / prefix / "run_paper.bat")
        paper_log = str(REPO_ROOT / prefix / "logs" / "paper_service.log")
        start_bot(paper_bat, paper_log, f"paper-bot-{prefix}")

        print()

    # 4. Start watchdog
    watchdog_path = REPO_ROOT / "common" / "tools" / "watchdog.py"
    if watchdog_path.exists():
        print("[Watchdog]")
        for acct in accounts:
            if not acct.get("paper_only", False) and not args.paper_only:
                acct_id = acct["id"]
                subprocess.Popen(
                    f'start "watchdog-{acct_id}" cmd /c "'
                    f'{VENV_PYTHON} -u {watchdog_path} --account {acct_id} '
                    f'>{REPO_ROOT / "logs" / f"watchdog_{acct_id}.log"} 2>&1"',
                    shell=True,
                )
                print(f"  Watchdog gestart voor {acct['name']}")

    print()
    print("=" * 60)
    print("  ALLES GESTART")
    print("=" * 60)


if __name__ == "__main__":
    main()
