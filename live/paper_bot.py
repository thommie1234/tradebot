#!/usr/bin/env python3
"""Standalone paper trading bot — runs independently from the live bot.

Scans all symbols on M5/M15/M30/H1/H4 timeframes, simulates trades
with real MT5 prices, tracks positions with SL/TP/trailing/BE.

Usage:
    python3 common/live/paper_bot.py --account-id ftmo_100k
    python3 common/live/paper_bot.py --account-id bright_100k

Systemd services:
    systemctl --user start paper-bot-ftmo
    systemctl --user start paper-bot-bf
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p
os.chdir(REPO_ROOT)

from config.loader import cfg, load_config

load_config()


def main():
    parser = argparse.ArgumentParser(description="Standalone paper trading bot")
    parser.add_argument("--account-id", required=True,
                        help="Account ID (ftmo_100k or bright_100k)")
    parser.add_argument("--scan-interval", type=int, default=1,
                        help="Base scan interval in minutes (default: 1)")
    args = parser.parse_args()

    # Pin to CPUs 20-27 to avoid interfering with live trading
    try:
        os.sched_setaffinity(0, set(range(20, 28)))
        print(f"[PAPER] Pinned to CPUs 20-27")
    except Exception:
        pass

    # Load account config
    import yaml
    accts_path = REPO_ROOT / "config" / "accounts.yaml"
    with open(accts_path) as f:
        accts = yaml.safe_load(f).get("accounts", {})
    acct_cfg = accts.get(args.account_id)
    if not acct_cfg:
        print(f"[ERROR] Account '{args.account_id}' not found in accounts.yaml")
        sys.exit(1)

    # Apply account-specific config overrides
    if acct_cfg.get("config_path"):
        cfg.CONFIG_PATH = str(REPO_ROOT / acct_cfg["config_path"])
    if acct_cfg.get("model_dir"):
        cfg.MODEL_DIR = str(REPO_ROOT / acct_cfg["model_dir"])

    # Initialize logger
    from audit.audit_logger import BlackoutLogger
    prefix = args.account_id.split("_")[0]
    # bright_100k → bf/ directory (not bright/), ttp_demo → ttp/
    DIR_PREFIX_MAP = {"bright": "bf"}
    dir_prefix = DIR_PREFIX_MAP.get(prefix, prefix)
    log_db = str(REPO_ROOT / dir_prefix / "audit" / "paper_bot_log.db")
    os.makedirs(os.path.dirname(log_db), exist_ok=True)
    logger = BlackoutLogger(db_path=log_db)

    # Initialize Discord (optional)
    discord = None
    try:
        from tools.discord_notifier import DiscordNotifier
        config_path = REPO_ROOT / "config" / "discord_config.json"
        if config_path.exists():
            with open(config_path) as f:
                dc = json.load(f)
            discord = DiscordNotifier(dc.get("webhook_url"))
    except Exception:
        pass

    # Initialize MT5
    from tools.mt5_bridge import mt5, initialize_mt5
    terminal_path = acct_cfg.get("terminal_path")
    ok, _err, _mode = initialize_mt5(terminal_path=terminal_path)
    if ok:
        info = mt5.account_info()
        if info:
            print(f"[PAPER] MT5 connected: {info.login} | "
                  f"balance=${info.balance:,.2f}")
        else:
            print(f"[PAPER] MT5 initialized, no account info")
    else:
        print(f"[WARNING] MT5 init failed — will retry on scan")

    # Initialize paper tracker
    from live.paper_tracker import PaperTracker
    risk_scale = acct_cfg.get("risk_scale", 1.0)
    pt_db = str(REPO_ROOT / dir_prefix / "audit" / "paper_trades.db")
    pt = PaperTracker(
        account_id=args.account_id,
        db_path=pt_db,
        logger=logger,
        discord=discord,
        risk_scale=risk_scale,
    )
    pt.load_models(logger)

    if not pt._tf_filters and not pt._filters:
        print("[ERROR] No models loaded — nothing to scan. Train models first.")
        sys.exit(1)

    # Count models per TF
    tf_counts = {}
    for (sym, tf) in pt._tf_filters:
        tf_counts[tf] = tf_counts.get(tf, 0) + 1
    tf_summary = ", ".join(f"{tf}={n}" for tf, n in sorted(tf_counts.items()))
    print(f"[PAPER] Loaded {len(pt._tf_filters)} models: {tf_summary}")

    # Start SL/TP monitor thread (1s interval, uses real MT5 ticks)
    pt.start_monitor(mt5, interval=0.0)
    print("[PAPER] SL/TP monitor started (max speed, no sleep)")

    # Graceful shutdown
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False
        print(f"\n[PAPER] Caught signal {signum}, shutting down...")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # TF scan intervals in minutes
    tf_intervals = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}

    print(f"[PAPER] Starting scan loop for {args.account_id}")
    logger.log('INFO', 'PaperBot', 'STARTED',
               f'{args.account_id}: {len(pt._tf_filters)} models, TFs: {tf_summary}')

    if discord:
        discord.send("PAPER BOT STARTED",
                      f"[{args.account_id}] {len(pt._tf_filters)} models ({tf_summary})",
                      "blue")

    last_daily = None

    # Trading schedule for market-close checks
    from risk.ftmo_guard import TradingSchedule
    trading_schedule = TradingSchedule()

    while running:
        try:
            now = datetime.now(timezone.utc)
            minute = now.hour * 60 + now.minute

            # Close positions whose market is closed (session end, weekend, etc.)
            pt.close_market_closed(mt5, trading_schedule)

            # Scan each TF if its interval has elapsed
            for tf, interval in tf_intervals.items():
                if minute % interval == 0:
                    try:
                        count = pt.run_full_scan(mt5, logger, timeframe=tf)
                        pt.on_bar_close(mt5, timeframe=tf)
                        if count and count > 0:
                            logger.log('INFO', 'PaperBot', 'SCAN_SIGNALS',
                                       f'{tf}: {count} signals found')
                    except Exception as e:
                        logger.log('ERROR', 'PaperBot', 'SCAN_ERROR',
                                   f'{tf}: {e}')

            # Daily summary at 00:00 UTC
            today = now.strftime("%Y-%m-%d")
            if now.hour == 0 and now.minute == 0 and last_daily != today:
                last_daily = today
                try:
                    stats = pt.daily_summary()
                    if discord and stats:
                        lines = [
                            f"Equity: ${stats.get('equity', 0):,.0f}",
                            f"Open: {stats.get('open_positions', 0)}",
                            f"Closed today: {stats.get('closed_today', 0)}",
                            f"Day P&L: ${stats.get('daily_pnl', 0):,.2f}",
                            f"Win rate: {stats.get('win_rate', 0):.1%}",
                        ]
                        discord.send(f"PAPER DAILY [{args.account_id}]",
                                     "\n".join(lines), "blue")
                except Exception as e:
                    logger.log('ERROR', 'PaperBot', 'DAILY_ERROR', str(e))

        except Exception as e:
            logger.log('ERROR', 'PaperBot', 'LOOP_ERROR', str(e))

        # Sleep until next minute boundary + 2s buffer
        now2 = datetime.now(timezone.utc)
        sleep_s = max(1, 60 - now2.second + 2)
        for _ in range(int(sleep_s)):
            if not running:
                break
            time.sleep(1)

    # Shutdown
    pt._monitor_running = False
    logger.log('INFO', 'PaperBot', 'STOPPED', f'{args.account_id}')
    print("[PAPER] Shutdown complete")


if __name__ == "__main__":
    main()
