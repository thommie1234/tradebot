"""Daily P/L summary with best day rule status — sent to Discord at 23:00 UTC."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def send_daily_summary(mt5, logger, discord, account_name: str = "",
                       account_size: float = 100000, initial_balance: float = 100000):
    """Generate and send daily P/L summary with best day rule check."""
    if mt5 is None or discord is None:
        return

    info = mt5.account_info()
    if not info:
        return

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get today's closed deals
    deals = mt5.history_deals_get(today_start, now)
    if not deals:
        deals = []

    today_pnl = 0
    today_trades = 0
    today_wins = 0
    today_commission = 0
    today_swap = 0
    sym_pnl = {}

    for d in deals:
        if d.entry != 1:  # exits only
            continue
        net = d.profit + d.commission + d.swap
        today_pnl += net
        today_trades += 1
        today_commission += d.commission
        today_swap += d.swap
        if net > 0:
            today_wins += 1
        sym = d.symbol
        if sym not in sym_pnl:
            sym_pnl[sym] = 0
        sym_pnl[sym] += net

    # Open positions P/L
    positions = mt5.positions_get()
    open_pnl = sum(p.profit for p in (positions or []) if p.magic >= 2000)
    open_count = sum(1 for p in (positions or []) if p.magic >= 2000)

    # Best day rule
    total_profit = info.balance - initial_balance
    wr = today_wins / today_trades * 100 if today_trades > 0 else 0

    # Build message
    name = f"[{account_name}] " if account_name else ""
    lines = [
        f"Balance: ${info.balance:,.2f}",
        f"Equity: ${info.equity:,.2f}",
        f"",
        f"Today: {today_trades} trades, {today_wins}W {today_trades - today_wins}L ({wr:.0f}% WR)",
        f"P/L: ${today_pnl:+,.2f} (comm: ${today_commission:,.2f}, swap: ${today_swap:,.2f})",
    ]

    if sym_pnl:
        lines.append("")
        top = sorted(sym_pnl.items(), key=lambda x: -x[1])[:5]
        for sym, pnl in top:
            lines.append(f"  {sym}: ${pnl:+,.2f}")

    if open_count > 0:
        lines.append(f"\nOpen: {open_count} positions, ${open_pnl:+,.2f}")

    # Best day rule
    lines.append(f"\nTotal profit: ${total_profit:+,.2f}")
    if total_profit > 0 and today_pnl > 0:
        best_day_pct = today_pnl / total_profit * 100
        other_days = total_profit - today_pnl
        lines.append(f"Best day rule: {best_day_pct:.1f}% {'OK' if best_day_pct <= 50 else 'NEED MORE DAYS'}")
        if best_day_pct > 50:
            needed = today_pnl - other_days
            lines.append(f"Need ${needed:+,.0f} more on other days")

    # DD status
    dd_pct = (initial_balance - info.equity) / initial_balance * 100 if info.equity < initial_balance else 0
    lines.append(f"\nMax DD used: {dd_pct:.2f}% / 10%")
    lines.append(f"DD floor: ${info.balance - 10000:,.2f}")

    body = "\n".join(lines)
    color = "green" if today_pnl > 0 else "red" if today_pnl < -100 else "blue"

    discord.send(f"{name}DAILY SUMMARY", body, color)

    if logger:
        logger.log('INFO', 'DailySummary', 'SENT',
                   f'P/L: ${today_pnl:+,.2f}, {today_trades} trades, balance ${info.balance:,.2f}')
