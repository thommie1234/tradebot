#!/usr/bin/env python3
"""
Discord Notifier - Real-time trade alerts
==========================================

Sends Discord notifications for:
- Trade entries (FILLED)
- Trade exits (with PnL)
- Emergency stop triggers
- Daily PnL milestones
"""

import requests
import json
from datetime import datetime

class DiscordNotifier:
    """Discord webhook notifier"""

    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None and webhook_url != ""

        if self.enabled:
            print(f"Discord notifier ENABLED")
        else:
            print(f"Discord notifier DISABLED (no webhook URL)")

    def send(self, title, description, color=None, fields=None):
        """Send Discord embed message"""
        if not self.enabled:
            return

        # Color codes
        colors = {
            'green': 0x00ff00,
            'red': 0xff0000,
            'yellow': 0xffff00,
            'blue': 0x0099ff,
            'orange': 0xff9900
        }

        embed_color = colors.get(color, 0x0099ff)

        embed = {
            "title": title,
            "description": description,
            "color": embed_color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Sovereign Execution Engine"
            }
        }

        if fields:
            embed["fields"] = fields

        payload = {
            "embeds": [embed]
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            if response.status_code != 204:
                print(f"Discord webhook error: {response.status_code}")
        except Exception as e:
            print(f"Discord notification failed: {str(e)}")

    def trade_entry(self, symbol, direction, entry_price, lot_size, tp, sl, ticket, confidence, account=None):
        """Notify trade entry"""
        acct_tag = f" [{account}]" if account else ""
        self.send(
            title=f"🎯 TRADE ENTRY{acct_tag}: {symbol} {direction}",
            description=f"Position opened with ML confidence {confidence*100:.1f}%",
            color='blue',
            fields=[
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Direction", "value": direction, "inline": True},
                {"name": "Ticket", "value": f"#{ticket}", "inline": True},
                {"name": "Entry Price", "value": f"${entry_price:.5f}", "inline": True},
                {"name": "Lot Size", "value": f"{lot_size:.2f}", "inline": True},
                {"name": "ML Filter", "value": f"{confidence*100:.1f}%", "inline": True},
                {"name": "Take Profit", "value": f"${tp:.5f}", "inline": True},
                {"name": "Stop Loss", "value": f"${sl:.5f}", "inline": True},
            ]
        )

    def trade_exit(self, symbol, direction, entry_price, exit_price, pnl, ticket, reason, account=None):
        """Notify trade exit"""
        color = 'green' if pnl > 0 else 'red'
        emoji = '✅' if pnl > 0 else '❌'
        acct_tag = f" [{account}]" if account else ""

        self.send(
            title=f"{emoji} TRADE EXIT{acct_tag}: {symbol} {direction}",
            description=f"Position closed: {reason}",
            color=color,
            fields=[
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Direction", "value": direction, "inline": True},
                {"name": "Ticket", "value": f"#{ticket}", "inline": True},
                {"name": "Entry", "value": f"${entry_price:.5f}", "inline": True},
                {"name": "Exit", "value": f"${exit_price:.5f}", "inline": True},
                {"name": "P&L", "value": f"${pnl:.2f}", "inline": True},
            ]
        )

    def emergency_stop(self, reason, daily_pnl, open_positions):
        """Notify emergency stop triggered"""
        self.send(
            title="🚨 EMERGENCY STOP TRIGGERED",
            description=reason,
            color='red',
            fields=[
                {"name": "Daily P&L", "value": f"${daily_pnl:.2f}", "inline": True},
                {"name": "Positions Closed", "value": str(open_positions), "inline": True},
                {"name": "Status", "value": "⛔ TRADING HALTED", "inline": False},
            ]
        )

    def daily_summary(self, date, trades, wins, losses, total_pnl, win_rate):
        """Daily summary notification"""
        color = 'green' if total_pnl > 0 else 'red'
        emoji = '📈' if total_pnl > 0 else '📉'

        self.send(
            title=f"{emoji} Daily Summary - {date}",
            description=f"Trading session completed",
            color=color,
            fields=[
                {"name": "Total Trades", "value": str(trades), "inline": True},
                {"name": "Wins", "value": str(wins), "inline": True},
                {"name": "Losses", "value": str(losses), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate*100:.1f}%", "inline": True},
                {"name": "Total P&L", "value": f"${total_pnl:.2f}", "inline": True},
            ]
        )

    def milestone(self, message, current_pnl):
        """P&L milestone notification"""
        self.send(
            title="🎊 MILESTONE REACHED",
            description=message,
            color='yellow',
            fields=[
                {"name": "Current Daily P&L", "value": f"${current_pnl:.2f}", "inline": True},
            ]
        )

if __name__ == "__main__":
    # Test Discord webhook
    import sys

    if len(sys.argv) < 2:
        print("Usage: python discord_notifier.py <webhook_url>")
        sys.exit(1)

    webhook_url = sys.argv[1]
    notifier = DiscordNotifier(webhook_url)

    print("Testing Discord notifications...")

    # Test trade entry
    notifier.trade_entry(
        symbol="SOLUSD",
        direction="BUY",
        entry_price=116.87,
        lot_size=1.0,
        tp=117.22,
        sl=116.52,
        ticket=98234926,
        confidence=0.552
    )

    print("✓ Test notification sent!")
