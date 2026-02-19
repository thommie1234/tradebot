"""
Drawdown guards — daily loss gate, profit lock, DD recovery mode.

Extracted from guardrails in execute_trade() (Gebod 43, 48, 54).
"""
from __future__ import annotations

from config.loader import cfg


class DrawdownGuard:
    """Enforces daily loss limit, profit lock, and drawdown recovery."""

    def __init__(self, logger, discord=None):
        self.logger = logger
        self.discord = discord
        self._dd_recovery_mode = False
        self._daily_loss_warned = False
        self._profit_lock_warned = False

    def check_daily_limits(self, account_info) -> tuple[bool, str]:
        """Check daily PnL limits. Returns (allowed, reason)."""
        if account_info is None:
            return True, ""

        daily_pnl_pct = (account_info.equity - account_info.balance) / cfg.ACCOUNT_SIZE
        if daily_pnl_pct <= -0.035:
            if self.discord and not self._daily_loss_warned:
                self.discord.send("DAILY LOSS LIMIT",
                                  f"PnL: {daily_pnl_pct:.2%}\nNo new trades until tomorrow.",
                                  "red")
                self._daily_loss_warned = True
            return False, f"daily PnL {daily_pnl_pct:.2%} hit -3.5% limit"

        if daily_pnl_pct >= 0.03:
            if self.discord and not self._profit_lock_warned:
                self.discord.send("PROFIT LOCKED",
                                  f"PnL: {daily_pnl_pct:.2%}\nNo new trades. Protecting gains.",
                                  "green")
                self._profit_lock_warned = True
            return False, f"daily PnL {daily_pnl_pct:.2%} hit +3% — locking profits"

        return True, ""

    def check_dd_recovery(self, account_info) -> bool:
        """Check if in drawdown recovery mode. Returns True if lots should be halved."""
        if account_info is None:
            return False

        dd_pct = (cfg.ACCOUNT_SIZE - account_info.equity) / cfg.ACCOUNT_SIZE
        if dd_pct >= 0.04:
            self._dd_recovery_mode = True
        elif dd_pct <= 0.01:
            if self._dd_recovery_mode:
                self.logger.log('INFO', 'DrawdownGuard', 'DD_RECOVERY_OFF',
                                f'Drawdown recovery OFF — equity recovered to {dd_pct:.2%} DD')
            self._dd_recovery_mode = False

        return self._dd_recovery_mode

    def reset_daily_flags(self):
        """Reset daily warning flags (call at start of new day)."""
        self._daily_loss_warned = False
        self._profit_lock_warned = False
