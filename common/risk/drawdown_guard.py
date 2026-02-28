"""
Drawdown guards — daily loss gate, profit lock, DD recovery mode.

Extracted from guardrails in execute_trade() (Gebod 43, 48, 54).
"""
from __future__ import annotations

from datetime import datetime, timezone

from config.loader import cfg


class DrawdownGuard:
    """Enforces daily loss limit, profit lock, and drawdown recovery.

    Supports parameterized thresholds for multi-account setups.
    Uses daily_start_balance (persisted across restarts via audit DB)
    to track true daily P&L including realized losses.
    """

    def __init__(self, logger, discord=None, account_name: str = "default",
                 daily_loss_pct: float = 0.035, profit_lock_pct: float = 0.03,
                 dd_recovery_threshold: float = 0.04, dd_recovery_exit: float = 0.01,
                 profit_gate_pct: float = 0.015, profit_gate_min_conf: float = 0.90,
                 daily_start_balance: float = 0):
        self.logger = logger
        self.discord = discord
        self.account_name = account_name
        self.daily_loss_pct = daily_loss_pct
        self.profit_lock_pct = profit_lock_pct
        self.dd_recovery_threshold = dd_recovery_threshold
        self.dd_recovery_exit = dd_recovery_exit
        self.profit_gate_pct = profit_gate_pct
        self.profit_gate_min_conf = profit_gate_min_conf
        self.daily_start_balance = daily_start_balance
        self._last_reset_date = datetime.now(timezone.utc).date()
        self._dd_recovery_mode = False
        self._daily_loss_warned = False
        self._profit_lock_warned = False
        self._profit_gate_warned = False

    def _check_daily_reset(self, account_info):
        """Reset daily tracking at UTC midnight."""
        today = datetime.now(timezone.utc).date()
        if today > self._last_reset_date:
            self.daily_start_balance = account_info.balance
            self._last_reset_date = today
            self.reset_daily_flags()
            self.logger.log('INFO', 'DrawdownGuard', 'DAILY_RESET',
                            f'[{self.account_name}] Daily start balance reset to '
                            f'${account_info.balance:,.2f}')

    def check_daily_limits(self, account_info) -> tuple[bool, str]:
        """Check daily PnL limits. Returns (allowed, reason).

        Uses daily_start_balance (real day-start, survives restarts) instead
        of current balance, so realized losses are correctly tracked.
        """
        if account_info is None:
            return True, ""

        self._check_daily_reset(account_info)

        # Use day-start balance for true daily P&L (realized + unrealized)
        base = self.daily_start_balance if self.daily_start_balance > 0 else account_info.balance
        if base <= 0:
            return True, ""  # No valid balance — allow trading, don't crash
        daily_pnl_pct = (account_info.equity - base) / base
        if daily_pnl_pct <= -self.daily_loss_pct:
            if self.discord and not self._daily_loss_warned:
                self.discord.send(f"[{self.account_name}] DAILY LOSS LIMIT",
                                  f"PnL: {daily_pnl_pct:.2%}\nNo new trades until tomorrow.",
                                  "red")
                self._daily_loss_warned = True
            return False, f"daily PnL {daily_pnl_pct:.2%} hit -{self.daily_loss_pct:.1%} limit"

        if daily_pnl_pct >= self.profit_lock_pct:
            if self.discord and not self._profit_lock_warned:
                self.discord.send(f"[{self.account_name}] PROFIT LOCKED",
                                  f"PnL: {daily_pnl_pct:.2%}\nNo new trades. Protecting gains.",
                                  "green")
                self._profit_lock_warned = True
            return False, f"daily PnL {daily_pnl_pct:.2%} hit +{self.profit_lock_pct:.0%} — locking profits"

        return True, ""

    def check_profit_gate(self, account_info, ml_confidence: float) -> tuple[bool, str]:
        """Soft profit gate: above +profit_gate_pct daily P&L, only allow
        high-confidence trades (>= profit_gate_min_conf).

        Returns (allowed, reason).
        """
        if account_info is None or self.profit_gate_pct <= 0:
            return True, ""

        base = self.daily_start_balance if self.daily_start_balance > 0 else account_info.balance
        if base <= 0:
            return True, ""
        daily_pnl_pct = (account_info.equity - base) / base

        if daily_pnl_pct >= self.profit_gate_pct and ml_confidence < self.profit_gate_min_conf:
            if self.discord and not self._profit_gate_warned:
                self.discord.send(
                    f"[{self.account_name}] PROFIT GATE ACTIVE",
                    f"Daily P&L: {daily_pnl_pct:.2%} (>{self.profit_gate_pct:.0%})\n"
                    f"Only trades with confidence >= {self.profit_gate_min_conf:.0%} allowed.",
                    "yellow")
                self._profit_gate_warned = True
            return False, (f"profit gate: daily +{daily_pnl_pct:.2%} > {self.profit_gate_pct:.0%}, "
                           f"conf {ml_confidence:.2f} < {self.profit_gate_min_conf}")

        return True, ""

    def check_dd_recovery(self, account_info) -> bool:
        """Check if in drawdown recovery mode. Returns True if lots should be halved."""
        if account_info is None:
            return False

        base = account_info.balance if account_info.balance > 0 else cfg.ACCOUNT_SIZE
        if base <= 0:
            return False
        dd_pct = (base - account_info.equity) / base
        if dd_pct >= self.dd_recovery_threshold:
            self._dd_recovery_mode = True
        elif dd_pct <= self.dd_recovery_exit:
            if self._dd_recovery_mode:
                self.logger.log('INFO', 'DrawdownGuard', 'DD_RECOVERY_OFF',
                                f'[{self.account_name}] Drawdown recovery OFF — '
                                f'equity recovered to {dd_pct:.2%} DD')
            self._dd_recovery_mode = False

        return self._dd_recovery_mode

    def reset_daily_flags(self):
        """Reset daily warning flags (call at start of new day)."""
        self._daily_loss_warned = False
        self._profit_lock_warned = False
        self._profit_gate_warned = False
