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
                 daily_start_balance: float = 0,
                 profit_trail_activate_pct: float = 0.0205,
                 profit_trail_buffer_pct: float = 0.0005):
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
        self.profit_trail_activate_pct = profit_trail_activate_pct
        self.profit_trail_buffer_pct = profit_trail_buffer_pct
        self._last_reset_date = datetime.now(timezone.utc).date()
        self._dd_recovery_mode = False
        self._daily_loss_warned = False
        self._profit_lock_warned = False
        self._profit_gate_warned = False
        self._profit_trail_active = False
        self._needs_daily_loss_protect = False
        self._profit_trail_hwm = 0.0
        self._profit_trail_floor = 0.0
        self._profit_trail_warned = False

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
            if not self._daily_loss_warned:
                self._daily_loss_warned = True
                self._needs_daily_loss_protect = True
                if self.discord:
                    self.discord.send(f"[{self.account_name}] DAILY LOSS LIMIT",
                                      f"PnL: {daily_pnl_pct:.2%}\nProtecting positions + no new trades.",
                                      "red")
            return False, f"daily PnL {daily_pnl_pct:.2%} hit -{self.daily_loss_pct:.1%} limit"

        # Profit lock disabled — trailing stop on each position handles exits
        # if daily_pnl_pct >= self.profit_lock_pct:
        #     return False, "profit locked"

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

    def check_profit_trail(self, account_info) -> tuple[bool, str]:
        """At profit_lock_pct (5%): move all SL to breakeven, block new trades.

        No close-all — spread fluctuations cause false triggers.
        The trailing stop on each position handles exits.

        Returns (should_move_to_be, reason). Never returns close-all.
        """
        if account_info is None or self.profit_lock_pct <= 0:
            return False, ""

        base = self.daily_start_balance if self.daily_start_balance > 0 else account_info.balance
        if base <= 0:
            return False, ""

        daily_pnl_pct = (account_info.equity - base) / base

        # At 5%+ daily P/L: move all SL to breakeven (once)
        if daily_pnl_pct >= self.profit_lock_pct and not self._profit_trail_active:
            self._profit_trail_active = True
            self.logger.log('INFO', 'DrawdownGuard', 'PROFIT_LOCK_BREAKEVEN',
                            f'[{self.account_name}] Daily P&L {daily_pnl_pct:.2%} hit '
                            f'{self.profit_lock_pct:.0%} — moving all SL to breakeven')
            if self.discord:
                self.discord.send(
                    f"[{self.account_name}] PROFIT LOCK — ALL SL TO BREAKEVEN",
                    f"Daily P&L: {daily_pnl_pct:.2%}\n"
                    f"All positions moved to breakeven.\n"
                    f"No new trades. Trailing stop manages exits.",
                    "green")
            return True, "profit lock: move all SL to breakeven"

        return False, ""

    def check_dd_recovery(self, account_info) -> bool:
        """Check if in drawdown recovery mode. Returns True if lots should be halved.

        Uses DAILY P/L (not floating DD) so it works even after trades are closed.
        Triggers at dd_recovery_threshold (1.5%) daily loss.
        Exits at dd_recovery_exit (0.5%) daily loss (recovery).
        """
        if account_info is None:
            return False

        # Use daily P/L instead of balance vs equity
        base = self.daily_start_balance if self.daily_start_balance > 0 else account_info.balance
        if base <= 0:
            return False
        daily_pnl_pct = (account_info.equity - base) / base  # negative = loss

        if daily_pnl_pct <= -self.dd_recovery_threshold:
            if not self._dd_recovery_mode:
                self.logger.log('WARNING', 'DrawdownGuard', 'DD_RECOVERY_ON',
                                f'[{self.account_name}] Daily loss {daily_pnl_pct:.2%} hit '
                                f'-{self.dd_recovery_threshold:.1%} — halving lot sizes')
                if self.discord:
                    self.discord.send(
                        f"[{self.account_name}] DD RECOVERY MODE",
                        f"Daily loss: {daily_pnl_pct:.2%}\nLot sizes halved until recovery.",
                        "orange")
            self._dd_recovery_mode = True
        elif daily_pnl_pct >= -self.dd_recovery_exit:
            if self._dd_recovery_mode:
                self.logger.log('INFO', 'DrawdownGuard', 'DD_RECOVERY_OFF',
                                f'[{self.account_name}] Daily loss recovered to {daily_pnl_pct:.2%} '
                                f'— normal lot sizes resumed')
            self._dd_recovery_mode = False

        return self._dd_recovery_mode

    def reset_daily_flags(self):
        """Reset daily warning flags (call at start of new day)."""
        self._daily_loss_warned = False
        self._profit_lock_warned = False
        self._profit_gate_warned = False
        self._profit_trail_active = False
        self._needs_daily_loss_protect = False
        self._profit_trail_hwm = 0.0
        self._profit_trail_floor = 0.0
        self._profit_trail_warned = False
