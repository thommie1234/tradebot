"""Tests for multi-account support — AccountContext, parameterized compliance."""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ftmo_compliance import FTMOCompliance
from risk.drawdown_guard import DrawdownGuard
from execution.account_context import AccountContext, load_accounts


# ── Parameterized Compliance ──────────────────────────────────────────


class TestParameterizedCompliance:
    """FTMOCompliance should use per-account limits, not hardcoded globals."""

    def test_custom_daily_loss_limit(self):
        """3% daily limit should trigger at 3%, not at default 5%."""
        ftmo = FTMOCompliance(100000, account_name="OneStep",
                              max_daily_loss_pct=0.03)
        # 3.5% loss — should be blocked with 3% limit
        ok, reason = ftmo.check_daily_loss(96500)
        assert not ok
        assert "3%" in reason or "OneStep" in reason

    def test_default_daily_loss_limit(self):
        """Default limit (5%) should still work when no param given."""
        ftmo = FTMOCompliance(100000)
        # 4% loss — should be allowed with default 5% limit
        ok, reason = ftmo.check_daily_loss(96000)
        assert ok

    def test_custom_total_dd_limit(self):
        """6% max DD should trigger at 6%, not at default 10%."""
        ftmo = FTMOCompliance(50000, account_name="FXIFY",
                              max_total_dd_pct=0.06)
        # 7% DD = $3,500 loss on $50k
        must_close, reason = ftmo.check_total_dd(46500)
        assert must_close
        assert "FXIFY" in reason

    def test_static_dd_uses_initial_balance(self):
        """Static DD type always measures from initial balance."""
        ftmo = FTMOCompliance(50000, account_name="BrightFunded",
                              max_total_dd_pct=0.10, dd_type="static")
        # Even if equity went up to 55k and back to 46k:
        # Static DD = (50000 - 46000) / 50000 = 8% — not triggered yet
        ftmo.update_high_water_mark(55000)
        must_close, _ = ftmo.check_total_dd(46000)
        assert not must_close  # 8% < 10%

        # 11% DD — should trigger
        must_close, reason = ftmo.check_total_dd(44000)
        assert must_close

    def test_account_name_in_messages(self):
        """Account name should appear in log messages."""
        ftmo = FTMOCompliance(100000, account_name="MyAccount",
                              max_daily_loss_pct=0.05)
        ok, reason = ftmo.check_daily_loss(94000)
        assert "MyAccount" in reason


# ── Parameterized DrawdownGuard ───────────────────────────────────────


class TestParameterizedDrawdownGuard:
    """DrawdownGuard should use per-account thresholds."""

    def test_custom_daily_loss_threshold(self):
        """Custom -2% limit should trigger earlier than default -3.5%."""
        from types import SimpleNamespace
        dg = DrawdownGuard(None, daily_loss_pct=0.02)
        acct = SimpleNamespace(balance=100000, equity=97500)
        ok, reason = dg.check_daily_limits(acct)
        assert not ok
        assert "2.0%" in reason or "-2.5%" in reason

    def test_custom_profit_lock(self):
        """Custom +2% profit lock."""
        from types import SimpleNamespace
        dg = DrawdownGuard(None, profit_lock_pct=0.02)
        acct = SimpleNamespace(balance=100000, equity=102500)
        ok, reason = dg.check_daily_limits(acct)
        assert not ok
        assert "locking" in reason.lower()

    def test_dd_recovery_custom_threshold(self):
        """Custom 3% recovery threshold (instead of default 4%)."""
        from types import SimpleNamespace

        class FakeLogger:
            def log(self, *a, **kw): pass

        dg = DrawdownGuard(FakeLogger(), dd_recovery_threshold=0.03, dd_recovery_exit=0.005)
        acct = SimpleNamespace(balance=100000, equity=96500)  # 3.5% DD
        assert dg.check_dd_recovery(acct) is True  # Should be in recovery


# ── Account Loading ───────────────────────────────────────────────────


class TestAccountLoading:
    """Verify accounts.yaml loading produces correct AccountContext instances."""

    def test_load_from_yaml(self):
        """Should load 2 accounts from accounts.yaml."""
        accounts = load_accounts()
        assert "ftmo_100k" in accounts
        assert "bright_50k" in accounts

    def test_ftmo_account_enabled(self):
        accounts = load_accounts()
        assert accounts["ftmo_100k"].enabled is True

    def test_bright_account_disabled(self):
        accounts = load_accounts()
        assert accounts["bright_50k"].enabled is False

    def test_ftmo_port(self):
        accounts = load_accounts()
        assert accounts["ftmo_100k"].bridge_port == 5056

    def test_bright_port(self):
        accounts = load_accounts()
        assert accounts["bright_50k"].bridge_port == 5057

    def test_dd_type_trailing(self):
        accounts = load_accounts()
        assert accounts["ftmo_100k"].account_cfg["dd_type"] == "trailing"

    def test_dd_type_static(self):
        accounts = load_accounts()
        assert accounts["bright_50k"].account_cfg["dd_type"] == "static"

    def test_risk_scale(self):
        accounts = load_accounts()
        assert accounts["ftmo_100k"].risk_scale == 1.0
        assert accounts["bright_50k"].risk_scale == 1.0


# ── AccountContext Isolation ──────────────────────────────────────────


class TestAccountContextIsolation:
    """Each AccountContext should be fully independent."""

    def test_separate_instances(self):
        """Two accounts should have separate compliance objects."""
        a1 = AccountContext("a1", {"name": "A1", "enabled": True, "bridge_port": 5055,
                                   "account_size": 100000})
        a2 = AccountContext("a2", {"name": "A2", "enabled": True, "bridge_port": 5056,
                                   "account_size": 50000})
        assert a1.account_id != a2.account_id
        assert a1.bridge_port != a2.bridge_port
        assert a1.account_size != a2.account_size

    def test_safe_mode_per_account(self):
        """Safe mode on one account shouldn't affect another."""
        a1 = AccountContext("a1", {"name": "A1", "enabled": True, "bridge_port": 5055})
        a2 = AccountContext("a2", {"name": "A2", "enabled": True, "bridge_port": 5056})
        a1.safe_mode = True
        assert a2.safe_mode is False

    def test_disabled_account_no_execute(self):
        """Disabled account should not execute trades."""
        a = AccountContext("x", {"name": "X", "enabled": False, "bridge_port": 9999})
        result = a.execute_trade("EURUSD", "BUY", 0.85)
        assert result is False
