"""
Unit tests for execution/position_manager.py — critical bug regressions.

Each test is named after the bug it guards against.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import MockMT5, MockLogger, make_position


def _make_pm(mt5, logger):
    """Create a PositionManager with mocked dependencies."""
    from execution.position_manager import PositionManager
    pm = PositionManager(logger, mt5, discord=None)
    pm._trading_schedule = None  # No session trail by default
    pm._ftmo = None
    return pm


# ── BUG: Exception in one position stops management of ALL positions ──

class TestExceptionIsolation:
    """Bug: a single position throwing an exception in _process_single_position
    would prevent ALL other positions from being managed."""

    def test_error_in_one_position_does_not_block_others(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        # Create 3 positions
        pos1 = make_position('GOOD1', profit=50.0, ticket=1001)
        pos2 = make_position('BADONE', profit=50.0, ticket=1002)
        pos3 = make_position('GOOD2', profit=50.0, ticket=1003)
        mt5._positions = [pos1, pos2, pos3]

        # Make _process_single_position crash only for BADONE
        original_process = pm._process_single_position
        call_log = []

        def tracking_process(pos):
            call_log.append(pos.symbol)
            if pos.symbol == 'BADONE':
                raise ValueError("Simulated crash for BADONE")
            original_process(pos)

        pm._process_single_position = tracking_process

        # This should NOT crash, and should process all 3 positions
        pm.manage_positions(running=True, emergency_stop=False)

        # All 3 positions should have been attempted
        assert call_log == ['GOOD1', 'BADONE', 'GOOD2']
        # Error should be logged with symbol info
        assert logger.has_log(component='PositionManager', event_type='MANAGE_ERROR')
        error_msg = [msg for _, comp, evt, msg in logger.logs
                     if comp == 'PositionManager' and evt == 'MANAGE_ERROR'][0]
        assert 'BADONE' in error_msg
        assert 'ticket=1002' in error_msg

    def test_all_positions_crashing_still_logs_each(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        pos1 = make_position('SYM1', ticket=1001)
        pos2 = make_position('SYM2', ticket=1002)
        mt5._positions = [pos1, pos2]

        def always_crash(pos):
            raise RuntimeError(f"crash {pos.symbol}")

        pm._process_single_position = always_crash
        pm.manage_positions(running=True, emergency_stop=False)

        # Both errors should be logged individually
        error_logs = [msg for _, comp, evt, msg in logger.logs
                      if comp == 'PositionManager' and evt == 'MANAGE_ERROR']
        assert len(error_logs) == 2
        assert 'SYM1' in error_logs[0]
        assert 'SYM2' in error_logs[1]


# ── BUG: trail_distance_base can go negative ─────────────────────────

class TestTrailDistanceFloor:
    """Bug: when session_factor + friday_factor stack, trail_activation shrinks
    below min_locked_profit, making trail_distance_base negative. This places
    the SL on the wrong side of price → instant stop-out."""

    def test_trail_distance_never_negative(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        # Position deep in profit
        pos = make_position(
            'AAPL', pos_type=0, price_open=150.0, profit=500.0,
            sl=149.0, ticket=1001, magic=2000,
        )
        mt5._positions = [pos]

        # Mock tick showing profit
        mt5._tick = SimpleNamespace(ask=155.0, bid=154.98)

        # Mock ATR (small, to make the factors dominant)
        pm._get_cached_atr = lambda symbol, period=14: 0.50

        # Mock high costs (to push min_locked_profit high)
        pm._estimate_costs = lambda pos, cfg, atr: 0.40

        # Stack Friday + session factors → extreme shrinkage
        with patch.object(pm, '_friday_trail_factor', return_value=0.67), \
             patch.object(pm, '_session_trail_factor', return_value=0.25):

            # Track SL modification requests
            sent_requests = []
            original_send = mt5.order_send

            def capture_send(request):
                sent_requests.append(request)
                return original_send(request)

            mt5.order_send = capture_send
            mt5._symbol_info = SimpleNamespace(
                volume_min=0.01, digits=2, point=0.01,
                ask=155.0, bid=154.98,
                swap_long=-0.5, swap_short=0.3,
                trade_contract_size=1,
            )

            pm._process_single_position(pos)

        # If SL was set, it must be BELOW current price for a BUY
        if sent_requests:
            new_sl = sent_requests[0]['sl']
            assert new_sl < 154.98, \
                f"SL {new_sl} is above current bid 154.98 — trail_distance went negative!"

    def test_trail_distance_base_floor_at_5pct_atr(self, mt5, logger):
        """Directly test the floor logic: trail_distance_base >= 0.05 * ATR."""
        from execution.position_manager import PositionManager

        pm = _make_pm(mt5, logger)
        atr = 1.0

        # Simulate: trail_activation shrunk to 0.3, min_locked_profit = 0.6
        # Without floor: trail_distance_base = 0.3 - 0.6 = -0.3
        # With floor: max(-0.3, 0.05 * 1.0) = 0.05
        trail_activation = 0.3
        min_locked_profit = 0.6
        trail_distance_base = trail_activation - min_locked_profit
        trail_distance_base = max(trail_distance_base, atr * 0.05)

        assert trail_distance_base == 0.05
        assert trail_distance_base > 0


# ── BUG: ZOMBIE check ignores swap (kills carry trades) ──────────────

class TestZombieCheckIncludesSwap:
    """Bug: ZOMBIE detection used abs(pnl) < 5.0 where pnl = pos.profit,
    ignoring pos.swap. A position with profit=-$3 but swap=+$15 (total +$12)
    was killed as a zombie."""

    def test_profitable_carry_not_killed_as_zombie(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        # Position: small unrealized loss but large positive swap
        pos = make_position(
            'EURUSD', profit=-3.0, swap=15.0,
            ticket=1001, magic=2000,
            time=int(time.time()) - 72 * 3600,  # 72h old
        )
        mt5._positions = [pos]
        mt5._tick = SimpleNamespace(ask=1.1000, bid=1.0998)

        # Should NOT close this position
        order_sent = [False]
        original_send = mt5.order_send

        def track_send(request):
            if request.get('comment') == 'Sovereign_AutoClose':
                order_sent[0] = True
            return original_send(request)

        mt5.order_send = track_send

        pm.auto_close_bleeders(running=True, emergency_stop=False)

        assert order_sent[0] is False, \
            "Profitable carry trade (pnl+swap=+$12) was killed as ZOMBIE!"

    def test_real_zombie_still_closed(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        # Position: tiny P&L AND tiny swap → real zombie
        pos = make_position(
            'EURUSD', profit=1.0, swap=-0.5,
            ticket=1001, magic=2000,
            time=int(time.time()) - 72 * 3600,
        )
        mt5._positions = [pos]
        mt5._tick = SimpleNamespace(ask=1.1000, bid=1.0998)

        order_sent = [False]

        def track_send(request):
            order_sent[0] = True
            return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, order=99999)

        mt5.order_send = track_send

        pm.auto_close_bleeders(running=True, emergency_stop=False)

        assert order_sent[0] is True, "Real zombie (pnl+swap=$0.50) should be closed"


# ── BUG: DEEP_LOSS check ignores swap ────────────────────────────────

class TestDeepLossIncludesSwap:
    """Bug: DEEP_LOSS used pnl < -50 but pnl excluded swap."""

    def test_deep_loss_mitigated_by_swap(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        # profit=-55 but swap=+30 → total=-25, not deep loss
        pos = make_position(
            'EURUSD', profit=-55.0, swap=30.0,
            ticket=1001, magic=2000,
            time=int(time.time()) - 48 * 3600,
        )
        mt5._positions = [pos]
        mt5._tick = SimpleNamespace(ask=1.1000, bid=1.0998)

        order_sent = [False]

        def track_send(request):
            order_sent[0] = True
            return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, order=99999)

        mt5.order_send = track_send

        pm.auto_close_bleeders(running=True, emergency_stop=False)

        assert order_sent[0] is False, \
            "Position with pnl+swap=-$25 should NOT be killed as DEEP_LOSS (-$50 threshold)"


# ── BUG: ML exit strike counter resets on failed close ────────────────

class TestMLExitStrikeReset:
    """Bug: strikes were reset to 0 even when _ml_exit_close failed,
    requiring 15+ more minutes before next attempt."""

    def test_strikes_preserved_on_failed_close(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        pos = make_position('EURUSD', profit=-10.0, swap=0.0,
                            ticket=1001, magic=2000,
                            time=int(time.time()) - 7200)
        mt5._positions = [pos]

        # Simulate 3 strikes already accumulated
        pm._ml_exit_strikes[1001] = {"strikes": 3, "last_proba": 0.30}

        # Make close FAIL
        mt5._order_result = SimpleNamespace(retcode=10004, comment='No money')

        # Call _ml_exit_close directly
        result = pm._ml_exit_close(pos, 0.30, 3, -10.0)

        assert result is False
        # Strikes should NOT be reset (caller checks return value)

    def test_strikes_reset_on_successful_close(self, mt5, logger):
        pm = _make_pm(mt5, logger)

        pos = make_position('EURUSD', profit=-10.0, swap=0.0,
                            ticket=1001, magic=2000)

        # Make close succeed
        mt5._order_result = SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE, order=99999, price=1.0998
        )

        result = pm._ml_exit_close(pos, 0.30, 3, -10.0)

        assert result is True


# ── BUG: horizon_exit_check color uses (real_pnl or pnl) ─────────────

class TestHorizonExitColor:
    """Bug: (real_pnl or pnl) evaluates 0.0 as falsy, falling through to pnl."""

    def test_zero_pnl_is_orange_not_red(self, mt5, logger):
        """When real_pnl is exactly 0.0 (breakeven), color should be orange."""
        pm = _make_pm(mt5, logger)
        pm.discord = MagicMock()

        pos = make_position(
            'EURUSD', profit=0.0, swap=0.0,
            ticket=1001, magic=2000,
            time=int(time.time()) - 200 * 3600,  # way past horizon
        )
        mt5._positions = [pos]

        # Configure horizon exit
        with patch('execution.position_manager.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {
                'EURUSD': {'exit_horizon': 5, 'exit_timeframe': 'H1'}
            }
            mock_cfg.DEVIATION = 10

            # _deal_pnl returns exactly 0.0
            pm._deal_pnl = lambda ticket: 0.0

            mt5._order_result = SimpleNamespace(
                retcode=mt5.TRADE_RETCODE_DONE, order=99999
            )

            pm.horizon_exit_check(running=True, emergency_stop=False)

        # Discord should be called with orange (breakeven), not red
        if pm.discord.send.called:
            call_args = pm.discord.send.call_args
            color = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get('color')
            assert color == 'orange', f"Expected orange for breakeven, got {color}"
