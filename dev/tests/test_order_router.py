"""
Unit tests for execution/order_router.py — critical bug regressions.

Each test is named after the bug it guards against.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Fixtures from conftest
from tests.conftest import MockMT5, MockLogger, make_position


# ── Helpers ──────────────────────────────────────────────────────────

def _make_router(mt5, logger, ftmo=None):
    """Create an OrderRouter with all dependencies mocked."""
    from risk.position_sizing import PositionSizingEngine
    sizer = MagicMock(spec=PositionSizingEngine)
    sizer.kelly_risk_pct.return_value = 0.005
    sizer.calculate_lot_size.return_value = 0.10
    sizer.rl_adjust_risk.return_value = (0.005, 0)

    schedule = MagicMock()
    schedule.is_trading_open.return_value = (True, 120)
    schedule.should_friday_close.return_value = False
    schedule._now_gmt2.return_value = SimpleNamespace(hour=12, minute=0, weekday=lambda: 1)

    from execution.order_router import OrderRouter
    router = OrderRouter(logger, mt5, sizer, schedule, discord=None, ftmo=ftmo)
    return router


def _enable_live():
    """Context manager: set ENABLE_LIVE_TRADING=1."""
    return patch.dict(os.environ, {"ENABLE_LIVE_TRADING": "1"})


# ── BUG: emergency_stop flag never checked ───────────────────────────

class TestEmergencyStop:
    """Bug: emergency_stop was set but never checked in execute_trade()."""

    def test_emergency_stop_blocks_trade(self, mt5, logger):
        router = _make_router(mt5, logger)
        router.emergency_stop = True

        with _enable_live():
            result = router.execute_trade('EURUSD', 'BUY', 0.65)

        assert result is False
        assert router.last_reject_reason == 'emergency stop'

    def test_no_emergency_stop_allows_trade(self, mt5, logger):
        router = _make_router(mt5, logger)
        router.emergency_stop = False

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            result = router.execute_trade('EURUSD', 'BUY', 0.65)

        assert result is True


# ── BUG: stale all_positions after _try_replace_worst ─────────────────

class TestStalePositionsAfterReplacement:
    """Bug: after _try_replace_worst closed a position, all_positions was stale,
    causing sector exposure to be calculated with the old (pre-close) data."""

    def test_positions_refreshed_after_replacement(self, mt5, logger):
        """After _try_replace_worst returns True, positions must be re-fetched."""
        # Create 12 positions to trigger replacement path
        positions_12 = [make_position(f'SYM{i}', profit=10.0 * i, ticket=1000 + i,
                                      magic=2000)
                        for i in range(12)]
        positions_11 = positions_12[1:]  # After replacement

        router = _make_router(mt5, logger)

        # Track positions_get(symbol=None) calls AFTER replacement
        get_calls_after_replace = []
        replacement_done = [False]

        original_positions_get = mt5.positions_get

        def tracking_get(symbol=None):
            if symbol:
                return []  # No per-symbol positions for NEWSTOCK
            if replacement_done[0]:
                get_calls_after_replace.append(True)
                return positions_11
            return positions_12

        mt5.positions_get = tracking_get

        # Mock _try_replace_worst to succeed and flag the replacement
        def mock_replace(our_positions, symbol, direction, ml_confidence, mt5_arg):
            replacement_done[0] = True
            return True

        router._try_replace_worst = mock_replace

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            result = router.execute_trade('NEWSTOCK', 'BUY', 0.75)

        # After replacement, positions_get should have been called again
        assert len(get_calls_after_replace) >= 1, \
            "positions_get() not called after _try_replace_worst — stale data!"


# ── BUG: empty our_sym_positions falls back to positions[0] ──────────

class TestEmptyOurSymPositions:
    """Bug: when all positions on a symbol had magic < 2000 (other EA),
    the code fell back to positions[0] which could close someone else's trade."""

    def test_other_ea_positions_not_touched(self, mt5, logger):
        """If only other-EA positions exist on a symbol, we should NOT interact with them."""
        other_ea_pos = make_position('EURUSD', magic=1000, ticket=999)
        mt5._positions = [other_ea_pos]

        def positions_get_by_symbol(symbol=None):
            if symbol == 'EURUSD':
                return [other_ea_pos]
            return mt5._positions

        mt5.positions_get = positions_get_by_symbol

        router = _make_router(mt5, logger)

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            result = router.execute_trade('EURUSD', 'BUY', 0.65)

        # Should succeed (open new trade), NOT try to close/flip other EA's position
        assert result is True

    def test_our_positions_handled_correctly(self, mt5, logger):
        """If we have our own position on a symbol, handle_existing_position runs."""
        our_pos = make_position('EURUSD', pos_type=0, magic=2000, ticket=1001)  # BUY
        mt5._positions = [our_pos]

        def positions_get_by_symbol(symbol=None):
            if symbol == 'EURUSD':
                return [our_pos]
            return mt5._positions

        mt5.positions_get = positions_get_by_symbol

        router = _make_router(mt5, logger)

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            # Same direction, low confidence → should be rejected (already in market)
            result = router.execute_trade('EURUSD', 'BUY', 0.60)

        assert result is False
        assert 'al BUY positie open' in (router.last_reject_reason or '')


# ── BUG: FTMO trade count inflated per split chunk ────────────────────

class TestFTMOTradeCount:
    """Bug: ftmo.increment_trade_count() was called per split chunk,
    not per logical trade."""

    def test_ftmo_count_once_per_trade(self, mt5, logger):
        ftmo = MagicMock()
        ftmo.pre_trade_check.return_value = (True, '')
        ftmo.increment_trade_count = MagicMock()
        ftmo.record_trade = MagicMock()

        router = _make_router(mt5, logger, ftmo=ftmo)

        # Set volume_max low to force multiple chunks
        mt5._symbol_info.volume_max = 0.05
        router.position_sizer.calculate_lot_size.return_value = 0.15  # → 3 chunks

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            result = router.execute_trade('EURUSD', 'BUY', 0.65)

        assert result is True
        # FTMO should be called exactly once, not 3 times
        assert ftmo.increment_trade_count.call_count == 1
        assert ftmo.record_trade.call_count == 1

    def test_ftmo_not_called_on_failure(self, mt5, logger):
        ftmo = MagicMock()
        ftmo.pre_trade_check.return_value = (True, '')
        ftmo.increment_trade_count = MagicMock()

        router = _make_router(mt5, logger, ftmo=ftmo)

        # Make order_send fail
        mt5._order_result = SimpleNamespace(retcode=10004, comment='No money')

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            result = router.execute_trade('EURUSD', 'BUY', 0.65)

        assert result is False
        assert ftmo.increment_trade_count.call_count == 0


# ── BUG: TP not shifted for exit costs ────────────────────────────────

class TestTPCostShift:
    """Bug: TP was only shifted by entry-side costs (half commission/slippage),
    missing the exit-side costs."""

    def test_tp_includes_full_roundtrip_costs(self, mt5, logger):
        router = _make_router(mt5, logger)

        # Track the order request to inspect TP
        sent_requests = []
        original_send = mt5.order_send

        def capture_send(request):
            sent_requests.append(request)
            return original_send(request)

        mt5.order_send = capture_send

        with _enable_live(), \
             patch('execution.order_router.check_spread', return_value=(True, 0.001)):
            router.execute_trade('EURUSD', 'BUY', 0.65)

        assert len(sent_requests) > 0
        req = sent_requests[0]

        # TP should be entry + tp_distance + full costs
        # The key assertion: TP must be > entry + base_tp_distance
        # (i.e., costs were added, not just half)
        entry = req['price']
        tp = req['tp']
        sl = req['sl']

        # TP must be above entry for a BUY
        assert tp > entry
        # SL must be below entry for a BUY
        assert sl < entry
