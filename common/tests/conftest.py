"""Shared fixtures for all tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Mock MT5 constants ──────────────────────────────────────────────
class MockMT5:
    """Fake MT5 bridge with standard constants and controllable returns."""

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 2
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388

    def __init__(self):
        self._positions = []
        self._account = SimpleNamespace(
            balance=100_000, equity=100_000, margin=5000,
            margin_free=95_000
        )
        self._order_result = SimpleNamespace(
            retcode=self.TRADE_RETCODE_DONE, order=12345, price=0
        )
        self._symbol_info = SimpleNamespace(
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            digits=5, point=0.00001,
            trade_contract_size=100_000,
            ask=1.1000, bid=1.0998,
            swap_long=-0.5, swap_short=0.3,
        )
        self._tick = SimpleNamespace(ask=1.1000, bid=1.0998)

    def positions_get(self, symbol=None):
        if symbol:
            return [p for p in self._positions if p.symbol == symbol] or None
        return self._positions if self._positions else []

    def account_info(self):
        return self._account

    def order_send(self, request):
        return self._order_result

    def symbol_info(self, symbol):
        return self._symbol_info

    def symbol_info_tick(self, symbol):
        return self._tick

    def copy_rates_from_pos(self, symbol, tf, start, count):
        # Return fake H1 bars for ATR calculation
        bars = []
        base = 1.1000
        for i in range(count):
            bars.append({
                'time': 1700000000 + i * 3600,
                'open': base, 'high': base + 0.002,
                'low': base - 0.002, 'close': base + 0.001,
                'tick_volume': 1000,
            })
        return bars

    def history_deals_get(self, start, end, position=None):
        return []


def make_position(symbol='EURUSD', pos_type=0, volume=1.0, profit=0.0,
                  swap=0.0, magic=2000, ticket=1001, sl=0.0, tp=0.0,
                  price_open=1.1000, price_current=1.1010, time=1700000000):
    """Create a fake MT5 position."""
    return SimpleNamespace(
        symbol=symbol, type=pos_type, volume=volume, profit=profit,
        swap=swap, magic=magic, ticket=ticket, sl=sl, tp=tp,
        price_open=price_open, price_current=price_current, time=time,
    )


class MockLogger:
    """Captures log calls for assertion."""

    def __init__(self):
        self.logs = []
        self.trades = []

    def log(self, level, component, event_type, message='', data=None):
        self.logs.append((level, component, event_type, message))

    def log_trade(self, *args, **kwargs):
        self.trades.append((args, kwargs))

    def has_log(self, component=None, event_type=None):
        for _, comp, evt, _ in self.logs:
            if (component is None or comp == component) and \
               (event_type is None or evt == event_type):
                return True
        return False


@pytest.fixture
def mt5():
    return MockMT5()


@pytest.fixture
def logger():
    return MockLogger()
