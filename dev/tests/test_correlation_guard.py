"""
Unit tests for risk/correlation_guard.py — critical bug regressions.

Tests the set-based counting fix (was per-position, now per-unique-symbol)
and symbol name matching fixes.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import MockMT5, MockLogger, make_position


def _make_guard(logger):
    from risk.correlation_guard import CorrelationGuard
    return CorrelationGuard(logger)


# ── BUG: Index counting per-position instead of per-unique-symbol ─────

class TestIndexUniqueSymbolCounting:
    """Bug: two positions on US30.cash (scaling in) counted as 2, blocking
    all other US index trades despite only 1 unique symbol exposure."""

    def test_two_positions_same_index_count_as_one(self, logger):
        guard = _make_guard(logger)

        # Two BUY positions on US30.cash (scaled in)
        positions = [
            make_position('US30.cash', pos_type=0, ticket=1001),
            make_position('US30.cash', pos_type=0, ticket=1002),
        ]

        # Should allow another US index (US100.cash) because unique count = 1
        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'US100.cash': {'asset_class': 'index'}}
            result = guard.check_correlation('US100.cash', 'BUY', positions, None)

        assert result is True, \
            "Two positions on same index should count as 1 unique symbol"

    def test_two_different_indices_counted_correctly(self, logger):
        guard = _make_guard(logger)

        # Two different US indices
        positions = [
            make_position('US30.cash', pos_type=0, ticket=1001),
            make_position('US500.cash', pos_type=0, ticket=1002),
        ]

        # Should block a third US index
        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'US100.cash': {'asset_class': 'index'}}
            result = guard.check_correlation('US100.cash', 'BUY', positions, None)

        assert result is False, \
            "Two different US indices should block a third (MAX_INDEX_SAME_DIR=2)"

    def test_opposite_direction_not_counted(self, logger):
        guard = _make_guard(logger)

        # One BUY and one SELL on US indices
        positions = [
            make_position('US30.cash', pos_type=0, ticket=1001),   # BUY
            make_position('US500.cash', pos_type=1, ticket=1002),  # SELL
        ]

        # BUY on another US index should be allowed (only 1 BUY counted)
        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'US100.cash': {'asset_class': 'index'}}
            result = guard.check_correlation('US100.cash', 'BUY', positions, None)

        assert result is True


# ── BUG: Equity counting per-position instead of per-unique-symbol ────

class TestEquityUniqueSymbolCounting:
    """Bug: same as index — two positions on AAPL counted as 2."""

    def test_two_positions_same_equity_count_as_one(self, logger):
        guard = _make_guard(logger)

        # Two BUY positions on AAPL (pyramid)
        positions = [
            make_position('AAPL', pos_type=0, ticket=1001),
            make_position('AAPL', pos_type=0, ticket=1002),
        ]

        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'NVDA': {'asset_class': 'equity'}}
            result = guard.check_correlation('NVDA', 'BUY', positions, None)

        assert result is True, \
            "Two positions on same equity should count as 1 unique symbol"

    def test_four_different_equities_blocks_fifth(self, logger):
        guard = _make_guard(logger)

        positions = [
            make_position('AAPL', pos_type=0, ticket=1001),
            make_position('GOOG', pos_type=0, ticket=1002),
            make_position('AMZN', pos_type=0, ticket=1003),
            make_position('TSLA', pos_type=0, ticket=1004),
        ]

        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'NVDA': {'asset_class': 'equity'}}
            result = guard.check_correlation('NVDA', 'BUY', positions, None)

        assert result is False, \
            "4 unique US equities should block a 5th (MAX_EQUITY_SAME_DIR=4)"


# ── BUG: Commodity counting per-position instead of per-unique-symbol ─

class TestCommodityUniqueSymbolCounting:
    """Bug: two positions on XAUUSD counted as 2, blocking XAGUSD."""

    def test_two_positions_same_metal_count_as_one(self, logger):
        guard = _make_guard(logger)

        positions = [
            make_position('XAUUSD', pos_type=0, ticket=1001),
            make_position('XAUUSD', pos_type=0, ticket=1002),
        ]

        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'XAGUSD': {'asset_class': 'commodity'}}
            result = guard.check_correlation('XAGUSD', 'BUY', positions, None)

        assert result is True, \
            "Two positions on XAUUSD should count as 1 unique metal"

    def test_two_different_metals_blocks_third(self, logger):
        guard = _make_guard(logger)

        positions = [
            make_position('XAUUSD', pos_type=0, ticket=1001),
            make_position('XAGUSD', pos_type=0, ticket=1002),
        ]

        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'GOLDUSD': {'asset_class': 'commodity'}}
            result = guard.check_correlation('GOLDUSD', 'BUY', positions, None)

        assert result is False, \
            "Two different metals should block a third (MAX_METAL_SAME_DIR=2)"

    def test_two_positions_same_oil_count_as_one(self, logger):
        guard = _make_guard(logger)

        positions = [
            make_position('USOIL.cash', pos_type=0, ticket=1001),
            make_position('USOIL.cash', pos_type=0, ticket=1002),
        ]

        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'UKOIL.cash': {'asset_class': 'commodity'}}
            result = guard.check_correlation('UKOIL.cash', 'BUY', positions, None)

        assert result is True, \
            "Two positions on USOIL should count as 1 unique oil"


# ── BUG: DOGUSD typo → should be DOGEUSD ─────────────────────────────

class TestCryptoSymbolNames:
    """Bug: CRYPTO_SYMBOLS had 'DOGUSD' but broker uses 'DOGEUSD'."""

    def test_dogeusd_recognized_as_crypto(self, logger):
        from risk.correlation_guard import CRYPTO_SYMBOLS
        assert 'DOGEUSD' in CRYPTO_SYMBOLS, \
            "DOGEUSD must be in CRYPTO_SYMBOLS (was typo'd as DOGUSD)"

    def test_dogusd_not_in_crypto(self, logger):
        from risk.correlation_guard import CRYPTO_SYMBOLS
        assert 'DOGUSD' not in CRYPTO_SYMBOLS, \
            "DOGUSD should not be in CRYPTO_SYMBOLS (typo)"

    def test_dogeusd_fallback_detection(self, logger):
        """Even without config, DOGEUSD should be caught by CRYPTO_SYMBOLS fallback."""
        guard = _make_guard(logger)

        positions = [
            make_position('BTCUSD', pos_type=0, ticket=1001),
            make_position('ETHUSD', pos_type=0, ticket=1002),
            make_position('SOLUSD', pos_type=0, ticket=1003),
        ]

        # No config for DOGEUSD — relies on CRYPTO_SYMBOLS fallback
        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {}  # No config at all
            result = guard.check_correlation('DOGEUSD', 'BUY', positions, None)

        assert result is False, \
            "DOGEUSD should be detected as crypto via fallback and blocked at MAX=3"


# ── BUG: Oil symbols bypass correlation guard without config ──────────

class TestOilFallbackDetection:
    """Bug: oil symbols without asset_class config fell through all checks
    and returned True (no correlation limit)."""

    def test_oil_without_config_still_caught(self, logger):
        guard = _make_guard(logger)

        positions = [
            make_position('USOIL.cash', pos_type=0, ticket=1001),
            make_position('UKOIL.cash', pos_type=0, ticket=1002),
        ]

        # No config → must be caught by OIL_SYMBOLS fallback
        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {}
            result = guard.check_correlation('WTIUSD', 'BUY', positions, None)

        assert result is False, \
            "Oil symbols without config should still be caught by OIL_SYMBOLS fallback"


# ── BUG: INDEX_REGIONS symbol mismatches ──────────────────────────────

class TestIndexRegionSymbols:
    """Bug: FR40 didn't match broker's FRA40.cash, AU200 didn't match AUS200.cash."""

    def test_fra40_in_eu_region(self):
        from risk.correlation_guard import INDEX_REGIONS
        assert 'FRA40' in INDEX_REGIONS['eu'], \
            "FRA40 must be in eu INDEX_REGIONS (broker sends FRA40.cash → normalised to FRA40)"

    def test_aus200_in_apac_region(self):
        from risk.correlation_guard import INDEX_REGIONS
        assert 'AUS200' in INDEX_REGIONS['apac'], \
            "AUS200 must be in apac INDEX_REGIONS (broker sends AUS200.cash → normalised to AUS200)"

    def test_normalise_strips_cash(self):
        from risk.correlation_guard import _normalise
        assert _normalise('US30.cash') == 'US30'
        assert _normalise('GER40.cash') == 'GER40'
        assert _normalise('FRA40.cash') == 'FRA40'
        assert _normalise('AUS200.cash') == 'AUS200'
        assert _normalise('EURUSD') == 'EURUSD'


# ── BUG: Crypto counting was already fixed (regression guard) ─────────

class TestCryptoCountingRegression:
    """Guard: crypto already uses set-based counting. Ensure it stays correct."""

    def test_two_positions_same_crypto_count_as_one(self, logger):
        guard = _make_guard(logger)

        positions = [
            make_position('BTCUSD', pos_type=0, ticket=1001),
            make_position('BTCUSD', pos_type=0, ticket=1002),
        ]

        with patch('risk.correlation_guard.cfg') as mock_cfg:
            mock_cfg.SYMBOLS = {'ETHUSD': {'asset_class': 'crypto'}}
            result = guard.check_correlation('ETHUSD', 'BUY', positions, None)

        assert result is True, \
            "Two positions on BTCUSD should count as 1 unique crypto"
