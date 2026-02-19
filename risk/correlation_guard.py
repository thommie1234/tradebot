"""
Correlation guard — multi-asset correlation-aware position limits (Gebod 45).

Tracks currency exposure (forex decomposition), crypto group, metals,
regional index groups, and equity sectors.
"""
from __future__ import annotations

from config.loader import cfg


# ── Asset group definitions ──────────────────────────────────────────
CRYPTO_SYMBOLS = {
    'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BCHUSD', 'ADAUSD',
    'SOLUSD', 'DOGUSD', 'DOTUSD', 'BNBUSD',
    # Broker-specific suffixes handled by _normalise()
}

METAL_SYMBOLS = {'XAUUSD', 'XAGUSD', 'GOLDUSD', 'SILVERUSD'}

INDEX_REGIONS: dict[str, set[str]] = {
    'us':   {'US30', 'US500', 'USTEC', 'US2000', 'SP500', 'NAS100', 'DJ30'},
    'eu':   {'DE40', 'FR40', 'EU50', 'STOXX50', 'DAX40', 'CAC40'},
    'uk':   {'UK100', 'FTSE100'},
    'apac': {'JP225', 'AU200', 'HK50', 'CN50', 'NI225'},
}

EQUITY_SECTORS: dict[str, set[str]] = {
    'us_equity': {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'},
    'eu_equity': {'SAP', 'ASML', 'LVMH', 'SIE', 'BMW'},
}

# ── Limits ───────────────────────────────────────────────────────────
MAX_CURRENCY_SAME_DIR = 3    # forex: max 3 positions per currency in same direction
MAX_CRYPTO_SAME_DIR = 4      # all crypto treated as one correlated group
MAX_METAL_SAME_DIR = 2       # gold + silver correlated
MAX_INDEX_SAME_DIR = 2       # per regional group
MAX_EQUITY_SAME_DIR = 3      # per equity sector


def _normalise(symbol: str) -> str:
    """Strip common broker suffixes (.r, .i, _SB, etc.)."""
    for suffix in ('.r', '.i', '.e', '_SB', '.ecn', '.pro'):
        if symbol.lower().endswith(suffix.lower()):
            symbol = symbol[:len(symbol) - len(suffix)]
    return symbol.upper()


def _pos_direction(pos) -> str:
    return 'BUY' if pos.type == 0 else 'SELL'


class CorrelationGuard:
    """Enforces correlation caps across asset classes."""

    # Legacy constant for backward compatibility
    USD_SAME_DIR_MAX = MAX_CURRENCY_SAME_DIR

    def __init__(self, logger):
        self.logger = logger

    # ── Public API ───────────────────────────────────────────────────

    def check_correlation(self, symbol: str, direction: str,
                          our_positions, mt5) -> bool:
        """Returns True if trade is allowed, False if blocked by any correlation limit."""
        sym = _normalise(symbol)
        sym_cfg = cfg.SYMBOLS.get(symbol, {})
        asset_class = sym_cfg.get('asset_class', '')

        if asset_class == 'forex':
            return self._check_forex(sym, direction, our_positions)
        if asset_class == 'crypto' or sym in CRYPTO_SYMBOLS:
            return self._check_crypto(sym, direction, our_positions)
        if asset_class == 'commodity' or sym in METAL_SYMBOLS:
            return self._check_commodity(sym, direction, our_positions)
        if asset_class == 'index':
            return self._check_index(sym, direction, our_positions)
        if asset_class == 'equity':
            return self._check_equity(sym, direction, our_positions)

        return True

    def check_usd_correlation(self, symbol: str, direction: str,
                               our_positions, mt5) -> bool:
        """Backward-compatible alias — delegates to check_correlation()."""
        return self.check_correlation(symbol, direction, our_positions, mt5)

    # ── Forex: currency decomposition ────────────────────────────────

    def _check_forex(self, sym: str, direction: str, positions) -> bool:
        """
        Decompose forex pairs into base/quote currency exposure.
        BUY EURUSD = +EUR, -USD.  SELL EURUSD = -EUR, +USD.
        Limit: max MAX_CURRENCY_SAME_DIR positions contributing same-direction
        exposure to any single currency.
        """
        if len(sym) < 6:
            return True

        base, quote = sym[:3], sym[3:6]

        # New trade's currency exposures: +currency means "long that currency"
        if direction == 'BUY':
            new_exposures = {base: 'long', quote: 'short'}
        else:
            new_exposures = {base: 'short', quote: 'long'}

        # Count existing currency exposures from open forex positions
        currency_long: dict[str, int] = {}
        currency_short: dict[str, int] = {}

        for p in positions:
            p_sym = _normalise(p.symbol)
            p_cfg = cfg.SYMBOLS.get(p.symbol, {})
            if p_cfg.get('asset_class', '') != 'forex':
                continue
            if len(p_sym) < 6:
                continue

            p_base, p_quote = p_sym[:3], p_sym[3:6]
            p_dir = _pos_direction(p)

            if p_dir == 'BUY':
                currency_long[p_base] = currency_long.get(p_base, 0) + 1
                currency_short[p_quote] = currency_short.get(p_quote, 0) + 1
            else:
                currency_short[p_base] = currency_short.get(p_base, 0) + 1
                currency_long[p_quote] = currency_long.get(p_quote, 0) + 1

        # Check if adding new trade would breach any currency limit
        for ccy, exp_dir in new_exposures.items():
            if exp_dir == 'long':
                current = currency_long.get(ccy, 0)
            else:
                current = currency_short.get(ccy, 0)

            if current >= MAX_CURRENCY_SAME_DIR:
                self.logger.log(
                    'WARNING', 'CorrelationGuard', 'CURRENCY_CAP',
                    f'Blocked {sym} {direction}: {ccy} already has {current} '
                    f'{exp_dir} exposures (max {MAX_CURRENCY_SAME_DIR})')
                return False

        return True

    # ── Crypto: single correlated group ──────────────────────────────

    def _check_crypto(self, sym: str, direction: str, positions) -> bool:
        same_dir = 0
        for p in positions:
            p_sym = _normalise(p.symbol)
            p_cfg = cfg.SYMBOLS.get(p.symbol, {})
            is_crypto = (p_cfg.get('asset_class', '') == 'crypto'
                         or p_sym in CRYPTO_SYMBOLS)
            if is_crypto and _pos_direction(p) == direction:
                same_dir += 1

        if same_dir >= MAX_CRYPTO_SAME_DIR:
            self.logger.log(
                'WARNING', 'CorrelationGuard', 'CRYPTO_CAP',
                f'Blocked {sym} {direction}: already {same_dir} crypto '
                f'{direction}s (max {MAX_CRYPTO_SAME_DIR})')
            return False
        return True

    # ── Commodity: metals group ──────────────────────────────────────

    def _check_commodity(self, sym: str, direction: str, positions) -> bool:
        is_metal = sym in METAL_SYMBOLS
        if not is_metal:
            return True  # non-metal commodities: no group limit

        same_dir = 0
        for p in positions:
            p_sym = _normalise(p.symbol)
            if p_sym in METAL_SYMBOLS and _pos_direction(p) == direction:
                same_dir += 1

        if same_dir >= MAX_METAL_SAME_DIR:
            self.logger.log(
                'WARNING', 'CorrelationGuard', 'METAL_CAP',
                f'Blocked {sym} {direction}: already {same_dir} metals '
                f'{direction}s (max {MAX_METAL_SAME_DIR})')
            return False
        return True

    # ── Index: regional groups ───────────────────────────────────────

    def _check_index(self, sym: str, direction: str, positions) -> bool:
        region = None
        for r, symbols in INDEX_REGIONS.items():
            if sym in symbols:
                region = r
                break
        if region is None:
            return True  # unknown index, no group limit

        same_dir = 0
        for p in positions:
            p_sym = _normalise(p.symbol)
            if p_sym in INDEX_REGIONS[region] and _pos_direction(p) == direction:
                same_dir += 1

        if same_dir >= MAX_INDEX_SAME_DIR:
            self.logger.log(
                'WARNING', 'CorrelationGuard', 'INDEX_CAP',
                f'Blocked {sym} {direction}: already {same_dir} {region} '
                f'indices {direction}s (max {MAX_INDEX_SAME_DIR})')
            return False
        return True

    # ── Equity: sector groups ────────────────────────────────────────

    def _check_equity(self, sym: str, direction: str, positions) -> bool:
        sector = None
        for s, symbols in EQUITY_SECTORS.items():
            if sym in symbols:
                sector = s
                break
        if sector is None:
            return True  # unknown equity, no group limit

        same_dir = 0
        for p in positions:
            p_sym = _normalise(p.symbol)
            if p_sym in EQUITY_SECTORS[sector] and _pos_direction(p) == direction:
                same_dir += 1

        if same_dir >= MAX_EQUITY_SAME_DIR:
            self.logger.log(
                'WARNING', 'CorrelationGuard', 'EQUITY_CAP',
                f'Blocked {sym} {direction}: already {same_dir} {sector} '
                f'{direction}s (max {MAX_EQUITY_SAME_DIR})')
            return False
        return True
