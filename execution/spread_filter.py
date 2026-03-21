"""
Spread filter — validates spread per asset class before trade entry.

Extracted from check_spread() in sovereign_bot.py.
"""
from __future__ import annotations

from config.loader import cfg


def check_spread(symbol: str, mt5, logger, sym_cfg: dict | None = None) -> tuple[bool, float]:
    """Check if spread is acceptable. Returns (ok, spread_pct)."""
    if mt5 is None:
        return True, 0.0

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.log('ERROR', 'SpreadFilter', 'TICK_FAILED',
                    f'Could not get tick for {symbol}')
        return False, 0.0

    spread = tick.ask - tick.bid
    spread_pct = spread / tick.ask if tick.ask > 0 else 0

    if sym_cfg is None:
        sym_cfg = cfg.SYMBOLS.get(symbol, {})
    max_spread = sym_cfg.get('max_spread_pct', 0.001)

    if spread_pct > max_spread:
        logger.log('WARNING', 'SpreadFilter', 'SPREAD_TOO_WIDE',
                    f'{symbol} spread {spread_pct*100:.3f}% > {max_spread*100:.3f}%')
        return False, spread_pct

    return True, spread_pct
