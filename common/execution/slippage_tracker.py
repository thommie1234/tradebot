"""
Slippage tracker — logs slippage per fill for post-trade analysis.

Extracted from execute_trade() slippage logging.
"""
from __future__ import annotations


def log_slippage(logger, symbol: str, direction: str, requested_price: float,
                 fill_price: float, lot_size: float):
    """Log slippage event."""
    slippage_pts = abs(fill_price - requested_price)
    slippage_bps = (slippage_pts / requested_price * 10_000) if requested_price > 0 else 0.0

    logger.log('INFO', 'SlippageTracker', 'SLIPPAGE',
               f'{symbol} {direction}: requested={requested_price:.5f} '
               f'filled={fill_price:.5f} slippage={slippage_bps:.2f}bps',
               data={'symbol': symbol, 'direction': direction,
                     'requested_price': requested_price, 'fill_price': fill_price,
                     'slippage_bps': slippage_bps, 'lot_size': lot_size})

    return slippage_bps
