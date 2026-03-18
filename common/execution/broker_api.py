"""
MT5 broker API — connection, symbol info, initialization.

Uses native MetaTrader5 module directly (no bridge wrapper).
"""
from __future__ import annotations

from tools.mt5_bridge import mt5, initialize_mt5

MT5_AVAILABLE = True


def get_mt5():
    """Get the MT5 module."""
    return mt5


def is_available() -> bool:
    return MT5_AVAILABLE


def initialize(logger) -> tuple[bool, str]:
    """Initialize MT5 connection. Returns (success, mode)."""
    try:
        ok, mt5_error, mode = initialize_mt5()
        if not ok:
            logger.log('WARNING', 'BrokerAPI', 'MT5_INIT_SKIPPED',
                        'Default broker init skipped (multi-account mode uses account_context)')
            return False, "skipped"
        logger.log('INFO', 'BrokerAPI', 'MT5_INIT_MODE',
                    f'MT5 initialized via {mode}')

        account_info = mt5.account_info()
        if account_info:
            logger.log('INFO', 'BrokerAPI', 'MT5_INITIALIZED',
                        f'Account {account_info.login} | '
                        f'Balance ${account_info.balance:,.2f} | '
                        f'Equity ${account_info.equity:,.2f}')

        return True, mode
    except Exception as e:
        logger.log('ERROR', 'BrokerAPI', 'MT5_INIT_ERROR', str(e))
        return False, "error"


def get_symbol_info(symbol: str) -> dict | None:
    """Get symbol info from MT5."""
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "trade_contract_size": info.trade_contract_size,
            "trade_tick_value": getattr(info, "trade_tick_value", 0.0),
            "trade_tick_size": getattr(info, "trade_tick_size", 0.0),
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "point": info.point,
            "digits": info.digits,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
            "margin_initial": info.margin_initial,
        }
    except Exception:
        return None


def shutdown():
    """Shutdown MT5 connection."""
    try:
        mt5.shutdown()
    except Exception:
        pass
