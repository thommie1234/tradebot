"""
MT5 broker API wrapper — connection, symbol info, initialization.

Extracted from SovereignExecution MT5 methods + mt5_bridge.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# MT5 bridge (direct in Wine, TCP on Linux)
MT5_AVAILABLE = False
mt5 = None

try:
    # Try importing from tools/ first (new location), fall back to old
    _repo_root = Path(__file__).resolve().parent.parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    # Try new location
    from tools.mt5_bridge import get_mt5_bridge, initialize_mt5 as bridge_initialize_mt5
    mt5 = get_mt5_bridge()
    MT5_AVAILABLE = True
except Exception as e:
    print(f"[WARN] MT5 bridge not available — running in offline mode: {e}")


def get_mt5():
    """Get the MT5 bridge instance."""
    return mt5


def is_available() -> bool:
    return MT5_AVAILABLE


def initialize(logger) -> tuple[bool, str]:
    """Initialize MT5 connection. Returns (success, mode)."""
    if not MT5_AVAILABLE:
        logger.log('WARNING', 'BrokerAPI', 'MT5_UNAVAILABLE',
                    'MetaTrader5 module not available')
        return False, "unavailable"
    try:
        ok, mt5_error, mode = bridge_initialize_mt5(mt5)
        if not ok:
            logger.log('ERROR', 'BrokerAPI', 'MT5_INIT_FAILED',
                        f'Failed: {mt5_error}')
            return False, "failed"
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
    if not MT5_AVAILABLE:
        return None
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "trade_contract_size": info.trade_contract_size,
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
    if MT5_AVAILABLE and mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass
