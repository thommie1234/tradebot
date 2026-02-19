"""
F8: Cross-Asset Lead-Lag Features.

Known lead-lag relationships in forex/commodities/crypto.
Leaders' recent returns predict followers' future moves.
"""
from __future__ import annotations

import numpy as np


# Known lead-lag pairs: leader → list of followers
LEAD_LAG_PAIRS = {
    # DXY leads all USD pairs (inverse for EURUSD/GBPUSD/AUDUSD/NZDUSD)
    "DXY.cash": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    # S&P 500 leads risk sentiment
    "US500.cash": ["EURUSD", "USDJPY", "AUDUSD", "BTCUSD"],
    # Gold leads safe-haven flows
    "XAUUSD": ["AUDUSD", "USDCHF", "XAGUSD"],
    # USDJPY leads JPY crosses
    "USDJPY": ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY"],
    # Oil leads CAD and energy-sensitive pairs
    "USOIL.cash": ["USDCAD", "AUDCAD"],
    # BTC leads altcoins
    "BTCUSD": ["ETHUSD", "SOLUSD", "LTCUSD", "XRPUSD", "ADAUSD", "DOGUSD", "BNBUSD"],
    # EUR/USD leads EUR crosses
    "EURUSD": ["EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF"],
    # GBP/USD leads GBP crosses
    "GBPUSD": ["GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD"],
}


def _find_leaders(symbol: str) -> list[str]:
    """Find leader symbols for a given follower."""
    leaders = []
    for leader, followers in LEAD_LAG_PAIRS.items():
        if symbol in followers:
            leaders.append(leader)
    return leaders


def build_lead_lag_features(symbol: str, mt5, logger) -> dict[str, float]:
    """
    Compute lead-lag features from leader symbols' recent H1 bars.

    Returns dict with keys:
        leader_ret1: weighted average return of leaders (1-bar)
        leader_ret3: weighted average return of leaders (3-bar)
        leader_momentum: consensus direction (-1 to +1)
    """
    leaders = _find_leaders(symbol)
    defaults = {"leader_ret1": 0.0, "leader_ret3": 0.0, "leader_momentum": 0.0}

    if not leaders or mt5 is None:
        return defaults

    ret1_list = []
    ret3_list = []
    direction_list = []

    for leader in leaders:
        try:
            rates = mt5.copy_rates_from_pos(leader, mt5.TIMEFRAME_H1, 0, 5)
            if rates is None or len(rates) < 4:
                continue

            # 1-bar return (most recent completed bar)
            c1 = float(rates[-2]['close'])
            c2 = float(rates[-3]['close'])
            if c2 > 0:
                ret1 = (c1 - c2) / c2
            else:
                ret1 = 0.0

            # 3-bar return
            c4 = float(rates[-4]['close']) if len(rates) >= 4 else c2
            if c4 > 0:
                ret3 = (c1 - c4) / c4
            else:
                ret3 = 0.0

            ret1_list.append(ret1)
            ret3_list.append(ret3)
            direction_list.append(1.0 if ret1 > 0 else (-1.0 if ret1 < 0 else 0.0))

        except Exception:
            continue

    if not ret1_list:
        return defaults

    return {
        "leader_ret1": float(np.mean(ret1_list)),
        "leader_ret3": float(np.mean(ret3_list)),
        "leader_momentum": float(np.mean(direction_list)),
    }


def build_lead_lag_features_batch(symbol: str, bars_close: np.ndarray,
                                   all_bars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Batch version for training: compute lead-lag features from pre-loaded bar data.

    Parameters
    ----------
    symbol : target symbol
    bars_close : close prices of target symbol (array)
    all_bars : dict of leader_symbol → close prices (aligned by time)

    Returns
    -------
    dict of feature_name → numpy array (same length as bars_close)
    """
    n = len(bars_close)
    leaders = _find_leaders(symbol)

    if not leaders:
        return {
            "leader_ret1": np.zeros(n),
            "leader_ret3": np.zeros(n),
            "leader_momentum": np.zeros(n),
        }

    leader_ret1_all = []
    leader_ret3_all = []
    leader_mom_all = []

    for leader in leaders:
        if leader not in all_bars:
            continue
        ldr_close = all_bars[leader]
        if len(ldr_close) != n:
            continue

        # 1-bar return
        ret1 = np.zeros(n)
        ret1[1:] = np.diff(ldr_close) / np.maximum(ldr_close[:-1], 1e-10)

        # 3-bar return
        ret3 = np.zeros(n)
        ret3[3:] = (ldr_close[3:] - ldr_close[:-3]) / np.maximum(ldr_close[:-3], 1e-10)

        leader_ret1_all.append(ret1)
        leader_ret3_all.append(ret3)
        leader_mom_all.append(np.sign(ret1))

    if not leader_ret1_all:
        return {
            "leader_ret1": np.zeros(n),
            "leader_ret3": np.zeros(n),
            "leader_momentum": np.zeros(n),
        }

    return {
        "leader_ret1": np.mean(leader_ret1_all, axis=0),
        "leader_ret3": np.mean(leader_ret3_all, axis=0),
        "leader_momentum": np.mean(leader_mom_all, axis=0),
    }
