"""
Triple-barrier labeling — dynamic PT/SL scaled by rolling volatility.

Copied from trading_prop/ml/labeling.py (canonical source).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TripleBarrierResult:
    label: np.ndarray
    tb_ret: np.ndarray
    exit_idx: np.ndarray
    upside: np.ndarray
    downside: np.ndarray


def apply_triple_barrier(
    close: np.ndarray,
    vol_proxy: np.ndarray,
    side: np.ndarray,
    horizon: int,
    pt_mult: float,
    sl_mult: float,
) -> TripleBarrierResult:
    """
    Dynamic triple-barrier:
    - pt/sl distances scale with rolling volatility proxy (e.g., vol20 / ATR proxy).
    - side is primary signal direction (+1 long, -1 short, 0 no-trade candidate).
    """
    n = len(close)
    label = np.full(n, np.nan, dtype=float)
    tb_ret = np.full(n, np.nan, dtype=float)
    exit_idx = np.full(n, -1, dtype=int)
    upside = np.full(n, np.nan, dtype=float)
    downside = np.full(n, np.nan, dtype=float)

    events = np.where((side != 0) & np.isfinite(vol_proxy))[0]
    for i in events:
        end = min(i + horizon, n - 1)
        if end <= i:
            continue
        entry = close[i]
        up = max(1e-12, pt_mult * vol_proxy[i])
        dn = max(1e-12, sl_mult * vol_proxy[i])
        upside[i] = up
        downside[i] = dn

        chosen = end
        ret = side[i] * ((close[end] / entry) - 1.0)
        lab = 1.0 if ret > 0.0 else 0.0
        for j in range(i + 1, end + 1):
            r = side[i] * ((close[j] / entry) - 1.0)
            if r >= up:
                chosen = j
                ret = r
                lab = 1.0
                break
            if r <= -dn:
                chosen = j
                ret = r
                lab = 0.0
                break

        label[i] = lab
        tb_ret[i] = ret
        exit_idx[i] = chosen

    return TripleBarrierResult(
        label=label,
        tb_ret=tb_ret,
        exit_idx=exit_idx,
        upside=upside,
        downside=downside,
    )
