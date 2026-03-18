"""
F4: Regime Detection — trending / mean-reverting / volatile.

Uses rolling Hurst exponent proxy (R/S analysis) and vol-of-vol.
Only scipy/numpy required (no HMM library).
"""
from __future__ import annotations

import numpy as np


class RegimeDetector:
    """
    Detect market regime: trending / mean-reverting / volatile.

    Regimes:
        0 = trending      (H > 0.6)
        1 = mean_reverting (H < 0.4)
        2 = volatile       (vol-of-vol > 1.5 std override)
    """

    REGIMES = {"trending": 0, "mean_reverting": 1, "volatile": 2}

    @staticmethod
    def detect(close: np.ndarray, vol: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect regime for each bar.

        Parameters
        ----------
        close : close prices array
        vol : rolling volatility array (e.g., vol20)
        window : lookback window for Hurst calculation

        Returns
        -------
        (regimes, hurst, regime_duration) — all same length as input
        """
        n = len(close)
        hurst = _rolling_hurst(close, window)
        vov = _rolling_vol_of_vol(vol, window)

        # Volatile override: vol-of-vol > 1.5× its own rolling mean
        vov_mean = np.full(n, np.nan)
        for i in range(window, n):
            segment = vov[i - window:i]
            valid = segment[np.isfinite(segment)]
            if len(valid) > 0:
                vov_mean[i] = np.mean(valid) + 1.5 * np.std(valid)
            else:
                vov_mean[i] = np.inf

        regimes = np.where(
            (np.isfinite(vov)) & (np.isfinite(vov_mean)) & (vov > vov_mean), 2,
            np.where(hurst > 0.6, 0,
            np.where(hurst < 0.4, 1, 0))
        )

        # Handle NaN regions (beginning)
        regimes = np.where(np.isfinite(hurst), regimes, 0).astype(np.int32)

        # Regime duration: consecutive bars in same regime
        regime_duration = np.ones(n, dtype=np.float64)
        for i in range(1, n):
            if regimes[i] == regimes[i - 1]:
                regime_duration[i] = regime_duration[i - 1] + 1.0
            else:
                regime_duration[i] = 1.0

        return regimes, hurst, regime_duration


def _rolling_hurst(close: np.ndarray, window: int) -> np.ndarray:
    """
    Simplified Hurst exponent via R/S (Rescaled Range) analysis.

    H > 0.5 → persistent (trending)
    H < 0.5 → anti-persistent (mean reverting)
    H = 0.5 → random walk
    """
    n = len(close)
    hurst = np.full(n, np.nan)

    # Work with log returns
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
    log_ret = np.concatenate([[0.0], log_ret])

    for i in range(window, n):
        segment = log_ret[i - window:i]
        if np.all(segment == 0):
            hurst[i] = 0.5
            continue

        # Mean-adjusted cumulative deviation
        mean_ret = np.mean(segment)
        deviation = np.cumsum(segment - mean_ret)

        # Range
        r = np.max(deviation) - np.min(deviation)

        # Standard deviation
        s = np.std(segment, ddof=1)
        if s < 1e-15:
            hurst[i] = 0.5
            continue

        # R/S statistic
        rs = r / s

        # Hurst exponent: H = log(R/S) / log(n)
        if rs > 0:
            hurst[i] = np.log(rs) / np.log(window)
        else:
            hurst[i] = 0.5

    # Clip to valid range
    hurst = np.clip(hurst, 0.0, 1.0)
    return hurst


def _rolling_vol_of_vol(vol: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation of volatility (vol-of-vol)."""
    n = len(vol)
    vov = np.full(n, np.nan)
    for i in range(window, n):
        segment = vol[i - window:i]
        valid = segment[np.isfinite(segment)]
        if len(valid) > 2:
            vov[i] = np.std(valid, ddof=1)
        else:
            vov[i] = 0.0
    return vov
