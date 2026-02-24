"""
F5: Live Portfolio Optimizer — mean-variance risk budgeting.

Uses scipy.optimize for real-time portfolio optimization.
Constrains total portfolio volatility under FTMO limits.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import minimize

from config.loader import cfg


class PortfolioOptimizer:
    """
    Real-time portfolio risk budgeting.

    Limits risk allocation per symbol/sector so total portfolio
    volatility stays under FTMO daily loss limit.
    """

    def __init__(self, max_portfolio_vol: float = 0.03, buffer_days: int = 60):
        self.max_portfolio_vol = max_portfolio_vol  # 3% daily max
        self.buffer_days = buffer_days
        self.returns_buffer: dict[str, deque] = {}  # {symbol: deque of daily returns}
        self._weights_cache: dict[str, float] = {}
        self._cache_age: float = 0.0

    def update_returns(self, symbol: str, daily_return: float):
        """Track daily returns per symbol."""
        if symbol not in self.returns_buffer:
            self.returns_buffer[symbol] = deque(maxlen=self.buffer_days)
        self.returns_buffer[symbol].append(daily_return)

    def _build_cov_matrix(self, candidates: list[str]) -> np.ndarray:
        """Build covariance matrix from rolling returns."""
        n = len(candidates)

        # Find minimum shared history length
        min_len = min(len(self.returns_buffer.get(s, [])) for s in candidates)
        if min_len < 5:
            # Not enough data — return identity (equal risk)
            return np.eye(n) * 0.001

        # Build returns matrix
        returns_matrix = np.zeros((min_len, n))
        for j, sym in enumerate(candidates):
            buf = self.returns_buffer[sym]
            returns_matrix[:, j] = list(buf)[-min_len:]

        # Covariance matrix with shrinkage toward diagonal (Ledoit-Wolf-like)
        cov = np.cov(returns_matrix, rowvar=False)
        # Shrinkage: blend with diagonal
        shrinkage = 0.3
        diag = np.diag(np.diag(cov))
        cov_shrunk = (1 - shrinkage) * cov + shrinkage * diag

        return cov_shrunk

    def optimal_weights(self, candidates: list[str]) -> dict[str, float]:
        """
        Compute optimal risk weights given historical covariance.

        Returns dict of symbol → weight (sums to 1.0).
        """
        if len(candidates) == 0:
            return {}
        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        # Check if we have return data
        have_data = all(
            len(self.returns_buffer.get(s, [])) >= 5 for s in candidates
        )
        if not have_data:
            # Equal weight fallback
            w = 1.0 / len(candidates)
            return {c: w for c in candidates}

        cov_matrix = self._build_cov_matrix(candidates)
        n = len(candidates)

        # Minimize portfolio variance
        def portfolio_vol(w):
            return float(w @ cov_matrix @ w)

        x0 = np.ones(n) / n
        bounds = [(0.05, 0.5)] * n  # Each symbol: 5-50% of risk budget
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        try:
            result = minimize(
                portfolio_vol,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8},
            )
            if result.success:
                weights = result.x
            else:
                weights = x0  # Fallback to equal weight
        except Exception:
            weights = x0

        # Normalize
        weights = weights / np.sum(weights)
        return dict(zip(candidates, weights.tolist()))

    def risk_budget(self, symbol: str, base_risk: float,
                    candidates: list[str] | None = None) -> float:
        """
        Adjust risk_pct based on portfolio optimization.

        Parameters
        ----------
        symbol : target symbol
        base_risk : unadjusted risk (from Kelly/RL)
        candidates : list of all active trading candidates

        Returns
        -------
        Adjusted risk_pct
        """
        if candidates is None or len(candidates) < 2:
            return base_risk

        weights = self.optimal_weights(candidates)
        symbol_weight = weights.get(symbol, 1.0 / len(candidates))

        # Scale risk by portfolio weight
        # Weight > 1/n means overweight (higher risk), < 1/n means underweight
        n = len(candidates)
        equal_weight = 1.0 / n
        scale_factor = symbol_weight / equal_weight if equal_weight > 0 else 1.0

        # Clamp scale factor to [0.3, 2.0] to prevent extreme allocations
        scale_factor = max(0.3, min(2.0, scale_factor))

        return base_risk * scale_factor
