"""
F3: RL Position Sizing — Contextual Bandit (LinUCB).

Learns optimal risk multiplier per market context.
No gym/stable-baselines needed — pure numpy/scipy.

Context: [ml_confidence, regime, volatility, drawdown_state, win_streak]
Actions: risk multipliers [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
Reward: realized P&L / risk_taken
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from config.loader import cfg


class ContextualBanditSizer:
    """
    LinUCB contextual bandit for position sizing.

    Chooses a risk multiplier based on market context.
    Updates after each trade with realized reward.
    """

    N_ARMS = 8
    ARMS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    CONTEXT_DIM = 5  # [ml_confidence, regime, volatility, dd_state, win_streak]

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        # Per arm: A matrix (d×d) and b vector (d×1)
        d = self.CONTEXT_DIM
        self.A = [np.eye(d) for _ in range(self.N_ARMS)]
        self.b = [np.zeros(d) for _ in range(self.N_ARMS)]
        self._win_streak = 0

    def build_context(self, ml_confidence: float, regime: int = 0,
                      volatility: float = 0.0, drawdown_pct: float = 0.0) -> np.ndarray:
        """Build context vector from market state."""
        # Normalize inputs
        conf_norm = (ml_confidence - 0.5) * 2.0  # [0.5, 1.0] → [0.0, 1.0]
        regime_norm = regime / 2.0  # {0, 1, 2} → [0.0, 1.0]
        vol_norm = min(volatility / 0.02, 1.0)  # Clip at 2% vol
        dd_norm = min(drawdown_pct / 0.05, 1.0)  # Clip at 5% DD
        streak_norm = min(self._win_streak / 5.0, 1.0)  # Clip at 5 streak

        return np.array([conf_norm, regime_norm, vol_norm, dd_norm, streak_norm],
                        dtype=np.float64)

    def select_arm(self, context: np.ndarray) -> tuple[int, float]:
        """
        Choose optimal risk multiplier given context.

        Uses LinUCB: θ = A⁻¹b, p = θᵀx + α√(xᵀA⁻¹x)
        Returns (arm_index, multiplier).
        """
        best_arm = 0
        best_ucb = -np.inf

        for a in range(self.N_ARMS):
            try:
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                ucb = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
                if ucb > best_ucb:
                    best_arm = a
                    best_ucb = ucb
            except np.linalg.LinAlgError:
                continue

        return best_arm, self.ARMS[best_arm]

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update after trade result."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Track win streak
        if reward > 0:
            self._win_streak += 1
        else:
            self._win_streak = 0

    def save(self, path: str | None = None):
        """Save bandit state to disk."""
        if path is None:
            path = os.path.join(cfg.MODEL_DIR, "rl_sizer.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "alpha": self.alpha,
            "win_streak": self._win_streak,
            "A": [a.tolist() for a in self.A],
            "b": [b.tolist() for b in self.b],
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path: str | None = None) -> bool:
        """Load bandit state from disk."""
        if path is None:
            path = os.path.join(cfg.MODEL_DIR, "rl_sizer.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            self.alpha = state["alpha"]
            self._win_streak = state.get("win_streak", 0)
            self.A = [np.array(a) for a in state["A"]]
            self.b = [np.array(b) for b in state["b"]]
            return True
        except Exception:
            return False
