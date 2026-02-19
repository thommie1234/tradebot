"""
F15: Deep Learning Ensemble — TabNet + XGBoost.

XGBoost = base model (always present).
TabNet = optional second model (requires torch + pytorch-tabnet).

Ensemble: weighted average of probabilities.
Default: 0.7 × XGBoost + 0.3 × TabNet (XGBoost proven, TabNet experimental).

Graceful fallback: if torch not installed → XGBoost only.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from config.loader import cfg


def _torch_available() -> bool:
    """Check if PyTorch and TabNet are installed."""
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        return True
    except ImportError:
        return False


def _cuda_usable() -> bool:
    """Check if CUDA actually works (not just 'available')."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Actually test a CUDA operation — catches sm_61 incompatibility
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


class DeepEnsemble:
    """
    TabNet + XGBoost ensemble.

    XGBoost = base model (always).
    TabNet = optional second model.
    """

    def __init__(self, symbol: str, xgb_weight: float = 0.7):
        self.symbol = symbol
        self.xgb_weight = xgb_weight
        self.tabnet_model = None
        self._has_torch = _torch_available()

    @property
    def tabnet_weight(self) -> float:
        return 1.0 - self.xgb_weight

    def train_tabnet(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Train TabNet with early stopping. Returns True if successful."""
        if not self._has_torch:
            return False

        try:
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier

            self.tabnet_model = TabNetClassifier(
                n_d=16,
                n_a=16,
                n_steps=4,
                gamma=1.5,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                mask_type="entmax",
                device_name="cuda" if _cuda_usable() else "cpu",
                verbose=0,
            )

            # Ensure labels are integer type
            y_tr = y_train.astype(np.int64)
            y_va = y_val.astype(np.int64)

            self.tabnet_model.fit(
                X_train.astype(np.float32), y_tr,
                eval_set=[(X_val.astype(np.float32), y_va)],
                max_epochs=100,
                patience=15,
                batch_size=256,
            )
            return True

        except Exception as e:
            self.tabnet_model = None
            return False

    def predict(self, features: np.ndarray, xgb_proba: float) -> float:
        """
        Ensemble prediction.

        Parameters
        ----------
        features : 2D array (1, n_features) for TabNet
        xgb_proba : XGBoost probability from existing pipeline

        Returns
        -------
        Blended probability
        """
        if self.tabnet_model is None:
            return xgb_proba

        try:
            tabnet_proba = self.tabnet_model.predict_proba(
                features.astype(np.float32)
            )[:, 1]
            blended = self.xgb_weight * xgb_proba + self.tabnet_weight * float(tabnet_proba[0])
            return float(np.clip(blended, 0.0, 1.0))
        except Exception:
            return xgb_proba

    def save(self, path: str | None = None):
        """Save TabNet model to disk."""
        if self.tabnet_model is None:
            return
        if path is None:
            path = os.path.join(cfg.MODEL_DIR, f"{self.symbol}_tabnet.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # pytorch-tabnet has built-in save
        try:
            save_path = path.replace(".pkl", "")
            self.tabnet_model.save_model(save_path)
        except Exception:
            # Fallback to pickle
            with open(path, "wb") as f:
                pickle.dump(self.tabnet_model, f)

    def load(self, path: str | None = None) -> bool:
        """Load TabNet model from disk."""
        if not self._has_torch:
            return False
        if path is None:
            path = os.path.join(cfg.MODEL_DIR, f"{self.symbol}_tabnet.pkl")

        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            save_path = path.replace(".pkl", "")
            # pytorch-tabnet expects path without extension for zip files
            zip_path = save_path + ".zip"
            if os.path.exists(zip_path):
                self.tabnet_model = TabNetClassifier()
                self.tabnet_model.load_model(zip_path)
                return True
            elif os.path.exists(path):
                with open(path, "rb") as f:
                    self.tabnet_model = pickle.load(f)
                return True
        except Exception:
            self.tabnet_model = None
        return False

    @property
    def is_available(self) -> bool:
        """Check if TabNet model is loaded and ready."""
        return self.tabnet_model is not None
