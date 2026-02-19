"""
F13: Model Versioning + A/B Testing.

Version management for XGBoost models with champion/challenger A/B testing.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

from config.loader import cfg

REPO_ROOT = Path(__file__).resolve().parent.parent


class ModelRegistry:
    """Version control for XGBoost models with A/B testing support."""

    VERSION_DIR = os.path.join(cfg.MODEL_DIR, "versions") if cfg.MODEL_DIR else str(
        REPO_ROOT / "models" / "sovereign_models" / "versions"
    )
    REGISTRY_PATH = str(REPO_ROOT / "models" / "registry.yaml")

    def __init__(self, logger=None):
        self.logger = logger
        self._registry = self._load_registry()
        os.makedirs(self.VERSION_DIR, exist_ok=True)

    def _load_registry(self) -> dict:
        if os.path.exists(self.REGISTRY_PATH):
            try:
                with open(self.REGISTRY_PATH) as f:
                    data = yaml.safe_load(f) or {}
                return data
            except Exception:
                return {}
        return {}

    def _save_registry(self):
        try:
            with open(self.REGISTRY_PATH, "w") as f:
                yaml.dump(self._registry, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            if self.logger:
                self.logger.log('WARNING', 'ModelRegistry', 'SAVE_ERROR', str(e))

    def save_version(self, symbol: str, model, metadata: dict | None = None) -> str:
        """Save a model version with timestamp + metadata. Returns version string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{symbol}_{timestamp}"
        model_path = os.path.join(self.VERSION_DIR, f"{version}.json")

        os.makedirs(self.VERSION_DIR, exist_ok=True)
        model.save_model(model_path)

        # Update registry
        if "symbols" not in self._registry:
            self._registry["symbols"] = {}
        if symbol not in self._registry["symbols"]:
            self._registry["symbols"][symbol] = {
                "champion": None,
                "versions": [],
                "ab_results": [],
            }

        entry = self._registry["symbols"][symbol]
        version_info = {
            "version": version,
            "path": model_path,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        entry["versions"].append(version_info)

        # Always promote latest version as champion (retrain = new champion)
        entry["champion"] = version

        self._save_registry()

        if self.logger:
            self.logger.log('INFO', 'ModelRegistry', 'VERSION_SAVED',
                            f'{symbol}: saved version {version}')
        return version

    def get_active_model(self, symbol: str):
        """Return (version, model_path) for the active champion model."""
        entry = self._registry.get("symbols", {}).get(symbol)
        if not entry or not entry.get("champion"):
            # Fallback to default model path
            default_path = os.path.join(cfg.MODEL_DIR, f"{symbol}.json")
            if os.path.exists(default_path):
                return "default", default_path
            return None, None

        champion_version = entry["champion"]
        for v in entry["versions"]:
            if v["version"] == champion_version:
                if os.path.exists(v["path"]):
                    return champion_version, v["path"]
                break

        # Fallback to default
        default_path = os.path.join(cfg.MODEL_DIR, f"{symbol}.json")
        if os.path.exists(default_path):
            return "default", default_path
        return None, None

    def promote(self, symbol: str, version: str):
        """Promote a version to champion."""
        entry = self._registry.get("symbols", {}).get(symbol)
        if not entry:
            return
        entry["champion"] = version

        # Also copy to the default model path for backward compatibility
        for v in entry["versions"]:
            if v["version"] == version and os.path.exists(v["path"]):
                default_path = os.path.join(cfg.MODEL_DIR, f"{symbol}.json")
                shutil.copy2(v["path"], default_path)
                break

        self._save_registry()
        if self.logger:
            self.logger.log('INFO', 'ModelRegistry', 'PROMOTED',
                            f'{symbol}: promoted {version} to champion')

    def ab_test_select(self, symbol: str, challenger_pct: float = 0.10):
        """Select champion or challenger model for A/B testing.

        Returns (version, model_path, is_challenger).
        """
        entry = self._registry.get("symbols", {}).get(symbol)
        if not entry:
            return self.get_active_model(symbol) + (False,)

        champion = entry.get("champion")
        versions = entry.get("versions", [])

        if len(versions) < 2 or not champion:
            version, path = self.get_active_model(symbol)
            return version, path, False

        # Find most recent non-champion version as challenger
        challenger = None
        for v in reversed(versions):
            if v["version"] != champion and os.path.exists(v["path"]):
                challenger = v
                break

        if challenger is None:
            version, path = self.get_active_model(symbol)
            return version, path, False

        # A/B split
        if random.random() < challenger_pct:
            return challenger["version"], challenger["path"], True
        else:
            version, path = self.get_active_model(symbol)
            return version, path, False

    def record_ab_result(self, symbol: str, version: str, pnl: float):
        """Log an A/B test result for a version."""
        entry = self._registry.get("symbols", {}).get(symbol)
        if not entry:
            return

        if "ab_results" not in entry:
            entry["ab_results"] = []

        entry["ab_results"].append({
            "version": version,
            "pnl": pnl,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last 200 results
        entry["ab_results"] = entry["ab_results"][-200:]
        self._save_registry()

    def evaluate_ab(self, symbol: str, min_trades: int = 30) -> dict:
        """Compare champion vs challenger performance."""
        entry = self._registry.get("symbols", {}).get(symbol)
        if not entry:
            return {"status": "no_data"}

        champion = entry.get("champion")
        results = entry.get("ab_results", [])

        champion_pnls = [r["pnl"] for r in results if r["version"] == champion]
        challenger_pnls = [r["pnl"] for r in results if r["version"] != champion]

        if len(champion_pnls) < min_trades or len(challenger_pnls) < min_trades:
            return {
                "status": "insufficient_data",
                "champion_trades": len(champion_pnls),
                "challenger_trades": len(challenger_pnls),
                "min_required": min_trades,
            }

        import numpy as np
        c_mean = np.mean(champion_pnls)
        ch_mean = np.mean(challenger_pnls)
        c_wr = np.mean(np.array(champion_pnls) > 0)
        ch_wr = np.mean(np.array(challenger_pnls) > 0)

        return {
            "status": "ready",
            "champion": {
                "version": champion,
                "trades": len(champion_pnls),
                "mean_pnl": float(c_mean),
                "total_pnl": float(sum(champion_pnls)),
                "win_rate": float(c_wr),
            },
            "challenger": {
                "trades": len(challenger_pnls),
                "mean_pnl": float(ch_mean),
                "total_pnl": float(sum(challenger_pnls)),
                "win_rate": float(ch_wr),
            },
            "challenger_better": ch_mean > c_mean,
        }

    def list_versions(self, symbol: str) -> list[dict]:
        """List all versions for a symbol."""
        entry = self._registry.get("symbols", {}).get(symbol)
        if not entry:
            return []
        return entry.get("versions", [])
