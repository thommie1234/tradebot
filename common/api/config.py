"""API configuration — loads from config/api_config.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "api_config.yaml"


def _load() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"API config not found at {CONFIG_FILE}. "
            "Copy api_config.yaml.example to api_config.yaml and fill in your secrets."
        )
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


_cfg = _load()

JWT_SECRET: str = _cfg["jwt_secret"]
API_KEY: str = _cfg["api_key"]
JWT_EXPIRY_DAYS: int = _cfg.get("jwt_expiry_days", 30)
BRIDGE_HOST: str = _cfg.get("bridge_host", "127.0.0.1")
BRIDGE_PORT: int = _cfg.get("bridge_port", 5056)
