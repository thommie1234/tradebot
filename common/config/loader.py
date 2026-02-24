"""
Config loader — reads all YAML configs into a frozen dataclass.

Usage:
    from config.loader import cfg, load_config
    load_config()  # call once at startup
    print(cfg.ml_threshold)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent.parent


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class SovereignConfig:
    """Production configuration — loaded from YAML + sovereign_configs.json"""

    # Per-symbol configs (loaded from JSON)
    SYMBOLS: dict = field(default_factory=dict)

    # Optuna XGBoost params per symbol (loaded from CSV)
    OPTUNA_PARAMS: dict = field(default_factory=dict)

    # FTMO limits
    MAX_DAILY_LOSS_PCT: float = 0.05
    MAX_TOTAL_LOSS_PCT: float = 0.10
    MAX_CONCURRENT_POSITIONS: int = 200
    ACCOUNT_SIZE: float = 100_000

    # ML
    ML_THRESHOLD: float = 0.55
    DISABLE_ZSCORE: bool = True
    TRAIN_SIZE: int = 800

    # F13: A/B Testing
    AB_TEST_ENABLED: bool = False
    AB_TEST_CHALLENGER_PCT: float = 0.10

    # Timing
    HEARTBEAT_INTERVAL: int = 10
    BLACKOUT_ROLLOVER_START: int = 23
    BLACKOUT_ROLLOVER_END: int = 0

    # Order execution
    DEVIATION: int = 10
    RR_RATIO: float = 3.0

    # Paths (resolved relative to REPO_ROOT)
    DB_PATH: str = ""
    MODEL_DIR: str = ""
    CONFIG_PATH: str = ""
    WFO_LOG: str = ""
    OPTUNA_CSV: str = ""
    INFO_DIR: str = ""
    TRADE_LOG_DIR: str = ""

    # Data
    DATA_ROOTS: list = field(default_factory=list)
    BAR_ROOTS: list = field(default_factory=list)

    def load(self):
        """Load symbol configs from JSON."""
        if os.path.exists(self.CONFIG_PATH):
            with open(self.CONFIG_PATH) as f:
                self.SYMBOLS = json.load(f)
            print(f"[CONFIG] Loaded {len(self.SYMBOLS)} symbols from {self.CONFIG_PATH}")
        else:
            print(f"[CONFIG] WARNING: {self.CONFIG_PATH} not found — run with --build-plan first")

    def load_optuna_params(self):
        """Load Optuna XGBoost params from CSV."""
        import polars as pl

        csv_path = os.getenv("OPTUNA_CSV_OVERRIDE", self.OPTUNA_CSV)
        if csv_path != self.OPTUNA_CSV:
            print(f"[CONFIG] Using OPTUNA_CSV_OVERRIDE: {csv_path}")
            self.OPTUNA_CSV = csv_path
        if not os.path.exists(self.OPTUNA_CSV):
            print(f"[CONFIG] WARNING: Optuna CSV not found at {self.OPTUNA_CSV}")
            return

        df = pl.read_csv(self.OPTUNA_CSV)
        ok = df.filter(pl.col("status") == "ok")

        for row in ok.iter_rows(named=True):
            sym = row["symbol"]
            # Use Optuna-optimized eta/colsample_bylevel if available, else defaults
            eta = float(row["best_eta"]) if row.get("best_eta") else 0.03
            colsample_bylevel = float(row["best_colsample_bylevel"]) if row.get("best_colsample_bylevel") else 0.75
            self.OPTUNA_PARAMS[sym] = {
                "xgb_params": {
                    "booster": "gbtree",
                    "tree_method": "hist",
                    "device": "cuda",
                    "sampling_method": "gradient_based",
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "max_depth": int(row["best_max_depth"]),
                    "eta": eta,
                    "gamma": float(row["best_gamma"]),
                    "subsample": float(row["best_subsample"]),
                    "colsample_bytree": float(row["best_colsample_bytree"]),
                    "colsample_bylevel": colsample_bylevel,
                    "reg_alpha": float(row["best_reg_alpha"]),
                    "reg_lambda": float(row["best_reg_lambda"]),
                    "min_child_weight": float(row["best_min_child_weight"]),
                    "max_bin": 512,
                    "grow_policy": "lossguide",
                    "verbosity": 0,
                },
                "num_boost_round": int(row["best_num_boost_round"]),
                "optuna_ev": float(row["best_ev"]),
                "timeframe": row.get("timeframe", "H1") or "H1",
                "fee_bps": float(row["fee_bps"]) if row.get("fee_bps") is not None else 3.0,
            }
        print(f"[CONFIG] Loaded Optuna params for {len(self.OPTUNA_PARAMS)} symbols")


def _resolve_path(rel: str) -> str:
    """Resolve a path relative to REPO_ROOT."""
    if not rel:
        return ""
    p = REPO_ROOT / rel
    return str(p)


# Module-level singleton
cfg = SovereignConfig()


def load_config():
    """Load all YAML configs and populate the singleton."""
    paths = _load_yaml("paths.yaml")
    ftmo = _load_yaml("ftmo.yaml")
    thresholds = _load_yaml("thresholds.yaml")
    risk = _load_yaml("risk.yaml")
    execution = _load_yaml("execution.yaml")

    # Paths
    cfg.DATA_ROOTS = paths.get("data_roots", cfg.DATA_ROOTS)
    cfg.BAR_ROOTS = paths.get("bar_roots", cfg.BAR_ROOTS)
    cfg.MODEL_DIR = _resolve_path(paths.get("model_dir", "models/sovereign_models"))
    cfg.DB_PATH = _resolve_path(paths.get("db_path", "logging/sovereign_log.db"))
    cfg.CONFIG_PATH = _resolve_path(paths.get("config_path", "config/sovereign_configs.json"))
    cfg.WFO_LOG = _resolve_path(paths.get("wfo_log", ""))
    cfg.OPTUNA_CSV = _resolve_path(paths.get("optuna_csv", ""))
    cfg.INFO_DIR = _resolve_path(paths.get("info_dir", "data/instrument_specs"))
    cfg.TRADE_LOG_DIR = _resolve_path(paths.get("trade_log_dir", "logging/trade_logs"))

    # FTMO
    cfg.MAX_DAILY_LOSS_PCT = ftmo.get("max_daily_loss_pct", cfg.MAX_DAILY_LOSS_PCT)
    cfg.MAX_TOTAL_LOSS_PCT = ftmo.get("max_total_loss_pct", cfg.MAX_TOTAL_LOSS_PCT)
    cfg.MAX_CONCURRENT_POSITIONS = ftmo.get("max_concurrent_positions", cfg.MAX_CONCURRENT_POSITIONS)
    cfg.ACCOUNT_SIZE = ftmo.get("account_size", cfg.ACCOUNT_SIZE)
    timing = ftmo.get("timing", {})
    cfg.HEARTBEAT_INTERVAL = timing.get("heartbeat_interval", cfg.HEARTBEAT_INTERVAL)

    # ML thresholds
    ml = thresholds.get("ml", {})
    cfg.ML_THRESHOLD = ml.get("threshold", cfg.ML_THRESHOLD)
    cfg.DISABLE_ZSCORE = ml.get("disable_zscore", cfg.DISABLE_ZSCORE)
    cfg.TRAIN_SIZE = ml.get("train_size", cfg.TRAIN_SIZE)

    # F13: A/B Testing
    ab = thresholds.get("ab_test", {})
    cfg.AB_TEST_ENABLED = ab.get("enabled", cfg.AB_TEST_ENABLED)
    cfg.AB_TEST_CHALLENGER_PCT = ab.get("challenger_pct", cfg.AB_TEST_CHALLENGER_PCT)

    # Execution
    cfg.DEVIATION = execution.get("deviation", cfg.DEVIATION)
    cfg.RR_RATIO = execution.get("rr_ratio", cfg.RR_RATIO)

    return cfg
