"""
ML inference — XGBoost per-symbol signal filter.

Extracted from SovereignMLFilter in sovereign_bot.py (lines 669-863).
"""
from __future__ import annotations

import os

import numpy as np

from config.loader import cfg


# Lazy ML imports
_ML_LOADED = False


def _ensure_ml_imports():
    global _ML_LOADED
    if _ML_LOADED:
        return
    import importlib
    global pl, xgb, build_bar_features, FEATURE_COLUMNS
    global apply_triple_barrier, make_time_bars, sanitize_training_frame
    global infer_spread_bps, infer_slippage_bps, make_ev_custom_objective

    import polars
    import xgboost
    pl = polars
    xgb = xgboost

    from engine.feature_builder import build_bar_features as _bbf, FEATURE_COLUMNS as _fc
    build_bar_features = _bbf
    FEATURE_COLUMNS = _fc

    from engine.labeling import apply_triple_barrier as _atb
    apply_triple_barrier = _atb

    train_mod = importlib.import_module("research.train_ml_strategy")
    make_time_bars = train_mod.make_time_bars
    sanitize_training_frame = train_mod.sanitize_training_frame
    infer_spread_bps = train_mod.infer_spread_bps
    infer_slippage_bps = train_mod.infer_slippage_bps

    pipeline_mod = importlib.import_module("research.integrated_pipeline")
    make_ev_custom_objective = pipeline_mod.make_ev_custom_objective

    _ML_LOADED = True


class SovereignMLFilter:
    """XGBoost signal filter per symbol — trained on WFO tick data"""

    def __init__(self, symbol, logger):
        self.symbol = symbol
        self.logger = logger
        self.model = None  # xgb.Booster
        self._model_version = "default"
        self._registry = None
        self._ensemble = None  # F15: DeepEnsemble (lazy loaded)

    def _get_registry(self):
        """Lazy-load model registry (F13)."""
        if self._registry is None:
            try:
                from models.model_registry import ModelRegistry
                self._registry = ModelRegistry(self.logger)
            except Exception:
                self._registry = False  # Mark as unavailable
        return self._registry if self._registry is not False else None

    def model_path(self) -> str:
        return os.path.join(cfg.MODEL_DIR, f"{self.symbol}.json")

    def load_model(self) -> bool:
        """Load cached model from disk, preferring registry if available."""
        _ensure_ml_imports()

        # F13: Try loading via registry first
        registry = self._get_registry()
        if registry:
            version, path = registry.get_active_model(self.symbol)
            if path and os.path.exists(path):
                self.model = xgb.Booster()
                self.model.load_model(path)
                self._model_version = version or "default"
                self.logger.log('INFO', 'SovereignMLFilter', 'MODEL_LOADED',
                                f'{self.symbol} v={self._model_version} loaded from {path}')
                return True

        # Fallback to direct file load
        path = self.model_path()
        if not os.path.exists(path):
            return False
        self.model = xgb.Booster()
        self.model.load_model(path)
        self._model_version = "default"
        self.logger.log('INFO', 'SovereignMLFilter', 'MODEL_LOADED',
                        f'{self.symbol} model loaded from {path}')

        # F15: Try to load TabNet ensemble
        self._load_ensemble()
        return True

    def load_challenger(self) -> tuple[str | None, object | None]:
        """Load challenger model for A/B testing (F13). Returns (version, booster)."""
        registry = self._get_registry()
        if not registry or not cfg.AB_TEST_ENABLED:
            return None, None

        version, path, is_challenger = registry.ab_test_select(
            self.symbol, cfg.AB_TEST_CHALLENGER_PCT
        )
        if not is_challenger or not path or not os.path.exists(path):
            return None, None

        _ensure_ml_imports()
        challenger = xgb.Booster()
        challenger.load_model(path)
        return version, challenger

    def train_model(self) -> bool:
        """Train model from bar/tick data using Optuna-tuned params."""
        _ensure_ml_imports()

        optuna_cfg = cfg.OPTUNA_PARAMS.get(self.symbol)
        if not optuna_cfg:
            self.logger.log('WARNING', 'SovereignMLFilter', 'NO_OPTUNA_PARAMS',
                            f'{self.symbol}: no Optuna params, skipping')
            return False

        train_tf = optuna_cfg.get("timeframe", "H1")
        ticks = None

        # 1. Try bar data first (more history)
        bars = self._load_bars(train_tf)
        if bars is not None and bars.height >= 500:
            self.logger.log('INFO', 'SovereignMLFilter', 'USING_BAR_DATA',
                            f'{self.symbol}: loaded {bars.height} {train_tf} bars from parquets')
            spread_bps = {"crypto": 15.0, "forex": 2.0, "equity": 3.0,
                          "index": 2.0}.get(
                cfg.SYMBOLS.get(self.symbol, {}).get("asset_class", "equity"), 3.0)
        else:
            # Fallback: tick data
            ticks = self._load_ticks()
            if ticks is None:
                self.logger.log('WARNING', 'SovereignMLFilter', 'NO_DATA',
                                f'{self.symbol}: no bar or tick data found')
                return False
            if ticks.select(pl.col("size").sum()).item() <= 0:
                ticks = ticks.with_columns(pl.lit(1.0).alias("size"))
            bars = make_time_bars(ticks.select(["time", "price", "size"]), train_tf)
            spread_bps = infer_spread_bps(ticks)

        if bars.height < 500:
            self.logger.log('WARNING', 'SovereignMLFilter', 'INSUFFICIENT_BARS',
                            f'{self.symbol}: only {bars.height} bars (need 500+)')
            return False

        # 2. Build features
        slippage_bps = infer_slippage_bps(self.symbol)
        feat = build_bar_features(bars, z_threshold=0.0)

        # 4. Apply triple barrier
        tb = apply_triple_barrier(
            close=feat["close"].to_numpy(),
            vol_proxy=feat["vol20"].to_numpy(),
            side=feat["primary_side"].to_numpy(),
            horizon=6, pt_mult=2.0, sl_mult=1.5,
        )
        feat = feat.with_columns([
            pl.Series("label", tb.label),
            pl.Series("target", tb.label),
            pl.Series("tb_ret", tb.tb_ret),
            pl.Series("avg_win", tb.upside),
            pl.Series("avg_loss", tb.downside),
            pl.Series("upside", tb.upside),
            pl.Series("downside", tb.downside),
            pl.lit(3.0).alias("fee_bps"),
            pl.lit(float(spread_bps)).alias("spread_bps"),
            pl.lit(float(slippage_bps)).alias("slippage_bps"),
        ]).filter(pl.col("target").is_finite())
        feat = sanitize_training_frame(feat)

        if feat.height < cfg.TRAIN_SIZE:
            self.logger.log('WARNING', 'SovereignMLFilter', 'INSUFFICIENT_SAMPLES',
                            f'{self.symbol}: {feat.height} samples after sanitization')
            return False

        # 5. Use all available data (or cap at TRAIN_SIZE if set low for tick data)
        use_size = max(cfg.TRAIN_SIZE, feat.height)  # use all available
        train_df = feat.tail(use_size)
        x = train_df.select(FEATURE_COLUMNS).to_numpy()
        y = train_df["target"].to_numpy().astype(np.float32)
        avg_win = train_df["avg_win"].to_numpy().astype(np.float64)
        avg_loss = train_df["avg_loss"].to_numpy().astype(np.float64)
        costs = (
            (train_df["fee_bps"] + train_df["spread_bps"] + train_df["slippage_bps"] * 2.0).to_numpy() / 1e4
        ).astype(np.float64)

        # 6. Train XGBoost with custom EV objective
        xgb_params = optuna_cfg["xgb_params"].copy()
        num_boost_round = optuna_cfg["num_boost_round"]
        obj = make_ev_custom_objective(
            float(np.mean(avg_win)),
            float(np.mean(avg_loss)),
            float(np.mean(costs)),
        )

        dtrain = xgb.DMatrix(x, label=y)
        try:
            bst = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                obj=obj,
                verbose_eval=False,
            )
        except Exception as e:
            # GPU fallback to CPU
            self.logger.log('WARNING', 'SovereignMLFilter', 'GPU_FALLBACK',
                            f'{self.symbol}: GPU train failed ({e}), trying CPU')
            xgb_params["device"] = "cpu"
            xgb_params.pop("sampling_method", None)
            bst = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                obj=obj,
                verbose_eval=False,
            )

        # 7. Save model
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        bst.save_model(self.model_path())
        self.model = bst

        # F15: Train TabNet ensemble (optional, graceful if torch not available)
        try:
            from engine.deep_ensemble import DeepEnsemble, _torch_available
            if _torch_available():
                # Use 80/20 train/val split for TabNet
                split_idx = int(len(x) * 0.8)
                ensemble = DeepEnsemble(self.symbol)
                success = ensemble.train_tabnet(
                    x[:split_idx], y[:split_idx],
                    x[split_idx:], y[split_idx:],
                )
                if success:
                    ensemble.save()
                    self._ensemble = ensemble
                    self.logger.log('INFO', 'SovereignMLFilter', 'TABNET_TRAINED',
                                    f'{self.symbol}: TabNet trained on {split_idx} samples')
        except Exception as e:
            self.logger.log('DEBUG', 'SovereignMLFilter', 'TABNET_SKIP',
                            f'{self.symbol}: TabNet training skipped ({e})')

        # F13: Save version via registry
        registry = self._get_registry()
        if registry:
            metadata = {
                "train_samples": int(len(y)),
                "optuna_ev": float(optuna_cfg.get("optuna_ev", 0)),
                "num_boost_round": int(num_boost_round),
            }
            version = registry.save_version(self.symbol, bst, metadata)
            self._model_version = version

        self.logger.log('INFO', 'SovereignMLFilter', 'MODEL_TRAINED',
                        f'{self.symbol}: trained on {len(y)} samples, saved to {self.model_path()}')
        return True

    def _load_bars(self, timeframe: str, max_years: int = 10):
        """Load bar data from pre-downloaded parquets."""
        _ensure_ml_imports()
        from datetime import datetime, timedelta, timezone
        from pathlib import Path

        bar_roots = getattr(cfg, 'BAR_ROOTS', [])
        if not bar_roots:
            return None

        variants = [self.symbol, self.symbol.replace("_", ""), self.symbol.replace("_", "/")]
        for root in bar_roots:
            for variant in variants:
                bar_dir = Path(root) / timeframe / variant
                if not bar_dir.is_dir():
                    continue
                pq_files = sorted(bar_dir.glob("*.parquet"))
                if not pq_files:
                    continue
                dfs = []
                for pf in pq_files:
                    try:
                        dfs.append(pl.read_parquet(pf))
                    except Exception:
                        continue
                if not dfs:
                    continue
                combined = pl.concat(dfs, how="vertical").sort("time")
                # Normalise columns
                rename = {}
                for col in combined.columns:
                    if col.lower() in ("time", "open", "high", "low", "close",
                                       "volume", "tick_volume"):
                        rename[col] = col.lower()
                combined = combined.rename(rename)
                if "volume" not in combined.columns and "tick_volume" in combined.columns:
                    combined = combined.rename({"tick_volume": "volume"})
                elif "volume" in combined.columns and "tick_volume" in combined.columns:
                    if combined.select(pl.col("volume").sum()).item() == 0:
                        combined = combined.drop("volume").rename({"tick_volume": "volume"})
                # Ensure datetime
                if combined["time"].dtype in (pl.Int64, pl.UInt64):
                    combined = combined.with_columns(
                        (pl.col("time") * 1_000_000).cast(pl.Datetime("us", "UTC")).alias("time")
                    )
                cols = ["time", "open", "high", "low", "close", "volume"]
                missing = [c for c in cols if c not in combined.columns]
                if missing:
                    continue
                combined = combined.select(cols).drop_nulls()
                # Trim to max_years
                cutoff = datetime.now(timezone.utc) - timedelta(days=max_years * 365)
                time_dtype = combined["time"].dtype
                combined = combined.filter(
                    pl.col("time") >= pl.lit(cutoff).cast(time_dtype)
                )
                return combined
        return None

    def _load_ticks(self):
        """Load tick data from parquet files."""
        _ensure_ml_imports()
        frames = []
        for root in cfg.DATA_ROOTS:
            sym_dir = os.path.join(root, self.symbol)
            if not os.path.isdir(sym_dir):
                continue
            for f in sorted(os.listdir(sym_dir)):
                if f.endswith(".parquet"):
                    df = pl.read_parquet(os.path.join(sym_dir, f)).select(
                        ["time", "bid", "ask", "last", "volume", "volume_real"]
                    )
                    if df.height > 0:
                        frames.append(df)
        if not frames:
            return None

        d = (
            pl.concat(frames, how="vertical")
            .sort("time")
            .with_columns([
                pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
                pl.when(pl.col("last") > 0)
                .then(pl.col("last"))
                .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
                .alias("price"),
                pl.when(pl.col("volume_real") > 0)
                .then(pl.col("volume_real"))
                .otherwise(pl.col("volume"))
                .alias("size"),
            ])
            .drop_nulls(["time", "price", "size"])
        )
        return d.select(["time", "bid", "ask", "price", "size"])

    def predict(self, features_np: np.ndarray) -> float:
        """Run inference on feature array. Returns probability [0, 1]."""
        if self.model is None:
            return 0.5
        _ensure_ml_imports()

        # Handle feature count mismatch: old models trained on 28, new on 39
        model_num_features = int(self.model.num_features())
        if features_np.shape[1] > model_num_features:
            features_np = features_np[:, :model_num_features]

        dmat = xgb.DMatrix(features_np)
        raw = self.model.predict(dmat)
        xgb_proba = float(raw[0])

        # F15: Ensemble with TabNet if available
        if self._ensemble is not None and self._ensemble.is_available:
            return self._ensemble.predict(features_np, xgb_proba)

        return xgb_proba

    def _load_ensemble(self):
        """Lazy-load TabNet ensemble model (F15)."""
        if self._ensemble is not None:
            return
        try:
            from engine.deep_ensemble import DeepEnsemble
            self._ensemble = DeepEnsemble(self.symbol)
            if self._ensemble.load():
                self.logger.log('INFO', 'SovereignMLFilter', 'TABNET_LOADED',
                                f'{self.symbol}: TabNet ensemble loaded')
            else:
                self._ensemble = DeepEnsemble(self.symbol)  # Keep instance but no model
        except Exception:
            self._ensemble = None

    def should_trade(self, features_np: np.ndarray, primary_side: int,
                     threshold: float | None = None) -> tuple[bool, float, str | None]:
        """
        Returns (should_trade, confidence, direction).
        z_threshold=0 means primary_side is always +/-1 (never 0).
        ML model alone decides whether to trade.
        Uses per-symbol threshold from sovereign_configs if available.
        """
        proba = self.predict(features_np)
        thr = threshold if threshold is not None else cfg.ML_THRESHOLD
        should = proba >= thr

        if primary_side > 0:
            direction = 'BUY'
        elif primary_side < 0:
            direction = 'SELL'
        else:
            direction = None
            should = False

        return should, proba, direction
