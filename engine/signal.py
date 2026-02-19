"""
Signal generation — H1 bars from MT5 → features → batch inference.

Extracted from SovereignBot.get_h1_features() + check_signals().
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np

from config.loader import cfg
from engine.inference import _ensure_ml_imports


# ---------------------------------------------------------------------------
# ScanCache — preload heavy I/O data before H1 bar close
# ---------------------------------------------------------------------------

class ScanCache:
    """Pre-loads tick parquets and lead-lag data so the :00 scan is fast."""

    def __init__(self):
        self.tick_data: dict = {}      # symbol → DataFrame | None
        self.lead_lag: dict = {}       # symbol → {leader_ret1, leader_ret3, leader_momentum}
        self.timestamp: float = 0      # epoch when cache was filled
        self._loading: bool = False

    def is_warm(self, max_age_seconds: int = 180) -> bool:
        """True if cache was filled within *max_age_seconds*."""
        return (time.time() - self.timestamp) < max_age_seconds

    def preload(self, symbols: list[str], mt5, logger) -> None:
        """Load tick data + lead-lag for all *symbols*.  ~6-8 s one-off."""
        self._loading = True
        t0 = time.time()

        # 1) Tick data from disk (the most expensive operation)
        for sym in symbols:
            try:
                self.tick_data[sym] = _load_recent_ticks(sym, logger)
            except Exception:
                self.tick_data[sym] = None

        # 2) Lead-lag features (many MT5 bridge calls)
        from engine.lead_lag import build_lead_lag_features
        for sym in symbols:
            try:
                self.lead_lag[sym] = build_lead_lag_features(sym, mt5, logger) or {}
            except Exception:
                self.lead_lag[sym] = {}

        self.timestamp = time.time()
        self._loading = False
        ms = (time.time() - t0) * 1000
        logger.log('INFO', 'ScanCache', 'PRELOADED',
                    f'{len(symbols)} symbols in {ms:.0f}ms')

    def clear(self) -> None:
        self.tick_data.clear()
        self.lead_lag.clear()
        self.timestamp = 0


def _load_recent_ticks(symbol: str, logger):
    """Try to load recent tick data from parquet files for tick features."""
    try:
        import os
        from engine.inference import pl
        for root in cfg.DATA_ROOTS:
            sym_dir = os.path.join(root, symbol)
            if not os.path.isdir(sym_dir):
                continue
            files = sorted(f for f in os.listdir(sym_dir) if f.endswith(".parquet"))
            if not files:
                continue
            # Load most recent file
            latest = os.path.join(sym_dir, files[-1])
            df = pl.read_parquet(latest).select(
                ["time", "bid", "ask", "last", "volume", "volume_real"]
            )
            if df.height > 0:
                return df.with_columns([
                    pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
                    pl.when(pl.col("last") > 0)
                    .then(pl.col("last"))
                    .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
                    .alias("price"),
                    pl.when(pl.col("volume_real") > 0)
                    .then(pl.col("volume_real"))
                    .otherwise(pl.col("volume"))
                    .alias("size"),
                ]).select(["time", "bid", "ask", "price", "size"])
    except Exception:
        pass
    return None


def get_h1_features(symbol: str, mt5, logger, cache: ScanCache | None = None):
    """Fetch H1 bars from MT5 and build features.

    When *cache* is warm, tick data and lead-lag features are read from it
    instead of hitting disk / MT5 again (preloaded at :58).

    Returns (features_np, primary_side) or (None, None).
    """
    if mt5 is None:
        return None, None

    _ensure_ml_imports()
    from engine.inference import pl, build_bar_features, FEATURE_COLUMNS  # noqa: F811

    # Need 200 bars for rolling windows (vol20, ret48, etc.)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
    if rates is None or len(rates) < 100:
        logger.log('WARNING', 'Signal', 'INSUFFICIENT_BARS',
                    f'{symbol}: only got {len(rates) if rates is not None else 0} bars')
        return None, None

    # Convert MT5 structured array to Polars DataFrame
    bars = pl.DataFrame({
        "time": [datetime.fromtimestamp(int(r['time']), tz=timezone.utc) for r in rates],
        "open": [float(r['open']) for r in rates],
        "high": [float(r['high']) for r in rates],
        "low": [float(r['low']) for r in rates],
        "close": [float(r['close']) for r in rates],
        "volume": [float(r['tick_volume']) for r in rates],
    }).with_columns(
        pl.col("time").cast(pl.Datetime("us", "UTC"))
    )

    # Build features
    try:
        feat = build_bar_features(bars, z_threshold=0.0)
    except Exception as e:
        logger.log('ERROR', 'Signal', 'FEATURE_ERROR', f'{symbol}: {e}')
        return None, None

    if feat.height < 2:
        return None, None

    # F8: Add lead-lag features from leader symbols
    try:
        if cache and cache.is_warm() and symbol in cache.lead_lag:
            ll_features = cache.lead_lag[symbol]
        else:
            from engine.lead_lag import build_lead_lag_features
            ll_features = build_lead_lag_features(symbol, mt5, logger)
        if ll_features:
            feat = feat.with_columns([
                pl.lit(ll_features.get("leader_ret1", 0.0)).alias("leader_ret1"),
                pl.lit(ll_features.get("leader_ret3", 0.0)).alias("leader_ret3"),
                pl.lit(ll_features.get("leader_momentum", 0.0)).alias("leader_momentum"),
            ])
    except Exception as e:
        logger.log('DEBUG', 'Signal', 'LEAD_LAG_ERROR', f'{symbol}: {e}')

    # F2: Add tick features if tick data available
    try:
        from engine.tick_features import build_tick_features
        if cache and cache.is_warm() and symbol in cache.tick_data:
            tick_data = cache.tick_data[symbol]
        else:
            tick_data = _load_recent_ticks(symbol, logger)
        if tick_data is not None and tick_data.height > 0:
            feat = build_tick_features(tick_data, feat)
    except Exception as e:
        logger.log('DEBUG', 'Signal', 'TICK_FEATURES_ERROR', f'{symbol}: {e}')

    # Get the last row's features (shift(1) safe — uses only completed bar data)
    last_row = feat.tail(1)
    features_np = last_row.select(FEATURE_COLUMNS).to_numpy()

    # Check for NaN/Inf
    if not np.all(np.isfinite(features_np)):
        logger.log('WARNING', 'Signal', 'NAN_FEATURES',
                    f'{symbol}: features contain NaN/Inf')
        return None, None

    primary_side = int(last_row["primary_side"][0])
    return features_np, primary_side


def _sentiment_adjust(confidence: float, direction: str | None,
                      sentiment_cache: dict, symbol: str) -> tuple[float, float]:
    """Adjust ML proba with sentiment. Returns (adjusted_proba, sentiment_boost).

    sentiment_cache maps symbol → score in [-1, +1].
    Boost = score * SCALE * direction_sign  (max ±0.03 on proba).
    Positive sentiment helps BUY, hurts SELL and vice versa.
    Broad market sentiment (_BROAD_*) is mixed in at half weight.
    """
    SCALE = 0.03
    if not sentiment_cache or direction is None:
        return confidence, 0.0

    # Symbol-specific sentiment
    sym_score = sentiment_cache.get(symbol, 0.0)

    # Broad sentiment (half weight)
    broad_keys = [k for k in sentiment_cache if k.startswith("_BROAD_")]
    broad_avg = (sum(sentiment_cache[k] for k in broad_keys) / len(broad_keys)
                 if broad_keys else 0.0)
    combined = sym_score + broad_avg * 0.5

    # Direction sign: BUY benefits from positive sentiment, SELL from negative
    dir_sign = 1.0 if direction == "BUY" else -1.0
    boost = combined * SCALE * dir_sign

    # Clamp boost to ±SCALE
    boost = max(-SCALE, min(SCALE, boost))
    return confidence + boost, boost


def check_signals(engine, filters, decay_tracker, trading_schedule,
                   feature_logger, discord, mt5, llm_scan_callback=None,
                   cache: ScanCache | None = None):
    """Check all symbols for H1 signals.

    Uses batch GPU inference: gathers all features first,
    then runs all models in one pass to minimize GPU context switches.
    When *cache* is warm, tick/lead-lag data is served from cache.

    Returns (signals_found, signals_executed, candidates_scanned).
    """
    _ensure_ml_imports()
    from engine.inference import FEATURE_COLUMNS

    signals_found = 0
    signals_executed = 0

    # Grab cached sentiment from order router
    sentiment_cache = getattr(engine.order_router, '_cached_sentiment', {})

    # Run model decay audit first
    newly_disabled = decay_tracker.audit_all(discord)
    if newly_disabled:
        engine.logger.log('WARNING', 'Signal', 'DECAY_DISABLED',
                          f'Disabled {len(newly_disabled)} symbols: {newly_disabled}')

    # Phase 1: Gather features for all eligible symbols (CPU-bound MT5 calls)
    # Skip symbols managed by multi-TF scanner (they run on their own timeframe)
    mtf_symbols = engine.multi_tf.get_multi_tf_symbols() if getattr(engine, 'multi_tf', None) else set()
    candidates = []
    for symbol in cfg.SYMBOLS:
        if engine.emergency_stop:
            break
        if symbol in mtf_symbols:
            continue
        if decay_tracker.is_disabled(symbol):
            continue
        is_open, _ = trading_schedule.is_trading_open(symbol)
        if not is_open:
            continue
        filt = filters.get(symbol)
        if filt is None or filt.model is None:
            continue

        features_np, primary_side = get_h1_features(symbol, mt5, engine.logger, cache=cache)
        if features_np is None:
            continue
        candidates.append((symbol, filt, features_np, primary_side))

    # Phase 2: Batch inference — one DMatrix per model (GPU stays hot)
    t0 = time.time()
    scan_results = []
    for symbol, filt, features_np, primary_side in candidates:
        if engine.emergency_stop:
            break

        # Per-symbol threshold (from sovereign_configs.json), fallback to global
        sym_threshold = cfg.SYMBOLS.get(symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)

        # F13: A/B test — optionally use challenger model
        used_version = getattr(filt, '_model_version', 'default')
        challenger_version, challenger_model = None, None
        if cfg.AB_TEST_ENABLED:
            challenger_version, challenger_model = filt.load_challenger()

        if challenger_model is not None:
            # Use challenger for this inference
            from engine.inference import xgb
            dmat = xgb.DMatrix(features_np)
            raw = challenger_model.predict(dmat)
            challenger_proba = float(raw[0])
            confidence = challenger_proba
            used_version = challenger_version
            if features_np[0][FEATURE_COLUMNS.index("z20")] > 0:
                direction = 'SELL'
            elif features_np[0][FEATURE_COLUMNS.index("z20")] < 0:
                direction = 'BUY'
            else:
                direction = None
            # Sentiment adjustment on challenger proba
            confidence, sent_boost = _sentiment_adjust(
                confidence, direction, sentiment_cache, symbol)
            should_trade = confidence >= sym_threshold and direction is not None
        else:
            should_trade, confidence, direction = filt.should_trade(
                features_np, primary_side, threshold=sym_threshold)
            # Sentiment adjustment on ML proba
            raw_confidence = confidence
            confidence, sent_boost = _sentiment_adjust(
                confidence, direction, sentiment_cache, symbol)
            should_trade = confidence >= sym_threshold and direction is not None

        # Build features dict for logging
        features_dict = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            features_dict[col] = float(features_np[0][i])
        features_dict["_model_version"] = used_version
        features_dict["_sentiment_boost"] = sent_boost

        if not should_trade or direction is None:
            sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
            engine.logger.log('DEBUG', 'Signal', 'NO_SIGNAL',
                              f'{symbol}: proba={confidence:.3f}{sent_info} side={primary_side}')
            feature_logger.log_trade_features(
                symbol, direction or "NONE", confidence, features_dict,
                status="FILTERED"
            )
            if primary_side == 0:
                reason = "z20 exact nul (zeer zeldzaam)"
            elif confidence < sym_threshold:
                reason = f"proba {confidence:.3f} < threshold {sym_threshold}"
                if sent_boost != 0:
                    reason += f" (sentiment {sent_boost:+.3f})"
            else:
                reason = "geen richting"
            scan_results.append({
                "symbol": symbol, "proba": confidence, "side": primary_side,
                "direction": direction or "NONE", "status": "SKIP",
                "reason": reason,
                "z20": features_dict.get("z20", 0),
                "rsi14": features_dict.get("rsi14", 0),
                "vol20": features_dict.get("vol20", 0),
            })
            continue

        signals_found += 1
        sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
        engine.logger.log('INFO', 'Signal', 'SIGNAL',
                          f'{symbol} {direction} (proba={confidence:.3f}{sent_info})')

        feature_logger.log_trade_features(
            symbol, direction, confidence, features_dict,
            status="SIGNAL"
        )

        success = engine.execute_trade(symbol, direction, confidence,
                                       features_dict=features_dict)
        if success:
            signals_executed += 1
            feature_logger.log_trade_features(
                symbol, direction, confidence, features_dict,
                status="EXECUTED"
            )
        reject_reason = getattr(engine.order_router, 'last_reject_reason', None)
        scan_results.append({
            "symbol": symbol, "proba": confidence, "side": primary_side,
            "direction": direction, "status": "EXECUTED" if success else "BLOCKED",
            "reason": "trade geplaatst" if success else (reject_reason or "guardrail blokkade"),
            "z20": features_dict.get("z20", 0),
            "rsi14": features_dict.get("rsi14", 0),
            "vol20": features_dict.get("vol20", 0),
        })

    inference_ms = (time.time() - t0) * 1000
    engine.logger.log('INFO', 'Signal', 'BATCH_INFERENCE',
                      f'{len(candidates)} symbols in {inference_ms:.0f}ms')

    # LLM scan commentary
    if llm_scan_callback and scan_results:
        llm_scan_callback(scan_results, signals_found, signals_executed)

    # Flush feature log
    feature_logger.flush()
    return signals_found, signals_executed, len(candidates)
