"""
Multi-timeframe scanner — runs symbols on their optimal timeframe.

Symbols in config/multi_tf.yaml get scanned on M15/M30/H4 etc.
Symbols NOT in multi_tf.yaml stay on the default H1 loop.

Usage from run_bot.py:
    scanner = MultiTFScanner(bot)
    scanner.load_config()

    # In the main loop, call this every ~10 seconds:
    scanner.tick()  # fires scans when timeframe boundaries are hit
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from config.loader import cfg
from engine.inference import SovereignMLFilter, _ensure_ml_imports
from engine.signal import get_h1_features, _sentiment_adjust, ScanCache

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "multi_tf.yaml"

# Timeframe → (minutes per bar, MT5 timeframe attribute name)
TF_MAP = {
    "M5":  (5,   "TIMEFRAME_M5"),
    "M15": (15,  "TIMEFRAME_M15"),
    "M30": (30,  "TIMEFRAME_M30"),
    "H1":  (60,  "TIMEFRAME_H1"),
    "H4":  (240, "TIMEFRAME_H4"),
}


class MultiTFScanner:
    """Manages scanning symbols on different timeframes."""

    def __init__(self, bot):
        """
        bot: SovereignBot instance (has .logger, .order_router, .filters,
             .decay_tracker, .trading_schedule, .feature_logger, .discord)
        """
        self.bot = bot
        self.symbols: dict[str, dict] = {}      # symbol → {timeframe, ...}
        self.filters: dict[str, SovereignMLFilter] = {}
        self._last_scan: dict[str, float] = {}  # timeframe → last scan epoch
        self._tf_groups: dict[str, list[str]] = {}  # timeframe → [symbols]
        self._last_bar_time: dict[str, int] = {}  # symbol → last bar epoch (per ticker)

    def load_config(self, allowed_symbols: set[str] | None = None) -> int:
        """Load multi_tf.yaml. Returns number of symbols loaded."""
        if not CONFIG_PATH.exists():
            return 0

        try:
            with open(CONFIG_PATH) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return 0

        self.symbols = data.get("symbols", {}) or {}
        if not self.symbols:
            return 0

        # Filter to account symbols if provided
        if allowed_symbols is not None:
            self.symbols = {sym: scfg for sym, scfg in self.symbols.items()
                            if sym in allowed_symbols}

        # Group symbols by timeframe
        self._tf_groups = {}
        for sym, sym_cfg in self.symbols.items():
            tf = sym_cfg.get("timeframe", "H1")
            if tf == "H1":
                continue  # H1 symbols stay in the default loop
            if tf not in TF_MAP:
                self.bot.logger.log('WARNING', 'MultiTF', 'BAD_TIMEFRAME',
                                    f'{sym}: unknown timeframe {tf}, skipping')
                continue
            self._tf_groups.setdefault(tf, []).append(sym)

        # Load models for multi-TF symbols
        _ensure_ml_imports()
        loaded = 0
        for sym in self.symbols:
            tf = self.symbols[sym].get("timeframe", "H1")
            if tf == "H1":
                continue
            filt = SovereignMLFilter(sym, self.bot.logger)
            if filt.load_model():
                self.filters[sym] = filt
                loaded += 1
            else:
                self.bot.logger.log('WARNING', 'MultiTF', 'NO_MODEL',
                                    f'{sym}: no model found, skipping')

        if loaded > 0:
            tf_summary = ", ".join(f"{tf}:{len(syms)}" for tf, syms in self._tf_groups.items())
            self.bot.logger.log('INFO', 'MultiTF', 'LOADED',
                                f'{loaded} symbols on non-H1 timeframes: {tf_summary}')

        return loaded

    def get_multi_tf_symbols(self) -> set[str]:
        """Return symbols managed by multi-TF (excluded from default H1 loop)."""
        return {sym for sym, cfg in self.symbols.items()
                if cfg.get("timeframe", "H1") != "H1"}

    def tick(self, mt5) -> tuple[int, int]:
        """Called every ~10s from main loop. Fires scans when timeframe boundaries hit.

        Also triggers a preload ~15s before each bar close so the scan is fast.

        Returns (total_signals_found, total_signals_executed).
        """
        if not self._tf_groups:
            return 0, 0

        total_found = 0
        total_executed = 0
        now = datetime.now()

        for tf, symbols in self._tf_groups.items():
            minutes, _ = TF_MAP[tf]

            # Preload ~15s before bar close
            if self._should_preload(tf, minutes, now):
                eligible = [
                    s for s in symbols
                    if not self.bot.decay_tracker.is_disabled(s)
                    and self.bot.trading_schedule.is_trading_open(s)[0]
                    and s in self.filters and self.filters[s].model is not None
                ]
                if eligible:
                    cache = getattr(self.bot, 'scan_cache', None)
                    if cache and not cache.is_warm():
                        self.bot.logger.log('INFO', 'ScanCache', 'MTF_PRELOAD_START',
                                            f'{tf}: {len(eligible)} symbols')
                        cache.preload(eligible, mt5, self.bot.logger)

            if self._should_scan(tf, minutes, now):
                found, executed = self._scan_timeframe(tf, symbols, mt5)
                total_found += found
                total_executed += executed
                self._last_scan[tf] = time.time()

        return total_found, total_executed

    def _should_preload(self, tf: str, minutes: int, now: datetime) -> bool:
        """True ~15s before the next bar close for *tf*."""
        if minutes <= 60:
            # M15, M30, H1: minute-based calculation works
            secs_into_bar = (now.minute % minutes) * 60 + now.second
        else:
            # H4+: use full hour+minute calculation
            total_minutes = now.hour * 60 + now.minute
            secs_into_bar = (total_minutes % minutes) * 60 + now.second

        bar_duration = minutes * 60
        secs_until_close = bar_duration - secs_into_bar

        # Fire when 10-20 seconds remain (wide enough for the 10s tick interval)
        if not (10 <= secs_until_close <= 20):
            return False

        # Don't preload if we already preloaded recently
        last = self._last_scan.get(f"_preload_{tf}", 0)
        if time.time() - last < bar_duration * 0.9:
            return False

        self._last_scan[f"_preload_{tf}"] = time.time()
        return True

    def _should_scan(self, tf: str, minutes: int, now: datetime) -> bool:
        """Check if we should scan this timeframe now.

        For sub-hourly TFs (M15, M30): fire at minute boundaries.
        For H4+: fire every hour at :00 — per-symbol new-bar checks happen
        inside _scan_timeframe so each ticker's candle schedule is respected.
        """
        if now.second > 30:
            return False

        if minutes <= 60:
            # M15, M30, H1: minute-based boundary check works fine
            if now.minute % minutes != 0:
                return False
            last = self._last_scan.get(tf, 0)
            if time.time() - last < minutes * 60 * 0.9:
                return False
            return True
        else:
            # H4+: check every hour at :00, per-symbol bar check filters inside
            if now.minute != 0:
                return False
            # Short cooldown (50 min) to prevent double-fire in the same hour
            last = self._last_scan.get(tf, 0)
            if time.time() - last < 50 * 60:
                return False
            return True

    def _has_new_bar(self, symbol: str, mt5, mt5_tf) -> bool:
        """Check if this specific symbol has a new bar since we last scanned it."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 2)
            if not rates or len(rates) < 2:
                return True  # Can't verify, allow scan
            # Use second-to-last bar (last completed bar), not the forming bar
            bar_time = int(rates[-2]['time'])
            last_bar = self._last_bar_time.get(symbol, 0)
            if bar_time > last_bar:
                self._last_bar_time[symbol] = bar_time
                return True
            return False
        except Exception:
            return True  # Can't verify, allow scan

    def _scan_timeframe(self, tf: str, symbols: list[str], mt5) -> tuple[int, int]:
        """Scan all symbols for a specific timeframe. Returns (found, executed)."""
        _ensure_ml_imports()
        from engine.inference import FEATURE_COLUMNS

        minutes, mt5_tf_attr = TF_MAP[tf]
        mt5_tf = getattr(mt5, mt5_tf_attr, None)
        if mt5_tf is None:
            self.bot.logger.log('ERROR', 'MultiTF', 'NO_MT5_TF',
                                f'MT5 has no attribute {mt5_tf_attr}')
            return 0, 0

        # For H4+: filter to symbols that actually have a new bar
        if minutes > 60:
            eligible = [s for s in symbols if self._has_new_bar(s, mt5, mt5_tf)]
            if not eligible:
                return 0, 0
        else:
            eligible = symbols

        self.bot.logger.log('INFO', 'MultiTF', 'SCAN_START',
                            f'{tf}: scanning {len(eligible)} symbols')

        sentiment_cache = getattr(self.bot.order_router, '_cached_sentiment', {})
        found = 0
        executed = 0
        scan_results = []
        t0 = time.time()

        for symbol in eligible:
            if self.bot.emergency_stop:
                break

            filt = self.filters.get(symbol)
            if filt is None or filt.model is None:
                continue

            if self.bot.decay_tracker.is_disabled(symbol):
                continue

            is_open, _ = self.bot.trading_schedule.is_trading_open(symbol)
            if not is_open:
                continue

            # Get features using the correct timeframe
            features_np, primary_side = self._get_features(
                symbol, mt5, mt5_tf, tf)
            if features_np is None:
                continue

            # Inference
            should_trade, confidence, direction = filt.should_trade(
                features_np, primary_side)

            # Sentiment adjustment
            confidence, sent_boost = _sentiment_adjust(
                confidence, direction, sentiment_cache, symbol)
            # Per-symbol threshold (from sovereign_configs.json), fallback to global
            sym_threshold = cfg.SYMBOLS.get(symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)
            should_trade = confidence >= sym_threshold and direction is not None

            # Build features dict
            features_dict = {}
            for i, col in enumerate(FEATURE_COLUMNS):
                features_dict[col] = float(features_np[0][i])
            features_dict["_model_version"] = getattr(filt, '_model_version', 'default')
            features_dict["_sentiment_boost"] = sent_boost
            features_dict["_timeframe"] = tf

            # Collect scan result for LLM commentary
            scan_results.append({
                "symbol": symbol,
                "side": primary_side,
                "proba": confidence,
                "status": "SIGNAL" if should_trade else "skip",
                "reason": (direction or "no dir") if should_trade else "onder threshold",
                "z20": features_dict.get("z20", 0),
                "rsi14": features_dict.get("rsi14", 0),
                "vol20": features_dict.get("vol20", 0),
            })

            if not should_trade or direction is None:
                sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
                thr_info = f" thr={sym_threshold:.2f}" if sym_threshold != cfg.ML_THRESHOLD else ""
                self.bot.logger.log('DEBUG', 'MultiTF', 'NO_SIGNAL',
                                    f'{symbol}[{tf}]: proba={confidence:.3f}{sent_info}{thr_info}')
                continue

            found += 1
            sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
            thr_info = f" thr={sym_threshold:.2f}" if sym_threshold != cfg.ML_THRESHOLD else ""
            self.bot.logger.log('INFO', 'MultiTF', 'SIGNAL',
                                f'{symbol}[{tf}] {direction} (proba={confidence:.3f}{sent_info}{thr_info})')

            success = self.bot.execute_trade(symbol, direction, confidence,
                                             features_dict=features_dict)
            if success:
                executed += 1
                scan_results[-1]["status"] = "TRADE"
                scan_results[-1]["reason"] = "trade geplaatst"

        ms = (time.time() - t0) * 1000
        self.bot.logger.log('INFO', 'MultiTF', 'SCAN_DONE',
                            f'{tf}: {len(symbols)} symbols in {ms:.0f}ms, '
                            f'signals={found}, executed={executed}')

        # LLM commentary + Discord notification
        if scan_results:
            llm_callback = getattr(self.bot, '_llm_mtf_commentary', None)
            if llm_callback:
                llm_callback(tf, scan_results, found, executed)

        return found, executed

    def _get_features(self, symbol, mt5, mt5_tf, tf_name):
        """Fetch bars on the given timeframe and build features.

        Same logic as get_h1_features but with configurable timeframe.
        """
        _ensure_ml_imports()
        from engine.inference import pl, build_bar_features, FEATURE_COLUMNS
        from datetime import timezone

        # Scale bar count: need 200 H1 bars equivalent
        minutes, _ = TF_MAP[tf_name]
        bars_needed = max(200, int(200 * 60 / minutes))  # More bars for shorter TFs

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars_needed)
        if rates is None or len(rates) < 100:
            return None, None

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

        import numpy as np

        try:
            feat = build_bar_features(bars, z_threshold=0.0)
        except Exception as e:
            self.bot.logger.log('ERROR', 'MultiTF', 'FEATURE_ERROR',
                                f'{symbol}[{tf_name}]: {e}')
            return None, None

        if feat.height < 2:
            return None, None

        # Lead-lag features (use bot scan_cache if warm)
        cache = getattr(self.bot, 'scan_cache', None)
        try:
            if cache and cache.is_warm() and symbol in cache.lead_lag:
                ll = cache.lead_lag[symbol]
            else:
                from engine.lead_lag import build_lead_lag_features
                ll = build_lead_lag_features(symbol, mt5, self.bot.logger)
            if ll:
                feat = feat.with_columns([
                    pl.lit(ll.get("leader_ret1", 0.0)).alias("leader_ret1"),
                    pl.lit(ll.get("leader_ret3", 0.0)).alias("leader_ret3"),
                    pl.lit(ll.get("leader_momentum", 0.0)).alias("leader_momentum"),
                ])
        except Exception:
            pass

        # Tick features (use bot scan_cache if warm)
        try:
            from engine.tick_features import build_tick_features
            if cache and cache.is_warm() and symbol in cache.tick_data:
                tick_data = cache.tick_data[symbol]
            else:
                from engine.signal import _load_recent_ticks
                tick_data = _load_recent_ticks(symbol, self.bot.logger)
            if tick_data is not None and tick_data.height > 0:
                feat = build_tick_features(tick_data, feat)
        except Exception:
            pass

        last_row = feat.tail(1)
        features_np = last_row.select(FEATURE_COLUMNS).to_numpy()

        if not np.all(np.isfinite(features_np)):
            return None, None

        primary_side = int(last_row["primary_side"][0])
        return features_np, primary_side
