"""
Multi-timeframe scanner — runs ALL symbols on ALL timeframes simultaneously.

Every TF from M1 to H4 is treated equally. No H1 bias.
Scans fire when timeframe boundaries are hit. Feature computation + ML
inference run in parallel across ALL ready TFs using ThreadPoolExecutor.

Returns Signal objects to caller (MasterOrderGenerator handles execution).

Usage from run_bot.py:
    scanner = MultiTFScanner(bot)
    scanner.load_config()

    # In the main loop, call this every ~5 seconds:
    signals = scanner.tick(mt5)  # returns list[_Signal], no execution
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import yaml

from config.loader import cfg
from engine.inference import SovereignMLFilter, _ensure_ml_imports
from engine.signal import _sentiment_adjust, ScanCache

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "multi_tf.yaml"

# Timeframe → (minutes per bar, MT5 timeframe attribute name)
TF_MAP = {
    "M1":  (1,   "TIMEFRAME_M1"),
    "M5":  (5,   "TIMEFRAME_M5"),
    "M15": (15,  "TIMEFRAME_M15"),
    "M30": (30,  "TIMEFRAME_M30"),
    "H1":  (60,  "TIMEFRAME_H1"),
    "H4":  (240, "TIMEFRAME_H4"),
}

# Scale workers to available CPU cores (clamped 4–16)
_MAX_WORKERS = max(4, min(16, (os.cpu_count() or 4)))


@dataclass
class _Signal:
    """Result from parallel feature+inference computation."""
    symbol: str
    tf: str
    should_trade: bool
    confidence: float
    direction: str | None
    features_dict: dict
    primary_side: int
    sent_boost: float
    scan_info: dict  # for LLM commentary


class MultiTFScanner:
    """Manages scanning ALL symbols on their assigned timeframe (M1–H4).

    Three-phase pipeline on each tick:
      1. Batch MT5 data fetch   (main thread — MT5 not thread-safe)
      2. Feature + ML inference  (ThreadPoolExecutor — CPU only, no MT5)
      3. Trade execution         (main thread — MT5 not thread-safe)
    """

    def __init__(self, bot):
        self.bot = bot
        self.symbols: dict[str, dict] = {}
        self.filters: dict[str, SovereignMLFilter] = {}
        self._last_scan: dict[str, float] = {}
        self._tf_groups: dict[str, list[str]] = {}
        self._last_bar_time: dict[str, int] = {}

    # ── Config loading ────────────────────────────────────────────

    def load_config(self, allowed_symbols: set[str] | None = None) -> int:
        """Load symbol→TF assignments. Returns number of symbols loaded.

        Sources (in priority order):
          1. cfg.SYMBOLS[sym].atr_timeframe  (from bf/config.json)
          2. multi_tf.yaml
          3. Default: H1
        """
        # Start with multi_tf.yaml
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH) as f:
                    data = yaml.safe_load(f) or {}
                self.symbols = data.get("symbols", {}) or {}
            except Exception:
                self.symbols = {}
        else:
            self.symbols = {}

        # Ensure ALL account symbols are included (even if not in multi_tf.yaml)
        for sym in cfg.SYMBOLS:
            if sym not in self.symbols:
                self.symbols[sym] = {"timeframe": "H1"}

        # Filter to account symbols if provided
        if allowed_symbols is not None:
            self.symbols = {sym: scfg for sym, scfg in self.symbols.items()
                            if sym in allowed_symbols}

        # Group symbols by timeframe
        self._tf_groups = {}
        for sym, sym_cfg in self.symbols.items():
            # Account config overrides multi_tf.yaml
            acct_tf = cfg.SYMBOLS.get(sym, {}).get("atr_timeframe")
            tf = acct_tf if acct_tf else sym_cfg.get("timeframe", "H1")
            sym_cfg["timeframe"] = tf
            if tf not in TF_MAP:
                self.bot.logger.log('WARNING', 'MultiTF', 'BAD_TIMEFRAME',
                                    f'{sym}: unknown timeframe {tf}, skipping')
                continue
            self._tf_groups.setdefault(tf, []).append(sym)

        # Load ALL models in parallel — not one by one
        _ensure_ml_imports()
        load_tasks = []
        for sym, sym_cfg in self.symbols.items():
            tf = sym_cfg.get("timeframe", "H1")
            if tf == "H1":
                model_dir = None
            else:
                tf_model_dir = os.path.join(cfg.MODEL_DIR, tf)
                model_dir = tf_model_dir if os.path.isdir(tf_model_dir) else None
            load_tasks.append((sym, tf, model_dir))

        loaded = 0
        t_load = time.time()
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(load_tasks) or 1)) as pool:
            def _load_one(sym, tf, mdir):
                filt = SovereignMLFilter(sym, self.bot.logger, model_dir=mdir)
                ok = filt.load_model()
                return sym, tf, filt, ok

            futures = {pool.submit(_load_one, s, t, m): s for s, t, m in load_tasks}
            for f in as_completed(futures):
                sym, tf, filt, ok = f.result()
                if ok:
                    self.filters[sym] = filt
                    loaded += 1
                else:
                    self.bot.logger.log('WARNING', 'MultiTF', 'NO_MODEL',
                                        f'{sym}[{tf}]: no model found, skipping')

        load_ms = (time.time() - t_load) * 1000

        if loaded > 0:
            tf_summary = ", ".join(
                f"{tf}:{len(syms)}" for tf, syms in sorted(self._tf_groups.items(),
                                                             key=lambda x: TF_MAP[x[0]][0]))
            self.bot.logger.log('INFO', 'MultiTF', 'LOADED',
                                f'{loaded} models in {load_ms:.0f}ms (parallel), '
                                f'{len(self._tf_groups)} timeframes: {tf_summary}')

        return loaded

    def get_multi_tf_symbols(self) -> set[str]:
        """Return ALL managed symbols (for backwards compat)."""
        return set(self.symbols.keys())

    # ── Main tick — two-phase pipeline (returns signals) ─────────

    def tick(self, mt5) -> list[_Signal]:
        """Called every ~5s from main loop.

        Handles preloading + scanning for ALL timeframes simultaneously.
        When multiple TFs fire at the same moment (e.g. :00 → M5+M15+M30+H1),
        ALL are fetched and computed in one pass — no TF gets priority.

        Returns list of _Signal objects. Caller (MasterOrderGenerator)
        handles execution.
        """
        if not self._tf_groups:
            return []

        now = datetime.now()

        # Handle preloads for ALL TFs simultaneously
        preload_tfs = []
        for tf, symbols in self._tf_groups.items():
            minutes, _ = TF_MAP[tf]
            if self._should_preload(tf, minutes, now):
                preload_tfs.append((tf, symbols))
        for tf, symbols in preload_tfs:
            self._preload_tf(tf, symbols, mt5)

        # Collect ALL TFs that need scanning NOW — no priority, all at once
        ready_tfs = []
        for tf, symbols in self._tf_groups.items():
            minutes, _ = TF_MAP[tf]
            if self._should_scan(tf, minutes, now):
                ready_tfs.append((tf, symbols))

        if not ready_tfs:
            return []

        # Run the two-phase parallel pipeline (fetch + compute, no execution)
        return self._parallel_scan(ready_tfs, mt5)

    def force_scan(self, mt5) -> list[_Signal]:
        """Force-scan all timeframes immediately."""
        if not self._tf_groups:
            return []
        all_tfs = list(self._tf_groups.items())
        self.bot.logger.log('INFO', 'MultiTF', 'FORCE_SCAN',
                            f'Force-scanning {sum(len(s) for _, s in all_tfs)} symbols '
                            f'across {len(all_tfs)} TFs')
        return self._parallel_scan(all_tfs, mt5, force=True)

    # ── Two-phase parallel scan (returns signals, no execution) ──

    def _parallel_scan(self, ready_tfs: list[tuple[str, list[str]]],
                       mt5, force: bool = False) -> list[_Signal]:
        """
        Phase 1: Batch MT5 data fetch (main thread, serial — MT5 not thread-safe)
        Phase 2: Feature computation + ML inference (parallel, max workers)

        Returns list of _Signal objects sorted by TF speed then confidence.
        No trade execution — caller handles that via MasterOrderGenerator.
        """
        _ensure_ml_imports()
        from engine.inference import FEATURE_COLUMNS

        t0 = time.time()
        sentiment_cache = getattr(self.bot.order_router, '_cached_sentiment', {})

        # ── Phase 1: Batch fetch ALL symbols across ALL ready TFs ──
        fetch_tasks = []
        for tf, symbols in ready_tfs:
            minutes, mt5_tf_attr = TF_MAP[tf]
            mt5_tf = getattr(mt5, mt5_tf_attr, None)
            if mt5_tf is None:
                continue

            for sym in symbols:
                if self.bot.emergency_stop:
                    break
                filt = self.filters.get(sym)
                if filt is None or filt.model is None:
                    continue
                if self.bot.decay_tracker.is_disabled(sym):
                    continue
                is_open, _ = self.bot.trading_schedule.is_trading_open(sym)
                if not is_open:
                    continue
                if minutes > 60 and not force:
                    if not self._has_new_bar(sym, mt5, mt5_tf):
                        continue
                fetch_tasks.append((sym, tf, mt5_tf, minutes))

        if not fetch_tasks:
            for tf, _ in ready_tfs:
                self._last_scan[tf] = time.time()
            return []

        # Log scan start — show ALL TFs firing simultaneously
        tf_counts = {}
        for _, tf, _, _ in fetch_tasks:
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
        tf_str = " + ".join(f"{tf}:{n}" for tf, n in sorted(tf_counts.items(),
                            key=lambda x: TF_MAP[x[0]][0]))
        self.bot.logger.log('INFO', 'MultiTF', 'SCAN_START',
                            f'{len(fetch_tasks)} symbols [{tf_str}]')

        # Batch fetch: get rates from MT5 for all symbols (serial, MT5 not thread-safe)
        prefetched = {}
        cache = getattr(self.bot, 'scan_cache', None)

        for sym, tf, mt5_tf, minutes in fetch_tasks:
            bars_needed = min(500, max(250, int(200 * 60 / minutes)))
            broker_sym = cfg.SYMBOLS.get(sym, {}).get("broker_symbol", sym)
            cache_key = (sym, tf)

            try:
                if cache and cache_key in cache.bars:
                    cached_rates = cache.bars[cache_key]
                    fresh = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, 2)
                    if fresh and len(fresh) >= 1:
                        cached_times = {r['time'] for r in cached_rates[-5:]}
                        merged = list(cached_rates)
                        for bar in fresh:
                            if bar['time'] not in cached_times:
                                merged.append(bar)
                            else:
                                for i in range(len(merged) - 1, -1, -1):
                                    if merged[i]['time'] == bar['time']:
                                        merged[i] = bar
                                        break
                        rates = merged[-bars_needed:]
                        cache.bars[cache_key] = rates
                    else:
                        rates = cached_rates
                else:
                    rates = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, bars_needed)

                if rates is not None and len(rates) >= 100:
                    prefetched[(sym, tf)] = rates
            except Exception as e:
                self.bot.logger.log('ERROR', 'MultiTF', 'FETCH_ERROR',
                                    f'{sym}[{tf}]: {e}')

        fetch_ms = (time.time() - t0) * 1000

        if not prefetched:
            for tf, _ in ready_tfs:
                self._last_scan[tf] = time.time()
            return []

        # Grab cached tick + lead-lag data (read-only, thread-safe)
        cached_ticks = {}
        cached_lead_lag = {}
        if cache and cache.is_warm():
            cached_ticks = cache.tick_data
            cached_lead_lag = cache.lead_lag

        # ── Phase 2: Parallel feature + ML inference (ALL at once) ─
        t1 = time.time()
        signals: list[_Signal] = []
        all_scan_results: dict[str, list] = {}

        def _compute_one(sym: str, tf: str, rates, filt: SovereignMLFilter):
            """Pure computation — NO MT5 calls. Thread-safe."""
            return self._compute_signal(
                sym, tf, rates, filt, sentiment_cache,
                cached_ticks, cached_lead_lag)

        # Use ALL available workers — no artificial cap during peak moments
        n_workers = min(_MAX_WORKERS, len(prefetched))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for (sym, tf), rates in prefetched.items():
                filt = self.filters.get(sym)
                if filt is None:
                    continue
                f = pool.submit(_compute_one, sym, tf, rates, filt)
                futures[f] = (sym, tf)

            for f in as_completed(futures):
                sym, tf = futures[f]
                try:
                    result = f.result(timeout=15)
                    if result is None:
                        continue

                    all_scan_results.setdefault(tf, []).append(result.scan_info)

                    if result.should_trade:
                        signals.append(result)
                    else:
                        sent_info = f" sent={result.sent_boost:+.3f}" if result.sent_boost != 0 else ""
                        sym_thr = cfg.SYMBOLS.get(sym, {}).get("prob_threshold", cfg.ML_THRESHOLD)
                        thr_info = f" thr={sym_thr:.2f}" if sym_thr != cfg.ML_THRESHOLD else ""
                        self.bot.logger.log('DEBUG', 'MultiTF', 'NO_SIGNAL',
                                            f'{sym}[{tf}]: proba={result.confidence:.3f}{sent_info}{thr_info}')
                except Exception as e:
                    self.bot.logger.log('ERROR', 'MultiTF', 'COMPUTE_ERROR',
                                        f'{sym}[{tf}]: {e}')

        compute_ms = (time.time() - t1) * 1000

        # Sort: shortest TF first (most time-sensitive), then highest confidence
        signals.sort(key=lambda s: (TF_MAP.get(s.tf, (999,))[0], -s.confidence))

        # Log signals found (no execution here — caller handles that)
        for sig in signals:
            sent_info = f" sent={sig.sent_boost:+.3f}" if sig.sent_boost != 0 else ""
            sym_thr = cfg.SYMBOLS.get(sig.symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)
            thr_info = f" thr={sym_thr:.2f}" if sym_thr != cfg.ML_THRESHOLD else ""
            self.bot.logger.log('INFO', 'MultiTF', 'SIGNAL',
                                f'{sig.symbol}[{sig.tf}] {sig.direction} '
                                f'(proba={sig.confidence:.3f}{sent_info}{thr_info})')

        total_ms = (time.time() - t0) * 1000

        # Mark all ready TFs as scanned
        for tf, _ in ready_tfs:
            self._last_scan[tf] = time.time()

        self.bot.logger.log('INFO', 'MultiTF', 'SCAN_DONE',
                            f'{len(prefetched)} symbols, signals={len(signals)} | '
                            f'fetch={fetch_ms:.0f}ms compute={compute_ms:.0f}ms '
                            f'total={total_ms:.0f}ms (workers={n_workers})')

        # LLM commentary per TF
        for tf, results in all_scan_results.items():
            llm_callback = getattr(self.bot, '_llm_mtf_commentary', None)
            if llm_callback and results:
                found_tf = sum(1 for r in results if r["status"] in ("SIGNAL", "TRADE"))
                exec_tf = sum(1 for r in results if r["status"] == "TRADE")
                llm_callback(tf, results, found_tf, exec_tf)

        return signals

    # ── Pure computation (thread-safe, no MT5) ────────────────────

    def _compute_signal(self, symbol: str, tf: str, rates,
                        filt: SovereignMLFilter, sentiment_cache: dict,
                        cached_ticks: dict, cached_lead_lag: dict
                        ) -> _Signal | None:
        """Build features + run ML inference. No MT5 calls — thread-safe."""
        _ensure_ml_imports()
        from engine.inference import pl, build_bar_features, FEATURE_COLUMNS
        import numpy as np

        # Build polars DataFrame from pre-fetched rates
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

        try:
            feat = build_bar_features(bars, z_threshold=0.0)
        except Exception:
            return None

        if feat.height < 2:
            return None

        # Lead-lag features (from cache only — no MT5 fallback in threads)
        try:
            ll = cached_lead_lag.get(symbol)
            if ll:
                feat = feat.with_columns([
                    pl.lit(ll.get("leader_ret1", 0.0)).alias("leader_ret1"),
                    pl.lit(ll.get("leader_ret3", 0.0)).alias("leader_ret3"),
                    pl.lit(ll.get("leader_momentum", 0.0)).alias("leader_momentum"),
                ])
        except Exception:
            pass

        # Tick features (from cache only)
        try:
            from engine.tick_features import build_tick_features
            tick_data = cached_ticks.get(symbol)
            if tick_data is not None and tick_data.height > 0:
                feat = build_tick_features(tick_data, feat)
        except Exception:
            pass

        last_row = feat.tail(1)
        features_np = last_row.select(FEATURE_COLUMNS).to_numpy()

        if not np.all(np.isfinite(features_np)):
            return None

        primary_side = int(last_row["primary_side"][0])

        # ML inference (thread-safe: predict_proba is read-only)
        should_trade, confidence, direction = filt.should_trade(features_np, primary_side)

        # Sentiment adjustment (dict reads, thread-safe)
        confidence, sent_boost = _sentiment_adjust(
            confidence, direction, sentiment_cache, symbol)

        sym_threshold = cfg.SYMBOLS.get(symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)
        should_trade = confidence >= sym_threshold and direction is not None

        # Build features dict
        features_dict = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            features_dict[col] = float(features_np[0][i])
        features_dict["_model_version"] = getattr(filt, '_model_version', 'default')
        features_dict["_sentiment_boost"] = sent_boost
        features_dict["_timeframe"] = tf

        scan_info = {
            "symbol": symbol,
            "side": primary_side,
            "proba": confidence,
            "status": "SIGNAL" if should_trade else "skip",
            "reason": (direction or "no dir") if should_trade else "onder threshold",
            "z20": features_dict.get("z20", 0),
            "rsi14": features_dict.get("rsi14", 0),
            "vol20": features_dict.get("vol20", 0),
        }

        return _Signal(
            symbol=symbol, tf=tf, should_trade=should_trade,
            confidence=confidence, direction=direction,
            features_dict=features_dict, primary_side=primary_side,
            sent_boost=sent_boost, scan_info=scan_info,
        )

    # ── Preloading ────────────────────────────────────────────────

    def _preload_tf(self, tf: str, symbols: list[str], mt5):
        """Preload tick + bar data for a TF ~30s before bar close."""
        eligible = [
            s for s in symbols
            if not self.bot.decay_tracker.is_disabled(s)
            and self.bot.trading_schedule.is_trading_open(s)[0]
            and s in self.filters and self.filters[s].model is not None
        ]
        if not eligible:
            return
        cache = getattr(self.bot, 'scan_cache', None)
        if not cache:
            return
        missing = [s for s in eligible if s not in cache.tick_data]
        bars_missing = [s for s in eligible if (s, tf) not in cache.bars]
        if missing or bars_missing or not cache.is_warm():
            tf_map = {s: tf for s in eligible}
            self.bot.logger.log('INFO', 'ScanCache', 'MTF_PRELOAD_START',
                                f'{tf}: {len(eligible)} symbols '
                                f'({len(missing)} tick, {len(bars_missing)} bars new)')
            cache.preload(eligible, mt5, self.bot.logger, tf_map=tf_map)

    # ── Timing helpers ────────────────────────────────────────────

    def _should_preload(self, tf: str, minutes: int, now: datetime) -> bool:
        """True ~30-45s before the next bar close."""
        if minutes <= 60:
            secs_into_bar = (now.minute % minutes) * 60 + now.second
        else:
            total_minutes = now.hour * 60 + now.minute
            secs_into_bar = (total_minutes % minutes) * 60 + now.second

        bar_duration = minutes * 60
        secs_until_close = bar_duration - secs_into_bar

        if minutes <= 5:
            preload_window = (25, 45)
        elif minutes <= 15:
            preload_window = (20, 40)
        else:
            preload_window = (15, 35)

        if not (preload_window[0] <= secs_until_close <= preload_window[1]):
            return False

        last = self._last_scan.get(f"_preload_{tf}", 0)
        if time.time() - last < bar_duration * 0.9:
            return False

        self._last_scan[f"_preload_{tf}"] = time.time()
        return True

    def _should_scan(self, tf: str, minutes: int, now: datetime) -> bool:
        """Check if we should scan this timeframe now."""
        if now.second > 30:
            return False

        if minutes <= 60:
            if now.minute % minutes != 0:
                return False
            last = self._last_scan.get(tf, 0)
            if time.time() - last < minutes * 60 * 0.9:
                return False
            return True
        else:
            if now.minute != 0:
                return False
            last = self._last_scan.get(tf, 0)
            if time.time() - last < 50 * 60:
                return False
            return True

    def _has_new_bar(self, symbol: str, mt5, mt5_tf) -> bool:
        """Check if this specific symbol has a new bar since we last scanned it."""
        try:
            broker_sym = cfg.SYMBOLS.get(symbol, {}).get("broker_symbol", symbol)
            rates = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, 2)
            if not rates or len(rates) < 2:
                return True
            bar_time = int(rates[-2]['time'])
            last_bar = self._last_bar_time.get(symbol, 0)
            if bar_time > last_bar:
                self._last_bar_time[symbol] = bar_time
                return True
            return False
        except Exception:
            return True

    def seconds_until_next_bar(self) -> float:
        """Return seconds until the next bar close across all active TFs.

        Used by the main loop to sleep efficiently.
        """
        if not self._tf_groups:
            return 60.0

        now = datetime.now()
        min_wait = 9999.0

        for tf in self._tf_groups:
            minutes, _ = TF_MAP[tf]
            if minutes <= 60:
                secs_into_bar = (now.minute % minutes) * 60 + now.second
            else:
                total_minutes = now.hour * 60 + now.minute
                secs_into_bar = (total_minutes % minutes) * 60 + now.second

            bar_duration = minutes * 60
            secs_until = bar_duration - secs_into_bar + 3  # 3s grace
            min_wait = min(min_wait, secs_until)

        return max(min_wait, 1.0)
