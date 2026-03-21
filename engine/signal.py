"""
Signal generation — H1 bars from MT5 → features → batch inference.

Extracted from SovereignBot.get_h1_features() + check_signals().
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

from config.loader import cfg
from engine.inference import _ensure_ml_imports
CROSS_SYNC_DIR = REPO_ROOT / "config" / "cross_sync"
CROSS_SYNC_MAX_AGE_SEC = 3900  # 65 min — signals expire after one H1 window


# ---------------------------------------------------------------------------
# Cross-account signal sync (file-based IPC)
# ---------------------------------------------------------------------------

def _write_cross_sync_signal(account_id: str, symbol: str, sig: dict) -> None:
    """Write a signal file so the other account's process can pick it up."""
    CROSS_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = CROSS_SYNC_DIR / f"{account_id}_{symbol}_{ts}.json"
    payload = {
        "source_account": account_id,
        "symbol": symbol,
        "direction": sig["direction"],
        "confidence": sig["confidence"],
        "features_dict": sig.get("features_dict", {}),
        "primary_side": sig.get("primary_side", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload))


def _read_cross_sync_signals(my_account_id: str) -> list[dict]:
    """Read signal files from OTHER accounts, return list of signals."""
    if not CROSS_SYNC_DIR.exists():
        return []
    now = time.time()
    signals = []
    for path in CROSS_SYNC_DIR.glob("*.json"):
        try:
            age = now - path.stat().st_mtime
            if age > CROSS_SYNC_MAX_AGE_SEC:
                path.unlink(missing_ok=True)  # expired
                continue
            data = json.loads(path.read_text())
            if data.get("source_account") == my_account_id:
                continue  # our own signal, skip
            data["_path"] = str(path)
            signals.append(data)
        except Exception:
            continue
    return signals


def _cleanup_cross_sync_signal(path_str: str) -> None:
    """Remove a processed signal file."""
    try:
        Path(path_str).unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ScanCache — preload heavy I/O data before H1 bar close
# ---------------------------------------------------------------------------

class ScanCache:
    """Pre-loads tick parquets, lead-lag data, and bars so the :00 scan is fast."""

    def __init__(self):
        self.tick_data: dict = {}      # symbol → DataFrame | None
        self.lead_lag: dict = {}       # symbol → {leader_ret1, leader_ret3, leader_momentum}
        self.bars: dict = {}           # (symbol, tf) → list[dict] (raw rates from MT5)
        self.timestamp: float = 0      # epoch when cache was filled
        self._loading: bool = False

    def is_warm(self, max_age_seconds: int = 180) -> bool:
        """True if cache was filled within *max_age_seconds*."""
        return (time.time() - self.timestamp) < max_age_seconds

    def preload(self, symbols: list[str], mt5, logger,
                tf_map: dict[str, str] | None = None) -> None:
        """Load tick data + lead-lag + bars for all *symbols*.

        tf_map: optional {symbol: timeframe_name} to preload bars on correct TF.
        """
        self._loading = True
        t0 = time.time()

        # 1) Tick data from disk (the most expensive operation)
        for sym in symbols:
            if sym in self.tick_data:
                continue
            try:
                self.tick_data[sym] = _load_recent_ticks(sym, logger)
            except Exception:
                self.tick_data[sym] = None

        # 2) Lead-lag features (many MT5 bridge calls)
        from engine.lead_lag import build_lead_lag_features
        for sym in symbols:
            if sym in self.lead_lag:
                continue
            try:
                self.lead_lag[sym] = build_lead_lag_features(sym, mt5, logger) or {}
            except Exception:
                self.lead_lag[sym] = {}

        # 3) Pre-fetch bars from MT5 so scan only needs 1 fresh bar
        if tf_map:
            TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
            TF_ATTRS = {"M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
                        "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4"}
            for sym in symbols:
                tf_name = tf_map.get(sym)
                if not tf_name or tf_name not in TF_ATTRS:
                    continue
                key = (sym, tf_name)
                if key in self.bars:
                    continue
                mt5_tf = getattr(mt5, TF_ATTRS[tf_name], None)
                if mt5_tf is None:
                    continue
                minutes = TF_MINUTES[tf_name]
                bars_needed = max(200, int(200 * 60 / minutes))
                broker_sym = cfg.SYMBOLS.get(sym, {}).get("broker_symbol", sym)
                try:
                    rates = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, bars_needed)
                    if rates and len(rates) >= 100:
                        self.bars[key] = rates
                except Exception:
                    pass

        self.timestamp = time.time()
        self._loading = False
        ms = (time.time() - t0) * 1000
        bars_count = sum(1 for k in self.bars if k[0] in symbols)
        logger.log('INFO', 'ScanCache', 'PRELOADED',
                    f'{len(symbols)} symbols in {ms:.0f}ms (bars: {bars_count})')

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


def get_h1_features(symbol: str, mt5, logger, cache: ScanCache | None = None,
                    broker_symbol: str | None = None):
    """Fetch H1 bars from MT5 and build features.

    When *cache* is warm, tick data and lead-lag features are read from it
    instead of hitting disk / MT5 again (preloaded at :58).

    Args:
        broker_symbol: Override broker symbol resolution (for paper-only symbols
                       not in cfg.SYMBOLS).

    Returns (features_np, primary_side) or (None, None).
    """
    if mt5 is None:
        return None, None

    _ensure_ml_imports()
    from engine.inference import pl, build_bar_features, FEATURE_COLUMNS  # noqa: F811

    # Resolve broker symbol (e.g. BTC_USD → BTC/USD)
    broker_sym = broker_symbol or cfg.SYMBOLS.get(symbol, {}).get("broker_symbol", symbol)

    # Need 200 bars for rolling windows (vol20, ret48, etc.)
    rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_H1, 0, 200)
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


_TF_ATTR = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
}


def get_tf_features(symbol: str, mt5, logger, timeframe: str = "H1",
                    broker_symbol: str | None = None):
    """Fetch bars for any timeframe from MT5 and build features.

    Same as get_h1_features() but for arbitrary timeframes.
    Returns (features_np, primary_side) or (None, None).
    """
    if mt5 is None:
        return None, None

    _ensure_ml_imports()
    from engine.inference import pl, build_bar_features, FEATURE_COLUMNS

    broker_sym = broker_symbol or cfg.SYMBOLS.get(symbol, {}).get("broker_symbol", symbol)
    mt5_tf = getattr(mt5, _TF_ATTR.get(timeframe, "TIMEFRAME_H1"))

    rates = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, 200)
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

    try:
        feat = build_bar_features(bars, z_threshold=0.0)
    except Exception:
        return None, None

    if feat.height < 2:
        return None, None

    last_row = feat.tail(1)
    features_np = last_row.select(FEATURE_COLUMNS).to_numpy()

    if not np.all(np.isfinite(features_np)):
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

    # Phase 2+3: Per-account inference + batch execution
    # Each account has its own ML models (trained with account-specific costs).
    # Features are shared, but inference and execution are per-account.
    t0 = time.time()
    scan_results = []

    # Get active accounts for per-account inference
    active_accounts = [a for a in engine.accounts.values()
                       if a.enabled and a.order_router is not None and a.filters
                       and not getattr(a, 'trading_paused', False)]

    # Fallback: if no per-account filters, use shared bot.filters (backwards compat)
    if not active_accounts:
        active_accounts = None  # Will use shared filters below

    # Helper: run inference for one (symbol, filter, features) combo
    def _run_inference(symbol, filt, features_np, primary_side):
        sym_threshold = cfg.SYMBOLS.get(symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)
        used_version = getattr(filt, '_model_version', 'default')
        challenger_version, challenger_model = None, None
        if cfg.AB_TEST_ENABLED:
            challenger_version, challenger_model = filt.load_challenger()

        if challenger_model is not None:
            from engine.inference import xgb
            dmat = xgb.DMatrix(features_np)
            raw = challenger_model.predict(dmat)
            confidence = float(raw[0])
            used_version = challenger_version
            if features_np[0][FEATURE_COLUMNS.index("z20")] > 0:
                direction = 'SELL'
            elif features_np[0][FEATURE_COLUMNS.index("z20")] < 0:
                direction = 'BUY'
            else:
                direction = None
            confidence, sent_boost = _sentiment_adjust(
                confidence, direction, sentiment_cache, symbol)
            should_trade = confidence >= sym_threshold and direction is not None
        else:
            should_trade, confidence, direction = filt.should_trade(
                features_np, primary_side, threshold=sym_threshold)
            confidence, sent_boost = _sentiment_adjust(
                confidence, direction, sentiment_cache, symbol)
            should_trade = confidence >= sym_threshold and direction is not None

        features_dict = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            features_dict[col] = float(features_np[0][i])
        features_dict["_model_version"] = used_version
        features_dict["_sentiment_boost"] = sent_boost
        return should_trade, confidence, direction, features_dict, sent_boost

    if active_accounts:
        # --- Per-account inference + execution ---
        for acct in active_accounts:
            acct_signals = []
            for symbol, _shared_filt, features_np, primary_side in candidates:
                if engine.emergency_stop:
                    break
                filt = acct.filters.get(symbol)
                if filt is None or filt.model is None:
                    continue

                should_trade, confidence, direction, features_dict, sent_boost = \
                    _run_inference(symbol, filt, features_np, primary_side)

                # Paper trackers: feed EVERY signal (even blocked ones)
                for pt in getattr(engine, 'paper_trackers', []):
                    try:
                        pt.on_signal(symbol, direction, confidence, features_dict, mt5)
                    except Exception as _pt_err:
                        engine.logger.log('ERROR', 'PaperTracker', 'SIGNAL_ERROR',
                                          f'{symbol}: {_pt_err}')

                if not should_trade or direction is None:
                    sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
                    engine.logger.log('DEBUG', 'Signal', 'NO_SIGNAL',
                                      f'[{acct.name}] {symbol}: proba={confidence:.3f}{sent_info}')
                    continue

                signals_found += 1
                sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
                engine.logger.log('INFO', 'Signal', 'SIGNAL',
                                  f'[{acct.name}] {symbol} {direction} (proba={confidence:.3f}{sent_info})')
                feature_logger.log_trade_features(
                    symbol, direction, confidence, features_dict, status="SIGNAL")
                acct_signals.append({
                    'symbol': symbol, 'direction': direction,
                    'confidence': confidence, 'features_dict': features_dict,
                    'primary_side': primary_side,
                })

            # Batch execute for this account
            if acct_signals:
                acct_signals.sort(key=lambda s: s['confidence'], reverse=True)

                # Query margin once, allocate budgets
                account_info = acct.mt5.account_info() if acct.mt5 else None
                if account_info:
                    n = len(acct_signals)
                    original_free = account_info.margin_free
                    total_budget = original_free * 0.90
                    total_conf = sum(s['confidence'] for s in acct_signals)
                    engine.logger.log('INFO', 'Signal', 'BATCH_MARGIN',
                                      f'[{acct.name}] {n} signals, budget ${total_budget:,.0f} '
                                      f'(reserve ${original_free * 0.10:,.0f})')

                    for sig in acct_signals:
                        weight = sig['confidence'] / total_conf if total_conf > 0 else 1.0 / n
                        budget = total_budget * weight
                        budget = max(total_budget * 0.10, min(total_budget * 0.50, budget))

                        success = acct.execute_trade(
                            sig['symbol'], sig['direction'], sig['confidence'],
                            features_dict=sig.get('features_dict'),
                            margin_budget=budget,
                        )
                        if success:
                            signals_executed += 1
                            feature_logger.log_trade_features(
                                sig['symbol'], sig['direction'], sig['confidence'],
                                sig['features_dict'], status="EXECUTED")
                            # Write cross-sync signal file for other accounts
                            _write_cross_sync_signal(acct.account_id, sig['symbol'], sig)
                        scan_results.append({
                            "symbol": sig['symbol'], "proba": sig['confidence'],
                            "side": sig['primary_side'],
                            "direction": sig['direction'],
                            "status": "EXECUTED" if success else "BLOCKED",
                            "reason": f"[{acct.name}] " + ("trade geplaatst" if success else "guardrail blokkade"),
                            "z20": sig['features_dict'].get("z20", 0),
                            "rsi14": sig['features_dict'].get("rsi14", 0),
                            "vol20": sig['features_dict'].get("vol20", 0),
                        })
        # --- Cross-account sync (file-based IPC) ---
        # Each bot runs as a separate process with one account.
        # Check for signal files written by other accounts and execute them.
        for acct in active_accounts:
            pending = _read_cross_sync_signals(acct.account_id)
            if not pending:
                continue
            for sig_data in pending:
                symbol = sig_data.get("symbol", "")
                direction = sig_data.get("direction", "")
                confidence = sig_data.get("confidence", 0)
                source = sig_data.get("source_account", "?")

                # Only sync symbols in this account's portfolio
                if symbol not in acct._internal_to_broker:
                    _cleanup_cross_sync_signal(sig_data["_path"])
                    continue

                # Skip if we already have a position in this symbol
                existing = acct.mt5.positions_get() if acct.mt5 else []
                broker_sym = acct._internal_to_broker[symbol]
                if existing and any(p.symbol == broker_sym for p in existing):
                    engine.logger.log('DEBUG', 'Signal', 'CROSS_SYNC_SKIP',
                        f'[{acct.name}] {symbol} already in market, skip sync from {source}')
                    _cleanup_cross_sync_signal(sig_data["_path"])
                    continue

                engine.logger.log('INFO', 'Signal', 'CROSS_SYNC',
                    f'[{acct.name}] Syncing {symbol} {direction} '
                    f'from {source} (conf={confidence:.3f})')

                acct_info = acct.mt5.account_info() if acct.mt5 else None
                budget = acct_info.margin_free * 0.45 if acct_info else None

                success = acct.execute_trade(
                    symbol, direction, confidence,
                    features_dict=sig_data.get("features_dict"),
                    margin_budget=budget,
                )
                if success:
                    signals_executed += 1
                    engine.logger.log('INFO', 'Signal', 'CROSS_SYNC_OK',
                        f'[{acct.name}] {symbol} {direction} synced from {source}')
                    scan_results.append({
                        "symbol": symbol, "proba": confidence,
                        "side": sig_data.get("primary_side", 0),
                        "direction": direction,
                        "status": "EXECUTED",
                        "reason": f"[{acct.name}] cross-sync from {source}",
                    })
                else:
                    engine.logger.log('INFO', 'Signal', 'CROSS_SYNC_BLOCKED',
                        f'[{acct.name}] {symbol} sync blocked by guardrails')
                _cleanup_cross_sync_signal(sig_data["_path"])

    else:
        # --- Fallback: shared filters (backwards compat / single-account) ---
        actionable_signals = []
        for symbol, filt, features_np, primary_side in candidates:
            if engine.emergency_stop:
                break
            sym_threshold = cfg.SYMBOLS.get(symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)
            should_trade, confidence, direction, features_dict, sent_boost = \
                _run_inference(symbol, filt, features_np, primary_side)

            # Paper trackers: feed EVERY signal (even blocked ones)
            for pt in getattr(engine, 'paper_trackers', []):
                try:
                    pt.on_signal(symbol, direction, confidence, features_dict, mt5)
                except Exception as _pt_err:
                    engine.logger.log('ERROR', 'PaperTracker', 'SIGNAL_ERROR',
                                      f'{symbol}: {_pt_err}')

            if not should_trade or direction is None:
                sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
                engine.logger.log('DEBUG', 'Signal', 'NO_SIGNAL',
                                  f'{symbol}: proba={confidence:.3f}{sent_info} side={primary_side}')
                feature_logger.log_trade_features(
                    symbol, direction or "NONE", confidence, features_dict, status="FILTERED")
                if primary_side == 0:
                    reason = "z20 exact nul (zeer zeldzaam)"
                elif confidence < sym_threshold:
                    reason = f"proba {confidence:.3f} < threshold {sym_threshold}"
                else:
                    reason = "geen richting"
                scan_results.append({
                    "symbol": symbol, "proba": confidence, "side": primary_side,
                    "direction": direction or "NONE", "status": "SKIP", "reason": reason,
                    "z20": features_dict.get("z20", 0), "rsi14": features_dict.get("rsi14", 0),
                    "vol20": features_dict.get("vol20", 0),
                })
                continue

            signals_found += 1
            sent_info = f" sent={sent_boost:+.3f}" if sent_boost != 0 else ""
            engine.logger.log('INFO', 'Signal', 'SIGNAL',
                              f'{symbol} {direction} (proba={confidence:.3f}{sent_info})')
            feature_logger.log_trade_features(
                symbol, direction, confidence, features_dict, status="SIGNAL")
            actionable_signals.append({
                'symbol': symbol, 'direction': direction,
                'confidence': confidence, 'features_dict': features_dict,
                'primary_side': primary_side,
            })

        if actionable_signals:
            actionable_signals.sort(key=lambda s: s['confidence'], reverse=True)
            batch_results = engine.execute_trade_batch(actionable_signals)
            for sig in actionable_signals:
                result = next((r for r in batch_results
                               if r['symbol'] == sig['symbol']), None)
                success = result['success'] if result else False
                if success:
                    signals_executed += 1
                    feature_logger.log_trade_features(
                        sig['symbol'], sig['direction'], sig['confidence'],
                        sig['features_dict'], status="EXECUTED")
                scan_results.append({
                    "symbol": sig['symbol'], "proba": sig['confidence'],
                    "side": sig['primary_side'], "direction": sig['direction'],
                    "status": "EXECUTED" if success else "BLOCKED",
                    "reason": "trade geplaatst" if success else "guardrail blokkade",
                    "z20": sig['features_dict'].get("z20", 0),
                    "rsi14": sig['features_dict'].get("rsi14", 0),
                    "vol20": sig['features_dict'].get("vol20", 0),
                })

    # LLM scan commentary
    if llm_scan_callback and scan_results:
        llm_scan_callback(scan_results, signals_found, signals_executed)

    # Flush feature log
    feature_logger.flush()
    return signals_found, signals_executed, len(candidates)
