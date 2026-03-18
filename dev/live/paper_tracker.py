"""
Paper Trading Tracker — shadow bot that simulates trades alongside the live bot.

Runs ALL symbols with a trained model (not just live ones), so we can discover
which symbols perform best in forward-testing. Uses REAL MT5 prices, real
commission from CostModel, real spreads, and the same trailing/BE/horizon
logic as the live bot.

Usage:
    pt = PaperTracker("ftmo_100k", "ftmo/audit/paper_trades.db")
    pt.load_models(logger)            # Load ALL available XGBoost models
    pt.start_monitor(mt5, interval=1.0)
    pt.run_full_scan(mt5, logger)     # Own H1 scan over ALL model symbols
    pt.on_bar_close(mt5)
    stats = pt.daily_summary()
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.loader import cfg
from research.cost_model import get_cost_model

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p


@dataclass
class PaperPosition:
    symbol: str
    direction: str              # "BUY" / "SELL"
    entry_price: float
    entry_time: str
    lot_size: float
    sl_price: float
    tp_price: float
    atr: float
    confidence: float
    cost_pct: float             # round-trip cost fraction
    commission_usd: float       # actual commission from MT5 or CostModel
    spread_bps: float
    exit_params: dict           # atr_sl_mult, atr_tp_mult, breakeven_atr, etc.
    contract_size: float = 1.0
    tick_value: float = 0.0
    tick_size: float = 0.0
    bars_held: int = 0
    high_water: float = 0.0     # max favorable price excursion
    db_id: int | None = None    # paper_trades.id after INSERT
    timeframe: str = "H1"       # M1, M5, M15, M30, H1, H4


class PaperTracker:
    """Shadow trading engine — simulates positions using real MT5 prices."""

    # Asset-class defaults (same as PositionManager._TRAIL_DEFAULTS)
    _TRAIL_DEFAULTS = {
        'crypto':    {'breakeven_atr': 1.0, 'trail_activation_atr': 3.0, 'trail_distance_atr': 1.5},
        'forex':     {'breakeven_atr': 0.5, 'trail_activation_atr': 1.5, 'trail_distance_atr': 0.8},
        'commodity': {'breakeven_atr': 0.5, 'trail_activation_atr': 1.5, 'trail_distance_atr': 0.8},
        'index':     {'breakeven_atr': 0.5, 'trail_activation_atr': 1.5, 'trail_distance_atr': 0.8},
        'equity':    {'breakeven_atr': 0.2, 'trail_activation_atr': 0.8, 'trail_distance_atr': 0.4},
    }

    # Default configs per asset class — used for symbols NOT in live config
    _DEFAULT_CONFIGS = {
        'crypto': {
            'asset_class': 'crypto', 'risk_per_trade': 0.002,
            'prob_threshold': 0.55, 'atr_period': 14,
            'atr_sl_mult': 2.0, 'atr_tp_mult': 6.0, 'max_spread_pct': 0.005,
            'exit_horizon': 24,
        },
        'forex': {
            'asset_class': 'forex', 'risk_per_trade': 0.004,
            'prob_threshold': 0.55, 'atr_period': 14,
            'atr_sl_mult': 1.5, 'atr_tp_mult': 4.5, 'max_spread_pct': 0.0005,
            'exit_horizon': 24,
        },
        'equity': {
            'asset_class': 'equity', 'risk_per_trade': 0.005,
            'prob_threshold': 0.55, 'atr_period': 14,
            'atr_sl_mult': 1.2, 'atr_tp_mult': 3.6, 'max_spread_pct': 0.001,
            'exit_horizon': 24,
        },
        'index': {
            'asset_class': 'index', 'risk_per_trade': 0.003,
            'prob_threshold': 0.55, 'atr_period': 14,
            'atr_sl_mult': 1.5, 'atr_tp_mult': 4.5, 'max_spread_pct': 0.0005,
            'exit_horizon': 24,
        },
        'commodity': {
            'asset_class': 'commodity', 'risk_per_trade': 0.003,
            'prob_threshold': 0.55, 'atr_period': 14,
            'atr_sl_mult': 1.5, 'atr_tp_mult': 4.5, 'max_spread_pct': 0.001,
            'exit_horizon': 24,
        },
    }

    # TF-specific multiplier overrides — scale exit params for shorter timeframes
    # Shorter TFs → tighter SL, tighter TP, shorter horizon, faster trailing
    _TF_SCALE = {
        'M5':  {'sl_scale': 0.3, 'tp_scale': 0.35, 'exit_horizon': 6,
                'trail_activation_scale': 0.3, 'trail_distance_scale': 0.3, 'breakeven_scale': 0.25},
        'M15': {'sl_scale': 0.7, 'tp_scale': 0.6, 'exit_horizon': 8,
                'trail_activation_scale': 0.6, 'trail_distance_scale': 0.6, 'breakeven_scale': 0.6},
        'M30': {'sl_scale': 0.8, 'tp_scale': 0.7, 'exit_horizon': 12,
                'trail_activation_scale': 0.7, 'trail_distance_scale': 0.7, 'breakeven_scale': 0.7},
        'H1':  {'sl_scale': 1.0, 'tp_scale': 1.0, 'exit_horizon': 24,
                'trail_activation_scale': 1.0, 'trail_distance_scale': 1.0, 'breakeven_scale': 1.0},
        'H4':  {'sl_scale': 1.2, 'tp_scale': 1.3, 'exit_horizon': 12,
                'trail_activation_scale': 1.2, 'trail_distance_scale': 1.2, 'breakeven_scale': 1.2},
    }

    def __init__(self, account_id: str, db_path: str, logger=None,
                 discord=None, risk_scale: float = 1.0):
        self.account_id = account_id
        self.cost_model = get_cost_model(account_id)
        self.equity = 100_000.0
        self.starting_equity = 100_000.0
        self.positions: list[PaperPosition] = []
        self.risk_scale = risk_scale
        self.logger = logger
        self.discord = discord

        # Load per-symbol configs from account's config.json (live symbols)
        self.sym_configs: dict[str, dict] = {}
        self._load_sym_configs(account_id)

        # Load Optuna-optimized exit params per symbol×TF (if available)
        self._exit_optuna: dict[str, dict] = {}  # key: "SYMBOL_TF"
        self._load_exit_optuna(account_id)

        # Model directory + loaded ML filters for ALL symbols
        self._model_dir: str = ""
        self._filters: dict = {}  # symbol → SovereignMLFilter
        self._all_symbols: set[str] = set()  # ALL symbols with models
        self._live_scanned: set[str] = set()  # symbols already fed by live scan

        # Monitor thread
        self._mt5 = None
        self._monitor_thread: threading.Thread | None = None
        self._monitor_running = False
        self._lock = threading.Lock()

        # SQLite
        abs_path = str(REPO_ROOT / db_path) if not Path(db_path).is_absolute() else db_path
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = abs_path
        self._init_db()

        # Restore open positions from DB
        self._restore_open_positions()

        # Trading schedule (market hours check)
        self._trading_schedule = None
        self._session_closed: dict[str, float] = {}  # symbol_TF -> timestamp of last CLOSED_SESSION
        try:
            from risk.ftmo_guard import TradingSchedule
            self._trading_schedule = TradingSchedule()
        except Exception:
            pass

    def _load_sym_configs(self, account_id: str):
        """Load symbol configs from account's config.json."""
        import yaml
        accts_path = Path(__file__).resolve().parent.parent / "config" / "accounts.yaml"
        if not accts_path.exists():
            return
        with open(accts_path) as f:
            accts = yaml.safe_load(f).get("accounts", {})
        acct_cfg = accts.get(account_id, {})
        config_path = acct_cfg.get("config_path", "")
        if config_path:
            import json
            full = REPO_ROOT / config_path
            if full.exists():
                with open(full) as f:
                    self.sym_configs = json.load(f)

    def _load_exit_optuna(self, account_id: str):
        """Load Optuna-optimized exit params from exit_summary.csv."""
        import csv as _csv
        prefix = account_id.split("_")[0]  # ftmo or bright→bf
        if prefix == "bright":
            prefix = "bf"
        csv_path = REPO_ROOT / prefix / "optuna" / "exit_fast_tfs" / "exit_summary.csv"
        if not csv_path.exists():
            return
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                ev = float(row.get("best_ev", -999))
                key = f"{row['symbol']}_{row['timeframe']}"
                self._exit_optuna[key] = {
                    "atr_sl_mult": float(row["best_atr_sl_mult"]),
                    "atr_tp_mult": float(row["best_atr_tp_mult"]),
                    "breakeven_atr": float(row["best_breakeven_atr"]),
                    "trail_activation_atr": float(row["best_trail_activation_atr"]),
                    "trail_distance_atr": float(row["best_trail_distance_atr"]),
                    "exit_horizon": int(row["best_horizon"]),
                    "ev": ev,
                }
        if self._exit_optuna and self.logger:
            self.logger.log('INFO', 'PaperTracker', 'EXIT_OPTUNA_LOADED',
                            f'[{account_id}] Loaded {len(self._exit_optuna)} optimized exit params')

    def _connect(self):
        """Open DB connection with WAL mode and busy timeout."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                sl_price REAL,
                tp_price REAL,
                lot_size REAL,
                confidence REAL,
                cost_pct REAL,
                commission_usd REAL,
                spread_bps REAL,
                contract_size REAL,
                status TEXT,
                exit_price REAL,
                exit_timestamp TEXT,
                pnl REAL,
                bars_held INTEGER,
                account_id TEXT,
                atr REAL,
                exit_params TEXT
            );
            CREATE TABLE IF NOT EXISTS paper_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                daily_pnl REAL,
                open_positions INTEGER,
                account_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_paper_trades_status
                ON paper_trades(status);
            CREATE INDEX IF NOT EXISTS idx_paper_trades_account
                ON paper_trades(account_id);
        """)
        # Migrations — add columns if missing
        for migration in [
            "ALTER TABLE paper_trades ADD COLUMN high_water REAL DEFAULT 0",
            "ALTER TABLE paper_trades ADD COLUMN timeframe TEXT DEFAULT 'H1'",
            "ALTER TABLE paper_trades ADD COLUMN tick_value REAL DEFAULT 0",
            "ALTER TABLE paper_trades ADD COLUMN tick_size REAL DEFAULT 0",
        ]:
            try:
                conn.execute(migration)
            except Exception:
                pass  # Column already exists
        conn.close()

    def _restore_open_positions(self):
        """Restore OPEN paper positions from DB on restart."""
        import json as _json
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status = 'OPEN' AND account_id = ?",
            (self.account_id,)
        ).fetchall()
        conn.close()
        for row in rows:
            exit_params = _json.loads(row["exit_params"]) if row["exit_params"] else {}
            # Read tick_value/tick_size from DB first, fall back to MT5
            tv = row["tick_value"] if "tick_value" in row.keys() else 0.0
            ts = row["tick_size"] if "tick_size" in row.keys() else 0.0
            tv = tv or 0.0
            ts = ts or 0.0
            if (tv == 0.0 or ts == 0.0) and self._mt5:
                broker_sym = self.sym_configs.get(row["symbol"], {}).get(
                    "broker_symbol", row["symbol"])
                si = self._mt5.symbol_info(broker_sym)
                if si:
                    tv = getattr(si, 'trade_tick_value', 0.0) or 0.0
                    ts = getattr(si, 'trade_tick_size', 0.0) or 0.0
            pos = PaperPosition(
                symbol=row["symbol"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                entry_time=row["timestamp"],
                lot_size=row["lot_size"],
                sl_price=row["sl_price"],
                tp_price=row["tp_price"],
                atr=row["atr"] or 0.0,
                confidence=row["confidence"],
                cost_pct=row["cost_pct"],
                commission_usd=row["commission_usd"] or 0.0,
                spread_bps=row["spread_bps"],
                exit_params=exit_params,
                contract_size=row["contract_size"] or 1.0,
                tick_value=tv,
                tick_size=ts,
                bars_held=row["bars_held"] or 0,
                high_water=row["high_water"] or 0.0,
                db_id=row["id"],
                timeframe=row["timeframe"] or "H1",
            )
            self.positions.append(pos)
        if rows and self.logger:
            self.logger.log('INFO', 'PaperTracker', 'RESTORED',
                            f'[{self.account_id}] Restored {len(rows)} open paper positions')

    # ── Model loading + full scan ──────────────────────────────────

    # Timeframes to scan — H1 is always from the root model dir,
    # others from TF-specific subdirectories (e.g. models/versions/M15/)
    SCAN_TIMEFRAMES = ["M15", "M30", "H1", "H4"]

    def load_models(self, logger=None):
        """Discover and load ALL XGBoost models from the account's model_dir.

        Loads H1 models from root dir, plus models from TF subdirectories
        (M1/, M5/, M15/, M30/, H4/) if they exist.
        """
        import yaml
        from engine.inference import SovereignMLFilter

        log = logger or self.logger
        accts_path = Path(__file__).resolve().parent.parent / "config" / "accounts.yaml"
        if not accts_path.exists():
            return
        with open(accts_path) as f:
            accts = yaml.safe_load(f).get("accounts", {})
        acct_cfg = accts.get(self.account_id, {})
        self._model_dir = str(REPO_ROOT / acct_cfg.get("model_dir", ""))
        if not self._model_dir or not os.path.isdir(self._model_dir):
            if log:
                log.log('WARNING', 'PaperTracker', 'NO_MODEL_DIR',
                         f'[{self.account_id}] Model dir not found: {self._model_dir}')
            return

        # _filters: symbol → SovereignMLFilter (H1 only, backward compat)
        # _tf_filters: (symbol, timeframe) → SovereignMLFilter
        self._tf_filters: dict[tuple[str, str], 'SovereignMLFilter'] = {}

        total_loaded = 0
        for tf in self.SCAN_TIMEFRAMES:
            if tf == "H1":
                tf_dir = self._model_dir
            else:
                tf_dir = os.path.join(self._model_dir, tf)
                if not os.path.isdir(tf_dir):
                    continue

            model_files = [f for f in os.listdir(tf_dir)
                           if f.endswith('.json') and f != 'rl_sizer.json'
                           and not os.path.isdir(os.path.join(tf_dir, f))]

            all_syms = [mf.replace('.json', '') for mf in model_files]
            live_names = set(self.sym_configs.keys())
            skip = set()
            for s in all_syms:
                no_us = s.replace('_', '')
                if no_us != s and no_us in all_syms:
                    skip.add(no_us)
            all_syms = [s for s in all_syms if s not in skip]

            for symbol in all_syms:
                filt = SovereignMLFilter(symbol, log, model_dir=tf_dir)
                if filt.load_model():
                    self._tf_filters[(symbol, tf)] = filt
                    if tf == "H1":
                        self._filters[symbol] = filt
                    self._all_symbols.add(symbol)
                    total_loaded += 1

                    if symbol not in self.sym_configs:
                        self.sym_configs[symbol] = self._generate_default_config(symbol)

        if log:
            tf_counts = {}
            for (sym, tf) in self._tf_filters:
                tf_counts[tf] = tf_counts.get(tf, 0) + 1
            tf_summary = ", ".join(f"{tf}={n}" for tf, n in sorted(tf_counts.items()))
            log.log('INFO', 'PaperTracker', 'MODELS_LOADED',
                     f'[{self.account_id}] {total_loaded} models loaded ({tf_summary})')

    def _generate_default_config(self, symbol: str) -> dict:
        """Generate sensible default config for a symbol based on its name."""
        from risk.position_sizing import SECTOR_MAP, ASSET_CLASS

        # Try exact match, then without underscores (BTC_USD → BTCUSD)
        sector = SECTOR_MAP.get(symbol)
        if sector is None:
            sector = SECTOR_MAP.get(symbol.replace('_', ''), "unknown")
        asset_class = ASSET_CLASS.get(sector, "forex")
        defaults = dict(self._DEFAULT_CONFIGS.get(asset_class, self._DEFAULT_CONFIGS['forex']))
        defaults['sector'] = sector
        defaults['asset_class'] = asset_class

        # Broker symbol mapping: most symbols are 1:1, but underscored ones
        # might need the underscore stripped (BTC_USD → BTCUSD)
        if '_' in symbol:
            defaults['broker_symbol'] = symbol.replace('_', '')

        return defaults

    def run_full_scan(self, mt5, logger, cache=None, timeframe: str | None = None):
        """Run inference scan over ALL symbols with models for one or all timeframes.

        Args:
            timeframe: If set, scan only this TF. If None, scan all available TFs.
        """
        if not self._tf_filters and not self._filters:
            return 0

        from engine.signal import get_h1_features, get_tf_features
        from engine.inference import FEATURE_COLUMNS

        scan_tfs = [timeframe] if timeframe else self.SCAN_TIMEFRAMES
        scanned = 0
        signals = 0

        for tf in scan_tfs:
            # Collect models for this timeframe
            tf_models = {sym: filt for (sym, t), filt in self._tf_filters.items() if t == tf}
            if not tf_models:
                continue

            # Skip symbols already fed by live scan for this TF
            for symbol in sorted(tf_models):
                if f"{symbol}_{tf}" in self._live_scanned:
                    continue

                filt = tf_models[symbol]
                if filt.model is None:
                    continue

                try:
                    broker_sym = self.sym_configs.get(symbol, {}).get("broker_symbol", symbol)

                    if tf == "H1":
                        features_np, primary_side = get_h1_features(
                            symbol, mt5, logger, cache=cache, broker_symbol=broker_sym)
                    else:
                        features_np, primary_side = get_tf_features(
                            symbol, mt5, logger, timeframe=tf, broker_symbol=broker_sym)

                    if features_np is None:
                        continue

                    scanned += 1

                    sym_threshold = self.sym_configs.get(symbol, {}).get(
                        "prob_threshold", cfg.ML_THRESHOLD)
                    should_trade, confidence, direction = filt.should_trade(
                        features_np, primary_side, threshold=sym_threshold)

                    if direction is not None:
                        features_dict = {}
                        for i, col in enumerate(FEATURE_COLUMNS):
                            if i < features_np.shape[1]:
                                features_dict[col] = float(features_np[0][i])
                        opened = self.on_signal(
                            symbol, direction, confidence, features_dict, mt5, timeframe=tf)
                        if opened:
                            signals += 1

                except Exception as e:
                    if logger:
                        logger.log('DEBUG', 'PaperTracker', 'SCAN_ERROR',
                                    f'[{self.account_id}] {symbol}/{tf}: {e}')

        self._live_scanned.clear()

        if logger and scanned > 0:
            logger.log('INFO', 'PaperTracker', 'FULL_SCAN',
                        f'[{self.account_id}] Scanned {scanned} symbol-TF combos, '
                        f'{signals} new paper trades')
        return signals

    # ── Commission from MT5 ────────────────────────────────────────

    def _get_commission_usd(self, symbol: str, lot_size: float, mt5) -> float:
        """Try to get real commission from MT5 symbol_info, fall back to CostModel.

        MT5 exposes commission per lot in the deal history but not directly
        on symbol_info. Some brokers populate symbol_info().session_* fields.
        We try order_calc_profit for 1 lot to estimate, then fall back.
        """
        # Method 1: CostModel has real per-instrument commission from CSV specs
        # (extracted from broker's MT5 instrument specification)
        fee_bps = self.cost_model.commission_bps(symbol)
        # Convert bps to USD: fee_bps / 10000 * entry_value
        # We'll compute this at entry time with actual price
        # For now return the bps rate — caller will compute USD amount
        return fee_bps  # will be converted to USD in on_signal

    # ── Signal handling ─────────────────────────────────────────────

    def on_signal(self, symbol: str, direction: str | None, confidence: float,
                  features_dict: dict, mt5, timeframe: str = "H1") -> bool:
        """Called for EVERY signal from check_signals() or run_full_scan().

        Paper tracker applies its own filters. Returns True if a paper
        trade was opened.
        """
        # Track that this symbol was fed by the live scan (avoid double-scanning)
        self._live_scanned.add(f"{symbol}_{timeframe}")

        if direction is None:
            return False

        sym_cfg = self.sym_configs.get(symbol, {})
        threshold = sym_cfg.get("prob_threshold", cfg.ML_THRESHOLD)
        if confidence < threshold:
            return False

        # Market hours check
        sc_key = f"{symbol}_{timeframe}"
        if self._trading_schedule:
            is_open, _ = self._trading_schedule.is_trading_open(symbol)
            if not is_open:
                return False
            # Market is open — clear any session-close cooldown
            self._session_closed.pop(sc_key, None)

        # Already in this symbol+timeframe?
        with self._lock:
            if any(p.symbol == symbol and p.timeframe == timeframe for p in self.positions):
                return False

        # Don't reopen if recently closed by session end (market closed)
        sc_time = self._session_closed.get(sc_key)
        if sc_time and time.time() - sc_time < 3600:
            return False

        # Max positions — higher than live (testing ALL symbols x ALL timeframes)
        max_pos = 200
        with self._lock:
            if len(self.positions) >= max_pos:
                return False

        # Get REAL tick price from MT5
        broker_sym = sym_cfg.get("broker_symbol", symbol)
        tick = mt5.symbol_info_tick(broker_sym)
        if not tick or tick.ask <= 0:
            return False

        entry_price = tick.ask if direction == "BUY" else tick.bid
        spread_bps = (tick.ask - tick.bid) / tick.ask * 10000 if tick.ask > 0 else 0

        # Spread filter
        max_spread = sym_cfg.get("max_spread_pct", 0.005)
        if (tick.ask - tick.bid) / tick.ask > max_spread:
            if self.logger:
                self.logger.log('DEBUG', 'PaperTracker', 'SPREAD_REJECT',
                                f'[{self.account_id}] {symbol} spread {spread_bps:.1f}bps > limit')
            return False

        # ATR from MT5 — use the signal's timeframe
        atr = self._calc_atr(broker_sym, mt5, sym_cfg.get("atr_period", 14),
                             timeframe=timeframe)
        if atr is None or atr <= 0:
            return False

        # Exit params: prefer Optuna-optimized per symbol×TF, fallback to scaled defaults
        optuna_key = f"{symbol}_{timeframe}"
        optuna_ep = self._exit_optuna.get(optuna_key)
        if optuna_ep:
            exit_params = {
                "atr_sl_mult": optuna_ep["atr_sl_mult"],
                "atr_tp_mult": optuna_ep["atr_tp_mult"],
                "breakeven_atr": optuna_ep["breakeven_atr"],
                "trail_activation_atr": optuna_ep["trail_activation_atr"],
                "trail_distance_atr": optuna_ep["trail_distance_atr"],
                "exit_horizon": optuna_ep["exit_horizon"],
            }
        else:
            asset_class = sym_cfg.get("asset_class", "forex")
            defaults = self._TRAIL_DEFAULTS.get(asset_class, self._TRAIL_DEFAULTS['forex'])
            tf_scale = self._TF_SCALE.get(timeframe, self._TF_SCALE['H1'])
            exit_params = {
                "atr_sl_mult": sym_cfg.get("atr_sl_mult", 1.5) * tf_scale['sl_scale'],
                "atr_tp_mult": sym_cfg.get("atr_tp_mult", 4.5) * tf_scale['tp_scale'],
                "breakeven_atr": sym_cfg.get("breakeven_atr", defaults['breakeven_atr']) * tf_scale['breakeven_scale'],
                "trail_activation_atr": sym_cfg.get("trail_activation_atr", defaults['trail_activation_atr']) * tf_scale['trail_activation_scale'],
                "trail_distance_atr": sym_cfg.get("trail_distance_atr", defaults['trail_distance_atr']) * tf_scale['trail_distance_scale'],
                "exit_horizon": tf_scale['exit_horizon'],
            }

        # SL / TP
        sl_mult = exit_params["atr_sl_mult"]
        tp_mult = exit_params["atr_tp_mult"]
        confidence_scale = max(1.0, min(2.0, confidence / 0.55))

        if direction == "BUY":
            sl_price = entry_price - atr * sl_mult
            tp_price = entry_price + atr * tp_mult * confidence_scale
        else:
            sl_price = entry_price + atr * sl_mult
            tp_price = entry_price - atr * tp_mult * confidence_scale

        sl_distance = abs(entry_price - sl_price)

        # Position sizing — same as live bot
        risk_pct = sym_cfg.get("risk_per_trade", 0.003) * self.risk_scale
        # Ensure symbol is selected in MarketWatch before reading info
        mt5.symbol_select(broker_sym, True)
        sym_info = mt5.symbol_info(broker_sym)
        contract_size = 1.0
        lot_step = 0.01
        min_lot = 0.01
        tick_value = 0.0
        tick_size = 0.0
        if sym_info:
            contract_size = getattr(sym_info, 'trade_contract_size', 1.0) or 1.0
            lot_step = getattr(sym_info, 'volume_step', 0.01) or 0.01
            min_lot = getattr(sym_info, 'volume_min', 0.01) or 0.01
            tick_value = getattr(sym_info, 'trade_tick_value', 0.0) or 0.0
            tick_size = getattr(sym_info, 'trade_tick_size', 0.0) or 0.0

        # Use tick_value/tick_size for proper quote→USD conversion (JPY, HUF, etc.)
        if tick_value > 0 and tick_size > 0:
            risk_per_lot = (sl_distance / tick_size) * tick_value
        else:
            # tick_value unavailable (market closed?) — skip trade to avoid wrong sizing
            self.logger.log('WARNING', 'PaperTracker', 'TICK_VALUE_ZERO',
                            f'{symbol}: tick_value=0, skipping paper trade (market likely closed)')
            return False
        if risk_per_lot <= 0:
            return False
        lot_size = (self.equity * risk_pct) / risk_per_lot
        if lot_step > 0:
            lot_size = round(lot_size / lot_step) * lot_step
        lot_size = max(min_lot, lot_size)

        # Cost calculation — commission per lot (round-trip)
        comm_spec = self.cost_model._lookup_spec(symbol)
        if comm_spec and comm_spec.get("commission_type") == "USD/LOT":
            commission_usd = comm_spec["commission"] * lot_size * 2
        else:
            fee_bps = self.cost_model.commission_bps(symbol)
            # Notional in USD = lots * contract_size * price
            notional_usd = lot_size * contract_size * entry_price
            commission_usd = (fee_bps / 10000) * notional_usd * 2
        cost_pct = self.cost_model.total_cost_pct(symbol, spread_bps=spread_bps, pessimistic=False)

        now = datetime.now(timezone.utc).isoformat()
        pos = PaperPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=now,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_price=tp_price,
            atr=atr,
            confidence=confidence,
            cost_pct=cost_pct,
            commission_usd=commission_usd,
            spread_bps=spread_bps,
            exit_params=exit_params,
            contract_size=contract_size,
            tick_value=tick_value,
            tick_size=tick_size,
            timeframe=timeframe,
        )

        # Insert into DB
        import json as _json
        conn = self._connect()
        cur = conn.execute(
            """INSERT INTO paper_trades
               (timestamp, symbol, direction, entry_price, sl_price, tp_price,
                lot_size, confidence, cost_pct, commission_usd, spread_bps,
                contract_size, status, bars_held, account_id, atr, exit_params, timeframe,
                tick_value, tick_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0, ?, ?, ?, ?, ?, ?)""",
            (now, symbol, direction, entry_price, sl_price, tp_price,
             lot_size, confidence, cost_pct, commission_usd, spread_bps,
             contract_size, self.account_id, atr,
             _json.dumps(exit_params), timeframe, tick_value, tick_size),
        )
        pos.db_id = cur.lastrowid
        conn.commit()
        conn.close()

        with self._lock:
            self.positions.append(pos)

        if self.logger:
            self.logger.log('INFO', 'PaperTracker', 'PAPER_ENTRY',
                            f'[{self.account_id}] {symbol} {direction} @ {entry_price:.5f} '
                            f'lots={lot_size:.2f} SL={sl_price:.5f} TP={tp_price:.5f} '
                            f'conf={confidence:.3f} comm=${commission_usd:.2f} '
                            f'spread={spread_bps:.1f}bps tf={timeframe}')
        return True

    # ── Continuous price monitor ────────────────────────────────────

    def start_monitor(self, mt5, interval: float = 1.0):
        """Start background thread monitoring paper positions with real prices."""
        self._mt5 = mt5
        # Fix tick_value/tick_size for positions restored before MT5 was available
        for pos in self.positions:
            if (pos.tick_value == 0.0 or pos.tick_size == 0.0) and mt5:
                broker_sym = self.sym_configs.get(pos.symbol, {}).get(
                    "broker_symbol", pos.symbol)
                si = mt5.symbol_info(broker_sym)
                if si:
                    pos.tick_value = getattr(si, 'trade_tick_value', 0.0) or 0.0
                    pos.tick_size = getattr(si, 'trade_tick_size', 0.0) or 0.0
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,),
            daemon=True, name=f"PaperMonitor-{self.account_id}",
        )
        self._monitor_thread.start()
        if self.logger:
            self.logger.log('INFO', 'PaperTracker', 'MONITOR_STARTED',
                            f'[{self.account_id}] Paper monitor started (interval={interval}s)')

    def stop_monitor(self):
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self._monitor_thread = None

    def _monitor_loop(self, interval: float):
        """Continuous monitoring — check REAL prices against SL/TP."""
        while self._monitor_running:
            with self._lock:
                positions_copy = list(self.positions)

            if not positions_copy:
                time.sleep(max(interval, 1.0))
                continue

            # Quick bridge health check — skip cycle if bridge is down
            try:
                test_sym = positions_copy[0].symbol
                broker_sym = self.sym_configs.get(test_sym, {}).get(
                    "broker_symbol", test_sym)
                probe = self._mt5.symbol_info_tick(broker_sym)
                if probe is None:
                    time.sleep(5)
                    continue
            except Exception:
                time.sleep(5)
                continue

            for pos in positions_copy:
                if not self._monitor_running:
                    break
                try:
                    broker_sym = self.sym_configs.get(pos.symbol, {}).get(
                        "broker_symbol", pos.symbol)
                    tick = self._mt5.symbol_info_tick(broker_sym)
                    if not tick or tick.bid <= 0:
                        continue

                    # Close price: bid for BUY (sell to close), ask for SELL (buy to close)
                    current = tick.bid if pos.direction == "BUY" else tick.ask
                    price_profit = (current - pos.entry_price) if pos.direction == "BUY" \
                        else (pos.entry_price - current)

                    # Update high water mark
                    if price_profit > pos.high_water:
                        pos.high_water = price_profit

                    # Check SL
                    sl_hit = False
                    if pos.direction == "BUY" and current <= pos.sl_price:
                        sl_hit = True
                    elif pos.direction == "SELL" and current >= pos.sl_price:
                        sl_hit = True

                    if sl_hit:
                        # Determine if SL was moved to profit (trailing/BE) or original SL
                        if pos.direction == "BUY" and pos.sl_price > pos.entry_price:
                            status = "CLOSED_TRAIL"
                        elif pos.direction == "SELL" and pos.sl_price < pos.entry_price:
                            status = "CLOSED_TRAIL"
                        else:
                            status = "CLOSED_SL"
                        # Use actual market price, not SL level — real fills
                        # happen at current bid/ask, not exactly at SL
                        self._close_position(pos, current, status)
                        continue

                    # Check TP
                    tp_hit = False
                    if pos.direction == "BUY" and current >= pos.tp_price:
                        tp_hit = True
                    elif pos.direction == "SELL" and current <= pos.tp_price:
                        tp_hit = True

                    if tp_hit:
                        # Use actual market price, not TP level
                        self._close_position(pos, current, "CLOSED_TP")
                        continue

                    # Trailing stop + breakeven
                    old_sl = pos.sl_price
                    self._update_trailing(pos, current, price_profit)

                    # Persist SL + high_water changes to DB
                    if pos.db_id and (pos.sl_price != old_sl or price_profit > 0):
                        try:
                            _conn = self._connect()
                            _conn.execute(
                                "UPDATE paper_trades SET sl_price = ?, high_water = ? WHERE id = ?",
                                (pos.sl_price, pos.high_water, pos.db_id))
                            _conn.commit()
                            _conn.close()
                        except Exception as _db_err:
                            if self.logger:
                                self.logger.log('ERROR', 'PaperTracker', 'DB_PERSIST_ERROR',
                                                f'[{self.account_id}] {pos.symbol}: {_db_err}')

                except Exception as e:
                    if self.logger:
                        self.logger.log('ERROR', 'PaperTracker', 'MONITOR_ERROR',
                                        f'[{self.account_id}] {pos.symbol}: {e}')

            if interval > 0:
                time.sleep(interval)

    def _update_trailing(self, pos: PaperPosition, current: float, price_profit: float):
        """Update trailing stop and breakeven for a paper position."""
        atr = pos.atr
        if atr <= 0:
            return

        ep = pos.exit_params
        asset_class = self.sym_configs.get(pos.symbol, {}).get("asset_class", "forex")
        defaults = self._TRAIL_DEFAULTS.get(asset_class, self._TRAIL_DEFAULTS['forex'])

        be_trigger = ep.get("breakeven_atr", defaults['breakeven_atr']) * atr
        trail_activation = ep.get("trail_activation_atr", defaults['trail_activation_atr']) * atr
        trail_distance = ep.get("trail_distance_atr", defaults['trail_distance_atr']) * atr

        if price_profit <= 0:
            return

        # Progressive trail (same tiers as position_manager)
        _PROG_TIERS = [
            (0.0, 1.00), (0.3, 0.80), (0.8, 0.60), (1.3, 0.40), (1.8, 0.30),
        ]
        if price_profit >= trail_activation:
            profit_beyond = (price_profit - trail_activation) / atr
            factor = _PROG_TIERS[-1][1]
            for i in range(len(_PROG_TIERS) - 1):
                p0, f0 = _PROG_TIERS[i]
                p1, f1 = _PROG_TIERS[i + 1]
                if profit_beyond <= p1:
                    t = (profit_beyond - p0) / (p1 - p0) if p1 > p0 else 0
                    factor = f0 + t * (f1 - f0)
                    break
            adj_trail = trail_distance * factor

            # Trailing SL
            if pos.direction == "BUY":
                candidate_sl = current - adj_trail
                if candidate_sl > pos.sl_price:
                    pos.sl_price = candidate_sl
            else:
                candidate_sl = current + adj_trail
                if pos.sl_price == 0 or candidate_sl < pos.sl_price:
                    pos.sl_price = candidate_sl

        # Breakeven
        elif price_profit >= be_trigger:
            # Estimate costs in price units — cap so BE SL never exceeds current price
            cost_price = pos.entry_price * pos.cost_pct
            cost_buf = cost_price + 0.05 * atr
            # Cap cost_buf at 80% of current profit to avoid setting SL beyond market
            max_buf = price_profit * 0.8
            cost_buf = min(cost_buf, max_buf)
            if pos.direction == "BUY":
                be_sl = pos.entry_price + cost_buf
                if be_sl > pos.sl_price:
                    pos.sl_price = be_sl
            else:
                be_sl = pos.entry_price - cost_buf
                if pos.sl_price == 0 or be_sl < pos.sl_price:
                    pos.sl_price = be_sl

    # ── Bar close handling ──────────────────────────────────────────

    def on_bar_close(self, mt5, timeframe: str = "H1"):
        """Called on bar close. Increment bars_held for matching TF positions, check horizon."""
        with self._lock:
            positions_copy = list(self.positions)

        updated_ids = []
        for pos in positions_copy:
            if pos.timeframe != timeframe:
                continue
            pos.bars_held += 1
            updated_ids.append((pos.bars_held, pos.db_id))
            horizon = pos.exit_params.get("exit_horizon", 24)
            if pos.bars_held >= horizon:
                broker_sym = self.sym_configs.get(pos.symbol, {}).get(
                    "broker_symbol", pos.symbol)
                tick = mt5.symbol_info_tick(broker_sym)
                if tick:
                    price = tick.bid if pos.direction == "BUY" else tick.ask
                    self._close_position(pos, price, "CLOSED_HORIZON")

        # Persist bars_held to DB so restarts don't lose count
        if updated_ids:
            conn = self._connect()
            conn.executemany(
                "UPDATE paper_trades SET bars_held = ? WHERE id = ?",
                updated_ids,
            )
            conn.commit()
            conn.close()

    # ── Position closing ────────────────────────────────────────────

    def _close_position(self, pos: PaperPosition, exit_price: float, status: str):
        """Close a paper position and log to DB."""
        # Raw PnL in price units → USD
        if pos.direction == "BUY":
            raw_pnl_price = exit_price - pos.entry_price
        else:
            raw_pnl_price = pos.entry_price - exit_price

        # Use tick_value/tick_size for proper quote→USD conversion
        if pos.tick_value > 0 and pos.tick_size > 0:
            raw_pnl_usd = (raw_pnl_price / pos.tick_size) * pos.tick_value * pos.lot_size
        else:
            # tick_value was 0 at open — should not happen after fix, but safe fallback
            # Re-read tick_value from MT5 as a last resort
            mt5 = self.mt5
            if mt5:
                mt5.symbol_select(pos.symbol, True)
                info = mt5.symbol_info(pos.symbol)
                if info and info.trade_tick_value > 0 and info.trade_tick_size > 0:
                    raw_pnl_usd = (raw_pnl_price / info.trade_tick_size) * info.trade_tick_value * pos.lot_size
                else:
                    raw_pnl_usd = raw_pnl_price * pos.lot_size * pos.contract_size
            else:
                raw_pnl_usd = raw_pnl_price * pos.lot_size * pos.contract_size
        # Deduct real commission
        pnl = raw_pnl_usd - pos.commission_usd

        now = datetime.now(timezone.utc).isoformat()

        # Update DB
        conn = self._connect()
        conn.execute(
            """UPDATE paper_trades SET
                status = ?, exit_price = ?, exit_timestamp = ?,
                pnl = ?, bars_held = ?, sl_price = ?
               WHERE id = ?""",
            (status, exit_price, now, pnl, pos.bars_held, pos.sl_price, pos.db_id),
        )
        conn.commit()
        conn.close()

        # Track cumulative P&L but keep equity fixed at starting_equity
        # so position sizing stays constant for fair signal evaluation
        self.equity = self.starting_equity

        # Remove from active positions
        with self._lock:
            self.positions = [p for p in self.positions if p.db_id != pos.db_id]

        if self.logger:
            self.logger.log('INFO', 'PaperTracker', 'PAPER_EXIT',
                            f'[{self.account_id}] {pos.symbol} {pos.direction} '
                            f'{status} @ {exit_price:.5f} PnL=${pnl:+.2f} '
                            f'bars={pos.bars_held} comm=${pos.commission_usd:.2f} '
                            f'equity=${self.equity:,.2f}')

    def close_market_closed(self, mt5, trading_schedule):
        """Close paper positions whose market is closed (session end, weekend, etc.)."""
        with self._lock:
            positions_copy = list(self.positions)
        closed = 0
        for pos in positions_copy:
            is_open, _ = trading_schedule.is_trading_open(pos.symbol)
            if is_open:
                continue
            broker_sym = self.sym_configs.get(pos.symbol, {}).get(
                "broker_symbol", pos.symbol)
            try:
                tick = mt5.symbol_info_tick(broker_sym)
                if tick and hasattr(tick, "bid") and tick.bid > 0:
                    price = tick.bid if pos.direction == "BUY" else tick.ask
                    self._close_position(pos, price, "CLOSED_SESSION")
                    self._session_closed[f"{pos.symbol}_{pos.timeframe}"] = time.time()
                    closed += 1
            except Exception:
                pass
        if closed and self.logger:
            self.logger.log('INFO', 'PaperTracker', 'SESSION_CLOSE',
                            f'[{self.account_id}] Closed {closed} positions (market closed)')
        return closed

    # ── ATR calculation ─────────────────────────────────────────────

    _TF_MAP = {
        "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
        "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    }

    def _calc_atr(self, broker_sym: str, mt5, period: int = 14,
                  timeframe: str = "H1") -> float | None:
        """Calculate ATR from real bars via MT5 for any timeframe."""
        try:
            mt5_tf = getattr(mt5, self._TF_MAP.get(timeframe, "TIMEFRAME_H1"))
            bars = mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, period + 5)
            if bars is None or len(bars) < period + 1:
                return None
            tr_list = []
            for i in range(1, len(bars)):
                hl = bars[i]['high'] - bars[i]['low']
                hpc = abs(bars[i]['high'] - bars[i - 1]['close'])
                lpc = abs(bars[i]['low'] - bars[i - 1]['close'])
                tr_list.append(max(hl, hpc, lpc))
            return sum(tr_list[-period:]) / period
        except Exception:
            return None

    # ── Daily summary ───────────────────────────────────────────────

    def daily_summary(self) -> dict:
        """Return paper account stats for Discord / daily report."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row

        # Today's closed trades
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        closed_today = conn.execute(
            """SELECT * FROM paper_trades
               WHERE account_id = ? AND status != 'OPEN'
               AND exit_timestamp LIKE ?""",
            (self.account_id, f"{today}%"),
        ).fetchall()

        total_pnl = sum(r["pnl"] or 0 for r in closed_today)
        wins = sum(1 for r in closed_today if (r["pnl"] or 0) > 0)
        losses = sum(1 for r in closed_today if (r["pnl"] or 0) <= 0)
        total_comm = sum(r["commission_usd"] or 0 for r in closed_today)

        # All-time stats
        all_closed = conn.execute(
            """SELECT COUNT(*) as n, SUM(pnl) as total,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
               FROM paper_trades
               WHERE account_id = ? AND status != 'OPEN'""",
            (self.account_id,),
        ).fetchone()

        conn.close()

        # Unrealized P&L from open positions
        unrealized = 0.0
        with self._lock:
            open_count = len(self.positions)
            for pos in self.positions:
                if self._mt5:
                    broker_sym = self.sym_configs.get(pos.symbol, {}).get(
                        "broker_symbol", pos.symbol)
                    tick = self._mt5.symbol_info_tick(broker_sym)
                    if tick:
                        current = tick.bid if pos.direction == "BUY" else tick.ask
                        pnl_price = (current - pos.entry_price) if pos.direction == "BUY" \
                            else (pos.entry_price - current)
                        if pos.tick_value > 0 and pos.tick_size > 0:
                            unrealized += (pnl_price / pos.tick_size) * pos.tick_value * pos.lot_size
                        else:
                            unrealized += pnl_price * pos.lot_size * pos.contract_size

        # Log equity snapshot
        conn = self._connect()
        conn.execute(
            """INSERT INTO paper_equity
               (timestamp, equity, daily_pnl, open_positions, account_id)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), self.equity, total_pnl,
             open_count, self.account_id),
        )
        conn.commit()
        conn.close()

        return {
            "account_id": self.account_id,
            "equity": self.equity,
            "daily_pnl": total_pnl,
            "daily_trades": len(closed_today),
            "daily_wins": wins,
            "daily_losses": losses,
            "daily_commission": total_comm,
            "win_rate": wins / len(closed_today) if closed_today else 0,
            "open_positions": open_count,
            "unrealized_pnl": unrealized,
            "all_time_trades": all_closed["n"] or 0,
            "all_time_pnl": all_closed["total"] or 0,
            "all_time_winrate": (all_closed["wins"] or 0) / (all_closed["n"] or 1),
            "return_pct": (self.equity - self.starting_equity) / self.starting_equity * 100,
        }

    def send_daily_discord(self):
        """Send daily paper trading summary to Discord."""
        if not self.discord:
            return
        stats = self.daily_summary()
        wr = f"{stats['win_rate']*100:.0f}%" if stats['daily_trades'] > 0 else "N/A"
        body = (
            f"**Equity:** ${stats['equity']:,.2f} "
            f"({stats['return_pct']:+.2f}%)\n"
            f"**Daily P&L:** ${stats['daily_pnl']:+,.2f} "
            f"({stats['daily_trades']} trades, WR {wr})\n"
            f"**Commission paid:** ${stats['daily_commission']:,.2f}\n"
            f"**Open positions:** {stats['open_positions']} "
            f"(unrealized: ${stats['unrealized_pnl']:+,.2f})\n"
            f"**All-time:** {stats['all_time_trades']} trades, "
            f"${stats['all_time_pnl']:+,.2f}, "
            f"WR {stats['all_time_winrate']*100:.0f}%"
        )
        color = "green" if stats['daily_pnl'] >= 0 else "red"
        self.discord.send(
            f"PAPER [{self.account_id}]", body, color,
        )
