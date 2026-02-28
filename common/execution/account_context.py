"""
Account context — bundles all per-account instances for multi-account trading.

Each account gets its own isolated:
  - MT5 bridge connection (different port per broker terminal)
  - Compliance (FTMOCompliance with account-specific limits)
  - DrawdownGuard (parameterized thresholds)
  - CorrelationGuard (applied to this account's positions only)
  - PositionSizingEngine
  - OrderRouter
  - PositionManager
  - Audit logger (separate DB per account)
  - HeartbeatMonitor

Shared across all accounts:
  - ML models / filters (same signals)
  - Feature builder
  - Trading schedule (same market hours)
  - Signal generation (computed once, routed to each account)
  - Model decay tracker
  - Config (cfg singleton)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit.audit_logger import BlackoutLogger
from execution.order_router import OrderRouter
from execution.position_manager import PositionManager
from live.healthcheck import HeartbeatMonitor
from risk.correlation_guard import CorrelationGuard
from risk.drawdown_guard import DrawdownGuard
from risk.position_sizing import PositionSizingEngine
from tools.ftmo_compliance import FTMOCompliance
from tools.mt5_bridge import MT5BridgeClient, get_mt5_bridge, initialize_mt5


class AccountContext:
    """Holds all per-account instances. One per trading account."""

    def __init__(self, account_id: str, account_cfg: dict, discord=None):
        self.account_id = account_id
        self.account_cfg = account_cfg
        self.name = account_cfg.get("name", account_id)
        self.enabled = account_cfg.get("enabled", False)
        self.bridge_port = account_cfg.get("bridge_port", 5055)
        self.account_size = account_cfg.get("account_size", 100000)
        self.risk_scale = account_cfg.get("risk_scale", 1.0)
        self.max_positions = account_cfg.get("max_positions", 12)
        self.max_positions_alpha = account_cfg.get("max_positions_alpha", 80)
        self.discord = discord
        self.trading_paused = account_cfg.get("trading_paused", False)

        # Per-account symbol configs (keyed by broker symbol name)
        self.symbols: dict = {}
        self.margin_leverage: dict = {}  # Stored separately from symbols
        self._internal_to_broker: dict[str, str] = {}  # e.g. UNIUSD → UNI/USD
        self._broker_to_internal: dict[str, str] = {}  # e.g. UNI/USD → UNIUSD
        self._load_symbols(account_cfg.get("config_path"))

        # Per-account ML models + Optuna params
        self.model_dir: str = str(REPO_ROOT / account_cfg.get("model_dir", "models/sovereign_models"))
        self.optuna_csv: str = str(REPO_ROOT / account_cfg.get("optuna_csv", ""))
        self.optuna_params: dict = {}
        self.filters: dict = {}  # symbol → SovereignMLFilter

        # These are initialized in initialize()
        self.mt5: MT5BridgeClient | None = None
        self.logger: BlackoutLogger | None = None
        self.ftmo: FTMOCompliance | None = None
        self.drawdown_guard: DrawdownGuard | None = None
        self.correlation_guard: CorrelationGuard | None = None
        self.position_sizer: PositionSizingEngine | None = None
        self.order_router: OrderRouter | None = None
        self.position_manager: PositionManager | None = None
        self.heartbeat: HeartbeatMonitor | None = None

        self.safe_mode = False
        self.emergency_stop = False
        self._tracked_tickets: dict = {}

    def _load_symbols(self, config_path: str | None):
        """Load per-account symbol configs from a dedicated JSON file."""
        if not config_path:
            return
        full_path = REPO_ROOT / config_path
        if not full_path.exists():
            return
        with open(full_path) as f:
            raw = json.load(f)
        for sym, sym_cfg in raw.items():
            if sym == "margin_leverage":
                self.margin_leverage = sym_cfg  # Store separately, NOT in symbols
                continue
            broker_sym = sym_cfg.get("broker_symbol", sym)
            self.symbols[broker_sym] = sym_cfg
            self._internal_to_broker[sym] = broker_sym
            self._broker_to_internal[broker_sym] = sym

    def load_optuna_params(self):
        """Load Optuna params from this account's CSV."""
        if not self.optuna_csv or not os.path.exists(self.optuna_csv):
            return
        try:
            import polars as pl_
            df = pl_.read_csv(self.optuna_csv)
            ok = df.filter(pl_.col("status") == "ok")
            for row in ok.iter_rows(named=True):
                sym = row["symbol"]
                eta = float(row["best_eta"]) if row.get("best_eta") else 0.03
                colsample_bylevel = float(row["best_colsample_bylevel"]) if row.get("best_colsample_bylevel") else 0.75
                self.optuna_params[sym] = {
                    "xgb_params": {
                        "booster": "gbtree", "tree_method": "hist",
                        "device": "cuda", "sampling_method": "gradient_based",
                        "objective": "binary:logistic", "eval_metric": "logloss",
                        "max_depth": int(row["best_max_depth"]),
                        "eta": eta,
                        "gamma": float(row["best_gamma"]),
                        "subsample": float(row["best_subsample"]),
                        "colsample_bytree": float(row["best_colsample_bytree"]),
                        "colsample_bylevel": colsample_bylevel,
                        "reg_alpha": float(row["best_reg_alpha"]),
                        "reg_lambda": float(row["best_reg_lambda"]),
                        "min_child_weight": float(row["best_min_child_weight"]),
                        "max_bin": 512, "grow_policy": "lossguide", "verbosity": 0,
                    },
                    "num_boost_round": int(row["best_num_boost_round"]),
                    "optuna_ev": float(row["best_ev"]),
                    "timeframe": row.get("timeframe", "H1") or "H1",
                    "fee_bps": float(row["fee_bps"]) if row.get("fee_bps") is not None else 3.0,
                }
        except Exception as e:
            print(f"[{self.name}] Failed to load Optuna CSV {self.optuna_csv}: {e}")

    def init_filters(self, logger):
        """Create per-account ML filters with account-specific model dir."""
        from engine.inference import SovereignMLFilter
        os.makedirs(self.model_dir, exist_ok=True)
        for broker_sym, sym_cfg in self.symbols.items():
            internal_sym = self._broker_to_internal.get(broker_sym, broker_sym)
            filt = SovereignMLFilter(internal_sym, logger, model_dir=self.model_dir)
            filt.load_model()
            self.filters[internal_sym] = filt

    def get_internal_symbol(self, broker_symbol: str) -> str:
        """Translate broker symbol name to internal/model name."""
        return self._broker_to_internal.get(broker_symbol, broker_symbol)

    def get_broker_symbol(self, internal_symbol: str) -> str:
        """Translate internal symbol name to broker symbol name."""
        return self._internal_to_broker.get(internal_symbol, internal_symbol)

    @staticmethod
    def _query_day_start_balance(audit_db: str) -> float | None:
        """Query the audit DB for today's first heartbeat balance.

        This survives bot restarts — the heartbeats table always has the
        real balance at day start, even if the bot was restarted mid-day.
        """
        import sqlite3
        from datetime import date
        try:
            conn = sqlite3.connect(audit_db)
            row = conn.execute(
                "SELECT account_balance FROM heartbeats "
                "WHERE timestamp >= ? ORDER BY id ASC LIMIT 1",
                (date.today().isoformat() + 'T00:00:00',)
            ).fetchone()
            conn.close()
            if row:
                return row[0]
        except Exception:
            pass
        return None

    def initialize(self, trading_schedule, on_safe_mode=None) -> bool:
        """Initialize all per-account instances. Returns True on success."""
        acfg = self.account_cfg

        # 1. Audit logger (separate DB per account)
        audit_db = os.path.join(str(REPO_ROOT), acfg.get("audit_db", f"audit/{self.account_id}.db"))
        self.logger = BlackoutLogger(db_path=audit_db)

        # 2. MT5 bridge connection
        self.mt5 = get_mt5_bridge(port=self.bridge_port, name=self.account_id)

        ok, mt5_error, mode = initialize_mt5(self.mt5)
        if not ok:
            self.logger.log('ERROR', 'AccountContext', 'MT5_INIT_FAILED',
                            f'[{self.name}] MT5 init failed on port {self.bridge_port}: {mt5_error}')
            return False

        account_info = self.mt5.account_info()
        if account_info:
            self.logger.log('INFO', 'AccountContext', 'MT5_INITIALIZED',
                            f'[{self.name}] Account {account_info.login} | '
                            f'Balance ${account_info.balance:,.2f} | '
                            f'Equity ${account_info.equity:,.2f} | '
                            f'Port {self.bridge_port}')
            initial_balance = account_info.balance
        else:
            initial_balance = self.account_size

        # Restore real day-start balance from audit DB (survives restarts)
        day_start = self._query_day_start_balance(audit_db)
        if day_start is not None and day_start != initial_balance:
            self.logger.log('INFO', 'AccountContext', 'DAY_START_RESTORED',
                            f'[{self.name}] Day-start balance from DB: ${day_start:,.2f} '
                            f'(current: ${initial_balance:,.2f}, '
                            f'day delta: ${initial_balance - day_start:+,.2f})')
        daily_start = day_start if day_start is not None else initial_balance

        # 3. Compliance (parameterized per account)
        self.ftmo = FTMOCompliance(
            initial_balance=daily_start,
            logger=self.logger,
            discord=self.discord,
            account_name=self.name,
            max_daily_loss_pct=acfg.get("max_daily_loss_pct", 0.05),
            max_total_dd_pct=acfg.get("max_total_dd_pct", 0.10),
            total_dd_warning_pct=acfg.get("total_dd_warning_pct", 0.08),
            dd_type=acfg.get("dd_type", "trailing"),
        )
        self.ftmo.load_last_trade_time(audit_db)

        # 4. Drawdown guard (parameterized per account, with persisted day-start)
        self.drawdown_guard = DrawdownGuard(
            self.logger, self.discord,
            account_name=self.name,
            daily_loss_pct=acfg.get("internal_daily_loss_pct", 0.035),
            profit_lock_pct=acfg.get("internal_profit_lock_pct", 0.03),
            dd_recovery_threshold=acfg.get("dd_recovery_threshold", 0.04),
            dd_recovery_exit=acfg.get("dd_recovery_exit", 0.01),
            profit_gate_pct=acfg.get("profit_gate_pct", 0.015),
            profit_gate_min_conf=acfg.get("profit_gate_min_conf", 0.80),
            daily_start_balance=daily_start,
        )

        # 5. Correlation guard (per account — checks this account's positions)
        self.correlation_guard = CorrelationGuard(self.logger)

        # 6. Position sizing engine
        self.position_sizer = PositionSizingEngine(self.logger, self.mt5)

        # 7. Order router (per account — sends orders to this MT5)
        self.order_router = OrderRouter(
            self.logger, self.mt5, self.position_sizer, trading_schedule,
            discord=self.discord, ftmo=self.ftmo,
            drawdown_guard=self.drawdown_guard,
            correlation_guard=self.correlation_guard,
            account_name=self.name,
            account_symbols=self.symbols or None,
            risk_scale=self.risk_scale,
        )
        self.order_router._margin_leverage = self.margin_leverage or None

        # 8. Position manager (per account — manages this account's positions)
        monitor_interval = acfg.get("monitor_interval", 0.5)
        self.position_manager = PositionManager(
            self.logger, self.mt5, self.discord,
            account_symbols=self.symbols or None,
            account_name=self.name,
        )
        self.position_manager._trading_schedule = trading_schedule
        self.position_manager._ftmo = self.ftmo

        # Floating profit close — wired into the monitor loop
        from live.emergency_kill import profit_close_all
        self.position_manager._profit_close_pct  = acfg.get("floating_profit_close_pct", 0)
        self.position_manager._profit_hard_pct   = acfg.get("floating_profit_hard_pct", 0)
        self.position_manager._profit_tighten_pct = acfg.get("floating_profit_tighten_pct", 0)
        self.position_manager._account_size      = self.account_size

        def _on_profit_close(floating: float, hard: bool):
            profit_close_all(self.logger, self.mt5, floating,
                             self.account_size, self.name, self.discord)
            if hard and on_safe_mode:
                on_safe_mode(self.account_id,
                             f'floating profit hard stop +{floating/self.account_size:.1%}')

        self.position_manager._on_profit_close = _on_profit_close

        self.position_manager.start_monitor(interval=monitor_interval)

        # 9. Heartbeat monitor (per account — monitors this MT5 connection)
        def _enter_safe_mode(reason):
            if on_safe_mode:
                on_safe_mode(self.account_id, reason)
            self.safe_mode = True
            self.order_router.safe_mode = True

        self.heartbeat = HeartbeatMonitor(
            self.logger, self.mt5,
            on_disconnect=_enter_safe_mode,
            discord=self.discord,
        )
        # Pre-seed day-start balance BEFORE thread starts to prevent
        # the monitor loop from overwriting with current (wrong) balance
        from datetime import datetime
        self.heartbeat.initial_balance = initial_balance
        self.heartbeat.daily_start_balance = daily_start
        self.heartbeat.last_reset_date = datetime.now().date()
        self.heartbeat.start()

        self.logger.log('INFO', 'AccountContext', 'ACCOUNT_READY',
                        f'[{self.name}] All components initialized | '
                        f'Port {self.bridge_port} | Risk scale {self.risk_scale}x')
        return True

    def execute_trade(self, symbol: str, direction: str, ml_confidence: float,
                      features_dict: dict | None = None,
                      margin_budget: float | None = None) -> bool:
        """Execute trade on this account with all guardrails.

        Symbol filtering: if this account has its own symbols config, only
        trade symbols that are in that config. Translates internal symbol
        names (e.g. UNIUSD) to broker names (e.g. UNI/USD) if needed.
        """
        if not self.enabled or self.safe_mode or self.emergency_stop or self.trading_paused:
            return False

        # Per-account symbol filtering
        if self.symbols:
            broker_sym = self._internal_to_broker.get(symbol, symbol)
            if broker_sym not in self.symbols:
                return False  # Not configured for this account
            symbol = broker_sym  # Use broker name for order execution

        return self.order_router.execute_trade(
            symbol, direction, ml_confidence,
            gpu_trading_pause=HeartbeatMonitor.GPU_TRADING_PAUSE,
            features_dict=features_dict,
            margin_budget=margin_budget,
        )

    def manage_positions(self):
        """Run position management for this account."""
        if self.position_manager and not self.emergency_stop:
            self.position_manager.manage_positions(
                running=not self.emergency_stop,
                emergency_stop=self.emergency_stop,
            )

    def check_total_dd(self) -> bool:
        """Check total drawdown. Returns True if must emergency close."""
        if not self.ftmo or not self.mt5:
            return False
        acct = self.mt5.account_info()
        if not acct:
            return False
        must_close, reason = self.ftmo.check_total_dd(acct.equity)
        return must_close

    def stop(self):
        """Graceful shutdown of this account's components."""
        if self.position_manager:
            self.position_manager.stop_monitor()
        if self.heartbeat:
            self.heartbeat.stop()
        if self.mt5:
            try:
                self.mt5.shutdown()
            except Exception:
                pass
        self.logger.log('INFO', 'AccountContext', 'ACCOUNT_STOPPED',
                        f'[{self.name}] Account stopped')

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"AccountContext({self.name}, port={self.bridge_port}, {status})"


def load_accounts(discord=None) -> dict[str, AccountContext]:
    """Load account configurations from config/accounts.yaml.

    Returns dict of account_id → AccountContext (not yet initialized).
    Falls back to a single default account from ftmo.yaml if accounts.yaml doesn't exist.
    """
    import yaml

    accounts_path = REPO_ROOT / "config" / "accounts.yaml"

    if accounts_path.exists():
        with open(accounts_path) as f:
            data = yaml.safe_load(f)
        accounts = {}
        for acct_id, acct_cfg in data.get("accounts", {}).items():
            accounts[acct_id] = AccountContext(acct_id, acct_cfg, discord=discord)
        return accounts

    # Fallback: build a single account from ftmo.yaml (backwards compatibility)
    ftmo_path = REPO_ROOT / "config" / "ftmo.yaml"
    if ftmo_path.exists():
        with open(ftmo_path) as f:
            ftmo_cfg = yaml.safe_load(f)
    else:
        ftmo_cfg = {}

    default_cfg = {
        "name": "FTMO (default)",
        "enabled": True,
        "bridge_port": int(os.getenv("MT5_BRIDGE_PORT", "5055")),
        "account_size": ftmo_cfg.get("account_size", 100000),
        "max_daily_loss_pct": ftmo_cfg.get("max_daily_loss_pct", 0.05),
        "max_total_dd_pct": ftmo_cfg.get("max_total_loss_pct", 0.10),
        "total_dd_warning_pct": 0.08,
        "dd_type": "trailing",
        "internal_daily_loss_pct": 0.035,
        "internal_profit_lock_pct": 0.03,
        "dd_recovery_threshold": 0.04,
        "dd_recovery_exit": 0.01,
        "risk_scale": 1.0,
        "max_positions": 12,
        "max_positions_alpha": 80,
        "audit_db": "audit/sovereign_log.db",
    }
    return {"default": AccountContext("default", default_cfg, discord=discord)}
