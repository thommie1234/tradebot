#!/usr/bin/env python3
"""
SOVEREIGN BOT — Production H1 Multi-Symbol Trading System
===========================================================

Orchestration layer: run_bot.py connects all modules.

Usage:
    python3 live/run_bot.py --dry-run          # Show plan, no trading
    python3 live/run_bot.py --train            # Train models from tick data
    python3 live/run_bot.py --live             # Live H1 trading loop
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Add repo root to path
_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

# Config
from config.loader import cfg, load_config

# Logging
from audit.audit_logger import BlackoutLogger
from audit.feature_logger import FeatureLogger

# Engine
from engine.inference import SovereignMLFilter, _ensure_ml_imports
from engine.decay_tracker import ModelDecayTracker
# check_signals / ScanCache no longer needed — all scanning via MultiTFScanner
from engine.multi_tf_scanner import MultiTFScanner

# Risk
from risk.position_sizing import PositionSizingEngine
from risk.ftmo_guard import TradingSchedule

# Execution
from execution.broker_api import MT5_AVAILABLE, mt5, initialize, get_symbol_info, shutdown
from execution.order_router import OrderRouter
from execution.position_manager import PositionManager
from execution.account_context import AccountContext, load_accounts

# Live
from live.healthcheck import HeartbeatMonitor
from live.emergency_kill import emergency_close_all, friday_auto_close, friday_progressive_close, profit_close_all, profit_lock_breakeven

# Position sizing constants and plan building
from risk.position_sizing import (
    ASSET_CLASS, MAX_SECTOR_EXPOSURE, RISK_PER_TRADE, SECTOR_MAP,
    build_position_plan, fractional_kelly, parse_wfo_log, print_plan,
)

# Discord notifications
DISCORD_AVAILABLE = False
try:
    from tools.discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    pass

# FTMO Compliance
FTMO_AVAILABLE = False
try:
    from tools.ftmo_compliance import FTMOCompliance
    FTMO_AVAILABLE = True
except ImportError:
    pass

# Sentiment engine
SENTIMENT_AVAILABLE = False
try:
    from tools.sentiment_engine import get_sentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    pass


class SovereignBot:
    """H1 multi-symbol trading bot — monitors all symbols on candle close"""

    def __init__(self, account_id: str | None = None):
        self._account_id = account_id

        # Initialize config
        load_config()

        # Override DB path BEFORE BlackoutLogger so each account uses its own audit DB
        if account_id:
            import yaml as _yaml
            _accts_path = Path(__file__).resolve().parent.parent / "config" / "accounts.yaml"
            if _accts_path.exists():
                with open(_accts_path) as _f:
                    _accts = _yaml.safe_load(_f).get("accounts", {})
                _audit_db = _accts.get(account_id, {}).get("audit_db")
                if _audit_db:
                    cfg.DB_PATH = str(REPO_ROOT / _audit_db)

        # Core components
        self.logger = BlackoutLogger()
        self.feature_logger = FeatureLogger()
        # Load BF session override if this is a BrightFunded account
        _bf_sessions = os.path.join(os.path.dirname(__file__), "..", "data",
                                     "instrument_specs", "bf_sessions.csv")
        _override = _bf_sessions if "bright" in (self._account_id or '') else None
        self.trading_schedule = TradingSchedule(override_csv=_override)
        self.position_sizer = PositionSizingEngine(self.logger, mt5)
        self.decay_tracker = ModelDecayTracker(self.logger)
        self.position_manager = PositionManager(self.logger, mt5)
        self.filters = {}
        self.running = False
        self.emergency_stop = False
        self.safe_mode = False
        self.last_signal_time = {}

        # F5: Portfolio optimizer
        self.portfolio_optimizer = None
        try:
            from risk.portfolio_optimizer import PortfolioOptimizer
            self.portfolio_optimizer = PortfolioOptimizer()
            self.logger.log('INFO', 'SovereignBot', 'PORTFOLIO_OPT_INIT',
                            'Portfolio optimizer initialized')
        except Exception as e:
            self.logger.log('DEBUG', 'SovereignBot', 'PORTFOLIO_OPT_SKIP', str(e))

        # F16: Trade journal
        self.trade_journal = None
        try:
            from research.trade_journal import TradeJournal
            self.trade_journal = TradeJournal()
            self.logger.log('INFO', 'SovereignBot', 'JOURNAL_INIT',
                            'Trade journal initialized')
        except Exception as e:
            self.logger.log('DEBUG', 'SovereignBot', 'JOURNAL_SKIP', str(e))

        # Discord
        self.discord = None
        if DISCORD_AVAILABLE:
            try:
                config_path = os.path.join(str(REPO_ROOT), "config", "discord_config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        discord_cfg = json.load(f)
                    if discord_cfg.get('enabled', False):
                        self.discord = DiscordNotifier(discord_cfg.get('webhook_url'))
                        self.logger.log('INFO', 'SovereignBot', 'DISCORD_INIT',
                                        'Discord connected')
            except Exception as e:
                print(f"Discord config load failed: {e}")

        self.position_manager.discord = self.discord
        self.position_manager._trading_schedule = self.trading_schedule
        self.position_manager.start_monitor(interval=0.5)  # Continuous trailing-stop monitoring

        # Paper tracker monitors will be started after mt5 is confirmed ready (in run())

        # Order router (primary — used for backwards compat and single-account mode)
        self.order_router = OrderRouter(
            self.logger, mt5, self.position_sizer, self.trading_schedule,
            discord=self.discord,
        )
        # F5: Pass portfolio optimizer to order router
        self.order_router.portfolio_optimizer = self.portfolio_optimizer

        # Heartbeat (primary — monitors default MT5 connection for market data)
        self.heartbeat = HeartbeatMonitor(
            self.logger, mt5,
            on_disconnect=self._enter_safe_mode,
            discord=self.discord,
        )

        # Multi-timeframe scanner (full load deferred to run() for account filtering)
        self.multi_tf = MultiTFScanner(self)

        # Scan cache — preloads tick data + lead-lag before bar close
        # scan_cache removed — MultiTFScanner handles preloading internally

        # FTMO (primary — set during run())
        self.ftmo = None

        # Multi-account support: load all account contexts
        self.accounts: dict[str, AccountContext] = load_accounts(discord=self.discord)
        if self._account_id:
            if self._account_id not in self.accounts:
                raise ValueError(f"Account '{self._account_id}' not found in accounts.yaml")
            self.accounts = {self._account_id: self.accounts[self._account_id]}

        # Paper trackers — shadow trading for strategy testing
        # Paper trading runs as separate process (paper_bot.py)

    def _enter_safe_mode(self, reason: str):
        if self.safe_mode:
            return
        self.safe_mode = True
        self.order_router.safe_mode = True
        self.logger.log('CRITICAL', 'SovereignBot', 'SAFE_MODE',
                        f'Entering SAFE MODE: {reason}')
        if self.discord:
            self.discord.send("SAFE MODE",
                              f"Sovereign bot entered SAFE MODE ({reason}).",
                              "red")

    def _account_safe_mode(self, account_id: str, reason: str):
        """Safe mode callback for individual accounts."""
        acct = self.accounts.get(account_id)
        if acct:
            acct.safe_mode = True
            acct.order_router.safe_mode = True
            self.logger.log('CRITICAL', 'SovereignBot', 'ACCOUNT_SAFE_MODE',
                            f'[{acct.name}] Entering SAFE MODE: {reason}')
            if self.discord:
                self.discord.send(f"[{acct.name}] SAFE MODE",
                                  f"Account entered SAFE MODE ({reason}).",
                                  "red")

    def _merge_account_symbols(self):
        """Add account-specific symbols to cfg.SYMBOLS for scanning.

        Symbols from per-account configs that are not already in cfg.SYMBOLS
        get added (using their internal/model name) so the scanner picks them up.
        Existing symbols are NOT overwritten — the primary config takes precedence.
        For existing symbols, broker_symbol is updated if the account has one
        (each broker may use different symbol names, e.g. DOGEUSD vs XDG/USD).
        """
        added = []
        for acct in self.accounts.values():
            if not acct.enabled or not acct.symbols:
                continue
            for broker_sym, sym_cfg in acct.symbols.items():
                internal_sym = acct.get_internal_symbol(broker_sym)
                if internal_sym not in cfg.SYMBOLS:
                    cfg.SYMBOLS[internal_sym] = sym_cfg
                    added.append(f"{internal_sym} (from {acct.name})")
                elif broker_sym != internal_sym:
                    # Symbol exists but this broker uses a different name —
                    # update broker_symbol so the scanner fetches the right bars
                    cfg.SYMBOLS[internal_sym]["broker_symbol"] = broker_sym
        if added:
            self.logger.log('INFO', 'SovereignBot', 'SYMBOLS_MERGED',
                            f'Added {len(added)} account-specific symbols: {", ".join(added)}')

    def init_filters(self):
        for symbol in cfg.SYMBOLS:
            self.filters[symbol] = SovereignMLFilter(symbol, self.logger)

    def train_models(self, force=False):
        _ensure_ml_imports()
        cfg.load_optuna_params()

        trained = 0
        loaded = 0
        skipped = 0

        for symbol, filt in self.filters.items():
            if not force and filt.load_model():
                loaded += 1
                continue
            if filt.train_model():
                trained += 1
            else:
                skipped += 1

        self.logger.log('INFO', 'SovereignBot', 'MODELS_READY',
                        f'Trained: {trained}, Loaded: {loaded}, Skipped: {skipped}')

    def _llm_scan_commentary(self, scan_results: list, found: int, executed: int):
        """Ask LLM to explain the scan results in plain Dutch."""
        if not scan_results:
            return
        try:
            import requests
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            lines = [f"H1 Scan {now_str} | {len(scan_results)} symbols | {found} signals | {executed} executed"]
            lines.append(f"ML threshold: {cfg.ML_THRESHOLD:.2f} (per-symbol overrides active)")
            include_z = not cfg.DISABLE_ZSCORE
            if include_z:
                lines.append("Symbol      | Side | Proba | z20    | RSI14 | Status")
            else:
                lines.append("Symbol      | Side | Proba | RSI14 | Status")
            lines.append("-" * 60)
            for r in sorted(scan_results, key=lambda x: -x["proba"]):
                if include_z:
                    lines.append(
                        f"{r['symbol']:<11} | {r['side']:+d}   | {r['proba']:.3f} | "
                        f"{r['z20']:+.2f}  | {r['rsi14']:.0f}   | {r['status']}: {r['reason']}"
                    )
                else:
                    lines.append(
                        f"{r['symbol']:<11} | {r['side']:+d}   | {r['proba']:.3f} | "
                        f"{r['rsi14']:.0f}   | {r['status']}: {r['reason']}"
                    )

            if MT5_AVAILABLE:
                positions = mt5.positions_get()
                our_pos = [p for p in (positions or []) if p.magic >= 2000]
                if our_pos:
                    lines.append(f"\nOpen posities ({len(our_pos)}):")
                    for p in our_pos:
                        d = "BUY" if p.type == 0 else "SELL"
                        lines.append(f"  {p.symbol} {d} {p.volume} lots | PnL=${p.profit+p.swap:+.2f}")

            table = "\n".join(lines)

            system = (
                "Je bent een trading-analist voor een algoritmische FTMO prop trading bot. "
                "De bot handelt op H1 candles met XGBoost ML modellen. "
                "Leg in 3-5 zinnen in het Nederlands uit wat er deze scan gebeurde: "
                "waarom er wel/niet gehandeld is, welke symbolen het dichtst bij een signaal zaten, "
                "en of er iets opvalt (RSI divergentie, volatiliteit). "
                "BELANGRIJK: Als er signalen zijn geblokkeerd (ALREADY_IN_MARKET, H4_MISALIGN, "
                "SPREAD_TOO_WIDE, DD_GATE, PROFIT_LOCK, CORR_BLOCK, etc.), benoem deze EXPLICIET "
                "per symbool met de reden waarom ze geblokkeerd zijn. "
                "Gebruik altijd de ML-threshold die in de tabel staat. "
                "Noem z-scores alleen als ze in de tabel staan. "
                "Wees bondig en direct. Geen disclaimers."
            )

            resp = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": table,
                    "system": system,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                commentary = resp.json().get("response", "").strip()
                if commentary:
                    self.logger.log('INFO', 'SovereignBot', 'SCAN_COMMENTARY', commentary[:500])
                    if self.discord and found > 0:
                        status_emoji = f"{found} signals, {executed} trades"
                        color = "green" if executed > 0 else "blue"
                        self.discord.send(
                            f"SCAN {now_str} | {status_emoji}",
                            commentary[:1900],
                            color,
                        )
        except Exception as e:
            self.logger.log('DEBUG', 'SovereignBot', 'SCAN_COMMENTARY_ERROR', str(e))

    def _llm_mtf_commentary(self, tf: str, scan_results: list, found: int, executed: int):
        """Ask LLM to explain multi-TF scan results, post to Discord."""
        if not scan_results:
            return
        try:
            import requests
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            lines = [f"{tf} Scan {now_str} | {len(scan_results)} symbols | {found} signals | {executed} executed"]
            lines.append(f"ML threshold: {cfg.ML_THRESHOLD:.2f} (per-symbol overrides active)")
            lines.append("Symbol      | Side | Proba | RSI14 | Status")
            lines.append("-" * 55)
            for r in sorted(scan_results, key=lambda x: -x["proba"]):
                lines.append(
                    f"{r['symbol']:<11} | {r['side']:+d}   | {r['proba']:.3f} | "
                    f"{r['rsi14']:.0f}   | {r['status']}: {r['reason']}"
                )

            if MT5_AVAILABLE:
                positions = mt5.positions_get()
                our_pos = [p for p in (positions or []) if p.magic >= 2000]
                if our_pos:
                    lines.append(f"\nOpen posities ({len(our_pos)}):")
                    for p in our_pos:
                        d = "BUY" if p.type == 0 else "SELL"
                        lines.append(f"  {p.symbol} {d} {p.volume} lots | PnL=${p.profit+p.swap:+.2f}")

            table = "\n".join(lines)

            system = (
                f"Je bent een trading-analist voor een algoritmische FTMO prop trading bot. "
                f"De bot handelt op {tf} candles met XGBoost ML modellen. "
                f"Dit zijn US/EU equity posities (TSLA, NVDA, AMZN, META, AAPL, LVMH). "
                f"Leg in 3-5 zinnen in het Nederlands uit wat er deze scan gebeurde: "
                f"waarom er wel/niet gehandeld is, welke symbolen het dichtst bij een signaal zaten, "
                f"en of er iets opvalt (RSI, volatiliteit). "
                f"BELANGRIJK: Als er signalen zijn geblokkeerd (ALREADY_IN_MARKET, H4_MISALIGN, "
                f"SPREAD_TOO_WIDE, DD_GATE, PROFIT_LOCK, CORR_BLOCK, etc.), benoem deze EXPLICIET "
                f"per symbool met de reden waarom ze geblokkeerd zijn. "
                f"Wees bondig en direct. Geen disclaimers."
            )

            resp = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": table,
                    "system": system,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                commentary = resp.json().get("response", "").strip()
                if commentary:
                    self.logger.log('INFO', 'MultiTF', 'SCAN_COMMENTARY', commentary[:500])
                    if self.discord and found > 0:
                        status_emoji = f"{found} signals, {executed} trades"
                        color = "green" if executed > 0 else "blue"
                        self.discord.send(
                            f"{tf} SCAN {now_str} | {status_emoji}",
                            commentary[:1900],
                            color,
                        )
        except Exception as e:
            self.logger.log('DEBUG', 'MultiTF', 'SCAN_COMMENTARY_ERROR', str(e))

    @staticmethod
    def seconds_until_next_h1() -> float:
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        wait = (next_hour - now).total_seconds() + 5
        return max(wait, 1.0)

    def _check_closed_positions(self):
        if not MT5_AVAILABLE:
            return
        try:
            current_positions = mt5.positions_get()
            current_tickets = set()
            if current_positions:
                for p in current_positions:
                    if p.magic >= 2000:
                        current_tickets.add(p.ticket)

            if not hasattr(self, '_tracked_tickets'):
                self._tracked_tickets = {}
                if current_positions:
                    for p in current_positions:
                        if p.magic >= 2000:
                            self._tracked_tickets[p.ticket] = {
                                'symbol': p.symbol,
                                'direction': 'BUY' if p.type == 0 else 'SELL',
                                'confidence': 0.0,
                            }
                return

            closed_tickets = set(self._tracked_tickets.keys()) - current_tickets
            for ticket in closed_tickets:
                info = self._tracked_tickets.pop(ticket)
                close_info = self._get_deal_close_info(ticket)

                # Fallback: if deal history unavailable, estimate from last tick
                if close_info is None:
                    try:
                        _sym = info.get('broker_symbol', info['symbol'])
                        _tick = mt5.symbol_info_tick(_sym)
                        if _tick:
                            _exit_p = _tick.bid if info['direction'] == 'BUY' else _tick.ask
                            _si = mt5.symbol_info(_sym)
                            _cs = _si.trade_contract_size if _si else 1.0
                            _lots = info.get('lot_size', 0)
                            if info['direction'] == 'BUY':
                                _est_pnl = (_exit_p - info.get('entry_price', _exit_p)) * _lots * _cs
                            else:
                                _est_pnl = (info.get('entry_price', _exit_p) - _exit_p) * _lots * _cs
                            close_info = {
                                'pnl': round(_est_pnl, 2),
                                'exit_price': _exit_p,
                                'exit_time': datetime.now(timezone.utc).isoformat(),
                            }
                            self.logger.log('WARN', 'SovereignBot', 'CLOSE_NO_DEALS',
                                            f'{info["symbol"]} ticket={ticket}: no deal history, '
                                            f'estimated exit={_exit_p} pnl=${_est_pnl:+.2f}')
                    except Exception as _fb_err:
                        self.logger.log('ERROR', 'SovereignBot', 'CLOSE_FALLBACK_ERROR',
                                        f'{info["symbol"]} ticket={ticket}: {_fb_err}')

                pnl = close_info['pnl'] if close_info else None
                if pnl is not None:
                    self.decay_tracker.record_trade(
                        info['symbol'], pnl, info['direction'], info['confidence']
                    )
                    self.logger.log('INFO', 'SovereignBot', 'TRADE_CLOSED',
                                    f'{info["symbol"]} {info["direction"]} '
                                    f'ticket={ticket} PnL={pnl:+.2f}')

                    # Update trades table with exit info
                    self.logger.close_trade(
                        ticket=ticket,
                        exit_price=close_info.get('exit_price', 0),
                        pnl=pnl,
                        exit_timestamp=close_info.get('exit_time', ''),
                    )

                    # F3: Update RL sizer with trade result
                    rl_arm = info.get('rl_arm')
                    if rl_arm is not None:
                        risk_taken = info.get('risk_pct', 0.003)
                        acct = mt5.account_info()
                        live_equity = acct.equity if acct else cfg.ACCOUNT_SIZE
                        reward = pnl / max(risk_taken * live_equity, 1.0)
                        self.position_sizer.rl_update(
                            rl_arm,
                            {
                                'ml_confidence': info.get('confidence', 0.55),
                                'regime': info.get('regime', 0),
                                'volatility': info.get('volatility', 0.0),
                                'drawdown_pct': 0.0,
                            },
                            reward,
                        )

                    # F16: Journal the closed trade
                    if self.trade_journal:
                        try:
                            self.trade_journal.journal_trade({
                                'symbol': info['symbol'],
                                'direction': info['direction'],
                                'pnl': pnl,
                                'hold_hours': info.get('hold_hours', 0),
                                'ml_confidence': info.get('confidence', 0),
                                'entry_price': info.get('entry_price', 0),
                                'exit_price': info.get('exit_price', 0),
                                'features': info.get('features', {}),
                            })
                        except Exception as e:
                            self.logger.log('DEBUG', 'SovereignBot', 'JOURNAL_ERROR', str(e))

                    # F5: Update portfolio optimizer returns
                    if self.portfolio_optimizer:
                        try:
                            acct2 = mt5.account_info()
                            daily_ret = pnl / max(acct2.equity if acct2 else cfg.ACCOUNT_SIZE, 1.0)
                            self.portfolio_optimizer.update_returns(info['symbol'], daily_ret)
                        except Exception:
                            pass

                    # Discord P&L is now reported by position_manager/emergency_kill
                    # at close time with _deal_pnl(). Only send here for SL/TP hits
                    # (broker-closed positions not handled by our code).
                    if self.discord:
                        # Check if this was closed by our code (has a Sovereign_ comment)
                        is_managed_close = False
                        try:
                            now_utc = datetime.now(timezone.utc)
                            deals = mt5.history_deals_get(
                                now_utc - timedelta(days=7), now_utc, position=ticket)
                            if deals:
                                for d in deals:
                                    if d.position_id == ticket and hasattr(d, 'comment'):
                                        if 'Sovereign_' in str(d.comment):
                                            is_managed_close = True
                                            break
                        except Exception:
                            pass

                        if not is_managed_close:
                            color = "green" if pnl > 0 else "red"
                            self.discord.send(
                                f"TRADE CLOSED: {info['symbol']}",
                                f"Direction: {info['direction']}\n"
                                f"Ticket: {ticket}\nP&L: ${pnl:+.2f}",
                                color,
                            )

            if current_positions:
                for p in current_positions:
                    if p.magic >= 2000 and p.ticket not in self._tracked_tickets:
                        ctx = getattr(self, '_last_trade_context', {}).get(p.symbol, {})
                        self._tracked_tickets[p.ticket] = {
                            'symbol': p.symbol,
                            'direction': 'BUY' if p.type == 0 else 'SELL',
                            'confidence': ctx.get('confidence', 0.0),
                            'rl_arm': ctx.get('rl_arm'),
                            'risk_pct': ctx.get('risk_pct', 0.003),
                            'regime': ctx.get('regime', 0),
                            'volatility': ctx.get('volatility', 0.0),
                        }
        except Exception as e:
            self.logger.log('ERROR', 'SovereignBot', 'CLOSED_CHECK_ERROR', str(e))

    def _reconcile_stale_positions(self):
        """Reconcile DB positions marked FILLED that are no longer open on MT5.

        On bot restart, positions may have been closed broker-side (SL/TP hit,
        session close) while the bot was down.  This method fetches deal history
        and updates the audit DB so FILLED ↔ MT5 stays in sync.
        """
        if not MT5_AVAILABLE:
            return
        try:
            import sqlite3 as _sql
            from collections import defaultdict

            # 1. All FILLED tickets in audit DB
            conn = _sql.connect(self.logger.db_path)
            conn.row_factory = _sql.Row
            filled = conn.execute(
                "SELECT ticket, symbol FROM trades "
                "WHERE status = 'FILLED' AND ticket IS NOT NULL"
            ).fetchall()
            if not filled:
                conn.close()
                return

            # 2. Currently live tickets on MT5
            live_positions = mt5.positions_get()
            live_tickets = set()
            if live_positions:
                for p in live_positions:
                    if p.magic >= 2000:
                        live_tickets.add(p.ticket)

            stale = [r for r in filled if r['ticket'] not in live_tickets]
            if not stale:
                conn.close()
                return

            self.logger.log('INFO', 'SovereignBot', 'RECONCILE_START',
                            f'{len(stale)} stale FILLED positions to reconcile')

            # 3. Fetch all deal history (one call, filter client-side)
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=30)
            all_deals = mt5.history_deals_get(start, now)
            by_pos = defaultdict(list)
            if all_deals:
                for d in all_deals:
                    pid = getattr(d, 'position_id', 0)
                    if pid > 0:
                        by_pos[pid].append(d)

            # 4. Update each stale position
            reconciled = 0
            no_history = 0
            for row in stale:
                ticket = row['ticket']
                pos_deals = by_pos.get(ticket, [])
                exit_deals = [d for d in pos_deals
                              if getattr(d, 'entry', -1) == 1]
                entry_deals = [d for d in pos_deals
                               if getattr(d, 'entry', -1) == 0]

                if exit_deals:
                    pnl = sum(d.profit + d.commission + d.swap
                              for d in exit_deals)
                    pnl += sum(d.commission + d.swap for d in entry_deals)
                    exit_price = exit_deals[-1].price
                    exit_time = datetime.fromtimestamp(
                        exit_deals[-1].time, tz=timezone.utc
                    ).isoformat()
                    conn.execute(
                        "UPDATE trades SET exit_price=?, exit_timestamp=?, "
                        "pnl=?, status='CLOSED' WHERE ticket=?",
                        (exit_price, exit_time, pnl, ticket),
                    )
                    reconciled += 1
                    self.logger.log('INFO', 'SovereignBot', 'RECONCILE_CLOSED',
                                    f'{row["symbol"]} ticket={ticket} '
                                    f'PnL=${pnl:+.2f}')
                else:
                    # No deal history — try last tick price as fallback
                    _est_exit = 0
                    _est_pnl = 0
                    try:
                        _sym = row['symbol']
                        _tick = mt5.symbol_info_tick(_sym)
                        if _tick:
                            # Need direction from DB
                            _row_full = conn.execute(
                                "SELECT direction, entry_price, lot_size FROM trades WHERE ticket=?",
                                (ticket,)
                            ).fetchone()
                            if _row_full:
                                _dir = _row_full['direction']
                                _entry = _row_full['entry_price']
                                _lots = _row_full['lot_size'] or 0
                                _est_exit = _tick.bid if _dir == 'BUY' else _tick.ask
                                _si = mt5.symbol_info(_sym)
                                _cs = _si.trade_contract_size if _si else 1.0
                                if _dir == 'BUY':
                                    _est_pnl = (_est_exit - _entry) * _lots * _cs
                                else:
                                    _est_pnl = (_entry - _est_exit) * _lots * _cs
                                _est_pnl = round(_est_pnl, 2)
                    except Exception:
                        pass

                    conn.execute(
                        "UPDATE trades SET status='CLOSED_NO_HISTORY', "
                        "exit_price=?, pnl=?, exit_timestamp=? WHERE ticket=?",
                        (_est_exit, _est_pnl,
                         datetime.now(timezone.utc).isoformat(), ticket),
                    )
                    no_history += 1

            conn.commit()
            conn.close()

            msg = (f'Reconciled {reconciled} positions from deal history, '
                   f'{no_history} without history')
            self.logger.log('INFO', 'SovereignBot', 'RECONCILE_DONE', msg)
            if self.discord and reconciled > 0:
                self.discord.send("AUDIT RECONCILIATION", msg, "blue")

        except Exception as e:
            self.logger.log('ERROR', 'SovereignBot', 'RECONCILE_ERROR', str(e))

    def _refresh_sentiment(self):
        if not SENTIMENT_AVAILABLE:
            return
        try:
            sentiment = {}
            for symbol in cfg.SYMBOLS:
                score = get_sentiment(symbol)
                if score != 0.0:
                    sentiment[symbol] = score
            for broad in ("_BROAD_CRYPTO", "_BROAD_FOREX", "_BROAD_RISK_OFF"):
                score = get_sentiment(broad)
                if score != 0.0:
                    sentiment[broad] = score
            self.order_router._cached_sentiment = sentiment
        except Exception as e:
            self.logger.log('WARNING', 'SovereignBot', 'SENTIMENT_REFRESH_ERROR', str(e))

    def _get_deal_pnl(self, ticket: int) -> float | None:
        info = self._get_deal_close_info(ticket)
        return info['pnl'] if info else None

    def _get_deal_close_info(self, ticket: int, _retries: int = 3) -> dict | None:
        """Get PnL, exit price, and exit time from MT5 deal history."""
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=7)
            deals = mt5.history_deals_get(start, now, position=ticket)
            if deals is None or len(deals) == 0:
                return None
            pos_deals = [d for d in deals if d.position_id == ticket]
            if not pos_deals:
                return None

            # Only count exit deals (entry=1) for P&L — entry deals have
            # profit=0 and their commission is already baked into the
            # exit deal's total cost on most brokers.
            exit_deals = [d for d in pos_deals if hasattr(d, 'entry') and d.entry == 1]
            if not exit_deals:
                # Exit deal not yet in history (race condition) — retry
                if _retries > 0:
                    import time
                    time.sleep(1)
                    return self._get_deal_close_info(ticket, _retries - 1)
                return None

            # Sum profit + commission + swap across all exit deals (split fills)
            total_pnl = sum(d.profit + d.commission + d.swap for d in exit_deals)
            # Add entry commission separately (not included in exit deal on some brokers)
            entry_deals = [d for d in pos_deals if hasattr(d, 'entry') and d.entry == 0]
            total_pnl += sum(d.commission + d.swap for d in entry_deals)

            exit_price = exit_deals[0].price
            exit_time = datetime.fromtimestamp(exit_deals[0].time, tz=timezone.utc).isoformat()

            return {'pnl': total_pnl, 'exit_price': exit_price, 'exit_time': exit_time}
        except Exception:
            return None

    def _send_daily_summary(self):
        if not MT5_AVAILABLE or not self.discord:
            return
        try:
            account = mt5.account_info()
            if not account:
                return

            today = datetime.now().date()
            if hasattr(self, '_last_summary_date') and self._last_summary_date == today:
                return
            self._last_summary_date = today

            equity = account.equity
            balance = account.balance
            initial = self.heartbeat.initial_balance or cfg.ACCOUNT_SIZE

            profit_target = initial * 0.10
            current_profit = equity - initial
            progress_pct = (current_profit / profit_target * 100) if profit_target > 0 else 0
            daily_loss_limit = initial * 0.05
            total_dd_limit = initial * 0.10

            daily_start = self.heartbeat.daily_start_balance or balance
            daily_pnl = equity - daily_start
            daily_pnl_pct = daily_pnl / daily_start * 100 if daily_start > 0 else 0

            positions = mt5.positions_get()
            open_count = sum(1 for p in positions if p.magic >= 2000) if positions else 0
            open_pnl = sum(p.profit for p in positions if p.magic >= 2000) if positions else 0

            conn = sqlite3.connect(self.logger.db_path, timeout=30)
            try:
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()
                trades_today = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE timestamp > ? AND status = 'FILLED'",
                    (yesterday,)
                ).fetchone()[0]
            finally:
                conn.close()

            bar_len = 20
            filled = int(max(0, min(progress_pct, 100)) / 100 * bar_len)
            progress_bar = "█" * filled + "░" * (bar_len - filled)

            body = (
                f"**Balance:** ${balance:,.2f}\n"
                f"**Equity:** ${equity:,.2f}\n"
                f"**Daily P&L:** ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)\n"
                f"**Open positions:** {open_count} (unrealized: ${open_pnl:+,.2f})\n"
                f"**Trades today:** {trades_today}\n\n"
                f"**FTMO Target Progress:**\n"
                f"`{progress_bar}` {progress_pct:.1f}%\n"
                f"Profit: ${current_profit:+,.2f} / ${profit_target:,.2f}\n\n"
                f"**Safety:**\n"
                f"Daily loss margin: ${daily_loss_limit + daily_pnl:,.2f} remaining\n"
                f"Total DD margin: ${total_dd_limit + current_profit:,.2f} remaining"
            )
            self.discord.send("DAILY SUMMARY", body, "blue")
        except Exception as e:
            self.logger.log('ERROR', 'SovereignBot', 'DAILY_SUMMARY_ERROR', str(e))

    def show_plan(self, equity: float):
        results = parse_wfo_log(cfg.WFO_LOG)
        plans = build_position_plan(
            results, account=equity, max_dd_pct=0.05,
            kelly_fraction=0.10, min_trades=100, min_pf=1.02,
        )
        print_plan(plans, equity, 0.05)

    def execute_trade(self, symbol, direction, ml_confidence, features_dict=None):
        """Delegate to all active accounts (multi-account) + primary order router.

        Returns True if at least one account executed the trade.
        """
        any_executed = False

        # Route to all initialized account contexts
        active_accounts = [a for a in self.accounts.values()
                           if a.enabled and a.order_router is not None]

        if active_accounts:
            for acct in active_accounts:
                try:
                    result = acct.execute_trade(symbol, direction, ml_confidence, features_dict)
                    if result:
                        any_executed = True
                        acct.logger.log('INFO', 'AccountContext', 'TRADE_EXECUTED',
                                        f'[{acct.name}] {symbol} {direction} conf={ml_confidence:.3f}')
                except Exception as e:
                    self.logger.log('ERROR', 'SovereignBot', 'ACCOUNT_TRADE_ERROR',
                                    f'[{acct.name}] {symbol}: {e}')
        else:
            # Fallback: use primary order router (backwards compat / single-account)
            result = self.order_router.execute_trade(
                symbol, direction, ml_confidence,
                gpu_trading_pause=HeartbeatMonitor.GPU_TRADING_PAUSE,
                features_dict=features_dict,
            )
            any_executed = result

        # Store RL context for trade closure update (F3)
        if any_executed and features_dict:
            if not hasattr(self, '_last_trade_context'):
                self._last_trade_context = {}
            self._last_trade_context[symbol] = {
                'rl_arm': getattr(self.order_router, '_last_rl_arm', None),
                'risk_pct': getattr(self.order_router, '_last_risk_pct', 0.003),
                'regime': int(features_dict.get('regime', 0)),
                'volatility': float(features_dict.get('vol20', 0.0)),
                'confidence': ml_confidence,
            }
        return any_executed

    def execute_trade_batch(self, signals: list[dict]) -> list[dict]:
        """Execute multiple signals with confidence-weighted margin allocation.

        Each signal dict: {symbol, direction, confidence, features_dict}
        Returns list of {symbol, direction, confidence, success} dicts.

        Margin is split proportionally by confidence with 10% reserve.
        Highest confidence signals execute first.
        """
        n = len(signals)
        if n == 0:
            return []

        results = []
        active_accounts = [a for a in self.accounts.values()
                           if a.enabled and a.order_router is not None]

        if not active_accounts:
            # Fallback: single account via primary order router (no batch budgeting)
            for sig in signals:
                success = self.execute_trade(
                    sig['symbol'], sig['direction'], sig['confidence'],
                    features_dict=sig.get('features_dict'),
                )
                results.append({
                    'symbol': sig['symbol'], 'direction': sig['direction'],
                    'confidence': sig['confidence'], 'success': success,
                })
            return results

        # Per-account: query margin once, calculate budgets, execute in order
        for acct in active_accounts:
            try:
                account_info = acct.mt5.account_info()
                if not account_info:
                    acct.logger.log('WARNING', 'SovereignBot', 'BATCH_NO_ACCOUNT_INFO',
                                    f'[{acct.name}] Cannot get account info for batch allocation')
                    continue

                original_free = account_info.margin_free
                total_budget = original_free * 0.90  # 10% reserve

                # Confidence-weighted allocation
                total_conf = sum(s['confidence'] for s in signals)
                budgets = []
                for s in signals:
                    weight = s['confidence'] / total_conf if total_conf > 0 else 1.0 / n
                    budget = total_budget * weight
                    # Floor at 10% of budget, cap at 50% (prevents single signal hogging)
                    budget = max(total_budget * 0.10, min(total_budget * 0.50, budget))
                    budgets.append(budget)

                reserve = original_free - total_budget
                acct.logger.log('INFO', 'SovereignBot', 'BATCH_MARGIN',
                                f'[{acct.name}] BATCH_MARGIN: {n} signals, '
                                f'budget ${total_budget:,.0f} (reserve ${reserve:,.0f})')

                # Execute in confidence order (signals already sorted by caller)
                for sig, budget in zip(signals, budgets):
                    acct.logger.log('INFO', 'SovereignBot', 'MARGIN_BUDGET',
                                    f'[{acct.name}] MARGIN_BUDGET: {sig["symbol"]} '
                                    f'max_margin=${budget:,.0f} '
                                    f'(conf={sig["confidence"]:.3f})')
                    try:
                        success = acct.execute_trade(
                            sig['symbol'], sig['direction'], sig['confidence'],
                            features_dict=sig.get('features_dict'),
                            margin_budget=budget,
                        )
                    except Exception as e:
                        self.logger.log('ERROR', 'SovereignBot', 'BATCH_TRADE_ERROR',
                                        f'[{acct.name}] {sig["symbol"]}: {e}')
                        success = False

                    if success:
                        acct.logger.log('INFO', 'AccountContext', 'TRADE_EXECUTED',
                                        f'[{acct.name}] {sig["symbol"]} {sig["direction"]} '
                                        f'conf={sig["confidence"]:.3f}')

                    # Track result (keyed by symbol — last account wins for multi-account)
                    # Find existing result or create new
                    existing = next((r for r in results if r['symbol'] == sig['symbol']), None)
                    if existing:
                        existing['success'] = existing['success'] or success
                    else:
                        results.append({
                            'symbol': sig['symbol'], 'direction': sig['direction'],
                            'confidence': sig['confidence'], 'success': success,
                        })

                    # Store RL context for successful trades
                    if success and sig.get('features_dict'):
                        if not hasattr(self, '_last_trade_context'):
                            self._last_trade_context = {}
                        self._last_trade_context[sig['symbol']] = {
                            'rl_arm': getattr(acct.order_router, '_last_rl_arm', None),
                            'risk_pct': getattr(acct.order_router, '_last_risk_pct', 0.003),
                            'regime': int(sig['features_dict'].get('regime', 0)),
                            'volatility': float(sig['features_dict'].get('vol20', 0.0)),
                            'confidence': sig['confidence'],
                        }

            except Exception as e:
                self.logger.log('ERROR', 'SovereignBot', 'BATCH_ACCOUNT_ERROR',
                                f'[{acct.name}] Batch execution failed: {e}')

        return results

    def run(self, mode: str = 'dry-run', scan_once: bool = False):
        print("=" * 70)
        print("  SOVEREIGN BOT — H1 Multi-Symbol Production System")
        if self._account_id:
            print(f"  Account: {self._account_id}")
        print("=" * 70)

        # Override cfg paths for single-account mode
        if self._account_id:
            acct = self.accounts[self._account_id]
            acfg = acct.account_cfg
            if acfg.get("config_path"):
                cfg.CONFIG_PATH = str(REPO_ROOT / acfg["config_path"])
            if acfg.get("model_dir"):
                cfg.MODEL_DIR = str(REPO_ROOT / acfg["model_dir"])
            if acfg.get("optuna_csv"):
                cfg.OPTUNA_CSV = str(REPO_ROOT / acfg["optuna_csv"])
            if acfg.get("audit_db"):
                cfg.DB_PATH = str(REPO_ROOT / acfg["audit_db"])

        cfg.load()
        if not cfg.SYMBOLS:
            print("[ERROR] No symbol configs loaded. Run with --build-plan first.")
            return

        # Merge symbols from per-account configs (adds BF-only symbols like US100.cash)
        self._merge_account_symbols()

        # Load multi-TF scanner (filtered to account symbols in single-account mode)
        mtf_allowed = set(cfg.SYMBOLS.keys()) if self._account_id else None
        mtf_count = self.multi_tf.load_config(allowed_symbols=mtf_allowed)
        if mtf_count > 0:
            self.logger.log('INFO', 'SovereignBot', 'MULTI_TF_INIT',
                            f'{mtf_count} symbols loaded across all timeframes')

        self.init_filters()
        print(f"\n[1] Initialized filters for {len(self.filters)} symbols")

        # Start MT5 — use account-specific terminal path if running single-account
        _mt5_terminal = None
        if self._account_id and self._account_id in self.accounts:
            _mt5_terminal = self.accounts[self._account_id].terminal_path
        print(f"  [DEBUG] MT5 module: {mt5.__name__}, terminal_path: {_mt5_terminal}")
        from tools.mt5_bridge import initialize_mt5 as _init_mt5
        _ok, _err, _mode = _init_mt5(terminal_path=_mt5_terminal)
        print(f"  [DEBUG] MT5 init result: ok={_ok}, mode={_mode}, error={_err}")
        mt5_ok = _ok
        mt5_mode = _mode
        if mt5_ok:
            self.logger.log('INFO', 'BrokerAPI', 'MT5_INIT_MODE',
                            f'MT5 initialized via {mt5_mode}')
            account_info = mt5.account_info()
            if account_info:
                self.logger.log('INFO', 'BrokerAPI', 'MT5_INITIALIZED',
                                f'Account {account_info.login} | '
                                f'Balance ${account_info.balance:,.2f} | '
                                f'Equity ${account_info.equity:,.2f}')

        # Initialize FTMO compliance — single source of truth for all FTMO rules
        if mt5_ok and FTMO_AVAILABLE:
            account_info = mt5.account_info()
            if account_info:
                self.ftmo = FTMOCompliance(
                    initial_balance=account_info.balance,
                    logger=self.logger,
                    discord=self.discord,
                )
                self.order_router.ftmo = self.ftmo
                self.position_manager._ftmo = self.ftmo
                # Load last trade time for inactivity tracking
                self.ftmo.load_last_trade_time(self.logger.db_path)

        if mt5_ok:
            self.heartbeat.start()
            # Reconcile stale FILLED positions against MT5 on startup
            self._reconcile_stale_positions()

        # Initialize multi-account contexts
        self._active_accounts: list[AccountContext] = []
        for acct_id, acct in self.accounts.items():
            if not acct.enabled:
                self.logger.log('INFO', 'SovereignBot', 'ACCOUNT_DISABLED',
                                f'[{acct.name}] Skipped (disabled)')
                continue
            try:
                ok = acct.initialize(self.trading_schedule,
                                     on_safe_mode=self._account_safe_mode)
                if ok:
                    self._active_accounts.append(acct)
                    self.logger.log('INFO', 'SovereignBot', 'ACCOUNT_INIT_OK',
                                    f'[{acct.name}] Initialized on port {acct.bridge_port}')
                else:
                    self.logger.log('ERROR', 'SovereignBot', 'ACCOUNT_INIT_FAIL',
                                    f'[{acct.name}] Failed to initialize')
            except Exception as e:
                self.logger.log('ERROR', 'SovereignBot', 'ACCOUNT_INIT_ERROR',
                                f'[{acct.name}] {e}')

        if self._active_accounts:
            names = [a.name for a in self._active_accounts]
            self.logger.log('INFO', 'SovereignBot', 'MULTI_ACCOUNT',
                            f'{len(self._active_accounts)} accounts active: {names}')
            if self.discord:
                self.discord.send("MULTI-ACCOUNT",
                                  f"{len(self._active_accounts)} accounts active:\n" +
                                  "\n".join(f"  - {a.name} (port {a.bridge_port})" for a in self._active_accounts),
                                  "blue")

        self.running = True
        self.order_router.safe_mode = self.safe_mode

        # Get equity
        equity = cfg.ACCOUNT_SIZE
        if mt5_ok:
            try:
                account_info = mt5.account_info()
                if account_info:
                    equity = account_info.equity
            except Exception:
                pass

        # Show plan
        print(f"\n[2] Position sizing plan (equity: ${equity:,.0f})")
        self.show_plan(equity)

        # Sector summary
        sectors_used = {}
        for sym, sym_cfg in cfg.SYMBOLS.items():
            s = sym_cfg.get('sector', 'unknown')
            sectors_used[s] = sectors_used.get(s, 0) + sym_cfg.get('risk_per_trade', 0)
        print(f"\n  Sector allocation:")
        for sec in sorted(sectors_used, key=lambda x: -sectors_used[x]):
            limit = MAX_SECTOR_EXPOSURE.get(sec, 0.02)
            print(f"    {sec:>12s}: {sectors_used[sec]:.2%} / {limit:.2%}")

        if mode == 'dry-run':
            print(f"\n  Mode: DRY RUN (no live trading)")
            print(f"  To go live: enable algo trading in MT5, then run with --live")
            self._stop()
            return

        # Train/load models
        print(f"\n[3] Loading ML models...")
        self.train_models()

        models_ready = sum(1 for f in self.multi_tf.filters.values() if f.model is not None)
        print(f"    Models ready: {models_ready} / {len(self.multi_tf.symbols)}")

        self.decay_tracker.load_baselines_from_config()
        print(f"    Decay tracker baselines: {len(self.decay_tracker.baselines)}")

        if models_ready == 0:
            print("[ERROR] No models available. Run with --train first.")
            self._stop()
            return

        if not mt5_ok and not self._active_accounts:
            print("[ERROR] MT5 not connected. Cannot run live mode.")
            self._stop()
            return

        # Live trading loop — all TFs handled by MultiTFScanner
        print(f"\n[4] LIVE MODE — multi-TF parallel scanner active")
        print(f"    Active symbols: {models_ready}")
        per_sym_thrs = {s: c.get("prob_threshold", cfg.ML_THRESHOLD)
                        for s, c in cfg.SYMBOLS.items() if "prob_threshold" in c}
        if per_sym_thrs:
            print(f"    ML threshold:   {cfg.ML_THRESHOLD} (default) | per-symbol: {per_sym_thrs}")
        else:
            print(f"    ML threshold:   {cfg.ML_THRESHOLD}")
        print(f"    Press Ctrl+C to stop\n")

        if self.discord:
            # Build startup summary grouped by TF
            lines = []
            tf_groups = {}
            for sym, sym_cfg in (self.multi_tf.symbols or {}).items():
                tf = sym_cfg.get("timeframe", "?")
                tf_groups.setdefault(tf, []).append(sym)
            for tf, syms in sorted(tf_groups.items()):
                loaded = [s for s in syms if s in self.multi_tf.filters and self.multi_tf.filters[s].model is not None]
                if loaded:
                    lines.append(f"{len(loaded)} symbols on {tf}: {', '.join(loaded)}")
            self.discord.send("SOVEREIGN BOT STARTED", "\n".join(lines) or "No symbols loaded", "blue")

        # Paper trading runs as separate process (paper_bot.py)

        try:
            if scan_once:
                self.logger.log('INFO', 'SovereignBot', 'FORCE_SCAN',
                                'Force-scanning all symbols across all TFs...')
                force_signals = self.multi_tf.force_scan(mt5)
                found = len(force_signals)
                self.logger.log('INFO', 'SovereignBot', 'FORCE_SCAN_RESULT',
                                f'Signals: {found}')

                if found == 0 and self.discord:
                    all_syms = list(self.multi_tf.symbols.keys())
                    any_open = any(self.trading_schedule.is_trading_open(s)[0] for s in all_syms)
                    if any_open:
                        self.discord.send("SCAN FAILED — NO DATA",
                                          f"Force scan got 0 signals for {len(all_syms)} symbols.\n"
                                          "MT5 may be down.", "red")

                return

            last_slow_check = 0
            while self.running and not self.emergency_stop:
                # Friday close checks at loop top
                friday_progressive_close(self.logger, mt5, self.trading_schedule,
                                         self.running, self.emergency_stop, self.discord)
                friday_auto_close(self.logger, mt5, self.trading_schedule,
                                  self.running, self.emergency_stop, self.discord)

                # Multi-TF scanner — returns signals, no execution
                try:
                    signals = self.multi_tf.tick(mt5)
                    if signals:
                        # Lazy snapshots — only when signals exist
                        from execution.master_order_generator import (
                            MasterOrderGenerator, AccountSnapshot, Signal)

                        if not hasattr(self, '_master_gen'):
                            self._master_gen = MasterOrderGenerator(self.logger)

                        active = [a for a in self.accounts.values()
                                  if a.enabled and a.order_router is not None]
                        snapshots = {}
                        for a in active:
                            try:
                                snapshots[a.account_id] = AccountSnapshot.from_account(a)
                            except Exception as e:
                                self.logger.log('ERROR', 'MasterGen', 'SNAPSHOT_ERROR',
                                                f'[{a.name}] {e}')

                        # Convert scanner _Signal to MasterGen Signal
                        mg_signals = [
                            Signal(
                                symbol=s.symbol, direction=s.direction,
                                confidence=s.confidence,
                                features_dict=s.features_dict,
                                timeframe=s.tf, sent_boost=s.sent_boost,
                            ) for s in signals
                        ]

                        order_specs = self._master_gen.process_signals(mg_signals, snapshots)

                        executed = 0
                        for acct_id, specs in order_specs.items():
                            if self.emergency_stop:
                                self.logger.log('WARNING', 'MasterGen', 'EMERGENCY_STOP',
                                                'Aborting spec execution — emergency stop')
                                break
                            acct = self.accounts.get(acct_id)
                            if not acct or not acct.order_router:
                                continue
                            for spec in specs:
                                if self.emergency_stop:
                                    break
                                if spec.dry_run:
                                    self.logger.log('INFO', 'MasterGen', 'DRY_RUN',
                                                    f'[{acct_id}] {spec.symbol} {spec.direction} '
                                                    f'conf={spec.ml_confidence:.3f}')
                                else:
                                    try:
                                        ok = acct.order_router.execute_order_spec(spec)
                                        if ok:
                                            executed += 1
                                            # Store RL context (F3) for trade closure feedback
                                            if spec.features_dict:
                                                if not hasattr(self, '_last_trade_context'):
                                                    self._last_trade_context = {}
                                                self._last_trade_context[spec.symbol] = {
                                                    'rl_arm': getattr(acct.order_router, '_last_rl_arm', None),
                                                    'risk_pct': getattr(acct.order_router, '_last_risk_pct', 0.003),
                                                    'regime': int(spec.features_dict.get('regime', 0)),
                                                    'volatility': float(spec.features_dict.get('vol20', 0.0)),
                                                    'confidence': spec.ml_confidence,
                                                }
                                    except Exception as e:
                                        self.logger.log('ERROR', 'MasterGen', 'EXEC_ERROR',
                                                        f'[{acct_id}] {spec.symbol}: {e}')

                        self.logger.log('INFO', 'SovereignBot', 'SCAN_RESULT',
                                        f'Signals: {len(signals)}, Executed: {executed}')
                except Exception as e:
                    self.logger.log('ERROR', 'MultiTF', 'TICK_ERROR', str(e))

                # Slow checks every 60s: FTMO, DD, bleeders, exits
                now_ts = time.time()
                if now_ts - last_slow_check >= 60:
                    # FTMO: Total DD hard stop check (primary account)
                    if self.ftmo:
                        acct_info = mt5.account_info()
                        if acct_info:
                            must_close, dd_reason = self.ftmo.check_total_dd(acct_info.equity)
                            if must_close:
                                self.logger.log('CRITICAL', 'SovereignBot', 'FTMO_HARD_STOP',
                                                dd_reason)
                                emergency_close_all(self.logger, mt5, self.discord)
                                self._enter_safe_mode(dd_reason)

                    # FTMO: Inactivity check (primary account)
                    if self.ftmo:
                        self.ftmo.check_inactivity()

                    # Multi-account: DD + floating profit checks for all accounts
                    for ma in self._active_accounts:
                        try:
                            if ma.check_total_dd():
                                self.logger.log('CRITICAL', 'SovereignBot', 'ACCOUNT_DD_STOP',
                                                f'[{ma.name}] Total DD hard stop triggered')
                                emergency_close_all(ma.logger, ma.mt5, self.discord)
                                self._account_safe_mode(ma.account_id, 'total DD hard stop')
                            if ma.ftmo:
                                ma.ftmo.check_inactivity()
                        except Exception as e:
                            self.logger.log('ERROR', 'SovereignBot', 'ACCT_DD_CHECK_ERROR',
                                            f'[{ma.name}] {e}')

                    friday_progressive_close(self.logger, mt5, self.trading_schedule,
                                             self.running, self.emergency_stop, self.discord)
                    friday_auto_close(self.logger, mt5, self.trading_schedule,
                                      self.running, self.emergency_stop, self.discord)
                    self.position_manager.auto_close_bleeders(self.running, self.emergency_stop)
                    self.position_manager.session_close_check(
                        self.trading_schedule, self.running, self.emergency_stop)
                    self.position_manager.horizon_exit_check(self.running, self.emergency_stop)
                    # ML exit: check if model still supports open positions
                    self.position_manager.ml_exit_check(
                        self.multi_tf.filters, None, self.multi_tf.symbols)
                    self._check_closed_positions()
                    self._refresh_sentiment()
                    self._send_daily_summary()
                    # Log trailing-stop monitor stats
                    stats = self.position_manager._monitor_stats
                    if stats["cycles"] > 0:
                        self.logger.log('DEBUG', 'TrailingStopMonitor', 'STATS',
                                        f'cycles={stats["cycles"]} '
                                        f'sl_moves={stats["sl_moves"]} '
                                        f'last_cycle={stats["last_cycle_ms"]:.1f}ms')
                    last_slow_check = now_ts

                # Sleep ~5s between ticks (or less if a bar is closing soon)
                wait = min(5.0, self.multi_tf.seconds_until_next_bar())
                time.sleep(max(wait, 1.0))


        except KeyboardInterrupt:
            self.logger.log('INFO', 'SovereignBot', 'INTERRUPTED',
                            'Bot interrupted by user')
        finally:
            self._stop()
            print("\nSovereign Bot stopped. Check logging/sovereign_log.db for full audit trail.")

    def _stop(self):
        self.logger.log('INFO', 'SovereignBot', 'STOP', 'Stopping bot')
        self.running = False
        self.position_manager.stop_monitor()
        self.heartbeat.stop()
        # Stop all account contexts
        for acct in getattr(self, '_active_accounts', []):
            try:
                acct.stop()
            except Exception as e:
                self.logger.log('ERROR', 'SovereignBot', 'ACCOUNT_STOP_ERROR',
                                f'[{acct.name}] {e}')
        shutdown()


def build_and_save_configs(wfo_log: str, account: float, max_dd: float):
    """Build position plans and save sovereign_configs.json."""
    results = parse_wfo_log(wfo_log)
    print(f"Parsed {len(results)} symbols from WFO log")

    plans = build_position_plan(
        results, account=account, max_dd_pct=max_dd,
        kelly_fraction=0.10, min_trades=100, min_pf=1.02,
    )
    active_plans = [p for p in plans if p.risk_per_trade > 0]
    print(f"Active symbols: {len(active_plans)}")

    configs = {}
    magic_base = 2000
    for i, p in enumerate(plans):
        if p.risk_per_trade <= 0:
            continue
        sector = SECTOR_MAP.get(p.symbol, "unknown")
        ac = ASSET_CLASS.get(sector, "forex")

        atr_configs = {
            "crypto": {"period": 14, "sl_mult": 2.0, "tp_mult": 6.0},
            "forex": {"period": 14, "sl_mult": 1.5, "tp_mult": 4.5},
            "commodity": {"period": 14, "sl_mult": 1.5, "tp_mult": 4.5},
            "index": {"period": 14, "sl_mult": 1.5, "tp_mult": 4.5},
            "equity": {"period": 14, "sl_mult": 1.2, "tp_mult": 3.6},
        }
        spread_limits = {
            "crypto": 0.005, "forex": 0.0005, "commodity": 0.001,
            "index": 0.0005, "equity": 0.001,
        }
        atr_cfg = atr_configs.get(ac, atr_configs["forex"])
        spread_limit = spread_limits.get(ac, 0.001)

        configs[p.symbol] = {
            "sector": sector,
            "asset_class": ac,
            "risk_per_trade": p.risk_per_trade,
            "kelly_fraction": p.kelly_fraction,
            "profit_factor": p.profit_factor,
            "max_dd": p.max_dd,
            "magic_number": magic_base + i,
            "max_spread_pct": spread_limit,
            "atr_period": atr_cfg["period"],
            "atr_sl_mult": atr_cfg["sl_mult"],
            "atr_tp_mult": atr_cfg["tp_mult"],
            "atr_timeframe": "H1",
        }

    config_path = Path(cfg.CONFIG_PATH)
    config_path.write_text(json.dumps(configs, indent=2))
    print(f"Saved {len(configs)} configs to {config_path}")

    print_plan(plans, account, max_dd)
    return configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sovereign Bot — H1 Multi-Symbol Trading System")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only, don't trade")
    parser.add_argument("--live", action="store_true", help="Live H1 trading loop")
    parser.add_argument("--train", action="store_true", help="Train XGBoost models from tick data")
    parser.add_argument("--build-plan", action="store_true",
                        help="Build position plan and save configs from WFO log")
    parser.add_argument("--show-plan", action="store_true", help="Show position sizing plan")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain top-N models with latest data (Sunday cron job)")
    parser.add_argument("--retrain-top", type=int, default=15)
    parser.add_argument("--wfo-log", default=None)
    parser.add_argument("--account", type=float, default=100_000)
    parser.add_argument("--max-dd", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--scan-now", action="store_true")
    parser.add_argument("--disable-zscore", action="store_true")
    parser.add_argument("--account-id", type=str, default=None,
                        help="Run single account (e.g. ftmo_100k, bright_100k)")
    args = parser.parse_args()

    # Load config first
    load_config()

    cfg.ML_THRESHOLD = args.threshold
    cfg.DISABLE_ZSCORE = bool(args.disable_zscore)
    if args.wfo_log:
        cfg.WFO_LOG = args.wfo_log
    cfg.ACCOUNT_SIZE = args.account

    if args.build_plan:
        build_and_save_configs(cfg.WFO_LOG, args.account, args.max_dd)
        sys.exit(0)

    if args.train:
        cfg.load()
        bot = SovereignBot(account_id=args.account_id)
        # Apply account-specific path overrides (model_dir, config_path, etc.)
        if bot._account_id and bot._account_id in bot.accounts:
            acfg = bot.accounts[bot._account_id].account_cfg
            if acfg.get("config_path"):
                cfg.CONFIG_PATH = str(REPO_ROOT / acfg["config_path"])
            if acfg.get("model_dir"):
                cfg.MODEL_DIR = str(REPO_ROOT / acfg["model_dir"])
            cfg.load()
        bot._merge_account_symbols()
        bot.init_filters()
        bot.train_models(force=True)
        print("\nModel training complete.")
        sys.exit(0)

    if args.scan_now:
        bot = SovereignBot(account_id=args.account_id)
        bot.run(mode='live', scan_once=True)
        sys.exit(0)

    if args.retrain:
        cfg.load()
        bot = SovereignBot(account_id=args.account_id)
        bot._merge_account_symbols()
        bot.init_filters()

        ranked = sorted(
            cfg.SYMBOLS.items(),
            key=lambda x: x[1].get("profit_factor", 0),
            reverse=True,
        )
        top_symbols = [s for s, _ in ranked[:args.retrain_top]]
        print(f"\nRetraining top {args.retrain_top} symbols: {top_symbols}")

        _ensure_ml_imports()
        cfg.load_optuna_params()

        retrained = 0
        for sym in top_symbols:
            filt = bot.filters.get(sym)
            if filt is None:
                continue
            print(f"\n  Retraining {sym}...")
            if filt.train_model():
                retrained += 1
                print(f"    OK")
            else:
                print(f"    SKIPPED")

        print(f"\nRetrained {retrained} / {len(top_symbols)} symbols.")
        if bot.discord:
            bot.discord.send(
                "SUNDAY RETRAIN COMPLETE",
                f"Retrained {retrained}/{len(top_symbols)} top symbols",
                "blue"
            )
        sys.exit(0)

    if args.show_plan:
        cfg.load()
        bot = SovereignBot(account_id=args.account_id)
        bot.show_plan(args.account)
        sys.exit(0)

    if args.live:
        bot = SovereignBot(account_id=args.account_id)
        bot.run(mode='live')
        sys.exit(0)

    # Default: dry-run
    bot = SovereignBot(account_id=args.account_id)
    bot.run(mode='dry-run')
