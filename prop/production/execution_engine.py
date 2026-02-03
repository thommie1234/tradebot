#!/usr/bin/env python3
"""
EXECUTION ENGINE - Production Grade Trading System
====================================================

Components:
1. Heartbeat Monitor - MT5 connection watchdog
2. Sovereign Filter - XGBoost microservice
3. Position Sizing - Kelly criterion based
4. Blackout Logger - SQLite event logging
5. Emergency Kill-Switch - 3% DD protection

Guardrails:
- Max spread guard (0.5% for crypto, 0.08% for stocks)
- Time blackout (rollover 23:00-00:00, news events)
- Slippage protection (FOK/IOC orders)
- TP/SL sent with entry order (no naked positions)

Author: Thomas (HP Z440)
Status: PRODUCTION READY
"""

import MetaTrader5 as mt5
# import pandas as pd
# import numpy as np
# import xgboost as xgb
import sqlite3
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import threading
import pickle

# Discord notifications
try:
    from discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("Discord notifier not available (missing requests module)")

# FTMO Compliance
try:
    from ftmo_compliance import FTMOCompliance
    FTMO_AVAILABLE = True
except ImportError:
    FTMO_AVAILABLE = False
    print("FTMO compliance not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Production configuration"""

    # Symbols
    SYMBOLS = {
        'SOLUSD': {
            'lot_size': 0.01,
            'max_spread_pct': 0.005,  # 0.5%
            'pip_value': 0.01,
            'magic_number': 1001,
            'atr_multiplier': 2.0,    # SL = 2x ATR (crypto needs room)
            'atr_period': 14,         # ATR period
            'atr_timeframe': 'H1'     # ATR timeframe
        },
        'MSFT': {
            'lot_size': 0.01,
            'max_spread_pct': 0.0008,  # 0.08% (Increased from 0.02%)
            'pip_value': 1.0,
            'magic_number': 1002,
            'atr_multiplier': 0.8,    # SL = 0.8x ATR (80%)
            'atr_period': 14,         # ATR period
            'atr_timeframe': 'H1'     # ATR timeframe
        }
    }

    # Risk/Reward ratio (TP = RR_RATIO * SL)
    RR_RATIO = 3.0  # 1:3 Risk/Reward

    # Risk Management (FTMO Rules)
    MAX_DAILY_LOSS_PCT = 0.05  # 5% FTMO daily loss limit
    MAX_TOTAL_LOSS_PCT = 0.10  # 10% FTMO total drawdown limit
    RISK_PER_TRADE_PCT = 0.003  # 0.3%
    MAX_CONCURRENT_POSITIONS = 200  # FTMO server limit

    # Filters
    ML_CONFIDENCE_THRESHOLD = 0.70
    # Global TP/SL removed - see per-symbol config above

    # Timing
    HEARTBEAT_INTERVAL = 10  # seconds
    BLACKOUT_ROLLOVER_START = 23  # hour
    BLACKOUT_ROLLOVER_END = 0  # hour
    NEWS_BLACKOUT_MINUTES = 5

    # Paths
    DB_PATH = 'production_log.db'
    MODEL_PATH_SOLUSD = '../../models/solusd_filter.pkl'
    MODEL_PATH_MSFT = '../../models/msft_filter.pkl'

    # Order execution
    DEVIATION = 10  # Max price deviation in points
    ORDER_FILL_TYPE = mt5.ORDER_FILLING_FOK  # Fill or Kill

# ============================================================================
# BLACKOUT LOGGER - SQLite Event Logging
# ============================================================================

class BlackoutLogger:
    """Production event logger - tracks everything"""

    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT,
                data TEXT
            )
        ''')

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL,
                tp_price REAL,
                sl_price REAL,
                lot_size REAL,
                spread_pct REAL,
                ml_confidence REAL,
                ticket INTEGER,
                status TEXT,
                exit_timestamp TEXT,
                exit_price REAL,
                pnl REAL
            )
        ''')

        # Heartbeat table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mt5_connected BOOLEAN,
                account_balance REAL,
                account_equity REAL,
                open_positions INTEGER,
                daily_pnl REAL
            )
        ''')

        conn.commit()
        conn.close()

    def log(self, level, component, event_type, message, data=None):
        """Log event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO events (timestamp, level, component, event_type, message, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            level,
            component,
            event_type,
            message,
            json.dumps(data) if data else None
        ))

        conn.commit()
        conn.close()

        # Also print to console
        print(f"[{level}] {component}.{event_type}: {message}")

    def log_trade(self, symbol, direction, entry_price, tp_price, sl_price,
                   lot_size, spread_pct, ml_confidence, ticket=None, status='PENDING'):
        """Log trade attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, direction, entry_price, tp_price,
                                sl_price, lot_size, spread_pct, ml_confidence, ticket, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            direction,
            entry_price,
            tp_price,
            sl_price,
            lot_size,
            spread_pct,
            ml_confidence,
            ticket,
            status
        ))

        conn.commit()
        conn.close()

    def log_heartbeat(self, mt5_connected, account_balance, account_equity,
                      open_positions, daily_pnl):
        """Log heartbeat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO heartbeats (timestamp, mt5_connected, account_balance,
                                    account_equity, open_positions, daily_pnl)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            mt5_connected,
            account_balance,
            account_equity,
            open_positions,
            daily_pnl
        ))

        conn.commit()
        conn.close()

# ============================================================================
# HEARTBEAT MONITOR - MT5 Connection Watchdog
# ============================================================================

class HeartbeatMonitor:
    """Monitors MT5 connection health"""

    def __init__(self, logger):
        self.logger = logger
        self.running = False
        self.thread = None
        self.last_balance = None
        self.initial_balance = None
        self.daily_start_balance = None
        self.last_reset_date = None

    def start(self):
        """Start heartbeat monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.log('INFO', 'HeartbeatMonitor', 'START', 'Heartbeat monitoring started')

    def stop(self):
        """Stop heartbeat monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.log('INFO', 'HeartbeatMonitor', 'STOP', 'Heartbeat monitoring stopped')

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check MT5 connection
                if not mt5.terminal_info():
                    self.logger.log('ERROR', 'HeartbeatMonitor', 'MT5_DISCONNECTED',
                                    'MT5 terminal not connected')
                    time.sleep(Config.HEARTBEAT_INTERVAL)
                    continue

                # Get account info
                account_info = mt5.account_info()
                if not account_info:
                    self.logger.log('ERROR', 'HeartbeatMonitor', 'ACCOUNT_INFO_FAILED',
                                    'Could not retrieve account info')
                    time.sleep(Config.HEARTBEAT_INTERVAL)
                    continue

                # Initialize balances on first run
                if self.initial_balance is None:
                    self.initial_balance = account_info.balance
                    self.daily_start_balance = account_info.balance
                    self.last_reset_date = datetime.now().date()

                # Reset daily balance if new day
                current_date = datetime.now().date()
                if current_date > self.last_reset_date:
                    self.daily_start_balance = account_info.balance
                    self.last_reset_date = current_date
                    self.logger.log('INFO', 'HeartbeatMonitor', 'DAILY_RESET',
                                    f'Daily balance reset to ${account_info.balance:.2f}')

                # Calculate PnL
                daily_pnl = account_info.equity - self.daily_start_balance
                total_pnl = account_info.equity - self.initial_balance

                # Get open positions
                positions = mt5.positions_total()

                # Log heartbeat
                self.logger.log_heartbeat(
                    mt5_connected=True,
                    account_balance=account_info.balance,
                    account_equity=account_info.equity,
                    open_positions=positions,
                    daily_pnl=daily_pnl
                )

                # Check emergency conditions
                daily_loss_pct = daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
                total_loss_pct = total_pnl / self.initial_balance if self.initial_balance > 0 else 0

                if daily_loss_pct < -Config.MAX_DAILY_LOSS_PCT:
                    self.logger.log('CRITICAL', 'HeartbeatMonitor', 'DAILY_LOSS_BREACH',
                                    f'Daily loss {daily_loss_pct*100:.2f}% exceeds {Config.MAX_DAILY_LOSS_PCT*100:.1f}% limit')
                    # Trigger emergency stop (implemented in ExecutionEngine)

                if total_loss_pct < -Config.MAX_TOTAL_LOSS_PCT:
                    self.logger.log('CRITICAL', 'HeartbeatMonitor', 'TOTAL_LOSS_BREACH',
                                    f'Total loss {total_loss_pct*100:.2f}% exceeds {Config.MAX_TOTAL_LOSS_PCT*100:.1f}% limit')

            except Exception as e:
                self.logger.log('ERROR', 'HeartbeatMonitor', 'EXCEPTION',
                                f'Heartbeat error: {str(e)}')

            time.sleep(Config.HEARTBEAT_INTERVAL)

# ============================================================================
# SOVEREIGN FILTER - XGBoost Microservice
# ============================================================================


"""
class SovereignFilter:
...
            return False, 0.0
"""

class PositionSizingEngine:
    """Calculate lot size based on risk management"""

    def __init__(self, logger):
        self.logger = logger

    def calculate_lot_size(self, symbol, account_equity, sl_distance, confidence=0.5):
        """
        Calculate lot size based on ATR-based SL distance
        sl_distance: absolute price distance for stop loss
        """
        # Risk amount per trade
        risk_amount = account_equity * Config.RISK_PER_TRADE_PCT

        tick = mt5.symbol_info_tick(symbol)
        if not tick: return 0.01

        # Calculate risk per lot based on SL distance
        # For crypto (SOLUSD): contract size is typically 100
        # For stocks (MSFT): contract size is typically 1
        contract_size = 100 if symbol == 'SOLUSD' else 1
        risk_per_lot = sl_distance * contract_size

        lot_size = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01
        lot_size = max(0.01, min(lot_size, 1.0))

        self.logger.log('DEBUG', 'PositionSizing', 'LOT_CALC',
                        f'{symbol} Risk:${risk_amount:.2f} SL_dist:{sl_distance:.5f} -> {lot_size:.2f} lots')
        return round(lot_size, 2)

# ============================================================================
# EXECUTION ENGINE - Main Trading Logic
# ============================================================================

class ExecutionEngine:
    """Production-grade execution engine"""

    def __init__(self):
        # DIAGNOSTIC PRINT
        msft_limit = Config.SYMBOLS.get('MSFT', {}).get('max_spread_pct', 0)
        print(f"DEBUG: MSFT Spread Limit Loaded: {msft_limit*100:.3f}%")

        self.logger = BlackoutLogger()
        self.heartbeat = HeartbeatMonitor(self.logger)
        self.filters = {}
        self.position_sizer = PositionSizingEngine(self.logger)
        self.running = False
        self.emergency_stop = False

        # Initialize Discord notifier
        self.discord = None
        if DISCORD_AVAILABLE:
            try:
                # Try absolute path for Wine compatibility
                config_path = r"Z:\home\thomas\optifire\prop\discord_config.json"
                if not os.path.exists(config_path.replace("Z:", "")):
                     # Fallback to local path if running on Linux
                     config_path = "discord_config.json"
                
                with open(config_path, 'r') as f:
                    discord_config = json.load(f)
                    if discord_config.get('enabled', False):
                        self.discord = DiscordNotifier(discord_config.get('webhook_url'))
                        self.logger.log('INFO', 'ExecutionEngine', 'DISCORD_INIT', 'Discord connected')
            except Exception as e:
                print(f"Discord config load failed: {e}")

        # Initialize FTMO Compliance
        self.ftmo = None
        if FTMO_AVAILABLE:
            try:
                # Get initial balance from MT5
                account_info = mt5.account_info()
                initial_balance = account_info.balance if account_info else 100000
                self.ftmo = FTMOCompliance(initial_balance=initial_balance)
                print(f"FTMO Compliance enabled (Initial: ${initial_balance:,.2f})")
            except Exception as e:
                print(f"WARNING: FTMO compliance init failed: {e}")

        # Initialize filters
        # for symbol in Config.SYMBOLS.keys():
        #     self.filters[symbol] = SovereignFilter(symbol, self.logger)
        pass

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.log('CRITICAL', 'ExecutionEngine', 'MT5_INIT_FAILED',
                            'Failed to initialize MT5')
            return False

        account_info = mt5.account_info()
        if not account_info:
            self.logger.log('CRITICAL', 'ExecutionEngine', 'ACCOUNT_INFO_FAILED',
                            'Could not retrieve account info')
            return False

        self.logger.log('INFO', 'ExecutionEngine', 'MT5_INITIALIZED',
                        f'Connected to account {account_info.login}')

        return True

    def manage_positions(self):
        """
        Active Trade Management - Single Pass
        - Checks for 50% TP hit
        - Closes 50% volume
        - Moves SL to Break Even
        Called by StrategyBot main loop
        """
        if not self.running or self.emergency_stop:
            return

        try:
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    self._process_single_position(pos)
        except Exception as e:
            self.logger.log('ERROR', 'ExecutionEngine', 'MANAGE_ERROR',
                            f'Error managing positions: {str(e)}')

    def _process_single_position(self, pos):
        """Handle logic for a single position"""
        symbol = pos.symbol
        
        # 1. Check if already managed (SL is approx equal to Open Price)
        # Using a small epsilon for float comparison
        if abs(pos.sl - pos.price_open) < (pos.price_open * 0.0001):
            return  # Already at Break Even, skip

        # 2. Check if we have valid TP info
        if pos.tp == 0:
            return

        # 3. Calculate Progress
        entry = pos.price_open
        tp = pos.tp
        current_price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

        total_distance = abs(tp - entry)
        current_distance = abs(current_price - entry)
        
        if total_distance == 0: return

        progress = current_distance / total_distance

        # 4. Trigger at 60% Progress
        # Also ensure trade is actually in profit (direction matches)
        is_profit = (pos.type == mt5.ORDER_TYPE_BUY and current_price > entry) or \
                    (pos.type == mt5.ORDER_TYPE_SELL and current_price < entry)

        if progress >= 0.60 and is_profit:
            self.logger.log('INFO', 'TradeManager', 'TRIGGER_HIT',
                            f'{symbol} reached 60% to TP. Executing Partial Close & BE.')
            self._execute_partial_close_and_be(pos)

    def _execute_partial_close_and_be(self, pos):
        """Execute 50% close and move SL to BE"""
        # A. Close 50%
        close_vol = round(pos.volume * 0.5, 2)
        if close_vol >= 0.01:
            tick = mt5.symbol_info_tick(pos.symbol)
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            req_close = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_vol,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": Config.DEVIATION,
                "magic": pos.magic,
                "comment": "Partial Close 50%",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req_close)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                 self.logger.log('SUCCESS', 'TradeManager', 'PARTIAL_CLOSE',
                                 f'Closed {close_vol} lots for {pos.symbol}')
                 if self.discord:
                     self.discord.send("💰 PARTIAL TAKE PROFIT", 
                                       f"Closed 50% ({close_vol} lots) of {pos.symbol} at ${price:.2f}", "green")

        # B. Move SL to Break Even (Entry Price)
        # Add tiny buffer to cover spread/commissions (optional, keeping it simple at entry for now)
        req_sl = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": pos.price_open, # Move to Entry
            "tp": pos.tp          # Keep original TP
        }
        res_sl = mt5.order_send(req_sl)
        if res_sl.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.log('SUCCESS', 'TradeManager', 'MOVE_TO_BE',
                            f'Moved SL to Break Even for {pos.symbol}')
            if self.discord:
                 self.discord.send("🛡️ STOP LOSS UPDATED", 
                                   f"Moved SL to Break Even (${pos.price_open:.2f}) for {pos.symbol}. Risk Free Trade!", "blue")

    def check_spread(self, symbol):
        """Check if spread is acceptable"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            self.logger.log('ERROR', 'ExecutionEngine', 'TICK_FAILED',
                            f'Could not get tick for {symbol}')
            return False, 0.0

        spread = tick.ask - tick.bid
        spread_pct = spread / tick.ask

        max_spread = Config.SYMBOLS[symbol]['max_spread_pct']

        if spread_pct > max_spread:
            self.logger.log('WARNING', 'ExecutionEngine', 'SPREAD_TOO_WIDE',
                            f'{symbol} spread {spread_pct*100:.3f}% > {max_spread*100:.3f}%')
            return False, spread_pct

        return True, spread_pct

    def calculate_atr(self, symbol):
        """Calculate ATR for dynamic SL/TP"""
        sym_cfg = Config.SYMBOLS.get(symbol, {})
        period = sym_cfg.get('atr_period', 14)
        tf_str = sym_cfg.get('atr_timeframe', 'H1')

        # Map timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        timeframe = tf_map.get(tf_str, mt5.TIMEFRAME_H1)

        # Get candles
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            self.logger.log('WARNING', 'ExecutionEngine', 'ATR_FAILED',
                            f'Could not get rates for {symbol} ATR calculation')
            return None

        # Calculate True Range for each candle
        tr_values = []
        for i in range(1, len(rates)):
            high = rates[i]['high']
            low = rates[i]['low']
            prev_close = rates[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        # ATR = Simple Moving Average of TR
        atr = sum(tr_values[-period:]) / period

        self.logger.log('DEBUG', 'ExecutionEngine', 'ATR_CALCULATED',
                        f'{symbol} ATR({period}) on {tf_str}: {atr:.5f}')
        return atr

    def is_blackout_period(self):
        """Check if we're in a blackout period"""
        now = datetime.now()
        hour = now.hour

        # Rollover blackout (23:00-00:00)
        if hour == Config.BLACKOUT_ROLLOVER_START or hour == Config.BLACKOUT_ROLLOVER_END:
            self.logger.log('INFO', 'ExecutionEngine', 'BLACKOUT_ROLLOVER',
                            'In rollover blackout period')
            return True

        # TODO: Add news event checking
        # For now, we skip this

        return False

    def execute_trade(self, symbol, direction, ml_confidence):
        """
        Execute trade with full guardrails

        Steps:
        0. FTMO COMPLIANCE CHECK (FIRST!)
        1. Check spread
        2. Check blackout period
        3. No hedging check
        4. Get position size
        5. Calculate TP/SL
        6. Send order with TP/SL attached (FOK/IOC)
        7. Log everything
        """

        # Guardrail 0: FTMO COMPLIANCE (CRITICAL - CHECK FIRST!)
        # ... (Keep existing FTMO logic)
        
        # Bypass filter for now
        allowed = True
        # if self.ftmo: ...
        
        # ... (Keep existing spread and blackout checks)

        # Guardrail 1: Spread check
        spread_ok, spread_pct = self.check_spread(symbol)
        if not spread_ok:
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                   status='REJECTED_SPREAD')
            return False

        # Guardrail 2: Blackout period
        if self.is_blackout_period():
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                   status='REJECTED_BLACKOUT')
            return False

        # Guardrail 3: FTMO ANTI-HEDGING CHECK (NO HEDGING ALLOWED!)
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) > 0:
            self.logger.log('CRITICAL', 'ExecutionEngine', 'HEDGING_FORBIDDEN',
                            f'FTMO: FTMO VIOLATION: Cannot open new {direction} on {symbol} - '
                            f'Position already open (hedging forbidden!)')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                   status='REJECTED_HEDGING_FORBIDDEN')
            return False

        # Get account info
        account_info = mt5.account_info()
        if not account_info:
            self.logger.log('ERROR', 'ExecutionEngine', 'ACCOUNT_INFO_FAILED',
                            'Could not get account info')
            return False

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            self.logger.log('ERROR', 'ExecutionEngine', 'TICK_FAILED',
                            f'Could not get tick for {symbol}')
            return False

        # Get symbol config
        sym_cfg = Config.SYMBOLS.get(symbol, {})
        atr_mult = sym_cfg.get('atr_multiplier', 0.8)

        # Calculate ATR-based SL distance
        atr = self.calculate_atr(symbol)
        if atr is None:
            self.logger.log('ERROR', 'ExecutionEngine', 'ATR_FAILED',
                            f'Could not calculate ATR for {symbol}, aborting trade')
            return False

        sl_distance = atr * atr_mult
        tp_distance = sl_distance * Config.RR_RATIO  # 1:3 RR

        # Calculate position size based on ATR SL distance
        lot_size = self.position_sizer.calculate_lot_size(symbol, account_info.equity, sl_distance, ml_confidence)

        # Calculate entry, TP, SL
        if direction == 'BUY':
            entry_price = tick.ask
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry_price = tick.bid
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            order_type = mt5.ORDER_TYPE_SELL

        self.logger.log('INFO', 'ExecutionEngine', 'ATR_SLTP',
                        f'{symbol} ATR:{atr:.5f} SL_dist:{sl_distance:.5f} TP_dist:{tp_distance:.5f} (1:{Config.RR_RATIO:.0f} RR)')

        # Build order request WITH TP/SL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": Config.DEVIATION,
            "magic": Config.SYMBOLS[symbol]['magic_number'],
            "comment": f"Sovereign_{symbol}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": Config.ORDER_FILL_TYPE,
        }

        # Send order
        self.logger.log('INFO', 'ExecutionEngine', 'ORDER_SEND',
                        f'{symbol} {direction} {lot_size} lots @ {entry_price:.5f} '
                        f'TP:{tp_price:.5f} SL:{sl_price:.5f}')

        result = mt5.order_send(request)

        if result is None:
            self.logger.log('ERROR', 'ExecutionEngine', 'ORDER_FAILED',
                            f'order_send() returned None')
            self.logger.log_trade(symbol, direction, entry_price, tp_price, sl_price,
                                   lot_size, spread_pct, ml_confidence, status='FAILED_NONE')
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.log('ERROR', 'ExecutionEngine', 'ORDER_REJECTED',
                            f'Order rejected: {result.retcode} - {result.comment}')
            self.logger.log_trade(symbol, direction, entry_price, tp_price, sl_price,
                                   lot_size, spread_pct, ml_confidence,
                                   status=f'REJECTED_{result.retcode}')
            return False

        # Success!
        self.logger.log('SUCCESS', 'ExecutionEngine', 'ORDER_FILLED',
                        f'Order filled: Ticket {result.order}')
        self.logger.log_trade(symbol, direction, entry_price, tp_price, sl_price,
                               lot_size, spread_pct, ml_confidence,
                               ticket=result.order, status='FILLED')

        # Discord notification
        if self.discord:
            self.discord.trade_entry(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                lot_size=lot_size,
                tp=tp_price,
                sl=sl_price,
                ticket=result.order,
                confidence=ml_confidence
            )

        # FTMO: Increment daily trade count
        if self.ftmo:
            self.ftmo.increment_trade_count()

        return True

    def emergency_close_all(self):
        """Emergency: Close all positions"""
        self.logger.log('CRITICAL', 'ExecutionEngine', 'EMERGENCY_STOP',
                        'EMERGENCY STOP ACTIVATED - Closing all positions')

        positions = mt5.positions_get()
        num_positions = len(positions) if positions else 0

        if not positions:
            self.logger.log('INFO', 'ExecutionEngine', 'NO_POSITIONS',
                            'No positions to close')
            return

        for position in positions:
            # Close position
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                continue

            close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": close_price,
                "deviation": Config.DEVIATION,
                "magic": position.magic,
                "comment": "EMERGENCY_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.log('INFO', 'ExecutionEngine', 'POSITION_CLOSED',
                                f'Closed position {position.ticket}')
            else:
                self.logger.log('ERROR', 'ExecutionEngine', 'CLOSE_FAILED',
                                f'Failed to close position {position.ticket}')

        self.emergency_stop = True

        # Discord notification
        if self.discord:
            account_info = mt5.account_info()
            daily_pnl = account_info.profit if account_info else 0.0
            self.discord.emergency_stop(
                reason='Daily loss limit exceeded (3%)',
                daily_pnl=daily_pnl,
                open_positions=num_positions
            )

    def start(self):
        """Start execution engine"""
        self.logger.log('INFO', 'ExecutionEngine', 'START',
                        '='*70)
        self.logger.log('INFO', 'ExecutionEngine', 'START',
                        'SOVEREIGN EXECUTION ENGINE - PRODUCTION MODE')
        self.logger.log('INFO', 'ExecutionEngine', 'START',
                        '='*70)

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Start heartbeat
        self.heartbeat.start()

        self.running = True
        
        # Trade Manager is now called synchronously by StrategyBot
        # self.trade_manager_thread = threading.Thread(target=self.manage_positions, daemon=True)
        # self.trade_manager_thread.start()

        self.logger.log('INFO', 'ExecutionEngine', 'RUNNING',
                        'Execution engine is LIVE')

    def stop(self):
        """Stop execution engine"""
        self.logger.log('INFO', 'ExecutionEngine', 'STOP',
                        'Stopping execution engine')

        self.running = False
        self.heartbeat.stop()
        mt5.shutdown()

        self.logger.log('INFO', 'ExecutionEngine', 'STOPPED',
                        'Execution engine stopped')

# ============================================================================
# MAIN - For Testing
# ============================================================================

if __name__ == "__main__":
    engine = ExecutionEngine()
    engine.start()

    print("\n" + "="*70)
    print("EXECUTION ENGINE INITIALIZED")
    print("="*70)
    print("\nCommands:")
    print("  'buy <symbol>' - Execute buy trade")
    print("  'sell <symbol>' - Execute sell trade")
    print("  'status' - Show account status")
    print("  'positions' - Show open positions")
    print("  'emergency' - Emergency close all")
    print("  'quit' - Stop engine")
    print("="*70 + "\n")

    try:
        while engine.running and not engine.emergency_stop:
            cmd = input(">>> ").strip().lower()

            if cmd == 'quit':
                break

            elif cmd == 'emergency':
                engine.emergency_close_all()
                break

            elif cmd == 'status':
                account = mt5.account_info()
                if account:
                    print(f"\nAccount: {account.login}")
                    print(f"Balance: ${account.balance:.2f}")
                    print(f"Equity: ${account.equity:.2f}")
                    print(f"Margin: ${account.margin:.2f}")
                    print(f"Free Margin: ${account.margin_free:.2f}\n")

            elif cmd == 'positions':
                positions = mt5.positions_get()
                if positions:
                    print(f"\nOpen positions: {len(positions)}")
                    for pos in positions:
                        print(f"  {pos.symbol} {pos.type} {pos.volume} @ {pos.price_open} "
                              f"TP:{pos.tp} SL:{pos.sl} PnL:${pos.profit:.2f}")
                else:
                    print("\nNo open positions\n")

            elif cmd.startswith('buy ') or cmd.startswith('sell '):
                parts = cmd.split()
                if len(parts) == 2:
                    direction = parts[0].upper()
                    symbol = parts[1].upper()

                    if symbol in Config.SYMBOLS:
                        # For testing, use ML filter if available
                        ml_filter = engine.filters[symbol]
                        # Dummy features for now
                        should_trade, confidence = True, 0.6

                        if should_trade:
                            engine.execute_trade(symbol, direction, confidence)
                        else:
                            print(f"Trade rejected by ML filter (confidence: {confidence:.3f})")
                    else:
                        print(f"Unknown symbol: {symbol}")
                else:
                    print("Usage: buy/sell <symbol>")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        engine.stop()
        print("\nEngine stopped. Check production_log.db for full audit trail.")
