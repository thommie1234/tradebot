#!/usr/bin/env python3
"""
STRATEGY BOT - Mean Reversion with Sovereign Filter
=====================================================

Monitors market for mean reversion signals and executes via Execution Engine
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from execution_engine import ExecutionEngine, Config
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator

class StrategyBot:
    """Mean reversion strategy with ML filtering"""

    def __init__(self, symbol):
        self.symbol = symbol
        self.engine = ExecutionEngine()
        self.last_signal_time = {}  # Track last signal time to avoid duplicates
        self.min_bars_between_signals = 5  # Minimum bars between signals (25 min on M5)

    def get_historical_data(self, bars=200):
        """Fetch recent bars from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, bars)

        if rates is None or len(rates) == 0:
            self.engine.logger.log('ERROR', 'StrategyBot', 'DATA_FETCH_FAILED',
                                   f'Could not fetch data for {self.symbol}')
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df

    def calculate_features(self, df):
        """Calculate regime features for ML filter"""
        df = df.copy()

        # ATR
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']
        df['atr_ma'] = df['atr'].rolling(50).mean()
        df['atr_regime'] = df['atr'] / df['atr_ma']

        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Range
        df['range'] = (df['high'] - df['low']) / df['close']
        df['range_ma'] = df['range'].rolling(20).mean()
        df['range_spike'] = df['range'] / df['range_ma']

        # Time
        df['hour'] = df['time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Recent returns
        returns = df['close'].pct_change()
        df['recent_return'] = returns.rolling(10).sum()

        # Spread estimate
        df['spread_pct'] = 0.0005
        df['estimated_spread'] = df['spread_pct'] * df['atr_regime']
        df['edge_after_atr_spread'] = 0.003 - df['estimated_spread']

        return df

    def check_signal(self):
        """
        Check for mean reversion signal

        Simple Rule:
        - BUY after red candle (close < open)
        - SELL after green candle (close > open)
        """

        # Get recent data
        df = self.get_historical_data(bars=200)

        if df is None or len(df) < 100:
            return None, None

        # Calculate features
        df = self.calculate_features(df)

        if df is None or len(df) < 50:
            return None, None

        # Get last 2 candles
        prev_candle = df.iloc[-2]
        current_candle = df.iloc[-1]

        # Simple mean reversion signal
        signal = None

        if prev_candle['close'] < prev_candle['open']:
            # Previous candle was red → BUY signal
            signal = 'BUY'
        elif prev_candle['close'] > prev_candle['open']:
            # Previous candle was green → SELL signal
            signal = 'SELL'

        if signal is None:
            return None, None

        # Check if we recently sent a signal
        last_time = self.last_signal_time.get(self.symbol, 0)
        current_time = time.time()

        if current_time - last_time < self.min_bars_between_signals * 60 * 5:  # M5 bars
            return None, None

        # Get current features for ML filter
        features = current_candle[['atr_pct', 'atr_regime', 'adx', 'adx_pos', 'adx_neg',
                                     'range_spike', 'hour_sin', 'hour_cos', 'recent_return',
                                     'edge_after_atr_spread']].to_dict()

        self.engine.logger.log('DEBUG', 'StrategyBot', 'SIGNAL_GENERATED',
                               f'{self.symbol} {signal} signal detected')

        return signal, features

    def run_once(self):
        """Run one iteration of strategy check"""

        # Check for emergency stop
        if self.engine.emergency_stop:
            self.engine.logger.log('WARNING', 'StrategyBot', 'EMERGENCY_STOP',
                                   'Emergency stop active, skipping signals')
            return

        # Check if engine is running
        if not self.engine.running:
            return

        # --- TRADE MANAGEMENT (Check open positions) ---
        self.engine.manage_positions()
        # -----------------------------------------------

        # Check for signal
        signal, features = self.check_signal()

        if signal is None:
            return

        # Ask ML filter
        ml_filter = self.engine.filters.get(self.symbol)
        if ml_filter:
            should_trade, confidence = ml_filter.should_trade(features)

            if not should_trade:
                self.engine.logger.log('INFO', 'StrategyBot', 'SIGNAL_FILTERED',
                                       f'{self.symbol} {signal} rejected by ML filter '
                                       f'(confidence: {confidence:.3f})')
                return
        else:
            confidence = 0.5

        # Execute trade
        success = self.engine.execute_trade(self.symbol, signal, confidence)

        if success:
            # Update last signal time
            self.last_signal_time[self.symbol] = time.time()

            self.engine.logger.log('SUCCESS', 'StrategyBot', 'TRADE_EXECUTED',
                                   f'{self.symbol} {signal} trade executed')

    def run(self, check_interval=60):
        """
        Main strategy loop

        Args:
            check_interval: Seconds between strategy checks (default 60 = 1 minute)
        """

        self.engine.start()

        self.engine.logger.log('INFO', 'StrategyBot', 'STRATEGY_START',
                               f'Strategy bot started for {self.symbol}')
        self.engine.logger.log('INFO', 'StrategyBot', 'CHECK_INTERVAL',
                               f'Checking for signals every {check_interval} seconds')

        try:
            while self.engine.running and not self.engine.emergency_stop:
                self.run_once()
                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.engine.logger.log('INFO', 'StrategyBot', 'INTERRUPTED',
                                   'Strategy bot interrupted by user')

        finally:
            self.engine.stop()

# ============================================================================
# MULTI-SYMBOL BOT
# ============================================================================

class MultiSymbolBot:
    """Run strategy on multiple symbols"""

    def __init__(self, symbols):
        self.symbols = symbols
        self.engine = ExecutionEngine()
        self.bots = {}

        # Create individual bots
        for symbol in symbols:
            bot = StrategyBot(symbol)
            bot.engine = self.engine  # Share same engine
            self.bots[symbol] = bot

    def run(self, check_interval=60):
        """Run all strategies"""

        self.engine.start()

        self.engine.logger.log('INFO', 'MultiSymbolBot', 'START',
                               f'Multi-symbol bot started for {", ".join(self.symbols)}')

        try:
            while self.engine.running and not self.engine.emergency_stop:
                # --- TRADE MANAGEMENT (Check open positions) ---
                self.engine.manage_positions()
                # -----------------------------------------------

                # Run each bot
                for symbol, bot in self.bots.items():
                    try:
                        bot.run_once()
                    except Exception as e:
                        self.engine.logger.log('ERROR', 'MultiSymbolBot', 'BOT_ERROR',
                                               f'Error in {symbol} bot: {str(e)}')

                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.engine.logger.log('INFO', 'MultiSymbolBot', 'INTERRUPTED',
                                   'Multi-symbol bot interrupted by user')

        finally:
            self.engine.stop()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("="*70)
    print("SOVEREIGN STRATEGY BOT - PRODUCTION MODE")
    print("="*70)
    print("")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single symbol:  python strategy_bot.py SOLUSD")
        print("  Multi symbol:   python strategy_bot.py SOLUSD MSFT")
        print("")
        sys.exit(1)

    symbols = sys.argv[1:]

    # Validate symbols
    valid_symbols = []
    for symbol in symbols:
        if symbol in Config.SYMBOLS:
            valid_symbols.append(symbol)
        else:
            print(f"❌ Unknown symbol: {symbol}")

    if not valid_symbols:
        print("No valid symbols provided")
        sys.exit(1)

    print(f"Trading symbols: {', '.join(valid_symbols)}")
    print("")

    if len(valid_symbols) == 1:
        # Single symbol bot
        bot = StrategyBot(valid_symbols[0])
        bot.run(check_interval=60)
    else:
        # Multi-symbol bot
        bot = MultiSymbolBot(valid_symbols)
        bot.run(check_interval=60)
