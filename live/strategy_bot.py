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
from mt5_bootstrap import initialize_mt5
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
# DATA DOWNLOADER
# ============================================================================
import shutil
import os
from datetime import timezone

def print_log(level, message):
    """Prints a formatted log message for the downloader."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [DOWNLOADER] [{level.upper()}] {message}")

def get_symbols_from_excel(file_path):
    """Reads a list of symbols from the first column of ALL sheets in an Excel file."""
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None, header=None)
        all_symbols = set()
        for sheet_name, df in all_sheets.items():
            if not df.empty:
                all_symbols.update(df.iloc[:, 0].dropna().tolist())
        
        symbols = sorted(list(all_symbols)) # Sort for consistent order
        print_log("info", f"Found {len(symbols)} unique symbols across all sheets in '{file_path}'.")
        return symbols
    except FileNotFoundError:
        print_log("critical", f"Symbols file not found at '{file_path}'. Exiting.")
        return []
    except Exception as e:
        print_log("critical", f"Error reading symbols file: {e}. Exiting.")
        return []

def resolve_mt5_symbol(requested_symbol):
    """Try a few broker symbol variants and return the first selectable one."""
    candidates = []
    for s in [
        requested_symbol,
        requested_symbol.replace("/", ""),
        requested_symbol.replace("/", "."),
        requested_symbol.replace(" ", ""),
    ]:
        if s and s not in candidates:
            candidates.append(s)

    for candidate in candidates:
        if mt5.symbol_select(candidate, True):
            return candidate
    return None

def sanitize_symbol_for_path(symbol):
    """Make symbol safe as folder name while preserving readability."""
    return symbol.replace("/", "_").replace("\\", "_")

def fetch_ticks_month(mt5_symbol, from_date, to_date, batch_size=5000):
    """
    Fetch monthly ticks in pages using copy_ticks_from.
    copy_ticks_range can return empty on some brokers even when data exists.
    """
    chunks = []
    cursor = from_date

    while cursor < to_date:
        ticks = mt5.copy_ticks_from(mt5_symbol, cursor, batch_size, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            break

        df = pd.DataFrame(ticks)
        if df.empty:
            break

        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[df['time'] < to_date]
        if df.empty:
            break

        chunks.append(df)

        # Move cursor past the latest millisecond to avoid duplicates/infinite loops.
        last_msc = int(df['time_msc'].iloc[-1])
        cursor = pd.to_datetime(last_msc + 1, unit='ms', utc=True).to_pydatetime()
        if len(df) < batch_size:
            break

    if not chunks:
        return None

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=['time_msc', 'bid', 'ask', 'last', 'volume', 'flags'])
    return out

def month_start_end(year, month):
    from_date = datetime(year, month, 1, tzinfo=timezone.utc)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    to_date = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
    return from_date, to_date

def month_has_ticks(mt5_symbol, year, month):
    """Cheap probe: check if at least one tick exists inside the month."""
    from_date, to_date = month_start_end(year, month)
    ticks = mt5.copy_ticks_from(mt5_symbol, from_date, 1, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        return False
    t = pd.to_datetime(int(ticks[0]['time']), unit='s', utc=True).to_pydatetime()
    return from_date <= t < to_date

def detect_symbol_month_bounds(mt5_symbol, months_desc, months_asc):
    """
    Detect latest and earliest month with data for a symbol.
    Returns (latest_key, earliest_key) as YYYYMM ints, or (None, None) if no data.
    """
    latest_key = None
    earliest_key = None

    for y, m in months_desc:
        if month_has_ticks(mt5_symbol, y, m):
            latest_key = y * 100 + m
            break

    if latest_key is None:
        return None, None

    for y, m in months_asc:
        if month_has_ticks(mt5_symbol, y, m):
            earliest_key = y * 100 + m
            break

    return latest_key, earliest_key

def get_available_space_gb(path):
    """Returns the free space on a drive in gigabytes."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free / (1024**3)
    except FileNotFoundError:
        return 0

def select_active_drive(drive_priority, current_drive_index, threshold_gb):
    """Selects the next available drive with enough free space."""
    for i in range(current_drive_index, len(drive_priority)):
        drive_path = drive_priority[i]
        free_space_gb = get_available_space_gb(drive_path)
        print_log("info", f"Checking drive '{drive_path}': {free_space_gb:.2f} GB free.")
        if free_space_gb > threshold_gb:
            print_log("info", f"Selected active drive: '{drive_path}'.")
            return i, drive_path
    
    print_log("critical", "No drives with enough free space found! Exiting.")
    return -1, None

def run_downloader():
    """Main function to download historical tick data."""
    SYMBOLS_FILE_PATH = os.getenv("SYMBOLS_FILE_PATH", "/home/tradebot/tradebots/data/symbols.xlsx")
    DRIVE_PRIORITY = [
        '/home/tradebot/ssd_data_1',
        '/home/tradebot/ssd_data_2',
        '/home/tradebot/data_1',
        '/home/tradebot/data_2',
        '/home/tradebot/data_3',
    ]
    DATA_ROOT_DIR_NAME = "tick_data"
    DRIVE_FREE_SPACE_THRESHOLD_GB = 10
    YEARS_TO_DOWNLOAD = 10

    print_log("info", "Starting tick data downloader integrated mode.")
    time.sleep(10) # Give MT5 a moment to fully initialize

    ok, mt5_error, mode = initialize_mt5()
    if not ok:
        print_log("critical", f"MT5 initialize() failed, error code = {mt5_error}")
        return
    print_log("info", f"MT5 init mode: {mode}")

    print_log("info", f"Connected to MetaTrader 5 build {mt5.version()}")

    symbols = get_symbols_from_excel(SYMBOLS_FILE_PATH)
    if not symbols:
        mt5.shutdown()
        return

    current_drive_idx, active_drive = select_active_drive(DRIVE_PRIORITY, 0, DRIVE_FREE_SPACE_THRESHOLD_GB)
    if active_drive is None:
        mt5.shutdown()
        return

    start_date = datetime.now(timezone.utc)
    start_year = start_date.year - YEARS_TO_DOWNLOAD + 1
    years_desc = list(range(start_date.year, start_year - 1, -1))
    months_desc_all = []
    months_asc_all = []
    for y in years_desc:
        if y == start_date.year:
            md = list(range(start_date.month, 0, -1))
            ma = list(range(1, start_date.month + 1))
        else:
            md = list(range(12, 0, -1))
            ma = list(range(1, 13))
        for m in md:
            months_desc_all.append((y, m))
        for m in ma:
            months_asc_all.append((y, m))

    # Pre-resolve symbols and detect available month ranges to avoid empty-month scans.
    symbol_meta = []
    for symbol in symbols:
        mt5_symbol = resolve_mt5_symbol(symbol)
        if not mt5_symbol:
            print_log("warning", f"Could not select symbol '{symbol}' in MT5. Skipping.")
            continue
        latest_key, earliest_key = detect_symbol_month_bounds(mt5_symbol, months_desc_all, months_asc_all)
        if latest_key is None:
            print_log("warning", f"No data in last {YEARS_TO_DOWNLOAD} years for '{symbol}'. Skipping symbol.")
            continue
        symbol_meta.append({
            "symbol": symbol,
            "mt5_symbol": mt5_symbol,
            "symbol_dir": sanitize_symbol_for_path(symbol),
            "latest_key": latest_key,
            "earliest_key": earliest_key,
        })
        print_log("info", f"Range {symbol}: {earliest_key}..{latest_key}")

    # Prioritize recency across ALL symbols: newest year/month first.
    for year in years_desc:
        if year == start_date.year:
            months_desc = list(range(start_date.month, 0, -1))
        else:
            months_desc = list(range(12, 0, -1))

        for month in months_desc:
            print_log("info", f"=== Processing batch {year}-{month:02d} across all symbols ===")
            month_key = year * 100 + month
            for meta in symbol_meta:
                symbol = meta["symbol"]
                mt5_symbol = meta["mt5_symbol"]
                symbol_dir = meta["symbol_dir"]
                if month_key > meta["latest_key"] or month_key < meta["earliest_key"]:
                    continue

                free_space_gb = get_available_space_gb(active_drive)
                if free_space_gb < DRIVE_FREE_SPACE_THRESHOLD_GB:
                    print_log("warning", f"Drive '{active_drive}' is low on space ({free_space_gb:.2f} GB left).")
                    new_drive_idx, new_active_drive = select_active_drive(DRIVE_PRIORITY, current_drive_idx + 1, DRIVE_FREE_SPACE_THRESHOLD_GB)
                    if new_active_drive is None:
                        print_log("critical", "No more drives with space. Stopping download.")
                        mt5.shutdown()
                        return
                    current_drive_idx = new_drive_idx
                    active_drive = new_active_drive

                target_dir = os.path.join(active_drive, DATA_ROOT_DIR_NAME, symbol_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                file_name = f"{year}-{month:02d}.parquet"
                target_file_path = os.path.join(target_dir, file_name)

                if os.path.exists(target_file_path):
                    print_log("info", f"Skipping '{target_file_path}' (already exists).")
                    continue

                if mt5_symbol != symbol:
                    print_log("info", f"Downloading {symbol} via MT5 symbol '{mt5_symbol}' for {year}-{month:02d}...")
                else:
                    print_log("info", f"Downloading {symbol} for {year}-{month:02d}...")

                from_date, to_date = month_start_end(year, month)

                try:
                    ticks_df = fetch_ticks_month(mt5_symbol, from_date, to_date, batch_size=5000)

                    if ticks_df is None or len(ticks_df) == 0:
                        print_log("warning", f"No ticks found for {symbol} in {year}-{month:02d}. MT5 error: {mt5.last_error()}")
                        continue

                    ticks_df.to_parquet(target_file_path, index=False, compression='snappy')
                    print_log("success", f"Saved {len(ticks_df)} ticks to '{target_file_path}'.")

                except Exception as e:
                    print_log("error", f"An error occurred while downloading {symbol} for {year}-{month:02d}: {e}")
    
    print_log("info", "All symbols processed. Downloader finished.")
    mt5.shutdown()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("="*70)
    print("SOVEREIGN STRATEGY BOT - PRODUCTION MODE")
    print("="*70)
    print("")

    if "--download-data" in sys.argv:
        # --- DOWNLOADER MODE ---
        run_downloader()

    else:
        # --- TRADING BOT MODE ---
        if len(sys.argv) < 2:
            print("Usage:")
            print("  Single symbol:  python strategy_bot.py SOLUSD")
            print("  Multi symbol:   python strategy_bot.py SOLUSD MSFT")
            print("  Data Downloader: python strategy_bot.py --download-data")
            print("")
            sys.exit(1)

        symbols = [arg for arg in sys.argv[1:] if not arg.startswith('--')]

        # Validate symbols
        valid_symbols = []
        for symbol in symbols:
            if symbol in Config.SYMBOLS:
                valid_symbols.append(symbol)
            else:
                print(f"❌ Unknown symbol: {symbol}")

        if not valid_symbols:
            print("No valid symbols provided for trading.")
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
