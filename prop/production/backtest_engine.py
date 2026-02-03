#!/usr/bin/env python3
"""
SOVEREIGN BACKTEST ENGINE - No Mercy Edition
=============================================
Multi-threaded backtesting with real-world costs

Author: Sovereign Trading System
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

class BacktestConfig:
    """Backtest settings - no fairy tales"""

    # Tickers to test (2 threads each)
    TICKERS = {
        # Crypto
        'BTCUSD': {'type': 'crypto', 'contract_size': 1},
        'ETHUSD': {'type': 'crypto', 'contract_size': 1},
        'SOLUSD': {'type': 'crypto', 'contract_size': 100},
        # Stocks
        'MSFT': {'type': 'stock', 'contract_size': 1},
        'NVDA': {'type': 'stock', 'contract_size': 1},
        'AAPL': {'type': 'stock', 'contract_size': 1},
        'TSLA': {'type': 'stock', 'contract_size': 1},
        'AMZN': {'type': 'stock', 'contract_size': 1},
        'GOOGL': {'type': 'stock', 'contract_size': 1},
        'META': {'type': 'stock', 'contract_size': 1},
    }

    # Real-world costs (no mercy)
    COSTS = {
        'crypto': {
            'spread_pct': 0.0015,      # 0.15% spread
            'slippage_pct': 0.001,     # 0.10% slippage (Wine/Linux penalty)
            'commission_pct': 0.0005,  # 0.05% commission per side
        },
        'stock': {
            'spread_pct': 0.0003,      # 0.03% spread
            'slippage_pct': 0.0005,    # 0.05% slippage
            'commission_pct': 0.0002,  # 0.02% commission per side
        }
    }

    # ATR Settings
    ATR_PERIOD = 14
    ATR_TIMEFRAME = mt5.TIMEFRAME_H1
    ATR_MULTIPLIER_CRYPTO = 2.0  # 2x ATR for crypto
    ATR_MULTIPLIER_STOCK = 0.8   # 0.8x ATR for stocks

    # Risk/Reward
    RR_RATIO = 3.0  # 1:3
    PARTIAL_CLOSE_THRESHOLD = 0.60  # 60%

    # Account
    INITIAL_BALANCE = 100000
    RISK_PER_TRADE = 0.003  # 0.3%

    # Backtest period
    LOOKBACK_DAYS = 365  # 1 year

    # Threading
    THREADS_PER_TICKER = 2
    MAX_WORKERS = 20


# ============================================================================
# ATR CALCULATOR
# ============================================================================

def calculate_atr(df, period=14):
    """Calculate ATR from OHLC dataframe"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


# ============================================================================
# SIGNAL GENERATOR (Simplified for backtest)
# ============================================================================

def generate_signals(df):
    """
    Generate BUY/SELL signals based on simple momentum
    Returns: DataFrame with 'signal' column (1=BUY, -1=SELL, 0=NONE)
    """
    # Simple momentum: price above/below 20-period SMA
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Signals
    df['signal'] = 0

    # BUY: price crosses above SMA20, SMA20 > SMA50, RSI < 70
    buy_condition = (
        (df['close'] > df['sma_20']) &
        (df['close'].shift(1) <= df['sma_20'].shift(1)) &
        (df['sma_20'] > df['sma_50']) &
        (df['rsi'] < 70)
    )

    # SELL: price crosses below SMA20, SMA20 < SMA50, RSI > 30
    sell_condition = (
        (df['close'] < df['sma_20']) &
        (df['close'].shift(1) >= df['sma_20'].shift(1)) &
        (df['sma_20'] < df['sma_50']) &
        (df['rsi'] > 30)
    )

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    return df


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Single-ticker backtest engine"""

    def __init__(self, symbol, config=BacktestConfig):
        self.symbol = symbol
        self.config = config
        self.ticker_config = config.TICKERS.get(symbol, {})
        self.ticker_type = self.ticker_config.get('type', 'stock')
        self.contract_size = self.ticker_config.get('contract_size', 1)
        self.costs = config.COSTS[self.ticker_type]

        # ATR multiplier based on type
        self.atr_mult = (config.ATR_MULTIPLIER_CRYPTO if self.ticker_type == 'crypto'
                        else config.ATR_MULTIPLIER_STOCK)

        # Results
        self.trades = []
        self.equity_curve = []
        self.balance = config.INITIAL_BALANCE
        self.peak_balance = config.INITIAL_BALANCE
        self.max_drawdown = 0
        self.max_drawdown_pct = 0

    def get_historical_data(self):
        """Fetch historical data from MT5"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.LOOKBACK_DAYS)

        rates = mt5.copy_rates_range(
            self.symbol,
            self.config.ATR_TIMEFRAME,
            start_date,
            end_date
        )

        if rates is None or len(rates) == 0:
            print(f"[{self.symbol}] No data available")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def calculate_costs(self, entry_price, exit_price, direction):
        """Calculate total trade costs"""
        spread_cost = entry_price * self.costs['spread_pct']
        slippage_cost = entry_price * self.costs['slippage_pct']
        commission_cost = (entry_price + exit_price) * self.costs['commission_pct']

        total_cost = spread_cost + slippage_cost + commission_cost
        return total_cost

    def simulate_trade_vectorized(self, entry_idx, entry_price, direction, sl_price, tp_price,
                                     highs, lows, closes, times):
        """
        FAST vectorized trade simulation using numpy
        """
        sl_distance = abs(entry_price - sl_price)
        risk_amount = self.balance * self.config.RISK_PER_TRADE
        lot_size = min(max(risk_amount / (sl_distance * self.contract_size), 0.01), 10.0)

        remaining_lots = lot_size
        partial_closed = False
        partial_profit = 0.0
        original_sl = sl_price

        n = len(highs)

        for i in range(n):
            high = highs[i]
            low = lows[i]
            close = closes[i]

            if direction == 1:  # BUY
                if low <= sl_price:
                    pnl = (sl_price - entry_price) * remaining_lots * self.contract_size
                    costs = self.calculate_costs(entry_price, sl_price, direction)
                    return {'entry_time': times[0], 'exit_time': times[i], 'direction': 'BUY',
                            'entry_price': entry_price, 'exit_price': sl_price, 'sl': original_sl,
                            'tp': tp_price, 'lots': lot_size, 'pnl': pnl + partial_profit - costs,
                            'costs': costs, 'result': 'SL_HIT', 'partial_closed': partial_closed}

                if not partial_closed and (close - entry_price) / (tp_price - entry_price) >= 0.6:
                    close_lots = remaining_lots * 0.5
                    partial_profit = (close - entry_price) * close_lots * self.contract_size
                    remaining_lots -= close_lots
                    partial_closed = True
                    sl_price = entry_price

                if high >= tp_price:
                    pnl = (tp_price - entry_price) * remaining_lots * self.contract_size
                    costs = self.calculate_costs(entry_price, tp_price, direction)
                    return {'entry_time': times[0], 'exit_time': times[i], 'direction': 'BUY',
                            'entry_price': entry_price, 'exit_price': tp_price, 'sl': original_sl,
                            'tp': tp_price, 'lots': lot_size, 'pnl': pnl + partial_profit - costs,
                            'costs': costs, 'result': 'TP_HIT', 'partial_closed': partial_closed}
            else:  # SELL
                if high >= sl_price:
                    pnl = (entry_price - sl_price) * remaining_lots * self.contract_size
                    costs = self.calculate_costs(entry_price, sl_price, direction)
                    return {'entry_time': times[0], 'exit_time': times[i], 'direction': 'SELL',
                            'entry_price': entry_price, 'exit_price': sl_price, 'sl': original_sl,
                            'tp': tp_price, 'lots': lot_size, 'pnl': pnl + partial_profit - costs,
                            'costs': costs, 'result': 'SL_HIT', 'partial_closed': partial_closed}

                if not partial_closed and (entry_price - close) / (entry_price - tp_price) >= 0.6:
                    close_lots = remaining_lots * 0.5
                    partial_profit = (entry_price - close) * close_lots * self.contract_size
                    remaining_lots -= close_lots
                    partial_closed = True
                    sl_price = entry_price

                if low <= tp_price:
                    pnl = (entry_price - tp_price) * remaining_lots * self.contract_size
                    costs = self.calculate_costs(entry_price, tp_price, direction)
                    return {'entry_time': times[0], 'exit_time': times[i], 'direction': 'SELL',
                            'entry_price': entry_price, 'exit_price': tp_price, 'sl': original_sl,
                            'tp': tp_price, 'lots': lot_size, 'pnl': pnl + partial_profit - costs,
                            'costs': costs, 'result': 'TP_HIT', 'partial_closed': partial_closed}
        return None

    def run(self):
        """Run the backtest"""
        print(f"[{self.symbol}] Starting backtest...")

        # Get data
        df = self.get_historical_data()
        if df is None:
            return None

        # Calculate ATR
        df['atr'] = calculate_atr(df, self.config.ATR_PERIOD)

        # Generate signals
        df = generate_signals(df)

        # Drop NaN rows
        df = df.dropna()

        print(f"[{self.symbol}] Data loaded: {len(df)} candles")

        # Iterate through signals
        in_trade = False

        for i in range(len(df) - 1):
            row = df.iloc[i]

            if in_trade:
                continue

            signal = row['signal']
            if signal == 0:
                continue

            # Entry setup
            entry_time = df.index[i]
            entry_price = row['close']
            atr = row['atr']

            sl_distance = atr * self.atr_mult
            tp_distance = sl_distance * self.config.RR_RATIO

            if signal == 1:  # BUY
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:  # SELL
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance

            # Simulate trade
            df_slice = df.iloc[i+1:]
            trade = self.simulate_trade(entry_time, entry_price, signal, sl_price, tp_price, df_slice)

            if trade:
                self.trades.append(trade)
                self.balance += trade['pnl']

                # Track drawdown
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance

                current_dd = self.peak_balance - self.balance
                current_dd_pct = current_dd / self.peak_balance

                if current_dd > self.max_drawdown:
                    self.max_drawdown = current_dd
                if current_dd_pct > self.max_drawdown_pct:
                    self.max_drawdown_pct = current_dd_pct

                self.equity_curve.append({
                    'time': trade['exit_time'],
                    'balance': self.balance,
                    'drawdown': current_dd,
                    'drawdown_pct': current_dd_pct
                })

        return self.get_results()

    def get_results(self):
        """Calculate final statistics"""
        if not self.trades:
            return {
                'symbol': self.symbol,
                'total_trades': 0,
                'error': 'No trades executed'
            }

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        total_profit = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        win_rate = len(wins) / len(self.trades) * 100

        total_costs = sum(t['costs'] for t in self.trades)
        partial_closes = len([t for t in self.trades if t['partial_closed']])

        # Find worst day
        trade_df = pd.DataFrame(self.trades)
        trade_df['date'] = pd.to_datetime(trade_df['exit_time']).dt.date
        daily_pnl = trade_df.groupby('date')['pnl'].sum()
        worst_day = daily_pnl.min() if len(daily_pnl) > 0 else 0
        worst_day_date = daily_pnl.idxmin() if len(daily_pnl) > 0 else None

        return {
            'symbol': self.symbol,
            'type': self.ticker_type,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.balance - self.config.INITIAL_BALANCE,
            'total_costs': total_costs,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct * 100,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE * 100,
            'partial_closes': partial_closes,
            'worst_day_pnl': worst_day,
            'worst_day_date': worst_day_date,
            'avg_win': total_profit / len(wins) if wins else 0,
            'avg_loss': total_loss / len(losses) if losses else 0,
        }


# ============================================================================
# STRESS TEST - CRASH DAYS
# ============================================================================

def find_crash_days(symbol, threshold=-0.05):
    """Find days with drops > threshold (e.g., -5%)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date, end_date)
    if rates is None:
        return []

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['return'] = df['close'].pct_change()

    crash_days = df[df['return'] <= threshold]

    return crash_days[['time', 'return', 'close']].to_dict('records')


# ============================================================================
# MULTI-THREADED RUNNER
# ============================================================================

def run_single_backtest(symbol):
    """Run backtest for a single symbol (called by thread)"""
    try:
        engine = BacktestEngine(symbol)
        return engine.run()
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_all_backtests():
    """Run backtests for all tickers in parallel"""
    print("=" * 70)
    print("SOVEREIGN BACKTEST ENGINE - No Mercy Edition")
    print("=" * 70)
    print(f"Tickers: {len(BacktestConfig.TICKERS)}")
    print(f"Threads per ticker: {BacktestConfig.THREADS_PER_TICKER}")
    print(f"Lookback: {BacktestConfig.LOOKBACK_DAYS} days")
    print(f"Initial Balance: ${BacktestConfig.INITIAL_BALANCE:,.0f}")
    print("=" * 70)
    print()

    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed!")
        return

    results = []

    # Run backtests in parallel
    with ThreadPoolExecutor(max_workers=BacktestConfig.MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_backtest, symbol): symbol
                   for symbol in BacktestConfig.TICKERS.keys()}

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[{symbol}] Error: {e}")

    mt5.shutdown()

    # Print results
    print_results(results)

    return results


def print_results(results):
    """Print formatted results"""
    print()
    print("=" * 70)
    print("BACKTEST RESULTS - The Three Numbers That Matter")
    print("=" * 70)
    print()

    # Sort by profit factor
    valid_results = [r for r in results if 'error' not in r and r['total_trades'] > 0]
    valid_results.sort(key=lambda x: x['profit_factor'], reverse=True)

    print(f"{'Symbol':<10} {'Trades':>7} {'Win%':>7} {'PF':>7} {'MaxDD%':>8} {'Return%':>9} {'Verdict':>10}")
    print("-" * 70)

    for r in valid_results:
        pf = r['profit_factor']
        win_rate = r['win_rate']
        max_dd = r['max_drawdown_pct']

        # Verdict
        if pf >= 1.5 and win_rate >= 50 and max_dd < 10:
            verdict = "STRONG"
        elif pf >= 1.2 and win_rate >= 40 and max_dd < 15:
            verdict = "OK"
        else:
            verdict = "WEAK"

        print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']:>6.1f}% {pf:>7.2f} "
              f"{max_dd:>7.1f}% {r['return_pct']:>8.1f}% {verdict:>10}")

    # Print errors
    errors = [r for r in results if 'error' in r]
    if errors:
        print()
        print("ERRORS:")
        for e in errors:
            print(f"  {e['symbol']}: {e['error']}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if valid_results:
        avg_pf = sum(r['profit_factor'] for r in valid_results if r['profit_factor'] != float('inf')) / len(valid_results)
        avg_wr = sum(r['win_rate'] for r in valid_results) / len(valid_results)
        avg_dd = sum(r['max_drawdown_pct'] for r in valid_results) / len(valid_results)
        total_return = sum(r['return_pct'] for r in valid_results)

        print(f"Avg Profit Factor: {avg_pf:.2f} {'(OK)' if avg_pf >= 1.2 else '(WEAK)'}")
        print(f"Avg Win Rate: {avg_wr:.1f}%")
        print(f"Avg Max Drawdown: {avg_dd:.1f}%")
        print(f"Total Return (all symbols): {total_return:.1f}%")

        # Worst days
        print()
        print("STRESS TEST - Worst Days:")
        for r in valid_results:
            if r.get('worst_day_pnl', 0) < -500:
                print(f"  {r['symbol']}: ${r['worst_day_pnl']:,.0f} on {r['worst_day_date']}")

    print()
    print("=" * 70)
    print("Legend: PF = Profit Factor (>1.2 = OK), MaxDD = Max Drawdown")
    print("Costs included: Spread + Slippage + Commission")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_all_backtests()
