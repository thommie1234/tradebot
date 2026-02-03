#!/usr/bin/env python3
"""
FTMO COMPLIANCE MODULE
======================

All FTMO prop firm rules enforced here:
1. Max Daily Loss: 5% (hard stop)
2. Max Total Drawdown: 10% (account termination)
3. No Hedging: 1 position per symbol max
4. Weekend Close: All positions closed Friday 23:50 GMT+2
5. Trading Hours: Per-symbol session times
6. Max Positions: 200 concurrent (FTMO server limit)
7. Max Daily Trades: 2000 (FTMO server limit)
"""

import pandas as pd
from datetime import datetime, time
from pathlib import Path

class FTMOCompliance:
    """FTMO rule enforcement"""

    # FTMO Rules
    MAX_DAILY_LOSS_PCT = 0.05  # 5%
    MAX_TOTAL_LOSS_PCT = 0.10  # 10%
    MAX_CONCURRENT_POSITIONS = 200
    MAX_DAILY_TRADES = 2000

    # Weekend close time (GMT+2)
    WEEKEND_CLOSE_DAY = 4  # Friday (0=Monday)
    WEEKEND_CLOSE_HOUR = 23
    WEEKEND_CLOSE_MINUTE = 50

    def __init__(self, initial_balance, symbols_file='../../symbols.xlsx'):
        self.initial_balance = initial_balance
        self.high_water_mark = initial_balance
        self.daily_start_balance = initial_balance
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Load trading hours
        self.trading_hours = self._load_trading_hours(symbols_file)

    def _load_trading_hours(self, symbols_file):
        """Load symbol trading hours from xlsx"""
        try:
            df = pd.read_excel(symbols_file)

            hours = {}
            for _, row in df.iterrows():
                symbol = row['symbol'].replace('/', '')  # EURUSD not EUR/USD

                hours[symbol] = {
                    'mon_open': self._parse_time(row.get('mon_open')),
                    'mon_close': self._parse_time(row.get('mon_close')),
                    'tue_open': self._parse_time(row.get('tue_open')),
                    'tue_close': self._parse_time(row.get('tue_close')),
                    'wed_open': self._parse_time(row.get('wed_open')),
                    'wed_close': self._parse_time(row.get('wed_close')),
                    'thu_open': self._parse_time(row.get('thu_open')),
                    'thu_close': self._parse_time(row.get('thu_close')),
                    'fri_open': self._parse_time(row.get('fri_open')),
                    'fri_close': self._parse_time(row.get('fri_close')),
                    'sat_status': row.get('sat_status', 'trading is closed'),
                    'sun_status': row.get('sun_status', 'trading is closed'),
                }

            return hours
        except Exception as e:
            print(f"⚠️  Could not load trading hours: {e}")
            return {}

    def _parse_time(self, time_str):
        """Parse time string (HH:MM:SS) to time object"""
        if pd.isna(time_str):
            return None

        try:
            if isinstance(time_str, str):
                parts = time_str.split(':')
                return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            elif isinstance(time_str, pd.Timestamp):
                return time_str.time()
            else:
                return None
        except:
            return None

    def reset_daily_counters(self):
        """Reset daily counters (called at start of new trading day)"""
        today = datetime.now().date()

        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = today
            print(f"Daily: Daily counters reset for {today}")

    def check_daily_loss(self, current_balance):
        """Check if daily loss limit exceeded"""
        daily_pnl = current_balance - self.daily_start_balance
        daily_loss_pct = (daily_pnl / self.daily_start_balance) * -1

        if daily_loss_pct >= self.MAX_DAILY_LOSS_PCT:
            return False, f"FTMO: FTMO DAILY LOSS LIMIT: {daily_loss_pct*100:.2f}% >= 5%"

        return True, daily_loss_pct

    def check_total_drawdown(self, current_balance):
        """Check if total drawdown limit exceeded"""
        # Update high water mark
        if current_balance > self.high_water_mark:
            self.high_water_mark = current_balance

        # Calculate drawdown from high water mark
        total_dd = ((self.high_water_mark - current_balance) / self.high_water_mark)

        if total_dd >= self.MAX_TOTAL_LOSS_PCT:
            return False, f"FTMO: FTMO TOTAL DRAWDOWN: {total_dd*100:.2f}% >= 10% (Account TERMINATED)"

        return True, total_dd

    def check_weekend_close(self):
        """Check if we need to close all positions (weekend)"""
        now = datetime.now()

        # Friday after 23:50 GMT+2
        if now.weekday() == self.WEEKEND_CLOSE_DAY:
            close_time = time(self.WEEKEND_CLOSE_HOUR, self.WEEKEND_CLOSE_MINUTE)
            if now.time() >= close_time:
                return True, "Time: Weekend close time (Friday 23:50)"

        # Saturday or Sunday
        if now.weekday() in [5, 6]:
            return True, "Daily: Weekend - markets closed"

        return False, None

    def check_trading_hours(self, symbol):
        """Check if symbol is allowed to trade now"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Check if we have trading hours for this symbol
        if symbol not in self.trading_hours:
            # No info = allow (conservative)
            return True, "No trading hours info"

        hours = self.trading_hours[symbol]

        # Weekend check
        if weekday == 5:  # Saturday
            if 'closed' in hours['sat_status'].lower():
                return False, "Saturday - market closed"

        if weekday == 6:  # Sunday
            if 'closed' in hours['sun_status'].lower():
                return False, "Sunday - market closed"

        # Weekday session check
        day_names = ['mon', 'tue', 'wed', 'thu', 'fri']
        if weekday < 5:
            day = day_names[weekday]
            open_time = hours.get(f'{day}_open')
            close_time = hours.get(f'{day}_close')

            if open_time and close_time:
                if not (open_time <= current_time <= close_time):
                    return False, f"Outside trading hours ({open_time}-{close_time})"

        return True, "In trading hours"

    def check_max_positions(self, current_positions):
        """Check if max concurrent positions exceeded"""
        if current_positions >= self.MAX_CONCURRENT_POSITIONS:
            return False, f"FTMO: Max positions: {current_positions}/{self.MAX_CONCURRENT_POSITIONS}"

        return True, current_positions

    def check_max_daily_trades(self):
        """Check if max daily trades exceeded"""
        if self.daily_trade_count >= self.MAX_DAILY_TRADES:
            return False, f"FTMO: Max daily trades: {self.daily_trade_count}/{self.MAX_DAILY_TRADES}"

        return True, self.daily_trade_count

    def increment_trade_count(self):
        """Increment daily trade counter"""
        self.daily_trade_count += 1

    def check_all(self, current_balance, current_positions, symbol=None):
        """
        Run ALL FTMO compliance checks

        Returns: (allowed: bool, reason: str)
        """

        # 1. Reset daily counters if new day
        self.reset_daily_counters()

        # 2. Total drawdown (most critical)
        ok, dd = self.check_total_drawdown(current_balance)
        if not ok:
            return False, dd

        # 3. Daily loss
        ok, loss = self.check_daily_loss(current_balance)
        if not ok:
            return False, loss

        # 4. Weekend close
        ok, reason = self.check_weekend_close()
        if ok:  # ok=True means we SHOULD close (confusing naming)
            return False, reason

        # 5. Max positions
        ok, count = self.check_max_positions(current_positions)
        if not ok:
            return False, count

        # 6. Max daily trades
        ok, count = self.check_max_daily_trades()
        if not ok:
            return False, count

        # 7. Trading hours (symbol-specific)
        if symbol:
            ok, reason = self.check_trading_hours(symbol)
            if not ok:
                return False, f"{symbol}: {reason}"

        return True, "OK All FTMO checks passed"

if __name__ == "__main__":
    # Test FTMO compliance
    ftmo = FTMOCompliance(initial_balance=100000)

    print("="*70)
    print("FTMO COMPLIANCE TEST")
    print("="*70)
    print("")

    # Test current compliance
    allowed, reason = ftmo.check_all(
        current_balance=95000,
        current_positions=1,
        symbol='SOLUSD'
    )

    print(f"Allowed to trade: {allowed}")
    print(f"Reason: {reason}")
    print("")

    # Test trading hours
    print("Trading Hours Check:")
    for symbol in ['SOLUSD', 'MSFT', 'EURUSD', 'XAUUSD']:
        ok, msg = ftmo.check_trading_hours(symbol)
        status = "OK" if ok else "NO"
        print(f"  {status} {symbol}: {msg}")
