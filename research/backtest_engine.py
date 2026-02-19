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
import os
from mt5_bootstrap import initialize_mt5
from xgboost import XGBClassifier

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
        'N25.cash': {'type': 'stock', 'contract_size': 1},
        'GBP_JPY': {'type': 'stock', 'contract_size': 1},
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
    RISK_PER_TRADE = 0.0015  # 0.15%
    MAX_LOT_SIZE = {
        'crypto': 0.20,
        'stock': 2.00,
    }
    MAX_DRAWDOWN_STOP_PCT = 0.15  # Stop symbol test after 15% DD

    # Backtest period
    LOOKBACK_DAYS = 365  # 1 year

    # Threading
    THREADS_PER_TICKER = 2
    MAX_WORKERS = max(1, (os.cpu_count() or 1))

    # ML gate defaults
    USE_ML_GATE = True
    META_THRESHOLD = 0.55
    EMBARGO_BARS = 24
    XGB_ESTIMATORS = 160
    XGB_MAX_DEPTH = 4
    XGB_LR = 0.05


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

    # Raw signal at candle close.
    df['signal_raw'] = 0

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

    df.loc[buy_condition, 'signal_raw'] = 1
    df.loc[sell_condition, 'signal_raw'] = -1

    # Shift(1) discipline:
    # a signal generated at t can only be acted on from t+1 onward.
    df['signal'] = df['signal_raw'].shift(1).fillna(0).astype(int)

    return df


def build_ml_features(df):
    x = df.copy()
    x["ret1_raw"] = np.log(x["close"]).diff()
    x["ret3_raw"] = x["close"].pct_change(3)
    x["ret12_raw"] = x["close"].pct_change(12)
    x["vol20_raw"] = x["ret1_raw"].rolling(20).std()
    q_low = x["vol20_raw"].quantile(0.01)
    q_high = x["vol20_raw"].quantile(0.99)
    if pd.notna(q_low) and pd.notna(q_high):
        x["vol20_raw"] = x["vol20_raw"].clip(lower=q_low, upper=q_high)
    ma20 = x["close"].rolling(20).mean()
    sd20 = x["close"].rolling(20).std()
    x["z20_raw"] = (x["close"] - ma20) / sd20
    x["range_raw"] = (x["high"] - x["low"]) / x["close"]
    x["vchg1_raw"] = x["tick_volume"].pct_change()
    x["vratio20_raw"] = x["tick_volume"] / x["tick_volume"].rolling(20).mean()
    hour = x.index.hour
    x["hour_sin_raw"] = np.sin(2 * np.pi * hour / 24.0)
    x["hour_cos_raw"] = np.cos(2 * np.pi * hour / 24.0)

    # Shift(1) discipline for all predictor features.
    for c in [
        "ret1_raw",
        "ret3_raw",
        "ret12_raw",
        "vol20_raw",
        "z20_raw",
        "range_raw",
        "vchg1_raw",
        "vratio20_raw",
        "hour_sin_raw",
        "hour_cos_raw",
    ]:
        x[c.replace("_raw", "")] = x[c].shift(1)
    x["primary_side"] = np.where(x["z20"] > 1.5, -1, np.where(x["z20"] < -1.5, 1, 0))
    return x


def triple_barrier_arrays(close, vol_proxy, side, horizon, pt_mult, sl_mult):
    n = len(close)
    label = np.full(n, np.nan)
    tb_ret = np.full(n, np.nan)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    idx = np.where((side != 0) & np.isfinite(vol_proxy))[0]
    for i in idx:
        end = min(i + horizon, n - 1)
        if end <= i:
            continue
        epx = close[i]
        pt = max(1e-12, pt_mult * vol_proxy[i])
        sl = max(1e-12, sl_mult * vol_proxy[i])
        up[i] = pt
        dn[i] = sl
        chosen_ret = side[i] * ((close[end] / epx) - 1.0)
        chosen_lab = 1.0 if chosen_ret > 0 else 0.0
        for j in range(i + 1, end + 1):
            r = side[i] * ((close[j] / epx) - 1.0)
            if r >= pt:
                chosen_ret = r
                chosen_lab = 1.0
                break
            if r <= -sl:
                chosen_ret = r
                chosen_lab = 0.0
                break
        label[i] = chosen_lab
        tb_ret[i] = chosen_ret
    return label, tb_ret, up, dn


def generate_ml_gated_signals(df, costs, cfg):
    d = build_ml_features(df)
    label, tb_ret, up, dn = triple_barrier_arrays(
        close=d["close"].to_numpy(),
        vol_proxy=d["vol20"].to_numpy(),
        side=d["primary_side"].to_numpy(),
        horizon=24,
        pt_mult=2.0,
        sl_mult=1.5,
    )
    d["label"] = label
    d["tb_ret"] = tb_ret
    d["up"] = up
    d["dn"] = dn

    feat_cols = [
        "ret1",
        "ret3",
        "ret12",
        "vol20",
        "z20",
        "range",
        "vchg1",
        "vratio20",
        "hour_sin",
        "hour_cos",
        "primary_side",
    ]
    m = d.dropna(subset=feat_cols + ["label", "tb_ret", "up", "dn"]).copy()
    if len(m) < 400:
        return generate_signals(df)

    X = m[feat_cols].to_numpy()
    y = m["label"].astype(int).to_numpy()
    gross = m["tb_ret"].to_numpy()
    upside = m["up"].to_numpy()
    downside = m["dn"].to_numpy()

    p_primary = np.full(len(m), np.nan)
    p_meta = np.full(len(m), np.nan)
    cost = costs["spread_pct"] + costs["slippage_pct"] + (2.0 * costs["commission_pct"])
    n = len(m)
    split_edges = np.linspace(0, n, 6, dtype=int)

    for k in range(1, len(split_edges) - 1):
        te_start = split_edges[k]
        te_end = split_edges[k + 1]
        tr_end = max(0, te_start - cfg.EMBARGO_BARS)
        if tr_end < 200 or te_end <= te_start:
            continue
        tr = np.arange(0, tr_end)
        te = np.arange(te_start, te_end)

        ytr = y[tr]
        if np.unique(ytr).size < 2:
            p_primary[te] = float(np.unique(ytr)[0]) if len(ytr) else 0.0
            p_meta[te] = 0.0
            continue

        pos = max(int(ytr.sum()), 1)
        neg = max(int(len(ytr) - ytr.sum()), 1)
        primary = XGBClassifier(
            n_estimators=cfg.XGB_ESTIMATORS,
            max_depth=cfg.XGB_MAX_DEPTH,
            learning_rate=cfg.XGB_LR,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
            scale_pos_weight=neg / pos,
        )
        primary.fit(X[tr], ytr)
        p_tr = primary.predict_proba(X[tr])[:, 1]
        p_te = primary.predict_proba(X[te])[:, 1]
        p_primary[te] = p_te

        y_meta = ((p_tr >= 0.5).astype(np.int8) == ytr.astype(np.int8)).astype(np.int8)
        if np.unique(y_meta).size < 2:
            p_meta[te] = float(np.unique(y_meta)[0])
        else:
            mpos = max(int(y_meta.sum()), 1)
            mneg = max(int(len(y_meta) - y_meta.sum()), 1)
            x_meta_tr = np.column_stack([X[tr], p_tr])
            x_meta_te = np.column_stack([X[te], p_te])
            meta = XGBClassifier(
                n_estimators=max(80, cfg.XGB_ESTIMATORS // 2),
                max_depth=max(3, cfg.XGB_MAX_DEPTH - 1),
                learning_rate=max(0.03, cfg.XGB_LR),
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
                random_state=43,
                scale_pos_weight=mneg / mpos,
            )
            meta.fit(x_meta_tr, y_meta)
            p_meta[te] = meta.predict_proba(x_meta_te)[:, 1]

    m["p_primary"] = p_primary
    m["p_meta"] = p_meta
    m["ev"] = m["p_primary"] * upside - (1.0 - m["p_primary"]) * downside - cost
    m["trade_ok"] = (m["ev"] > 0.0) & (m["p_meta"] >= cfg.META_THRESHOLD)
    m["signal"] = np.where(m["trade_ok"], m["primary_side"], 0)

    out = df.copy()
    out["signal"] = 0
    out.loc[m.index, "signal"] = m["signal"].astype(int)
    out["signal"] = out["signal"].shift(1).fillna(0).astype(int)
    return out


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
        slippage_mult = float(os.getenv("BACKTEST_SLIPPAGE_MULT", "1.0"))
        spread_cost = entry_price * self.costs['spread_pct']
        slippage_cost = entry_price * self.costs['slippage_pct'] * slippage_mult
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
        max_lot = self.config.MAX_LOT_SIZE.get(self.ticker_type, 1.0)
        lot_size = min(max(risk_amount / (sl_distance * self.contract_size), 0.01), max_lot)

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

    def simulate_trade(self, entry_time, entry_price, direction, sl_price, tp_price, df_slice):
        """Compatibility wrapper around the vectorized simulator."""
        if df_slice is None or df_slice.empty:
            return None

        trade = self.simulate_trade_vectorized(
            entry_idx=0,
            entry_price=entry_price,
            direction=direction,
            sl_price=sl_price,
            tp_price=tp_price,
            highs=df_slice['high'].to_numpy(),
            lows=df_slice['low'].to_numpy(),
            closes=df_slice['close'].to_numpy(),
            times=df_slice.index.to_numpy(),
        )
        if trade:
            trade['entry_time'] = entry_time
        return trade

    def run(self):
        """Run the backtest"""
        print(f"[{self.symbol}] Starting backtest...")

        # Get data
        df = self.get_historical_data()
        if df is None:
            return None

        # ATR is shifted by 1 bar to avoid using current bar info at entry.
        df['atr'] = calculate_atr(df, self.config.ATR_PERIOD).shift(1)

        # Generate signals
        if self.config.USE_ML_GATE:
            df = generate_ml_gated_signals(df, self.costs, self.config)
        else:
            df = generate_signals(df)

        # Drop NaN rows
        df = df.dropna()

        print(f"[{self.symbol}] Data loaded: {len(df)} candles")

        # Iterate through signals (single active trade per symbol)
        i = 0
        while i < len(df) - 1:
            row = df.iloc[i]

            signal = row['signal']
            if signal == 0:
                i += 1
                continue

            # Entry setup (next actionable bar open after shifted signal).
            entry_time = df.index[i]
            entry_price = row['open']
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
                # Skip ahead to trade exit to avoid overlapping positions.
                exit_idx = int(df.index.searchsorted(pd.to_datetime(trade['exit_time'])))
                i = max(i + 1, exit_idx + 1)
                if self.max_drawdown_pct >= self.config.MAX_DRAWDOWN_STOP_PCT:
                    break
                continue

            i += 1

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
        ok, mt5_error, _ = initialize_mt5()
        if not ok:
            return {'symbol': symbol, 'error': f"MT5 init failed in worker: {mt5_error}"}
        engine = BacktestEngine(symbol)
        result = engine.run()
        mt5.shutdown()
        return result
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_all_backtests():
    """Run backtests for all tickers in parallel"""
    selected = list(BacktestConfig.TICKERS.keys())
    if len(sys.argv) > 1:
        selected = [s for s in sys.argv[1:] if s in BacktestConfig.TICKERS]
        if not selected:
            print("No valid symbols passed. Falling back to full ticker set.")
            selected = list(BacktestConfig.TICKERS.keys())

    print("=" * 70)
    print("SOVEREIGN BACKTEST ENGINE - No Mercy Edition")
    print("=" * 70)
    print(f"Tickers: {len(selected)}")
    print(f"Threads per ticker: {BacktestConfig.THREADS_PER_TICKER}")
    print(f"Lookback: {BacktestConfig.LOOKBACK_DAYS} days")
    print(f"Initial Balance: ${BacktestConfig.INITIAL_BALANCE:,.0f}")
    print("=" * 70)
    print()

    results = []

    with ThreadPoolExecutor(max_workers=min(BacktestConfig.MAX_WORKERS, len(selected))) as executor:
        futures = {executor.submit(run_single_backtest, symbol): symbol
                   for symbol in selected}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[{symbol}] Error: {e}")

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
