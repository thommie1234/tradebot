"""Test range detection + strakke trail filters on FTMO trade data.

Filters:
1. ATR ratio: current ATR < 70% of 20-period avg ATR → range mode
2. ADX < 20 → range mode
3. Consecutive losses: after 2 losses on same symbol → switch mode
4. BB width: narrow Bollinger Bands → range mode

In range mode: trail_distance halved (strakker)

Tests each filter solo, in pairs, and all combined.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['MT5_MODULE'] = 'MetaTrader5_FTMO'
import MetaTrader5_FTMO as mt5
from datetime import datetime, timedelta, timezone
import numpy as np
from collections import defaultdict

mt5.initialize(r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")

now = datetime.now(timezone.utc)
start = now - timedelta(days=3)
deals = mt5.history_deals_get(start, now)

# Build trades with entry info
trades = []
entries = {}
for d in (deals or []):
    if d.symbol == '':
        continue
    if d.entry == 0:
        entries[d.position_id] = {
            'symbol': d.symbol, 'dir': 'BUY' if d.type == 0 else 'SELL',
            'price': d.price, 'volume': d.volume,
            'time': datetime.fromtimestamp(d.time, tz=timezone.utc),
        }
    elif d.entry == 1:
        ent = entries.get(d.position_id)
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        net = d.profit + d.commission + d.swap
        trades.append({
            'symbol': d.symbol,
            'dir': ent['dir'] if ent else ('BUY' if d.type == 1 else 'SELL'),
            'entry_price': ent['price'] if ent else d.price,
            'exit_price': d.price,
            'volume': d.volume,
            'net': net,
            'profit': d.profit,
            'open_time': ent['time'] if ent else t,
            'close_time': t,
            'day': t.strftime('%m-%d'),
        })

trades.sort(key=lambda x: x['close_time'])

# Precompute indicators per symbol per trade time
tf_map = {'EURUSD': mt5.TIMEFRAME_M15, 'USDJPY': mt5.TIMEFRAME_M15,
          'GBPUSD': mt5.TIMEFRAME_M30, 'GBPAUD': mt5.TIMEFRAME_M30,
          'GBPCAD': mt5.TIMEFRAME_H1, 'NZDUSD': mt5.TIMEFRAME_M30,
          'FRA40.cash': mt5.TIMEFRAME_M30, 'US100.cash': mt5.TIMEFRAME_H1,
          'NVDA': mt5.TIMEFRAME_H1, 'XAUUSD': mt5.TIMEFRAME_M15}

def get_indicators(symbol, at_time):
    """Get ATR ratio, ADX, BB width at a given time."""
    tf = tf_map.get(symbol, mt5.TIMEFRAME_M30)
    bars = mt5.copy_rates_range(symbol, tf, at_time - timedelta(hours=48), at_time)
    if bars is None or len(bars) < 30:
        return None

    highs = np.array([b[2] for b in bars])
    lows = np.array([b[3] for b in bars])
    closes = np.array([b[4] for b in bars])

    # ATR
    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    if len(tr) < 20:
        return None

    current_atr = np.mean(tr[-5:])
    avg_atr = np.mean(tr[-20:])
    atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

    # ADX (simplified)
    plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1] - lows[1:], 0)

    mask = plus_dm > minus_dm
    plus_dm[~mask] = 0
    minus_dm[mask] = 0

    atr_14 = np.convolve(tr, np.ones(14)/14, mode='valid')
    if len(atr_14) < 1:
        return None

    plus_di = np.mean(plus_dm[-14:]) / np.mean(tr[-14:]) * 100 if np.mean(tr[-14:]) > 0 else 0
    minus_di = np.mean(minus_dm[-14:]) / np.mean(tr[-14:]) * 100 if np.mean(tr[-14:]) > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    adx = dx  # simplified, normally smoothed

    # Bollinger Band width
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    bb_width = (2 * std20) / sma20 * 100 if sma20 > 0 else 0

    # Historical BB width for comparison
    if len(closes) >= 40:
        hist_bb = []
        for i in range(20, len(closes)):
            s = np.mean(closes[i-20:i])
            sd = np.std(closes[i-20:i])
            hist_bb.append((2 * sd) / s * 100 if s > 0 else 0)
        avg_bb = np.mean(hist_bb) if hist_bb else bb_width
    else:
        avg_bb = bb_width

    bb_ratio = bb_width / avg_bb if avg_bb > 0 else 1.0

    return {
        'atr_ratio': atr_ratio,
        'adx': adx,
        'bb_width': bb_width,
        'bb_ratio': bb_ratio,
    }

# Precompute indicators for all trades
print("Computing indicators...")
trade_indicators = []
for t in trades:
    ind = get_indicators(t['symbol'], t['open_time'])
    trade_indicators.append(ind)
print(f"Done. {len(trades)} trades, {sum(1 for i in trade_indicators if i)} with indicators")

# === SIMULATION ===

def simulate(trades, indicators, is_range_fn, name):
    """Simulate with strakker trail in range mode.

    In range mode: we assume the trade captures 50% less profit (strakker trail exits earlier).
    For losses: same SL, but fewer trades (skip after detection).
    """
    state = defaultdict(lambda: {'losses': 0, 'trades': 0, 'pnl': 0})
    total_pnl = 0
    kept = 0
    skipped = 0

    for t, ind in zip(trades, indicators):
        key = t['symbol'] + '_' + t['day']

        range_mode = is_range_fn(ind, state[key])

        if range_mode:
            # In range: skip if already losing on this symbol today
            if state[key]['losses'] >= 2:
                skipped += 1
                # Update state but don't count P/L
                state[key]['trades'] += 1
                if t['net'] < 0:
                    state[key]['losses'] += 1
                state[key]['pnl'] += t['net']
                continue

            # In range: if trade is a winner, assume strakker trail captures 60% of profit
            if t['net'] > 0:
                adjusted = t['net'] * 0.6
            else:
                adjusted = t['net']  # losses stay same (SL is SL)
            total_pnl += adjusted
        else:
            total_pnl += t['net']

        kept += 1
        state[key]['trades'] += 1
        if t['net'] < 0:
            state[key]['losses'] += 1
        state[key]['pnl'] += t['net']

    return name, kept, skipped, total_pnl

baseline = sum(t['net'] for t in trades)
print(f"\nBaseline: {len(trades)} trades, ${baseline:+,.2f}")
print()

# Define range detection functions
def f_none(ind, state):
    return False

def f_atr(ind, state):
    if ind is None: return False
    return ind['atr_ratio'] < 0.7

def f_adx(ind, state):
    if ind is None: return False
    return ind['adx'] < 20

def f_bb(ind, state):
    if ind is None: return False
    return ind['bb_ratio'] < 0.7

def f_consec(ind, state):
    return state['losses'] >= 2

def f_atr_consec(ind, state):
    return f_atr(ind, state) or f_consec(ind, state)

def f_adx_consec(ind, state):
    return f_adx(ind, state) or f_consec(ind, state)

def f_bb_consec(ind, state):
    return f_bb(ind, state) or f_consec(ind, state)

def f_atr_adx(ind, state):
    return f_atr(ind, state) or f_adx(ind, state)

def f_atr_bb(ind, state):
    return f_atr(ind, state) or f_bb(ind, state)

def f_adx_bb(ind, state):
    return f_adx(ind, state) or f_bb(ind, state)

def f_all_indicators(ind, state):
    return f_atr(ind, state) or f_adx(ind, state) or f_bb(ind, state)

def f_all(ind, state):
    return f_atr(ind, state) or f_adx(ind, state) or f_bb(ind, state) or f_consec(ind, state)

filters = [
    (f_none, "Baseline (geen filter)"),
    (f_atr, "ATR ratio < 0.7"),
    (f_adx, "ADX < 20"),
    (f_bb, "BB width < 0.7x avg"),
    (f_consec, "Na 2 losses/sym/dag"),
    (f_atr_consec, "ATR + 2 losses"),
    (f_adx_consec, "ADX + 2 losses"),
    (f_bb_consec, "BB + 2 losses"),
    (f_atr_adx, "ATR + ADX"),
    (f_atr_bb, "ATR + BB"),
    (f_adx_bb, "ADX + BB"),
    (f_all_indicators, "ATR + ADX + BB"),
    (f_all, "Alle filters"),
]

print(f"{'Filter':<25s} {'Kept':>5s} {'Skip':>5s} {'P/L':>10s} {'vs Base':>10s}")
print('-' * 60)

for func, name in filters:
    result = simulate(trades, trade_indicators, func, name)
    diff = result[3] - baseline
    print(f"{result[0]:<25s} {result[1]:5d} {result[2]:5d} {result[3]:+10.2f} {diff:+10.2f}")

# Per-day breakdown for best filters
print("\n=== Per dag breakdown ===")
for func, name in [(f_consec, "2 losses"), (f_adx_consec, "ADX+2loss"), (f_all, "Alle")]:
    print(f"\n--- {name} ---")
    for day in sorted(set(t['day'] for t in trades)):
        day_idx = [i for i, t in enumerate(trades) if t['day'] == day]
        day_trades = [trades[i] for i in day_idx]
        day_ind = [trade_indicators[i] for i in day_idx]

        base_pnl = sum(t['net'] for t in day_trades)
        _, _, _, filt_pnl = simulate(day_trades, day_ind, func, name)
        diff = filt_pnl - base_pnl
        print(f"  {day}: base ${base_pnl:+,.2f} -> ${filt_pnl:+,.2f} (diff ${diff:+,.2f})")

mt5.shutdown()
