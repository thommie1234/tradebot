"""Dynamic trailing stop — adjust trail distance based on market regime.

Instead of skipping trades, we adjust the trail distance:
- Trending (high ADX/ATR/BB): wide trail (current settings)
- Ranging (low ADX/ATR/BB): tight trail (capture quick, give back less)

Simulate with tick data on all trades from last 2 days.
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

# Build trades
entries = {}
trades = []
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
        if not ent:
            continue
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        net = d.profit + d.commission + d.swap
        trades.append({
            'symbol': d.symbol,
            'dir': ent['dir'],
            'entry_price': ent['price'],
            'exit_price': d.price,
            'volume': ent['volume'],
            'net': net,
            'open_time': ent['time'],
            'close_time': t,
            'day': t.strftime('%m-%d'),
        })

trades.sort(key=lambda x: x['open_time'])

tf_map = {'EURUSD': mt5.TIMEFRAME_M15, 'USDJPY': mt5.TIMEFRAME_M15,
          'GBPUSD': mt5.TIMEFRAME_M30, 'GBPAUD': mt5.TIMEFRAME_M30,
          'GBPCAD': mt5.TIMEFRAME_H1, 'NZDUSD': mt5.TIMEFRAME_M30,
          'FRA40.cash': mt5.TIMEFRAME_M30, 'US100.cash': mt5.TIMEFRAME_H1,
          'NVDA': mt5.TIMEFRAME_H1, 'XAUUSD': mt5.TIMEFRAME_M15}

# Config per symbol (current settings)
sym_config = {
    'EURUSD': {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'GBPUSD': {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'USDJPY': {'be': 0.15, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'GBPCAD': {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'GBPAUD': {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'NZDUSD': {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'FRA40.cash': {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3},
    'US100.cash': {'be': 1.5, 'trail_act': 2.0, 'trail_dist': 1.0, 'sl': 0.3},
    'NVDA': {'be': 0.5, 'trail_act': 1.5, 'trail_dist': 0.5, 'sl': 0.5},
    'XAUUSD': {'be': 1.07, 'trail_act': 2.0, 'trail_dist': 0.94, 'sl': 0.5},
}


def get_regime(symbol, at_time):
    """Returns regime score 0-1 (0=strong range, 1=strong trend)."""
    tf = tf_map.get(symbol, mt5.TIMEFRAME_M30)
    bars = mt5.copy_rates_range(symbol, tf, at_time - timedelta(hours=48), at_time)
    if bars is None or len(bars) < 30:
        return 0.5  # unknown, use default

    highs = np.array([b[2] for b in bars])
    lows = np.array([b[3] for b in bars])
    closes = np.array([b[4] for b in bars])

    # ATR ratio
    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    current_atr = np.mean(tr[-5:])
    avg_atr = np.mean(tr[-20:])
    atr_score = min(1.0, (current_atr / avg_atr) if avg_atr > 0 else 0.5)

    # ADX simplified
    plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1] - lows[1:], 0)
    mask = plus_dm > minus_dm
    plus_dm_clean = plus_dm.copy(); plus_dm_clean[~mask] = 0
    minus_dm_clean = minus_dm.copy(); minus_dm_clean[mask] = 0

    plus_di = np.mean(plus_dm_clean[-14:]) / np.mean(tr[-14:]) * 100 if np.mean(tr[-14:]) > 0 else 0
    minus_di = np.mean(minus_dm_clean[-14:]) / np.mean(tr[-14:]) * 100 if np.mean(tr[-14:]) > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    adx_score = min(1.0, dx / 40)  # 0 at ADX=0, 1 at ADX=40+

    # BB width ratio
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    bb_width = (2 * std20) / sma20 if sma20 > 0 else 0
    hist_bb = []
    for i in range(20, len(closes)):
        s = np.mean(closes[i-20:i])
        sd = np.std(closes[i-20:i])
        hist_bb.append((2 * sd) / s if s > 0 else 0)
    avg_bb = np.mean(hist_bb) if hist_bb else bb_width
    bb_score = min(1.0, (bb_width / avg_bb) if avg_bb > 0 else 0.5)

    # Combined regime score (0=range, 1=trend)
    return (atr_score * 0.4 + adx_score * 0.35 + bb_score * 0.25)


def simulate_trade_with_trail(trade, atr, trail_dist_atr, be_atr, trail_act_atr, sl_atr):
    """Simulate a single trade with given trail settings using tick data."""
    sym = trade['symbol']
    ticks = mt5.copy_ticks_range(sym, trade['open_time'],
                                  trade['close_time'] + timedelta(minutes=5), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 10:
        return trade['net']  # fallback to actual

    entry = trade['entry_price']
    is_buy = trade['dir'] == 'BUY'

    sl_dist = atr * sl_atr
    be_dist = atr * be_atr
    trail_act_dist = atr * trail_act_atr
    trail_dist = atr * trail_dist_atr

    sl_price = entry - sl_dist if is_buy else entry + sl_dist
    best = entry
    trail_on = False
    tsl = sl_price
    be_done = False

    for tick in ticks:
        bid, ask = tick[1], tick[2]
        c = bid if is_buy else ask
        fav = (c - entry) if is_buy else (entry - c)

        if is_buy:
            if c > best: best = c
        else:
            if c < best: best = c

        if not be_done and fav >= be_dist:
            tsl = entry + (0.0001 if is_buy else -0.0001)
            be_done = True

        if not trail_on and fav >= trail_act_dist:
            trail_on = True
            if is_buy:
                tsl = max(tsl, c - trail_dist)
            else:
                tsl = min(tsl, c + trail_dist)

        if trail_on:
            if is_buy:
                ns = c - trail_dist
                if ns > tsl: tsl = ns
            else:
                ns = c + trail_dist
                if ns < tsl: tsl = ns

        # Check SL/trail hit
        if is_buy and c <= tsl:
            pnl_pts = tsl - entry
            si = mt5.symbol_info(sym)
            if si and si.trade_tick_size > 0:
                return (pnl_pts / si.trade_tick_size) * si.trade_tick_value * trade['volume']
            return pnl_pts * trade['volume']
        elif not is_buy and c >= tsl:
            pnl_pts = entry - tsl
            si = mt5.symbol_info(sym)
            if si and si.trade_tick_size > 0:
                return (pnl_pts / si.trade_tick_size) * si.trade_tick_value * trade['volume']
            return pnl_pts * trade['volume']

    # Still open at end — use last tick
    c = ticks[-1][1] if is_buy else ticks[-1][2]
    pnl_pts = (c - entry) if is_buy else (entry - c)
    si = mt5.symbol_info(sym)
    if si and si.trade_tick_size > 0:
        return (pnl_pts / si.trade_tick_size) * si.trade_tick_value * trade['volume']
    return pnl_pts * trade['volume']


# Precompute ATR per trade
print("Computing ATR and regime per trade...")
trade_data = []
for t in trades:
    tf = tf_map.get(t['symbol'], mt5.TIMEFRAME_M30)
    bars = mt5.copy_rates_range(t['symbol'], tf, t['open_time'] - timedelta(hours=24), t['open_time'])
    if bars is not None and len(bars) >= 14:
        highs = np.array([b[2] for b in bars[-14:]])
        lows = np.array([b[3] for b in bars[-14:]])
        closes = np.array([b[4] for b in bars[-15:-1]])
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
        atr = float(np.mean(tr))
    else:
        atr = 0.001

    regime = get_regime(t['symbol'], t['open_time'])
    trade_data.append({'trade': t, 'atr': atr, 'regime': regime})

print(f"Done. {len(trade_data)} trades")

# === TEST STRATEGIES ===

strategies = [
    ("Baseline (huidige settings)", None),
    ("Dynamic: trail_dist * regime", "linear"),
    ("Dynamic: trail_dist * regime^2", "quadratic"),
    ("Stepped: range<0.4 → 50% trail", "stepped_50"),
    ("Stepped: range<0.4 → 40% trail", "stepped_40"),
    ("Stepped: range<0.3 → 50% trail", "stepped_30_50"),
    ("Dynamic BE: range → BE sneller", "dynamic_be"),
    ("Full dynamic: BE + trail + SL", "full_dynamic"),
]

print(f"\n{'Strategy':<35s} {'P/L':>10s} {'vs Base':>10s}")
print('-' * 58)

baseline_pnl = 0
for name, mode in strategies:
    total_pnl = 0

    for td in trade_data:
        t = td['trade']
        atr = td['atr']
        regime = td['regime']
        cfg = sym_config.get(t['symbol'], {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3})

        if mode is None:
            # Baseline
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'], cfg['be'], cfg['trail_act'], cfg['sl'])
        elif mode == "linear":
            # Trail distance scales linearly with regime (range=tight, trend=wide)
            factor = 0.3 + 0.7 * regime  # 0.3x at full range, 1.0x at full trend
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'] * factor, cfg['be'], cfg['trail_act'], cfg['sl'])
        elif mode == "quadratic":
            factor = 0.3 + 0.7 * (regime ** 2)
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'] * factor, cfg['be'], cfg['trail_act'], cfg['sl'])
        elif mode == "stepped_50":
            factor = 0.5 if regime < 0.4 else 1.0
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'] * factor, cfg['be'], cfg['trail_act'] * factor, cfg['sl'])
        elif mode == "stepped_40":
            factor = 0.4 if regime < 0.4 else 1.0
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'] * factor, cfg['be'], cfg['trail_act'] * factor, cfg['sl'])
        elif mode == "stepped_30_50":
            factor = 0.5 if regime < 0.3 else 1.0
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'] * factor, cfg['be'], cfg['trail_act'] * factor, cfg['sl'])
        elif mode == "dynamic_be":
            # Range: BE faster (lower threshold)
            be_factor = 0.4 + 0.6 * regime  # range: BE at 40% of normal
            pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'], cfg['be'] * be_factor, cfg['trail_act'], cfg['sl'])
        elif mode == "full_dynamic":
            # Everything scales with regime
            factor = 0.4 + 0.6 * regime
            pnl = simulate_trade_with_trail(t, atr,
                                             cfg['trail_dist'] * factor,
                                             cfg['be'] * factor,
                                             cfg['trail_act'] * factor,
                                             cfg['sl'])

        total_pnl += pnl

    if mode is None:
        baseline_pnl = total_pnl

    diff = total_pnl - baseline_pnl
    print(f"{name:<35s} {total_pnl:+10.2f} {diff:+10.2f}")

# Per day for best strategies
print("\n=== Per dag ===")
for name, mode in [("Baseline", None), ("Full dynamic", "full_dynamic"), ("Stepped 50%", "stepped_50")]:
    print(f"\n--- {name} ---")
    for day in sorted(set(td['trade']['day'] for td in trade_data)):
        day_data = [td for td in trade_data if td['trade']['day'] == day]
        day_pnl = 0
        for td in day_data:
            t = td['trade']
            atr = td['atr']
            regime = td['regime']
            cfg = sym_config.get(t['symbol'], {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3})

            if mode is None:
                pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist'], cfg['be'], cfg['trail_act'], cfg['sl'])
            elif mode == "full_dynamic":
                factor = 0.4 + 0.6 * regime
                pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist']*factor, cfg['be']*factor, cfg['trail_act']*factor, cfg['sl'])
            elif mode == "stepped_50":
                factor = 0.5 if regime < 0.4 else 1.0
                pnl = simulate_trade_with_trail(t, atr, cfg['trail_dist']*factor, cfg['be'], cfg['trail_act']*factor, cfg['sl'])
            day_pnl += pnl
        print(f"  {day}: ${day_pnl:+,.2f}")

mt5.shutdown()
