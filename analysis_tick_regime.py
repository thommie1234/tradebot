"""Tick-level regime detection — detect range vs trend from tick behavior.

Concept: count tick direction changes in the last N ticks before trade entry.
Many direction changes = choppy/range = trail strakker.
Few direction changes = trending = trail ruim.

Metric: "choppiness ratio" = direction_changes / total_ticks (0-1)
High ratio (>0.5) = range, Low ratio (<0.3) = trend.

Test: adjust trail_dist based on this ratio at time of trade entry.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['MT5_MODULE'] = 'MetaTrader5_FTMO'
import MetaTrader5_FTMO as mt5
from datetime import datetime, timedelta, timezone
import numpy as np

mt5.initialize(r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")

now = datetime.now(timezone.utc)
start = now - timedelta(days=3)
deals = mt5.history_deals_get(start, now)

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
            'symbol': d.symbol, 'dir': ent['dir'],
            'entry_price': ent['price'], 'volume': ent['volume'],
            'net': net, 'open_time': ent['time'], 'close_time': t,
            'day': t.strftime('%m-%d'),
        })

trades.sort(key=lambda x: x['open_time'])

tf_map = {'EURUSD': mt5.TIMEFRAME_M15, 'USDJPY': mt5.TIMEFRAME_M15,
          'GBPUSD': mt5.TIMEFRAME_M30, 'GBPAUD': mt5.TIMEFRAME_M30,
          'GBPCAD': mt5.TIMEFRAME_H1, 'NZDUSD': mt5.TIMEFRAME_M30,
          'FRA40.cash': mt5.TIMEFRAME_M30, 'US100.cash': mt5.TIMEFRAME_H1,
          'NVDA': mt5.TIMEFRAME_H1, 'XAUUSD': mt5.TIMEFRAME_M15}

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


def get_choppiness(symbol, at_time, lookback_minutes=30):
    """Get choppiness ratio from ticks before trade entry.

    Returns: ratio (0=pure trend, 1=pure chop), net_direction (-1 to +1)
    """
    ticks = mt5.copy_ticks_range(symbol,
                                  at_time - timedelta(minutes=lookback_minutes),
                                  at_time, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 20:
        return 0.5, 0  # unknown

    bids = np.array([t[1] for t in ticks])

    # Direction changes
    diffs = np.diff(bids)
    diffs = diffs[diffs != 0]  # remove no-change ticks
    if len(diffs) < 5:
        return 0.5, 0

    signs = np.sign(diffs)
    changes = np.sum(np.abs(np.diff(signs)) > 0)
    chop_ratio = changes / len(signs)

    # Net direction: how much of total movement is in one direction
    total_move = np.sum(np.abs(diffs))
    net_move = abs(bids[-1] - bids[0])
    efficiency = net_move / total_move if total_move > 0 else 0

    return chop_ratio, efficiency


def get_atr(symbol, at_time):
    tf = tf_map.get(symbol, mt5.TIMEFRAME_M30)
    bars = mt5.copy_rates_range(symbol, tf, at_time - timedelta(hours=24), at_time)
    if bars is None or len(bars) < 14:
        return 0.001
    highs = np.array([b[2] for b in bars[-14:]])
    lows = np.array([b[3] for b in bars[-14:]])
    closes = np.array([b[4] for b in bars[-15:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
    return float(np.mean(tr))


def simulate_trade(trade, atr, trail_dist_mult, be_mult, trail_act_mult, sl_mult):
    """Simulate single trade with given settings using tick data."""
    sym = trade['symbol']
    ticks = mt5.copy_ticks_range(sym, trade['open_time'],
                                  trade['close_time'] + timedelta(minutes=5), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 10:
        return trade['net']

    entry = trade['entry_price']
    is_buy = trade['dir'] == 'BUY'

    sl_dist = atr * sl_mult
    be_dist = atr * be_mult
    trail_act_dist = atr * trail_act_mult
    trail_dist = atr * trail_dist_mult

    sl_price = entry - sl_dist if is_buy else entry + sl_dist
    best = entry
    trail_on = False
    tsl = sl_price
    be_done = False

    for tick in ticks:
        bid, ask = tick[1], tick[2]
        c = bid if is_buy else ask
        fav = (c - entry) if is_buy else (entry - c)

        if is_buy and c > best: best = c
        if not is_buy and c < best: best = c

        if not be_done and fav >= be_dist:
            tsl = entry + (0.0001 if is_buy else -0.0001)
            be_done = True

        if not trail_on and fav >= trail_act_dist:
            trail_on = True
            tsl = max(tsl, c - trail_dist) if is_buy else min(tsl, c + trail_dist)

        if trail_on:
            ns = (c - trail_dist) if is_buy else (c + trail_dist)
            if is_buy and ns > tsl: tsl = ns
            if not is_buy and ns < tsl: tsl = ns

        if (is_buy and c <= tsl) or (not is_buy and c >= tsl):
            pnl_pts = (tsl - entry) if is_buy else (entry - tsl)
            si = mt5.symbol_info(sym)
            if si and si.trade_tick_size > 0:
                return (pnl_pts / si.trade_tick_size) * si.trade_tick_value * trade['volume']
            return pnl_pts * trade['volume']

    c = ticks[-1][1] if is_buy else ticks[-1][2]
    pnl_pts = (c - entry) if is_buy else (entry - c)
    si = mt5.symbol_info(sym)
    if si and si.trade_tick_size > 0:
        return (pnl_pts / si.trade_tick_size) * si.trade_tick_value * trade['volume']
    return pnl_pts * trade['volume']


# === PRECOMPUTE ===
print("Computing tick regime + ATR per trade...")
trade_data = []
for t in trades:
    atr = get_atr(t['symbol'], t['open_time'])

    # Test different lookback windows
    chop_10, eff_10 = get_choppiness(t['symbol'], t['open_time'], 10)
    chop_30, eff_30 = get_choppiness(t['symbol'], t['open_time'], 30)
    chop_60, eff_60 = get_choppiness(t['symbol'], t['open_time'], 60)

    trade_data.append({
        'trade': t, 'atr': atr,
        'chop_10': chop_10, 'eff_10': eff_10,
        'chop_30': chop_30, 'eff_30': eff_30,
        'chop_60': chop_60, 'eff_60': eff_60,
    })

print(f"Done. {len(trade_data)} trades")

# Show choppiness per trade
print(f"\n{'Time':>8s} {'Sym':>10s} {'Dir':>5s} {'Net':>8s} {'Chop10':>7s} {'Eff10':>6s} {'Chop30':>7s} {'Eff30':>6s} {'Day':>6s}")
print('-' * 75)
for td in trade_data:
    t = td['trade']
    print(f"{t['open_time'].strftime('%H:%M'):>8s} {t['symbol']:>10s} {t['dir']:>5s} {t['net']:+8.0f} "
          f"{td['chop_10']:7.3f} {td['eff_10']:6.3f} {td['chop_30']:7.3f} {td['eff_30']:6.3f} {t['day']:>6s}")

# === STRATEGIES ===
print(f"\n{'Strategy':<40s} {'P/L':>10s} {'vs Base':>10s}")
print('-' * 63)

baseline_pnl = 0

strategies = [
    ("Baseline", "baseline"),
    # Choppiness-based trail scaling (10min lookback)
    ("Chop10: trail *= (1-chop)", "chop10_linear"),
    ("Chop10: trail *= (1-chop)^2", "chop10_quad"),
    # 30min lookback
    ("Chop30: trail *= (1-chop)", "chop30_linear"),
    ("Chop30: trail *= (1-chop)^2", "chop30_quad"),
    # 60min lookback
    ("Chop60: trail *= (1-chop)", "chop60_linear"),
    # Efficiency-based (net move / total move)
    ("Eff30: trail *= eff", "eff30"),
    ("Eff30: trail *= max(0.3, eff)", "eff30_floor"),
    # Combined: chop + efficiency
    ("Chop30*Eff30: trail *= combo", "combo30"),
    # Stepped
    ("Chop30 > 0.5: trail 50%", "stepped_chop30"),
    ("Chop30 > 0.45: trail 60%", "stepped_chop30_45"),
    ("Eff30 < 0.3: trail 50%", "stepped_eff30"),
    # Dynamic BE
    ("Chop30 > 0.5: BE 50% faster", "chop_be"),
    # Full: trail + BE + trail_act
    ("Full chop30: all settings scale", "full_chop30"),
    # Hybrid: chop + after loss
    ("Chop30>0.45 + after loss: trail 50%", "hybrid"),
]

for name, mode in strategies:
    total_pnl = 0
    losses_per_sym_day = {}

    for td in trade_data:
        t = td['trade']
        atr = td['atr']
        cfg = sym_config.get(t['symbol'], {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3})

        key = t['symbol'] + '_' + t['day']
        if key not in losses_per_sym_day:
            losses_per_sym_day[key] = 0

        td_mult = cfg['trail_dist']
        be_mult = cfg['be']
        ta_mult = cfg['trail_act']
        sl_mult = cfg['sl']

        if mode == "baseline":
            pass
        elif mode == "chop10_linear":
            factor = max(0.3, 1.0 - td['chop_10'])
            td_mult *= factor
        elif mode == "chop10_quad":
            factor = max(0.3, (1.0 - td['chop_10']) ** 2)
            td_mult *= factor
        elif mode == "chop30_linear":
            factor = max(0.3, 1.0 - td['chop_30'])
            td_mult *= factor
        elif mode == "chop30_quad":
            factor = max(0.3, (1.0 - td['chop_30']) ** 2)
            td_mult *= factor
        elif mode == "chop60_linear":
            factor = max(0.3, 1.0 - td['chop_60'])
            td_mult *= factor
        elif mode == "eff30":
            factor = max(0.2, td['eff_30'])
            td_mult *= factor
        elif mode == "eff30_floor":
            factor = max(0.3, td['eff_30'])
            td_mult *= factor
        elif mode == "combo30":
            factor = max(0.3, td['eff_30'] * (1.0 - td['chop_30']))
            td_mult *= factor
        elif mode == "stepped_chop30":
            if td['chop_30'] > 0.5:
                td_mult *= 0.5
                ta_mult *= 0.5
        elif mode == "stepped_chop30_45":
            if td['chop_30'] > 0.45:
                td_mult *= 0.6
                ta_mult *= 0.6
        elif mode == "stepped_eff30":
            if td['eff_30'] < 0.3:
                td_mult *= 0.5
                ta_mult *= 0.5
        elif mode == "chop_be":
            if td['chop_30'] > 0.5:
                be_mult *= 0.5
        elif mode == "full_chop30":
            factor = max(0.3, 1.0 - td['chop_30'])
            td_mult *= factor
            be_mult *= factor
            ta_mult *= factor
        elif mode == "hybrid":
            is_chop = td['chop_30'] > 0.45
            had_loss = losses_per_sym_day.get(key, 0) >= 1
            if is_chop and had_loss:
                td_mult *= 0.5
                ta_mult *= 0.5

        pnl = simulate_trade(t, atr, td_mult, be_mult, ta_mult, sl_mult)
        total_pnl += pnl

        if pnl < 0:
            losses_per_sym_day[key] = losses_per_sym_day.get(key, 0) + 1

    if mode == "baseline":
        baseline_pnl = total_pnl

    diff = total_pnl - baseline_pnl
    marker = ' ***' if diff > 0 else ''
    print(f"{name:<40s} {total_pnl:+10.2f} {diff:+10.2f}{marker}")

# Per-day for promising strategies
print("\n=== Per dag ===")
for name, mode in [("Baseline", "baseline"), ("Hybrid", "hybrid"), ("Stepped chop30", "stepped_chop30")]:
    print(f"\n--- {name} ---")
    for day in sorted(set(td['trade']['day'] for td in trade_data)):
        day_data = [td for td in trade_data if td['trade']['day'] == day]
        day_pnl = 0
        losses_per_sym_day = {}
        for td in day_data:
            t = td['trade']
            atr = td['atr']
            cfg = sym_config.get(t['symbol'], {'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'sl': 0.3})
            key = t['symbol'] + '_' + t['day']
            if key not in losses_per_sym_day: losses_per_sym_day[key] = 0

            td_m = cfg['trail_dist']
            ta_m = cfg['trail_act']

            if mode == "hybrid":
                if td['chop_30'] > 0.45 and losses_per_sym_day.get(key, 0) >= 1:
                    td_m *= 0.5; ta_m *= 0.5
            elif mode == "stepped_chop30":
                if td['chop_30'] > 0.5:
                    td_m *= 0.5; ta_m *= 0.5

            pnl = simulate_trade(t, atr, td_m, cfg['be'], ta_m, cfg['sl'])
            day_pnl += pnl
            if pnl < 0: losses_per_sym_day[key] = losses_per_sym_day.get(key, 0) + 1
        print(f"  {day}: ${day_pnl:+,.2f}")

mt5.shutdown()
