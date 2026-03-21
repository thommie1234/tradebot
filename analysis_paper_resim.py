"""Re-simulate paper trades with LIVE exit settings using MT5 tick data.

Paper trades used various exit params. We re-run them all with the
actual live config (SL=0.30, BE=0.50, Trail=0.75/0.38) to get
accurate P/L for scenario analysis.

Then analyze all 4 scenarios:
1. Per-symbol edge differences
2. Signal flip frequency vs performance
3. RRR in range vs trend
4. Session/time-of-day patterns
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['MT5_MODULE'] = 'MetaTrader5_FTMO'
import MetaTrader5_FTMO as mt5
import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
from collections import defaultdict

mt5.initialize(r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")

# Live exit settings
LIVE_CONFIG = {
    'EURUSD': {'sl': 0.3, 'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_M15},
    'GBPUSD': {'sl': 0.3, 'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_M30},
    'USDJPY': {'sl': 0.3, 'be': 0.15, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_M15},
    'GBPCAD': {'sl': 0.3, 'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_H1},
    'GBPAUD': {'sl': 0.3, 'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_M30},
    'NZDUSD': {'sl': 0.3, 'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_M30},
    'FRA40.cash': {'sl': 0.3, 'be': 0.5, 'trail_act': 0.75, 'trail_dist': 0.38, 'tf': mt5.TIMEFRAME_M30},
    'US100.cash': {'sl': 0.3, 'be': 1.5, 'trail_act': 2.0, 'trail_dist': 1.0, 'tf': mt5.TIMEFRAME_H1},
    'NVDA': {'sl': 0.5, 'be': 0.5, 'trail_act': 1.5, 'trail_dist': 0.5, 'tf': mt5.TIMEFRAME_H1},
}

PORTFOLIO = list(LIVE_CONFIG.keys())


def simulate_with_ticks(symbol, direction, entry_price, open_time, atr, cfg):
    """Re-simulate a trade with live exit settings using tick data."""
    ticks = mt5.copy_ticks_range(symbol, open_time,
                                  open_time + timedelta(hours=6), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 10:
        return None, None, None

    is_buy = direction == 'BUY'
    sl_dist = atr * cfg['sl']
    be_dist = atr * cfg['be']
    trail_act_dist = atr * cfg['trail_act']
    trail_dist = atr * cfg['trail_dist']

    sl_price = entry_price - sl_dist if is_buy else entry_price + sl_dist
    best = entry_price
    trail_on = False
    tsl = sl_price
    be_done = False
    exit_reason = 'TIME'

    for tick in ticks:
        bid, ask = tick[1], tick[2]
        c = bid if is_buy else ask
        fav = (c - entry_price) if is_buy else (entry_price - c)

        if is_buy and c > best: best = c
        if not is_buy and c < best: best = c

        if not be_done and fav >= be_dist:
            tsl = entry_price + (0.00001 if is_buy else -0.00001)
            be_done = True

        if not trail_on and fav >= trail_act_dist:
            trail_on = True
            tsl = max(tsl, c - trail_dist) if is_buy else min(tsl, c + trail_dist)

        if trail_on:
            ns = (c - trail_dist) if is_buy else (c + trail_dist)
            if is_buy and ns > tsl: tsl = ns
            if not is_buy and ns < tsl: tsl = ns

        if (is_buy and c <= tsl) or (not is_buy and c >= tsl):
            pnl_pts = (tsl - entry_price) if is_buy else (entry_price - tsl)
            exit_reason = 'TRAIL' if trail_on else ('BE' if be_done else 'SL')
            mfe = (best - entry_price) if is_buy else (entry_price - best)
            return pnl_pts, exit_reason, mfe

    # Timeout
    c = ticks[-1][1] if is_buy else ticks[-1][2]
    pnl_pts = (c - entry_price) if is_buy else (entry_price - c)
    mfe = (best - entry_price) if is_buy else (entry_price - best)
    return pnl_pts, 'TIME', mfe


def get_adx_at_time(symbol, at_time, tf):
    """Get ADX value at trade entry time."""
    bars = mt5.copy_rates_range(symbol, tf, at_time - timedelta(hours=48), at_time)
    if bars is None or len(bars) < 30:
        return None
    highs = np.array([b[2] for b in bars])
    lows = np.array([b[3] for b in bars])
    closes = np.array([b[4] for b in bars])

    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    if len(tr) < 14:
        return None

    plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1] - lows[1:], 0)
    mask = plus_dm > minus_dm
    pdm = plus_dm.copy(); pdm[~mask] = 0
    mdm = minus_dm.copy(); mdm[mask] = 0
    plus_di = np.mean(pdm[-14:]) / np.mean(tr[-14:]) * 100 if np.mean(tr[-14:]) > 0 else 0
    minus_di = np.mean(mdm[-14:]) / np.mean(tr[-14:]) * 100 if np.mean(tr[-14:]) > 0 else 0
    adx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0

    current_atr = np.mean(tr[-5:])
    avg_atr = np.mean(tr[-20:])
    atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

    return {'adx': adx, 'atr_ratio': atr_ratio, 'atr': float(np.mean(tr[-14:]))}


# Load paper trades for portfolio symbols
print("Loading paper trades...")
conn = sqlite3.connect('bf/audit/paper_trades.db')
paper_trades = conn.execute('''
    SELECT symbol, direction, entry_price, timestamp, atr, timeframe, confidence, pnl
    FROM paper_trades
    WHERE status != 'OPEN' AND ABS(pnl) < 50000
      AND symbol IN ({})
    ORDER BY timestamp
'''.format(','.join(f'"{s}"' for s in PORTFOLIO))).fetchall()
conn.close()

print(f"Loaded {len(paper_trades)} paper trades")

# Filter to matching timeframes only
filtered = []
for t in paper_trades:
    sym, direction, entry, ts, atr, tf, conf, orig_pnl = t
    cfg = LIVE_CONFIG.get(sym)
    if not cfg:
        continue
    # Match timeframe
    tf_match = {
        'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
    }
    if tf_match.get(tf) != cfg['tf']:
        continue
    filtered.append(t)

print(f"Filtered to {len(filtered)} trades on matching timeframes")

# Re-simulate with ticks + collect indicators
print("Re-simulating with live exits + tick data...")
results = []
count = 0
for t in filtered:
    sym, direction, entry, ts, atr, tf, conf, orig_pnl = t
    cfg = LIVE_CONFIG[sym]

    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except:
        continue

    if atr <= 0:
        continue

    pnl_pts, exit_reason, mfe = simulate_with_ticks(sym, direction, entry, dt, atr, cfg)
    if pnl_pts is None:
        continue

    # Get indicators
    indicators = get_adx_at_time(sym, dt, cfg['tf'])
    if not indicators:
        continue

    # Convert to USD (approximation using tick value)
    si = mt5.symbol_info(sym)
    if si and si.trade_tick_size > 0:
        pnl_usd = (pnl_pts / si.trade_tick_size) * si.trade_tick_value
    else:
        pnl_usd = pnl_pts

    hour = dt.hour
    session = 'asian' if hour < 7 else ('london' if hour < 13 else ('ny' if hour < 21 else 'late'))

    results.append({
        'symbol': sym, 'direction': direction, 'entry': entry,
        'pnl_pts': pnl_pts, 'pnl_usd': pnl_usd, 'mfe': mfe,
        'exit_reason': exit_reason, 'confidence': conf,
        'adx': indicators['adx'], 'atr_ratio': indicators['atr_ratio'],
        'atr': indicators['atr'],
        'hour': hour, 'session': session,
        'day': dt.strftime('%m-%d'), 'time': dt,
    })

    count += 1
    if count % 50 == 0:
        print(f"  {count}/{len(filtered)} done...")

print(f"Done. {len(results)} re-simulated trades")

# === SCENARIO ANALYSES ===

print(f"\n{'='*70}")
print("SCENARIO 1: Per-symbool edge")
print(f"{'='*70}")
print(f"{'Symbol':12s} {'Trades':>6s} {'WR%':>5s} {'P/L':>10s} {'AvgW':>8s} {'AvgL':>8s} {'RRR':>5s} {'PF':>5s}")
print('-' * 62)

sym_data = defaultdict(list)
for r in results:
    sym_data[r['symbol']].append(r)

for sym in PORTFOLIO:
    trades = sym_data.get(sym, [])
    if not trades:
        continue
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    wr = len(wins) / len(trades) * 100
    avg_w = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
    avg_l = np.mean([t['pnl_usd'] for t in losses]) if losses else 0
    rrr = abs(avg_w / avg_l) if avg_l != 0 else 0
    pf = sum(t['pnl_usd'] for t in wins) / abs(sum(t['pnl_usd'] for t in losses)) if losses else 0
    total = sum(t['pnl_usd'] for t in trades)
    print(f"{sym:12s} {len(trades):6d} {wr:5.1f} {total:+10.2f} {avg_w:+8.2f} {avg_l:+8.2f} {rrr:5.2f} {pf:5.2f}")

print(f"\n{'='*70}")
print("SCENARIO 2: Signal stability (direction flips)")
print(f"{'='*70}")

# Count consecutive same-direction signals per symbol
flip_results = defaultdict(lambda: {'stable': [], 'flipped': []})
prev_dir = {}
for r in sorted(results, key=lambda x: x['time']):
    sym = r['symbol']
    if sym in prev_dir and prev_dir[sym] != r['direction']:
        flip_results[sym]['flipped'].append(r)
    else:
        flip_results[sym]['stable'].append(r)
    prev_dir[sym] = r['direction']

print(f"{'Symbol':12s} {'Stable WR':>10s} {'Stable PnL':>10s} {'Flip WR':>10s} {'Flip PnL':>10s}")
print('-' * 55)
total_stable = []
total_flip = []
for sym in PORTFOLIO:
    s = flip_results.get(sym, {'stable': [], 'flipped': []})
    if not s['stable'] and not s['flipped']:
        continue
    s_wr = len([t for t in s['stable'] if t['pnl_usd'] > 0]) / len(s['stable']) * 100 if s['stable'] else 0
    f_wr = len([t for t in s['flipped'] if t['pnl_usd'] > 0]) / len(s['flipped']) * 100 if s['flipped'] else 0
    s_pnl = sum(t['pnl_usd'] for t in s['stable'])
    f_pnl = sum(t['pnl_usd'] for t in s['flipped'])
    total_stable.extend(s['stable'])
    total_flip.extend(s['flipped'])
    print(f"{sym:12s} {s_wr:9.1f}% {s_pnl:+10.2f} {f_wr:9.1f}% {f_pnl:+10.2f}")
s_wr = len([t for t in total_stable if t['pnl_usd'] > 0]) / len(total_stable) * 100 if total_stable else 0
f_wr = len([t for t in total_flip if t['pnl_usd'] > 0]) / len(total_flip) * 100 if total_flip else 0
print(f"{'TOTAAL':12s} {s_wr:9.1f}% {sum(t['pnl_usd'] for t in total_stable):+10.2f} {f_wr:9.1f}% {sum(t['pnl_usd'] for t in total_flip):+10.2f}")

print(f"\n{'='*70}")
print("SCENARIO 3: RRR in range vs trend (ADX)")
print(f"{'='*70}")

for label, check in [('Range (ADX<20)', lambda r: r['adx'] < 20),
                      ('Mild trend (20-30)', lambda r: 20 <= r['adx'] < 30),
                      ('Strong trend (30+)', lambda r: r['adx'] >= 30)]:
    subset = [r for r in results if check(r)]
    if not subset:
        print(f"{label}: no trades")
        continue
    wins = [t for t in subset if t['pnl_usd'] > 0]
    losses = [t for t in subset if t['pnl_usd'] <= 0]
    wr = len(wins) / len(subset) * 100
    avg_w = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
    avg_l = np.mean([t['pnl_usd'] for t in losses]) if losses else 0
    rrr = abs(avg_w / avg_l) if avg_l != 0 else 0
    total = sum(t['pnl_usd'] for t in subset)
    print(f"{label:25s} {len(subset):4d} trades  WR={wr:.1f}%  AvgW={avg_w:+.2f}  AvgL={avg_l:+.2f}  RRR={rrr:.2f}  P/L={total:+.2f}")

# Also by ATR ratio
print()
for label, check in [('Low vol (ATR<0.7x)', lambda r: r['atr_ratio'] < 0.7),
                      ('Normal (0.7-1.3x)', lambda r: 0.7 <= r['atr_ratio'] < 1.3),
                      ('High vol (>1.3x)', lambda r: r['atr_ratio'] >= 1.3)]:
    subset = [r for r in results if check(r)]
    if not subset:
        print(f"{label}: no trades")
        continue
    wins = [t for t in subset if t['pnl_usd'] > 0]
    losses = [t for t in subset if t['pnl_usd'] <= 0]
    wr = len(wins) / len(subset) * 100
    avg_w = np.mean([t['pnl_usd'] for t in wins]) if wins else 0
    avg_l = np.mean([t['pnl_usd'] for t in losses]) if losses else 0
    rrr = abs(avg_w / avg_l) if avg_l != 0 else 0
    total = sum(t['pnl_usd'] for t in subset)
    print(f"{label:25s} {len(subset):4d} trades  WR={wr:.1f}%  AvgW={avg_w:+.2f}  AvgL={avg_l:+.2f}  RRR={rrr:.2f}  P/L={total:+.2f}")

print(f"\n{'='*70}")
print("SCENARIO 4: Session/tijd analyse")
print(f"{'='*70}")

for label, check in [('Asian (00-07)', lambda r: r['hour'] < 7),
                      ('London (07-13)', lambda r: 7 <= r['hour'] < 13),
                      ('NY overlap (13-17)', lambda r: 13 <= r['hour'] < 17),
                      ('NY (17-21)', lambda r: 17 <= r['hour'] < 21),
                      ('Late (21-00)', lambda r: r['hour'] >= 21)]:
    subset = [r for r in results if check(r)]
    if not subset:
        print(f"{label}: no trades")
        continue
    wins = [t for t in subset if t['pnl_usd'] > 0]
    losses = [t for t in subset if t['pnl_usd'] <= 0]
    wr = len(wins) / len(subset) * 100
    total = sum(t['pnl_usd'] for t in subset)
    avg = total / len(subset)
    print(f"{label:25s} {len(subset):4d} trades  WR={wr:.1f}%  P/L={total:+.2f}  Avg={avg:+.2f}")

# Per hour detail
print(f"\nPer uur:")
print(f"{'Uur':>4s} {'Trades':>6s} {'WR%':>5s} {'P/L':>10s} {'Avg':>8s}")
print('-' * 38)
for h in range(24):
    subset = [r for r in results if r['hour'] == h]
    if not subset:
        continue
    wins = len([t for t in subset if t['pnl_usd'] > 0])
    wr = wins / len(subset) * 100
    total = sum(t['pnl_usd'] for t in subset)
    avg = total / len(subset)
    bar = '+' * int(max(0, total) / 5) + '-' * int(max(0, -total) / 5)
    print(f"{h:4d} {len(subset):6d} {wr:5.1f} {total:+10.2f} {avg:+8.2f}  {bar}")

# Confidence analysis
print(f"\n{'='*70}")
print("BONUS: Confidence vs Performance")
print(f"{'='*70}")
for label, check in [('Low conf (0.55-0.62)', lambda r: 0.55 <= (r['confidence'] or 0) < 0.62),
                      ('Med conf (0.62-0.70)', lambda r: 0.62 <= (r['confidence'] or 0) < 0.70),
                      ('High conf (0.70-0.80)', lambda r: 0.70 <= (r['confidence'] or 0) < 0.80),
                      ('Very high (0.80+)', lambda r: (r['confidence'] or 0) >= 0.80)]:
    subset = [r for r in results if check(r)]
    if not subset:
        continue
    wins = [t for t in subset if t['pnl_usd'] > 0]
    wr = len(wins) / len(subset) * 100
    total = sum(t['pnl_usd'] for t in subset)
    print(f"{label:25s} {len(subset):4d} trades  WR={wr:.1f}%  P/L={total:+.2f}")

mt5.shutdown()
