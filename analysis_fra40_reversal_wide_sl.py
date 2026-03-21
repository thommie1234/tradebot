"""FRA40 reversal — test wider SL with same trail settings."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import numpy as np

mt5.initialize(r"C:\Program Files\BrightFunded MT5 Terminal\terminal64.exe")

now = datetime.now(timezone.utc)
start = now - timedelta(days=30)
deals = mt5.history_deals_get(start, now)

exits = []
for d in (deals or []):
    if 'FRA40' not in d.symbol:
        continue
    if d.entry == 1:
        t = datetime.fromtimestamp(d.time, tz=timezone.utc)
        original_dir = 'SELL' if d.type == 0 else 'BUY'
        exits.append({
            'dir': original_dir,
            'exit': d.price,
            'close_time': t,
        })

print(f'FRA40 exits: {len(exits)}')

TICK_VALUE = 0.0115
LOTS = 30

# Fixed trail, varying SL
configs = [
    {'sl': 15, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 20, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 25, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 30, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 35, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 40, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 50, 'be': 5, 'trail_act': 8, 'trail_dist': 4},
    {'sl': 20, 'be': 8, 'trail_act': 10, 'trail_dist': 5},
    {'sl': 30, 'be': 8, 'trail_act': 10, 'trail_dist': 5},
    {'sl': 40, 'be': 8, 'trail_act': 10, 'trail_dist': 5},
    {'sl': 50, 'be': 10, 'trail_act': 12, 'trail_dist': 6},
]

print(f'\n{"SL":>4s} {"BE":>4s} {"Trail":>8s} | {"WR%":>5s} {"Total":>8s} {"Avg":>6s} {"AvgW":>6s} {"AvgL":>6s} | {"SL":>3s} {"BE":>3s} {"TR":>3s} {"TM":>3s}')
print('-' * 75)

for cfg in configs:
    results = []

    for e in exits:
        rev_dir = 'SELL' if e['dir'] == 'BUY' else 'BUY'

        ticks = mt5.copy_ticks_range('FRA40.cash', e['close_time'],
                                      e['close_time'] + timedelta(hours=3), mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) < 20:
            continue

        entry = e['exit']
        sl = entry + cfg['sl'] if rev_dir == 'SELL' else entry - cfg['sl']
        best_price = entry
        trailing_active = False
        trailing_sl = sl
        be_done = False
        exit_price = None
        exit_reason = None

        for tick in ticks:
            bid, ask = tick[1], tick[2]

            if rev_dir == 'BUY':
                current = bid
                favorable = current - entry

                if current > best_price:
                    best_price = current

                if not be_done and favorable >= cfg['be']:
                    trailing_sl = entry + 0.5
                    be_done = True

                if not trailing_active and favorable >= cfg['trail_act']:
                    trailing_active = True
                    trailing_sl = max(trailing_sl, current - cfg['trail_dist'])

                if trailing_active:
                    new_sl = current - cfg['trail_dist']
                    if new_sl > trailing_sl:
                        trailing_sl = new_sl

                if current <= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = 'TR' if trailing_active else ('BE' if be_done else 'SL')
                    break
            else:
                current = ask
                favorable = entry - current

                if current < best_price:
                    best_price = current

                if not be_done and favorable >= cfg['be']:
                    trailing_sl = entry - 0.5
                    be_done = True

                if not trailing_active and favorable >= cfg['trail_act']:
                    trailing_active = True
                    trailing_sl = min(trailing_sl, current + cfg['trail_dist'])

                if trailing_active:
                    new_sl = current + cfg['trail_dist']
                    if new_sl < trailing_sl:
                        trailing_sl = new_sl

                if current >= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = 'TR' if trailing_active else ('BE' if be_done else 'SL')
                    break

        if exit_price is None:
            exit_price = bid if rev_dir == 'BUY' else ask
            exit_reason = 'TM'

        pnl_pts = (exit_price - entry) if rev_dir == 'BUY' else (entry - exit_price)
        pnl_usd = pnl_pts * LOTS * TICK_VALUE

        results.append({'pnl_pts': pnl_pts, 'pnl_usd': pnl_usd, 'reason': exit_reason})

    if not results:
        continue

    wins = sum(1 for r in results if r['pnl_pts'] > 0)
    total = sum(r['pnl_usd'] for r in results)
    avg = total / len(results)
    avg_w = np.mean([r['pnl_usd'] for r in results if r['pnl_pts'] > 0]) if wins > 0 else 0
    avg_l = np.mean([r['pnl_usd'] for r in results if r['pnl_pts'] <= 0]) if wins < len(results) else 0

    sl_n = sum(1 for r in results if r['reason'] == 'SL')
    be_n = sum(1 for r in results if r['reason'] == 'BE')
    tr_n = sum(1 for r in results if r['reason'] == 'TR')
    tm_n = sum(1 for r in results if r['reason'] == 'TM')

    wr = wins / len(results) * 100
    marker = ' ***' if total > 0 else ''
    print(f'{cfg["sl"]:4d} {cfg["be"]:4d} {cfg["trail_act"]:4d}/{cfg["trail_dist"]:<3} | {wr:5.0f} {total:+8.0f} {avg:+6.1f} {avg_w:+6.1f} {avg_l:+6.1f} | {sl_n:3d} {be_n:3d} {tr_n:3d} {tm_n:3d}{marker}')

mt5.shutdown()
