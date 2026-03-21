"""FRA40 reversal backtest with trailing stop simulation using tick data."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import numpy as np

mt5.initialize(r"C:\Program Files\BrightFunded MT5 Terminal\terminal64.exe")

now = datetime.now(timezone.utc)
start = now - timedelta(days=30)
deals = mt5.history_deals_get(start, now)

# Get FRA40 exit deals
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
            'pnl': d.profit + d.commission,
            'volume': d.volume,
            'close_time': t,
        })

print(f'FRA40 exits: {len(exits)}')

# Test multiple SL/Trail configs
configs = [
    {'name': 'Tight',   'sl_pts': 5,  'be_pts': 3,  'trail_act': 5,  'trail_dist': 3},
    {'name': 'Medium',  'sl_pts': 8,  'be_pts': 5,  'trail_act': 8,  'trail_dist': 4},
    {'name': 'Wide',    'sl_pts': 12, 'be_pts': 6,  'trail_act': 10, 'trail_dist': 5},
    {'name': 'Ultra',   'sl_pts': 15, 'be_pts': 8,  'trail_act': 12, 'trail_dist': 6},
    {'name': 'ATR03',   'sl_pts': 10, 'be_pts': 4,  'trail_act': 7,  'trail_dist': 3.5},
]

TICK_VALUE = 0.0115  # FRA40 tick value approx
LOTS = 30

for cfg in configs:
    results = []

    for e in exits:
        rev_dir = 'SELL' if e['dir'] == 'BUY' else 'BUY'

        ticks = mt5.copy_ticks_range('FRA40.cash', e['close_time'],
                                      e['close_time'] + timedelta(hours=3), mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) < 20:
            continue

        entry = e['exit']
        sl = entry + cfg['sl_pts'] if rev_dir == 'SELL' else entry - cfg['sl_pts']
        be_trigger = cfg['be_pts']
        trail_act = cfg['trail_act']
        trail_dist = cfg['trail_dist']

        # Simulate tick by tick
        best_price = entry
        trailing_active = False
        trailing_sl = sl
        be_done = False
        exit_price = None
        exit_reason = None

        for tick in ticks:
            bid = tick[1]
            ask = tick[2]

            if rev_dir == 'BUY':
                current = bid  # exit at bid for BUY
                favorable = current - entry

                # Update best price
                if current > best_price:
                    best_price = current

                # Breakeven
                if not be_done and favorable >= be_trigger:
                    trailing_sl = entry + 0.5  # just above entry
                    be_done = True

                # Trail activation
                if not trailing_active and favorable >= trail_act:
                    trailing_active = True
                    trailing_sl = max(trailing_sl, current - trail_dist)

                # Trail update
                if trailing_active:
                    new_sl = current - trail_dist
                    if new_sl > trailing_sl:
                        trailing_sl = new_sl

                # Check SL hit
                if current <= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = 'TRAIL' if trailing_active else ('BE' if be_done else 'SL')
                    break

            else:  # SELL
                current = ask  # exit at ask for SELL
                favorable = entry - current

                if current < best_price:
                    best_price = current

                if not be_done and favorable >= be_trigger:
                    trailing_sl = entry - 0.5
                    be_done = True

                if not trailing_active and favorable >= trail_act:
                    trailing_active = True
                    trailing_sl = min(trailing_sl, current + trail_dist)

                if trailing_active:
                    new_sl = current + trail_dist
                    if new_sl < trailing_sl:
                        trailing_sl = new_sl

                if current >= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = 'TRAIL' if trailing_active else ('BE' if be_done else 'SL')
                    break

        if exit_price is None:
            # Still open after 3h — close at last tick
            exit_price = bid if rev_dir == 'BUY' else ask
            exit_reason = 'TIME'

        if rev_dir == 'BUY':
            pnl_pts = exit_price - entry
        else:
            pnl_pts = entry - exit_price

        pnl_usd = pnl_pts * LOTS * TICK_VALUE

        results.append({
            'pnl_pts': pnl_pts,
            'pnl_usd': pnl_usd,
            'reason': exit_reason,
            'rev_dir': rev_dir,
            'best': (best_price - entry) if rev_dir == 'BUY' else (entry - best_price),
        })

    if not results:
        continue

    wins = sum(1 for r in results if r['pnl_pts'] > 0)
    total_pnl = sum(r['pnl_usd'] for r in results)
    avg_pnl = total_pnl / len(results)
    avg_winner = np.mean([r['pnl_usd'] for r in results if r['pnl_pts'] > 0]) if wins > 0 else 0
    avg_loser = np.mean([r['pnl_usd'] for r in results if r['pnl_pts'] <= 0]) if wins < len(results) else 0

    sl_count = sum(1 for r in results if r['reason'] == 'SL')
    be_count = sum(1 for r in results if r['reason'] == 'BE')
    trail_count = sum(1 for r in results if r['reason'] == 'TRAIL')
    time_count = sum(1 for r in results if r['reason'] == 'TIME')

    print(f'\n=== {cfg["name"]}: SL={cfg["sl_pts"]}pts BE={cfg["be_pts"]}pts Trail={cfg["trail_act"]}/{cfg["trail_dist"]}pts ===')
    print(f'  Trades: {len(results)}  WR: {wins}/{len(results)} ({wins/len(results)*100:.0f}%)')
    print(f'  Total P/L: ${total_pnl:+.0f}  Avg: ${avg_pnl:+.0f}')
    print(f'  Avg winner: ${avg_winner:+.0f}  Avg loser: ${avg_loser:+.0f}')
    print(f'  Exits: SL={sl_count} BE={be_count} Trail={trail_count} Time={time_count}')

    # Show individual trades
    if cfg['name'] == 'Medium' or cfg['name'] == 'ATR03':
        print(f'  {"Close":>12s} {"Rev":>5s} {"P/L":>8s} {"Reason":>6s} {"Best":>8s}')
        for i, (e, r) in enumerate(zip(exits[:len(results)], results)):
            print(f'  {e["close_time"].strftime("%m-%d %H:%M"):>12s} {r["rev_dir"]:>5s} ${r["pnl_usd"]:+8.1f} {r["reason"]:>6s} {r["best"]:+8.2f}pts')

mt5.shutdown()
