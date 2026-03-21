"""FRA40 reversal backtest — after each trade close, go opposite direction."""
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

print(f'FRA40 exit deals on BF: {len(exits)}')
header = f"{'Close':>12s} {'Orig':>5s} {'Exit':>10s} {'P/L':>8s} | {'Rev':>5s} {'30m':>8s} {'60m':>8s} {'120m':>8s}"
print(header)
print('-' * 75)

rev_results = []
for e in exits:
    rev_dir = 'SELL' if e['dir'] == 'BUY' else 'BUY'

    ticks = mt5.copy_ticks_range('FRA40.cash', e['close_time'],
                                  e['close_time'] + timedelta(hours=2), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 20:
        continue

    entry_price = e['exit']
    row = {'dir': rev_dir, '30m': 0, '60m': 0, '120m': 0}

    for minutes in [30, 60, 120]:
        cutoff = e['close_time'] + timedelta(minutes=minutes)
        subset = [t for t in ticks if datetime.fromtimestamp(t[0], tz=timezone.utc) <= cutoff]
        if not subset:
            continue

        bids = [t[1] for t in subset]
        asks = [t[2] for t in subset]

        if rev_dir == 'SELL':
            mfe = entry_price - min(asks)
        else:
            mfe = max(bids) - entry_price

        row[f'{minutes}m'] = mfe * 0.6  # capture 60% of max move

    for k in ['30m', '60m', '120m']:
        row[f'{k}_pnl'] = row[k] * 30 * 0.0115

    rev_results.append(row)
    print(f'{e["close_time"].strftime("%m-%d %H:%M"):>12s} {e["dir"]:>5s} {entry_price:10.2f} {e["pnl"]:+8.2f} | '
          f'{rev_dir:>5s} {row["30m"]:+8.2f} {row["60m"]:+8.2f} {row["120m"]:+8.2f} pts')

if rev_results:
    print(f'\n=== FRA40 Reversal ({len(rev_results)} trades, 60% capture, 30 lots) ===')
    for period in ['30m', '60m', '120m']:
        moves = [r[period] for r in rev_results]
        pnls = [r[f'{period}_pnl'] for r in rev_results]
        winners = sum(1 for m in moves if m > 0)
        print(f'  {period}: gem={np.mean(moves):+.2f} pts  totaal=${sum(pnls):+.0f}  '
              f'winners={winners}/{len(moves)} ({winners/len(moves)*100:.0f}%)')

mt5.shutdown()
