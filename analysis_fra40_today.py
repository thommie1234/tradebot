"""FRA40 reversal backtest — today only, with tick data."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import numpy as np

mt5.initialize(r"C:\Program Files\BrightFunded MT5 Terminal\terminal64.exe")

now = datetime.now(timezone.utc)
today = now.replace(hour=0, minute=0, second=0)
deals = mt5.history_deals_get(today, now)

exits = []
for d in (deals or []):
    if 'FRA40' not in d.symbol or d.entry != 1:
        continue
    t = datetime.fromtimestamp(d.time, tz=timezone.utc)
    original_dir = 'SELL' if d.type == 0 else 'BUY'
    exits.append({'dir': original_dir, 'exit': d.price, 'pnl': d.profit, 'close_time': t})

print(f"FRA40 exits vandaag: {len(exits)}")

TV = 0.0115
LOTS = 30

configs = [
    ("SL15 TR5/3",  15, 4, 5, 3),
    ("SL20 TR8/4",  20, 5, 8, 4),
    ("SL25 TR8/4",  25, 5, 8, 4),
    ("SL30 TR10/5", 30, 8, 10, 5),
    ("SL20 TR12/6", 20, 8, 12, 6),
]

for name, sl_pts, be_pts, trail_act, trail_dist in configs:
    results = []

    for e in exits:
        rev = 'SELL' if e['dir'] == 'BUY' else 'BUY'
        ticks = mt5.copy_ticks_range('FRA40.cash', e['close_time'],
                                      e['close_time'] + timedelta(hours=2), mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) < 20:
            continue

        entry = e['exit']
        sl = entry + sl_pts if rev == 'SELL' else entry - sl_pts
        best = entry
        trail_on = False
        tsl = sl
        be_on = False
        ep = None
        er = None

        for tick in ticks:
            bid, ask = tick[1], tick[2]
            if rev == 'BUY':
                c = bid
                fav = c - entry
                if c > best: best = c
                if not be_on and fav >= be_pts: tsl = entry + 0.5; be_on = True
                if not trail_on and fav >= trail_act: trail_on = True; tsl = max(tsl, c - trail_dist)
                if trail_on:
                    ns = c - trail_dist
                    if ns > tsl: tsl = ns
                if c <= tsl: ep = tsl; er = 'TR' if trail_on else ('BE' if be_on else 'SL'); break
            else:
                c = ask
                fav = entry - c
                if c < best: best = c
                if not be_on and fav >= be_pts: tsl = entry - 0.5; be_on = True
                if not trail_on and fav >= trail_act: trail_on = True; tsl = min(tsl, c + trail_dist)
                if trail_on:
                    ns = c + trail_dist
                    if ns < tsl: tsl = ns
                if c >= tsl: ep = tsl; er = 'TR' if trail_on else ('BE' if be_on else 'SL'); break

        if ep is None:
            ep = bid if rev == 'BUY' else ask
            er = 'TM'

        pnl = ((ep - entry) if rev == 'BUY' else (entry - ep))
        mfe = (best - entry) if rev == 'BUY' else (entry - best)
        results.append((e['close_time'], rev, pnl, pnl * LOTS * TV, er, mfe))

    if not results:
        continue

    wins = sum(1 for r in results if r[2] > 0)
    total = sum(r[3] for r in results)

    print(f"\n=== {name} ===")
    print(f"  {'Time':>8s} {'Rev':>5s} {'P/L$':>8s} {'pts':>7s} {'MFE':>7s} {'Exit':>4s}")
    for r in results:
        print(f"  {r[0].strftime('%H:%M'):>8s} {r[1]:>5s} ${r[3]:+8.1f} {r[2]:+7.1f} {r[5]:+7.1f} {r[4]:>4s}")
    print(f"  Total: ${total:+.0f}  WR: {wins}/{len(results)}")

mt5.shutdown()
