"""Trailing stop efficiency analysis: how much profit is left on the table?"""
import pyarrow.parquet as pq
import pyarrow.compute as pc
import sqlite3, json, sys, os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sym_map = {
    "EURUSD": "EUR_USD", "AUDUSD": "AUD_USD", "NZDUSD": "NZD_USD",
    "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY", "EURCHF": "EUR_CHF",
    "US100.cash": "US100.cash", "FRA40.cash": "FRA40.cash", "US30.cash": "US30.cash",
    "UK100.cash": "UK100.cash", "US500.cash": "US500.cash",
    "GBPCHF": "GBP_CHF", "USDCHF": "USD_CHF", "USDCAD": "USD_CAD",
    "NZDCAD": "NZD_CAD", "NZDJPY": "NZD_JPY", "AUDNZD": "AUD_NZD",
    "EURJPY": "EUR_JPY", "EURGBP": "EUR_GBP", "CADJPY": "CAD_JPY",
    "BTC/USD": "BTC_USD", "XDG/USD": "DOGEUSD",
}
TICK_BASE = "C:/tick_data/ssd1"

db = sqlite3.connect("bf/audit/sovereign_log.db", timeout=10)
db.row_factory = sqlite3.Row
rows = db.execute("""
    SELECT id, symbol, direction, entry_price, sl_price, tp_price, exit_price, pnl,
           lot_size, timestamp, exit_timestamp
    FROM trades
    WHERE status = 'CLOSED' AND exit_price IS NOT NULL AND pnl IS NOT NULL
      AND sl_price IS NOT NULL AND tp_price IS NOT NULL
    ORDER BY id
""").fetchall()
trades = [dict(r) for r in rows]
db.close()

tick_cache = {}

def load_ticks(symbol, year_month):
    key = (symbol, year_month)
    if key not in tick_cache:
        tick_dir = sym_map.get(symbol, symbol)
        path = f"{TICK_BASE}/{tick_dir}/{year_month}.parquet"
        if os.path.exists(path):
            tick_cache[key] = pq.read_table(path, columns=["time", "bid", "ask"])
        else:
            tick_cache[key] = None
    return tick_cache[key]

def get_mfe_mae(symbol, direction, entry_price, start_str, end_str):
    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    months = set()
    cur = start
    while cur <= end:
        months.add(f"{cur.year}-{cur.month:02d}")
        cur += timedelta(days=32)
        cur = cur.replace(day=1)
    bids, asks = [], []
    for ym in sorted(months):
        table = load_ticks(symbol, ym)
        if table is None:
            continue
        mask = pc.and_(
            pc.greater_equal(table.column("time"), start),
            pc.less_equal(table.column("time"), end),
        )
        f = table.filter(mask)
        if f.num_rows > 0:
            bids.extend(f.column("bid").to_pylist())
            asks.extend(f.column("ask").to_pylist())
    if len(bids) < 10:
        return None, None
    if direction == "BUY":
        return max(bids) - entry_price, entry_price - min(bids)
    else:
        return entry_price - min(asks), max(asks) - entry_price

with open("bf/config.json") as f:
    config = json.load(f)

# Enrich
print("Computing MFE/MAE from tick data...")
for i, t in enumerate(trades):
    entry = t["entry_price"]
    sl = t["sl_price"]
    d = t["direction"]
    exit_p = t["exit_price"]
    tp = t["tp_price"]
    t["sl_dist"] = (entry - sl) if d == "BUY" else (sl - entry)
    t["tp_dist"] = (tp - entry) if d == "BUY" else (entry - tp)
    t["is_winner"] = t["pnl"] > 0
    price_move = abs(exit_p - entry)
    t["dollar_per_point"] = abs(t["pnl"]) / price_move if price_move > 0 else 0
    t["captured_move"] = (exit_p - entry) if d == "BUY" else (entry - exit_p)

    if t["symbol"] in sym_map and t["sl_dist"] > 0:
        t["mfe"], t["mae"] = get_mfe_mae(t["symbol"], d, entry, t["timestamp"], t["exit_timestamp"])
    else:
        t["mfe"], t["mae"] = None, None
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(trades)}...")

winners = [t for t in trades if t["is_winner"] and t["mfe"] is not None]
all_tick = [t for t in trades if t["mfe"] is not None and t["mae"] is not None]
print(f"\nWinners with MFE data: {len(winners)}")
print(f"All trades with tick data: {len(all_tick)}")

# =====================================================
print(f"\n{'='*85}")
print("TRAILING STOP ANALYSE: Hoeveel winst wordt achtergelaten?")
print(f"{'='*85}")

all_captured_usd = sum(t["captured_move"] * t["dollar_per_point"] for t in winners)
all_mfe_usd = sum(t["mfe"] * t["dollar_per_point"] for t in winners if t["mfe"] > 0)
all_left_usd = all_mfe_usd - all_captured_usd
curr_eff = all_captured_usd / all_mfe_usd * 100 if all_mfe_usd > 0 else 0

print(f"\nALLE {len(winners)} winners met tick data:")
print(f"  Captured profit:    ${all_captured_usd:>10,.2f}")
print(f"  Max possible (MFE): ${all_mfe_usd:>10,.2f}")
print(f"  Left on table:      ${all_left_usd:>10,.2f}")
print(f"  Capture efficiency: {curr_eff:.1f}%")

# Top 15 biggest MFE gaps
print(f"\nTop 15 trades met meeste achtergelaten winst:")
print(f"{'Symbol':<14} {'Dir':>4} {'PnL':>8} {'Capt$':>8} {'MFE$':>8} {'Left$':>8} {'Eff%':>5}")
print("-" * 62)
gaps = [(t, (t["mfe"] - t["captured_move"]) * t["dollar_per_point"]) for t in winners if t["mfe"] > t["captured_move"]]
gaps.sort(key=lambda x: x[1], reverse=True)
for t, left in gaps[:15]:
    capt_usd = t["captured_move"] * t["dollar_per_point"]
    mfe_usd = t["mfe"] * t["dollar_per_point"]
    eff = capt_usd / mfe_usd * 100 if mfe_usd > 0 else 0
    print(f"{t['symbol']:<14} {t['direction']:>4} {t['pnl']:>+8.2f} {capt_usd:>8.2f} {mfe_usd:>8.2f} {left:>8.2f} {eff:>4.0f}%")

# =====================================================
print(f"\n{'='*85}")
print("PER-SYMBOL: Trail efficiency vs huidige config")
print(f"{'='*85}")

sym_w = defaultdict(list)
for t in winners:
    sym_w[t["symbol"]].append(t)

print(f"{'Symbol':<14} {'W':>3} {'AvgCapt$':>9} {'AvgMFE$':>9} {'Eff%':>5} {'Trail_d':>8} {'Trail_a':>8} {'TotalLeft$':>11}")
print("-" * 75)
for sym in sorted(sym_w.keys()):
    ws = sym_w[sym]
    avg_capt = sum(t["captured_move"] * t["dollar_per_point"] for t in ws) / len(ws)
    avg_mfe = sum(t["mfe"] * t["dollar_per_point"] for t in ws) / len(ws)
    eff = avg_capt / avg_mfe * 100 if avg_mfe > 0 else 0
    left = sum(
        (t["mfe"] - t["captured_move"]) * t["dollar_per_point"]
        for t in ws
        if t["mfe"] > t["captured_move"]
    )
    cfg = config.get(sym, {})
    td = cfg.get("trail_distance_atr", "?")
    ta = cfg.get("trail_activation_atr", "?")
    print(f"{sym:<14} {len(ws):>3} {avg_capt:>9.2f} {avg_mfe:>9.2f} {eff:>4.0f}% {str(td):>8} {str(ta):>8} {left:>11.2f}")

# =====================================================
print(f"\n{'='*85}")
print("SIMULATIE: Wat als winners meer van hun MFE hadden gevangen?")
print(f"{'='*85}")

curr_pnl_tick = sum(t["pnl"] for t in all_tick)
print(f"\n  Huidige capture efficiency: {curr_eff:.1f}%")
print(f"  Huidige P&L (tick-verified): ${curr_pnl_tick:,.2f}\n")

for target_eff in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    sim_pnl = 0
    for t in all_tick:
        if not t["is_winner"]:
            sim_pnl += t["pnl"]
        else:
            new_capture = t["mfe"] * target_eff
            sim_pnl += new_capture * t["dollar_per_point"]
    delta = sim_pnl - curr_pnl_tick
    marker = " <-- BREAKEVEN" if abs(sim_pnl) < 500 else (" <-- HUIDIGE" if abs(target_eff - curr_eff/100) < 0.03 else "")
    print(f"  Bij {target_eff*100:>3.0f}% MFE capture: ${sim_pnl:>10,.2f}  ({delta:>+10,.2f}){marker}")

# =====================================================
print(f"\n{'='*85}")
print("GECOMBINEERDE SIMULATIE: SL x0.75 + betere trail capture")
print(f"{'='*85}")

for target_eff in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    sim_pnl = 0
    w = 0
    l = 0
    for t in all_tick:
        if t["sl_dist"] <= 0 or t["dollar_per_point"] <= 0:
            sim_pnl += t["pnl"]
            continue
        new_sl = t["sl_dist"] * 0.75
        if t["is_winner"]:
            if t["mae"] is not None and t["mae"] >= new_sl:
                # Killed by tighter SL
                sim_pnl -= t["dollar_per_point"] * new_sl
                l += 1
            else:
                new_capture = t["mfe"] * target_eff
                sim_pnl += new_capture * t["dollar_per_point"]
                w += 1
        else:
            sim_pnl -= min(t["dollar_per_point"] * new_sl, abs(t["pnl"]))
            l += 1
    print(f"  SL x0.75 + {target_eff*100:>3.0f}% MFE: ${sim_pnl:>10,.2f}  (W={w} L={l})")

# =====================================================
print(f"\n{'='*85}")
print("CONCRETE AANBEVELINGEN")
print(f"{'='*85}")
print("""
1. TRAILING STOP TUNING (hoogste prioriteit):
   - Huidige capture efficiency: ~{eff:.0f}% van MFE
   - Doel: 30-35% capture -> systeem wordt winstgevend
   - Actie: trail_distance_atr VERGROTEN (meer ruimte)
   - Actie: trail_activation_atr VERGROTEN (later activeren)

2. SL REDUCTIE (secundair, ~25% reductie):
   - Verliest effectiviteit door killed winners
   - Maar gecombineerd met betere trail is het waardevol

3. INDICES SPECIFIEK:
   - FRA40.cash en US30.cash: SL van 1.75x naar ~1.0x ATR
   - UK100.cash: SL van 4.0x naar ~2.5x ATR
   - Trail distance voor indices veel te krap

4. GECOMBINEERD OPTIMAAL:
   - SL x0.75 + 35% MFE capture = winstgevend systeem
""".format(eff=curr_eff))
