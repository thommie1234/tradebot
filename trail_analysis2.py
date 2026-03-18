"""Trail analysis with proper tick data coverage filter."""
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

# Tick data end times per symbol (from inspection)
TICK_ENDS = {
    "EUR_USD": "2026-03-10T04:03:56",
    "AUD_USD": "2026-03-10T04:01:32",
    "NZD_USD": "2026-03-10T04:01:00",
    "GBP_USD": "2026-03-10T04:03:00",
    "USD_JPY": "2026-03-10T04:03:00",
    "EUR_CHF": "2026-03-10T04:03:00",
    "US100.cash": "2026-03-10T04:05:52",
    "FRA40.cash": "2026-03-09T23:54:36",
    "US30.cash": "2026-03-10T04:05:00",
    "UK100.cash": "2026-03-10T04:05:00",
    "US500.cash": "2026-03-10T04:05:00",
    "GBP_CHF": "2026-03-10T04:03:00",
    "USD_CHF": "2026-03-10T04:03:00",
    "USD_CAD": "2026-03-10T04:03:00",
    "NZD_CAD": "2026-03-10T04:03:00",
    "NZD_JPY": "2026-03-10T04:03:00",
    "AUD_NZD": "2026-03-10T04:03:00",
    "EUR_JPY": "2026-03-10T04:03:00",
    "EUR_GBP": "2026-03-10T04:03:00",
    "CAD_JPY": "2026-03-10T04:03:00",
    "BTC_USD": "2026-03-10T17:11:27",
    "DOGEUSD": "2026-03-10T17:11:00",
}

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
all_trades = [dict(r) for r in rows]
db.close()

# Filter to trades with FULL tick coverage
trades = []
skipped_no_map = 0
skipped_no_cover = 0
for t in all_trades:
    sym = t["symbol"]
    if sym not in sym_map:
        skipped_no_map += 1
        continue
    tick_dir = sym_map[sym]
    tick_end = TICK_ENDS.get(tick_dir)
    if tick_end is None:
        skipped_no_map += 1
        continue
    # Trade must have exited BEFORE tick data ends
    exit_ts = t["exit_timestamp"][:19]  # strip fractional seconds
    if exit_ts > tick_end:
        skipped_no_cover += 1
        continue
    trades.append(t)

print(f"Total trades: {len(all_trades)}")
print(f"Skipped (no symbol map): {skipped_no_map}")
print(f"Skipped (exit after tick data ends): {skipped_no_cover}")
print(f"Fully covered trades: {len(trades)}")

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
print("\nComputing MFE/MAE from tick data (fully covered trades only)...")
valid = []
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

    if t["sl_dist"] > 0:
        t["mfe"], t["mae"] = get_mfe_mae(t["symbol"], d, entry, t["timestamp"], t["exit_timestamp"])
        if t["mfe"] is not None:
            valid.append(t)

print(f"Trades with valid MFE/MAE: {len(valid)}")

winners = [t for t in valid if t["is_winner"]]
losers = [t for t in valid if not t["is_winner"]]
print(f"Winners: {len(winners)}, Losers: {len(losers)}")

# =====================================================
print(f"\n{'='*85}")
print("TRAILING STOP ANALYSE (volledig tick-gedekte trades)")
print(f"{'='*85}")

capt_usd = sum(t["captured_move"] * t["dollar_per_point"] for t in winners)
mfe_usd = sum(t["mfe"] * t["dollar_per_point"] for t in winners if t["mfe"] > 0)
left_usd = mfe_usd - capt_usd
eff = capt_usd / mfe_usd * 100 if mfe_usd > 0 else 0

print(f"\n{len(winners)} winners:")
print(f"  Captured profit:    ${capt_usd:>10,.2f}")
print(f"  Max possible (MFE): ${mfe_usd:>10,.2f}")
print(f"  Left on table:      ${left_usd:>10,.2f}")
print(f"  Capture efficiency: {eff:.1f}%")

total_pnl = sum(t["pnl"] for t in valid)
total_loss = sum(t["pnl"] for t in losers)
print(f"\n  Total P&L (covered): ${total_pnl:,.2f}")
print(f"  Total losses:        ${total_loss:,.2f}")
print(f"  Total wins:          ${capt_usd:,.2f}")

# Per-trade MFE analysis for winners
print(f"\n{'='*85}")
print("PER WINNER: Captured vs MFE")
print(f"{'='*85}")
print(f"{'Symbol':<14} {'Dir':>4} {'PnL':>8} {'Capt':>10} {'MFE':>10} {'Eff%':>5} {'MFE/SL':>7}")
print("-" * 65)
for t in sorted(winners, key=lambda x: x["pnl"], reverse=True):
    c = t["captured_move"]
    m = t["mfe"]
    e = c / m * 100 if m > 0 else 0
    mfe_sl = m / t["sl_dist"] if t["sl_dist"] > 0 else 0
    print(f"{t['symbol']:<14} {t['direction']:>4} {t['pnl']:>+8.2f} {c:>+10.5f} {m:>10.5f} {e:>4.0f}% {mfe_sl:>6.1f}x")

# Per-symbol
print(f"\n{'='*85}")
print("PER-SYMBOL: Trail efficiency")
print(f"{'='*85}")
sym_w = defaultdict(list)
sym_l = defaultdict(list)
for t in valid:
    if t["is_winner"]:
        sym_w[t["symbol"]].append(t)
    else:
        sym_l[t["symbol"]].append(t)

all_syms = sorted(set(list(sym_w.keys()) + list(sym_l.keys())))
print(f"{'Symbol':<14} {'W':>3} {'L':>3} {'Eff%':>5} {'AvgW$':>8} {'AvgL$':>8} {'MFE_left$':>10} {'Trail_d':>8}")
print("-" * 70)
for sym in all_syms:
    ws = sym_w.get(sym, [])
    ls = sym_l.get(sym, [])
    if ws:
        c_usd = sum(t["captured_move"] * t["dollar_per_point"] for t in ws)
        m_usd = sum(t["mfe"] * t["dollar_per_point"] for t in ws if t["mfe"] > 0)
        e = c_usd / m_usd * 100 if m_usd > 0 else 0
        avg_w = c_usd / len(ws)
        left = m_usd - c_usd
    else:
        e = 0
        avg_w = 0
        left = 0
    avg_l = sum(t["pnl"] for t in ls) / len(ls) if ls else 0
    cfg = config.get(sym, {})
    td = cfg.get("trail_distance_atr", "?")
    print(f"{sym:<14} {len(ws):>3} {len(ls):>3} {e:>4.0f}% {avg_w:>8.2f} {avg_l:>8.2f} {left:>10.2f} {str(td):>8}")

# =====================================================
# KEY SIMULATION: What if trail captured more MFE?
print(f"\n{'='*85}")
print("SIMULATIE: Impact van betere MFE capture")
print(f"{'='*85}")

print(f"\nHuidige efficiency: {eff:.1f}%")
print(f"Huidige P&L: ${total_pnl:,.2f}\n")

for target in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
    sim = 0
    for t in valid:
        if not t["is_winner"]:
            sim += t["pnl"]
        else:
            # Simulate capturing target% of MFE
            new_capture = t["mfe"] * target
            sim += new_capture * t["dollar_per_point"]
    delta = sim - total_pnl
    marker = ""
    if sim > 0 and (sim - delta) <= 0:
        marker = " <-- WINSTGEVEND"
    print(f"  {target*100:>3.0f}% MFE capture: ${sim:>10,.2f}  ({delta:>+10,.2f}){marker}")

# =====================================================
# COMBINED: Tighter SL + better trail
print(f"\n{'='*85}")
print("GECOMBINEERDE SIMULATIE: SL reductie + betere trail")
print(f"{'='*85}")
print(f"\n{'SL factor':>10} {'30% MFE':>10} {'40% MFE':>10} {'50% MFE':>10} {'60% MFE':>10} {'70% MFE':>10}")
print("-" * 65)

for sl_f in [1.0, 0.85, 0.75, 0.65, 0.55]:
    results = []
    for target in [0.30, 0.40, 0.50, 0.60, 0.70]:
        sim = 0
        for t in valid:
            new_sl = t["sl_dist"] * sl_f
            if t["is_winner"]:
                if t["mae"] is not None and t["mae"] >= new_sl:
                    sim -= t["dollar_per_point"] * new_sl
                else:
                    sim += t["mfe"] * target * t["dollar_per_point"]
            else:
                sim -= min(t["dollar_per_point"] * new_sl, abs(t["pnl"]))
        results.append(sim)
    vals = "".join(f"${v:>9,.0f}" + " " for v in results)
    print(f"  SL x{sl_f:.2f}: {vals}")

# =====================================================
print(f"\n{'='*85}")
print("MAE ANALYSE: Max Adverse Excursion voor winners")
print(f"{'='*85}")

print(f"\n{'MAE als % van SL':>20} {'#Winners':>10} {'Cumul%':>8}")
print("-" * 42)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for th in thresholds:
    cnt = sum(1 for t in winners if t["mae"] is not None and t["mae"] / t["sl_dist"] <= th)
    pct = cnt / len(winners) * 100 if winners else 0
    bar = "#" * int(pct / 2)
    print(f"  MAE <= {th*100:>3.0f}% SL: {cnt:>6}/{len(winners)}  {pct:>5.1f}%  {bar}")

# =====================================================
print(f"\n{'='*85}")
print("CONCLUSIE")
print(f"{'='*85}")
print(f"""
DATA (alleen trades met volledige tick coverage, n={len(valid)}):
  Winners: {len(winners)}, Losers: {len(losers)}
  Win rate: {len(winners)/(len(winners)+len(losers))*100:.0f}%
  Avg win: ${capt_usd/len(winners):,.2f}, Avg loss: ${total_loss/len(losers):,.2f}
  Trail capture efficiency: {eff:.1f}% van MFE
  Total P&L: ${total_pnl:,.2f}

PROBLEEM DIAGNOSE:
  De trailing stop vangt gemiddeld {eff:.0f}% van het maximale prijsverloop (MFE).
  {'Dit is GOED - de trail doet zijn werk.' if eff > 60 else 'Dit is LAAG - er wordt veel winst achtergelaten.'}
  {'Het probleem zit dus NIET in de trail maar in de SL-grootte.' if eff > 60 else 'Het probleem zit in zowel trail als SL.'}

  Verliezen (avg ${abs(total_loss/len(losers)):,.0f}) zijn te groot t.o.v. winsten (avg ${capt_usd/len(winners):,.0f}).
""")
