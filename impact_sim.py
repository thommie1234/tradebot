"""Simulate impact of old vs new config on 176 real trades."""
import sqlite3, json, sys, random
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

# SL multipliers
old_sl = {
    "EURUSD": 0.84, "AUDUSD": 0.735, "NZDUSD": 0.96, "USDJPY": 0.45,
    "GBPUSD": 0.735, "EURCHF": 1.20, "US100.cash": 0.45,
    "FRA40.cash": 1.75, "US30.cash": 1.75,
}
new_sl = {
    "EURUSD": 0.63, "AUDUSD": 0.55, "NZDUSD": 0.72, "USDJPY": 0.45,
    "GBPUSD": 0.735, "EURCHF": 1.20, "US100.cash": 0.30,
    "FRA40.cash": 1.0, "US30.cash": 1.0,
}

for t in trades:
    sym = t["symbol"]
    entry = t["entry_price"]
    sl = t["sl_price"]
    d = t["direction"]
    exit_p = t["exit_price"]
    t["sl_dist"] = (entry - sl) if d == "BUY" else (sl - entry)
    t["is_winner"] = t["pnl"] > 0
    price_move = abs(exit_p - entry)
    t["dollar_per_point"] = abs(t["pnl"]) / price_move if price_move > 0 else 0
    if sym in old_sl and sym in new_sl:
        t["sl_factor"] = new_sl[sym] / old_sl[sym]
    else:
        t["sl_factor"] = 1.0


def simulate(trades, use_new_sl=False):
    total_pnl = 0
    w_cnt = 0
    l_cnt = 0
    w_pnl = 0
    l_pnl = 0
    killed = 0
    sym_pnl = defaultdict(float)

    for t in trades:
        sym = t["symbol"]
        factor = t["sl_factor"] if use_new_sl else 1.0

        if t["sl_dist"] <= 0 or t["dollar_per_point"] <= 0 or factor >= 1.0:
            total_pnl += t["pnl"]
            sym_pnl[sym] += t["pnl"]
            if t["pnl"] > 0:
                w_cnt += 1
                w_pnl += t["pnl"]
            else:
                l_cnt += 1
                l_pnl += t["pnl"]
            continue

        new_sl_dist = t["sl_dist"] * factor

        if t["is_winner"]:
            random.seed(t["id"] + 42)
            # Kill rates from tick-verified MAE analysis
            if factor <= 0.60:
                kill_rate = 0.25
            elif factor <= 0.70:
                kill_rate = 0.22
            elif factor <= 0.80:
                kill_rate = 0.18
            else:
                kill_rate = 0.12

            if random.random() < kill_rate:
                killed += 1
                new_pnl = -(t["dollar_per_point"] * new_sl_dist)
                l_cnt += 1
                l_pnl += new_pnl
                total_pnl += new_pnl
                sym_pnl[sym] += new_pnl
            else:
                w_cnt += 1
                w_pnl += t["pnl"]
                total_pnl += t["pnl"]
                sym_pnl[sym] += t["pnl"]
        else:
            new_loss = t["dollar_per_point"] * new_sl_dist
            new_pnl = -min(new_loss, abs(t["pnl"]))
            l_cnt += 1
            l_pnl += new_pnl
            total_pnl += new_pnl
            sym_pnl[sym] += new_pnl

    return {
        "total": total_pnl,
        "wins": w_cnt,
        "losses": l_cnt,
        "avg_w": w_pnl / w_cnt if w_cnt else 0,
        "avg_l": l_pnl / l_cnt if l_cnt else 0,
        "killed": killed,
        "sym_pnl": dict(sym_pnl),
        "w_pnl": w_pnl,
        "l_pnl": l_pnl,
    }


old = simulate(trades, use_new_sl=False)
new = simulate(trades, use_new_sl=True)

print("=" * 85)
print("IMPACT SIMULATIE: Oude config vs Nieuwe config (176 trades)")
print("=" * 85)

print(f"\n{'':>20} {'OUDE CONFIG':>15} {'NIEUWE CONFIG':>15} {'VERSCHIL':>12}")
print("-" * 65)
print(f"{'Totaal P&L':>20} ${old['total']:>13,.2f} ${new['total']:>13,.2f} ${new['total']-old['total']:>+10,.2f}")
print(f"{'Winners':>20} {old['wins']:>15} {new['wins']:>15} {new['wins']-old['wins']:>+12}")
print(f"{'Losers':>20} {old['losses']:>15} {new['losses']:>15} {new['losses']-old['losses']:>+12}")
wr_old = 100 * old["wins"] / (old["wins"] + old["losses"])
wr_new = 100 * new["wins"] / (new["wins"] + new["losses"])
print(f"{'Win rate':>20} {wr_old:>14.1f}% {wr_new:>14.1f}%")
print(f"{'Avg win':>20} ${old['avg_w']:>13,.2f} ${new['avg_w']:>13,.2f} ${new['avg_w']-old['avg_w']:>+10,.2f}")
print(f"{'Avg loss':>20} ${old['avg_l']:>13,.2f} ${new['avg_l']:>13,.2f} ${new['avg_l']-old['avg_l']:>+10,.2f}")
wl_old = abs(old["avg_w"] / old["avg_l"]) if old["avg_l"] else 0
wl_new = abs(new["avg_w"] / new["avg_l"]) if new["avg_l"] else 0
print(f"{'W/L ratio':>20} {wl_old:>15.2f} {wl_new:>15.2f} {wl_new-wl_old:>+12.2f}")
print(f"{'Killed winners':>20} {old['killed']:>15} {new['killed']:>15}")

# Per symbol
print(f"\n{'='*85}")
print("PER SYMBOOL (gewijzigde symbolen)")
print(f"{'='*85}")
changed = ["EURUSD", "AUDUSD", "NZDUSD", "US100.cash", "FRA40.cash", "US30.cash"]

print(f"{'Symbol':<14} {'SL':>10} {'PnL oud':>10} {'PnL nieuw':>10} {'Verschil':>10}")
print("-" * 58)
total_improvement = 0
for sym in changed:
    o = old["sym_pnl"].get(sym, 0)
    n = new["sym_pnl"].get(sym, 0)
    diff = n - o
    total_improvement += diff
    sl_change = f"{old_sl[sym]}->{new_sl[sym]}"
    print(f"{sym:<14} {sl_change:>10} ${o:>9,.2f} ${n:>9,.2f} ${diff:>+9,.2f}")

print(f"\n{'Ongewijzigd':<14}")
for sym in ["USDJPY", "GBPUSD", "EURCHF"]:
    o = old["sym_pnl"].get(sym, 0)
    print(f"  {sym:<12} ${o:>9,.2f}  (ongewijzigd)")

# Other symbols (not in active config)
other_pnl = sum(
    old["sym_pnl"].get(s, 0)
    for s in old["sym_pnl"]
    if s not in changed and s not in ["USDJPY", "GBPUSD", "EURCHF"]
)
print(f"  {'Overige':<12} ${other_pnl:>9,.2f}  (disabled symbols)")

# Bottom line
saved = new["total"] - old["total"]
saved_losses = new["l_pnl"] - old["l_pnl"]
cost_kills = new["w_pnl"] - old["w_pnl"]

print(f"\n{'='*85}")
print("BOTTOM LINE")
print(f"{'='*85}")
print(f"  Besparing op verliezen:  ${saved_losses:>+10,.2f}  (elke loss is kleiner)")
print(f"  Kosten killed winners:   ${cost_kills:>+10,.2f}  (winners die nu gestopt worden)")
print(f"  ─────────────────────────────────────")
print(f"  NETTO BESPARING:         ${saved:>+10,.2f}  over 176 trades")
print(f"  Per trade:               ${saved/len(trades):>+10,.2f}")

weeks = 2.5  # data spans ~2.5 weeks
tpw = len(trades) / weeks
print(f"\n  Trades/week (historisch): {tpw:.0f}")
print(f"  Geschatte besparing/week:  ${saved/weeks:>+,.2f}")
print(f"  Geschatte besparing/maand: ${saved/weeks*4.3:>+,.2f}")

print(f"\n  MAAR: trailing stop wijzigingen (grotere trail_distance)")
print(f"  voor indices zijn NIET meegenomen in deze simulatie.")
print(f"  De grotere trail_distance (0.09->0.15 US100, 0.25->0.40 FRA40/US30)")
print(f"  zou de avg win voor indices VERHOGEN, wat extra verbetering geeft.")
print(f"  Dit effect is moeilijk exact te simuleren zonder volledige tick replay.")
