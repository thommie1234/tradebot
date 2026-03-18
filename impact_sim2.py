"""Correct simulation: risk_per_trade is fixed, so dollar risk per trade is constant.
Tighter SL = bigger lots = same $ loss on SL hit, but MORE $ per pip on winners.
The question is: does the bigger winner size offset the extra stopouts?"""
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

# Load trades
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

# SL config
old_sl_cfg = {
    "EURUSD": 0.84, "AUDUSD": 0.735, "NZDUSD": 0.96, "USDJPY": 0.45,
    "GBPUSD": 0.735, "EURCHF": 1.20, "US100.cash": 0.45,
    "FRA40.cash": 1.75, "US30.cash": 1.75,
}
new_sl_cfg = {
    "EURUSD": 0.63, "AUDUSD": 0.55, "NZDUSD": 0.72, "USDJPY": 0.45,
    "GBPUSD": 0.735, "ECHCHF": 1.20, "US100.cash": 0.30,
    "FRA40.cash": 1.0, "US30.cash": 1.0,
}

# Tick data helpers
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

def get_mae(symbol, direction, entry_price, start_str, end_str):
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
    prices = []
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
            col = "bid" if direction == "BUY" else "ask"
            prices.extend(f.column(col).to_pylist())
    if len(prices) < 10:
        return None
    if direction == "BUY":
        return entry_price - min(prices)
    else:
        return max(prices) - entry_price

# Enrich trades
print("Computing MAE from tick data...")
for i, t in enumerate(trades):
    entry = t["entry_price"]
    sl = t["sl_price"]
    d = t["direction"]
    exit_p = t["exit_price"]
    t["sl_dist"] = (entry - sl) if d == "BUY" else (sl - entry)
    t["is_winner"] = t["pnl"] > 0
    price_move = abs(exit_p - entry)
    t["dollar_per_point"] = abs(t["pnl"]) / price_move if price_move > 0 else 0
    # Captured price movement (signed)
    t["captured_move"] = (exit_p - entry) if d == "BUY" else (entry - exit_p)

    # SL factor for this symbol
    sym = t["symbol"]
    if sym in old_sl_cfg and sym in new_sl_cfg:
        t["sl_factor"] = new_sl_cfg[sym] / old_sl_cfg[sym]
    else:
        t["sl_factor"] = 1.0

    # Get MAE from tick data
    if sym in sym_map and t["sl_dist"] > 0:
        t["mae"] = get_mae(sym, d, entry, t["timestamp"], t["exit_timestamp"])
    else:
        t["mae"] = None

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(trades)}...")

with_mae = [t for t in trades if t["mae"] is not None]
print(f"\nTrades with MAE: {len(with_mae)} / {len(trades)}")

# =====================================================
# CORRECT SIMULATION
# =====================================================
# Key insight: risk_per_trade * equity = FIXED dollar risk
# So: lots_old = risk$ / (sl_dist_old * contract_value)
#     lots_new = risk$ / (sl_dist_new * contract_value) = lots_old * (sl_dist_old / sl_dist_new)
#
# For a LOSER hitting full SL:
#   loss_old = lots_old * sl_dist_old * CV = risk$  (always the same!)
#   loss_new = lots_new * sl_dist_new * CV = risk$  (always the same!)
#   -> DOLLAR LOSS IS IDENTICAL
#
# For a WINNER:
#   profit_old = lots_old * captured_move * CV = risk$ * (captured_move / sl_dist_old)
#   profit_new = lots_new * captured_move * CV = risk$ * (captured_move / sl_dist_new)
#   -> profit_new = profit_old * (sl_dist_old / sl_dist_new) = profit_old / sl_factor
#   -> WINNERS ARE BIGGER (because more lots, same pip capture)
#
# For a KILLED WINNER (would have been a winner but tighter SL stops it):
#   loss = risk$ (same as any SL hit)
#
# Net effect:
#   + Surviving winners are BIGGER (/ sl_factor)
#   - Some winners become losers (killed by tighter SL)
#   = Losses are IDENTICAL in dollar terms

print(f"\n{'='*85}")
print("CORRECTE SIMULATIE: risk_per_trade = vast -> dollar risk per trade = vast")
print(f"{'='*85}")

# First, estimate fixed dollar risk from actual data
# risk$ = lot_size * sl_dist * contract_value
# For forex: CV = 100000
risks = []
for t in trades:
    if t["symbol"] in ["EURUSD", "AUDUSD", "NZDUSD", "GBPUSD", "EURCHF"]:
        risk = t["lot_size"] * t["sl_dist"] * 100000
        risks.append(risk)
avg_risk = sum(risks) / len(risks) if risks else 280
print(f"\nGemiddelde dollar risk per trade: ${avg_risk:.2f}")

print(f"\n{'SL factor':>10} {'Wins':>5} {'Loss':>5} {'WR':>5} {'AvgW':>8} {'AvgL':>8} {'W/L':>6} {'Net PnL':>12} {'Kill':>5}")
print("-" * 80)

import random

for sl_factor_override in [1.0, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]:
    total_pnl = 0
    w_cnt = 0
    l_cnt = 0
    w_pnl = 0
    l_pnl = 0
    killed = 0

    for t in trades:
        sym = t["symbol"]
        # Use per-symbol factor if available, otherwise use override
        if sl_factor_override == 1.0:
            factor = 1.0
        elif sym in old_sl_cfg and sym in new_sl_cfg:
            # Scale: if override is 0.75 and symbol factor is 0.75, use symbol factor
            # Otherwise interpolate based on override
            factor = t["sl_factor"]
            # But we want to test uniform factors too
            factor = sl_factor_override
        else:
            factor = 1.0

        if factor >= 1.0 or t["sl_dist"] <= 0:
            total_pnl += t["pnl"]
            if t["pnl"] > 0:
                w_cnt += 1
                w_pnl += t["pnl"]
            else:
                l_cnt += 1
                l_pnl += t["pnl"]
            continue

        new_sl_dist = t["sl_dist"] * factor
        lot_scale = 1.0 / factor  # lots increase inversely

        if t["is_winner"]:
            # Check if winner gets killed by tighter SL
            is_killed = False
            if t["mae"] is not None:
                is_killed = t["mae"] >= new_sl_dist
            else:
                # Estimate from observed rates
                random.seed(t["id"] + 99)
                if factor <= 0.60:
                    kill_rate = 0.28
                elif factor <= 0.70:
                    kill_rate = 0.22
                elif factor <= 0.80:
                    kill_rate = 0.18
                else:
                    kill_rate = 0.12
                is_killed = random.random() < kill_rate

            if is_killed:
                killed += 1
                # Killed winner -> loss = fixed dollar risk (same as any SL hit)
                # Use original trade's risk amount
                risk_usd = t["lot_size"] * t["sl_dist"] * 100000 if t["symbol"] not in ["US100.cash", "FRA40.cash", "US30.cash", "UK100.cash", "US500.cash"] else abs(t["pnl"]) if not t["is_winner"] else avg_risk
                # Actually simpler: loss = same as avg loser for this symbol
                # Or: loss = dollar_per_point * sl_dist (original risk)
                loss = t["dollar_per_point"] * t["sl_dist"]  # original dollar risk
                l_cnt += 1
                l_pnl -= loss
                total_pnl -= loss
            else:
                # Surviving winner: profit is BIGGER because more lots
                # new_profit = old_profit * (1/factor)
                new_profit = t["pnl"] * (1.0 / factor)
                w_cnt += 1
                w_pnl += new_profit
                total_pnl += new_profit
        else:
            # Loser: dollar loss = same (risk$ is fixed)
            # The loss is the SAME regardless of SL width
            # UNLESS the trade didn't hit full SL (partial loss)
            # 94% hit full SL, so mostly the same
            total_pnl += t["pnl"]  # same loss
            l_cnt += 1
            l_pnl += t["pnl"]

    avgw = w_pnl / w_cnt if w_cnt else 0
    avgl = l_pnl / l_cnt if l_cnt else 0
    wl = abs(avgw / avgl) if avgl else 0
    wr = 100 * w_cnt / (w_cnt + l_cnt)
    print(f"{sl_factor_override:>10.2f} {w_cnt:>5} {l_cnt:>5} {wr:>4.0f}% {avgw:>8.1f} {avgl:>8.1f} {wl:>6.2f} {total_pnl:>12.2f} {killed:>5}")

# Now simulate with actual per-symbol factors from our config change
print(f"\n{'='*85}")
print("MET ONZE WERKELIJKE CONFIG WIJZIGING (per-symbool SL factor)")
print(f"{'='*85}")

for label, use_new in [("OUDE CONFIG", False), ("NIEUWE CONFIG", True)]:
    total_pnl = 0
    w_cnt = 0
    l_cnt = 0
    w_pnl = 0
    l_pnl = 0
    killed = 0

    for t in trades:
        factor = t["sl_factor"] if use_new else 1.0

        if factor >= 1.0 or t["sl_dist"] <= 0:
            total_pnl += t["pnl"]
            if t["pnl"] > 0:
                w_cnt += 1
                w_pnl += t["pnl"]
            else:
                l_cnt += 1
                l_pnl += t["pnl"]
            continue

        new_sl_dist = t["sl_dist"] * factor

        if t["is_winner"]:
            is_killed = False
            if t["mae"] is not None:
                is_killed = t["mae"] >= new_sl_dist
            else:
                random.seed(t["id"] + 77)
                if factor <= 0.60:
                    kill_rate = 0.28
                elif factor <= 0.70:
                    kill_rate = 0.22
                elif factor <= 0.80:
                    kill_rate = 0.18
                else:
                    kill_rate = 0.12
                is_killed = random.random() < kill_rate

            if is_killed:
                killed += 1
                loss = t["dollar_per_point"] * t["sl_dist"]
                l_cnt += 1
                l_pnl -= loss
                total_pnl -= loss
            else:
                new_profit = t["pnl"] * (1.0 / factor)
                w_cnt += 1
                w_pnl += new_profit
                total_pnl += new_profit
        else:
            # Same dollar loss
            total_pnl += t["pnl"]
            l_cnt += 1
            l_pnl += t["pnl"]

    avgw = w_pnl / w_cnt if w_cnt else 0
    avgl = l_pnl / l_cnt if l_cnt else 0
    wl = abs(avgw / avgl) if avgl else 0
    wr = 100 * w_cnt / (w_cnt + l_cnt)
    print(f"\n  {label}:")
    print(f"    Wins: {w_cnt}, Losses: {l_cnt}, WR: {wr:.1f}%")
    print(f"    Avg win: ${avgw:.2f}, Avg loss: ${avgl:.2f}")
    print(f"    W/L ratio: {wl:.2f}")
    print(f"    Net P&L: ${total_pnl:,.2f}")
    print(f"    Killed winners: {killed}")

print(f"\n{'='*85}")
print("UITLEG")
print(f"{'='*85}")
print("""
Met vaste risk_per_trade:
  - Krappere SL = grotere positie (meer lots)
  - Dollar verlies per SL-hit = ALTIJD HETZELFDE (~$280)
  - Maar dollar WINST per winner = GROTER (meer lots * zelfde pip capture)
  - Trade-off: sommige winners worden gedood door krappere SL

  Netto effect hangt af van:
  + Grotere winsten op overlevende winners
  - Extra verliezen door gedode winners (elk ~$280)
""")
