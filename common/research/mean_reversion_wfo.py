#!/usr/bin/env python3
"""
Mean-Reversion Walk-Forward Validation
---------------------------------------
For each equity ticker:
  1. Split data: 60% train / 40% test
  2. Find best params on train set
  3. Run those exact params on unseen test set
  4. Report out-of-sample performance

This answers: "Would optimized params actually work on future data?"

Usage:
    python3 research/mean_reversion_wfo.py
"""
import sys
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_ROOTS = [
    "/home/tradebot/ssd_data_1/tick_data",
    "/home/tradebot/ssd_data_2/tick_data",
    "/home/tradebot/data_1/tick_data",
]

INITIAL_EQUITY = 100_000
TRAIN_FRAC = 0.60

# Parameter grid
LOOKBACK_DAYS = [3, 5, 10]
MIN_MOVE_PCTS = [2.0, 3.0, 5.0]
EXIT_DAYS = [2, 3]
MAX_RISK_PCTS = [5.0]
SL_PCTS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]

# Equities only (clear winners from full backtest)
EQUITIES = [
    "AAPL", "AIRF", "ALVG", "AMZN", "BABA", "BAC", "BAYGn", "DBKGn",
    "META", "GOOG", "IBE", "LVMH", "MSFT", "NFLX", "NVDA", "PFE",
    "RACE", "T", "TSLA", "V", "VOWG_p", "WMT", "ZM",
]

# Also test the other promising asset classes
EXTRAS = [
    "USOIL.cash", "UKOIL.cash", "WHEAT.c", "COTTON.c", "COCOA.c",
    "CORN.c", "SOYBEAN.c", "US100.cash", "US500.cash", "US30.cash",
    "US2000.cash", "GER40.cash", "JP225.cash", "XAU_USD", "XPD_USD",
    "XMRUSD", "AAVUSD", "BCHUSD", "SOLUSD", "BTCUSD",
]

ALL_SYMBOLS = EQUITIES + EXTRAS

EU_EQUITIES = {"AIRF", "ALVG", "BAYGn", "DBKGn", "IBE", "LVMH", "RACE", "VOWG_p"}

SYMBOL_SWAPS = {
    "AMZN": (-4.61, -3.64), "TSLA": (-9.40, -7.44), "NVDA": (-4.23, -3.35),
    "PFE": (-0.63, -0.49), "RACE": (-8.58, -6.79), "LVMH": (-9.12, -6.88),
}

BARS_PER_DAY_MAP = {
    "equities_us": 390, "equities_eu": 510,
    "forex": 1430, "crypto": 1430,
    "cash": 1360, "metals": 1360,
}

# Commission per asset class
COMMISSION_MAP = {
    "equities": 0.004, "crypto": 0.065, "cash": 0.0,
    "metals": 0.0014, "forex": 0.005,
}


def get_symbol_info(sym):
    """Return asset class, bars_per_day, commission, swaps for a symbol."""
    if sym in EQUITIES:
        ac = "equities"
        bpd = BARS_PER_DAY_MAP["equities_eu"] if sym in EU_EQUITIES else BARS_PER_DAY_MAP["equities_us"]
    elif sym.endswith(".cash") or sym.endswith(".c"):
        ac = "cash"
        bpd = BARS_PER_DAY_MAP["cash"]
    elif sym.endswith("USD") and sym not in EQUITIES:
        ac = "crypto"
        bpd = BARS_PER_DAY_MAP["crypto"]
    elif "XA" in sym or "XP" in sym or "XC" in sym:
        ac = "metals"
        bpd = BARS_PER_DAY_MAP["metals"]
    else:
        ac = "forex"
        bpd = BARS_PER_DAY_MAP["forex"]

    comm = COMMISSION_MAP.get(ac, 0.005)
    swap_l, swap_s = SYMBOL_SWAPS.get(sym, (-5.0, -4.0))
    return ac, bpd, comm, swap_l, swap_s


def load_m1_bars(symbol):
    frames = []
    for root in DATA_ROOTS:
        p = Path(root) / symbol
        if p.exists():
            for f in sorted(p.glob("*.parquet")):
                frames.append(pl.scan_parquet(f))
    if not frames:
        return pl.DataFrame()

    ticks = pl.concat(frames).collect()

    time_col = None
    for c in ["time_msc", "timestamp", "time"]:
        if c in ticks.columns:
            time_col = c
            break
    if time_col is None:
        return pl.DataFrame()

    price_col = None
    for c in ["bid", "last", "close", "price"]:
        if c in ticks.columns:
            price_col = c
            break
    if price_col is None:
        return pl.DataFrame()

    if ticks[time_col].dtype == pl.Int64:
        ticks = ticks.with_columns(
            pl.from_epoch(pl.col(time_col), time_unit="ms").alias("dt")
        )
    else:
        ticks = ticks.with_columns(pl.col(time_col).alias("dt"))

    m1 = (
        ticks.sort("dt")
        .group_by_dynamic("dt", every="1m")
        .agg([
            pl.col(price_col).first().alias("open"),
            pl.col(price_col).max().alias("high"),
            pl.col(price_col).min().alias("low"),
            pl.col(price_col).last().alias("close"),
            pl.len().alias("volume"),
        ])
        .filter(pl.col("volume") > 0)
        .sort("dt")
    )
    return m1


def estimate_spread(closes, highs, lows):
    ranges = (highs - lows) / closes
    ranges = ranges[ranges > 0]
    if len(ranges) < 100:
        return 0.10
    return round(float(np.percentile(ranges, 10) * 100), 4)


def backtest(closes, highs, lows, lookback_bars, exit_bars,
             min_move_pct, max_risk_pct, sl_pct,
             spread_pct, commission_pct, swap_long, swap_short,
             bars_per_day):
    n = len(closes)
    if n < lookback_bars + exit_bars + 50:
        return None

    equity = INITIAL_EQUITY
    pnls = []
    i = lookback_bars

    while i < n - exit_bars:
        past_price = closes[i - lookback_bars]
        current_price = closes[i]
        if current_price <= 0 or past_price <= 0:
            i += 1
            continue
        ret_pct = (current_price - past_price) / past_price * 100

        abs_move = abs(ret_pct)
        if abs_move < min_move_pct:
            i += 1
            continue

        direction = 1 if ret_pct < 0 else -1
        risk_pct = min(abs_move, max_risk_pct) / 100.0
        position_value = equity * risk_pct
        shares = position_value / current_price
        entry_price = current_price

        if sl_pct > 0:
            if direction == 1:
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                sl_price = entry_price * (1 + sl_pct / 100)
        else:
            sl_price = None

        exit_end = min(i + exit_bars, n - 1)
        actual_exit = exit_end
        hit_sl = False

        for j in range(i + 1, exit_end + 1):
            if sl_price is not None:
                if direction == 1 and lows[j] <= sl_price:
                    actual_exit = j
                    hit_sl = True
                    break
                if direction == -1 and highs[j] >= sl_price:
                    actual_exit = j
                    hit_sl = True
                    break
            ref_price = closes[max(0, j - lookback_bars)]
            if ref_price <= 0:
                continue
            new_ret = (closes[j] - ref_price) / ref_price * 100
            if direction == 1 and new_ret >= 0:
                actual_exit = j
                break
            if direction == -1 and new_ret <= 0:
                actual_exit = j
                break

        exit_price = sl_price if hit_sl else closes[actual_exit]

        spread_cost = 2 * (entry_price * spread_pct / 100) * shares
        comm_cost = 2 * (entry_price * commission_pct / 100) * shares
        bars_held = actual_exit - i
        nights = max(1, int(np.ceil(bars_held / bars_per_day)))
        swap_per_night = swap_long if direction == 1 else swap_short
        swap_cost = abs(swap_per_night) * 0.01 * shares * nights * 1.4
        cost = spread_cost + comm_cost + swap_cost

        if direction == 1:
            pnl = (exit_price - entry_price) * shares - cost
        else:
            pnl = (entry_price - exit_price) * shares - cost

        equity += pnl
        pnls.append(pnl)
        i = actual_exit + 1

    if not pnls:
        return None

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    peak = INITIAL_EQUITY
    max_dd = 0
    eq = INITIAL_EQUITY
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "total_pnl": round(sum(pnls), 2),
        "return_pct": round((equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "pf": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        "avg_pnl": round(np.mean(pnls), 2),
    }


def optimize(closes, highs, lows, spread_pct, commission_pct,
             swap_long, swap_short, bars_per_day):
    """Find best params on this data slice."""
    grid = list(product(LOOKBACK_DAYS, MIN_MOVE_PCTS, EXIT_DAYS, MAX_RISK_PCTS, SL_PCTS))
    best_pnl = -1e18
    best_params = None
    best_result = None

    for lookback_d, min_move, exit_d, max_risk, sl_pct in grid:
        lookback_bars = int(lookback_d * bars_per_day)
        exit_bars = int(exit_d * bars_per_day)

        r = backtest(closes, highs, lows, lookback_bars, exit_bars,
                     min_move, max_risk, sl_pct,
                     spread_pct, commission_pct, swap_long, swap_short,
                     bars_per_day)
        if r and r["trades"] >= 5 and r["total_pnl"] > best_pnl:
            best_pnl = r["total_pnl"]
            best_params = (lookback_d, min_move, exit_d, max_risk, sl_pct)
            best_result = r

    return best_params, best_result


def main():
    print("=" * 110)
    print(f"  MEAN-REVERSION WALK-FORWARD VALIDATION  (train {TRAIN_FRAC:.0%} / test {1-TRAIN_FRAC:.0%})")
    print("=" * 110)

    grid_size = len(list(product(LOOKBACK_DAYS, MIN_MOVE_PCTS, EXIT_DAYS, MAX_RISK_PCTS, SL_PCTS)))
    print(f"  Grid: {grid_size} combos | Symbols: {len(ALL_SYMBOLS)}\n")

    results = []

    for i, sym in enumerate(ALL_SYMBOLS, 1):
        ac, bpd, comm, swap_l, swap_s = get_symbol_info(sym)
        print(f"  [{i:>2}/{len(ALL_SYMBOLS)}] {sym:<14} ({ac})", end=" ", flush=True)

        m1 = load_m1_bars(sym)
        if m1.is_empty():
            print("no data")
            continue

        closes = m1["close"].to_numpy()
        highs = m1["high"].to_numpy()
        lows = m1["low"].to_numpy()
        n = len(closes)
        print(f"{n:>10,} bars", end=" ", flush=True)

        spread = estimate_spread(closes, highs, lows)

        # Split: train / test
        split = int(n * TRAIN_FRAC)
        train_c, train_h, train_l = closes[:split], highs[:split], lows[:split]
        test_c, test_h, test_l = closes[split:], highs[split:], lows[split:]

        # 1. Optimize on train
        best_params, train_result = optimize(
            train_c, train_h, train_l,
            spread, comm, swap_l, swap_s, bpd
        )

        if best_params is None:
            print("→ no viable params on train")
            continue

        lookback_d, min_move, exit_d, max_risk, sl_pct = best_params

        # 2. Run same params on test (out-of-sample)
        lookback_bars = int(lookback_d * bpd)
        exit_bars = int(exit_d * bpd)
        test_result = backtest(
            test_c, test_h, test_l,
            lookback_bars, exit_bars, min_move, max_risk, sl_pct,
            spread, comm, swap_l, swap_s, bpd
        )

        sl_str = "none" if sl_pct == 0 else f"{sl_pct:.1f}%"
        params_str = f"look={lookback_d}d move={min_move:.0f}% exit={exit_d}d SL={sl_str}"

        if test_result and test_result["trades"] >= 3:
            t = test_result
            marker = "+" if t["total_pnl"] > 0 else ""
            train_pnl = train_result["total_pnl"]
            print(f"→ train=${train_pnl:>+8,.0f}  TEST=${t['total_pnl']:>+8,.0f}  "
                  f"WR={t['win_rate']:.0f}%  trades={t['trades']}  DD={t['max_dd_pct']:.1f}%  "
                  f"({params_str})")
            results.append({
                "symbol": sym,
                "asset_class": ac,
                "params": params_str,
                "train_pnl": train_result["total_pnl"],
                "train_trades": train_result["trades"],
                "train_wr": train_result["win_rate"],
                "test_pnl": test_result["total_pnl"],
                "test_trades": test_result["trades"],
                "test_wr": test_result["win_rate"],
                "test_dd": test_result["max_dd_pct"],
                "test_pf": test_result["pf"],
                "spread": spread,
            })
        else:
            print(f"→ train=${train_result['total_pnl']:>+8,.0f}  TEST=no trades  ({params_str})")

    # ==================== SUMMARY ====================
    print(f"\n{'=' * 110}")
    print(f"  WALK-FORWARD RESULTS: TRAIN vs TEST (sorted by test PnL)")
    print(f"{'=' * 110}")
    print(f"  {'Ticker':<14} {'Class':<10} {'Train PnL':>10} {'Train#':>7} "
          f"{'TEST PnL':>10} {'Test#':>6} {'WR%':>5} {'DD%':>6} {'PF':>5}  Params")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*7} "
          f"{'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*5}  {'-'*30}")

    results.sort(key=lambda x: x["test_pnl"], reverse=True)

    test_profitable = 0
    total_test_pnl = 0
    train_profitable = 0

    for r in results:
        if r["test_pnl"] > 0:
            test_profitable += 1
        if r["train_pnl"] > 0:
            train_profitable += 1
        total_test_pnl += r["test_pnl"]
        print(f"  {r['symbol']:<14} {r['asset_class']:<10} "
              f"${r['train_pnl']:>+9,.0f} {r['train_trades']:>6} "
              f"${r['test_pnl']:>+9,.0f} {r['test_trades']:>5} "
              f"{r['test_wr']:>4.0f}% {r['test_dd']:>5.1f}% {r['test_pf']:>5.2f}"
              f"  {r['params']}")

    print(f"\n  {'─' * 90}")
    print(f"  Train profitable: {train_profitable}/{len(results)}")
    print(f"  TEST profitable:  {test_profitable}/{len(results)}  "
          f"({'OVERFIT' if test_profitable < train_profitable * 0.5 else 'ROBUST' if test_profitable >= train_profitable * 0.7 else 'MIXED'})")
    print(f"  Total TEST PnL:   ${total_test_pnl:>+,.0f}")

    # Consistency check: how many tickers profitable in BOTH train AND test?
    both_positive = sum(1 for r in results if r["train_pnl"] > 0 and r["test_pnl"] > 0)
    train_pos_test_neg = sum(1 for r in results if r["train_pnl"] > 0 and r["test_pnl"] <= 0)
    print(f"  Both positive:    {both_positive}/{len(results)}")
    print(f"  Train+ but Test-: {train_pos_test_neg}/{len(results)} (overfit indicator)")

    # Per asset class
    print(f"\n{'=' * 110}")
    print(f"  PER ASSET CLASS (TEST set only)")
    print(f"{'=' * 110}")
    classes = {}
    for r in results:
        ac = r["asset_class"]
        if ac not in classes:
            classes[ac] = {"pnl": 0, "n": 0, "profitable": 0}
        classes[ac]["pnl"] += r["test_pnl"]
        classes[ac]["n"] += 1
        if r["test_pnl"] > 0:
            classes[ac]["profitable"] += 1

    for ac in sorted(classes.keys(), key=lambda x: classes[x]["pnl"], reverse=True):
        c = classes[ac]
        print(f"  {ac:<15} {c['profitable']}/{c['n']} profitable  "
              f"test_total=${c['pnl']:>+10,.0f}  avg=${c['pnl']/c['n']:>+8,.0f}")


if __name__ == "__main__":
    main()
