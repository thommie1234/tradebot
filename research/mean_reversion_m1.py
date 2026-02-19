#!/usr/bin/env python3
"""
Mean-Reversion M1 Backtest — ALL TICKERS, Per-Ticker Optimization
-----------------------------------------------------------------
Strategy: If an instrument moves X% in N days → trade opposite direction.
Auto-discovers all tickers with tick data, applies realistic costs per asset class.

Usage:
    python3 research/mean_reversion_m1.py
"""
import csv
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

INFO_DIR = REPO / "trading_prop" / "information"

INITIAL_EQUITY = 100_000

# Parameter grid
LOOKBACK_DAYS = [3, 5, 10]
MIN_MOVE_PCTS = [2.0, 3.0, 5.0]
EXIT_DAYS = [2, 3]
MAX_RISK_PCTS = [5.0]
SL_PCTS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]

# Real swap data (from MT5 bridge, in points; 1 point = $0.01 for stocks)
SYMBOL_SWAPS = {
    "AMZN": (-4.61, -3.64), "TSLA": (-9.40, -7.44), "NVDA": (-4.23, -3.35),
    "PFE": (-0.63, -0.49), "RACE": (-8.58, -6.79), "LVMH": (-9.12, -6.88),
}

# Bars per trading day per asset class
BARS_PER_DAY = {
    "equities_us": 390,   # 6.5h
    "equities_eu": 510,   # 8.5h
    "forex": 1430,        # ~24h
    "crypto": 1430,       # ~24h
    "cash": 1360,         # ~22h
    "metals": 1360,       # ~22h
}

# EU-traded equities
EU_EQUITIES = {"AIRF", "ALVG", "BAYGn", "DBKGn", "IBE", "LVMH", "RACE", "VOWG_p"}


def build_symbol_catalog() -> dict:
    """Build catalog of all tradeable symbols with costs per asset class."""
    catalog = {}

    # Equities: 0.004% commission
    equities = [
        "AAPL", "AIRF", "ALVG", "AMZN", "BABA", "BAC", "BAYGn", "DBKGn",
        "META", "GOOG", "IBE", "LVMH", "MSFT", "NFLX", "NVDA", "PFE",
        "RACE", "T", "TSLA", "V", "VOWG_p", "WMT", "ZM",
    ]
    for sym in equities:
        bpd = BARS_PER_DAY["equities_eu"] if sym in EU_EQUITIES else BARS_PER_DAY["equities_us"]
        swap_l, swap_s = SYMBOL_SWAPS.get(sym, (-5.0, -4.0))
        catalog[sym] = {
            "display": sym, "asset_class": "equities",
            "commission_pct": 0.004, "bars_per_day": bpd,
            "swap_long": swap_l, "swap_short": swap_s,
        }

    # Forex: $5/lot ≈ 0.005% commission
    forex = [
        "AUD_CAD", "AUD_JPY", "AUD_NZD", "AUD_CHF", "AUD_USD",
        "GBP_AUD", "GBP_CAD", "GBP_JPY", "GBP_NZD", "GBP_CHF", "GBP_USD",
        "CAD_JPY", "CAD_CHF",
        "EUR_AUD", "EUR_GBP", "EUR_CAD", "EUR_JPY", "EUR_CHF", "EUR_USD", "EUR_NZD",
        "NZD_CAD", "NZD_CHF", "NZD_JPY", "NZD_USD",
        "CHF_JPY", "USD_CAD", "USD_CHF", "USD_JPY",
    ]
    for sym in forex:
        catalog[sym] = {
            "display": sym.replace("_", "/"), "asset_class": "forex",
            "commission_pct": 0.005, "bars_per_day": BARS_PER_DAY["forex"],
            "swap_long": -5.0, "swap_short": -4.0,
        }

    # Forex exotic: $5/lot
    forex_exotic = [
        "EUR_CZK", "EUR_HUF", "EUR_NOK", "EUR_PLN",
        "USD_CNH", "USD_CZK", "USD_HKD", "USD_HUF", "USD_ILS",
        "USD_MXN", "USD_NOK", "USD_PLN", "USD_SGD", "USD_ZAR", "USD_SEK",
    ]
    for sym in forex_exotic:
        catalog[sym] = {
            "display": sym.replace("_", "/"), "asset_class": "forex_exotic",
            "commission_pct": 0.005, "bars_per_day": BARS_PER_DAY["forex"],
            "swap_long": -8.0, "swap_short": -6.0,
        }

    # Crypto: 0.065% commission
    crypto = [
        "BTCUSD", "DASHUSD", "ETHUSD", "LTCUSD", "XRPUSD", "XMRUSD",
        "NEOUSD", "ADAUSD", "DOTUSD", "DOGEUSD", "SOLUSD", "AVAUSD",
        "BCHUSD", "ETCUSD", "BNBUSD", "SANUSD", "LNKUSD", "NERUSD",
        "ALGUSD", "ICPUSD", "AAVUSD", "BARUSD", "GALUSD", "GRTUSD",
        "IMXUSD", "MANUSD", "VECUSD", "XLMUSD", "UNIUSD", "FETUSD", "XTZUSD",
    ]
    for sym in crypto:
        catalog[sym] = {
            "display": sym, "asset_class": "crypto",
            "commission_pct": 0.065, "bars_per_day": BARS_PER_DAY["crypto"],
            "swap_long": -10.0, "swap_short": -8.0,
        }

    # Cash/Indices: mostly no commission (some 0.0014%)
    cash_no_comm = [
        "AUS200.cash", "US30.cash", "SPN35.cash", "EU50.cash", "FRA40.cash",
        "GER40.cash", "HK50.cash", "JP225.cash", "N25.cash", "US100.cash",
        "US500.cash", "UK100.cash", "UKOIL.cash", "USOIL.cash", "US2000.cash",
        "COCOA.c", "COFFEE.c", "CORN.c", "SOYBEAN.c", "WHEAT.c",
        "COTTON.c", "SUGAR.c",
    ]
    cash_with_comm = ["NATGAS.cash", "DXY.cash", "HEATOIL.c"]
    for sym in cash_no_comm:
        catalog[sym] = {
            "display": sym, "asset_class": "cash",
            "commission_pct": 0.0, "bars_per_day": BARS_PER_DAY["cash"],
            "swap_long": -5.0, "swap_short": -4.0,
        }
    for sym in cash_with_comm:
        catalog[sym] = {
            "display": sym, "asset_class": "cash",
            "commission_pct": 0.0014, "bars_per_day": BARS_PER_DAY["cash"],
            "swap_long": -5.0, "swap_short": -4.0,
        }

    # Metals: 0.0014% commission
    metals = [
        "XAG_USD", "XAG_EUR", "XAG_AUD",
        "XAU_USD", "XAU_EUR", "XAU_AUD",
        "XPD_USD", "XPT_USD", "XCU_USD",
    ]
    for sym in metals:
        catalog[sym] = {
            "display": sym.replace("_", "/"), "asset_class": "metals",
            "commission_pct": 0.0014, "bars_per_day": BARS_PER_DAY["metals"],
            "swap_long": -5.0, "swap_short": -4.0,
        }

    return catalog


def discover_tickers_with_data(catalog: dict) -> list:
    """Find which catalog symbols actually have parquet data."""
    found = []
    for data_name in sorted(catalog.keys()):
        for root in DATA_ROOTS:
            p = Path(root) / data_name
            if p.exists() and list(p.glob("*.parquet")):
                found.append(data_name)
                break
    return found


def load_m1_bars(symbol: str) -> pl.DataFrame:
    """Load tick data and resample to M1 bars."""
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


def backtest(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
             lookback_bars: int, exit_bars: int,
             min_move_pct: float, max_risk_pct: float, sl_pct: float,
             spread_pct: float = 0.05, commission_pct: float = 0.0,
             swap_long: float = 0.0, swap_short: float = 0.0,
             bars_per_day: int = 390) -> dict:
    """Run mean-reversion backtest with realistic costs."""
    n = len(closes)
    if n < lookback_bars + exit_bars + 100:
        return None

    equity = INITIAL_EQUITY
    pnls = []
    total_costs = 0.0
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

        # SL price
        if sl_pct > 0:
            if direction == 1:
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                sl_price = entry_price * (1 + sl_pct / 100)
        else:
            sl_price = None

        # Exit logic
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

        # Costs: spread (round-trip) + commission (round-trip) + swap
        spread_cost = 2 * (entry_price * spread_pct / 100) * shares
        comm_cost = 2 * (entry_price * commission_pct / 100) * shares

        bars_held = actual_exit - i
        nights = max(1, int(np.ceil(bars_held / bars_per_day)))
        swap_per_night = swap_long if direction == 1 else swap_short
        point_value = 0.01
        swap_cost = abs(swap_per_night) * point_value * shares * nights * 1.4

        cost = spread_cost + comm_cost + swap_cost
        total_costs += cost

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
        "total_costs": round(total_costs, 2),
    }


def optimize_ticker(sym_data: dict, info: dict) -> list:
    """Run full grid search for one ticker, return list of results."""
    grid = list(product(LOOKBACK_DAYS, MIN_MOVE_PCTS, EXIT_DAYS, MAX_RISK_PCTS, SL_PCTS))
    results = []

    for lookback_d, min_move, exit_d, max_risk, sl_pct in grid:
        lookback_bars = int(lookback_d * info["bars_per_day"])
        exit_bars = int(exit_d * info["bars_per_day"])

        r = backtest(
            sym_data["close"], sym_data["high"], sym_data["low"],
            lookback_bars, exit_bars, min_move, max_risk, sl_pct,
            spread_pct=info.get("spread_pct", 0.05),
            commission_pct=info["commission_pct"],
            swap_long=info["swap_long"],
            swap_short=info["swap_short"],
            bars_per_day=info["bars_per_day"],
        )
        if r and r["trades"] >= 5:
            r["lookback_d"] = lookback_d
            r["min_move"] = min_move
            r["exit_d"] = exit_d
            r["sl_pct"] = sl_pct
            results.append(r)

    results.sort(key=lambda x: x["total_pnl"], reverse=True)
    return results


def estimate_spread_from_data(closes: np.ndarray, highs: np.ndarray,
                              lows: np.ndarray) -> float:
    """Estimate typical spread as % using high-low range at M1 level."""
    # Minimum intrabar range is a proxy for spread
    ranges = (highs - lows) / closes
    ranges = ranges[ranges > 0]
    if len(ranges) < 100:
        return 0.10  # fallback
    # Use 10th percentile of range as spread estimate
    return round(float(np.percentile(ranges, 10) * 100), 4)


def main():
    print("=" * 100)
    print("  MEAN-REVERSION M1 — ALL TICKERS, PER-TICKER OPTIMIZATION (REALISTIC COSTS)")
    print("=" * 100)

    # Build catalog from FTMO CSVs
    catalog = build_symbol_catalog()
    print(f"  Catalog: {len(catalog)} symbols from FTMO specs")

    # Discover which ones have tick data
    available = discover_tickers_with_data(catalog)
    print(f"  With data: {len(available)} symbols")

    # Also find tickers with data but NOT in catalog (bonus data)
    all_data_dirs = set()
    for root in DATA_ROOTS:
        rp = Path(root)
        if rp.exists():
            for d in rp.iterdir():
                if d.is_dir() and list(d.glob("*.parquet")):
                    all_data_dirs.add(d.name)
    extra = sorted(all_data_dirs - set(catalog.keys()))
    if extra:
        print(f"  Extra tickers (not in FTMO catalog, skipping): {', '.join(extra)}")

    grid_size = len(list(product(LOOKBACK_DAYS, MIN_MOVE_PCTS, EXIT_DAYS, MAX_RISK_PCTS, SL_PCTS)))
    print(f"  Grid: {grid_size} combos per ticker = {grid_size * len(available)} total runs\n")

    # Load data and optimize per ticker
    ticker_results = {}
    for i, sym in enumerate(available, 1):
        info = catalog[sym]
        print(f"  [{i:>3}/{len(available)}] Loading {sym:<12} ({info['asset_class']})...", end=" ", flush=True)

        m1 = load_m1_bars(sym)
        if m1.is_empty():
            print("no data")
            continue

        sym_data = {
            "close": m1["close"].to_numpy(),
            "high": m1["high"].to_numpy(),
            "low": m1["low"].to_numpy(),
        }
        n_bars = len(sym_data["close"])
        print(f"{n_bars:>10,} bars", end=" ", flush=True)

        # Estimate spread from data if not in our hardcoded list
        info["spread_pct"] = estimate_spread_from_data(
            sym_data["close"], sym_data["high"], sym_data["low"]
        )

        results = optimize_ticker(sym_data, info)
        if results:
            best = results[0]
            sl_str = "none" if best["sl_pct"] == 0 else f"{best['sl_pct']:.1f}%"
            marker = "+" if best["total_pnl"] > 0 else ""
            print(f"→ best: ${marker}{best['total_pnl']:>10,.0f}  "
                  f"(look={best['lookback_d']}d move={best['min_move']:.0f}% "
                  f"exit={best['exit_d']}d SL={sl_str}  "
                  f"trades={best['trades']}  WR={best['win_rate']:.0f}%  "
                  f"spread≈{info['spread_pct']:.3f}%)")
            ticker_results[sym] = {
                "best": best,
                "all": results,
                "info": info,
                "n_bars": n_bars,
            }
        else:
            print("→ no viable combos")

    # ==================== SUMMARY ====================
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY: BEST PARAMS PER TICKER (sorted by PnL)")
    print(f"{'=' * 100}")
    print(f"  {'Ticker':<12} {'Class':<12} {'Look':>5} {'Move':>5} {'Exit':>5} {'SL':>5} "
          f"{'Trades':>7} {'WR%':>6} {'PnL':>12} {'MaxDD':>7} {'PF':>6} {'Costs':>9} {'Spread':>7}")
    print(f"  {'-'*12} {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*7} {'-'*6} {'-'*12} {'-'*7} {'-'*6} {'-'*9} {'-'*7}")

    sorted_tickers = sorted(ticker_results.items(), key=lambda x: x[1]["best"]["total_pnl"], reverse=True)

    total_pnl = 0
    profitable = 0
    for sym, data in sorted_tickers:
        b = data["best"]
        info = data["info"]
        sl_str = "none" if b["sl_pct"] == 0 else f"{b['sl_pct']:.1f}%"
        marker = "+" if b["total_pnl"] > 0 else ""
        total_pnl += b["total_pnl"]
        if b["total_pnl"] > 0:
            profitable += 1
        print(f"  {sym:<12} {info['asset_class']:<12} {b['lookback_d']:>4}d {b['min_move']:>4.0f}% "
              f"{b['exit_d']:>4}d {sl_str:>5} {b['trades']:>7} {b['win_rate']:>5.1f}% "
              f"${b['total_pnl']:>+10,.0f} {b['max_dd_pct']:>6.2f}% {b['pf']:>5.2f} "
              f"${b['total_costs']:>8,.0f} {info['spread_pct']:>6.3f}%")

    print(f"\n  {'─' * 80}")
    print(f"  Total PnL (all tickers, each with own best params): ${total_pnl:>+,.0f}")
    print(f"  Profitable tickers: {profitable}/{len(ticker_results)}")

    # Per asset-class summary
    print(f"\n{'=' * 100}")
    print(f"  PER ASSET CLASS SUMMARY")
    print(f"{'=' * 100}")
    classes = {}
    for sym, data in sorted_tickers:
        ac = data["info"]["asset_class"]
        if ac not in classes:
            classes[ac] = {"pnl": 0, "tickers": 0, "profitable": 0}
        classes[ac]["pnl"] += data["best"]["total_pnl"]
        classes[ac]["tickers"] += 1
        if data["best"]["total_pnl"] > 0:
            classes[ac]["profitable"] += 1

    for ac in sorted(classes.keys(), key=lambda x: classes[x]["pnl"], reverse=True):
        c = classes[ac]
        print(f"  {ac:<15} {c['profitable']}/{c['tickers']} profitable  "
              f"total=${c['pnl']:>+10,.0f}  avg=${c['pnl']/c['tickers']:>+8,.0f}")

    # Top 10 most robust (best PnL / lowest DD ratio)
    print(f"\n{'=' * 100}")
    print(f"  TOP 10 RISK-ADJUSTED (PnL / MaxDD ratio)")
    print(f"{'=' * 100}")
    risk_adj = []
    for sym, data in sorted_tickers:
        b = data["best"]
        if b["max_dd_pct"] > 0 and b["total_pnl"] > 0:
            ratio = b["total_pnl"] / b["max_dd_pct"]
            risk_adj.append((sym, ratio, data))
    risk_adj.sort(key=lambda x: x[1], reverse=True)

    for sym, ratio, data in risk_adj[:10]:
        b = data["best"]
        sl_str = "none" if b["sl_pct"] == 0 else f"{b['sl_pct']:.1f}%"
        print(f"  {sym:<12} PnL/DD={ratio:>8,.0f}  PnL=${b['total_pnl']:>+10,.0f}  "
              f"DD={b['max_dd_pct']:.2f}%  (look={b['lookback_d']}d move={b['min_move']:.0f}% "
              f"exit={b['exit_d']}d SL={sl_str})")


if __name__ == "__main__":
    main()
