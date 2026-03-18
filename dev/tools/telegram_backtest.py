"""Telegram Signal Backtest — reads channel history and checks what signals would have yielded."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

from tools.telegram_signals import (
    API_ID, API_HASH, SESSION_FILE, SYMBOL_MAP,
    parse_signal, DEFAULT_CHANNELS,
)


async def scrape_history(channels: list = None, days: int = 30, limit_per_channel: int = 5000):
    """Scrape historical messages and parse all signals."""
    if channels is None:
        channels = DEFAULT_CHANNELS

    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    await client.start()
    me = await client.get_me()
    print(f"Logged in as: {me.first_name}\n")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_signals = []

    for ch in channels:
        try:
            if isinstance(ch, int):
                for try_id in [ch, int(f"-100{abs(ch)}")]:
                    try:
                        entity = await client.get_entity(try_id)
                        break
                    except Exception:
                        continue
                else:
                    print(f"  Cannot find channel {ch}")
                    continue
            else:
                entity = await client.get_entity(ch)

            title = getattr(entity, 'title', None) or getattr(entity, 'username', str(ch))
            print(f"Scanning: {title}...")

            count = 0
            signals = 0
            async for msg in client.iter_messages(entity, limit=limit_per_channel):
                if msg.date < cutoff:
                    break
                if not msg.text:
                    continue
                count += 1

                signal = parse_signal(msg.text, title)
                if signal:
                    signal.timestamp = msg.date.isoformat()
                    all_signals.append({
                        "channel": title,
                        "timestamp": msg.date.isoformat(),
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "entry_price": signal.entry_price,
                        "entry_low": signal.entry_low,
                        "entry_high": signal.entry_high,
                        "sl_price": signal.sl_price,
                        "tp_price": signal.tp_price,
                        "raw_text": signal.raw_text[:200],
                    })
                    signals += 1

            print(f"  Messages: {count}, Signals found: {signals}")

        except Exception as e:
            print(f"  Error with {ch}: {e}")

    await client.disconnect()
    return all_signals


def _load_ticks(symbol: str, start: datetime, end: datetime):
    """Load tick data from parquet files for a symbol+date range."""
    import pyarrow.parquet as pq
    import pandas as pd

    # Map MT5 symbol to tick data folder name
    tick_name_map = {
        "XAUUSD": "XAU_USD", "XAGUSD": "XAG_USD", "BTC_USD": "BTC_USD",
        "ETH_USD": "ETH_USD", "USOIL.cash": "USOIL",
    }
    # For forex: EURUSD → EURUSD (try both)
    folder_name = tick_name_map.get(symbol, symbol)

    tick_root = Path(r"C:\tick_data\ssd1")
    tick_dir = tick_root / folder_name
    if not tick_dir.exists():
        # Try underscore version: EURUSD → EUR_USD
        alt = symbol[:3] + "_" + symbol[3:]
        tick_dir = tick_root / alt
    if not tick_dir.exists():
        return None

    # Load relevant months
    months = set()
    d = start
    while d <= end:
        months.add(d.strftime("%Y-%m"))
        d += timedelta(days=28)
    months.add(end.strftime("%Y-%m"))

    frames = []
    for m in sorted(months):
        fp = tick_dir / f"{m}.parquet"
        if fp.exists():
            df = pq.read_table(str(fp)).to_pandas()
            frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # Filter to range — ensure timestamps are tz-aware
    start_ts = pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
    df = df.loc[mask].reset_index(drop=True)
    return df if len(df) > 0 else None


def backtest_signals(signals: list[dict]) -> list[dict]:
    """Backtest signals against tick-level data with full trailing stop simulation.

    For each signal, replays ticks after entry and simulates:
    - ATR-based SL/TP
    - Breakeven move
    - Trailing stop activation + trailing
    """
    try:
        from tools.mt5_bridge import mt5, initialize_mt5
        ok, _, _ = initialize_mt5()
        if not ok:
            print("MT5 not connected — using tick data only for ATR")
    except Exception:
        mt5 = None

    from config.loader import cfg
    cfg.CONFIG_PATH = str(REPO_ROOT / "bf" / "config.json")
    cfg.load()

    results = []
    for sig in signals:
        symbol = sig["symbol"]
        direction = sig["direction"]
        entry_time_str = sig["timestamp"]

        # Parse entry time
        try:
            entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
        except Exception:
            sig["result"] = "PARSE_ERROR"
            results.append(sig)
            continue

        # Get symbol config — use config if available, otherwise sensible defaults
        sym_cfg = cfg.SYMBOLS.get(symbol, {})
        if not sym_cfg:
            # Default params for symbols not in config
            defaults = {
                "XAUUSD": {"atr_timeframe": "M30", "atr_period": 14, "atr_sl_mult": 1.0,
                           "atr_tp_mult": 3.0, "breakeven_atr": 0.29, "trail_activation_atr": 0.57,
                           "trail_distance_atr": 0.40, "exit_horizon": 12},
                "BTC_USD": {"atr_timeframe": "H1", "atr_period": 14, "atr_sl_mult": 0.8,
                            "atr_tp_mult": 6.0, "breakeven_atr": 0.7, "trail_activation_atr": 2.0,
                            "trail_distance_atr": 0.8, "exit_horizon": 24},
            }
            sym_cfg = defaults.get(symbol, {
                "atr_timeframe": "H1", "atr_period": 14, "atr_sl_mult": 1.0,
                "atr_tp_mult": 2.5, "breakeven_atr": 0.3, "trail_activation_atr": 0.6,
                "trail_distance_atr": 0.35, "exit_horizon": 16,
            })

        # Load tick data for this signal
        import numpy as np

        atr_period = sym_cfg.get("atr_period", 14)
        tick_data = _load_ticks(symbol, entry_time - timedelta(days=5), entry_time + timedelta(days=3))
        if tick_data is None or len(tick_data) < 100:
            sig["result"] = "NO_TICK_DATA"
            results.append(sig)
            continue

        # Calculate ATR from tick data — resample to bars first
        tf_map = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h"}
        atr_tf = sym_cfg.get("atr_timeframe", "H1")
        resample_tf = tf_map.get(atr_tf, "1h")

        tick_data = tick_data.set_index("time")
        mid = (tick_data["bid"] + tick_data["ask"]) / 2
        bars_df = mid.resample(resample_tf).ohlc().dropna()
        if len(bars_df) < atr_period + 2:
            sig["result"] = "INSUFFICIENT_BARS"
            results.append(sig)
            continue

        # Find bars before entry
        entry_ts = entry_time
        pre_bars = bars_df[bars_df.index < entry_ts].tail(atr_period)
        if len(pre_bars) < atr_period:
            sig["result"] = "INSUFFICIENT_BARS"
            results.append(sig)
            continue

        highs = pre_bars["high"].values
        lows = pre_bars["low"].values
        closes = pre_bars["close"].values
        tr = np.maximum(highs - lows, np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1))
        ))[1:]
        atr = float(np.mean(tr))

        sl_mult = sym_cfg.get("atr_sl_mult", 1.0)
        tp_mult = sym_cfg.get("atr_tp_mult", 2.0)
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        # Entry price from signal or first tick after signal
        entry_price = sig["entry_price"]

        if direction == "BUY":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Trailing stop params
        breakeven_dist = atr * sym_cfg.get("breakeven_atr", 0.25)
        trail_activation_dist = atr * sym_cfg.get("trail_activation_atr", 0.75)
        trail_distance = atr * sym_cfg.get("trail_distance_atr", 0.4)

        # Get ticks after entry for simulation
        post_ticks = tick_data[tick_data.index >= entry_ts]
        exit_horizon_hours = sym_cfg.get("exit_horizon", 12) * {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}.get(atr_tf, 60) / 60
        max_exit_time = entry_time + timedelta(hours=exit_horizon_hours)

        result = "OPEN"
        exit_price = entry_price
        max_fav = 0.0
        max_adv = 0.0
        current_sl = sl_price
        high_water = entry_price
        be_moved = False
        trailing_active = False

        for _, tick in post_ticks.iterrows():
            if tick.name > max_exit_time:
                result = "HORIZON"
                exit_price = float((tick["bid"] + tick["ask"]) / 2)
                break

            price = float(tick["bid"]) if direction == "BUY" else float(tick["ask"])

            if direction == "BUY":
                check_price = float(tick["bid"])   # SL checked against bid
                tp_check = float(tick["ask"])       # Actually bid for BUY exit

                if float(tick["ask"]) > high_water:
                    high_water = float(tick["ask"])

                fav = high_water - entry_price
                adv = entry_price - check_price

                if not be_moved and fav >= breakeven_dist:
                    current_sl = entry_price
                    be_moved = True
                if not trailing_active and fav >= trail_activation_dist:
                    trailing_active = True
                if trailing_active:
                    trail_sl = high_water - trail_distance
                    if trail_sl > current_sl:
                        current_sl = trail_sl

                if check_price <= current_sl:
                    result = "TRAIL_SL" if trailing_active else ("BE_HIT" if be_moved else "SL_HIT")
                    exit_price = current_sl
                    break
                if check_price >= tp_price:
                    result = "TP_HIT"
                    exit_price = tp_price
                    break
            else:
                check_price = float(tick["ask"])
                if float(tick["bid"]) < high_water:
                    high_water = float(tick["bid"])

                fav = entry_price - high_water
                adv = check_price - entry_price

                if not be_moved and fav >= breakeven_dist:
                    current_sl = entry_price
                    be_moved = True
                if not trailing_active and fav >= trail_activation_dist:
                    trailing_active = True
                if trailing_active:
                    trail_sl = high_water + trail_distance
                    if trail_sl < current_sl:
                        current_sl = trail_sl

                if check_price >= current_sl:
                    result = "TRAIL_SL" if trailing_active else ("BE_HIT" if be_moved else "SL_HIT")
                    exit_price = current_sl
                    break
                if check_price <= tp_price:
                    result = "TP_HIT"
                    exit_price = tp_price
                    break

            max_fav = max(max_fav, fav)
            max_adv = max(max_adv, adv)

        if result == "OPEN":
            result = "HORIZON"

        if direction == "BUY":
            pnl_pips = exit_price - entry_price
        else:
            pnl_pips = entry_price - exit_price

        sig["entry_price_actual"] = round(entry_price, 5)
        sig["exit_price"] = round(exit_price, 5)
        sig["sl_price_bot"] = round(sl_price, 5)
        sig["tp_price_bot"] = round(tp_price, 5)
        sig["atr"] = round(atr, 5)
        sig["pnl_pips"] = round(pnl_pips, 5)
        sig["pnl_r"] = round(pnl_pips / sl_dist, 2) if sl_dist > 0 else 0
        sig["result"] = result
        sig["bars_held"] = 0
        sig["max_fav"] = round(max_fav, 5)
        sig["max_adv"] = round(max_adv, 5)

        results.append(sig)

    return results


def print_results(results: list[dict]):
    """Print backtest summary."""
    # Filter to backtested signals only
    traded = [r for r in results if r.get("result") in ("SL_HIT", "TP_HIT", "HORIZON", "TRAIL_SL", "BE_HIT")]
    if not traded:
        print("\nNo backtestable signals found.")
        return

    wins = [r for r in traded if r["pnl_r"] > 0]
    losses = [r for r in traded if r["pnl_r"] <= 0]

    print(f"\n{'='*70}")
    print(f"  TELEGRAM SIGNAL BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Total signals parsed: {len(results)}")
    print(f"  Backtested:           {len(traded)}")
    print(f"  Wins:                 {len(wins)}")
    print(f"  Losses:               {len(losses)}")
    wr = len(wins) / len(traded) * 100 if traded else 0
    print(f"  Win Rate:             {wr:.1f}%")
    avg_r = sum(r["pnl_r"] for r in traded) / len(traded) if traded else 0
    total_r = sum(r["pnl_r"] for r in traded)
    print(f"  Avg R:                {avg_r:+.2f}R")
    print(f"  Total R:              {total_r:+.2f}R")
    print(f"{'='*70}")

    # Per channel
    channels = set(r["channel"] for r in traded)
    print(f"\n  Per Channel:")
    print(f"  {'Channel':<25} {'Trades':>6} {'WR':>6} {'Total R':>8} {'Avg R':>7}")
    print(f"  {'-'*55}")
    for ch in sorted(channels):
        ch_trades = [r for r in traded if r["channel"] == ch]
        ch_wins = [r for r in ch_trades if r["pnl_r"] > 0]
        ch_wr = len(ch_wins) / len(ch_trades) * 100 if ch_trades else 0
        ch_total_r = sum(r["pnl_r"] for r in ch_trades)
        ch_avg_r = ch_total_r / len(ch_trades) if ch_trades else 0
        print(f"  {ch:<25} {len(ch_trades):>6} {ch_wr:>5.0f}% {ch_total_r:>+7.1f}R {ch_avg_r:>+6.2f}R")

    # Per symbol
    symbols = set(r["symbol"] for r in traded)
    print(f"\n  Per Symbol:")
    print(f"  {'Symbol':<14} {'Trades':>6} {'WR':>6} {'Total R':>8} {'Avg R':>7}")
    print(f"  {'-'*45}")
    for sym in sorted(symbols):
        sym_trades = [r for r in traded if r["symbol"] == sym]
        sym_wins = [r for r in sym_trades if r["pnl_r"] > 0]
        sym_wr = len(sym_wins) / len(sym_trades) * 100 if sym_trades else 0
        sym_total_r = sum(r["pnl_r"] for r in sym_trades)
        sym_avg_r = sym_total_r / len(sym_trades) if sym_trades else 0
        print(f"  {sym:<14} {len(sym_trades):>6} {sym_wr:>5.0f}% {sym_total_r:>+7.1f}R {sym_avg_r:>+6.2f}R")

    # Recent signals
    print(f"\n  Last 10 Signals:")
    print(f"  {'Time':<18} {'Ch':<15} {'Sym':<10} {'Dir':<5} {'Result':<8} {'R':>6}")
    print(f"  {'-'*65}")
    for r in sorted(traded, key=lambda x: x["timestamp"], reverse=True)[:10]:
        ts = r["timestamp"][:16].replace("T", " ")
        ch = r["channel"][:14]
        color = "+" if r["pnl_r"] > 0 else ""
        print(f"  {ts:<18} {ch:<15} {r['symbol']:<10} {r['direction']:<5} {r['result']:<8} {r['pnl_r']:>+5.1f}R")

    # Skipped signals
    skipped = [r for r in results if r.get("result") not in ("SL_HIT", "TP_HIT", "HORIZON")]
    if skipped:
        reasons = {}
        for r in skipped:
            reason = r.get("result", "UNKNOWN")
            reasons[reason] = reasons.get(reason, 0) + 1
        print(f"\n  Skipped: {len(skipped)} signals")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Telegram Signal Backtest")
    parser.add_argument("--days", type=int, default=30, help="Days of history to scan")
    parser.add_argument("--limit", type=int, default=5000, help="Max messages per channel")
    parser.add_argument("--no-mt5", action="store_true", help="Skip MT5 backtest, just show parsed signals")
    args = parser.parse_args()

    print("Scraping Telegram channel history...\n")
    signals = await scrape_history(days=args.days, limit_per_channel=args.limit)

    print(f"\nTotal signals parsed: {len(signals)}")

    if args.no_mt5:
        # Just show parsed signals
        for s in signals:
            print(f"  {s['timestamp'][:16]} {s['channel'][:15]:<15} {s['direction']:<5} {s['symbol']:<12} entry={s['entry_price']:.5f}")
    else:
        print("\nBacktesting against MT5 price data...")
        results = backtest_signals(signals)
        print_results(results)

    # Save raw data
    out_path = str(REPO_ROOT / "bf" / "audit" / "telegram_backtest.json")
    with open(out_path, "w") as f:
        json.dump(signals if args.no_mt5 else results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
