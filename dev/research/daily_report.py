#!/usr/bin/env python3
"""
Daily trading analysis report generator.

Collects all trading data from both accounts (FTMO + BF) for a given day,
sends to a local LLM (Ollama), and writes a structured markdown report.

Usage:
    python3 common/analysis/daily_report.py                  # yesterday
    python3 common/analysis/daily_report.py --date 2026-03-03
    python3 common/analysis/daily_report.py --model qwen2.5:32b
"""

import argparse
import json
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

ACCOUNTS = {
    "FTMO": {
        "db": REPO_ROOT / "ftmo" / "audit" / "sovereign_log.db",
        "port": 5056,
        "start_equity": 100_000,
    },
    "BF": {
        "db": REPO_ROOT / "bf" / "audit" / "sovereign_log.db",
        "port": 5057,
        "start_equity": 100_000,
    },
}

OUTPUT_DIR = REPO_ROOT / "research" / "daily"
DEFAULT_MODEL = "qwen2.5:32b"
OLLAMA_URL = "http://localhost:11434/api/generate"

MARKET_BENCHMARKS = [
    # (symbol, label, category)
    ("US100.cash", "Nasdaq 100", "US Indices"),
    ("US500.cash", "S&P 500", "US Indices"),
    ("US30.cash", "Dow Jones", "US Indices"),
    ("US2000.cash", "Russell 2000", "US Indices"),
    ("GER40.cash", "DAX", "EU Indices"),
    ("UK100.cash", "FTSE 100", "EU Indices"),
    ("FRA40.cash", "CAC 40", "EU Indices"),
    ("EU50.cash", "Euro Stoxx 50", "EU Indices"),
    ("JP225.cash", "Nikkei 225", "Asia Indices"),
    ("BTCUSD", "Bitcoin", "Crypto"),
    ("ETHUSD", "Ethereum", "Crypto"),
    ("EURUSD", "EUR/USD", "Forex"),
    ("GBPUSD", "GBP/USD", "Forex"),
    ("USDJPY", "USD/JPY", "Forex"),
    ("DXY.cash", "Dollar Index", "Forex"),
    ("USOIL.cash", "WTI Crude Oil", "Commodities"),
    ("NATGAS.cash", "Natural Gas", "Commodities"),
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def get_trades(db_path: Path, date: str) -> list[dict]:
    """Get all trades for a given date from audit DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Trades opened on this date
    cur.execute("""
        SELECT ticket, symbol, direction, lot_size, entry_price, exit_price,
               pnl, status, ml_confidence, timestamp, exit_timestamp,
               tp_price, sl_price, spread_pct
        FROM trades
        WHERE DATE(timestamp) = ? OR DATE(exit_timestamp) = ?
        ORDER BY timestamp
    """, (date, date))

    trades = [dict(row) for row in cur.fetchall()]
    conn.close()
    return trades


def get_deal_history(port: int, date: str) -> list[dict]:
    """Get deal history from MT5 for fee/commission data."""
    try:
        import MetaTrader5 as mt5

        dt = datetime.strptime(date, "%Y-%m-%d")
        start = dt.replace(hour=0, minute=0, second=0)
        end = dt.replace(hour=23, minute=59, second=59)
        deals = mt5.history_deals_get(start, end + timedelta(seconds=1))

        if not deals:
            return []

        result = []
        for d in deals:
            if not hasattr(d, 'symbol') or not d.symbol or d.type not in (0, 1):
                continue
            result.append({
                "ticket": d.ticket,
                "position_id": d.position_id,
                "symbol": d.symbol,
                "type": "BUY" if d.type == 0 else "SELL",
                "entry": "IN" if d.entry == 0 else "OUT" if d.entry == 1 else str(d.entry),
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "commission": d.commission,
                "swap": getattr(d, "swap", 0),
                "fee": getattr(d, "fee", 0),
            })
        return result
    except Exception as e:
        print(f"  Warning: could not fetch deals from MT5 port {port}: {e}")
        return []


def compute_account_stats(trades: list[dict], deals: list[dict]) -> dict:
    """Compute summary statistics for an account's daily activity."""
    closed = [t for t in trades if t["status"] == "CLOSED" and t["pnl"] is not None]
    opened = [t for t in trades if t["timestamp"] and t["timestamp"][:10] == trades[0]["timestamp"][:10]] if trades else []
    rejected = [t for t in trades if "REJECTED" in (t["status"] or "")]
    filled_open = [t for t in trades if t["status"] == "FILLED"]

    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = sum(abs(t["pnl"]) for t in losses) if losses else 0
    net_pnl = sum(t["pnl"] for t in closed)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Fee data from deals
    closing_deals = [d for d in deals if d["entry"] == "OUT"]
    total_commission = sum(abs(d["commission"]) for d in deals)
    total_swap = sum(abs(d.get("swap", 0)) for d in deals)
    total_fees = total_commission + total_swap

    # Per-symbol breakdown
    by_symbol = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in closed:
        s = by_symbol[t["symbol"]]
        s["trades"] += 1
        s["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            s["wins"] += 1

    # Confidence stats
    conf_brackets = {}
    for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        bracket_trades = [t for t in closed if t["ml_confidence"] and low <= t["ml_confidence"] < high]
        if bracket_trades:
            bw = sum(1 for t in bracket_trades if t["pnl"] > 0)
            bp = sum(t["pnl"] for t in bracket_trades)
            conf_brackets[f"{low:.1f}-{high:.1f}"] = {
                "trades": len(bracket_trades), "wins": bw, "pnl": round(bp, 2)
            }

    # Rejection reasons
    rejection_reasons = defaultdict(int)
    for t in rejected:
        rejection_reasons[t["status"]] += 1

    return {
        "n_closed": len(closed),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": len(wins) / len(closed) * 100 if closed else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0,
        "best_trade": round(max((t["pnl"] for t in closed), default=0), 2),
        "worst_trade": round(min((t["pnl"] for t in closed), default=0), 2),
        "total_commission": round(total_commission, 2),
        "total_swap": round(total_swap, 2),
        "total_fees": round(total_fees, 2),
        "n_rejected": len(rejected),
        "rejection_reasons": dict(rejection_reasons),
        "n_still_open": len(filled_open),
        "by_symbol": {k: v for k, v in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)},
        "confidence_brackets": conf_brackets,
        "trades": [
            {
                "ticket": t["ticket"],
                "symbol": t["symbol"],
                "direction": t["direction"],
                "lot_size": t["lot_size"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "pnl": t["pnl"],
                "status": t["status"],
                "confidence": round(t["ml_confidence"], 3) if t["ml_confidence"] else None,
                "opened": t["timestamp"][11:16] if t["timestamp"] else None,
                "closed": t["exit_timestamp"][11:16] if t["exit_timestamp"] else None,
                "sl": t["sl_price"],
                "tp": t["tp_price"],
            }
            for t in trades if t["status"] not in ("REJECTED_SAME_DIR", "REJECTED_MARKET_CLOSED")
        ],
    }


def get_account_equity(port: int) -> dict | None:
    """Get current account equity from MT5."""
    try:
        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if acc:
            return {"equity": acc.equity, "balance": acc.balance}
    except Exception:
        pass
    return None


def get_market_conditions(port: int, date: str) -> list[dict]:
    """Fetch daily OHLC for market benchmarks to provide macro context.

    Uses the target date's daily bar and the previous day's close to compute
    the daily change. Works for both current and historical dates.
    """
    try:
        import MetaTrader5 as mt5
    except Exception:
        return []

    dt = datetime.strptime(date, "%Y-%m-%d")
    results = []

    for symbol, label, category in MARKET_BENCHMARKS:
        try:
            # Fetch daily bars covering our target date + a few days before
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1,
                                         dt - timedelta(days=7), dt + timedelta(days=1))
            if not rates or len(rates) < 2:
                continue

            # Find the bar matching our target date and the one before it
            target_bar = prev_bar = None
            for i, r in enumerate(rates):
                bar_date = datetime.utcfromtimestamp(r["time"]).strftime("%Y-%m-%d")
                if bar_date == date:
                    target_bar = r
                    if i > 0:
                        prev_bar = rates[i - 1]
                    break

            if not target_bar or not prev_bar:
                continue

            prev_close = prev_bar["close"]
            change_pct = (target_bar["close"] - prev_close) / prev_close * 100
            range_pct = (target_bar["high"] - target_bar["low"]) / prev_close * 100

            results.append({
                "symbol": symbol,
                "label": label,
                "category": category,
                "open": target_bar["open"],
                "high": target_bar["high"],
                "low": target_bar["low"],
                "close": target_bar["close"],
                "prev_close": prev_close,
                "change_pct": round(change_pct, 2),
                "range_pct": round(range_pct, 2),
                "volume": target_bar["tick_volume"],
            })
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Je bent een senior trading analyst die dagelijkse performance rapporten schrijft voor een algoritmisch tradingplatform met twee prop firm accounts.

Je krijgt:
1. Ruwe handelsdata van FTMO en BrightFunded
2. Marktcondities (dagelijkse koersen van indices, crypto, forex, commodities)

Schrijf in het Nederlands. Wees STRIKT data-gedreven. Elke bewering moet kloppen met de aangeleverde cijfers — controleer getallen voordat je vergelijkingen maakt (bijv. als FTMO DD=2.30% en BF DD=1.53%, dan is FTMO's DD HOGER, niet lager).

## Rapport structuur

### 1. Header
Datum, dag, en een one-liner samenvatting van de dag (bijv. "Rode dag voor indices, BTC veerkrachtig").

### 2. Marktcondities
Korte analyse van wat de brede markt deed:
- Welke asset classes stegen/daalden en hoeveel?
- Was er hoge volatiliteit (range%)?
- Opvallende macro-bewegingen (bijv. olie +6%, dollar sterk)?
- Hoe verhoudt onze trading zich tot de markt? (handelden we mee of tegen de trend?)

### 3. Account Performance
Per account (FTMO, BF):
- P&L, winrate, profit factor
- Fees breakdown (commissie + swap)
- Equity stand en drawdown
- Trade tabel: tijd open→close, symbool, richting, lots, entry→exit, P&L, confidence

### 4. Analyse
- **Winnaars & Verliezers**: Wat waren de beste en slechtste trades? Waarom? Verband met marktbeweging?
- **Confidence vs Performance**: Presteren hoge-confidence trades (>0.8) beter dan lage? Geef exacte cijfers per bracket.
- **Asset Class Breakdown**: Welke asset classes waren winstgevend, welke niet?
- **Timing**: Op welke momenten van de dag werd er gehandeld? Were er clusters?
- **Rejections**: Hoeveel signalen werden geblokkeerd en waarom? Waren dat achteraf goede of slechte blocks?

### 5. Vergelijking FTMO vs BF
Side-by-side tabel. Verklaar de verschillen (bijv. andere symbolen, andere tijdframes, andere fees).

### 6. Takeaways & Risico's
- 3-5 concrete, actionable conclusies
- Specifieke risico's voor morgen (open posities, drawdown niveau, marktrisico)
- Benadruk als de bot consistently verliest op bepaalde symbolen of asset classes

## Regels
- NOOIT getallen verdraaien. Als je twijfelt, citeer het exacte getal uit de data.
- Vergelijk ALTIJD met de marktbeweging: een verlies op een -5% index-dag is anders dan op een +2% dag.
- Meld ALTIJD als confidence >0.9 trades verlies opleveren — dat duidt op model-overfitting.
- Fees apart benoemen: de trader moet weten wat naar de broker gaat.
- FTMO limieten: max 5% daily loss, max 10% total drawdown — bereken hoeveel ruimte er nog is.
- BrightFunded: 0 commissie maar verdient via spread + swap — vergelijk effectieve kosten met FTMO."""


def build_prompt(date: str, account_data: dict[str, dict],
                 market: list[dict] | None = None) -> str:
    """Build the data prompt for the LLM."""
    weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%A")

    sections = [f"# Handelsdata voor {date} ({weekday})\n"]

    # Market conditions section
    if market:
        sections.append("## Marktcondities")
        by_cat = defaultdict(list)
        for m in market:
            by_cat[m["category"]].append(m)

        for cat, items in by_cat.items():
            sections.append(f"\n### {cat}")
            for m in items:
                arrow = "+" if m["change_pct"] > 0.1 else "-" if m["change_pct"] < -0.1 else "="
                sections.append(
                    f"  {arrow} {m['label']:20s} {m['close']:>12,.2f} "
                    f"({m['change_pct']:+.2f}%) range={m['range_pct']:.2f}%"
                )

        # Summary stats
        avg_us = [m["change_pct"] for m in market if "US " in m["label"] or "Nasdaq" in m["label"] or "Dow" in m["label"] or "Russell" in m["label"] or "S&P" in m["label"]]
        avg_eu = [m["change_pct"] for m in market if m["category"] == "EU Indices"]
        avg_crypto = [m["change_pct"] for m in market if m["category"] == "Crypto"]

        sections.append("\n### Samenvatting")
        if avg_us:
            sections.append(f"  US Indices gemiddeld: {sum(avg_us)/len(avg_us):+.2f}%")
        if avg_eu:
            sections.append(f"  EU Indices gemiddeld: {sum(avg_eu)/len(avg_eu):+.2f}%")
        if avg_crypto:
            sections.append(f"  Crypto gemiddeld: {sum(avg_crypto)/len(avg_crypto):+.2f}%")
        sections.append("")

    for name, data in account_data.items():
        stats = data["stats"]
        sections.append(f"\n## {name}")
        sections.append(f"Gesloten trades: {stats['n_closed']} ({stats['n_wins']}W/{stats['n_losses']}L, {stats['win_rate']:.0f}% WR)")
        sections.append(f"Net P&L: ${stats['net_pnl']:+,.2f}")
        sections.append(f"Gross profit: ${stats['gross_profit']:+,.2f} | Gross loss: ${-stats['gross_loss']:+,.2f}")
        sections.append(f"Profit Factor: {stats['profit_factor']}")
        sections.append(f"Avg win: ${stats['avg_win']:+.2f} | Avg loss: ${stats['avg_loss']:+.2f}")
        sections.append(f"Best trade: ${stats['best_trade']:+.2f} | Worst: ${stats['worst_trade']:+.2f}")
        sections.append(f"Fees — Commission: ${stats['total_commission']:.2f}, Swap: ${stats['total_swap']:.2f}")
        sections.append(f"Rejections: {stats['n_rejected']} {dict(stats['rejection_reasons'])}")
        sections.append(f"Still open: {stats['n_still_open']}")

        if data.get("equity"):
            eq = data["equity"]
            dd = (100_000 - eq["equity"]) / 100_000 * 100
            sections.append(f"Equity: ${eq['equity']:,.2f} (DD: {dd:.2f}%)")

        sections.append("\n### Trades")
        for t in stats["trades"]:
            pnl_str = f"${t['pnl']:+.2f}" if t["pnl"] is not None else "—"
            conf_str = f"{t['confidence']:.3f}" if t["confidence"] else "—"
            sections.append(
                f"  {t['opened'] or '??:??'} | #{t['ticket']} {t['symbol']} {t['direction']} "
                f"{t['lot_size']}L @ {t['entry_price']} → {t['exit_price'] or 'open'} | "
                f"{pnl_str} | {t['status']} | conf={conf_str}"
            )

        sections.append("\n### Per Symbol")
        for sym, s in stats["by_symbol"].items():
            wr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
            sections.append(f"  {sym}: {s['trades']}t ({wr:.0f}% WR) ${s['pnl']:+.2f}")

        if stats["confidence_brackets"]:
            sections.append("\n### Confidence Brackets")
            for bracket, b in stats["confidence_brackets"].items():
                wr = b["wins"] / b["trades"] * 100 if b["trades"] else 0
                sections.append(f"  {bracket}: {b['trades']}t ({wr:.0f}% WR) ${b['pnl']:+.2f}")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(model: str, system: str, prompt: str) -> str:
    """Call Ollama API and return generated text."""
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 8192,
            "num_ctx": 16384,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    print(f"  Calling Ollama ({model})...")
    with urllib.request.urlopen(req, timeout=1200) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result.get("response", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate daily trading analysis")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to analyze (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Only collect data, skip LLM generation")
    args = parser.parse_args()

    if args.date:
        date = args.date
    else:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    print(f"Daily Report: {date} ({weekday})")

    # Collect data from both accounts
    account_data = {}
    for name, cfg in ACCOUNTS.items():
        print(f"\n  Collecting {name}...")

        if not cfg["db"].exists():
            print(f"    DB not found: {cfg['db']}")
            continue

        trades = get_trades(cfg["db"], date)
        deals = get_deal_history(cfg["port"], date)
        stats = compute_account_stats(trades, deals)
        equity = get_account_equity(cfg["port"])

        account_data[name] = {"stats": stats, "equity": equity}

        print(f"    {stats['n_closed']} closed trades, ${stats['net_pnl']:+.2f}")

    # Collect market conditions
    print("\n  Collecting market conditions...")
    market = get_market_conditions(5056, date)  # use FTMO bridge for market data
    if not market:
        market = get_market_conditions(5057, date)  # fallback to BF
    print(f"    {len(market)} benchmarks loaded")

    if not any(d["stats"]["n_closed"] > 0 for d in account_data.values()):
        print(f"\nGeen trades gevonden voor {date} — geen rapport gegenereerd.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{date}.md"
        out_path.write_text(f"# Daily Analysis — {date} ({weekday})\n\nGeen trades op deze dag.\n")
        print(f"  Written: {out_path}")
        return

    # Build prompt
    data_prompt = build_prompt(date, account_data, market=market)

    if args.no_llm:
        print("\n--- RAW DATA ---")
        print(data_prompt)
        return

    # Call LLM
    report = call_ollama(args.model, SYSTEM_PROMPT, data_prompt)

    if not report.strip():
        print("ERROR: LLM returned empty response")
        sys.exit(1)

    # Write report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{date}.md"
    out_path.write_text(report + "\n")
    print(f"\n  Report written: {out_path}")
    print(f"  Length: {len(report)} chars")


if __name__ == "__main__":
    main()
