#!/usr/bin/env python3
"""
LLM Trade Advisor — Local Llama 3.1 8B on Tesla P40
=====================================================

Runs alongside the Sovereign Bot as a "second opinion" engine.
Analyzes open positions, swap costs, and market context to provide
actionable advice via Discord.

Uses Ollama API (localhost:11434) for inference.

Author: Thomas (HP Z440)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = str(SCRIPT_DIR / "sovereign_log.db")

# Ollama config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _ollama_generate(prompt: str, system: str = "", temperature: float = 0.3) -> str | None:
    """Call Ollama generate API."""
    import requests
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 800,
                },
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None
    except Exception as e:
        print(f"[LLM_ADVISOR] Ollama error: {e}")
        return None


def _get_open_positions(mt5_bridge) -> list[dict]:
    """Get open positions from MT5."""
    positions = mt5_bridge.positions_get()
    if not positions:
        return []
    result = []
    for p in positions:
        if p.magic < 2000:
            continue
        result.append({
            "symbol": p.symbol,
            "direction": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume,
            "entry": p.price_open,
            "current_price": p.price_current,
            "profit": p.profit,
            "swap": p.swap,
            "sl": p.sl,
            "tp": p.tp,
            "ticket": p.ticket,
            "time_open": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
        })
    return result


def _get_account_summary(mt5_bridge) -> dict | None:
    """Get account info."""
    info = mt5_bridge.account_info()
    if not info:
        return None
    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "profit": info.profit,
    }


def _get_recent_trades(db_path: str, hours: int = 24) -> list[dict]:
    """Get recent filled trades from DB."""
    conn = sqlite3.connect(db_path)
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT timestamp, symbol, direction, entry_price, lot_size, status "
        "FROM trades WHERE timestamp > ? AND status = 'FILLED' "
        "ORDER BY timestamp DESC LIMIT 20",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [
        {"time": r[0], "symbol": r[1], "direction": r[2],
         "entry": r[3], "lots": r[4], "status": r[5]}
        for r in rows
    ]


SYSTEM_PROMPT = """You are a quantitative risk officer for an FTMO prop firm challenge ($100K account).
You produce HARD verdicts, not vague suggestions. Every position gets a verdict.

FTMO HARD LIMITS (violation = account termination):
- Max daily loss: 5% ($5,000) — measured from start-of-day equity
- Max total drawdown: 10% ($10,000) — measured from initial balance
- Profit target: 10% ($10,000) — to pass the challenge
- No hedging (opposing positions on same symbol)
- Non-crypto must close before Friday 23:50 UTC+2

YOUR DECISION RULES (follow these exactly):
1. SWAP BLEEDING: If |swap| / |unrealized_pnl| > 0.25 AND P&L is flat/negative → verdict CLOSE
2. ZOMBIE POSITION: Held >48h with P&L < $5 absolute → verdict CLOSE (dead capital)
3. RISK OVERLOAD: If total unrealized loss > 3% of account → verdict REDUCE (close worst 2)
4. FTMO DANGER ZONE: If equity < $91,000 → verdict EMERGENCY (close all losers immediately)
5. WINNER MANAGEMENT: If unrealized P&L > 2x swap cost and trending → verdict HOLD

OUTPUT FORMAT (strict):
For each position, output exactly one line:
[CLOSE/HOLD/REDUCE] SYMBOL — reason (include $ amounts)

Then a 2-line PORTFOLIO SUMMARY:
- FTMO distance: X% daily / Y% total (safe/warning/danger)
- Action priority: what to do in the next 4 hours

Be brutally honest. No filler text. Numbers only. The trader's $100K depends on your precision."""


INITIAL_BALANCE = float(os.getenv("FTMO_INITIAL_BALANCE", "100000"))


def analyze_positions(mt5_bridge, db_path: str = DB_PATH) -> str | None:
    """Run LLM analysis on current portfolio state."""
    positions = _get_open_positions(mt5_bridge)
    account = _get_account_summary(mt5_bridge)
    recent = _get_recent_trades(db_path)

    if not account:
        return None

    if not positions:
        return None  # Nothing to analyze

    total_swap = sum(p["swap"] for p in positions)
    total_pnl = sum(p["profit"] for p in positions)
    equity = account["equity"]

    # Pre-compute FTMO distances (don't make the LLM do math)
    daily_loss_used_pct = max(0, -account["profit"]) / INITIAL_BALANCE * 100
    daily_loss_remaining = INITIAL_BALANCE * 0.05 + account["profit"]
    total_dd_pct = (INITIAL_BALANCE - equity) / INITIAL_BALANCE * 100
    total_dd_remaining = INITIAL_BALANCE * 0.10 - (INITIAL_BALANCE - equity)
    profit_progress = (equity - INITIAL_BALANCE) / (INITIAL_BALANCE * 0.10) * 100

    # Determine weekday for weekend warning
    now_utc = datetime.now(timezone.utc)
    is_friday = now_utc.weekday() == 4
    hours_to_weekend = 0
    if is_friday:
        close_time = now_utc.replace(hour=21, minute=50, second=0)
        hours_to_weekend = max(0, (close_time - now_utc).total_seconds() / 3600)

    prompt = f"""TIME: {now_utc.strftime('%Y-%m-%d %H:%M UTC')} ({'FRIDAY' if is_friday else now_utc.strftime('%A')})

ACCOUNT (initial: ${INITIAL_BALANCE:,.0f}):
  Equity: ${equity:,.2f} | Balance: ${account['balance']:,.2f}
  Unrealized P&L: ${total_pnl:+,.2f} | Total swap drain: ${total_swap:+,.2f}
  Free margin: ${account['free_margin']:,.2f}

FTMO DISTANCES (pre-computed):
  Daily loss: {daily_loss_used_pct:.2f}% used, ${daily_loss_remaining:,.2f} remaining before limit
  Total DD: {total_dd_pct:.2f}% used, ${total_dd_remaining:,.2f} remaining before limit
  Profit target: {profit_progress:.1f}% complete
  {"⚠ FRIDAY: " + f"{hours_to_weekend:.1f}h until weekend close deadline" if is_friday else ""}

OPEN POSITIONS ({len(positions)}):
"""
    for p in positions:
        # Pre-compute per-position metrics
        hold_hours = 0
        try:
            opened = datetime.fromisoformat(p["time_open"])
            hold_hours = (now_utc - opened).total_seconds() / 3600
        except Exception:
            pass

        pnl = p["profit"]
        swap = p["swap"]
        # Swap-to-alpha ratio: |swap| / |pnl| — infinity if pnl is near zero
        if abs(pnl) > 0.50:
            swap_alpha = abs(swap) / abs(pnl)
        else:
            swap_alpha = 99.9 if abs(swap) > 0.10 else 0.0

        # Flags
        flags = []
        if swap_alpha > 0.25:
            flags.append("SWAP-BLEEDING")
        if hold_hours > 48 and abs(pnl) < 5:
            flags.append("ZOMBIE")
        if pnl < -100:
            flags.append("BIG-LOSER")
        if is_friday and p["symbol"] not in ("BTCUSD", "ETHUSD", "XRPUSD",
                "LTCUSD", "BCHUSD", "SOLUSD", "AAVUSD", "LNKUSD", "XMRUSD",
                "ALGUSD", "XTZUSD", "VECUSD", "GRTUSD", "GALUSD", "IMXUSD",
                "XLMUSD", "MANUSD", "AVAUSD", "ETCUSD", "UNIUSD", "FETUSD",
                "ICPUSD", "NERUSD", "SANUSD"):
            flags.append("WEEKEND-RISK")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        prompt += (
            f"  {p['symbol']} {p['direction']} {p['volume']}lot "
            f"| P&L: ${pnl:+.2f} | Swap: ${swap:+.2f} | "
            f"Swap/Alpha: {swap_alpha:.0%} | Held: {hold_hours:.0f}h"
            f"{flag_str}\n"
        )

    prompt += f"""
TOTALS: P&L=${total_pnl:+,.2f}, Swap=${total_swap:+,.2f}
Trades last 24h: {len(recent)} filled

Give your verdict for each position and portfolio summary."""

    return _ollama_generate(prompt, system=SYSTEM_PROMPT)


def run_advisor_loop(mt5_bridge, discord_notifier=None, interval_min: int = 60,
                     db_path: str = DB_PATH):
    """Run the advisor on a schedule."""
    print(f"[LLM_ADVISOR] Starting (model={OLLAMA_MODEL}, interval={interval_min}min)")

    while True:
        try:
            advice = analyze_positions(mt5_bridge, db_path)
            if advice:
                print(f"[LLM_ADVISOR] Analysis:\n{advice}\n")
                if discord_notifier:
                    discord_notifier.send(
                        "LLM TRADE ADVISOR",
                        advice[:1900],  # Discord limit
                        "purple",
                    )
        except Exception as e:
            print(f"[LLM_ADVISOR] Error: {e}")

        time.sleep(interval_min * 60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))

    from mt5_bridge import get_mt5_bridge, initialize_mt5

    mt5 = get_mt5_bridge()
    ok, err, mode = initialize_mt5(mt5)
    if not ok:
        print(f"MT5 init failed: {err}")
        sys.exit(1)

    # Single analysis
    if "--once" in sys.argv:
        result = analyze_positions(mt5)
        if result:
            print(result)
        else:
            print("No analysis generated")
        sys.exit(0)

    # With Discord
    discord = None
    try:
        from discord_notifier import DiscordNotifier
        config_path = str(SCRIPT_DIR / "discord_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("enabled"):
                discord = DiscordNotifier(cfg.get("webhook_url"))
    except Exception:
        pass

    run_advisor_loop(mt5, discord_notifier=discord)
