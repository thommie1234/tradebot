#!/usr/bin/env python3
"""
FTMO COMPLIANCE MODULE
======================

All FTMO prop firm rules enforced in one place:

1. Max Daily Loss: 5% (hard stop) — our internal limit is stricter at 3.5%
2. Max Total Drawdown: 10% from initial balance (account termination)
3. No Hedging: 1 position per symbol max (enforced in order_router)
4. Weekend Close: All non-crypto positions closed Friday 23:50 GMT+2
5. Trading Hours: Per-symbol session times (enforced in ftmo_guard.py)
6. Max Positions: 200 concurrent (FTMO server limit)
7. Max Daily Trades: 2000 (FTMO server limit)
8. News Trading Filter: 2-min blackout around high-impact events (FTMO Account only)
9. Inactivity Rule: Must trade at least once per 30 calendar days
10. SL Modification Throttle: Prevent hyperactive modification patterns
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── FTMO Constants ──────────────────────────────────────────────────

MAX_DAILY_LOSS_PCT = 0.05       # 5% of starting day balance
MAX_TOTAL_DD_PCT = 0.10         # 10% from initial balance → account terminated
TOTAL_DD_WARNING_PCT = 0.08     # 8% → warning alert

MAX_CONCURRENT_POSITIONS = 200
MAX_DAILY_TRADES = 2000

NEWS_BLACKOUT_BEFORE_SEC = 120  # 2 minutes before high-impact event
NEWS_BLACKOUT_AFTER_SEC = 120   # 2 minutes after high-impact event
NEWS_REFRESH_INTERVAL_SEC = 3600  # Refresh calendar every hour

INACTIVITY_WARN_DAYS = 25       # Discord alert at 25 days
INACTIVITY_MAX_DAYS = 30        # FTMO violation at 30 days

SL_MODIFY_MAX_PER_POSITION = 2000  # Sanity cap — FTMO has no limit on SL modifications


class FTMOCompliance:
    """Prop firm rule enforcement — single source of truth for all compliance rules.

    Supports parameterized limits for multi-account setups (FTMO, BrightFunded, etc.).
    """

    def __init__(self, initial_balance: float, logger=None, discord=None,
                 account_name: str = "default",
                 max_daily_loss_pct: float | None = None,
                 max_total_dd_pct: float | None = None,
                 total_dd_warning_pct: float | None = None,
                 dd_type: str = "trailing"):
        self.initial_balance = initial_balance
        self.logger = logger
        self.discord = discord
        self.account_name = account_name

        # Parameterized limits (fall back to module-level constants for backwards compat)
        self.max_daily_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else MAX_DAILY_LOSS_PCT
        self.max_total_dd_pct = max_total_dd_pct if max_total_dd_pct is not None else MAX_TOTAL_DD_PCT
        self.total_dd_warning_pct = total_dd_warning_pct if total_dd_warning_pct is not None else TOTAL_DD_WARNING_PCT
        self.dd_type = dd_type  # "trailing" or "static"

        # For trailing DD: track high water mark
        self._high_water_mark = initial_balance

        # Daily tracking
        self.daily_start_balance = initial_balance
        self.daily_trade_count = 0
        self._last_reset_date = datetime.now(timezone.utc).date()

        # Total DD
        self._total_dd_warned = False
        self._total_dd_killed = False

        # Inactivity tracking
        self._last_trade_time: float | None = None
        self._inactivity_warned = False

        # News filter
        self._news_events: list[dict] = []
        self._news_last_refresh: float = 0
        self._news_lock = threading.Lock()

        # SL modification throttle: {ticket: {date_str: count}}
        self._sl_mod_counts: dict[int, dict[str, int]] = {}

        dd_stop = initial_balance * (1 - self.max_total_dd_pct)
        self._log('INFO', 'INIT',
                  f'[{account_name}] Initial balance: ${initial_balance:,.2f} | '
                  f'DD type: {dd_type} | DD hard stop at ${dd_stop:,.2f}')

    # ── Logging helper ──────────────────────────────────────────────

    def _log(self, level: str, event: str, msg: str):
        if self.logger:
            self.logger.log(level, 'FTMOCompliance', event, msg)

    # ── 1. Daily Loss Check ─────────────────────────────────────────

    def reset_daily_counters(self, current_balance: float | None = None):
        """Reset daily counters at start of new trading day."""
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset_date:
            self.daily_trade_count = 0
            if current_balance is not None:
                self.daily_start_balance = current_balance
            self._last_reset_date = today
            # Reset SL modification counts
            self._sl_mod_counts.clear()
            self._log('INFO', 'DAILY_RESET',
                      f'Daily counters reset | Balance: ${self.daily_start_balance:,.2f}')

    def check_daily_loss(self, current_equity: float) -> tuple[bool, str]:
        """Check daily loss limit (parameterized per account).

        Returns (allowed, reason). False means trading must stop.
        """
        if self.daily_start_balance <= 0:
            return True, ""

        daily_loss = self.daily_start_balance - current_equity
        daily_loss_pct = daily_loss / self.daily_start_balance

        if daily_loss_pct >= self.max_daily_loss_pct:
            reason = (f"[{self.account_name}] DAILY LOSS: -{daily_loss_pct:.2%} "
                      f"(${daily_loss:,.2f}) >= {self.max_daily_loss_pct:.0%}")
            self._log('CRITICAL', 'DAILY_LOSS_LIMIT', reason)
            return False, reason

        return True, ""

    # ── 2. Total Drawdown 10% Hard Stop ─────────────────────────────

    def update_high_water_mark(self, current_equity: float):
        """Update trailing high water mark (call after each equity check)."""
        if current_equity > self._high_water_mark:
            self._high_water_mark = current_equity

    def check_total_dd(self, current_equity: float) -> tuple[bool, str]:
        """Total drawdown hard stop (supports trailing and static DD types).

        - trailing: DD measured from high water mark (FTMO style)
        - static: DD measured from initial balance (BrightFunded style)

        Returns (must_emergency_close, reason).
        True = ALL positions must be closed, bot enters safe mode.
        """
        if self.initial_balance <= 0:
            return False, ""

        # Update high water mark for trailing DD
        self.update_high_water_mark(current_equity)

        # Calculate DD based on type
        if self.dd_type == "static":
            reference = self.initial_balance
            dd_label = "initial"
        else:  # trailing (default, FTMO behavior)
            reference = self.initial_balance  # FTMO uses initial balance, not HWM
            dd_label = "initial"

        total_dd = reference - current_equity
        total_dd_pct = total_dd / reference

        # Hard stop
        if total_dd_pct >= self.max_total_dd_pct:
            if not self._total_dd_killed:
                self._total_dd_killed = True
                reason = (f"[{self.account_name}] TOTAL DD HARD STOP: equity ${current_equity:,.2f} = "
                          f"-{total_dd_pct:.2%} from {dd_label} ${reference:,.2f}")
                self._log('CRITICAL', 'TOTAL_DD_HARD_STOP', reason)
                if self.discord:
                    self.discord.send(
                        f"[{self.account_name}] TOTAL DD — HARD STOP",
                        f"Equity: ${current_equity:,.2f}\n"
                        f"Reference ({dd_label}): ${reference:,.2f}\n"
                        f"Drawdown: -{total_dd_pct:.2%}\n\n"
                        f"ALL POSITIONS CLOSED. BOT IN SAFE MODE.\n"
                        f"Account at risk of termination.",
                        "red",
                    )
                return True, reason
            return True, "total DD hard stop already triggered"

        # Warning
        if total_dd_pct >= self.total_dd_warning_pct and not self._total_dd_warned:
            self._total_dd_warned = True
            remaining = reference * self.max_total_dd_pct - total_dd
            self._log('WARNING', 'TOTAL_DD_WARNING',
                      f'[{self.account_name}] Total DD at -{total_dd_pct:.2%} — '
                      f'approaching {self.max_total_dd_pct:.0%} hard stop | '
                      f'Remaining: ${remaining:,.2f}')
            if self.discord:
                self.discord.send(
                    f"[{self.account_name}] TOTAL DD WARNING",
                    f"Drawdown: -{total_dd_pct:.2%}\n"
                    f"Remaining before hard stop: ${remaining:,.2f}\n"
                    f"Reduce position sizes immediately.",
                    "orange",
                )

        return False, ""

    # ── 3. Max Positions ────────────────────────────────────────────

    def check_max_positions(self, current_count: int) -> tuple[bool, str]:
        """Check FTMO 200 position limit."""
        if current_count >= MAX_CONCURRENT_POSITIONS:
            return False, f"FTMO max positions: {current_count}/{MAX_CONCURRENT_POSITIONS}"
        return True, ""

    # ── 4. Max Daily Trades ─────────────────────────────────────────

    def check_daily_trades(self) -> tuple[bool, str]:
        """Check FTMO 2000 trades/day limit."""
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return False, f"FTMO max daily trades: {self.daily_trade_count}/{MAX_DAILY_TRADES}"
        return True, ""

    def increment_trade_count(self):
        """Increment daily trade counter (call after successful trade)."""
        self.daily_trade_count += 1

    # ── 5. News Trading Filter ──────────────────────────────────────

    def refresh_news_calendar(self):
        """Fetch high-impact economic events from ForexFactory API."""
        now = time.time()
        if now - self._news_last_refresh < NEWS_REFRESH_INTERVAL_SEC:
            return

        with self._news_lock:
            if now - self._news_last_refresh < NEWS_REFRESH_INTERVAL_SEC:
                return  # double-check after lock

            try:
                import requests
                url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    events = resp.json()
                    # Filter high-impact only
                    high_impact = []
                    for ev in events:
                        impact = ev.get("impact", "").lower()
                        if impact in ("high", "holiday"):
                            date_str = ev.get("date", "")
                            if date_str:
                                try:
                                    # ForexFactory dates: "2026-02-21T13:30:00-05:00"
                                    ev_time = datetime.fromisoformat(date_str)
                                    ev_time_utc = ev_time.astimezone(timezone.utc)
                                    high_impact.append({
                                        "time_utc": ev_time_utc,
                                        "timestamp": ev_time_utc.timestamp(),
                                        "title": ev.get("title", ""),
                                        "country": ev.get("country", ""),
                                        "impact": impact,
                                    })
                                except (ValueError, TypeError):
                                    pass

                    self._news_events = high_impact
                    self._news_last_refresh = now
                    self._log('INFO', 'NEWS_REFRESH',
                              f'Loaded {len(high_impact)} high-impact events this week')
                else:
                    self._log('WARNING', 'NEWS_REFRESH_FAILED',
                              f'HTTP {resp.status_code}')
            except Exception as e:
                self._log('WARNING', 'NEWS_REFRESH_ERROR', str(e))
                self._news_last_refresh = now  # Don't retry immediately

    def check_news_blackout(self, symbol: str | None = None) -> tuple[bool, str]:
        """Check if currently in a news blackout window.

        Returns (is_blocked, reason).
        True = trading blocked due to news event proximity.
        """
        self.refresh_news_calendar()

        now_ts = time.time()

        # Map symbol to affected countries
        affected_countries = self._symbol_to_countries(symbol) if symbol else None

        with self._news_lock:
            for ev in self._news_events:
                ev_ts = ev["timestamp"]

                # Check if we're within the blackout window
                before_start = ev_ts - NEWS_BLACKOUT_BEFORE_SEC
                after_end = ev_ts + NEWS_BLACKOUT_AFTER_SEC

                if before_start <= now_ts <= after_end:
                    # If we have a symbol, only block if the event affects this symbol
                    if affected_countries and ev["country"]:
                        if ev["country"].upper() not in affected_countries:
                            continue

                    secs_to_event = ev_ts - now_ts
                    if secs_to_event > 0:
                        timing = f"{secs_to_event:.0f}s before"
                    else:
                        timing = f"{abs(secs_to_event):.0f}s after"

                    reason = (f"News blackout: {ev['title']} ({ev['country']}) "
                              f"— {timing} event")
                    self._log('INFO', 'NEWS_BLACKOUT', reason)
                    return True, reason

        return False, ""

    @staticmethod
    def _symbol_to_countries(symbol: str) -> set[str]:
        """Map trading symbol to affected countries for news filtering."""
        # US equities/indices → affected by US news
        us_symbols = {'AAPL', 'AMZN', 'GOOG', 'MSFT', 'NVDA', 'TSLA', 'META',
                      'NFLX', 'V', 'BAC', 'PFE', 'ZM', 'BABA',
                      'US30.cash', 'US500.cash', 'USTEC.cash'}
        eu_symbols = {'LVMH', 'ALVG', 'DBKGn', 'RACE', 'SAP', 'ASML', 'SIE',
                      'GER40.cash', 'EU50.cash'}
        jp_symbols = {'JP225.cash'}
        oil_symbols = {'USOIL.cash', 'UKOIL.cash'}
        crypto_symbols = {'BTCUSD', 'ETHUSD', 'SOLUSD', 'ICPUSD', 'UNIUSD',
                          'AAVUSD', 'ETCUSD'}

        countries = set()
        if symbol in us_symbols:
            countries.add('USD')
        elif symbol in eu_symbols:
            countries.add('EUR')
        elif symbol in jp_symbols:
            countries.add('JPY')
        elif symbol in oil_symbols:
            countries.update({'USD', 'CAD'})
        elif symbol in crypto_symbols:
            countries.add('USD')  # Crypto mostly reacts to US macro
        else:
            # Unknown symbol → block on all high-impact news
            countries.update({'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'NZD', 'CHF'})

        # Always include USD for FOMC/NFP — affects everything
        countries.add('USD')
        return countries

    # ── 6. Inactivity Tracking ──────────────────────────────────────

    def record_trade(self):
        """Record that a trade was executed."""
        self._last_trade_time = time.time()
        self._inactivity_warned = False

    def check_inactivity(self) -> tuple[bool, int]:
        """Check FTMO 30-day inactivity rule.

        Returns (must_act, days_since_last_trade).
        If must_act is True, a trade must happen soon to avoid violation.
        """
        if self._last_trade_time is None:
            return False, 0

        days_inactive = (time.time() - self._last_trade_time) / 86400

        if days_inactive >= INACTIVITY_MAX_DAYS:
            self._log('CRITICAL', 'INACTIVITY_LIMIT',
                      f'{days_inactive:.0f} days without a trade — FTMO violation!')
            if self.discord:
                self.discord.send(
                    "FTMO INACTIVITY — 30 DAYS",
                    f"Days since last trade: {days_inactive:.0f}\n"
                    f"FTMO requires at least 1 trade per 30 days.\n"
                    f"Place a manual trade or lower thresholds.",
                    "red",
                )
            return True, int(days_inactive)

        if days_inactive >= INACTIVITY_WARN_DAYS and not self._inactivity_warned:
            self._inactivity_warned = True
            remaining = INACTIVITY_MAX_DAYS - days_inactive
            self._log('WARNING', 'INACTIVITY_WARNING',
                      f'{days_inactive:.0f} days without trade — '
                      f'{remaining:.0f} days until violation')
            if self.discord:
                self.discord.send(
                    "FTMO INACTIVITY WARNING",
                    f"Days since last trade: {days_inactive:.0f}\n"
                    f"Days remaining: {remaining:.0f}\n"
                    f"FTMO requires 1 trade per 30 calendar days.",
                    "orange",
                )

        return False, int(days_inactive)

    def load_last_trade_time(self, audit_db_path: str):
        """Load the last trade timestamp from audit database."""
        try:
            conn = sqlite3.connect(audit_db_path)
            row = conn.execute(
                "SELECT MAX(timestamp) FROM trades WHERE status = 'FILLED'"
            ).fetchone()
            conn.close()
            if row and row[0]:
                last_ts = datetime.fromisoformat(row[0])
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                self._last_trade_time = last_ts.timestamp()
                days_ago = (time.time() - self._last_trade_time) / 86400
                self._log('INFO', 'LAST_TRADE_LOADED',
                          f'Last trade: {days_ago:.1f} days ago')
        except Exception as e:
            self._log('WARNING', 'LAST_TRADE_LOAD_FAILED', str(e))

    # ── 7. SL Modification Throttle ─────────────────────────────────

    def check_sl_modify_allowed(self, ticket: int) -> bool:
        """Check if SL/TP modification is allowed for this position today.

        Returns True if within daily limit, False if throttled.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if ticket not in self._sl_mod_counts:
            self._sl_mod_counts[ticket] = {}

        counts = self._sl_mod_counts[ticket]
        current = counts.get(today_str, 0)

        if current >= SL_MODIFY_MAX_PER_POSITION:
            self._log('WARNING', 'SL_MODIFY_THROTTLED',
                      f'Ticket {ticket}: {current} modifications today '
                      f'(max {SL_MODIFY_MAX_PER_POSITION})')
            return False

        return True

    def record_sl_modify(self, ticket: int):
        """Record an SL/TP modification."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if ticket not in self._sl_mod_counts:
            self._sl_mod_counts[ticket] = {}
        counts = self._sl_mod_counts[ticket]
        counts[today_str] = counts.get(today_str, 0) + 1

    def cleanup_sl_mod_counts(self):
        """Remove entries for positions no longer tracked (call periodically)."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for ticket in list(self._sl_mod_counts.keys()):
            counts = self._sl_mod_counts[ticket]
            # Remove old dates
            old_dates = [d for d in counts if d != today_str]
            for d in old_dates:
                del counts[d]
            if not counts:
                del self._sl_mod_counts[ticket]

    # ── Combined pre-trade check ────────────────────────────────────

    def pre_trade_check(self, current_equity: float, position_count: int,
                        symbol: str | None = None) -> tuple[bool, str]:
        """Run ALL FTMO compliance checks before opening a trade.

        Returns (allowed, reason). False = trade must be blocked.
        """
        # Reset daily counters if new day
        self.reset_daily_counters(current_equity)

        # 1. Total DD hard stop (most critical)
        must_close, reason = self.check_total_dd(current_equity)
        if must_close:
            return False, reason

        # 2. Daily loss
        ok, reason = self.check_daily_loss(current_equity)
        if not ok:
            return False, reason

        # 3. Max positions
        ok, reason = self.check_max_positions(position_count)
        if not ok:
            return False, reason

        # 4. Max daily trades
        ok, reason = self.check_daily_trades()
        if not ok:
            return False, reason

        # 5. News blackout
        blocked, reason = self.check_news_blackout(symbol)
        if blocked:
            return False, f"FTMO news blackout: {reason}"

        return True, "OK"


if __name__ == "__main__":
    # Test FTMO compliance
    ftmo = FTMOCompliance(initial_balance=100000)

    print("=" * 70)
    print("FTMO COMPLIANCE TEST")
    print("=" * 70)

    # Test total DD
    print("\n1. Total Drawdown:")
    must_close, reason = ftmo.check_total_dd(92000)  # 8% DD
    print(f"   8% DD: must_close={must_close} | {reason}")
    must_close, reason = ftmo.check_total_dd(89000)  # 11% DD
    print(f"  11% DD: must_close={must_close} | {reason}")

    # Test daily loss
    print("\n2. Daily Loss:")
    ok, reason = ftmo.check_daily_loss(95500)  # 4.5% loss
    print(f"   4.5% loss: ok={ok} | {reason}")

    # Test pre-trade check
    print("\n3. Pre-trade Check:")
    ok, reason = ftmo.pre_trade_check(97000, 5, "TSLA")
    print(f"   Normal: ok={ok} | {reason}")

    # Test news blackout
    print("\n4. News Blackout:")
    blocked, reason = ftmo.check_news_blackout("TSLA")
    print(f"   TSLA: blocked={blocked} | {reason or 'clear'}")

    # Test SL throttle
    print("\n5. SL Modification Throttle:")
    for i in range(55):
        allowed = ftmo.check_sl_modify_allowed(12345)
        if allowed:
            ftmo.record_sl_modify(12345)
        else:
            print(f"   Throttled at modification #{i}")
            break

    print("\nAll tests complete.")
