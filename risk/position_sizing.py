"""
Position sizing engine — Half-Kelly + lot sizing + sector/correlation maps.

Extracted from PositionSizingEngine in sovereign_bot.py (lines 1290-1432)
and constants/plan-building from trading_prop/production/position_sizing.py.
"""
from __future__ import annotations

import re
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.loader import cfg

# ---------------------------------------------------------------------------
# Sector / Correlation Groups
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    # Crypto
    "AAVUSD": "crypto", "ADAUSD": "crypto", "ALGUSD": "crypto",
    "AVAUSD": "crypto", "BARUSD": "crypto", "BCHUSD": "crypto",
    "BNBUSD": "crypto", "BTCUSD": "crypto", "DASHUSD": "crypto",
    "DOGEUSD": "crypto", "DOTUSD": "crypto", "ETCUSD": "crypto",
    "ETHUSD": "crypto", "FETUSD": "crypto", "GALUSD": "crypto",
    "GRTUSD": "crypto", "ICPUSD": "crypto", "IMXUSD": "crypto",
    "LNKUSD": "crypto", "LTCUSD": "crypto", "MANUSD": "crypto",
    "NEOUSD": "crypto", "NERUSD": "crypto", "SANUSD": "crypto",
    "SOLUSD": "crypto", "UNIUSD": "crypto", "VECUSD": "crypto",
    "XLMUSD": "crypto", "XMRUSD": "crypto", "XRPUSD": "crypto",
    "XTZUSD": "crypto",
    # JPY crosses (both underscore and broker format)
    "AUD_JPY": "jpy", "AUDJPY": "jpy",
    "CAD_JPY": "jpy", "CADJPY": "jpy",
    "CHF_JPY": "jpy", "CHFJPY": "jpy",
    "EUR_JPY": "jpy", "EURJPY": "jpy",
    "GBP_JPY": "jpy", "GBPJPY": "jpy",
    "NZD_JPY": "jpy", "NZDJPY": "jpy",
    "USD_JPY": "jpy", "USDJPY": "jpy",
    # EUR crosses
    "EUR_AUD": "eur", "EURAUD": "eur",
    "EUR_CAD": "eur", "EURCAD": "eur",
    "EUR_CHF": "eur", "EURCHF": "eur",
    "EUR_CZK": "eur", "EURCZK": "eur",
    "EUR_GBP": "eur", "EURGBP": "eur",
    "EUR_HUF": "eur", "EURHUF": "eur",
    "EUR_NOK": "eur", "EURNOK": "eur",
    "EUR_NZD": "eur", "EURNZD": "eur",
    "EUR_PLN": "eur", "EURPLN": "eur",
    "EUR_USD": "eur", "EURUSD": "eur",
    # GBP crosses
    "GBP_AUD": "gbp", "GBPAUD": "gbp",
    "GBP_CAD": "gbp", "GBPCAD": "gbp",
    "GBP_CHF": "gbp", "GBPCHF": "gbp",
    "GBP_NZD": "gbp", "GBPNZD": "gbp",
    "GBP_USD": "gbp", "GBPUSD": "gbp",
    # USD majors (non-JPY, non-EUR, non-GBP)
    "USD_CAD": "usd", "USDCAD": "usd",
    "USD_CHF": "usd", "USDCHF": "usd",
    "USD_CNH": "usd", "USDCNH": "usd",
    "USD_CZK": "usd", "USDCZK": "usd",
    "USD_HKD": "usd", "USDHKD": "usd",
    "USD_HUF": "usd", "USDHUF": "usd",
    "USD_ILS": "usd", "USDILS": "usd",
    "USD_MXN": "usd", "USDMXN": "usd",
    "USD_NOK": "usd", "USDNOK": "usd",
    "USD_PLN": "usd", "USDPLN": "usd",
    "USD_SEK": "usd", "USDSEK": "usd",
    "USD_SGD": "usd", "USDSGD": "usd",
    "USD_ZAR": "usd", "USDZAR": "usd",
    # AUD/NZD
    "AUD_CAD": "aud", "AUDCAD": "aud",
    "AUD_CHF": "aud", "AUDCHF": "aud",
    "AUD_NZD": "aud", "AUDNZD": "aud",
    "AUD_USD": "aud", "AUDUSD": "aud",
    "NZD_CAD": "nzd", "NZDCAD": "nzd",
    "NZD_CHF": "nzd", "NZDCHF": "nzd",
    "NZD_USD": "nzd", "NZDUSD": "nzd",
    # CAD
    "CAD_CHF": "unknown", "CADCHF": "unknown",
    # Commodities
    "NATGAS.cash": "energy", "USOIL.cash": "energy", "UKOIL.cash": "energy",
    "HEATOIL.c": "energy",
    "XAU_USD": "gold", "XAUUSD": "gold", "XAU_AUD": "gold", "XAU_EUR": "gold",
    "XAG_USD": "silver", "XAGUSD": "silver", "XAG_AUD": "silver", "XAG_EUR": "silver",
    "XCU_USD": "metals", "XCUUSD": "metals",
    "XPD_USD": "metals", "XPDUSD": "metals",
    "XPT_USD": "metals", "XPTUSD": "metals",
    "WHEAT.c": "agri", "CORN.c": "agri", "SOYBEAN.c": "agri",
    "COCOA.c": "agri", "COFFEE.c": "agri", "COTTON.c": "agri",
    "SUGAR.c": "agri",
    # Indices
    "US100.cash": "us_idx", "US30.cash": "us_idx", "US500.cash": "us_idx",
    "US2000.cash": "us_idx",
    "GER40.cash": "eu_idx", "FRA40.cash": "eu_idx", "EU50.cash": "eu_idx",
    "SPN35.cash": "eu_idx", "N25.cash": "eu_idx",
    "UK100.cash": "uk_idx", "AUS200.cash": "apac_idx",
    "JP225.cash": "apac_idx", "HK50.cash": "apac_idx",
    "DXY.cash": "usd",
    # Equities
    "AAPL": "us_equity", "AMZN": "us_equity", "GOOG": "us_equity",
    "META": "us_equity", "MSFT": "us_equity", "NFLX": "us_equity",
    "NVDA": "us_equity", "TSLA": "us_equity", "BABA": "us_equity",
    "BAC": "us_equity", "PFE": "us_equity", "T": "us_equity",
    "V": "us_equity", "WMT": "us_equity", "ZM": "us_equity",
    "ALVG": "eu_equity", "AIRF": "eu_equity", "BAYGn": "eu_equity",
    "DBKGn": "eu_equity", "IBE": "eu_equity", "LVMH": "eu_equity",
    "RACE": "eu_equity", "VOWG_p": "eu_equity",
}

ASSET_CLASS = {
    "crypto": "crypto",
    "jpy": "forex", "eur": "forex", "gbp": "forex", "usd": "forex",
    "aud": "forex", "nzd": "forex",
    "energy": "commodity", "gold": "commodity", "silver": "commodity",
    "metals": "commodity", "agri": "commodity",
    "us_idx": "index", "eu_idx": "index", "uk_idx": "index", "apac_idx": "index",
    "us_equity": "equity", "eu_equity": "equity",
}

RISK_PER_TRADE = {
    "forex": 0.0040,
    "equity": 0.0025,
    "crypto": 0.0005,
    "commodity": 0.0035,
    "index": 0.0030,
}

MAX_SECTOR_EXPOSURE = {
    "crypto": 0.020, "jpy": 0.020, "eur": 0.015, "gbp": 0.015,
    "usd": 0.015, "aud": 0.010, "nzd": 0.010, "energy": 0.015,
    "gold": 0.010, "silver": 0.010, "metals": 0.010, "agri": 0.015,
    "us_idx": 0.015, "eu_idx": 0.010, "uk_idx": 0.005, "apac_idx": 0.010,
    "us_equity": 0.025, "eu_equity": 0.010,
}


# ---------------------------------------------------------------------------
# Plan-building dataclasses and functions
# ---------------------------------------------------------------------------

@dataclass
class SymbolResult:
    symbol: str
    trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_dd: float
    sharpe: float
    avg_trade: float


@dataclass
class PositionPlan:
    symbol: str
    sector: str
    asset_class: str
    trades: int
    win_rate: float
    profit_factor: float
    max_dd: float
    sharpe: float
    kelly_full: float
    kelly_fraction: float
    risk_per_trade: float
    allocation_usd: float
    pnl_usd: float
    pnl_per_month: float
    capped_by_sector: bool = False


def fractional_kelly(win_rate: float, profit_factor: float, fraction: float = 0.1) -> float:
    if profit_factor <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    w = win_rate
    r = profit_factor * w / (1 - w)
    kelly = w - (1 - w) / r
    if kelly <= 0:
        return 0.0
    return kelly * fraction


def parse_wfo_log(log_path: str) -> dict[str, SymbolResult]:
    results = {}
    text = Path(log_path).read_text()
    marker = "ALL SYMBOLS RESULTS"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"Could not find '{marker}' in {log_path}")
    lines = text[idx:].splitlines()
    for line in lines:
        m = re.match(
            r'\s+(\S+)\s+[+\-]\d+\.\d+\s+(\d+)\s+(\d+\.\d+)%\s+(\d+\.\d+)\s+([+\-]\d+\.\d+)%\s+([+\-]\d+\.\d+)%\s+([+\-]\d+\.\d+)',
            line
        )
        if not m:
            continue
        sym = m.group(1)
        trades = int(m.group(2))
        wr = float(m.group(3)) / 100
        pf = float(m.group(4))
        ret = float(m.group(5)) / 100
        dd = float(m.group(6)) / 100
        sharpe = float(m.group(7))
        avg_trade = ret / max(trades, 1)
        results[sym] = SymbolResult(
            symbol=sym, trades=trades, win_rate=wr,
            profit_factor=pf, total_return=ret, max_dd=dd,
            sharpe=sharpe, avg_trade=avg_trade,
        )
    return results


def build_position_plan(
    results: dict[str, SymbolResult],
    account: float = 100_000,
    max_dd_pct: float = 0.05,
    kelly_fraction: float = 0.10,
    min_trades: int = 100,
    min_pf: float = 1.0,
    min_sharpe: float = 0.0,
) -> list[PositionPlan]:
    dd_budget = account * max_dd_pct
    viable = {
        sym: r for sym, r in results.items()
        if r.trades >= min_trades and r.profit_factor > min_pf and r.sharpe >= min_sharpe
    }
    plans = []
    for sym, r in viable.items():
        sector = SECTOR_MAP.get(sym, "unknown")
        ac = ASSET_CLASS.get(sector, "forex")
        base_risk = RISK_PER_TRADE.get(ac, 0.003)
        kelly_full = fractional_kelly(r.win_rate, r.profit_factor, fraction=1.0)
        kelly_frac = fractional_kelly(r.win_rate, r.profit_factor, fraction=kelly_fraction)
        risk = min(base_risk, kelly_frac) if kelly_frac > 0 else 0.0
        plans.append(PositionPlan(
            symbol=sym, sector=sector, asset_class=ac,
            trades=r.trades, win_rate=r.win_rate,
            profit_factor=r.profit_factor, max_dd=r.max_dd,
            sharpe=r.sharpe,
            kelly_full=kelly_full, kelly_fraction=kelly_frac,
            risk_per_trade=risk,
            allocation_usd=0, pnl_usd=0, pnl_per_month=0,
        ))
    sector_used = {}
    plans.sort(key=lambda p: -p.profit_factor)
    for p in plans:
        if p.risk_per_trade <= 0:
            continue
        max_sector = MAX_SECTOR_EXPOSURE.get(p.sector, 0.02)
        used = sector_used.get(p.sector, 0.0)
        remaining = max_sector - used
        if remaining <= 0:
            p.risk_per_trade = 0.0
            p.capped_by_sector = True
            continue
        if p.risk_per_trade > remaining:
            p.risk_per_trade = remaining
            p.capped_by_sector = True
        sector_used[p.sector] = used + p.risk_per_trade
    total_risk = sum(p.risk_per_trade for p in plans if p.risk_per_trade > 0)
    for p in plans:
        if p.risk_per_trade <= 0:
            continue
        abs_dd = max(abs(p.max_dd), 0.01)
        budget_share = (p.risk_per_trade / total_risk) * dd_budget if total_risk > 0 else 0
        p.allocation_usd = budget_share / abs_dd
        r = results[p.symbol]
        p.pnl_usd = p.allocation_usd * r.total_return
        oos_months = max(r.trades / (2.0 * 21), 1)
        p.pnl_per_month = p.pnl_usd / oos_months
    return plans


def print_plan(plans: list[PositionPlan], account: float, max_dd_pct: float):
    active = [p for p in plans if p.risk_per_trade > 0]
    capped = [p for p in plans if p.capped_by_sector]
    print(f"\n{'='*110}")
    print(f"  SOVEREIGN POSITION SIZING PLAN")
    print(f"  Account: ${account:,.0f} | Max DD: {max_dd_pct:.0%} (${account*max_dd_pct:,.0f}) | Active: {len(active)} symbols")
    print(f"{'='*110}")
    sectors = {}
    for p in active:
        sectors.setdefault(p.sector, []).append(p)
    print(f"\n  {'Symbol':>14s} {'Sector':>8s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'Kelly%':>7s} {'Risk%':>7s} {'Alloc$':>10s} {'P&L$':>10s} {'$/mo':>8s} {'Cap?':>5s}")
    print(f"  {'-'*110}")
    total_pnl = total_alloc = total_risk = 0
    for p in sorted(active, key=lambda x: -x.pnl_per_month):
        cap = "SEC" if p.capped_by_sector else ""
        print(
            f"  {p.symbol:>14s} {p.sector:>8s} {p.trades:>7d} {p.win_rate:>5.1%} {p.profit_factor:>6.3f} "
            f"{p.kelly_full:>6.1%} {p.risk_per_trade:>6.2%} "
            f"${p.allocation_usd:>9,.0f} ${p.pnl_usd:>+9,.0f} ${p.pnl_per_month:>+7,.0f} {cap:>5s}"
        )
        total_pnl += p.pnl_usd
        total_alloc += p.allocation_usd
        total_risk += p.risk_per_trade
    print(f"  {'-'*110}")
    print(f"  {'TOTAL':>14s} {'':>8s} {'':>7s} {'':>6s} {'':>6s} "
          f"{'':>7s} {total_risk:>6.2%} "
          f"${total_alloc:>9,.0f} ${total_pnl:>+9,.0f} {'':>8s}")
    print(f"\n  SECTOR EXPOSURE:")
    sector_risk = {}
    for p in active:
        sector_risk[p.sector] = sector_risk.get(p.sector, 0) + p.risk_per_trade
    for sec in sorted(sector_risk, key=lambda s: -sector_risk[s]):
        limit = MAX_SECTOR_EXPOSURE.get(sec, 0.02)
        used = sector_risk[sec]
        status = "FULL" if abs(used - limit) < 0.0001 else "OK"
        print(f"    {sec:>12s}: {used:.2%} / {limit:.2%} [{status}]")
    print(f"\n  WORST-CASE CONCURRENT RISK:")
    print(f"    If ALL {len(active)} positions hit SL simultaneously: {total_risk:.2%} of account (${account*total_risk:,.0f})")
    max_concurrent = min(len(active), 15)
    concurrent_risk = sum(p.risk_per_trade for p in sorted(active, key=lambda x: -x.risk_per_trade)[:max_concurrent])
    print(f"    Top {max_concurrent} largest positions simultaneously: {concurrent_risk:.2%} (${account*concurrent_risk:,.0f})")
    print(f"    FTMO daily loss limit: 5.00% (${account*0.05:,.0f})")
    safe = concurrent_risk < 0.04
    print(f"    Safety margin: {'PASS' if safe else 'WARNING'} — "
          f"{'within' if safe else 'EXCEEDS'} 4% practical limit")
    print(f"\n  SWAN TEST (Sleep Well At Night):")
    print(f"    10 consecutive losses at max risk: {10 * max(p.risk_per_trade for p in active):.2%} "
          f"(${10 * max(p.risk_per_trade for p in active) * account:,.0f})")
    if capped:
        print(f"\n  SECTOR-CAPPED SYMBOLS ({len(capped)}):")
        for p in capped:
            if p.risk_per_trade > 0:
                print(f"    {p.symbol:>14s} [{p.sector}] risk reduced to {p.risk_per_trade:.3%}")
            else:
                print(f"    {p.symbol:>14s} [{p.sector}] EXCLUDED (sector full)")


class PositionSizingEngine:
    """Calculate lot size based on risk management with live Kelly sizing"""

    # FTMO constraints
    DAILY_DD_LIMIT = 0.05
    TOTAL_DD_LIMIT = 0.10
    MAX_CONCURRENT = 8
    DAILY_BUDGET = 0.035
    MIN_TRADES_FOR_KELLY = 20
    KELLY_FRACTION = 0.5
    KELLY_FLOOR = 0.001
    KELLY_CAP = None

    def __init__(self, logger, mt5=None):
        self.logger = logger
        self.mt5 = mt5
        self.KELLY_CAP = self.DAILY_BUDGET / self.MAX_CONCURRENT
        self._kelly_cache = {}
        self._kelly_cache_ttl = 3600
        # F3: RL position sizer (lazy loaded)
        self._rl_sizer = None
        self._rl_loaded = False

    def _get_rl_sizer(self):
        """Lazy-load RL sizer (F3)."""
        if not self._rl_loaded:
            self._rl_loaded = True
            try:
                from risk.rl_sizer import ContextualBanditSizer
                self._rl_sizer = ContextualBanditSizer()
                if not self._rl_sizer.load():
                    self.logger.log('INFO', 'PositionSizing', 'RL_SIZER_NEW',
                                    'No saved RL sizer found, starting fresh')
            except Exception as e:
                self._rl_sizer = None
                self.logger.log('DEBUG', 'PositionSizing', 'RL_SIZER_SKIP', str(e))
        return self._rl_sizer

    def rl_adjust_risk(self, base_risk: float, ml_confidence: float = 0.55,
                       regime: int = 0, volatility: float = 0.0,
                       drawdown_pct: float = 0.0) -> tuple[float, int | None]:
        """
        F3: Apply RL multiplier to base risk. Returns (adjusted_risk, arm_index).
        If RL sizer unavailable, returns (base_risk, None).
        """
        sizer = self._get_rl_sizer()
        if sizer is None:
            return base_risk, None

        context = sizer.build_context(ml_confidence, regime, volatility, drawdown_pct)
        arm_idx, multiplier = sizer.select_arm(context)

        adjusted = base_risk * multiplier
        # Never exceed KELLY_CAP
        adjusted = min(adjusted, self.KELLY_CAP)

        self.logger.log('DEBUG', 'PositionSizing', 'RL_ADJUST',
                        f'RL arm={arm_idx} mult={multiplier:.2f} '
                        f'risk {base_risk:.4%} → {adjusted:.4%}')
        return adjusted, arm_idx

    def rl_update(self, arm: int, context_args: dict, reward: float):
        """F3: Update RL sizer after trade result."""
        sizer = self._get_rl_sizer()
        if sizer is None or arm is None:
            return
        context = sizer.build_context(**context_args)
        sizer.update(arm, context, reward)
        sizer.save()

    def kelly_risk_pct(self, symbol: str) -> float:
        """Compute half-Kelly risk fraction from MT5 closed trade history."""
        now = _time.time()
        cached = self._kelly_cache.get(symbol)
        if cached and (now - cached[1]) < self._kelly_cache_ttl:
            return cached[0]

        fallback = 0.003

        if self.mt5 is None:
            return fallback

        try:
            from datetime import datetime, timedelta, timezone
            from_date = datetime.now(timezone.utc) - timedelta(days=180)
            deals = self.mt5.history_deals_get(from_date, datetime.now(timezone.utc))
            if deals is None or len(deals) == 0:
                self._kelly_cache[symbol] = (fallback, now)
                return fallback

            pnls = []
            for d in deals:
                if d.magic < 2000:
                    continue
                if d.symbol != symbol:
                    continue
                if d.entry != 1:
                    continue
                pnl = d.profit + d.swap + d.commission
                pnls.append(pnl)

            if len(pnls) < self.MIN_TRADES_FOR_KELLY:
                all_pnls = []
                for d in deals:
                    if d.magic < 2000 or d.entry != 1:
                        continue
                    all_pnls.append(d.profit + d.swap + d.commission)

                if len(all_pnls) < self.MIN_TRADES_FOR_KELLY:
                    self._kelly_cache[symbol] = (fallback, now)
                    return fallback
                pnls = all_pnls

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            if not wins or not losses:
                self._kelly_cache[symbol] = (fallback, now)
                return fallback

            p = len(wins) / len(pnls)
            q = 1.0 - p
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            b = avg_win / avg_loss if avg_loss > 0 else 1.0

            kelly_full = (p * b - q) / b if b > 0 else 0
            kelly_full = max(kelly_full, 0)

            risk = kelly_full * self.KELLY_FRACTION
            risk = max(risk, self.KELLY_FLOOR)
            risk = min(risk, self.KELLY_CAP)

            self.logger.log('INFO', 'PositionSizing', 'KELLY',
                            f'{symbol}: {len(pnls)} trades, WR={p:.0%}, '
                            f'avg_w=${avg_win:.2f}, avg_l=${avg_loss:.2f}, '
                            f'b={b:.2f}, kelly={kelly_full:.3%}, '
                            f'half_kelly={risk:.3%} (cap={self.KELLY_CAP:.3%})')

            self._kelly_cache[symbol] = (round(risk, 6), now)
            return round(risk, 6)

        except Exception as e:
            self.logger.log('WARNING', 'PositionSizing', 'KELLY_ERROR', str(e))
            self._kelly_cache[symbol] = (fallback, now)
            return fallback

    def calculate_lot_size(self, symbol, account_equity, risk_pct, sl_distance,
                           symbol_info=None):
        """
        lot_size = (equity * risk_pct) / (sl_distance * contract_size)
        Clamped to [min_lot, max_lot], rounded to lot_step.
        """
        if sl_distance <= 0:
            return 0.0

        risk_amount = account_equity * risk_pct

        contract_size = 1.0
        min_lot = 0.01
        max_lot = 100.0
        lot_step = 0.01

        if symbol_info:
            contract_size = symbol_info.get("trade_contract_size", 1.0)
            min_lot = symbol_info.get("volume_min", 0.01)
            max_lot = symbol_info.get("volume_max", 100.0)
            lot_step = symbol_info.get("volume_step", 0.01)

        risk_per_lot = sl_distance * contract_size
        if risk_per_lot <= 0:
            return min_lot

        lot_size = risk_amount / risk_per_lot

        if lot_step > 0:
            lot_size = round(lot_size / lot_step) * lot_step

        lot_size = max(min_lot, min(lot_size, max_lot))

        self.logger.log('DEBUG', 'PositionSizing', 'LOT_CALC',
                        f'{symbol} Risk:${risk_amount:.2f} SL_dist:{sl_distance:.5f} -> {lot_size:.2f} lots')
        return round(lot_size, 2)
