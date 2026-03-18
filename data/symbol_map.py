"""
Universal symbol mapping — one source of truth for all broker name translations.

Columns:
  common        — how you'd normally say it (human-readable)
  ftmo          — FTMO MT5 terminal name (None = not available)
  bf            — BrightFunded MT5 terminal name (None = not available)
  data_dir      — canonical directory name for bar/tick data storage
  asset_class   — crypto, forex, forex_exotic, index, commodity, metal, equity

Usage:
    from data.symbol_map import lookup, ftmo_to_bf, bf_to_ftmo, all_bf, all_ftmo

    lookup("dogecoin")          # → {common: "DOGE", ftmo: "DOGEUSD", bf: "XDG/USD", ...}
    ftmo_to_bf("DOGEUSD")       # → "XDG/USD"
    bf_to_ftmo("XDG/USD")       # → "DOGEUSD"
    all_bf()                    # → list of all BF broker names
    all_ftmo()                  # → list of all FTMO broker names
    data_dir_to_broker("BTC_USD", "bf")  # → "BTC/USD"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    common: str
    ftmo: str | None
    bf: str | None
    data_dir: str
    asset_class: str


# fmt: off
SYMBOLS: list[Symbol] = [
    # ── Crypto ──────────────────────────────────────────────────────────
    Symbol("BTC",     "BTCUSD",   "BTC/USD",    "BTC_USD",   "crypto"),
    Symbol("ETH",     "ETHUSD",   "ETH/USD",    "ETH_USD",   "crypto"),
    Symbol("SOL",     "SOLUSD",   "SOL/USD",    "SOL_USD",   "crypto"),
    Symbol("DOGE",    "DOGEUSD",  "XDG/USD",    "DOGE_USD",  "crypto"),
    Symbol("XRP",     "XRPUSD",   "XRP/USD",    "XRP_USD",   "crypto"),
    Symbol("ADA",     "ADAUSD",   "ADA/USD",    "ADA_USD",   "crypto"),
    Symbol("DOT",     "DOTUSD",   "DOT/USD",    "DOT_USD",   "crypto"),
    Symbol("LINK",    "LNKUSD",   "LINK/USD",   "LINK_USD",  "crypto"),
    Symbol("UNI",     "UNIUSD",   "UNI/USD",    "UNI_USD",   "crypto"),
    Symbol("AAVE",    "AAVUSD",   "AAVE/USD",   "AAVE_USD",  "crypto"),
    Symbol("LTC",     "LTCUSD",   "LTC/USD",    "LTC_USD",   "crypto"),
    Symbol("BNB",     "BNBUSD",   "BNB/USD",    "BNB_USD",   "crypto"),
    Symbol("ALGO",    "ALGUSD",   "ALGO/USD",   "ALGO_USD",  "crypto"),
    Symbol("XLM",     "XLMUSD",   "XLM/USD",    "XLM_USD",   "crypto"),
    Symbol("NEO",     "NEOUSD",   "NEO/USD",    "NEO_USD",   "crypto"),
    Symbol("DASH",    "DASHUSD",  "DASH/USD",   "DASH_USD",  "crypto"),
    Symbol("MATIC",   None,       "MATIC/USD",  "MATIC_USD", "crypto"),
    Symbol("NEAR",    None,       "NEAR/USD",   "NEAR_USD",  "crypto"),
    Symbol("FIL",     None,       "FIL/USD",    "FIL_USD",   "crypto"),
    Symbol("OP",      None,       "OP/USD",     "OP_USD",    "crypto"),
    Symbol("ARB",     None,       "ARB/USD",    "ARB_USD",   "crypto"),
    Symbol("SUI",     None,       "SUI/USD",    "SUI_USD",   "crypto"),
    Symbol("SEI",     None,       "SEI/USD",    "SEI_USD",   "crypto"),
    Symbol("APT",     None,       "APT/USD",    "APT_USD",   "crypto"),
    Symbol("STX",     None,       "STX/USD",    "STX_USD",   "crypto"),
    Symbol("INJ",     None,       "INJ/USD",    "INJ_USD",   "crypto"),
    Symbol("RUNE",    None,       "RUNE/USD",   "RUNE_USD",  "crypto"),
    Symbol("TRX",     None,       "TRX/USD",    "TRX_USD",   "crypto"),
    Symbol("CRV",     None,       "CRV/USD",    "CRV_USD",   "crypto"),
    Symbol("DYDX",    None,       "DYDX/USD",   "DYDX_USD",  "crypto"),
    Symbol("LDO",     None,       "LDO/USD",    "LDO_USD",   "crypto"),
    Symbol("RNDR",    None,       "RNDR/USD",   "RNDR_USD",  "crypto"),
    Symbol("BCH",     "BCHUSD",   None,         "BCHUSD",    "crypto"),
    Symbol("ETC",     "ETCUSD",   None,         "ETCUSD",    "crypto"),
    Symbol("XMR",     "XMRUSD",   None,         "XMRUSD",    "crypto"),
    Symbol("XTZ",     "XTZUSD",   None,         "XTZUSD",    "crypto"),
    Symbol("ICP",     "ICPUSD",   None,         "ICPUSD",    "crypto"),
    Symbol("FET",     "FETUSD",   None,         "FETUSD",    "crypto"),
    Symbol("GRT",     "GRTUSD",   None,         "GRTUSD",    "crypto"),
    Symbol("IMX",     "IMXUSD",   None,         "IMXUSD",    "crypto"),
    Symbol("MANA",    "MANUSD",   None,         "MANUSD",    "crypto"),
    Symbol("BAR",     "BARUSD",   None,         "BARUSD",    "crypto"),
    Symbol("GAL",     "GALUSD",   None,         "GALUSD",    "crypto"),
    Symbol("VET",     "VECUSD",   None,         "VECUSD",    "crypto"),
    Symbol("NER",     "NERUSD",   None,         "NERUSD",    "crypto"),
    Symbol("SAN",     "SANUSD",   None,         "SANUSD",    "crypto"),
    Symbol("AVAX",    "AVAUSD",   None,         "AVAUSD",    "crypto"),

    # ── Forex Majors ───────────────────────────────────────────────────
    Symbol("EURUSD",  "EURUSD",   "EURUSD",     "EURUSD",    "forex"),
    Symbol("GBPUSD",  "GBPUSD",   "GBPUSD",     "GBPUSD",    "forex"),
    Symbol("USDJPY",  "USDJPY",   "USDJPY",     "USDJPY",    "forex"),
    Symbol("USDCHF",  "USDCHF",   "USDCHF",     "USDCHF",    "forex"),
    Symbol("AUDUSD",  "AUDUSD",   "AUDUSD",     "AUDUSD",    "forex"),
    Symbol("NZDUSD",  "NZDUSD",   "NZDUSD",     "NZDUSD",    "forex"),
    Symbol("USDCAD",  "USDCAD",   "USDCAD",     "USDCAD",    "forex"),

    # ── Forex Crosses ──────────────────────────────────────────────────
    Symbol("EURJPY",  "EURJPY",   "EURJPY",     "EURJPY",    "forex"),
    Symbol("EURGBP",  "EURGBP",   "EURGBP",     "EURGBP",    "forex"),
    Symbol("EURAUD",  "EURAUD",   "EURAUD",     "EURAUD",    "forex"),
    Symbol("EURCAD",  "EURCAD",   "EURCAD",     "EURCAD",    "forex"),
    Symbol("EURCHF",  "EURCHF",   "EURCHF",     "EURCHF",    "forex"),
    Symbol("EURNZD",  "EURNZD",   "EURNZD",     "EURNZD",    "forex"),
    Symbol("GBPJPY",  "GBPJPY",   "GBPJPY",     "GBPJPY",    "forex"),
    Symbol("GBPAUD",  "GBPAUD",   "GBPAUD",     "GBPAUD",    "forex"),
    Symbol("GBPCAD",  "GBPCAD",   "GBPCAD",     "GBPCAD",    "forex"),
    Symbol("GBPCHF",  "GBPCHF",   "GBPCHF",     "GBPCHF",    "forex"),
    Symbol("GBPNZD",  "GBPNZD",   "GBPNZD",     "GBPNZD",    "forex"),
    Symbol("AUDJPY",  "AUDJPY",   "AUDJPY",     "AUDJPY",    "forex"),
    Symbol("AUDNZD",  "AUDNZD",   "AUDNZD",     "AUDNZD",    "forex"),
    Symbol("AUDCAD",  "AUDCAD",   "AUDCAD",     "AUDCAD",    "forex"),
    Symbol("AUDCHF",  "AUDCHF",   "AUDCHF",     "AUDCHF",    "forex"),
    Symbol("NZDJPY",  "NZDJPY",   "NZDJPY",     "NZDJPY",    "forex"),
    Symbol("NZDCAD",  "NZDCAD",   "NZDCAD",     "NZDCAD",    "forex"),
    Symbol("NZDCHF",  "NZDCHF",   "NZDCHF",     "NZDCHF",    "forex"),
    Symbol("CADJPY",  "CADJPY",   "CADJPY",     "CADJPY",    "forex"),
    Symbol("CADCHF",  "CADCHF",   "CADCHF",     "CADCHF",    "forex"),
    Symbol("CHFJPY",  "CHFJPY",   "CHFJPY",     "CHFJPY",    "forex"),

    # ── Forex Exotic ───────────────────────────────────────────────────
    Symbol("USDSEK",  "USDSEK",   "USDSEK",     "USDSEK",    "forex_exotic"),
    Symbol("USDMXN",  "USDMXN",   "USDMXN",     "USDMXN",    "forex_exotic"),
    Symbol("USDZAR",  "USDZAR",   "USDZAR",     "USDZAR",    "forex_exotic"),
    Symbol("USDNOK",  "USDNOK",   "USDNOK",     "USDNOK",    "forex_exotic"),
    Symbol("USDPLN",  "USDPLN",   "USDPLN",     "USDPLN",    "forex_exotic"),
    Symbol("USDSGD",  "USDSGD",   "USDSGD",     "USDSGD",    "forex_exotic"),
    Symbol("USDHKD",  "USDHKD",   "USDHKD",     "USDHKD",    "forex_exotic"),
    Symbol("EURPLN",  "EURPLN",   "EURPLN",     "EURPLN",    "forex_exotic"),
    Symbol("EURNOK",  "EURNOK",   "EURNOK",     "EURNOK",    "forex_exotic"),
    Symbol("EURSEK",  None,       "EURSEK",     "EURSEK",    "forex_exotic"),
    Symbol("USDCNH",  "USDCNH",   None,         "USDCNH",    "forex_exotic"),
    Symbol("USDCZK",  "USDCZK",   None,         "USDCZK",    "forex_exotic"),
    Symbol("USDHUF",  "USDHUF",   None,         "USDHUF",    "forex_exotic"),
    Symbol("USDILS",  "USDILS",   None,         "USDILS",    "forex_exotic"),
    Symbol("EURCZK",  "EURCZK",   None,         "EURCZK",    "forex_exotic"),
    Symbol("EURHUF",  "EURHUF",   None,         "EURHUF",    "forex_exotic"),

    # ── Indices ────────────────────────────────────────────────────────
    Symbol("US30",    "US30.cash",  "US30.cash",  "US30.cash",   "index"),
    Symbol("US100",   "US100.cash", "US100.cash", "US100.cash",  "index"),
    Symbol("US500",   "US500.cash", "US500.cash", "US500.cash",  "index"),
    Symbol("EU50",    "EU50.cash",  "EU50.cash",  "EU50.cash",   "index"),
    Symbol("UK100",   "UK100.cash", "UK100.cash", "UK100.cash",  "index"),
    Symbol("FRA40",   "FRA40.cash", "FRA40.cash", "FRA40.cash",  "index"),
    Symbol("GER40",   "GER40.cash", None,         "GER40.cash",  "index"),
    Symbol("JP225",   "JP225.cash", None,         "JP225.cash",  "index"),
    Symbol("AUS200",  "AUS200.cash",None,         "AUS200.cash", "index"),
    Symbol("HK50",    "HK50.cash",  None,         "HK50.cash",   "index"),
    Symbol("US2000",  "US2000.cash",None,         "US2000.cash", "index"),
    Symbol("SPN35",   "SPN35.cash", None,         "SPN35.cash",  "index"),
    Symbol("N25",     "N25.cash",   None,         "N25.cash",    "index"),
    Symbol("DXY",     "DXY.cash",   None,         "DXY.cash",    "index"),

    # ── Metals ─────────────────────────────────────────────────────────
    Symbol("GOLD",    "XAUUSD",   "XAU/USD",    "XAU_USD",   "metal"),
    Symbol("SILVER",  "XAGUSD",   "XAG/USD",    "XAG_USD",   "metal"),
    Symbol("PALLADIUM","XPDUSD",  "XPD/USD",    "XPD_USD",   "metal"),
    Symbol("PLATINUM","XPTUSD",   "XPT/USD",    "XPT_USD",   "metal"),
    Symbol("COPPER",  "XCUUSD",   None,         "XCUUSD",    "metal"),
    Symbol("GOLD_AUD","XAUAUD",   None,         "XAUAUD",    "metal"),
    Symbol("GOLD_EUR","XAUEUR",   None,         "XAUEUR",    "metal"),
    Symbol("SILVER_AUD","XAGAUD", None,         "XAGAUD",    "metal"),
    Symbol("SILVER_EUR","XAGEUR", None,         "XAGEUR",    "metal"),

    # ── Commodities ────────────────────────────────────────────────────
    Symbol("OIL_US",  "USOIL.cash",None,        "USOIL.cash",  "commodity"),
    Symbol("OIL_UK",  "UKOIL.cash",None,        "UKOIL.cash",  "commodity"),
    Symbol("NATGAS",  "NATGAS.cash",None,        "NATGAS.cash", "commodity"),
    Symbol("HEATOIL", "HEATOIL.c", None,        "HEATOIL.c",   "commodity"),
    Symbol("COCOA",   "COCOA.c",   None,        "COCOA.c",     "commodity"),
    Symbol("COFFEE",  "COFFEE.c",  None,        "COFFEE.c",    "commodity"),
    Symbol("CORN",    "CORN.c",    None,        "CORN.c",      "commodity"),
    Symbol("COTTON",  "COTTON.c",  None,        "COTTON.c",    "commodity"),
    Symbol("SOYBEAN", "SOYBEAN.c", None,        "SOYBEAN.c",   "commodity"),
    Symbol("SUGAR",   "SUGAR.c",   None,        "SUGAR.c",     "commodity"),
    Symbol("WHEAT",   "WHEAT.c",   None,        "WHEAT.c",     "commodity"),

    # ── Equities ───────────────────────────────────────────────────────
    Symbol("AAPL",    "AAPL",     None,         "AAPL",      "equity"),
    Symbol("AMZN",    "AMZN",     None,         "AMZN",      "equity"),
    Symbol("GOOG",    "GOOG",     None,         "GOOG",      "equity"),
    Symbol("META",    "META",     None,         "META",      "equity"),
    Symbol("MSFT",    "MSFT",     None,         "MSFT",      "equity"),
    Symbol("NFLX",    "NFLX",     None,         "NFLX",      "equity"),
    Symbol("NVDA",    "NVDA",     None,         "NVDA",      "equity"),
    Symbol("TSLA",    "TSLA",     None,         "TSLA",      "equity"),
    Symbol("BABA",    "BABA",     None,         "BABA",      "equity"),
    Symbol("BAC",     "BAC",      None,         "BAC",       "equity"),
    Symbol("PFE",     "PFE",      None,         "PFE",       "equity"),
    Symbol("T",       "T",        None,         "T",         "equity"),
    Symbol("V",       "V",        None,         "V",         "equity"),
    Symbol("WMT",     "WMT",      None,         "WMT",       "equity"),
    Symbol("ZM",      "ZM",       None,         "ZM",        "equity"),
    Symbol("ALVG",    "ALVG",     None,         "ALVG",      "equity"),
    Symbol("AIRF",    "AIRF",     None,         "AIRF",      "equity"),
    Symbol("BAYGn",   "BAYGn",    None,         "BAYGn",     "equity"),
    Symbol("DBKGn",   "DBKGn",    None,         "DBKGn",     "equity"),
    Symbol("IBE",     "IBE",      None,         "IBE",       "equity"),
    Symbol("LVMH",    "LVMH",     None,         "LVMH",      "equity"),
    Symbol("RACE",    "RACE",     None,         "RACE",      "equity"),
    Symbol("VOWG_p",  "VOWG_p",   None,         "VOWG_p",    "equity"),
]
# fmt: on


# ── Index lookups (built once at import) ──────────────────────────────

_by_common: dict[str, Symbol] = {}
_by_ftmo: dict[str, Symbol] = {}
_by_bf: dict[str, Symbol] = {}
_by_data_dir: dict[str, Symbol] = {}

for _s in SYMBOLS:
    _by_common[_s.common.upper()] = _s
    if _s.ftmo:
        _by_ftmo[_s.ftmo.upper()] = _s
    if _s.bf:
        _by_bf[_s.bf.upper()] = _s
    _by_data_dir[_s.data_dir.upper()] = _s


def lookup(name: str) -> Symbol | None:
    """Find a symbol by any name (common, ftmo, bf, or data_dir)."""
    key = name.strip().upper()
    return (
        _by_common.get(key)
        or _by_ftmo.get(key)
        or _by_bf.get(key)
        or _by_data_dir.get(key)
    )


def ftmo_to_bf(ftmo_name: str) -> str | None:
    """Translate FTMO broker name → BF broker name."""
    s = _by_ftmo.get(ftmo_name.strip().upper())
    return s.bf if s else None


def bf_to_ftmo(bf_name: str) -> str | None:
    """Translate BF broker name → FTMO broker name."""
    s = _by_bf.get(bf_name.strip().upper())
    return s.ftmo if s else None


def to_data_dir(name: str) -> str | None:
    """Any name → canonical data directory name."""
    s = lookup(name)
    return s.data_dir if s else None


def data_dir_to_broker(data_dir: str, broker: str) -> str | None:
    """Data directory name → broker-specific name. broker = 'ftmo' or 'bf'."""
    s = _by_data_dir.get(data_dir.strip().upper())
    if s is None:
        return None
    return s.ftmo if broker == "ftmo" else s.bf


def all_ftmo(asset_class: str | None = None) -> list[str]:
    """All FTMO broker names, optionally filtered by asset_class."""
    return [
        s.ftmo for s in SYMBOLS
        if s.ftmo and (asset_class is None or s.asset_class == asset_class)
    ]


def all_bf(asset_class: str | None = None) -> list[str]:
    """All BF broker names, optionally filtered by asset_class."""
    return [
        s.bf for s in SYMBOLS
        if s.bf and (asset_class is None or s.asset_class == asset_class)
    ]


def all_data_dirs(broker: str | None = None, asset_class: str | None = None) -> list[str]:
    """All data directory names, optionally filtered by broker availability and asset_class."""
    out = []
    for s in SYMBOLS:
        if asset_class and s.asset_class != asset_class:
            continue
        if broker == "ftmo" and not s.ftmo:
            continue
        if broker == "bf" and not s.bf:
            continue
        out.append(s.data_dir)
    return out


if __name__ == "__main__":
    # Print the full mapping table
    print(f"{'COMMON':<12} {'FTMO':<14} {'BF':<14} {'DATA_DIR':<14} {'CLASS'}")
    print("-" * 70)
    for s in SYMBOLS:
        print(f"{s.common:<12} {(s.ftmo or '-'):<14} {(s.bf or '-'):<14} {s.data_dir:<14} {s.asset_class}")
    print(f"\nTotal: {len(SYMBOLS)} symbols")
    print(f"  FTMO: {len(all_ftmo())}")
    print(f"  BF:   {len(all_bf())}")
    print(f"  Both: {len([s for s in SYMBOLS if s.ftmo and s.bf])}")
