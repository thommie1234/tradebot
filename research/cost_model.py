"""
Account-agnostic cost model — loads per-instrument costs from YAML + CSV specs.

Usage:
    from research.cost_model import load_cost_model

    cm = load_cost_model("ftmo_100k")
    fee  = cm.commission_bps("NVDA")
    slip = cm.slippage_bps("NVDA")
    cost = cm.total_cost_pct("NVDA", spread_bps=5.6)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COST_MODEL_DIR = REPO_ROOT / "config" / "cost_models"
SPEC_DIR = REPO_ROOT / "data" / "instrument_specs"

# Account ID aliases — accept short names too
_ALIASES: dict[str, str] = {
    "ftmo": "ftmo_100k",
    "bf": "bright_100k",
    "brightfunded": "bright_100k",
    "ttp": "ttp_demo",
}


def _normalise_symbol(csv_symbol: str) -> str:
    """Convert CSV symbol format (AUD/CAD, XAU/USD) to tick-data format."""
    return csv_symbol.strip().replace("/", "_")


def _classify_asset(asset_class_broker: str, symbol: str) -> str:
    """Map broker asset_class string to a slippage_model key."""
    ac = asset_class_broker.lower()
    if "crypto" in ac:
        return "crypto"
    if "exotic" in ac:
        return "forex_exotic"
    if ac == "forex":
        return "forex"
    if "metal" in ac:
        return "metals"
    if "equit" in ac:
        return "equity"
    if "cash" in ac:
        # Sub-classify: energy/commodity vs index
        if any(x in symbol for x in ("OIL", "NATGAS", "HEATOIL", "DXY")):
            return "energy"
        if any(x in symbol for x in ("COCOA", "COFFEE", "CORN", "SOYBEAN",
                                       "WHEAT", "COTTON", "SUGAR")):
            return "commodity"
        return "index"
    return "default"


@dataclass
class CostModel:
    """Per-instrument cost model for a specific account."""

    account_id: str
    slippage_model: dict[str, float] = field(default_factory=dict)
    _specs: dict[str, dict] = field(default_factory=dict, repr=False)

    def _lookup_spec(self, symbol: str) -> dict | None:
        """Look up instrument spec with symbol normalization fallbacks."""
        spec = self._specs.get(symbol)
        if spec is None:
            spec = self._specs.get(symbol.replace("_", "/"))   # BTC_USD → BTC/USD
        if spec is None:
            spec = self._specs.get(symbol.replace("_", ""))    # BTC_USD → BTCUSD
        if spec is None:
            # Try adding slash at common positions (BTCUSD → BTC/USD)
            for i in (3, 4):
                if len(symbol) > i:
                    spec = self._specs.get(symbol[:i] + "/" + symbol[i:])
                    if spec is not None:
                        break
        return spec

    def commission_bps(self, symbol: str) -> float:
        """Return broker commission in basis points for *symbol*.

        Commission types:
          PERCENT/VOLUME — value is already in %, e.g. 0.065 → 6.5 bps
          USD/LOT        — $5 per 100k lot → ~0.5 bps (at par)
          NO_COMMISSION  — 0 bps (spread-only instruments)
        """
        spec = self._lookup_spec(symbol)
        if spec is None:
            return 3.0  # conservative fallback

        comm = spec["commission"]
        ctype = spec["commission_type"]

        if ctype == "NO_COMMISSION" or comm == 0.0:
            return 0.0
        if ctype == "PERCENT/VOLUME":
            return comm * 100.0
        if ctype == "USD/LOT":
            try:
                lot_size = float(spec.get("contract_size", ""))
            except (ValueError, TypeError):
                lot_size = 100_000.0
            return (comm / lot_size) * 1e4

        return 3.0

    def slippage_bps(self, symbol: str) -> float:
        """Return realistic slippage estimate in bps for *symbol*."""
        spec = self._lookup_spec(symbol)
        if spec is None:
            return self.slippage_model.get("default", 3.0)

        key = _classify_asset(spec.get("asset_class_broker", ""), symbol)
        return self.slippage_model.get(key, self.slippage_model.get("default", 3.0))

    def total_cost_pct(self, symbol: str, spread_bps: float = 0.0,
                        pessimistic: bool = False) -> float:
        """Total round-trip cost as a fraction (not percent).

        cost = (commission + spread + 2 × slippage) / 10000

        If pessimistic=True, slippage is multiplied by 1.5× for conservative
        backtest estimates (better to overestimate costs than underestimate).
        """
        fee = self.commission_bps(symbol)
        slip = self.slippage_bps(symbol)
        if pessimistic:
            slip *= 1.5
        return (fee + spread_bps + slip * 2.0) / 1e4

    def cost_breakdown(self, symbol: str, spread_bps: float = 0.0,
                       pessimistic: bool = False) -> dict:
        """Return detailed cost breakdown for logging."""
        fee = self.commission_bps(symbol)
        slip = self.slippage_bps(symbol)
        slip_used = slip * 1.5 if pessimistic else slip
        total = (fee + spread_bps + slip_used * 2.0) / 1e4
        return {
            "fee_bps": fee,
            "spread_bps": spread_bps,
            "slip_bps": slip,
            "slip_used_bps": slip_used,
            "total_bps": (fee + spread_bps + slip_used * 2.0),
            "total_pct": total,
            "pessimistic": pessimistic,
        }

    def has_spec(self, symbol: str) -> bool:
        return self._lookup_spec(symbol) is not None


def load_cost_model(account_id: str) -> CostModel:
    """Load a CostModel from YAML config + instrument spec CSVs.

    Args:
        account_id: e.g. "ftmo_100k", "bright_100k", or aliases "ftmo", "bf"
    """
    account_id = _ALIASES.get(account_id, account_id)

    # Find YAML config
    yaml_path = None
    for candidate in COST_MODEL_DIR.glob("*.yaml"):
        with open(candidate) as f:
            data = yaml.safe_load(f)
        if data and data.get("account_id") == account_id:
            yaml_path = candidate
            break

    if yaml_path is None:
        raise FileNotFoundError(
            f"No cost model YAML found for account_id={account_id!r} "
            f"in {COST_MODEL_DIR}"
        )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    slippage_model = cfg.get("slippage_model", {})
    spec_csvs = cfg.get("spec_csvs", [])

    # Load instrument specs from CSVs
    specs: dict[str, dict] = {}
    for csv_name in spec_csvs:
        csv_path = SPEC_DIR / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            n_cols = len(headers)
            for fields in reader:
                if not fields or not fields[0].strip():
                    continue
                # Handle descriptions containing commas (extra field)
                if len(fields) == n_cols + 1:
                    fields = [fields[0], fields[1] + "," + fields[2]] + fields[3:]
                row = dict(zip(headers, fields))
                raw_sym = row.get("symbol", "").strip()
                if not raw_sym:
                    continue
                sym = _normalise_symbol(raw_sym)
                try:
                    commission = float(row.get("commission", "0") or "0")
                except ValueError:
                    commission = 0.0
                specs[sym] = {
                    "commission": commission,
                    "commission_type": (row.get("commission_type", "") or "").strip(),
                    "contract_size": (row.get("contract_size", "") or "").strip(),
                    "asset_class_broker": (row.get("asset_class", "") or "").strip(),
                    "margin_percent": row.get("margin_percent", ""),
                    "leverage": (row.get("leverage", "") or "").strip(),
                }

    return CostModel(
        account_id=account_id,
        slippage_model=slippage_model,
        _specs=specs,
    )


# Module-level cache
_CACHE: dict[str, CostModel] = {}


def get_cost_model(account_id: str) -> CostModel:
    """Cached version of load_cost_model — loads once per account per process."""
    account_id = _ALIASES.get(account_id, account_id)
    if account_id not in _CACHE:
        _CACHE[account_id] = load_cost_model(account_id)
    return _CACHE[account_id]
