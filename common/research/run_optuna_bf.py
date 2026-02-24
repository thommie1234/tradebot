#!/usr/bin/env python3
"""
Run Optuna optimization with BrightFunded broker costs.

BrightFunded charges NO commission (spread-only model).
Costs are loaded from data/instrument_specs/brightfunded.csv.
Spreads are inferred from BF tick data (downloaded from BF terminal).

Usage:
    # All BF symbols, H1, 80 trials
    python3 research/run_optuna_bf.py --timeframe H1 --trials 80

    # Specific symbols
    python3 research/run_optuna_bf.py --symbols EU50.cash,US100.cash,US30.cash --trials 200

    # H4 timeframe with HTF features
    python3 research/run_optuna_bf.py --timeframe H4 --trials 80 --htf
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# All BrightFunded symbols (directory names as stored in bars/tick_data)
BF_SYMBOLS = [
    # Crypto (slash → underscore)
    "BTC_USD", "ETH_USD", "SOL_USD", "UNI_USD", "AAVE_USD", "ADA_USD",
    "DOT_USD", "XRP_USD", "LTC_USD", "BNB_USD", "LINK_USD", "DASH_USD",
    "ALGO_USD", "XLM_USD", "NEO_USD",
    # Indices
    "EU50.cash", "US100.cash", "US500.cash", "US30.cash", "UK100.cash", "FRA40.cash",
    # Metals (slash → underscore)
    "XAU_USD", "XAG_USD", "XPD_USD", "XPT_USD",
    # Forex (BF uses no slashes, same as FTMO bar dirs)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDJPY", "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDJPY", "NZDCAD", "NZDCHF",
    "CADJPY", "CADCHF", "CHFJPY", "USDSEK",
]

# BF crypto symbols use underscore (BTC_USD) — add to SECTOR_MAP so
# cluster_for_symbol() returns "crypto" instead of defaulting to "index".
BF_CRYPTO_SECTORS = {
    "BTC_USD": "crypto", "ETH_USD": "crypto", "SOL_USD": "crypto",
    "UNI_USD": "crypto", "AAVE_USD": "crypto", "ADA_USD": "crypto",
    "DOT_USD": "crypto", "XRP_USD": "crypto", "LTC_USD": "crypto",
    "BNB_USD": "crypto", "LINK_USD": "crypto", "DASH_USD": "crypto",
    "ALGO_USD": "crypto", "XLM_USD": "crypto", "NEO_USD": "crypto",
}


def _patch_for_bf():
    """Configure the orchestrator to use BF instrument specs (CSV-based costs)."""
    import research.optuna_orchestrator as orch
    from risk.position_sizing import SECTOR_MAP

    # 1. Swap instrument specs to BF CSV (zero commission, correct asset classes)
    orch.INFO_CSVS = ["brightfunded.csv"]
    orch._SYMBOL_SPECS = None  # force reload from new CSV

    # 2. Add BF crypto symbol names to SECTOR_MAP for correct clustering
    SECTOR_MAP.update(BF_CRYPTO_SECTORS)

    # 3. Override slippage — BF spreads are wider, so less hidden slippage
    _orig_slippage = orch.broker_slippage_bps

    def bf_slippage_bps(symbol: str) -> float:
        """BF slippage: lower than FTMO because spreads already account for costs."""
        specs = orch._get_symbol_specs()
        spec = specs.get(symbol)
        if spec is None:
            return _orig_slippage(symbol)

        broker_class = spec.get("asset_class_broker", "")
        if "Crypto" in broker_class:
            return 5.0   # Spreads already wide on BF
        if "Metals" in broker_class:
            return 2.0
        if "Cash" in broker_class:
            return 1.5
        if "Exotic" in broker_class:
            return 2.5
        if broker_class == "Forex":
            return 1.5
        return 2.0

    orch.broker_slippage_bps = bf_slippage_bps


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optuna BF — BrightFunded costs")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (default: all BF)")
    parser.add_argument("--timeframe", type=str, default="H1")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trial-jobs", type=int, default=2)
    parser.add_argument("--htf", action="store_true")
    parser.add_argument("--no-kill", action="store_true")
    parser.add_argument("--min-years", type=int, default=0)
    parser.add_argument("--bar-roots", type=str, default="")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else BF_SYMBOLS

    # Set output dir to distinguish from FTMO runs
    out_dir = f"models/optuna_bf_{args.timeframe}_{len(symbols)}sym"

    # Patch orchestrator to use BF instrument specs CSV
    _patch_for_bf()

    # Build sys.argv for the orchestrator
    sys.argv = [
        "optuna_orchestrator.py",
        "--symbols", ",".join(symbols),
        "--timeframe", args.timeframe,
        "--trials", str(args.trials),
        "--workers", str(args.workers),
        "--trial-jobs", str(args.trial_jobs),
        "--out-dir", out_dir,
    ]
    if args.htf:
        sys.argv.append("--htf")
    if args.no_kill:
        sys.argv.append("--no-kill")
    if args.min_years:
        sys.argv.extend(["--min-years", str(args.min_years)])
    if args.bar_roots:
        sys.argv.extend(["--bar-roots", args.bar_roots])

    # Verify cost loading
    from research.optuna_orchestrator import broker_commission_bps, broker_slippage_bps
    sample = symbols[0] if symbols else "EURUSD"
    fee = broker_commission_bps(sample)
    print(f"[BF-Optuna] Cost verification: {sample} → fee={fee:.1f}bps (expect 0.0)")

    print(f"[BF-Optuna] Symbols: {len(symbols)}")
    print(f"[BF-Optuna] Timeframe: {args.timeframe}")
    print(f"[BF-Optuna] Trials: {args.trials}")
    print(f"[BF-Optuna] Commission: 0 bps (spread-only, from brightfunded.csv)")
    print(f"[BF-Optuna] Output: {out_dir}")
    print()

    from research.optuna_orchestrator import main as orch_main
    orch_main()


if __name__ == "__main__":
    main()
