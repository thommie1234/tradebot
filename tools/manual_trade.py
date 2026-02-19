#!/usr/bin/env python3
"""Manual trade execution — replicates full bot logic (ATR stops, risk sizing, margin clamp).

Usage:
    python3 tools/manual_trade.py AMZN SELL              # fresh entry
    python3 tools/manual_trade.py AMZN SELL --flip        # close existing + open opposite
    python3 tools/manual_trade.py AMZN --close            # only close
    python3 tools/manual_trade.py PFE BUY --confidence 0.70  # override confidence
    python3 tools/manual_trade.py NVDA BUY --dry-run      # show plan, no execution
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.mt5_bridge import MT5BridgeClient, initialize_mt5

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sovereign_configs.json"
DEVIATION = 10


# ── Helpers ──────────────────────────────────────────────────────────

def load_configs(symbol: str) -> tuple[dict, dict]:
    """Load per-symbol config + margin_leverage from sovereign_configs.json."""
    with open(CONFIG_PATH) as f:
        all_cfg = json.load(f)
    sym_cfg = all_cfg.get(symbol, {})
    margin_leverage = all_cfg.get("margin_leverage",
                                   {"equity": 3.5, "forex": 30, "index": 10,
                                    "commodity": 10, "crypto": 2})
    return sym_cfg, margin_leverage


def calculate_atr(mt5, symbol: str, period: int = 14) -> float | None:
    """ATR(period) from H1 bars — same formula as OrderRouter._calculate_atr()."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period + 1)
    if rates is None or len(rates) < period + 1:
        return None
    tr_values = []
    for i in range(1, len(rates)):
        h, l, pc = rates[i]["high"], rates[i]["low"], rates[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_values.append(tr)
    return sum(tr_values[-period:]) / period


def estimate_leverage(mt5, account_info, asset_class: str,
                      margin_leverage: dict) -> float:
    """Effective leverage from open positions — same as OrderRouter._estimate_leverage()."""
    try:
        if account_info.margin > 0:
            positions = mt5.positions_get()
            if positions:
                total_notional = 0.0
                for p in positions:
                    si = mt5.symbol_info(p.symbol)
                    if si:
                        total_notional += (p.volume * si.trade_contract_size
                                           * p.price_current)
                if total_notional > 0:
                    return total_notional / account_info.margin
    except Exception:
        pass
    return margin_leverage.get(asset_class, 3.5)


def fmt_lots(lots: float, vol_step: float) -> str:
    """Format lot size with appropriate decimals based on volume_step."""
    if vol_step >= 1:
        return f"{lots:.0f}"
    elif vol_step >= 0.1:
        return f"{lots:.1f}"
    return f"{lots:.2f}"


def close_positions(mt5, symbol: str, dry_run: bool = False) -> list[dict]:
    """Close all positions on symbol. Returns list of close results."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        print(f"No open positions on {symbol}")
        return []

    results = []
    for pos in positions:
        pos_dir = "BUY" if pos.type == 0 else "SELL"
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask
        pnl = pos.profit + getattr(pos, "swap", 0)

        if dry_run:
            print(f"[DRY] Would close {pos_dir} ticket {pos.ticket} "
                  f"@ {close_price:.2f} (PnL: ${pnl:+.2f})")
            results.append({"ticket": pos.ticket, "pnl": pnl, "ok": True})
            continue

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": DEVIATION,
            "magic": getattr(pos, "magic", 2000),
            "comment": "Manual_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Closed {pos_dir} ticket {pos.ticket} "
                  f"@ {close_price:.2f} (PnL: ${pnl:+.2f})")
            results.append({"ticket": pos.ticket, "pnl": pnl, "ok": True})
        else:
            rc = result.retcode if result else "None"
            comment = getattr(result, "comment", "") if result else ""
            print(f"FAILED to close ticket {pos.ticket}: {rc} {comment}")
            results.append({"ticket": pos.ticket, "ok": False})

    return results


def verify_position(mt5, symbol: str, ticket: int, vol_step: float):
    """Verify position exists with correct SL/TP after execution."""
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for p in positions:
            if getattr(p, "ticket", 0) == ticket:
                p_dir = "BUY" if p.type == 0 else "SELL"
                print(f"VERIFIED: {p_dir} {fmt_lots(p.volume, vol_step)} lots "
                      f"@ {p.price_open:.2f} SL={p.sl:.2f} TP={p.tp:.2f}")
                return True
    print(f"WARNING: ticket {ticket} not found in positions_get()")
    return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manual trade execution — replicates full bot logic")
    parser.add_argument("symbol", help="Trading symbol (e.g. AMZN, NVDA, PFE)")
    parser.add_argument("direction", nargs="?", choices=["BUY", "SELL"],
                        help="BUY or SELL (required unless --close)")
    parser.add_argument("--flip", action="store_true",
                        help="Close existing position first, then open opposite")
    parser.add_argument("--close", action="store_true",
                        help="Only close existing position(s), no new trade")
    parser.add_argument("--confidence", type=float, default=0.65,
                        help="ML confidence override (default: 0.65)")
    parser.add_argument("--risk", type=float, default=None,
                        help="Override risk_per_trade from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show calculations without executing")
    parser.add_argument("--port", type=int, default=None,
                        help="MT5 bridge port (default: env MT5_BRIDGE_PORT or 5056)")
    args = parser.parse_args()

    if not args.close and not args.direction:
        parser.error("direction (BUY/SELL) is required unless --close is used")

    symbol = args.symbol
    port = args.port or int(os.getenv("MT5_BRIDGE_PORT", "5056"))

    # ── 1. Connect ───────────────────────────────────────────────────
    print(f"Connecting to MT5 bridge on port {port}...")
    mt5 = MT5BridgeClient(port=port)
    ok, err, mode = initialize_mt5(mt5)
    if not ok:
        print(f"FATAL: MT5 init failed: {err}")
        sys.exit(1)
    print(f"MT5 connected ({mode})\n")

    # ── 2. Load config ───────────────────────────────────────────────
    sym_cfg, margin_leverage = load_configs(symbol)
    if not sym_cfg:
        print(f"WARNING: {symbol} not in sovereign_configs.json — using defaults")
        sym_cfg = {
            "asset_class": "equity", "atr_period": 14,
            "atr_sl_mult": 1.5, "atr_tp_mult": 4.5,
            "risk_per_trade": 0.003, "magic_number": 2000,
        }

    # ── 3. Close-only mode ───────────────────────────────────────────
    if args.close:
        close_positions(mt5, symbol, dry_run=args.dry_run)
        if not args.dry_run:
            remaining = mt5.positions_get(symbol=symbol)
            if not remaining:
                print(f"VERIFIED: No positions remaining on {symbol}")
            else:
                print(f"WARNING: {len(remaining)} position(s) still open on {symbol}")
        mt5.shutdown()
        return

    direction = args.direction

    # ── 4. Account info + tick ───────────────────────────────────────
    account = mt5.account_info()
    if not account:
        print("FATAL: Could not get account info")
        sys.exit(1)

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"FATAL: Could not get tick for {symbol}")
        sys.exit(1)

    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        print(f"FATAL: Could not get symbol info for {symbol}")
        sys.exit(1)

    # ── 5. ATR ───────────────────────────────────────────────────────
    atr_period = sym_cfg.get("atr_period", 14)
    atr = calculate_atr(mt5, symbol, atr_period)
    if atr is None:
        print(f"FATAL: Could not calculate ATR({atr_period}) for {symbol}")
        sys.exit(1)

    # ── 6. SL/TP ────────────────────────────────────────────────────
    sl_mult = sym_cfg.get("atr_sl_mult", 1.5)
    tp_mult = sym_cfg.get("atr_tp_mult", 4.5)
    confidence = args.confidence

    sl_distance = atr * sl_mult
    tp_distance = atr * tp_mult

    # F8: Confidence-scaled TP — 0.55->1.0x, 0.75->1.36x, 0.85->1.55x, cap 2.0x
    tp_confidence_factor = max(1.0, min(2.0, confidence / 0.55))
    tp_distance *= tp_confidence_factor

    entry_price = tick.ask if direction == "BUY" else tick.bid

    if direction == "BUY":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    # ── 7. Position sizing ──────────────────────────────────────────
    config_risk = sym_cfg.get("risk_per_trade", 0.003)
    risk_pct = args.risk if args.risk is not None else config_risk

    # F4: Confidence-scaled sizing — 0.55->0.5x, 0.70->1.0x, 0.85->1.5x, cap 2.0x
    conf_multiplier = max(0.5, min(2.0, 0.5 + (confidence - 0.55) * 3.33))
    effective_risk = risk_pct * conf_multiplier

    contract_size = getattr(sym_info, "trade_contract_size", 1.0)
    vol_step = getattr(sym_info, "volume_step", 0.01)
    vol_min = getattr(sym_info, "volume_min", 0.01)
    vol_max = getattr(sym_info, "volume_max", 10000.0)

    if sl_distance <= 0 or contract_size <= 0:
        print("FATAL: sl_distance or contract_size is zero")
        sys.exit(1)

    lots = (account.equity * effective_risk) / (sl_distance * contract_size)

    # Round to volume step
    if vol_step > 0:
        lots = round(lots / vol_step) * vol_step
        lots = round(lots, 8)
    lots = max(vol_min, min(vol_max, lots))

    # ── 8. Margin clamp ─────────────────────────────────────────────
    asset_class = sym_cfg.get("asset_class", "equity")
    eff_leverage = estimate_leverage(mt5, account, asset_class, margin_leverage)
    margin_per_lot = entry_price * contract_size

    if margin_per_lot > 0:
        margin_needed = (lots * margin_per_lot) / eff_leverage
        max_margin = account.margin_free * 0.80
        if margin_needed > max_margin > 0:
            old_lots = lots
            lots = (max_margin * eff_leverage) / margin_per_lot
            if vol_step > 0:
                lots = round(lots / vol_step) * vol_step
                lots = round(lots, 8)
            lots = max(vol_min, lots)
            print(f"MARGIN CLAMP: {fmt_lots(old_lots, vol_step)} -> "
                  f"{fmt_lots(lots, vol_step)} lots "
                  f"(free=${account.margin_free:.0f}, "
                  f"leverage={eff_leverage:.1f}x)")

    # Final margin calculation for display
    margin_needed = (lots * margin_per_lot) / eff_leverage if margin_per_lot > 0 else 0
    margin_pct = (margin_needed / account.margin_free * 100) if account.margin_free > 0 else 0
    risk_usd = account.equity * effective_risk

    # ── 9. Display plan ──────────────────────────────────────────────
    sl_delta = f"+{sl_distance:.2f}" if direction == "SELL" else f"-{sl_distance:.2f}"
    tp_delta = f"-{tp_distance:.2f}" if direction == "SELL" else f"+{tp_distance:.2f}"
    lots_str = fmt_lots(lots, vol_step)

    print(f"{symbol} {direction} | ATR: {atr:.2f} | "
          f"SL: {sl_price:.2f} ({sl_delta}) | TP: {tp_price:.2f} ({tp_delta})")
    print(f"Lots: {lots_str} | Risk: {effective_risk:.1%} (${risk_usd:.0f}) | "
          f"Margin: ${margin_needed:,.0f} ({margin_pct:.0f}% of free)")
    print("---")

    # ── 10. Dry-run check ────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] No orders executed.")
        mt5.shutdown()
        return

    # ── 11. Flip/close existing ──────────────────────────────────────
    if args.flip:
        close_positions(mt5, symbol)

    # ── 12. Open order — send WITHOUT SL/TP (avoids "Invalid stops") ─
    magic = sym_cfg.get("magic_number", 2000)
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

    # Refresh tick after potential close
    tick = mt5.symbol_info_tick(symbol)
    entry_price = tick.ask if direction == "BUY" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": entry_price,
        "deviation": DEVIATION,
        "magic": magic,
        "comment": f"Manual_{symbol}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
        rc = result.retcode if result else "None"
        comment = getattr(result, "comment", "") if result else ""
        print(f"ORDER FAILED: {rc} {comment}")
        mt5.shutdown()
        sys.exit(1)

    ticket = result.order
    fill_price = getattr(result, "price", 0) or entry_price
    print(f"Opened {direction} {lots_str} lots @ {fill_price:.2f} ticket {ticket}")

    # Recalculate SL/TP based on actual fill price
    if direction == "BUY":
        sl_price = fill_price - sl_distance
        tp_price = fill_price + tp_distance
    else:
        sl_price = fill_price + sl_distance
        tp_price = fill_price - tp_distance

    # ── 13. Modify with SL/TP ────────────────────────────────────────
    modify_request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": sl_price,
        "tp": tp_price,
    }

    mod_result = mt5.order_send(modify_request)
    if mod_result and mod_result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Modified SL={sl_price:.2f} TP={tp_price:.2f}")
    else:
        rc = mod_result.retcode if mod_result else "None"
        comment = getattr(mod_result, "comment", "") if mod_result else ""
        print(f"WARNING: SL/TP modify failed: {rc} {comment}")
        print(f"  -> Manually set SL={sl_price:.2f} TP={tp_price:.2f}")

    # ── 14. Verify ───────────────────────────────────────────────────
    verify_position(mt5, symbol, ticket, vol_step)

    mt5.shutdown()


if __name__ == "__main__":
    main()
