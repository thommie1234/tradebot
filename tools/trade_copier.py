"""Trade Copier — mirrors trades between accounts with per-account risk adjustment.

Watches a shared signal directory for new trade signals from any account,
and copies them to all other configured accounts.

Signal flow:
  1. order_router writes JSON signal on fill → C:\\tradebots\\trade_signals\\
  2. This copier watches for new signals
  3. Executes on the OTHER account with adjusted risk

Usage:
  # Run for BF (copies FTMO trades to BF):
  python trade_copier.py --account bright_100k

  # Run for FTMO (copies BF trades to FTMO):
  set MT5_MODULE=MetaTrader5_FTMO
  python trade_copier.py --account ftmo_100k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import glob
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / 'config').is_dir() and (_p / 'engine').is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
REPO_ROOT = _p

SIGNAL_DIR = str(REPO_ROOT / "trade_signals")
PROCESSED_DIR = os.path.join(SIGNAL_DIR, "processed")

# Risk per account (override per symbol if needed)
ACCOUNT_RISK = {
    "bright_100k": 0.001,   # 0.1% survival mode
    "ftmo_100k": 0.003,     # 0.3% standard
}

# Symbols to copy (None = copy all)
# Set to a list to filter, e.g. ["EURUSD", "GBPUSD"]
COPY_FILTER = None

# Cooldown: don't copy same symbol+direction within N seconds
COOLDOWN_S = 120


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def write_signal(account_id: str, symbol: str, direction: str,
                 entry_price: float, lot_size: float, sl_price: float,
                 tp_price: float, ticket: int, magic: int,
                 confidence: float = 0.0, atr: float = 0.0):
    """Write a trade signal file. Called from order_router on fill."""
    os.makedirs(SIGNAL_DIR, exist_ok=True)

    ts = datetime.now(timezone.utc)
    signal = {
        "timestamp": ts.isoformat(),
        "source_account": account_id,
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "lot_size": lot_size,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "ticket": ticket,
        "magic": magic,
        "confidence": confidence,
        "atr": atr,
    }

    filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{account_id}_{symbol}_{direction}.json"
    filepath = os.path.join(SIGNAL_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(signal, f, indent=2)

    return filepath


def execute_copy(signal: dict, target_account: str) -> tuple[bool, str]:
    """Execute a copied trade on the target account."""
    from tools.mt5_bridge import mt5

    symbol = signal["symbol"]
    direction = signal["direction"]
    sl_price = signal["sl_price"]
    tp_price = signal["tp_price"]
    entry_price = signal["entry_price"]
    atr = signal.get("atr", 0)

    # Get account info
    account_info = mt5.account_info()
    if not account_info:
        return False, "No account info"

    # Get symbol info
    mt5.symbol_select(symbol, True)
    sym_info = mt5.symbol_info(symbol)
    if not sym_info:
        return False, f"Symbol {symbol} not found"

    # Check if already have position on this symbol
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for p in positions:
            pos_dir = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            if pos_dir == direction:
                return False, f"Already have {direction} position on {symbol}"

    # Calculate lot size with target account's risk
    risk_pct = ACCOUNT_RISK.get(target_account, 0.002)
    equity = account_info.equity

    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return False, "Invalid SL distance"

    tick_value = sym_info.trade_tick_value
    tick_size = sym_info.trade_tick_size
    contract_size = sym_info.trade_contract_size

    if tick_value > 0 and tick_size > 0:
        risk_per_lot = (sl_distance / tick_size) * tick_value
    else:
        risk_per_lot = sl_distance * contract_size

    if risk_per_lot <= 0:
        return False, "Cannot calculate risk per lot"

    risk_amount = equity * risk_pct
    lot_size = risk_amount / risk_per_lot

    # Round to volume step
    vol_step = sym_info.volume_step
    vol_min = sym_info.volume_min
    vol_max = sym_info.volume_max

    if vol_step > 0:
        lot_size = round(lot_size / vol_step) * vol_step
    lot_size = max(vol_min, min(lot_size, vol_max))
    lot_size = round(lot_size, 8)

    # Margin check
    margin_per_lot = entry_price * contract_size
    if margin_per_lot > 0:
        margin_needed = lot_size * margin_per_lot / 30  # assume ~30x leverage for forex
        if margin_needed > account_info.margin_free * 0.8:
            old = lot_size
            lot_size = (account_info.margin_free * 0.8 * 30) / margin_per_lot
            if vol_step > 0:
                lot_size = round(lot_size / vol_step) * vol_step
            lot_size = max(vol_min, lot_size)
            lot_size = round(lot_size, 8)
            _safe_print(f"  [MARGIN] Clamped {old:.2f} -> {lot_size:.2f} lots")

    if lot_size < vol_min:
        return False, f"Lot size {lot_size} below minimum {vol_min}"

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, "No tick data"

    is_buy = direction == "BUY"
    price = tick.ask if is_buy else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL

    # Adjust SL/TP relative to current price (not original entry)
    if atr > 0:
        # Use ATR-based SL/TP from config
        sl_mult = 0.30  # default
        tp_mult = 10.0
        new_sl = price - (atr * sl_mult) if is_buy else price + (atr * sl_mult)
        new_tp = price + (atr * tp_mult) if is_buy else price - (atr * tp_mult)
    else:
        # Mirror SL/TP distance from original
        sl_dist = abs(signal["entry_price"] - sl_price)
        tp_dist = abs(signal["entry_price"] - tp_price)
        new_sl = price - sl_dist if is_buy else price + sl_dist
        new_tp = price + tp_dist if is_buy else price - tp_dist

    # Use magic number range for copies (9000+)
    magic = 9000 + (hash(symbol) % 1000)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": round(new_sl, sym_info.digits),
        "tp": round(new_tp, sym_info.digits),
        "deviation": 20,
        "magic": magic,
        "comment": f"Copy_{signal['source_account'][:4]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True, f"Copied: {lot_size:.2f} lots @ {price:.5f} (ticket={result.order})"
    else:
        rc = result.retcode if result else "None"
        comment = result.comment if result else "No result"
        return False, f"Order failed: rc={rc} {comment}"


def run_copier(target_account: str):
    """Main loop — watch for signals and copy to target account."""
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    from tools.mt5_bridge import mt5, initialize_mt5
    ok, err, mode = initialize_mt5()
    if not ok:
        _safe_print(f"[COPIER] MT5 init failed: {err}")
        return

    info = mt5.account_info()
    _safe_print(f"[COPIER] Target: {target_account}")
    _safe_print(f"[COPIER] MT5: {info.login} | Balance: ${info.balance:,.2f}")
    _safe_print(f"[COPIER] Risk: {ACCOUNT_RISK.get(target_account, 0.002)*100:.1f}%")
    _safe_print(f"[COPIER] Watching: {SIGNAL_DIR}")
    _safe_print(f"[COPIER] Running... (Ctrl+C to stop)")
    _safe_print("=" * 50)

    cooldown = {}  # "symbol_direction" -> last_copy_time

    while True:
        try:
            signals = sorted(glob.glob(os.path.join(SIGNAL_DIR, "*.json")))

            for filepath in signals:
                try:
                    with open(filepath) as f:
                        signal = json.load(f)
                except Exception:
                    continue

                # Skip signals from our own account
                if signal.get("source_account") == target_account:
                    _move_signal(filepath, "skip_own")
                    continue

                # Symbol filter
                if COPY_FILTER and signal["symbol"] not in COPY_FILTER:
                    _move_signal(filepath, "skip_filter")
                    continue

                # Cooldown check
                cd_key = f"{signal['symbol']}_{signal['direction']}"
                last = cooldown.get(cd_key, 0)
                if time.time() - last < COOLDOWN_S:
                    _move_signal(filepath, "skip_cooldown")
                    continue

                # Age check — skip signals older than 5 minutes
                try:
                    sig_time = datetime.fromisoformat(signal["timestamp"])
                    age = (datetime.now(timezone.utc) - sig_time).total_seconds()
                    if age > 300:
                        _move_signal(filepath, "skip_old")
                        continue
                except Exception:
                    pass

                # Execute copy
                _safe_print(f"\n[COPY] {signal['direction']} {signal['symbol']} "
                           f"from {signal['source_account']}")

                ok, result = execute_copy(signal, target_account)
                status = "OK" if ok else "FAIL"
                _safe_print(f"  [{status}] {result}")

                if ok:
                    cooldown[cd_key] = time.time()

                _move_signal(filepath, status.lower())

        except KeyboardInterrupt:
            _safe_print("\n[COPIER] Stopped.")
            break
        except Exception as e:
            _safe_print(f"[COPIER] Error: {e}")

        time.sleep(2)  # Check every 2 seconds


def _move_signal(filepath: str, status: str):
    """Move processed signal to processed directory."""
    try:
        filename = os.path.basename(filepath)
        dest = os.path.join(PROCESSED_DIR, f"{status}_{filename}")
        os.rename(filepath, dest)
    except Exception:
        try:
            os.remove(filepath)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Trade Copier")
    parser.add_argument("--account", type=str, required=True,
                        help="Target account ID (bright_100k or ftmo_100k)")
    args = parser.parse_args()
    run_copier(args.account)


if __name__ == "__main__":
    main()
