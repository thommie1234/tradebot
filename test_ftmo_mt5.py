"""Test FTMO MT5 connection — run with MT5_MODULE=MetaTrader5_FTMO"""
import os
import sys

# Show which module we're using
print(f"MT5_MODULE env: {os.environ.get('MT5_MODULE', 'NOT SET')}")

# Import the way run_bot.py does it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

from tools.mt5_bridge import mt5, initialize_mt5
print(f"mt5 module: {mt5.__name__}")
print(f"sys.modules['MetaTrader5']: {sys.modules.get('MetaTrader5', 'NOT LOADED')}")

# Check if BF bot has MetaTrader5 loaded
import MetaTrader5
print(f"import MetaTrader5 gives: {MetaTrader5.__name__}")

# Try init like run_bot.py does
terminal_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
print(f"\nTrying initialize_mt5(terminal_path={terminal_path})...")
ok, err, mode = initialize_mt5(terminal_path=terminal_path)
print(f"Result: ok={ok}, mode={mode}, error={err}")

if ok:
    info = mt5.account_info()
    if info:
        print(f"Account: {info.login}, Balance: ${info.balance:,.2f}")
    else:
        print(f"account_info() returned None, last_error: {mt5.last_error()}")

    ti = mt5.terminal_info()
    print(f"terminal_info: {ti is not None}")
    if ti:
        print(f"  path: {ti.path}")
        print(f"  connected: {ti.connected}")
else:
    print(f"FAILED. last_error: {mt5.last_error()}")

# Check if another terminal is also accessible
print("\n--- Checking what default init would connect to ---")
mt5.shutdown()
ok2 = mt5.initialize()
if ok2:
    info2 = mt5.account_info()
    print(f"Default init: Account {info2.login}, Balance ${info2.balance:,.2f}")
    mt5.shutdown()
else:
    print(f"Default init also failed: {mt5.last_error()}")
