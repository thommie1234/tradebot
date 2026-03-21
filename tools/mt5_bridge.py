"""MT5 connection helper — thin wrapper around native MetaTrader5 module.

After bridge removal: no more MT5BridgeClient class. The native MetaTrader5
module is used directly everywhere. This file only keeps:
  - initialize_mt5() — retry-based init with terminal path fallback
  - Backwards-compat aliases so existing imports don't break during migration

Multi-terminal support:
  Set MT5_MODULE=MetaTrader5_FTMO to use a duplicated package for FTMO.
  Each copy has its own DLL/IPC pipe, so they can connect to different terminals.
"""
from __future__ import annotations

import importlib
import os
import sys
import time

# Load the right MetaTrader5 module — default or FTMO copy
# When MT5_MODULE is set (e.g. MetaTrader5_FTMO), redirect ALL imports of
# MetaTrader5 to the copy. This ensures every module in the process uses
# the same DLL/IPC pipe, even if they do `import MetaTrader5` directly.
_mt5_module_name = os.getenv("MT5_MODULE", "MetaTrader5")
mt5 = importlib.import_module(_mt5_module_name)
if _mt5_module_name != "MetaTrader5":
    sys.modules["MetaTrader5"] = mt5


class MT5BridgeError(RuntimeError):
    pass


def initialize_mt5(mt5_obj=None, max_attempts=2, retry_seconds=2,
                   terminal_path: str | None = None):
    """Initialize MT5 with explicit terminal path, then fallback to default.

    Args:
        mt5_obj: Ignored (kept for backwards compat). Always uses native mt5.
        max_attempts: Retry count per mode.
        retry_seconds: Delay between retries.
        terminal_path: Path to terminal64.exe. Falls back to MT5_TERMINAL_PATH env.

    Returns:
        (success: bool, last_error: tuple, mode: str)
    """
    if terminal_path is None:
        terminal_path = os.getenv("MT5_TERMINAL_PATH")

    modes = []
    if terminal_path:
        # Use forward slashes and portable mode for multi-terminal support
        fwd_path = terminal_path.replace("\\", "/")
        modes.append(("path", {"path": fwd_path, "portable": True}))
    else:
        # Only use default fallback when no terminal_path is configured
        modes.append(("default", {}))

    last_error = mt5.last_error()

    for mode, kwargs in modes:
        for _ in range(max_attempts):
            if mt5.initialize(**kwargs):
                return True, mt5.last_error(), mode
            last_error = mt5.last_error()
            mt5.shutdown()
            time.sleep(retry_seconds)

    return False, last_error, "failed"


# Backwards-compat aliases — will be removed once all imports are updated
MT5BridgeClient = None  # Sentinel: any isinstance() checks will fail gracefully
get_mt5_bridge = None
