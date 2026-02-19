#!/usr/bin/env python3
"""Shared MT5 bootstrap helpers for Wine-based production scripts."""

import os
import time

import MetaTrader5 as mt5


DEFAULT_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"


def initialize_mt5(max_attempts=2, retry_seconds=2):
    """
    Initialize MT5 with an explicit terminal path first, then fallback to default.
    Returns: (ok: bool, last_error: tuple, mode: str)
    """
    terminal_path = os.getenv("MT5_TERMINAL_PATH", DEFAULT_TERMINAL_PATH)
    modes = [("path", {"path": terminal_path}), ("default", {})]
    last_error = mt5.last_error()

    for mode, kwargs in modes:
        for _ in range(max_attempts):
            if mt5.initialize(**kwargs):
                return True, mt5.last_error(), mode
            last_error = mt5.last_error()
            mt5.shutdown()
            time.sleep(retry_seconds)

    return False, last_error, "failed"
