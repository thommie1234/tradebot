#!/usr/bin/env python3
"""MT5 bridge: direct MetaTrader5 module (Wine) or TCP proxy (Linux)."""
from __future__ import annotations

import json
import os
import socket
import time
from types import SimpleNamespace


class MT5BridgeError(RuntimeError):
    pass


def _dict_to_obj(data):
    if data is None:
        return None
    return SimpleNamespace(**data)


class MT5BridgeClient:
    # Fallback constants so mt5.TIMEFRAME_H1 etc. always work even if bridge is down
    _FALLBACK_CONSTANTS = {
        "TIMEFRAME_M1": 1, "TIMEFRAME_M2": 2, "TIMEFRAME_M3": 3,
        "TIMEFRAME_M4": 4, "TIMEFRAME_M5": 5, "TIMEFRAME_M6": 6,
        "TIMEFRAME_M10": 10, "TIMEFRAME_M12": 12, "TIMEFRAME_M15": 15,
        "TIMEFRAME_M20": 20, "TIMEFRAME_M30": 30, "TIMEFRAME_H1": 16385,
        "TIMEFRAME_H2": 16386, "TIMEFRAME_H3": 16387, "TIMEFRAME_H4": 16388,
        "TIMEFRAME_H6": 16390, "TIMEFRAME_H8": 16392, "TIMEFRAME_H12": 16396,
        "TIMEFRAME_D1": 16408, "TIMEFRAME_W1": 32769, "TIMEFRAME_MN1": 49153,
        "TRADE_ACTION_DEAL": 1, "TRADE_ACTION_PENDING": 5,
        "TRADE_ACTION_SLTP": 6, "TRADE_ACTION_MODIFY": 7,
        "TRADE_ACTION_REMOVE": 8, "ORDER_TYPE_BUY": 0, "ORDER_TYPE_SELL": 1,
        "ORDER_FILLING_FOK": 0, "ORDER_FILLING_IOC": 1, "ORDER_FILLING_BOC": 2,
        "ORDER_TIME_GTC": 0, "POSITION_TYPE_BUY": 0, "POSITION_TYPE_SELL": 1,
        "COPY_TICKS_ALL": 1, "COPY_TICKS_INFO": 2, "COPY_TICKS_TRADE": 4,
        "TRADE_RETCODE_DONE": 10009, "TRADE_RETCODE_DONE_PARTIAL": 10010,
        "TRADE_RETCODE_REQUOTE": 10004, "TRADE_RETCODE_REJECT": 10006,
        "TRADE_RETCODE_CANCEL": 10007, "TRADE_RETCODE_PLACED": 10008,
        "TRADE_RETCODE_ERROR": 10011, "TRADE_RETCODE_TIMEOUT": 10012,
        "TRADE_RETCODE_INVALID": 10013, "TRADE_RETCODE_INVALID_VOLUME": 10014,
        "TRADE_RETCODE_INVALID_PRICE": 10015, "TRADE_RETCODE_NO_MONEY": 10019,
        "TRADE_RETCODE_MARKET_CLOSED": 10018, "TRADE_RETCODE_TOO_MANY_REQUESTS": 10024,
    }

    def __init__(self, host="127.0.0.1", port=5056, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._constants = dict(self._FALLBACK_CONSTANTS)
        # Persistent connection
        self._sock: socket.socket | None = None
        self._lock = __import__('threading').Lock()

    def _connect(self):
        """Establish or reuse a persistent TCP connection."""
        if self._sock is not None:
            return
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._sock.settimeout(self.timeout)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            self._sock = None
            raise

    def _disconnect(self):
        """Close the persistent connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send(self, command, params=None):
        payload = {"command": command}
        if params is not None:
            payload["params"] = params
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")

        with self._lock:
            # Try persistent connection first, fall back to new connection on failure
            for attempt in range(2):
                try:
                    self._connect()
                    self._sock.sendall(data)
                    raw = self._recv_line()
                    if raw is None:
                        # Server closed connection — reconnect and retry
                        self._disconnect()
                        if attempt == 0:
                            continue
                        return None
                    return json.loads(raw)
                except (OSError, json.JSONDecodeError):
                    self._disconnect()
                    if attempt == 0:
                        continue  # Retry once with fresh connection
                    return None
        return None

    def _recv_line(self) -> str | None:
        """Read one newline-terminated response from the persistent socket."""
        chunks = []
        while True:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                return None  # Server closed connection
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).decode("utf-8").strip()
        return raw if raw else None

    def _ensure_constants(self):
        if self._constants:
            return True
        resp = self._send("get_constants")
        if not resp or not resp.get("ok"):
            # Use fallbacks so we never fail on constants
            self._constants = dict(self._FALLBACK_CONSTANTS)
            return True
        self._constants = resp.get("result", {}) or {}
        # Merge fallbacks for any missing keys
        for k, v in self._FALLBACK_CONSTANTS.items():
            self._constants.setdefault(k, v)
        return True

    def __getattr__(self, name):
        if name.isupper():
            if self._ensure_constants() and name in self._constants:
                return self._constants[name]
        raise AttributeError(name)

    def ping(self):
        resp = self._send("ping")
        return bool(resp and resp.get("ok"))

    def initialize(self, **kwargs):
        resp = self._send("initialize", {"kwargs": kwargs})
        return bool(resp and resp.get("ok") and resp.get("result"))

    def shutdown(self):
        resp = self._send("shutdown")
        return bool(resp and resp.get("ok"))

    def last_error(self):
        resp = self._send("last_error")
        if resp and resp.get("ok"):
            return tuple(resp.get("result") or ())
        return (0, "bridge_unavailable")

    def terminal_info(self):
        resp = self._send("terminal_info")
        if resp and resp.get("ok"):
            return _dict_to_obj(resp.get("result"))
        return None

    def account_info(self):
        resp = self._send("account_info")
        if resp and resp.get("ok"):
            return _dict_to_obj(resp.get("result"))
        return None

    def positions_total(self):
        resp = self._send("positions_total")
        if resp and resp.get("ok"):
            return resp.get("result")
        return None

    def positions_get(self, **kwargs):
        resp = self._send("positions_get", {"kwargs": kwargs})
        if resp and resp.get("ok"):
            result = resp.get("result") or []
            return [_dict_to_obj(item) for item in result]
        return None

    def symbol_info(self, symbol):
        resp = self._send("symbol_info", {"symbol": symbol})
        if resp and resp.get("ok"):
            return _dict_to_obj(resp.get("result"))
        return None

    def symbol_select(self, symbol, enable=True):
        resp = self._send("symbol_select", {"symbol": symbol, "enable": enable})
        return bool(resp and resp.get("ok") and resp.get("result"))

    def symbol_info_tick(self, symbol):
        resp = self._send("symbol_info_tick", {"symbol": symbol})
        if resp and resp.get("ok"):
            return _dict_to_obj(resp.get("result"))
        return None

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        resp = self._send(
            "copy_rates_from_pos",
            {"symbol": symbol, "timeframe": timeframe, "start_pos": start_pos, "count": count},
        )
        if resp and resp.get("ok"):
            return resp.get("result")
        return None

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        resp = self._send(
            "copy_rates_range",
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "date_from": date_from.timestamp(),
                "date_to": date_to.timestamp(),
            },
        )
        if resp and resp.get("ok"):
            return resp.get("result")
        return None

    def order_send(self, request):
        resp = self._send("order_send", {"request": request})
        if resp and resp.get("ok"):
            return _dict_to_obj(resp.get("result"))
        return None

    def history_deals_get(self, date_from, date_to, **kwargs):
        params = {
            "date_from": date_from.timestamp(),
            "date_to": date_to.timestamp(),
        }
        if "position" in kwargs:
            params["position"] = kwargs["position"]
        if "group" in kwargs:
            params["group"] = kwargs["group"]
        resp = self._send("history_deals_get", params)
        if resp and resp.get("ok"):
            result = resp.get("result") or []
            return [_dict_to_obj(item) for item in result]
        return None


# Named bridge instances for multi-account support
_bridge_instances: dict[str, MT5BridgeClient] = {}


def get_mt5_bridge(port: int | None = None, name: str | None = None):
    """Get or create an MT5 bridge client.

    Args:
        port: TCP port for bridge proxy. Defaults to MT5_BRIDGE_PORT env var or 5055.
        name: Optional name for caching. If given, reuses existing instance for same name.
              If None, creates a new instance each time (backwards compat).
    """
    mode = os.getenv("MT5_BRIDGE_MODE", "tcp").lower()
    if mode == "direct":
        try:
            import MetaTrader5 as mt5
            return mt5
        except Exception:
            pass

    host = os.getenv("MT5_BRIDGE_HOST", "127.0.0.1")
    if port is None:
        port = int(os.getenv("MT5_BRIDGE_PORT", "5056"))
    timeout = int(os.getenv("MT5_BRIDGE_TIMEOUT", "5"))

    if name is not None:
        if name not in _bridge_instances:
            _bridge_instances[name] = MT5BridgeClient(host=host, port=port, timeout=timeout)
        return _bridge_instances[name]

    return MT5BridgeClient(host=host, port=port, timeout=timeout)


def initialize_mt5(mt5_obj, max_attempts=2, retry_seconds=2):
    """Initialize MT5 with an explicit terminal path first, then fallback to default."""
    terminal_path = os.getenv("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    modes = [("path", {"path": terminal_path}), ("default", {})]
    last_error = mt5_obj.last_error()

    for mode, kwargs in modes:
        for _ in range(max_attempts):
            if mt5_obj.initialize(**kwargs):
                return True, mt5_obj.last_error(), mode
            last_error = mt5_obj.last_error()
            mt5_obj.shutdown()
            time.sleep(retry_seconds)

    return False, last_error, "failed"
