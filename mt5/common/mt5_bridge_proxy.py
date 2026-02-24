#!/usr/bin/env python3
"""MT5 TCP/JSON proxy server (Wine-side).

Shared bridge proxy code — runs inside Wine with Python 3.11.
Each account runs its own instance on a different port.

Usage via run_wine.sh:
    MT5_BRIDGE_PORT=5056 ../common/run_wine.sh mt5_bridge_proxy.py
"""
from __future__ import annotations

import json
import os
import socketserver

import MetaTrader5 as mt5


def _serialize(obj):
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "_asdict"):
        return {k: _serialize(v) for k, v in obj._asdict().items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        names = obj.dtype.names
        return [
            {name: _serialize(val) for name, val in zip(names, row)}
            for row in obj.tolist()
        ]
    try:
        return _serialize(obj.item())
    except Exception:
        return str(obj)


def _get_constants():
    constants = {}
    for name in dir(mt5):
        if not name.isupper():
            continue
        value = getattr(mt5, name)
        if isinstance(value, (int, float, str)):
            constants[name] = value
    return constants


CONSTANTS = _get_constants()


class MT5Handler(socketserver.StreamRequestHandler):
    """Handles one or more requests per TCP connection (keep-alive)."""

    def handle(self):
        while True:
            try:
                line = self.rfile.readline()
            except Exception:
                return
            if not line or not line.strip():
                return
            line = line.decode("utf-8").strip()
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                self._send({"ok": False, "error": "invalid_json"})
                continue

            cmd = req.get("command")
            params = req.get("params") or {}
            try:
                result = self._dispatch(cmd, params)
                resp = {"ok": True, "result": result, "last_error": mt5.last_error()}
            except Exception as exc:
                resp = {"ok": False, "error": str(exc), "last_error": mt5.last_error()}
            try:
                self._send(resp)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _send(self, payload):
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _dispatch(self, cmd, params):
        if cmd == "ping":
            return True
        if cmd == "get_constants":
            return CONSTANTS
        if cmd == "initialize":
            kwargs = params.get("kwargs") or {}
            return mt5.initialize(**kwargs)
        if cmd == "shutdown":
            return mt5.shutdown()
        if cmd == "last_error":
            return mt5.last_error()
        if cmd == "terminal_info":
            return _serialize(mt5.terminal_info())
        if cmd == "account_info":
            return _serialize(mt5.account_info())
        if cmd == "positions_total":
            return mt5.positions_total()
        if cmd == "positions_get":
            kwargs = params.get("kwargs") or {}
            return _serialize(mt5.positions_get(**kwargs))
        if cmd == "symbol_info":
            return _serialize(mt5.symbol_info(params.get("symbol")))
        if cmd == "symbol_select":
            return mt5.symbol_select(params.get("symbol"), params.get("enable", True))
        if cmd == "symbol_info_tick":
            return _serialize(mt5.symbol_info_tick(params.get("symbol")))
        if cmd == "copy_rates_from_pos":
            return _serialize(
                mt5.copy_rates_from_pos(
                    params.get("symbol"),
                    params.get("timeframe"),
                    params.get("start_pos"),
                    params.get("count"),
                )
            )
        if cmd == "copy_rates_range":
            from datetime import datetime, timezone
            date_from = datetime.fromtimestamp(params.get("date_from"), tz=timezone.utc)
            date_to = datetime.fromtimestamp(params.get("date_to"), tz=timezone.utc)
            return _serialize(
                mt5.copy_rates_range(
                    params.get("symbol"),
                    params.get("timeframe"),
                    date_from,
                    date_to,
                )
            )
        if cmd == "order_send":
            return _serialize(mt5.order_send(params.get("request")))
        if cmd == "history_deals_get":
            from datetime import datetime, timezone
            date_from = datetime.fromtimestamp(params.get("date_from"), tz=timezone.utc)
            date_to = datetime.fromtimestamp(params.get("date_to"), tz=timezone.utc)
            kwargs = {}
            if params.get("position") is not None:
                kwargs["position"] = params.get("position")
            if params.get("group") is not None:
                kwargs["group"] = params.get("group")
            result = mt5.history_deals_get(date_from, date_to, **kwargs)
            return _serialize(result)
        raise ValueError(f"unknown_command:{cmd}")


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def server_bind(self):
        import socket as _socket
        self.socket.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        super().server_bind()


def main():
    host = os.getenv("MT5_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("MT5_BRIDGE_PORT", "5055"))
    with _ReusableTCPServer((host, port), MT5Handler) as server:
        server.daemon_threads = True
        print(f"[MT5_BRIDGE] listening on {host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
