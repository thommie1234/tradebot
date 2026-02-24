#!/usr/bin/env python3
"""MT5 TCP/JSON proxy server (Wine-side).

Robustness features:
- Auto-reconnect to MT5 on IPC failures
- Request throttling to prevent overload
- Exception isolation per request (one bad request won't kill the server)
- Heartbeat logging every 5 minutes
"""
from __future__ import annotations

import json
import os
import socketserver
import sys
import threading
import time
import traceback

import MetaTrader5 as mt5


# ── Globals ──────────────────────────────────────────────────────────
_lock = threading.Lock()          # serialize MT5 calls (MT5 is not thread-safe)
_stats = {"requests": 0, "errors": 0, "reconnects": 0, "started": time.time()}


def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [MT5_BRIDGE] {msg}", flush=True)


# ── MT5 connection management ───────────────────────────────────────
def _ensure_mt5():
    """Check MT5 connection, reconnect if needed."""
    try:
        info = mt5.terminal_info()
        if info is not None:
            return True
    except Exception:
        pass

    # Connection lost — try to reconnect
    _log("MT5 connection lost, reconnecting...")
    _stats["reconnects"] += 1

    for attempt in range(3):
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(2)
        try:
            if mt5.initialize():
                _log(f"MT5 reconnected (attempt {attempt + 1})")
                return True
        except Exception as e:
            _log(f"Reconnect attempt {attempt + 1} failed: {e}")

    _log("MT5 reconnect FAILED after 3 attempts")
    return False


# ── Serialization ────────────────────────────────────────────────────
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


# ── Request handler ──────────────────────────────────────────────────
class MT5Handler(socketserver.StreamRequestHandler):
    """Handles one or more requests per TCP connection.

    Supports two modes:
    - Single-shot: client sends 1 request, gets 1 response, disconnects (legacy)
    - Keep-alive: client sends multiple newline-delimited requests on one connection
    """

    def handle(self):
        while True:
            try:
                line = self.rfile.readline()
            except Exception:
                return
            if not line or not line.strip():
                return  # Client disconnected or empty line

            line = line.decode("utf-8").strip()
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                self._send({"ok": False, "error": "invalid_json"})
                continue

            cmd = req.get("command")
            params = req.get("params") or {}

            with _lock:
                _stats["requests"] += 1

                try:
                    # Ensure MT5 is connected before dispatching
                    if cmd != "ping" and cmd != "get_constants":
                        if not _ensure_mt5():
                            self._send({"ok": False, "error": "mt5_disconnected", "last_error": [-1, "MT5 not connected"]})
                            continue

                    result = self._dispatch(cmd, params)
                    try:
                        last_err = mt5.last_error()
                    except Exception:
                        last_err = [0, ""]
                    resp = {"ok": True, "result": result, "last_error": last_err}
                except Exception as exc:
                    _stats["errors"] += 1
                    try:
                        last_err = mt5.last_error()
                    except Exception:
                        last_err = [-1, str(exc)]
                    resp = {"ok": False, "error": str(exc), "last_error": last_err}
                    if _stats["errors"] <= 20 or _stats["errors"] % 100 == 0:
                        _log(f"ERROR in {cmd}: {exc}")

            try:
                self._send(resp)
            except (BrokenPipeError, ConnectionResetError):
                return  # Client gone

    def _send(self, payload):
        try:
            data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            self.wfile.write(data)
            self.wfile.flush()
        except Exception:
            pass  # Client gone

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
        if cmd == "stats":
            uptime = time.time() - _stats["started"]
            return {**_stats, "uptime_hours": round(uptime / 3600, 2)}
        raise ValueError(f"unknown_command:{cmd}")


# ── Heartbeat thread ─────────────────────────────────────────────────
def _heartbeat():
    """Log stats every 5 minutes."""
    while True:
        time.sleep(300)
        uptime = time.time() - _stats["started"]
        _log(
            f"HEARTBEAT | uptime={uptime/3600:.1f}h "
            f"reqs={_stats['requests']} errs={_stats['errors']} "
            f"reconn={_stats['reconnects']}"
        )


# ── Server ───────────────────────────────────────────────────────────
class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128  # backlog for pending connections

    def server_bind(self):
        import socket as _socket
        self.socket.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        super().server_bind()


def main():
    host = os.getenv("MT5_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("MT5_BRIDGE_PORT", "5055"))

    # Start heartbeat
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    _log(f"Starting on {host}:{port}")
    _log(f"MT5 initialized: {mt5.terminal_info() is not None}")

    with _ReusableTCPServer((host, port), MT5Handler) as server:
        _log(f"Listening on {host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            _log("Shutting down")
        finally:
            mt5.shutdown()


if __name__ == "__main__":
    main()
