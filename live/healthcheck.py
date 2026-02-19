"""
Healthcheck — MT5 ping, equity tracking, temperature monitoring.

Extracted from HeartbeatMonitor in sovereign_bot.py (lines 406-662).
"""
from __future__ import annotations

import glob as glob_mod
import os
import subprocess
import threading
import time

from config.loader import cfg


class HeartbeatMonitor:
    """Monitors MT5 connection health."""

    TEMP_WARN = 70
    TEMP_CRITICAL = 75
    GPU_TRADING_PAUSE = False
    TEMP_CHECK_INTERVAL = 300

    PING_FAIL_RESTART_THRESHOLD = 3   # restart bridge after N consecutive failures
    BRIDGE_RESTART_COOLDOWN = 300     # seconds between restart attempts

    def __init__(self, logger, mt5_module, on_disconnect=None, discord=None):
        self.logger = logger
        self.mt5 = mt5_module
        self.on_disconnect = on_disconnect
        self.discord = discord
        self.running = False
        self.thread = None
        self.initial_balance = None
        self.daily_start_balance = None
        self.last_reset_date = None
        self._proxy_down = False
        self._consecutive_ping_fails = 0
        self._last_bridge_restart = 0
        self._last_temp_check = 0
        self._temp_warned = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.log('INFO', 'HeartbeatMonitor', 'START', 'Heartbeat monitoring started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.log('INFO', 'HeartbeatMonitor', 'STOP', 'Heartbeat monitoring stopped')

    @staticmethod
    def read_temperatures():
        temps = {}
        try:
            for hwmon in glob_mod.glob("/sys/class/hwmon/hwmon*"):
                name_file = os.path.join(hwmon, "name")
                if not os.path.exists(name_file):
                    continue
                with open(name_file) as f:
                    name = f.read().strip()
                if name == "coretemp":
                    t_file = os.path.join(hwmon, "temp1_input")
                    if os.path.exists(t_file):
                        with open(t_file) as f:
                            temps["cpu"] = int(f.read().strip()) / 1000.0
                    break
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,name",
                 "--format=csv,noheader"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            gpu_temps = {}
            for line in out.splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    gpu_temps[parts[1].strip()] = int(parts[0].strip())
            if gpu_temps:
                hottest_name = max(gpu_temps, key=gpu_temps.get)
                temps["gpu"] = gpu_temps[hottest_name]
                temps["gpu_name"] = hottest_name
                temps["gpu_all"] = gpu_temps
        except Exception:
            pass

        try:
            for hwmon in glob_mod.glob("/sys/class/hwmon/hwmon*"):
                name_file = os.path.join(hwmon, "name")
                if not os.path.exists(name_file):
                    continue
                with open(name_file) as f:
                    name = f.read().strip()
                if name.startswith("nvme"):
                    t_file = os.path.join(hwmon, "temp1_input")
                    if os.path.exists(t_file):
                        with open(t_file) as f:
                            temps["nvme"] = int(f.read().strip()) / 1000.0
                    break
        except Exception:
            pass

        return temps

    def _check_temperatures(self):
        now = time.time()
        if now - self._last_temp_check < self.TEMP_CHECK_INTERVAL:
            return
        self._last_temp_check = now

        temps = self.read_temperatures()
        if not temps:
            return

        cpu = temps.get("cpu", 0)
        gpu = temps.get("gpu", 0)
        nvme = temps.get("nvme", 0)
        hottest = max(cpu, gpu, nvme)

        self.logger.log("DEBUG", "TempMonitor", "TEMPS",
                        f"CPU:{cpu:.0f}°C  GPU:{gpu:.0f}°C  NVMe:{nvme:.0f}°C")

        if hottest >= self.TEMP_CRITICAL:
            self.logger.log("CRITICAL", "TempMonitor", "TEMP_CRITICAL",
                            f"CRITICAL: CPU:{cpu:.0f}°C  GPU:{gpu:.0f}°C  NVMe:{nvme:.0f}°C — TRADING PAUSED")
            HeartbeatMonitor.GPU_TRADING_PAUSE = True
            if self.discord:
                gpu_detail = ""
                for gname, gtemp in temps.get("gpu_all", {}).items():
                    gpu_detail += f"\n  {gname}: {gtemp}°C"
                self.discord.send(
                    "GPU OVERHEAT — TRADING PAUSED",
                    f"CPU: {cpu:.0f}°C\nGPU: {gpu:.0f}°C{gpu_detail}\n"
                    f"NVMe: {nvme:.0f}°C\n\nHard limit: {self.TEMP_CRITICAL}°C\n"
                    f"No new trades until temps drop below {self.TEMP_WARN}°C",
                    "red",
                )
            self._temp_warned = True

        elif hottest >= self.TEMP_WARN:
            if not self._temp_warned:
                self.logger.log("WARNING", "TempMonitor", "TEMP_WARNING",
                                f"WARNING: CPU:{cpu:.0f}°C  GPU:{gpu:.0f}°C  NVMe:{nvme:.0f}°C")
                if self.discord:
                    self.discord.send(
                        "TEMPERATURE WARNING",
                        f"CPU: {cpu:.0f}°C\nGPU: {gpu:.0f}°C\nNVMe: {nvme:.0f}°C\n\n"
                        f"Threshold: {self.TEMP_WARN}°C",
                        "orange",
                    )
                self._temp_warned = True

        elif self._temp_warned and hottest < self.TEMP_WARN - 10:
            was_paused = HeartbeatMonitor.GPU_TRADING_PAUSE
            HeartbeatMonitor.GPU_TRADING_PAUSE = False
            self._temp_warned = False
            self.logger.log("INFO", "TempMonitor", "TEMP_NORMAL",
                            f"Temps normal: CPU:{cpu:.0f}°C  GPU:{gpu:.0f}°C  NVMe:{nvme:.0f}°C"
                            + (" — TRADING RESUMED" if was_paused else ""))
            if self.discord:
                self.discord.send("TEMPERATURE NORMAL",
                                  f"CPU: {cpu:.0f}°C\nGPU: {gpu:.0f}°C\nNVMe: {nvme:.0f}°C",
                                  "green")

    def _try_restart_bridge(self):
        """Auto-restart MT5 bridge proxy after consecutive ping failures."""
        now = time.time()
        if now - self._last_bridge_restart < self.BRIDGE_RESTART_COOLDOWN:
            return  # cooldown active

        self._last_bridge_restart = now
        self._consecutive_ping_fails = 0
        self.logger.log('WARNING', 'HeartbeatMonitor', 'BRIDGE_AUTO_RESTART',
                        'Restarting mt5-bridge-proxy after consecutive ping failures')
        try:
            result = subprocess.run(
                ["systemctl", "--user", "restart", "mt5-bridge-proxy"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                self.logger.log('INFO', 'HeartbeatMonitor', 'BRIDGE_RESTARTED',
                                'mt5-bridge-proxy restarted successfully')
                if self.discord:
                    self.discord.send("MT5 BRIDGE AUTO-RESTART",
                                      "Bridge was unresponsive — restarted automatically.\n"
                                      "Waiting for reconnection...", "orange")
                # Give bridge time to start
                time.sleep(10)
            else:
                self.logger.log('ERROR', 'HeartbeatMonitor', 'BRIDGE_RESTART_FAILED',
                                f'systemctl restart failed: {result.stderr.strip()}')
                if self.discord:
                    self.discord.send("MT5 BRIDGE RESTART FAILED",
                                      f"Could not restart bridge:\n{result.stderr.strip()}", "red")
        except Exception as e:
            self.logger.log('ERROR', 'HeartbeatMonitor', 'BRIDGE_RESTART_ERROR', str(e))

    def _monitor_loop(self):
        from datetime import datetime
        from execution.broker_api import MT5_AVAILABLE

        while self.running:
            try:
                if not MT5_AVAILABLE or self.mt5 is None:
                    self.logger.log('ERROR', 'HeartbeatMonitor', 'MT5_UNAVAILABLE',
                                    'MT5 bridge not available')
                    time.sleep(cfg.HEARTBEAT_INTERVAL)
                    continue

                if hasattr(self.mt5, "ping"):
                    if not self.mt5.ping():
                        self._consecutive_ping_fails += 1
                        self.logger.log('ERROR', 'HeartbeatMonitor', 'MT5_PING_FAILED',
                                        f'MT5 bridge ping failed ({self._consecutive_ping_fails}x)')
                        if not self._proxy_down:
                            self._proxy_down = True
                            if self.discord:
                                self.discord.send("MT5 PROXY DOWN",
                                                  "No response from MT5 bridge. Entering SAFE MODE.",
                                                  "red")
                            if self.on_disconnect:
                                self.on_disconnect("mt5_proxy_down")

                        # Auto-restart bridge after N consecutive failures
                        if self._consecutive_ping_fails >= self.PING_FAIL_RESTART_THRESHOLD:
                            self._try_restart_bridge()

                        time.sleep(cfg.HEARTBEAT_INTERVAL)
                        continue

                if not self.mt5.terminal_info():
                    self.logger.log('ERROR', 'HeartbeatMonitor', 'MT5_DISCONNECTED',
                                    'MT5 terminal not connected')
                    time.sleep(cfg.HEARTBEAT_INTERVAL)
                    continue
                self._consecutive_ping_fails = 0
                if self._proxy_down:
                    self._proxy_down = False
                    self.logger.log('INFO', 'HeartbeatMonitor', 'MT5_PROXY_RECOVERED',
                                    'MT5 bridge reconnected')
                    if self.discord:
                        self.discord.send("MT5 PROXY RECOVERED",
                                          "MT5 bridge is responding again.", "green")

                account_info = self.mt5.account_info()
                if not account_info:
                    time.sleep(cfg.HEARTBEAT_INTERVAL)
                    continue

                if self.initial_balance is None:
                    self.initial_balance = account_info.balance
                    self.daily_start_balance = account_info.balance
                    self.last_reset_date = datetime.now().date()

                current_date = datetime.now().date()
                if current_date > self.last_reset_date:
                    self.daily_start_balance = account_info.balance
                    self.last_reset_date = current_date
                    self.logger.log('INFO', 'HeartbeatMonitor', 'DAILY_RESET',
                                    f'Daily balance reset to ${account_info.balance:.2f}')

                daily_pnl = account_info.equity - self.daily_start_balance
                positions = self.mt5.positions_total()

                self.logger.log_heartbeat(
                    mt5_connected=True,
                    account_balance=account_info.balance,
                    account_equity=account_info.equity,
                    open_positions=positions,
                    daily_pnl=daily_pnl
                )

                daily_loss_pct = daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0
                total_pnl = account_info.equity - self.initial_balance
                total_loss_pct = total_pnl / self.initial_balance if self.initial_balance > 0 else 0

                if daily_loss_pct < -cfg.MAX_DAILY_LOSS_PCT:
                    self.logger.log('CRITICAL', 'HeartbeatMonitor', 'DAILY_LOSS_BREACH',
                                    f'Daily loss {daily_loss_pct*100:.2f}% exceeds limit')

                if total_loss_pct < -cfg.MAX_TOTAL_LOSS_PCT:
                    self.logger.log('CRITICAL', 'HeartbeatMonitor', 'TOTAL_LOSS_BREACH',
                                    f'Total loss {total_loss_pct*100:.2f}% exceeds limit')

                self._check_temperatures()

            except Exception as e:
                self.logger.log('ERROR', 'HeartbeatMonitor', 'EXCEPTION', str(e))

            time.sleep(cfg.HEARTBEAT_INTERVAL)
