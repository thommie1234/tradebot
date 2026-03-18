"""Sovereign Trading System — Launcher & Monitor GUI.

Modern dark UI to start, stop, and monitor all trading bots.
"""
import subprocess
import threading
import time
import os
import sys
import tkinter as tk
from datetime import datetime

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Process definitions ──────────────────────────────────────────
TRADEBOTS = r"C:\tradebots"
PREDMARKET = r"C:\predmarket"
VENV_PYTHON = os.path.join(TRADEBOTS, r".venv\Scripts\python.exe")
PREDMARKET_PYTHON = os.path.join(PREDMARKET, r".venv\Scripts\python.exe")

PROCESSES = [
    ("BF MT5", {
        "cmd": [r"C:\Program Files\BrightFunded MT5 Terminal\terminal64.exe", "/portable"],
        "cwd": None, "env": {}, "group": "terminal", "icon": "MT5",
    }),
    ("FTMO MT5", {
        "cmd": [r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe", "/portable"],
        "cwd": None, "env": {}, "group": "terminal", "icon": "MT5",
    }),
    ("BF Live", {
        "cmd": [VENV_PYTHON, "-u", r"common\live\run_bot.py", "--live", "--account-id", "bright_100k"],
        "cwd": TRADEBOTS,
        "env": {"ENABLE_LIVE_TRADING": "1", "PYTHONUNBUFFERED": "1"},
        "group": "bot", "icon": "LIVE",
    }),
    ("FTMO Live", {
        "cmd": [VENV_PYTHON, "-u", r"common\live\run_bot.py", "--live", "--account-id", "ftmo_100k"],
        "cwd": TRADEBOTS,
        "env": {"ENABLE_LIVE_TRADING": "1", "PYTHONUNBUFFERED": "1", "MT5_MODULE": "MetaTrader5_FTMO"},
        "group": "bot", "icon": "LIVE",
    }),
    ("BF Paper", {
        "cmd": [VENV_PYTHON, "-u", r"common\live\paper_bot.py", "--account-id", "bright_100k"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1"},
        "group": "paper", "icon": "SIM",
    }),
    ("FTMO Paper", {
        "cmd": [VENV_PYTHON, "-u", r"common\live\paper_bot.py", "--account-id", "ftmo_100k"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1", "MT5_MODULE": "MetaTrader5_FTMO"},
        "group": "paper", "icon": "SIM",
    }),
    ("PredMarket", {
        "cmd": [PREDMARKET_PYTHON, "scheduler.py"],
        "cwd": PREDMARKET,
        "env": {"PYTHONUNBUFFERED": "1"},
        "group": "other", "icon": "PRED",
    }),
    ("Telegram", {
        "cmd": [VENV_PYTHON, "-u", r"common\tools\telegram_signals.py"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1", "MT5_MODULE": "MetaTrader5_FTMO"},
        "group": "other", "icon": "TG",
    }),
    ("Copier BF", {
        "cmd": [VENV_PYTHON, "-u", r"common\tools\trade_copier.py", "--account", "bright_100k"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1"},
        "group": "other", "icon": "CPY",
    }),
    ("Copier FTMO", {
        "cmd": [VENV_PYTHON, "-u", r"common\tools\trade_copier.py", "--account", "ftmo_100k"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1", "MT5_MODULE": "MetaTrader5_FTMO"},
        "group": "other", "icon": "CPY",
    }),
]

COLORS = {
    "bg": "#0f0f1a",
    "card": "#1a1a2e",
    "card_hover": "#1f1f3a",
    "accent": "#e94560",
    "accent2": "#0f3460",
    "green": "#00d26a",
    "red": "#ff4757",
    "yellow": "#ffa502",
    "text": "#e0e0e0",
    "text_dim": "#6c7293",
    "terminal_bg": "#0d1117",
    "terminal_fg": "#c9d1d9",
}

GROUP_COLORS = {
    "terminal": "#536dfe",
    "bot": "#e94560",
    "paper": "#00bcd4",
    "other": "#ffa502",
}


class ProcessCard(ctk.CTkFrame):
    """A card representing a single process."""

    def __init__(self, parent, name, config, launcher):
        super().__init__(parent, fg_color=COLORS["card"], corner_radius=12,
                         border_width=1, border_color=COLORS["accent2"])
        self.name = name
        self.config = config
        self.launcher = launcher

        self.grid_columnconfigure(1, weight=1)

        # Icon badge
        group_color = GROUP_COLORS.get(config["group"], COLORS["accent"])
        icon_frame = ctk.CTkFrame(self, width=44, height=44,
                                   fg_color=group_color, corner_radius=8)
        icon_frame.grid(row=0, column=0, rowspan=2, padx=(12, 8), pady=10)
        icon_frame.grid_propagate(False)
        ctk.CTkLabel(icon_frame, text=config["icon"],
                      font=ctk.CTkFont("Consolas", 11, "bold"),
                      text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        # Name
        ctk.CTkLabel(self, text=name,
                      font=ctk.CTkFont("Segoe UI", 14, "bold"),
                      text_color=COLORS["text"]
                      ).grid(row=0, column=1, sticky="sw", pady=(10, 0))

        # Status line
        self.status_var = tk.StringVar(value="STOPPED")
        self.uptime_var = tk.StringVar(value="")
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=1, column=1, sticky="nw", pady=(0, 10))

        self.status_dot = ctk.CTkLabel(status_frame, text="",
                                        width=8, height=8,
                                        font=ctk.CTkFont(size=8),
                                        text_color=COLORS["red"])
        self.status_dot.pack(side="left", padx=(0, 4))

        self.status_label = ctk.CTkLabel(status_frame,
                                          textvariable=self.status_var,
                                          font=ctk.CTkFont("Consolas", 11),
                                          text_color=COLORS["text_dim"])
        self.status_label.pack(side="left")

        self.uptime_label = ctk.CTkLabel(status_frame,
                                          textvariable=self.uptime_var,
                                          font=ctk.CTkFont("Consolas", 10),
                                          text_color=COLORS["text_dim"])
        self.uptime_label.pack(side="left", padx=(8, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=10)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="Start", width=60, height=28,
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color=COLORS["green"], hover_color="#00b359",
            text_color="#000", corner_radius=6,
            command=self._start)
        self.start_btn.pack(side="left", padx=2)

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="Stop", width=60, height=28,
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color=COLORS["red"], hover_color="#cc3a47",
            text_color="#fff", corner_radius=6,
            command=self._stop)
        self.stop_btn.pack(side="left", padx=2)

        self.log_btn = ctk.CTkButton(
            btn_frame, text="Logs", width=50, height=28,
            font=ctk.CTkFont("Segoe UI", 11),
            fg_color=COLORS["accent2"], hover_color="#1a4a80",
            text_color="#fff", corner_radius=6,
            command=self._show_logs)
        self.log_btn.pack(side="left", padx=2)

    def _start(self):
        self.launcher.start_process(self.name)

    def _stop(self):
        self.launcher.stop_process(self.name)

    def _show_logs(self):
        self.launcher.show_logs(self.name)

    def set_status(self, status, uptime=""):
        self.status_var.set(status)
        self.uptime_var.set(uptime)
        if status == "RUNNING":
            self.status_dot.configure(text_color=COLORS["green"])
            self.status_label.configure(text_color=COLORS["green"])
            self.configure(border_color=COLORS["green"])
        elif status == "CRASHED":
            self.status_dot.configure(text_color=COLORS["yellow"])
            self.status_label.configure(text_color=COLORS["yellow"])
            self.configure(border_color=COLORS["yellow"])
        else:
            self.status_dot.configure(text_color=COLORS["red"])
            self.status_label.configure(text_color=COLORS["text_dim"])
            self.configure(border_color=COLORS["accent2"])


class SovereignLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sovereign Trading System")
        self.geometry("1100x780")
        self.configure(fg_color=COLORS["bg"])

        self.procs = {}       # name -> subprocess.Popen
        self.proc_logs = {}   # name -> list of log lines
        self.cards = {}       # name -> ProcessCard
        self.running = True
        self._current_log = None

        self._build_ui()
        self._start_updater()

    def _build_ui(self):
        # ── Header ──
        header = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=0, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="SOVEREIGN",
                      font=ctk.CTkFont("Consolas", 24, "bold"),
                      text_color=COLORS["accent"]).pack(side="left", padx=20)
        ctk.CTkLabel(header, text="TRADING SYSTEM",
                      font=ctk.CTkFont("Segoe UI", 14),
                      text_color=COLORS["text_dim"]).pack(side="left", padx=(0, 20))

        self.clock_label = ctk.CTkLabel(header, text="",
                                         font=ctk.CTkFont("Consolas", 12),
                                         text_color=COLORS["text_dim"])
        self.clock_label.pack(side="right", padx=20)

        # Buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=10)

        ctk.CTkButton(btn_frame, text="START ALL", width=100, height=32,
                        font=ctk.CTkFont("Segoe UI", 12, "bold"),
                        fg_color=COLORS["green"], hover_color="#00b359",
                        text_color="#000", corner_radius=8,
                        command=self._start_all).pack(side="left", padx=4)

        ctk.CTkButton(btn_frame, text="STOP ALL", width=100, height=32,
                        font=ctk.CTkFont("Segoe UI", 12, "bold"),
                        fg_color=COLORS["red"], hover_color="#cc3a47",
                        text_color="#fff", corner_radius=8,
                        command=self._stop_all).pack(side="left", padx=4)

        # ── Main content ──
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=10)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # ── Process grid ──
        grid_scroll = ctk.CTkScrollableFrame(main, fg_color="transparent",
                                              height=280)
        grid_scroll.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        grid_scroll.grid_columnconfigure(0, weight=1)
        grid_scroll.grid_columnconfigure(1, weight=1)

        for i, (name, cfg) in enumerate(PROCESSES):
            card = ProcessCard(grid_scroll, name, cfg, self)
            card.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="ew")
            self.cards[name] = card
            self.proc_logs[name] = []

        # ── Log viewer ──
        log_frame = ctk.CTkFrame(main, fg_color=COLORS["card"], corner_radius=12)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))

        self.log_title = ctk.CTkLabel(log_header, text="System Log",
                                       font=ctk.CTkFont("Consolas", 13, "bold"),
                                       text_color=COLORS["accent"])
        self.log_title.pack(side="left")

        ctk.CTkButton(log_header, text="Clear", width=50, height=24,
                        font=ctk.CTkFont("Segoe UI", 10),
                        fg_color=COLORS["accent2"], corner_radius=4,
                        command=self._clear_log).pack(side="right")

        ctk.CTkButton(log_header, text="System", width=60, height=24,
                        font=ctk.CTkFont("Segoe UI", 10),
                        fg_color=COLORS["accent2"], corner_radius=4,
                        command=lambda: self.show_logs(None)).pack(side="right", padx=4)

        self.log_text = ctk.CTkTextbox(
            log_frame, fg_color=COLORS["terminal_bg"],
            text_color=COLORS["terminal_fg"],
            font=ctk.CTkFont("Consolas", 10),
            corner_radius=8, border_width=0,
            state="disabled", wrap="word"
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.proc_logs["__system__"] = []
        self._system_log("Sovereign Trading System ready.")
        self._system_log("Click START ALL or start individual processes.")

    def _system_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.proc_logs["__system__"].append(line)
        if self._current_log is None:
            self._append_log(line)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def show_logs(self, name):
        self._current_log = name
        title = name if name else "System Log"
        self.log_title.configure(text=title)
        self._clear_log()
        key = name if name else "__system__"
        lines = self.proc_logs.get(key, [])[-500:]
        self.log_text.configure(state="normal")
        for line in lines:
            self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def start_process(self, name):
        if name in self.procs and self.procs[name].poll() is None:
            self._system_log(f"{name} already running.")
            return

        cfg = dict(PROCESSES)[name]
        env = os.environ.copy()
        env.update(cfg["env"])

        try:
            kwargs = dict(
                cwd=cfg.get("cwd"),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            # Terminals get their own window, bots run hidden
            if cfg["group"] == "terminal":
                kwargs.pop("stdout")
                kwargs.pop("stderr")
                kwargs["creationflags"] = 0
            else:
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            proc = subprocess.Popen(cfg["cmd"], **kwargs)
            proc._start_time = time.time()
            self.procs[name] = proc
            self.cards[name].set_status("RUNNING")
            self._system_log(f"{name} started (PID {proc.pid})")

            # Start log reader for non-terminal processes
            if cfg["group"] != "terminal":
                t = threading.Thread(target=self._read_output, args=(name, proc),
                                      daemon=True)
                t.start()
        except Exception as e:
            self._system_log(f"ERROR starting {name}: {e}")

    def stop_process(self, name):
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            self._system_log(f"{name} stopped.")
        self.cards[name].set_status("STOPPED")

    def _start_all(self):
        self._system_log("Starting all systems...")

        def _seq():
            # Terminals first
            for name, cfg in PROCESSES:
                if cfg["group"] == "terminal":
                    self.after(0, self.start_process, name)
            self._system_log("Waiting 15s for MT5 terminals...")
            time.sleep(15)

            # Live bots
            for name, cfg in PROCESSES:
                if cfg["group"] == "bot":
                    self.after(0, self.start_process, name)
            time.sleep(5)

            # Paper + other
            for name, cfg in PROCESSES:
                if cfg["group"] in ("paper", "other"):
                    self.after(0, self.start_process, name)

            self.after(0, self._system_log, "All systems started!")

        threading.Thread(target=_seq, daemon=True).start()

    def _stop_all(self):
        self._system_log("Stopping all systems...")
        for group in ["other", "paper", "bot", "terminal"]:
            for name, cfg in PROCESSES:
                if cfg["group"] == group:
                    self.stop_process(name)
        self._system_log("All systems stopped.")

    def _read_output(self, name, proc):
        try:
            for line in iter(proc.stdout.readline, b''):
                if not self.running:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip() + "\n"
                self.proc_logs[name].append(decoded)
                if len(self.proc_logs[name]) > 2000:
                    self.proc_logs[name] = self.proc_logs[name][-1500:]
                if self._current_log == name:
                    self.after(0, self._append_log, decoded)
        except Exception:
            pass

    def _start_updater(self):
        def _loop():
            while self.running:
                now = time.time()
                for name, _ in PROCESSES:
                    proc = self.procs.get(name)
                    if proc and proc.poll() is None:
                        s = int(now - getattr(proc, '_start_time', now))
                        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
                        self.cards[name].set_status("RUNNING", f"{h:02d}:{m:02d}:{sec:02d}")
                    elif proc and proc.poll() is not None:
                        if self.cards[name].status_var.get() == "RUNNING":
                            self.cards[name].set_status("CRASHED")
                            self.after(0, self._system_log,
                                       f"{name} crashed (exit {proc.returncode})")
                self.clock_label.configure(
                    text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
                time.sleep(1)

        threading.Thread(target=_loop, daemon=True).start()

    def on_close(self):
        self.running = False
        self.destroy()


def main():
    app = SovereignLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
