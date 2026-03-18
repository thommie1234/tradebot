"""Sovereign Trading System — Launcher & Monitor GUI.

Modern dark UI to start, stop, and monitor all trading bots.
Includes per-account risk management panel.
"""
import json
import subprocess
import threading
import time
import os
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import yaml

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Paths ────────────────────────────────────────────────────────
DEV_ROOT = Path(__file__).resolve().parent
TRADEBOTS = str(DEV_ROOT)
PREDMARKET = r"C:\predmarket"
VENV_PYTHON = str(DEV_ROOT.parent / ".venv" / "Scripts" / "python.exe")
PREDMARKET_PYTHON = os.path.join(PREDMARKET, r".venv\Scripts\python.exe")
ACCOUNTS_YAML = DEV_ROOT / "config" / "accounts.yaml"

# ── Process definitions (dev/ paths, no common/) ────────────────
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
        "cmd": [VENV_PYTHON, "-u", r"live\run_bot.py", "--live", "--account-id", "bright_100k"],
        "cwd": TRADEBOTS,
        "env": {"ENABLE_LIVE_TRADING": "1", "PYTHONUNBUFFERED": "1"},
        "group": "bot", "icon": "LIVE", "account": "bright_100k",
    }),
    ("FTMO Live", {
        "cmd": [VENV_PYTHON, "-u", r"live\run_bot.py", "--live", "--account-id", "ftmo_100k"],
        "cwd": TRADEBOTS,
        "env": {"ENABLE_LIVE_TRADING": "1", "PYTHONUNBUFFERED": "1", "MT5_MODULE": "MetaTrader5_FTMO"},
        "group": "bot", "icon": "LIVE", "account": "ftmo_100k",
    }),
    ("BF Paper", {
        "cmd": [VENV_PYTHON, "-u", r"live\paper_bot.py", "--account-id", "bright_100k"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1"},
        "group": "paper", "icon": "SIM",
    }),
    ("FTMO Paper", {
        "cmd": [VENV_PYTHON, "-u", r"live\paper_bot.py", "--account-id", "ftmo_100k"],
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
        "cmd": [VENV_PYTHON, "-u", r"tools\telegram_signals.py"],
        "cwd": TRADEBOTS,
        "env": {"PYTHONUNBUFFERED": "1", "MT5_MODULE": "MetaTrader5_FTMO"},
        "group": "other", "icon": "TG",
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


# ── Account config helpers ───────────────────────────────────────

def load_accounts_yaml() -> dict:
    """Load accounts.yaml and return the accounts dict."""
    if not ACCOUNTS_YAML.exists():
        return {}
    with open(ACCOUNTS_YAML) as f:
        data = yaml.safe_load(f) or {}
    return data.get("accounts", {})


def save_accounts_yaml(accounts: dict):
    """Write accounts dict back to accounts.yaml."""
    with open(ACCOUNTS_YAML) as f:
        data = yaml.safe_load(f) or {}
    data["accounts"] = accounts
    with open(ACCOUNTS_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_symbol_config(config_path: str) -> dict:
    """Load per-account symbol config JSON."""
    full = DEV_ROOT / config_path
    if not full.exists():
        return {}
    with open(full) as f:
        return json.load(f)


def save_symbol_config(config_path: str, data: dict):
    """Save per-account symbol config JSON."""
    full = DEV_ROOT / config_path
    with open(full, "w") as f:
        json.dump(data, f, indent=2)


# ── Process Card ─────────────────────────────────────────────────

class ProcessCard(ctk.CTkFrame):
    """A card representing a single process."""

    def __init__(self, parent, name, config, launcher):
        super().__init__(parent, fg_color=COLORS["card"], corner_radius=12,
                         border_width=1, border_color=COLORS["accent2"])
        self.name = name
        self.config = config
        self.launcher = launcher

        self.grid_columnconfigure(1, weight=1)

        group_color = GROUP_COLORS.get(config["group"], COLORS["accent"])
        icon_frame = ctk.CTkFrame(self, width=44, height=44,
                                   fg_color=group_color, corner_radius=8)
        icon_frame.grid(row=0, column=0, rowspan=2, padx=(12, 8), pady=10)
        icon_frame.grid_propagate(False)
        ctk.CTkLabel(icon_frame, text=config["icon"],
                      font=ctk.CTkFont("Consolas", 11, "bold"),
                      text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(self, text=name,
                      font=ctk.CTkFont("Segoe UI", 14, "bold"),
                      text_color=COLORS["text"]
                      ).grid(row=0, column=1, sticky="sw", pady=(10, 0))

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


# ── Risk Management Panel ────────────────────────────────────────

class RiskPanel(ctk.CTkFrame):
    """Per-account risk adjustment panel."""

    def __init__(self, parent, system_log_fn):
        super().__init__(parent, fg_color="transparent")
        self._system_log = system_log_fn
        self._account_widgets = {}  # account_id -> {widgets}

        self.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(self, text="RISK MANAGEMENT",
                      font=ctk.CTkFont("Consolas", 16, "bold"),
                      text_color=COLORS["accent"]).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self._accounts_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._accounts_frame.grid(row=1, column=0, sticky="nsew")
        self._accounts_frame.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._load_accounts()

    def _load_accounts(self):
        accounts = load_accounts_yaml()
        row = 0
        for acct_id, acct_cfg in accounts.items():
            if not acct_cfg.get("enabled", False):
                continue
            self._build_account_card(acct_id, acct_cfg, row)
            row += 1

    def _build_account_card(self, acct_id: str, acct_cfg: dict, row: int):
        card = ctk.CTkFrame(self._accounts_frame, fg_color=COLORS["card"],
                             corner_radius=12, border_width=1,
                             border_color=COLORS["accent2"])
        card.grid(row=row, column=0, sticky="ew", pady=6, padx=4)
        card.grid_columnconfigure(1, weight=1)

        # Account header
        name = acct_cfg.get("name", acct_id)
        ctk.CTkLabel(card, text=name,
                      font=ctk.CTkFont("Segoe UI", 15, "bold"),
                      text_color=COLORS["text"]).grid(row=0, column=0, columnspan=3,
                                                        sticky="w", padx=14, pady=(12, 6))

        # ── risk_scale slider ──
        risk_scale = acct_cfg.get("risk_scale", 1.0)
        rs_var = tk.DoubleVar(value=risk_scale)

        rs_frame = ctk.CTkFrame(card, fg_color="transparent")
        rs_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=14, pady=4)
        rs_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(rs_frame, text="Risk Scale:",
                      font=ctk.CTkFont("Consolas", 12),
                      text_color=COLORS["text_dim"]).grid(row=0, column=0, sticky="w")

        rs_value_label = ctk.CTkLabel(rs_frame, text=f"{risk_scale:.2f}",
                                       font=ctk.CTkFont("Consolas", 13, "bold"),
                                       text_color=COLORS["green"] if risk_scale <= 1.0 else COLORS["yellow"])
        rs_value_label.grid(row=0, column=2, sticky="e", padx=(8, 0))

        def _on_rs_change(val):
            v = round(float(val), 2)
            rs_var.set(v)
            color = COLORS["green"] if v <= 1.0 else COLORS["yellow"]
            if v > 1.5:
                color = COLORS["red"]
            rs_value_label.configure(text=f"{v:.2f}", text_color=color)

        rs_slider = ctk.CTkSlider(rs_frame, from_=0.1, to=2.0,
                                    number_of_steps=19,
                                    variable=rs_var,
                                    command=_on_rs_change,
                                    fg_color=COLORS["accent2"],
                                    progress_color=COLORS["green"],
                                    button_color=COLORS["text"],
                                    button_hover_color=COLORS["accent"])
        rs_slider.grid(row=0, column=1, sticky="ew", padx=10)

        # ── max_positions ──
        max_pos = acct_cfg.get("max_positions", 10)
        mp_var = tk.IntVar(value=max_pos)

        mp_frame = ctk.CTkFrame(card, fg_color="transparent")
        mp_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=4)
        mp_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(mp_frame, text="Max Positions:",
                      font=ctk.CTkFont("Consolas", 12),
                      text_color=COLORS["text_dim"]).grid(row=0, column=0, sticky="w")

        mp_value_label = ctk.CTkLabel(mp_frame, text=str(max_pos),
                                       font=ctk.CTkFont("Consolas", 13, "bold"),
                                       text_color=COLORS["text"])
        mp_value_label.grid(row=0, column=2, sticky="e", padx=(8, 0))

        def _on_mp_change(val):
            v = int(round(float(val)))
            mp_var.set(v)
            mp_value_label.configure(text=str(v))

        ctk.CTkSlider(mp_frame, from_=1, to=20, number_of_steps=19,
                        variable=mp_var, command=_on_mp_change,
                        fg_color=COLORS["accent2"],
                        progress_color=COLORS["accent"],
                        button_color=COLORS["text"],
                        button_hover_color=COLORS["accent"]
                        ).grid(row=0, column=1, sticky="ew", padx=10)

        # ── Per-symbol risk table ──
        config_path = acct_cfg.get("config_path", "")
        symbols_data = load_symbol_config(config_path) if config_path else {}

        sym_frame = ctk.CTkFrame(card, fg_color=COLORS["terminal_bg"], corner_radius=8)
        sym_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=14, pady=(8, 4))
        sym_frame.grid_columnconfigure(1, weight=1)
        sym_frame.grid_columnconfigure(2, weight=0)

        # Header
        ctk.CTkLabel(sym_frame, text="Symbol",
                      font=ctk.CTkFont("Consolas", 10, "bold"),
                      text_color=COLORS["text_dim"]).grid(row=0, column=0, sticky="w", padx=8, pady=2)
        ctk.CTkLabel(sym_frame, text="Risk %",
                      font=ctk.CTkFont("Consolas", 10, "bold"),
                      text_color=COLORS["text_dim"]).grid(row=0, column=1, sticky="w", padx=8, pady=2)
        ctk.CTkLabel(sym_frame, text="Enabled",
                      font=ctk.CTkFont("Consolas", 10, "bold"),
                      text_color=COLORS["text_dim"]).grid(row=0, column=2, sticky="e", padx=8, pady=2)

        sym_widgets = {}
        sr = 1
        for sym, sym_cfg in symbols_data.items():
            if sym == "margin_leverage" or not isinstance(sym_cfg, dict):
                continue

            rpt = sym_cfg.get("risk_per_trade", 0.003)
            enabled = sym_cfg.get("enabled", True)

            ctk.CTkLabel(sym_frame, text=sym,
                          font=ctk.CTkFont("Consolas", 11),
                          text_color=COLORS["text"]).grid(row=sr, column=0, sticky="w", padx=8, pady=1)

            rpt_var = tk.StringVar(value=f"{rpt * 100:.2f}")
            rpt_entry = ctk.CTkEntry(sym_frame, textvariable=rpt_var, width=60, height=24,
                                      font=ctk.CTkFont("Consolas", 11),
                                      fg_color=COLORS["card"], border_color=COLORS["accent2"])
            rpt_entry.grid(row=sr, column=1, sticky="w", padx=8, pady=1)

            en_var = tk.BooleanVar(value=enabled)
            en_cb = ctk.CTkCheckBox(sym_frame, text="", variable=en_var, width=20,
                                     fg_color=COLORS["green"],
                                     hover_color=COLORS["accent"],
                                     border_color=COLORS["accent2"])
            en_cb.grid(row=sr, column=2, sticky="e", padx=8, pady=1)

            sym_widgets[sym] = {"risk_var": rpt_var, "enabled_var": en_var}
            sr += 1

        # ── Save button ──
        save_btn = ctk.CTkButton(card, text="Save Risk Settings", height=32,
                                   font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                   fg_color=COLORS["accent"], hover_color="#c73650",
                                   text_color="#fff", corner_radius=8,
                                   command=lambda: self._save_account(
                                       acct_id, config_path, rs_var, mp_var, sym_widgets))
        save_btn.grid(row=4, column=0, columnspan=3, padx=14, pady=(8, 14), sticky="ew")

        self._account_widgets[acct_id] = {
            "risk_scale": rs_var,
            "max_positions": mp_var,
            "symbols": sym_widgets,
            "config_path": config_path,
        }

    def _save_account(self, acct_id: str, config_path: str,
                       rs_var: tk.DoubleVar, mp_var: tk.IntVar,
                       sym_widgets: dict):
        """Save risk settings to accounts.yaml + symbol config.json."""
        # Update accounts.yaml
        accounts = load_accounts_yaml()
        if acct_id in accounts:
            accounts[acct_id]["risk_scale"] = round(rs_var.get(), 2)
            accounts[acct_id]["max_positions"] = mp_var.get()
            save_accounts_yaml(accounts)

        # Update symbol config.json
        if config_path:
            sym_data = load_symbol_config(config_path)
            for sym, widgets in sym_widgets.items():
                if sym in sym_data and isinstance(sym_data[sym], dict):
                    try:
                        new_risk = float(widgets["risk_var"].get()) / 100.0
                        new_risk = max(0.0001, min(0.05, new_risk))  # Clamp 0.01% - 5%
                        sym_data[sym]["risk_per_trade"] = round(new_risk, 4)
                    except ValueError:
                        pass
                    sym_data[sym]["enabled"] = widgets["enabled_var"].get()
            save_symbol_config(config_path, sym_data)

        self._system_log(f"[{acct_id}] Risk settings saved: "
                         f"scale={rs_var.get():.2f}, max_pos={mp_var.get()}")


# ── Main Launcher ────────────────────────────────────────────────

class SovereignLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sovereign Trading System")
        self.geometry("1200x850")
        self.configure(fg_color=COLORS["bg"])

        self.procs = {}
        self.proc_logs = {}
        self.cards = {}
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

        # ── Tabview: Processes | Risk Management ──
        self.tabview = ctk.CTkTabview(self, fg_color=COLORS["bg"],
                                        segmented_button_fg_color=COLORS["card"],
                                        segmented_button_selected_color=COLORS["accent"],
                                        segmented_button_unselected_color=COLORS["accent2"])
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)

        proc_tab = self.tabview.add("Processes")
        risk_tab = self.tabview.add("Risk Management")

        # ── Processes Tab ──
        proc_tab.grid_columnconfigure(0, weight=1)
        proc_tab.grid_rowconfigure(1, weight=1)

        grid_scroll = ctk.CTkScrollableFrame(proc_tab, fg_color="transparent", height=280)
        grid_scroll.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        grid_scroll.grid_columnconfigure(0, weight=1)
        grid_scroll.grid_columnconfigure(1, weight=1)

        for i, (name, cfg) in enumerate(PROCESSES):
            card = ProcessCard(grid_scroll, name, cfg, self)
            card.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="ew")
            self.cards[name] = card
            self.proc_logs[name] = []

        # Log viewer
        log_frame = ctk.CTkFrame(proc_tab, fg_color=COLORS["card"], corner_radius=12)
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

        # ── Risk Management Tab ──
        risk_tab.grid_columnconfigure(0, weight=1)
        risk_tab.grid_rowconfigure(0, weight=1)

        self.risk_panel = RiskPanel(risk_tab, self._system_log)
        self.risk_panel.grid(row=0, column=0, sticky="nsew")

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
            for name, cfg in PROCESSES:
                if cfg["group"] == "terminal":
                    self.after(0, self.start_process, name)
            self._system_log("Waiting 15s for MT5 terminals...")
            time.sleep(15)

            for name, cfg in PROCESSES:
                if cfg["group"] == "bot":
                    self.after(0, self.start_process, name)
            time.sleep(5)

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
