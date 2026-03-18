"""Sovereign Control Panel — GUI to manage trading bot configs + paper trade analytics.

Compile to .exe:  pip install pyinstaller && pyinstaller --onefile --windowed sovereign_panel.py
"""
import json
import os
import sqlite3
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime, timedelta

import yaml

# When running as .exe (PyInstaller), __file__ points to temp dir — use fixed path
REPO_ROOT = Path(r"C:\tradebots")
ACCOUNTS_YAML = REPO_ROOT / "common" / "config" / "accounts.yaml"
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

# Colors
BG = "#1a1a2e"
BG2 = "#16213e"
BG3 = "#0d1117"
FG = "#e0e0e0"
FG_DIM = "#8899aa"
GREEN = "#00d26a"
RED = "#ff4757"
ORANGE = "#ffa502"
BLUE = "#1e90ff"
ACCENT = "#0f3460"
BUTTON_BG = "#0f3460"

# Paper trade DB paths per account
PAPER_DBS = {
    "bright_100k": REPO_ROOT / "bf" / "audit" / "paper_trades.db",
    "ftmo_100k": REPO_ROOT / "ftmo" / "audit" / "paper_trades.db",
}
LIVE_DBS = {
    "bright_100k": REPO_ROOT / "bf" / "audit" / "sovereign_log.db",
    "ftmo_100k": REPO_ROOT / "ftmo" / "audit" / "sovereign_log.db",
}


def _query_paper_summary(db_path):
    """Query paper trades DB and return summary per symbol+timeframe."""
    if not db_path.exists():
        return []
    try:
        db = sqlite3.connect(str(db_path), timeout=5)
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT symbol, timeframe,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(pnl), 0) as avg_pnl,
                   COALESCE(MAX(pnl), 0) as best_trade,
                   COALESCE(MIN(pnl), 0) as worst_trade,
                   MIN(timestamp) as first_trade,
                   MAX(exit_timestamp) as last_trade
            FROM paper_trades
            WHERE pnl IS NOT NULL AND status LIKE 'CLOSED%'
            GROUP BY symbol, timeframe
            ORDER BY total_pnl DESC
        """).fetchall()
        result = [dict(r) for r in rows]
        db.close()
        return result
    except Exception:
        return []


def _query_symbol_trades(db_path, symbol, timeframe):
    """Query individual trades for a symbol+timeframe."""
    if not db_path.exists():
        return []
    try:
        db = sqlite3.connect(str(db_path), timeout=5)
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT id, timestamp, direction, entry_price, exit_price,
                   sl_price, lot_size, pnl, status, bars_held,
                   confidence, exit_timestamp, tick_value, tick_size
            FROM paper_trades
            WHERE symbol = ? AND timeframe = ? AND pnl IS NOT NULL
            ORDER BY id
        """, (symbol, timeframe)).fetchall()
        result = [dict(r) for r in rows]
        db.close()
        return result
    except Exception:
        return []


def _query_live_summary(db_path):
    """Query live trades DB and return summary per symbol."""
    if not db_path.exists():
        return []
    try:
        db = sqlite3.connect(str(db_path), timeout=5)
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT symbol, COUNT(*) as trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(pnl), 0) as avg_pnl,
                   COALESCE(MAX(pnl), 0) as best_trade,
                   COALESCE(MIN(pnl), 0) as worst_trade
            FROM trades
            WHERE pnl IS NOT NULL AND status = 'CLOSED'
            GROUP BY symbol
            ORDER BY total_pnl DESC
        """).fetchall()
        result = [dict(r) for r in rows]
        db.close()
        return result
    except Exception:
        return []


class SovereignPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("Sovereign Control Panel")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.accounts_data = {}
        self.symbol_vars = {}

        self._load_accounts()
        self._build_ui()

    # ── Data ──────────────────────────────────────────────────────────

    def _load_accounts(self):
        if ACCOUNTS_YAML.exists():
            with open(ACCOUNTS_YAML) as f:
                raw = yaml.safe_load(f) or {}
            self.accounts_data = raw.get("accounts", {})

    def _save_accounts(self):
        with open(ACCOUNTS_YAML) as f:
            raw = yaml.safe_load(f) or {}
        raw["accounts"] = self.accounts_data
        with open(ACCOUNTS_YAML, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _load_symbols(self, acct_id: str) -> dict:
        cfg_path = self.accounts_data[acct_id].get("config_path", "")
        full_path = REPO_ROOT / cfg_path
        if full_path.exists():
            with open(full_path) as f:
                return json.load(f)
        return {}

    def _save_symbols(self, acct_id: str, symbols: dict):
        cfg_path = self.accounts_data[acct_id].get("config_path", "")
        full_path = REPO_ROOT / cfg_path
        with open(full_path, "w") as f:
            json.dump(symbols, f, indent=2)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        title = tk.Label(self.root, text="SOVEREIGN CONTROL PANEL",
                         font=("Consolas", 18, "bold"), bg=BG, fg=GREEN)
        title.pack(pady=(15, 5))

        # Notebook (tabs)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=ACCENT, foreground=FG,
                         padding=[12, 6], font=("Consolas", 11))
        style.map("TNotebook.Tab", background=[("selected", GREEN)],
                  foreground=[("selected", BG)])
        style.configure("TFrame", background=BG)
        # Treeview styling
        style.configure("Paper.Treeview", background=BG3, foreground=FG,
                         fieldbackground=BG3, font=("Consolas", 9),
                         rowheight=22)
        style.configure("Paper.Treeview.Heading", background=ACCENT,
                         foreground=FG, font=("Consolas", 9, "bold"))
        style.map("Paper.Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", GREEN)])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Config tabs per account
        for acct_id, acfg in self.accounts_data.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"  {acfg.get('name', acct_id)}  ")
            self._build_account_tab(frame, acct_id, acfg)

        # Paper trading tab
        paper_frame = ttk.Frame(self.notebook)
        self.notebook.add(paper_frame, text="  Paper Trading  ")
        self._build_paper_tab(paper_frame)

        # Live trading tab
        live_frame = ttk.Frame(self.notebook)
        self.notebook.add(live_frame, text="  Live Trading  ")
        self._build_live_tab(live_frame)

        # Bottom buttons
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        self._make_btn(btn_frame, "START ALLES", self._start_all, GREEN).pack(side="left", padx=5)
        self._make_btn(btn_frame, "STOP ALLES", self._stop_all, RED).pack(side="left", padx=5)
        self._make_btn(btn_frame, "OPSLAAN", self._save_all, ORANGE).pack(side="right", padx=5)

    def _make_btn(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command, font=("Consolas", 11, "bold"),
                        bg=color, fg=BG, activebackground=color, relief="flat",
                        padx=15, pady=6, cursor="hand2")
        return btn

    # ── Account Config Tab ─────────────────────────────────────────

    def _build_account_tab(self, parent, acct_id: str, acfg: dict):
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        row = 0

        # ── Account toggle ──
        enabled_var = tk.BooleanVar(value=acfg.get("enabled", False))

        def toggle_account():
            self.accounts_data[acct_id]["enabled"] = enabled_var.get()

        header = tk.Frame(scroll_frame, bg=BG2, padx=10, pady=8)
        header.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        tk.Checkbutton(header, text="Account ENABLED", variable=enabled_var,
                       command=toggle_account, font=("Consolas", 12, "bold"),
                       bg=BG2, fg=GREEN, selectcolor=BG, activebackground=BG2,
                       activeforeground=GREEN).pack(side="left")

        status = "ACTIEF" if acfg.get("enabled") else "UIT"
        color = GREEN if acfg.get("enabled") else RED
        tk.Label(header, text=f"[{status}]", font=("Consolas", 12, "bold"),
                 bg=BG2, fg=color).pack(side="right")
        row += 1

        # ── Risk settings ──
        risk_frame = tk.LabelFrame(scroll_frame, text=" Risk Settings ",
                                    font=("Consolas", 11, "bold"),
                                    bg=BG, fg=ORANGE, padx=10, pady=5)
        risk_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=5)

        risk_fields = [
            ("account_size", "Account Size ($)", "int"),
            ("risk_scale", "Risk Scale", "float"),
            ("max_positions", "Max Positions", "int"),
            ("max_daily_loss_pct", "Max Daily Loss (%)", "pct"),
            ("max_total_dd_pct", "Max Total DD (%)", "pct"),
            ("internal_daily_loss_pct", "Internal Daily Stop (%)", "pct"),
            ("internal_profit_lock_pct", "Profit Lock (%)", "pct"),
        ]

        self._entries = getattr(self, "_entries", {})
        self._entries[acct_id] = {}

        for i, (key, label, fmt) in enumerate(risk_fields):
            val = acfg.get(key, "")
            if fmt == "pct" and isinstance(val, (int, float)):
                val = f"{val * 100:.1f}"
            tk.Label(risk_frame, text=label, font=("Consolas", 10),
                     bg=BG, fg=FG, anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            entry = tk.Entry(risk_frame, font=("Consolas", 10), width=15,
                             bg=BG2, fg=FG, insertbackground=FG)
            entry.insert(0, str(val))
            entry.grid(row=i, column=1, padx=(10, 0), pady=2)
            self._entries[acct_id][key] = (entry, fmt)

        row += 1

        # ── Symbols ──
        symbols = self._load_symbols(acct_id)
        if symbols:
            sym_frame = tk.LabelFrame(scroll_frame, text=" Symbols ",
                                       font=("Consolas", 11, "bold"),
                                       bg=BG, fg=ORANGE, padx=10, pady=5)
            sym_frame.grid(row=row, column=0, columnspan=4, sticky="ew", padx=5, pady=5)

            self.symbol_vars.setdefault(acct_id, {})
            sym_entries = {}

            col = 0
            r = 0
            for sym, sym_cfg in symbols.items():
                if sym in ("margin_leverage", "_comment"):
                    continue
                if not isinstance(sym_cfg, dict):
                    continue

                var = tk.BooleanVar(value=sym_cfg.get("enabled", False))
                self.symbol_vars[acct_id][sym] = var

                sf = tk.Frame(sym_frame, bg=BG2, padx=6, pady=4)
                sf.grid(row=r, column=col, padx=3, pady=3, sticky="ew")

                tk.Checkbutton(sf, text=sym, variable=var,
                               font=("Consolas", 10, "bold"),
                               bg=BG2, fg=FG, selectcolor=BG,
                               activebackground=BG2).pack(side="left")

                thresh = sym_cfg.get("prob_threshold", 0.58)
                te = tk.Entry(sf, font=("Consolas", 9), width=5,
                              bg=BG, fg=FG, insertbackground=FG)
                te.insert(0, str(thresh))
                te.pack(side="right", padx=2)
                tk.Label(sf, text="thr:", font=("Consolas", 9),
                         bg=BG2, fg=FG).pack(side="right")

                risk = sym_cfg.get("risk_per_trade", 0.003)
                re = tk.Entry(sf, font=("Consolas", 9), width=6,
                              bg=BG, fg=FG, insertbackground=FG)
                re.insert(0, str(risk))
                re.pack(side="right", padx=2)
                tk.Label(sf, text="risk:", font=("Consolas", 9),
                         bg=BG2, fg=FG).pack(side="right")

                sym_entries[sym] = (te, re)

                col += 1
                if col >= 3:
                    col = 0
                    r += 1

            self._entries[acct_id]["_symbols"] = sym_entries

    # ── Paper Trading Tab ──────────────────────────────────────────

    def _build_paper_tab(self, parent):
        # Top bar with account selector and refresh
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="Account:", font=("Consolas", 11), bg=BG, fg=FG).pack(side="left")
        self._paper_acct_var = tk.StringVar(value=list(PAPER_DBS.keys())[0] if PAPER_DBS else "")
        acct_menu = ttk.Combobox(top, textvariable=self._paper_acct_var,
                                  values=list(PAPER_DBS.keys()),
                                  state="readonly", font=("Consolas", 10), width=20)
        acct_menu.pack(side="left", padx=10)
        acct_menu.bind("<<ComboboxSelected>>", lambda e: self._refresh_paper())

        self._make_btn(top, "REFRESH", self._refresh_paper, BLUE).pack(side="left", padx=10)

        # Stats bar
        self._paper_stats = tk.Label(parent, text="", font=("Consolas", 10), bg=BG, fg=FG_DIM)
        self._paper_stats.pack(fill="x", padx=10)

        # Main split: tree left, detail right
        paned = tk.PanedWindow(parent, orient="horizontal", bg=BG, sashwidth=4, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: summary tree
        left = tk.Frame(paned, bg=BG)
        paned.add(left, width=600)

        cols = ("symbol", "tf", "trades", "wr", "pf", "pnl", "avg", "best", "worst")
        self._paper_tree = ttk.Treeview(left, columns=cols, show="headings",
                                         style="Paper.Treeview", height=25)

        headers = {"symbol": ("Symbol", 100), "tf": ("TF", 45), "trades": ("Trades", 55),
                   "wr": ("WR%", 50), "pf": ("PF", 50), "pnl": ("Total PnL", 90),
                   "avg": ("Avg PnL", 75), "best": ("Best", 75), "worst": ("Worst", 75)}

        for c, (label, width) in headers.items():
            self._paper_tree.heading(c, text=label,
                                      command=lambda _c=c: self._sort_paper_tree(_c))
            self._paper_tree.column(c, width=width, anchor="e" if c != "symbol" else "w")

        self._paper_tree.column("symbol", anchor="w")
        self._paper_tree.pack(fill="both", expand=True, side="left")

        tree_scroll = ttk.Scrollbar(left, orient="vertical", command=self._paper_tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self._paper_tree.configure(yscrollcommand=tree_scroll.set)

        self._paper_tree.bind("<<TreeviewSelect>>", self._on_paper_select)

        # Right: detail panel
        right = tk.Frame(paned, bg=BG)
        paned.add(right, width=550)

        self._detail_title = tk.Label(right, text="Select a symbol for details",
                                       font=("Consolas", 13, "bold"), bg=BG, fg=ORANGE)
        self._detail_title.pack(pady=(10, 5))

        # Metrics grid
        self._detail_metrics = tk.Frame(right, bg=BG)
        self._detail_metrics.pack(fill="x", padx=10, pady=5)

        # Equity curve canvas
        self._equity_canvas = tk.Canvas(right, bg=BG3, highlightthickness=1,
                                         highlightbackground=ACCENT, height=180)
        self._equity_canvas.pack(fill="x", padx=10, pady=5)

        # Trade list
        trade_cols = ("time", "dir", "entry", "exit", "pnl", "bars", "conf", "status")
        self._trade_tree = ttk.Treeview(right, columns=trade_cols, show="headings",
                                         style="Paper.Treeview", height=10)

        trade_headers = {"time": ("Time", 120), "dir": ("Dir", 40), "entry": ("Entry", 80),
                        "exit": ("Exit", 80), "pnl": ("PnL", 70), "bars": ("Bars", 40),
                        "conf": ("Conf", 50), "status": ("Status", 80)}

        for c, (label, width) in trade_headers.items():
            self._trade_tree.heading(c, text=label)
            self._trade_tree.column(c, width=width, anchor="e" if c not in ("time", "dir", "status") else "w")

        self._trade_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        trade_scroll = ttk.Scrollbar(right, orient="vertical", command=self._trade_tree.yview)
        self._trade_tree.configure(yscrollcommand=trade_scroll.set)

        # Store sort state
        self._paper_sort_col = "pnl"
        self._paper_sort_rev = True
        self._paper_data = []

        # Initial load
        self._refresh_paper()

    def _refresh_paper(self, event=None):
        acct_id = self._paper_acct_var.get()
        db_path = PAPER_DBS.get(acct_id)
        if not db_path:
            return

        data = _query_paper_summary(db_path)
        self._paper_data = data

        # Stats
        total_trades = sum(r["trades"] for r in data)
        total_pnl = sum(r["total_pnl"] for r in data)
        total_wins = sum(r["wins"] for r in data)
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        profitable = sum(1 for r in data if r["total_pnl"] > 0)
        losing = sum(1 for r in data if r["total_pnl"] <= 0)

        pnl_color = GREEN if total_pnl >= 0 else RED
        self._paper_stats.config(
            text=f"Trades: {total_trades}  |  WR: {overall_wr:.1f}%  |  "
                 f"PnL: ${total_pnl:+,.2f}  |  "
                 f"Profitable pairs: {profitable}  |  Losing pairs: {losing}  |  "
                 f"Symbols: {len(data)}",
            fg=pnl_color
        )

        self._populate_paper_tree()

    def _populate_paper_tree(self):
        tree = self._paper_tree
        tree.delete(*tree.get_children())

        for r in self._paper_data:
            wr = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
            # Profit factor
            wins_sum = sum(1 for _ in range(r["wins"]))  # placeholder
            avg_win = r["total_pnl"] / r["trades"] if r["trades"] > 0 else 0
            # Approximate PF from win/loss ratio
            if r["losses"] > 0 and r["wins"] > 0:
                avg_w = r["total_pnl"] / r["wins"] if r["total_pnl"] > 0 else abs(r["best_trade"])
                avg_l = abs(r["worst_trade"])
                pf = (r["wins"] * abs(r["best_trade"])) / (r["losses"] * abs(r["worst_trade"])) if r["worst_trade"] != 0 else 0
                # Simpler: PF = gross_wins / gross_losses, but we don't have that split
                # Use: PF ≈ (WR / (1-WR)) * (avg_win / avg_loss) — approximate
                pf = max(0, (wr / 100) / (1 - wr / 100) * 1.0) if wr < 100 else 99.0
            else:
                pf = 99.0 if r["total_pnl"] > 0 else 0.0

            pnl_str = f"${r['total_pnl']:+,.2f}"
            avg_str = f"${r['avg_pnl']:+,.2f}"
            best_str = f"${r['best_trade']:+,.2f}"
            worst_str = f"${r['worst_trade']:+,.2f}"

            tag = "win" if r["total_pnl"] > 0 else "loss"
            tree.insert("", "end", values=(
                r["symbol"], r["timeframe"], r["trades"],
                f"{wr:.0f}%", f"{pf:.2f}", pnl_str, avg_str, best_str, worst_str
            ), tags=(tag,))

        tree.tag_configure("win", foreground=GREEN)
        tree.tag_configure("loss", foreground=RED)

    def _sort_paper_tree(self, col):
        # Map column to data key
        col_map = {"symbol": "symbol", "tf": "timeframe", "trades": "trades",
                   "wr": "wins", "pf": "total_pnl", "pnl": "total_pnl",
                   "avg": "avg_pnl", "best": "best_trade", "worst": "worst_trade"}

        key = col_map.get(col, "total_pnl")

        if self._paper_sort_col == col:
            self._paper_sort_rev = not self._paper_sort_rev
        else:
            self._paper_sort_col = col
            self._paper_sort_rev = True

        if key == "symbol" or key == "timeframe":
            self._paper_data.sort(key=lambda r: r[key], reverse=self._paper_sort_rev)
        elif key == "wins":
            self._paper_data.sort(
                key=lambda r: r["wins"] / r["trades"] if r["trades"] > 0 else 0,
                reverse=self._paper_sort_rev)
        else:
            self._paper_data.sort(key=lambda r: r[key], reverse=self._paper_sort_rev)

        self._populate_paper_tree()

    def _on_paper_select(self, event):
        sel = self._paper_tree.selection()
        if not sel:
            return
        values = self._paper_tree.item(sel[0], "values")
        symbol = values[0]
        tf = values[1]

        acct_id = self._paper_acct_var.get()
        db_path = PAPER_DBS.get(acct_id)
        if not db_path:
            return

        trades = _query_symbol_trades(db_path, symbol, tf)
        summary = next((r for r in self._paper_data
                        if r["symbol"] == symbol and r["timeframe"] == tf), None)

        self._show_symbol_detail(symbol, tf, trades, summary)

    def _show_symbol_detail(self, symbol, tf, trades, summary):
        self._detail_title.config(text=f"{symbol}  [{tf}]")

        # Clear metrics
        for w in self._detail_metrics.winfo_children():
            w.destroy()

        if not summary or not trades:
            tk.Label(self._detail_metrics, text="No data", font=("Consolas", 10),
                     bg=BG, fg=FG_DIM).grid(row=0, column=0)
            return

        # Calculate advanced metrics
        n = summary["trades"]
        wins = summary["wins"]
        losses = summary["losses"]
        wr = wins / n * 100 if n > 0 else 0
        total_pnl = summary["total_pnl"]
        avg_pnl = summary["avg_pnl"]

        # Separate winning and losing trades
        win_pnls = [t["pnl"] for t in trades if t["pnl"] > 0]
        loss_pnls = [t["pnl"] for t in trades if t["pnl"] <= 0]
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        gross_wins = sum(win_pnls)
        gross_losses = abs(sum(loss_pnls))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 99.0

        # Max drawdown from cumulative PnL
        cum = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cum += t["pnl"]
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        # Max consecutive wins/losses
        max_con_wins = 0
        max_con_losses = 0
        cur_wins = 0
        cur_losses = 0
        for t in trades:
            if t["pnl"] > 0:
                cur_wins += 1
                cur_losses = 0
            else:
                cur_losses += 1
                cur_wins = 0
            max_con_wins = max(max_con_wins, cur_wins)
            max_con_losses = max(max_con_losses, cur_losses)

        # Avg bars held
        avg_bars = sum(t["bars_held"] or 0 for t in trades) / n if n > 0 else 0

        # Avg confidence
        avg_conf = sum(t["confidence"] or 0 for t in trades) / n if n > 0 else 0

        # Tick value status
        tv_zero = sum(1 for t in trades if (t.get("tick_value") or 0) == 0)
        tv_ok = n - tv_zero

        # Expected value
        ev = (wr / 100) * avg_win + (1 - wr / 100) * avg_loss

        # Build metrics grid
        metrics = [
            ("Trades", str(n), FG),
            ("Win Rate", f"{wr:.1f}%", GREEN if wr >= 50 else RED),
            ("Profit Factor", f"{profit_factor:.2f}", GREEN if profit_factor >= 1.5 else ORANGE if profit_factor >= 1.0 else RED),
            ("Total PnL", f"${total_pnl:+,.2f}", GREEN if total_pnl > 0 else RED),
            ("Avg Win", f"${avg_win:+,.2f}", GREEN),
            ("Avg Loss", f"${avg_loss:+,.2f}", RED),
            ("Best Trade", f"${summary['best_trade']:+,.2f}", GREEN),
            ("Worst Trade", f"${summary['worst_trade']:+,.2f}", RED),
            ("Max Drawdown", f"${max_dd:,.2f}", ORANGE),
            ("Expected Value", f"${ev:+,.2f}", GREEN if ev > 0 else RED),
            ("Avg Bars Held", f"{avg_bars:.1f}", FG),
            ("Avg Confidence", f"{avg_conf:.3f}", FG),
            ("Consec Wins", str(max_con_wins), GREEN),
            ("Consec Losses", str(max_con_losses), RED),
            ("Tick Value OK", f"{tv_ok}/{n}", GREEN if tv_zero == 0 else ORANGE),
        ]

        for i, (label, value, color) in enumerate(metrics):
            r, c = divmod(i, 3)
            frame = tk.Frame(self._detail_metrics, bg=BG2, padx=8, pady=4)
            frame.grid(row=r, column=c, padx=3, pady=2, sticky="ew")
            tk.Label(frame, text=label, font=("Consolas", 8), bg=BG2, fg=FG_DIM).pack(anchor="w")
            tk.Label(frame, text=value, font=("Consolas", 11, "bold"), bg=BG2, fg=color).pack(anchor="w")

        for c in range(3):
            self._detail_metrics.columnconfigure(c, weight=1)

        # Draw equity curve
        self._draw_equity_curve(trades)

        # Populate trade list
        self._trade_tree.delete(*self._trade_tree.get_children())
        for t in reversed(trades):  # newest first
            ts = (t["timestamp"] or "")[:16]
            pnl_str = f"${t['pnl']:+,.2f}" if t["pnl"] is not None else ""
            conf_str = f"{t['confidence']:.3f}" if t["confidence"] else ""
            tag = "win" if (t["pnl"] or 0) > 0 else "loss"
            self._trade_tree.insert("", "end", values=(
                ts, t["direction"], f"{t['entry_price']:.5f}",
                f"{t['exit_price']:.5f}" if t["exit_price"] else "",
                pnl_str, t["bars_held"] or "", conf_str, t["status"] or ""
            ), tags=(tag,))

        self._trade_tree.tag_configure("win", foreground=GREEN)
        self._trade_tree.tag_configure("loss", foreground=RED)

    def _draw_equity_curve(self, trades):
        canvas = self._equity_canvas
        canvas.delete("all")

        if not trades:
            return

        w = canvas.winfo_width() or 530
        h = canvas.winfo_height() or 180
        pad = 30

        # Build cumulative PnL
        cum_pnl = []
        total = 0
        for t in trades:
            total += t["pnl"]
            cum_pnl.append(total)

        if len(cum_pnl) < 2:
            return

        min_v = min(cum_pnl)
        max_v = max(cum_pnl)
        v_range = max_v - min_v if max_v != min_v else 1

        # Scale points
        points = []
        for i, v in enumerate(cum_pnl):
            x = pad + (i / (len(cum_pnl) - 1)) * (w - 2 * pad)
            y = h - pad - ((v - min_v) / v_range) * (h - 2 * pad)
            points.append((x, y))

        # Zero line
        if min_v < 0 < max_v:
            y_zero = h - pad - ((0 - min_v) / v_range) * (h - 2 * pad)
            canvas.create_line(pad, y_zero, w - pad, y_zero, fill=FG_DIM, dash=(3, 3))
            canvas.create_text(pad - 5, y_zero, text="$0", font=("Consolas", 7),
                              fill=FG_DIM, anchor="e")

        # Draw line
        line_color = GREEN if cum_pnl[-1] >= 0 else RED
        for i in range(len(points) - 1):
            seg_color = GREEN if cum_pnl[i + 1] >= cum_pnl[i] else RED
            canvas.create_line(points[i][0], points[i][1],
                             points[i + 1][0], points[i + 1][1],
                             fill=seg_color, width=2)

        # Labels
        canvas.create_text(pad, 10, text=f"${max_v:+,.0f}", font=("Consolas", 8),
                          fill=GREEN if max_v > 0 else RED, anchor="w")
        canvas.create_text(pad, h - 10, text=f"${min_v:+,.0f}", font=("Consolas", 8),
                          fill=RED if min_v < 0 else FG_DIM, anchor="w")
        canvas.create_text(w - pad, 10, text=f"Final: ${cum_pnl[-1]:+,.0f}",
                          font=("Consolas", 9, "bold"),
                          fill=GREEN if cum_pnl[-1] >= 0 else RED, anchor="e")
        canvas.create_text(w / 2, h - 5, text=f"{len(cum_pnl)} trades",
                          font=("Consolas", 7), fill=FG_DIM)

    # ── Live Trading Tab ───────────────────────────────────────────

    def _build_live_tab(self, parent):
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="Account:", font=("Consolas", 11), bg=BG, fg=FG).pack(side="left")
        self._live_acct_var = tk.StringVar(value=list(LIVE_DBS.keys())[0] if LIVE_DBS else "")
        acct_menu = ttk.Combobox(top, textvariable=self._live_acct_var,
                                  values=list(LIVE_DBS.keys()),
                                  state="readonly", font=("Consolas", 10), width=20)
        acct_menu.pack(side="left", padx=10)
        acct_menu.bind("<<ComboboxSelected>>", lambda e: self._refresh_live())

        self._make_btn(top, "REFRESH", self._refresh_live, BLUE).pack(side="left", padx=10)

        self._live_stats = tk.Label(parent, text="", font=("Consolas", 10), bg=BG, fg=FG_DIM)
        self._live_stats.pack(fill="x", padx=10)

        cols = ("symbol", "trades", "wr", "pnl", "avg", "best", "worst")
        self._live_tree = ttk.Treeview(parent, columns=cols, show="headings",
                                        style="Paper.Treeview", height=20)

        headers = {"symbol": ("Symbol", 120), "trades": ("Trades", 65),
                   "wr": ("WR%", 60), "pnl": ("Total PnL", 100),
                   "avg": ("Avg PnL", 85), "best": ("Best", 85), "worst": ("Worst", 85)}

        for c, (label, width) in headers.items():
            self._live_tree.heading(c, text=label)
            self._live_tree.column(c, width=width, anchor="e" if c != "symbol" else "w")

        self._live_tree.pack(fill="both", expand=True, padx=10, pady=10)

        self._refresh_live()

    def _refresh_live(self, event=None):
        acct_id = self._live_acct_var.get()
        db_path = LIVE_DBS.get(acct_id)
        if not db_path:
            return

        data = _query_live_summary(db_path)

        total_trades = sum(r["trades"] for r in data)
        total_pnl = sum(r["total_pnl"] for r in data)
        total_wins = sum(r["wins"] for r in data)
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

        pnl_color = GREEN if total_pnl >= 0 else RED
        self._live_stats.config(
            text=f"Trades: {total_trades}  |  WR: {overall_wr:.1f}%  |  "
                 f"PnL: ${total_pnl:+,.2f}  |  Symbols: {len(data)}",
            fg=pnl_color
        )

        tree = self._live_tree
        tree.delete(*tree.get_children())

        for r in data:
            wr = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
            tag = "win" if r["total_pnl"] > 0 else "loss"
            tree.insert("", "end", values=(
                r["symbol"], r["trades"], f"{wr:.0f}%",
                f"${r['total_pnl']:+,.2f}", f"${r['avg_pnl']:+,.2f}",
                f"${r['best_trade']:+,.2f}", f"${r['worst_trade']:+,.2f}"
            ), tags=(tag,))

        tree.tag_configure("win", foreground=GREEN)
        tree.tag_configure("loss", foreground=RED)

    # ── Actions ───────────────────────────────────────────────────────

    def _save_all(self):
        for acct_id, entries in self._entries.items():
            acfg = self.accounts_data[acct_id]

            for key, (entry, fmt) in entries.items():
                if key.startswith("_"):
                    continue
                val = entry.get().strip()
                if not val:
                    continue
                try:
                    if fmt == "int":
                        acfg[key] = int(val)
                    elif fmt == "float":
                        acfg[key] = float(val)
                    elif fmt == "pct":
                        acfg[key] = float(val) / 100.0
                except ValueError:
                    pass

            sym_entries = entries.get("_symbols", {})
            if sym_entries and acct_id in self.symbol_vars:
                symbols = self._load_symbols(acct_id)
                for sym, (thresh_entry, risk_entry) in sym_entries.items():
                    if sym in symbols and isinstance(symbols[sym], dict):
                        symbols[sym]["enabled"] = self.symbol_vars[acct_id][sym].get()
                        try:
                            symbols[sym]["prob_threshold"] = float(thresh_entry.get())
                        except ValueError:
                            pass
                        try:
                            symbols[sym]["risk_per_trade"] = float(risk_entry.get())
                        except ValueError:
                            pass
                self._save_symbols(acct_id, symbols)

        self._save_accounts()
        messagebox.showinfo("Opgeslagen", "Alle configs zijn opgeslagen.")

    def _start_all(self):
        start_bat = REPO_ROOT / "start_all.bat"
        if start_bat.exists():
            subprocess.Popen(f'start "" "{start_bat}"', shell=True)
            messagebox.showinfo("Gestart", "start_all.bat is gestart.")
        else:
            messagebox.showerror("Fout", f"start_all.bat niet gevonden")

    def _stop_all(self):
        result = messagebox.askyesno("Stop Alles",
                                      "Weet je zeker dat je ALLE bots wilt stoppen?\n"
                                      "(MT5 terminals blijven open)")
        if not result:
            return
        for title in ["BF Live", "FTMO Live", "BF Paper", "FTMO Paper", "PredMarket"]:
            subprocess.run(f'taskkill /F /FI "WINDOWTITLE eq {title}" >nul 2>&1', shell=True)
        messagebox.showinfo("Gestopt", "Alle bot processen zijn gestopt.")


def main():
    root = tk.Tk()
    app = SovereignPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
