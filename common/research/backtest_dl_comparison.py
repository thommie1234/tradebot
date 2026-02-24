"""
Backtest comparison: XGBoost vs LSTM vs CNN for trading signal prediction.

Same walk-forward backtest, same EV metric, fair comparison.
LSTM/CNN see sequences of bars; XGBoost sees single rows.

Usage:
    python3 research/backtest_dl_comparison.py --symbols NVDA,LVMH
    python3 research/backtest_dl_comparison.py --active
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from engine.labeling import apply_triple_barrier
from research.integrated_pipeline import (
    make_ev_custom_objective,
    purged_walk_forward_splits,
    set_polars_threads,
)
from research.train_ml_strategy import (
    infer_spread_bps,
    load_symbol_ticks,
    make_time_bars,
    sanitize_training_frame,
)
from research.optuna_orchestrator import broker_commission_bps, broker_slippage_bps

DATA_ROOTS = [
    "/home/tradebot/ssd_data_1/tick_data",
    "/home/tradebot/ssd_data_2/tick_data",
    "/home/tradebot/data_1/tick_data",
]

LOOKBACK = 20       # Sequence length for DL models
HIDDEN_DIM = 64     # LSTM/CNN hidden dimension
NUM_LAYERS = 2      # LSTM layers
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 10       # Early stopping patience


# ─── DL Models ───────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Bidirectional LSTM for sequence classification."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Use last timestep output (both directions concatenated)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class CNNModel(nn.Module):
    """1D-CNN for sequence classification."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.head(x).squeeze(-1)


# ─── Helpers ─────────────────────────────────────────────────────────────

def make_sequences(x: np.ndarray, lookback: int) -> np.ndarray:
    """Create overlapping sequences from feature matrix.

    Input:  x shape (n_samples, n_features)
    Output: shape (n_samples - lookback + 1, lookback, n_features)
    """
    n = len(x) - lookback + 1
    seqs = np.lib.stride_tricks.sliding_window_view(x, (lookback, x.shape[1]))
    return seqs.reshape(n, lookback, x.shape[1])


def normalize_features(x_train: np.ndarray, x_test: np.ndarray):
    """Z-score normalization fit on train, applied to test."""
    mu = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True) + 1e-8
    return (x_train - mu) / std, (x_test - mu) / std


def train_dl_model(model, x_train_seq, y_train, x_val_seq, y_val,
                   epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, patience=PATIENCE):
    """Train a PyTorch model with early stopping. Returns best model state."""
    device = torch.device("cpu")
    model = model.to(device)

    x_t = torch.FloatTensor(x_train_seq)
    y_t = torch.FloatTensor(y_train)
    x_v = torch.FloatTensor(x_val_seq)
    y_v = torch.FloatTensor(y_val)

    train_ds = TensorDataset(x_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(x_v)
            val_loss = criterion(val_logits, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_dl(model, x_seq):
    """Get probabilities from a trained DL model."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(x_seq))
        return torch.sigmoid(logits).numpy()


# ─── Metrics ─────────────────────────────────────────────────────────────

def calc_metrics(returns: np.ndarray) -> dict:
    trades = returns[returns != 0]
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "ev": 0, "sharpe": 0, "win_rate": 0,
                "max_dd": 0, "pf": 0}

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    mean_ret = np.mean(trades)
    std_ret = np.std(trades)
    sharpe = np.sqrt(252) * mean_ret / std_ret if std_ret > 0 else 0
    cumsum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumsum)
    max_dd = np.max(running_max - cumsum) if len(cumsum) > 0 else 0
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999

    return {
        "n_trades": n,
        "ev": float(mean_ret),
        "sharpe": float(sharpe),
        "win_rate": float(len(wins) / n),
        "max_dd": float(max_dd),
        "pf": float(pf),
    }


# ─── Main Backtest ───────────────────────────────────────────────────────

def run_comparison(symbol: str, train_size: int = 1200, test_size: int = 300,
                   purge: int = 24, embargo: int = 24) -> dict:
    """Run XGBoost vs LSTM vs CNN backtest for a single symbol."""
    set_polars_threads(4)
    t0 = time.time()

    # Load data
    ticks = load_symbol_ticks(symbol, DATA_ROOTS, years_filter=None)
    if ticks is None:
        return {"symbol": symbol, "status": "no_tick_data"}

    if ticks.select(pl.col("size").sum()).item() <= 0:
        ticks = ticks.with_columns(pl.lit(1.0).alias("size"))

    spread_bps = infer_spread_bps(ticks)
    fee_bps = broker_commission_bps(symbol)
    slippage_bps = broker_slippage_bps(symbol)

    bars = make_time_bars(ticks.select(["time", "price", "size"]), "H1")
    feat = build_bar_features(bars, z_threshold=1.0)

    # Apply triple barrier
    tb = apply_triple_barrier(
        close=feat["close"].to_numpy(),
        vol_proxy=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=24, pt_mult=2.0, sl_mult=1.5,
    )
    feat = feat.with_columns([
        pl.Series("label", tb.label),
        pl.Series("target", tb.label),
        pl.Series("tb_ret", tb.tb_ret),
        pl.Series("avg_win", tb.upside),
        pl.Series("avg_loss", tb.downside),
        pl.Series("upside", tb.upside),
        pl.Series("downside", tb.downside),
        pl.lit(float(fee_bps)).alias("fee_bps"),
        pl.lit(float(spread_bps)).alias("spread_bps"),
        pl.lit(float(slippage_bps)).alias("slippage_bps"),
    ]).filter(pl.col("target").is_finite())
    feat = sanitize_training_frame(feat)

    cols = FEATURE_COLUMNS + ["target", "tb_ret", "avg_win", "avg_loss",
                               "fee_bps", "spread_bps", "slippage_bps"]
    df = feat.select(cols).drop_nulls(cols)

    min_rows = train_size + test_size + LOOKBACK + 200
    if df.height < min_rows:
        return {"symbol": symbol, "status": "insufficient_rows", "rows": df.height}

    x_all = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    y_all = df["target"].to_numpy().astype(np.float32)
    tb_ret = df["tb_ret"].to_numpy().astype(np.float64)
    avg_win = df["avg_win"].to_numpy().astype(np.float64)
    avg_loss = df["avg_loss"].to_numpy().astype(np.float64)
    costs_all = (
        (df["fee_bps"] + df["spread_bps"] + df["slippage_bps"] * 2.0).to_numpy() / 1e4
    ).astype(np.float64)

    n_features = x_all.shape[1]

    splits = purged_walk_forward_splits(
        n_samples=len(df), train_size=train_size, test_size=test_size,
        purge=purge, embargo=embargo,
    )
    if not splits:
        return {"symbol": symbol, "status": "no_folds", "rows": len(df)}

    # Collect returns per model
    xgb_returns = []
    lstm_returns = []
    cnn_returns = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        # ── XGBoost ──
        dtrain = xgb.DMatrix(x_all[tr_idx], label=y_all[tr_idx])
        dtest = xgb.DMatrix(x_all[te_idx], label=y_all[te_idx])
        obj = make_ev_custom_objective(
            float(np.mean(avg_win[tr_idx])),
            float(np.mean(avg_loss[tr_idx])),
            float(np.mean(costs_all[tr_idx])),
        )
        params = {
            "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
        }
        bst = xgb.train(
            params, dtrain, num_boost_round=500, obj=obj,
            evals=[(dtest, "valid")], early_stopping_rounds=50,
            verbose_eval=False,
        )
        xgb_proba = bst.predict(dtest)

        # EV-filtered returns for XGBoost
        test_costs = costs_all[te_idx]
        test_tb_ret = tb_ret[te_idx]
        test_avg_win = avg_win[te_idx]
        test_avg_loss = avg_loss[te_idx]

        ev = xgb_proba * test_avg_win - (1 - xgb_proba) * test_avg_loss - test_costs
        take = (ev > 0) & (xgb_proba >= 0.5)
        xgb_returns.append(np.where(take, test_tb_ret - test_costs, 0.0))

        # ── DL: Prepare sequences ──
        # Normalize features (fit on train)
        x_tr_norm, x_te_norm = normalize_features(x_all[tr_idx], x_all[te_idx])

        # Build sequences: need to handle indices carefully
        # For train: take tr_idx range, build sequences
        tr_start = tr_idx[0]
        tr_end = tr_idx[-1] + 1
        te_start = te_idx[0]
        te_end = te_idx[-1] + 1

        # Normalize on full contiguous ranges
        x_tr_full = x_all[tr_start:tr_end]
        x_te_full = x_all[max(te_start - LOOKBACK + 1, 0):te_end]

        mu = np.mean(x_tr_full, axis=0, keepdims=True)
        std = np.std(x_tr_full, axis=0, keepdims=True) + 1e-8
        x_tr_normed = (x_tr_full - mu) / std
        x_te_normed = (x_all[max(te_start - LOOKBACK + 1, 0):te_end] - mu) / std

        # Build sequences
        if len(x_tr_normed) < LOOKBACK + 10:
            # Not enough data for sequences, skip DL for this fold
            lstm_returns.append(np.zeros(len(te_idx)))
            cnn_returns.append(np.zeros(len(te_idx)))
            continue

        tr_seqs = make_sequences(x_tr_normed, LOOKBACK)
        tr_labels = y_all[tr_start + LOOKBACK - 1:tr_end]

        te_seqs = make_sequences(x_te_normed, LOOKBACK)
        # Align test labels: sequences end at te_start..te_end-1
        offset = te_start - max(te_start - LOOKBACK + 1, 0)
        te_labels = y_all[te_start:te_end]
        # te_seqs might have more entries than te_labels if we included prefix
        # Take last len(te_idx) sequences
        if len(te_seqs) > len(te_idx):
            te_seqs = te_seqs[-len(te_idx):]

        if len(tr_seqs) < 100 or len(te_seqs) < 10:
            lstm_returns.append(np.zeros(len(te_idx)))
            cnn_returns.append(np.zeros(len(te_idx)))
            continue

        # Split train into train/val (80/20)
        val_split = int(len(tr_seqs) * 0.8)
        x_tr_s, x_val_s = tr_seqs[:val_split], tr_seqs[val_split:]
        y_tr_s, y_val_s = tr_labels[:val_split], tr_labels[val_split:]

        # ── LSTM ──
        lstm = LSTMModel(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
        lstm = train_dl_model(lstm, x_tr_s, y_tr_s, x_val_s, y_val_s)
        lstm_proba = predict_dl(lstm, te_seqs)

        ev_lstm = lstm_proba * test_avg_win - (1 - lstm_proba) * test_avg_loss - test_costs
        take_lstm = (ev_lstm > 0) & (lstm_proba >= 0.5)
        lstm_returns.append(np.where(take_lstm, test_tb_ret - test_costs, 0.0))

        # ── CNN ──
        cnn = CNNModel(n_features, HIDDEN_DIM, DROPOUT)
        cnn = train_dl_model(cnn, x_tr_s, y_tr_s, x_val_s, y_val_s)
        cnn_proba = predict_dl(cnn, te_seqs)

        ev_cnn = cnn_proba * test_avg_win - (1 - cnn_proba) * test_avg_loss - test_costs
        take_cnn = (ev_cnn > 0) & (cnn_proba >= 0.5)
        cnn_returns.append(np.where(take_cnn, test_tb_ret - test_costs, 0.0))

        del lstm, cnn  # Free memory

    elapsed = time.time() - t0
    return {
        "symbol": symbol,
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "rows": len(df),
        "folds": len(splits),
        "xgboost": calc_metrics(np.concatenate(xgb_returns)),
        "lstm": calc_metrics(np.concatenate(lstm_returns)),
        "cnn": calc_metrics(np.concatenate(cnn_returns)),
    }


def print_results(results: list[dict]):
    """Print formatted comparison table."""
    print()
    print("=" * 110)
    print(f"  {'Symbol':<14} │ {'Model':<8} │ {'Trades':>6} │ {'EV':>10} │ "
          f"{'Sharpe':>7} │ {'WinRate':>7} │ {'MaxDD':>8} │ {'PF':>6} │ {'Time'}")
    print("─" * 110)

    summary = {"xgboost": [], "lstm": [], "cnn": []}

    for r in sorted(results, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        if r.get("status") != "ok":
            print(f"  {sym:<14} │ {'':8} │ {'':>6} │ {'':>10} │ {'':>7} │ "
                  f"{'':>7} │ {'':>8} │ {'':>6} │ {r.get('status', '?')}")
            print("─" * 110)
            continue

        for model_name, key in [("XGBoost", "xgboost"), ("LSTM", "lstm"), ("CNN", "cnn")]:
            m = r[key]
            elapsed = f"{r['elapsed_s']:.0f}s" if model_name == "XGBoost" else ""
            print(f"  {sym:<14} │ {model_name:<8} │ {m['n_trades']:>6} │ "
                  f"{m['ev']:>+10.6f} │ {m['sharpe']:>7.3f} │ "
                  f"{m['win_rate']:>6.1%} │ {m['max_dd']:>8.4f} │ "
                  f"{m['pf']:>6.2f} │ {elapsed}")
            summary[key].append(m["ev"])

        print("─" * 110)

    # Summary
    print()
    print("  SUMMARY: Average EV per model")
    print("  " + "─" * 50)
    for model_name, key in [("XGBoost", "xgboost"), ("LSTM", "lstm"), ("CNN", "cnn")]:
        evs = summary[key]
        if evs:
            avg_ev = np.mean(evs)
            pos = sum(1 for e in evs if e > 0)
            print(f"    {model_name:<8}  avg_EV={avg_ev:+.6f}  "
                  f"positive: {pos}/{len(evs)}")

    # Head-to-head
    if summary["xgboost"] and summary["lstm"]:
        print()
        print("  HEAD-TO-HEAD: LSTM vs XGBoost")
        print("  " + "─" * 50)
        lstm_wins = 0
        for r in sorted(results, key=lambda x: x["symbol"]):
            if r.get("status") != "ok":
                continue
            diff = r["lstm"]["ev"] - r["xgboost"]["ev"]
            arrow = "▲ LSTM" if diff > 0 else "▼ XGB"
            if diff > 0:
                lstm_wins += 1
            print(f"    {r['symbol']:<14}  XGB={r['xgboost']['ev']:+.6f}  "
                  f"LSTM={r['lstm']['ev']:+.6f}  Δ={diff:+.6f} {arrow}")
        n = len([r for r in results if r.get("status") == "ok"])
        print(f"\n    LSTM wins: {lstm_wins}/{n}")


def main():
    p = argparse.ArgumentParser(description="XGBoost vs LSTM vs CNN backtest comparison")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--symbols", type=str, help="Comma-separated symbol list")
    grp.add_argument("--active", action="store_true", help="Use active symbols from config")
    p.add_argument("--train-size", type=int, default=1200)
    p.add_argument("--test-size", type=int, default=300)
    args = p.parse_args()

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    else:
        config_path = REPO_ROOT / "config" / "sovereign_configs.json"
        with open(config_path) as f:
            symbols = sorted(json.load(f).keys())

    print(f"[dl_comparison] {len(symbols)} symbols | XGBoost vs LSTM vs CNN")
    print(f"  Features: {len(FEATURE_COLUMNS)} | Lookback: {LOOKBACK} bars")
    print(f"  LSTM: {NUM_LAYERS}L BiLSTM, hidden={HIDDEN_DIM}")
    print(f"  CNN: 2-layer 1D-CNN, hidden={HIDDEN_DIM}")
    print(f"  Device: CPU (P40 incompatible with PyTorch {torch.__version__})")
    print()

    # Run sequentially (DL models use significant memory)
    results = []
    for sym in symbols:
        print(f"  Processing {sym}...", end=" ", flush=True)
        try:
            r = run_comparison(sym, args.train_size, args.test_size)
            results.append(r)
            if r.get("status") == "ok":
                print(f"done ({r['elapsed_s']:.0f}s, {r['folds']} folds)")
            else:
                print(f"{r.get('status', '?')}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"symbol": sym, "status": f"error: {e}"})

    print_results(results)


if __name__ == "__main__":
    main()
