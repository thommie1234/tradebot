#!/usr/bin/env python3
"""Leak-aware ML pipeline on Parquet ticks with triple-barrier + meta-labeling."""

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


YEAR_RE = re.compile(r"year=(\d{4})")


@dataclass
class TBResult:
    label: np.ndarray
    tb_ret: np.ndarray
    exit_idx: np.ndarray


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-glob", default="/home/tradebot/ticks_partitioned/year=*/month=*/*.parquet")
    p.add_argument("--years", default="", help="Comma-separated, e.g. 2021,2022,2023")
    p.add_argument("--max-files", type=int, default=300)
    p.add_argument("--max-rows", type=int, default=2_000_000)
    p.add_argument("--target-bars-per-day", type=int, default=96)
    p.add_argument("--z-threshold", type=float, default=1.5)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.0)
    p.add_argument("--horizon-bars", type=int, default=24)
    p.add_argument("--fee-bps", type=float, default=3.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--gap", type=int, default=24)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--out-dir", default="ml_outputs")
    return p.parse_args()


def path_year(path):
    m = YEAR_RE.search(path)
    return int(m.group(1)) if m else None


def load_ticks(args):
    paths = sorted(glob.glob(args.data_glob))
    if args.years:
        years = {int(x.strip()) for x in args.years.split(",") if x.strip()}
        paths = [p for p in paths if path_year(p) in years]
    paths = paths[: args.max_files]
    chunks = []
    rows = 0
    for p in paths:
        df = pd.read_parquet(p, columns=["timestamp", "price", "size"])
        if df.empty:
            continue
        chunks.append(df)
        rows += len(df)
        if rows >= args.max_rows:
            break
    if not chunks:
        raise RuntimeError("No data loaded. Check --data-glob/--years.")
    out = pd.concat(chunks, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out = out.sort_values("timestamp").dropna(subset=["price", "size"])
    return out


def make_volume_bars(df, target_bars_per_day):
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    med_daily_vol = d.groupby("date")["size"].sum().median()
    vol_threshold = max(float(med_daily_vol) / float(target_bars_per_day), 1e-9)
    bar_id = np.floor(d["size"].cumsum() / vol_threshold).astype(np.int64)
    d["bar_id"] = bar_id
    bars = d.groupby("bar_id", as_index=False).agg(
        timestamp=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
    )
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    return bars, vol_threshold


def build_features(bars, z_threshold):
    b = bars.copy()
    b["ret1"] = np.log(b["close"]).diff()
    b["ret3"] = b["close"].pct_change(3)
    b["ret12"] = b["close"].pct_change(12)
    b["vol20"] = b["ret1"].rolling(20).std()
    ma20 = b["close"].rolling(20).mean()
    sd20 = b["close"].rolling(20).std()
    b["z20"] = (b["close"] - ma20) / sd20
    b["range"] = (b["high"] - b["low"]) / b["close"]
    b["vchg1"] = b["volume"].pct_change()
    b["vma20"] = b["volume"].rolling(20).mean()
    b["vratio20"] = b["volume"] / b["vma20"]
    b["hour"] = b["timestamp"].dt.hour
    b["hour_sin"] = np.sin(2 * np.pi * b["hour"] / 24.0)
    b["hour_cos"] = np.cos(2 * np.pi * b["hour"] / 24.0)
    b["primary_side"] = np.where(b["z20"] > z_threshold, -1, np.where(b["z20"] < -z_threshold, 1, 0))
    return b


def triple_barrier(close, vol, side, horizon, pt_mult, sl_mult):
    n = len(close)
    label = np.full(n, np.nan, dtype=float)
    tb_ret = np.full(n, np.nan, dtype=float)
    exit_idx = np.full(n, -1, dtype=int)
    events = np.where((side != 0) & np.isfinite(vol))[0]
    for i in events:
        end = min(i + horizon, n - 1)
        if end <= i:
            continue
        pt = pt_mult * vol[i]
        sl = sl_mult * vol[i]
        epx = close[i]
        chosen = end
        ret = side[i] * ((close[end] / epx) - 1.0)
        lab = 1.0 if ret > 0 else 0.0
        for j in range(i + 1, end + 1):
            r = side[i] * ((close[j] / epx) - 1.0)
            if r >= pt:
                chosen = j
                ret = r
                lab = 1.0
                break
            if r <= -sl:
                chosen = j
                ret = r
                lab = 0.0
                break
        label[i] = lab
        tb_ret[i] = ret
        exit_idx[i] = chosen
    return TBResult(label=label, tb_ret=tb_ret, exit_idx=exit_idx)


def calc_metrics(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return dict(n_trades=0, win_rate=0.0, sharpe=0.0, sortino=0.0, max_drawdown_pct=0.0, profit_factor=0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float((r > 0).mean())
    mean = r.mean()
    std = r.std(ddof=1) if r.size > 1 else 0.0
    neg = r[r < 0]
    dstd = neg.std(ddof=1) if neg.size > 1 else 0.0
    sharpe = 0.0 if std == 0 else math.sqrt(252.0) * mean / std
    sortino = 0.0 if dstd == 0 else math.sqrt(252.0) * mean / dstd
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak == 0, 0.0, eq / peak - 1.0)
    mdd = float(dd.min()) if dd.size else 0.0
    gp = wins.sum() if wins.size else 0.0
    gl = abs(losses.sum()) if losses.size else 0.0
    pf = float(gp / gl) if gl > 0 else float("inf")
    return dict(
        n_trades=int(r.size),
        win_rate=win_rate,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown_pct=float(mdd),
        profit_factor=pf,
    )


def run_pipeline(args):
    ticks = load_ticks(args)
    bars, vol_threshold = make_volume_bars(ticks, args.target_bars_per_day)
    feat = build_features(bars, args.z_threshold)
    tb = triple_barrier(
        close=feat["close"].to_numpy(),
        vol=feat["vol20"].to_numpy(),
        side=feat["primary_side"].to_numpy(),
        horizon=args.horizon_bars,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
    )

    feat["label"] = tb.label
    feat["tb_ret"] = tb.tb_ret
    feat["exit_idx"] = tb.exit_idx
    feat = feat[np.isfinite(feat["label"])].copy()
    feat["exit_time"] = feat["exit_idx"].map(lambda i: bars.iloc[int(i)]["timestamp"] if i >= 0 else pd.NaT)
    feat["exit_year"] = pd.to_datetime(feat["exit_time"]).dt.year
    feat = feat.dropna()

    features = [
        "ret1",
        "ret3",
        "ret12",
        "vol20",
        "z20",
        "range",
        "vchg1",
        "vratio20",
        "hour_sin",
        "hour_cos",
        "primary_side",
    ]
    feat = feat.dropna(subset=features + ["label", "tb_ret"])
    X = feat[features].to_numpy()
    y = feat["label"].astype(int).to_numpy()
    v = feat["vol20"].to_numpy()
    gross = feat["tb_ret"].to_numpy()
    exit_year = feat["exit_year"].astype(int).to_numpy()

    tscv = TimeSeriesSplit(n_splits=args.splits, gap=args.gap)
    cost = (args.fee_bps + args.slippage_bps) / 1e4
    rows = []

    for split_id, (tr, te) in enumerate(tscv.split(X), start=1):
        ytr = y[tr]
        unique = np.unique(ytr)
        if unique.size < 2:
            p_te = np.full(len(te), float(unique[0]), dtype=float)
            upside = args.pt_mult * v[te]
            downside = args.sl_mult * v[te]
            ev = p_te * upside - (1.0 - p_te) * downside - cost
            take = ev > 0.0
            net = np.where(take, gross[te] - cost, 0.0)
            split_years = exit_year[te]
            for yr in sorted(np.unique(split_years)):
                m = split_years == yr
                stats = calc_metrics(net[m][take[m]])
                rows.append(
                    dict(
                        year=int(yr),
                        thread=split_id,
                        n_obs=int(m.sum()),
                        n_trades=stats["n_trades"],
                        win_rate=stats["win_rate"],
                        sharpe=stats["sharpe"],
                        sortino=stats["sortino"],
                        max_drawdown_pct=stats["max_drawdown_pct"],
                        profit_factor=stats["profit_factor"],
                    )
                )
            continue

        pos = max(ytr.sum(), 1)
        neg = max((len(ytr) - ytr.sum()), 1)
        spw = neg / pos
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=spw,
        )
        model.fit(X[tr], ytr)
        p_tr = model.predict_proba(X[tr])[:, 1].reshape(-1, 1)
        p_te_raw = model.predict_proba(X[te])[:, 1].reshape(-1, 1)
        calibrator = LogisticRegression(max_iter=200)
        calibrator.fit(p_tr, ytr)
        p_te = calibrator.predict_proba(p_te_raw)[:, 1]

        upside = args.pt_mult * v[te]
        downside = args.sl_mult * v[te]
        ev = p_te * upside - (1.0 - p_te) * downside - cost
        take = ev > 0.0
        net = np.where(take, gross[te] - cost, 0.0)

        split_years = exit_year[te]
        for yr in sorted(np.unique(split_years)):
            m = split_years == yr
            stats = calc_metrics(net[m][take[m]])
            rows.append(
                dict(
                    year=int(yr),
                    thread=split_id,
                    n_obs=int(m.sum()),
                    n_trades=stats["n_trades"],
                    win_rate=stats["win_rate"],
                    sharpe=stats["sharpe"],
                    sortino=stats["sortino"],
                    max_drawdown_pct=stats["max_drawdown_pct"],
                    profit_factor=stats["profit_factor"],
                )
            )

    out = pd.DataFrame(rows).sort_values(["year", "thread"]).reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_parquet = os.path.join(args.out_dir, "ml_year_thread_metrics.parquet")
    out_csv = os.path.join(args.out_dir, "ml_year_thread_metrics.csv")
    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)
    print(f"Loaded ticks: {len(ticks):,}")
    print(f"Built bars: {len(bars):,} (volume threshold={vol_threshold:.4f})")
    print(f"Labeled events: {len(feat):,}")
    print(f"Saved: {out_parquet}")
    print(f"Saved: {out_csv}")
    if not out.empty:
        print("\nHead:")
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    run_pipeline(parse_args())
