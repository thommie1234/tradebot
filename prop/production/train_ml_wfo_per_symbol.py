#!/usr/bin/env python3
"""Per-symbol walk-forward ML training on MT5 tick parquet files."""

import argparse
import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-roots",
        default="/home/tradebot/ssd_data_1/tick_data,/home/tradebot/ssd_data_2/tick_data,/home/tradebot/data_1/tick_data,/home/tradebot/data_2/tick_data,/home/tradebot/data_3/tick_data",
    )
    p.add_argument("--years", default="", help="Comma separated filter, e.g. 2017,2018,2019")
    p.add_argument("--target-bars-per-day", type=int, default=96)
    p.add_argument("--z-threshold", type=float, default=1.5)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.0)
    p.add_argument("--horizon-bars", type=int, default=24)
    p.add_argument("--gap-bars", type=int, default=24)
    p.add_argument("--fee-bps", type=float, default=3.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--symbol-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--xgb-jobs", type=int, default=1)
    p.add_argument("--symbols", default="", help="Optional comma-separated symbol filter")
    p.add_argument("--symbols-file", default="", help="Optional file with one symbol per line")
    p.add_argument("--out-dir", default="ml_boxes")
    return p.parse_args()


def discover_symbols(data_roots):
    symbols = set()
    for root in data_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                symbols.add(name)
    return sorted(symbols)


def symbol_files(symbol, data_roots):
    files = []
    for root in data_roots:
        files.extend(glob.glob(os.path.join(root, symbol, "*.parquet")))
    return sorted(files)


def load_symbol_ticks(symbol, data_roots, years_filter):
    frames = []
    for fp in symbol_files(symbol, data_roots):
        base = os.path.basename(fp).split(".")[0]
        year = int(base.split("-")[0])
        if years_filter and year not in years_filter:
            continue
        df = pd.read_parquet(fp, columns=["time", "bid", "ask", "last", "volume", "volume_real"])
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.sort_values("time")
    d["price"] = np.where(d["last"] > 0, d["last"], (d["bid"] + d["ask"]) / 2.0)
    d["size"] = np.where(d["volume_real"] > 0, d["volume_real"], d["volume"])
    if float(pd.Series(d["size"]).fillna(0).sum()) <= 0:
        # Some symbols expose zero volume in ticks; fallback to tick-count bars.
        d["size"] = 1.0
    d = d.dropna(subset=["time", "price", "size"])
    return d[["time", "price", "size"]]


def make_volume_bars(ticks, target_bars_per_day):
    d = ticks.copy()
    d["date"] = d["time"].dt.date
    med_daily_vol = d.groupby("date")["size"].sum().median()
    vol_threshold = max(float(med_daily_vol) / float(target_bars_per_day), 1e-9)
    d["bar_id"] = np.floor(d["size"].cumsum() / vol_threshold).astype(np.int64)
    bars = d.groupby("bar_id", as_index=False).agg(
        time=("time", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
    )
    bars = bars.sort_values("time").reset_index(drop=True)
    return bars


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
    b["hour"] = b["time"].dt.hour
    b["hour_sin"] = np.sin(2 * np.pi * b["hour"] / 24.0)
    b["hour_cos"] = np.cos(2 * np.pi * b["hour"] / 24.0)
    b["primary_side"] = np.where(b["z20"] > z_threshold, -1, np.where(b["z20"] < -z_threshold, 1, 0))
    return b


def triple_barrier(close, vol, side, horizon, pt_mult, sl_mult):
    n = len(close)
    label = np.full(n, np.nan, dtype=float)
    tb_ret = np.full(n, np.nan, dtype=float)
    exit_idx = np.full(n, -1, dtype=int)
    idx = np.where((side != 0) & np.isfinite(vol))[0]
    for i in idx:
        end = min(i + horizon, n - 1)
        if end <= i:
            continue
        epx = close[i]
        pt = pt_mult * vol[i]
        sl = sl_mult * vol[i]
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
    return label, tb_ret, exit_idx


def recency_weights(years):
    y = np.asarray(years, dtype=float)
    ymin = y.min()
    ymax = y.max()
    if ymax <= ymin:
        return np.ones_like(y)
    frac = (y - ymin) / (ymax - ymin)
    # 10 years ago ~= 0.1, current ~= 1.0
    return 0.1 * np.power(10.0, frac)


def calc_metrics(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return dict(n_trades=0, win_rate=0.0, sharpe=0.0, sortino=0.0, max_drawdown_pct=0.0, profit_factor=0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float((r > 0).mean())
    m = r.mean()
    s = r.std(ddof=1) if r.size > 1 else 0.0
    n = r[r < 0]
    ds = n.std(ddof=1) if n.size > 1 else 0.0
    sharpe = 0.0 if s == 0 else math.sqrt(252.0) * m / s
    sortino = 0.0 if ds == 0 else math.sqrt(252.0) * m / ds
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


def process_symbol(symbol, args_dict):
    args = argparse.Namespace(**args_dict)
    roots = [x for x in args.data_roots.split(",") if x]
    years_filter = {int(x.strip()) for x in args.years.split(",") if x.strip()} if args.years else set()
    ticks = load_symbol_ticks(symbol, roots, years_filter)
    if ticks is None or ticks.empty:
        return dict(symbol=symbol, status="no_data")

    bars = make_volume_bars(ticks, args.target_bars_per_day)
    feat = build_features(bars, args.z_threshold)
    label, tb_ret, exit_idx = triple_barrier(
        feat["close"].to_numpy(),
        feat["vol20"].to_numpy(),
        feat["primary_side"].to_numpy(),
        args.horizon_bars,
        args.pt_mult,
        args.sl_mult,
    )
    feat["label"] = label
    feat["tb_ret"] = tb_ret
    feat["exit_idx"] = exit_idx
    feat = feat[np.isfinite(feat["label"])].copy()
    feat["year"] = feat["time"].dt.year.astype(int)
    feat = feat.dropna()

    features = ["ret1", "ret3", "ret12", "vol20", "z20", "range", "vchg1", "vratio20", "hour_sin", "hour_cos", "primary_side"]
    feat = feat.dropna(subset=features + ["label", "tb_ret"])
    if feat.empty:
        return dict(symbol=symbol, status="no_features")

    years = sorted(feat["year"].unique().tolist())
    if len(years) < 3:
        return dict(symbol=symbol, status="not_enough_years", years=len(years))

    X = feat[features].to_numpy()
    y = feat["label"].astype(int).to_numpy()
    vol = feat["vol20"].to_numpy()
    gross = feat["tb_ret"].to_numpy()
    yr = feat["year"].to_numpy()
    w = recency_weights(yr)
    cost = (args.fee_bps + args.slippage_bps) / 1e4

    rows = []
    fold_id = 0
    for i in range(2, len(years)):
        train_years = {years[i - 2], years[i - 1]}
        test_year = years[i]
        tr_idx = np.where(np.isin(yr, list(train_years)))[0]
        te_idx = np.where(yr == test_year)[0]
        if tr_idx.size == 0 or te_idx.size == 0:
            continue

        first_test = te_idx.min()
        tr_idx = tr_idx[tr_idx <= max(0, first_test - args.gap_bars)]
        if tr_idx.size == 0:
            continue

        ytr = y[tr_idx]
        fold_id += 1
        uniq = np.unique(ytr)
        if uniq.size < 2:
            p = np.full(te_idx.size, float(uniq[0]), dtype=float)
        else:
            pos = max(ytr.sum(), 1)
            neg = max(len(ytr) - ytr.sum(), 1)
            model = XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=args.xgb_jobs,
                random_state=42,
                scale_pos_weight=neg / pos,
            )
            model.fit(X[tr_idx], ytr, sample_weight=w[tr_idx])
            p_tr = model.predict_proba(X[tr_idx])[:, 1].reshape(-1, 1)
            p_te_raw = model.predict_proba(X[te_idx])[:, 1].reshape(-1, 1)
            cal = LogisticRegression(max_iter=300)
            cal.fit(p_tr, ytr, sample_weight=w[tr_idx])
            p = cal.predict_proba(p_te_raw)[:, 1]

        upside = args.pt_mult * vol[te_idx]
        downside = args.sl_mult * vol[te_idx]
        ev = p * upside - (1.0 - p) * downside - cost
        take = ev > 0.0
        net = np.where(take, gross[te_idx] - cost, 0.0)
        stats = calc_metrics(net[take])
        rows.append(
            dict(
                symbol=symbol,
                year=int(test_year),
                thread=int(fold_id),
                train_years="-".join(str(x) for x in sorted(train_years)),
                n_obs=int(te_idx.size),
                n_trades=stats["n_trades"],
                win_rate=stats["win_rate"],
                sharpe=stats["sharpe"],
                sortino=stats["sortino"],
                max_drawdown_pct=stats["max_drawdown_pct"],
                profit_factor=stats["profit_factor"],
            )
        )

    if not rows:
        return dict(symbol=symbol, status="no_folds")

    out_df = pd.DataFrame(rows).sort_values(["year", "thread"]).reset_index(drop=True)
    box_dir = os.path.join(args.out_dir, symbol)
    os.makedirs(box_dir, exist_ok=True)
    out_parquet = os.path.join(box_dir, "wfo_metrics.parquet")
    out_csv = os.path.join(box_dir, "wfo_metrics.csv")
    out_df.to_parquet(out_parquet, index=False, compression="zstd")
    out_df.to_csv(out_csv, index=False)
    return dict(symbol=symbol, status="ok", folds=len(rows), parquet=out_parquet)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data_roots = [x for x in args.data_roots.split(",") if x]
    symbols = discover_symbols(data_roots)
    if args.symbols:
        allowed = {x.strip() for x in args.symbols.split(",") if x.strip()}
        symbols = [s for s in symbols if s in allowed]
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            allowed_file = {ln.strip() for ln in f if ln.strip()}
        symbols = [s for s in symbols if s in allowed_file]
    if not symbols:
        raise SystemExit("No symbol folders found under data roots.")

    args_dict = vars(args).copy()
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.symbol_workers) as ex:
            fut = {ex.submit(process_symbol, sym, args_dict): sym for sym in symbols}
            for f in as_completed(fut):
                res = f.result()
                results.append(res)
                print(res)
    except PermissionError:
        # Restricted environments can block multiprocessing semaphores.
        with ThreadPoolExecutor(max_workers=max(1, args.symbol_workers)) as ex:
            fut = {ex.submit(process_symbol, sym, args_dict): sym for sym in symbols}
            for f in as_completed(fut):
                res = f.result()
                results.append(res)
                print(res)

    summary = pd.DataFrame(results)
    sum_parquet = os.path.join(args.out_dir, "summary.parquet")
    sum_csv = os.path.join(args.out_dir, "summary.csv")
    summary.to_parquet(sum_parquet, index=False, compression="zstd")
    summary.to_csv(sum_csv, index=False)
    print(f"Saved summary: {sum_parquet}")


if __name__ == "__main__":
    main()
