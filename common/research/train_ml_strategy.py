#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from engine.feature_builder import FEATURE_COLUMNS, build_bar_features
from engine.labeling import apply_triple_barrier


def parse_args():
    p = argparse.ArgumentParser(
        description="Polars-based leak-safe WFO training with dynamic triple-barrier + meta-labeling."
    )
    p.add_argument(
        "--data-roots",
        default="/home/tradebot/ssd_data_1/tick_data,/home/tradebot/ssd_data_2/tick_data,/home/tradebot/data_1/tick_data,/home/tradebot/data_2/tick_data,/home/tradebot/data_3/tick_data",
    )
    p.add_argument("--years", default="", help="Comma separated filter, e.g. 2018,2019,2020")
    p.add_argument("--symbols", default="", help="Optional comma-separated symbols filter")
    p.add_argument("--symbols-file", default="", help="Optional file with one symbol per line")
    p.add_argument(
        "--timeframes",
        default="M1,M5,M15,M30,H1",
        help="Comma-separated timeframes to test (up to H1), e.g. M1,M5,M15,M30,H1",
    )
    p.add_argument("--target-bars-per-day", type=int, default=96)
    p.add_argument("--z-threshold", type=float, default=1.5)
    p.add_argument("--pt-mult", type=float, default=2.0)
    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--horizon-bars", type=int, default=24)
    p.add_argument("--gap-bars", type=int, default=24, help="Embargo gap bars between train/test")
    p.add_argument("--wf-train-ratio", type=float, default=0.75, help="Walk-forward train window ratio")
    p.add_argument("--wf-step-ratio", type=float, default=0.25, help="Walk-forward test/step ratio")
    p.add_argument("--fee-bps", type=float, default=3.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--reg-alpha", type=float, default=0.05, help="L1 regularization for XGBoost")
    p.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization for XGBoost")
    p.add_argument("--symbol-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--xgb-jobs", type=int, default=1)
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Prefer GPU (cuda) with automatic CPU fallback by default.",
    )
    p.add_argument("--meta-threshold", type=float, default=0.5)
    p.add_argument("--out-dir", default="ml_boxes")
    return p.parse_args()


def discover_symbols(data_roots):
    symbols = set()
    for root in data_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if os.path.isdir(os.path.join(root, name)):
                symbols.add(name)
    return sorted(symbols)


def symbol_files(symbol, data_roots):
    out = []
    for root in data_roots:
        d = os.path.join(root, symbol)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                out.append(os.path.join(d, f))
    return out


def load_symbol_ticks(symbol, data_roots, years_filter):
    frames = []
    for fp in symbol_files(symbol, data_roots):
        base = os.path.basename(fp).split(".")[0]
        try:
            y = int(base.split("-")[0])
        except Exception:
            y = None
        if years_filter and y not in years_filter:
            continue
        df = pl.read_parquet(fp).select(["time", "bid", "ask", "last", "volume", "volume_real"])
        if df.height == 0:
            continue
        frames.append(df)
    if not frames:
        return None
    d = pl.concat(frames, how="vertical").sort("time").with_columns(
        [
            pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
            pl.when(pl.col("last") > 0)
            .then(pl.col("last"))
            .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
            .alias("price"),
            pl.when(pl.col("volume_real") > 0).then(pl.col("volume_real")).otherwise(pl.col("volume")).alias("size"),
        ]
    ).drop_nulls(["time", "price", "size"])
    if d.select(pl.col("size").sum()).item() <= 0:
        d = d.with_columns(pl.lit(1.0).alias("size"))
    return d.select(["time", "bid", "ask", "price", "size"])


def timeframe_minutes(tf: str) -> int:
    tf = tf.upper().strip()
    if tf == "M1":
        return 1
    if tf == "M5":
        return 5
    if tf == "M15":
        return 15
    if tf == "M30":
        return 30
    if tf == "H1":
        return 60
    if tf == "H4":
        return 240
    raise ValueError(f"unsupported timeframe: {tf}")


def make_time_bars(ticks: pl.DataFrame, tf: str) -> pl.DataFrame:
    mins = timeframe_minutes(tf)
    every = f"{mins}m"
    bars = (
        ticks.sort("time")
        .group_by_dynamic("time", every=every, period=every, closed="left", label="right")
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("size").sum().alias("volume"),
            ]
        )
        .drop_nulls(["open", "high", "low", "close", "volume"])
        .sort("time")
    )
    return bars


def make_volume_bars(ticks: pl.DataFrame, target_bars_per_day: int) -> pl.DataFrame:
    d = ticks.with_columns(pl.col("time").dt.date().alias("date"))
    med_daily_vol = (
        d.group_by("date")
        .agg(pl.col("size").sum().alias("daily_size"))
        .select(pl.col("daily_size").median())
        .item()
    )
    vol_threshold = max(float(med_daily_vol) / float(target_bars_per_day), 1e-9)
    bars = (
        d.with_columns((pl.col("size").cum_sum() / vol_threshold).floor().cast(pl.Int64).alias("bar_id"))
        .group_by("bar_id")
        .agg(
            [
                pl.col("time").last().alias("time"),
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("size").sum().alias("volume"),
            ]
        )
        .sort("time")
    )
    return bars


def recency_weights(years: np.ndarray) -> np.ndarray:
    # Exponential decay: oldest ~= 0.1, newest ~= 1.0
    y = years.astype(float)
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if ymax <= ymin:
        return np.ones_like(y)
    frac = (y - ymin) / (ymax - ymin)
    return 0.1 * np.power(10.0, frac)


def infer_spread_bps(ticks: pl.DataFrame) -> float:
    d = ticks.with_columns(
        (((pl.col("ask") - pl.col("bid")) / pl.col("price")) * 1e4).alias("spread_bps")
    ).filter(pl.col("spread_bps").is_finite() & (pl.col("spread_bps") > 0))
    if d.height == 0:
        return 0.0
    return float(d.select(pl.col("spread_bps").median()).item())


def infer_slippage_bps(symbol: str) -> float:
    s = symbol.upper()
    crypto_prefixes = ("BTC", "ETH", "SOL", "ADA", "DOT", "DOGE", "LTC", "BCH", "XRP", "XMR", "DASH", "NEO")
    if s.endswith(".CASH"):
        return 4.0
    if "_" in s:
        return 3.0
    if s.startswith(crypto_prefixes) and s.endswith("USD"):
        return 8.0
    return 5.0


def sanitize_training_frame(feat: pl.DataFrame) -> pl.DataFrame:
    # Convert inf/-inf to null and clip extreme outliers before model fit.
    cols = FEATURE_COLUMNS + ["label", "tb_ret", "upside", "downside"]
    exprs = []
    for c in cols:
        exprs.append(
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        )
    feat = feat.with_columns(exprs)

    # Hard caps to prevent numeric explosions from bad ticks/symbol glitches.
    clip_map = {
        "ret1": (-1.0, 1.0),
        "ret3": (-1.0, 1.0),
        "ret12": (-1.0, 1.0),
        "vol20": (0.0, 1.0),
        "z20": (-15.0, 15.0),
        "range": (0.0, 1.0),
        "vchg1": (-10.0, 10.0),
        "vratio20": (0.0, 20.0),
        "hour_sin": (-1.0, 1.0),
        "hour_cos": (-1.0, 1.0),
        "primary_side": (-1.0, 1.0),
        "tb_ret": (-1.0, 1.0),
        "upside": (0.0, 1.0),
        "downside": (0.0, 1.0),
        # Volatility regime
        "parkinson_vol": (0.0, 1.0),
        "garman_klass": (0.0, 1.0),
        "atr_ratio": (0.0, 10.0),
        "vol_of_vol": (0.0, 1.0),
        # Momentum / trend
        "ret24": (-1.0, 1.0),
        "ret48": (-1.0, 1.0),
        "ma_cross": (0.5, 2.0),
        "adx_proxy": (0.0, 5.0),
        # Price structure
        "body_ratio": (0.0, 1.0),
        "upper_shadow": (0.0, 1.0),
        "lower_shadow": (0.0, 1.0),
        # Volume microstructure
        "vratio5": (0.0, 20.0),
        "volume_trend": (0.0, 10.0),
        "vwap_dev": (0.5, 2.0),
        # Calendar
        "dow_sin": (-1.0, 1.0),
        "dow_cos": (-1.0, 1.0),
        # Autocorrelation
        "autocorr5": (-1.0, 1.0),
    }
    feat = feat.with_columns(
        [pl.col(k).clip(v[0], v[1]).alias(k) for k, v in clip_map.items() if k in feat.columns]
    )
    feat = feat.drop_nulls(cols)
    return feat


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


def fit_xgb_with_fallback(model_params, x_train, y_train, sample_weight, device_mode):
    """
    Train XGBoost with GPU-first policy and CPU fallback.
    Returns (model, used_device).
    """
    devices = []
    if device_mode == "cuda":
        devices = ["cuda"]
    elif device_mode == "cpu":
        devices = ["cpu"]
    else:
        devices = ["cuda", "cpu"]

    last_error = None
    for dev in devices:
        try:
            params = dict(model_params)
            params["tree_method"] = "hist"
            params["device"] = dev
            model = XGBClassifier(**params)
            model.fit(x_train, y_train, sample_weight=sample_weight)
            return model, dev
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"xgboost_fit_failed: {last_error}")


def process_symbol(symbol, args_dict):
    args = argparse.Namespace(**args_dict)
    roots = [x for x in args.data_roots.split(",") if x]
    years_filter = {int(x.strip()) for x in args.years.split(",") if x.strip()} if args.years else set()
    ticks = load_symbol_ticks(symbol, roots, years_filter)
    if ticks is None or ticks.height == 0:
        return dict(symbol=symbol, status="no_data")

    spread_bps = infer_spread_bps(ticks)
    slippage_bps = infer_slippage_bps(symbol)
    timeframes = [x.strip().upper() for x in args.timeframes.split(",") if x.strip()]
    all_rows = []

    for tf in timeframes:
        bars = make_time_bars(ticks, tf)
        feat = build_bar_features(bars, args.z_threshold)

        tb = apply_triple_barrier(
            close=feat["close"].to_numpy(),
            vol_proxy=feat["vol20"].to_numpy(),
            side=feat["primary_side"].to_numpy(),
            horizon=args.horizon_bars,
            pt_mult=args.pt_mult,
            sl_mult=args.sl_mult,
        )

        feat = feat.with_columns(
            [
                pl.Series("label", tb.label),
                pl.Series("tb_ret", tb.tb_ret),
                pl.Series("exit_idx", tb.exit_idx),
                pl.Series("upside", tb.upside),
                pl.Series("downside", tb.downside),
                pl.col("time").dt.year().alias("year"),
            ]
        )
        feat = feat.filter(pl.col("label").is_finite())
        feat = sanitize_training_frame(feat)
        if feat.height == 0:
            continue

        X = feat.select(FEATURE_COLUMNS).to_numpy()
        y = feat["label"].cast(pl.Int8).to_numpy()
        gross = feat["tb_ret"].to_numpy()
        yr = feat["year"].to_numpy()
        upside = feat["upside"].to_numpy()
        downside = feat["downside"].to_numpy()
        w = recency_weights(yr)
        cost = (args.fee_bps + spread_bps + slippage_bps) / 1e4

        n = len(feat)
        train_size = max(300, int(n * args.wf_train_ratio))
        test_size = max(120, int(n * args.wf_step_ratio) - args.gap_bars)
        if train_size + args.gap_bars + test_size > n:
            continue

        fold_id = 0
        start = 0
        tf_rows = []
        while True:
            tr_start = start
            tr_end = tr_start + train_size
            te_start = tr_end + args.gap_bars
            te_end = te_start + test_size
            if te_end > n:
                break

            tr_idx = np.arange(tr_start, tr_end)
            te_idx = np.arange(te_start, te_end)
            if tr_idx.size < 100 or te_idx.size < 50:
                break

            ytr = y[tr_idx]
            fold_id += 1
            uniq = np.unique(ytr)
            if uniq.size < 2:
                p_primary_te = np.full(te_idx.size, float(uniq[0]), dtype=float)
                p_meta_te = np.zeros(te_idx.size, dtype=float)
                used_device = "na"
            else:
                pos = max(int(ytr.sum()), 1)
                neg = max(int(len(ytr) - ytr.sum()), 1)
                primary_params = dict(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    reg_alpha=args.reg_alpha,
                    reg_lambda=args.reg_lambda,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=args.xgb_jobs,
                    random_state=42,
                    scale_pos_weight=neg / pos,
                )
                primary, used_device = fit_xgb_with_fallback(
                    primary_params, X[tr_idx], ytr, w[tr_idx], args.device
                )
                p_primary_tr = primary.predict_proba(X[tr_idx])[:, 1]
                p_primary_te = primary.predict_proba(X[te_idx])[:, 1]

                # Meta-label: did primary prediction match realized direction?
                y_meta = ((p_primary_tr >= 0.5).astype(np.int8) == ytr.astype(np.int8)).astype(np.int8)
                meta_pos = max(int(y_meta.sum()), 1)
                meta_neg = max(int(len(y_meta) - y_meta.sum()), 1)
                x_meta_tr = np.column_stack([X[tr_idx], p_primary_tr])
                x_meta_te = np.column_stack([X[te_idx], p_primary_te])

                meta_params = dict(
                    n_estimators=max(150, args.n_estimators // 2),
                    max_depth=max(3, args.max_depth - 1),
                    learning_rate=max(0.02, args.learning_rate),
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    reg_alpha=args.reg_alpha,
                    reg_lambda=args.reg_lambda,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=args.xgb_jobs,
                    random_state=43,
                    scale_pos_weight=meta_neg / meta_pos,
                )
                meta, _meta_device = fit_xgb_with_fallback(
                    meta_params, x_meta_tr, y_meta, w[tr_idx], args.device
                )
                p_meta_te = meta.predict_proba(x_meta_te)[:, 1]

            ev = p_primary_te * upside[te_idx] - (1.0 - p_primary_te) * downside[te_idx] - cost
            take = (ev > 0.0) & (p_meta_te >= args.meta_threshold)
            net = np.where(take, gross[te_idx] - cost, 0.0)
            stats = calc_metrics(net[take])
            te_years = yr[te_idx]
            year_label = int(np.max(te_years)) if te_years.size else -1
            tf_rows.append(
                dict(
                    symbol=symbol,
                    timeframe=tf,
                    year=year_label,
                    thread=int(fold_id),
                    train_years=f"{int(np.min(yr[tr_idx]))}-{int(np.max(yr[tr_idx]))}",
                    n_obs=int(te_idx.size),
                    n_trades=stats["n_trades"],
                    win_rate=stats["win_rate"],
                    sharpe=stats["sharpe"],
                    sortino=stats["sortino"],
                    max_drawdown_pct=stats["max_drawdown_pct"],
                    profit_factor=stats["profit_factor"],
                    device=used_device,
                    spread_bps=float(spread_bps),
                    slippage_bps=float(slippage_bps),
                    total_cost_bps=float(args.fee_bps + spread_bps + slippage_bps),
                )
            )
            start += test_size

        all_rows.extend(tf_rows)

    if not all_rows:
        return dict(symbol=symbol, status="no_folds")

    out_df = pl.from_dicts(all_rows).sort(["timeframe", "year", "thread"])
    box_dir = os.path.join(args.out_dir, symbol)
    os.makedirs(box_dir, exist_ok=True)
    out_parquet = os.path.join(box_dir, "wfo_metrics.parquet")
    out_csv = os.path.join(box_dir, "wfo_metrics.csv")
    out_df.write_parquet(out_parquet, compression="zstd")
    out_df.write_csv(out_csv)
    return dict(symbol=symbol, status="ok", folds=len(all_rows), parquet=out_parquet)


def run():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    roots = [x for x in args.data_roots.split(",") if x]
    symbols = discover_symbols(roots)

    if args.symbols:
        requested = {s.strip() for s in args.symbols.split(",") if s.strip()}
        symbols = [s for s in symbols if s in requested]

    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            requested = {line.strip() for line in f if line.strip()}
        symbols = [s for s in symbols if s in requested]

    if not symbols:
        raise SystemExit("No symbols found to process.")

    args_dict = vars(args).copy()
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.symbol_workers) as ex:
            futures = {ex.submit(process_symbol, s, args_dict): s for s in symbols}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = dict(symbol=s, status="error", error=str(e))
                print(r)
                results.append(r)
    except PermissionError:
        # Fallback for restricted environments where multiprocessing semaphores are blocked.
        # Use threads to keep symbol-level parallelism; XGBoost/Numpy run in native code.
        with ThreadPoolExecutor(max_workers=max(1, args.symbol_workers)) as ex:
            futures = {ex.submit(process_symbol, s, args_dict): s for s in symbols}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = dict(symbol=s, status="error", error=str(e))
                print(r)
                results.append(r)

    ok = [r for r in results if r.get("status") == "ok"]
    summary = pl.from_dicts(results)
    summary.write_parquet(os.path.join(args.out_dir, "summary.parquet"), compression="zstd")
    summary.write_csv(os.path.join(args.out_dir, "summary.csv"))
    print(f"Completed symbols: {len(ok)}/{len(results)}")


if __name__ == "__main__":
    run()
