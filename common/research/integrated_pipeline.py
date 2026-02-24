from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl
import xgboost as xgb


def set_polars_threads(n_threads: int = 28) -> None:
    """
    Request a Polars thread budget for lazy execution.
    """
    os.environ["POLARS_MAX_THREADS"] = str(max(1, int(n_threads)))


def frac_diff_weights(d: float, window: int) -> np.ndarray:
    """
    Fixed-window fractional differentiation weights.
    """
    if d < 0.0 or d > 1.0:
        raise ValueError("d must be in [0, 1]")
    if window < 2:
        raise ValueError("window must be >= 2")

    w = np.zeros(window, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = -w[k - 1] * (d - (k - 1)) / float(k)
    return w


def frac_diff_expr(price_col: str, d: float, window: int, out_col: str = "frac_diff") -> pl.Expr:
    """
    Fractional differentiation expression using fixed window and shifted columns.
    """
    w = frac_diff_weights(d=d, window=window)
    expr = pl.lit(0.0)
    for lag in range(window):
        expr = expr + (pl.col(price_col).shift(lag) * float(w[lag]))
    return expr.alias(out_col)


def build_stationary_features_lf(
    lf: pl.LazyFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    d: float = 0.45,
    frac_window: int = 128,
    z_window: int = 40,
) -> pl.LazyFrame:
    """
    Build stationarity-aware features with strict shift(1) discipline.
    """
    if d < 0.3 or d > 0.6:
        raise ValueError("d must be in [0.3, 0.6]")

    two_pi = 2.0 * math.pi
    return (
        lf.sort("time")
        .with_columns(
            [
                frac_diff_expr(price_col=price_col, d=d, window=frac_window, out_col="fd_raw"),
                (pl.col(price_col) / pl.col(price_col).shift(1)).log().alias("ret1_raw"),
                (pl.col(price_col) / pl.col(price_col).shift(5) - 1.0).alias("ret5_raw"),
                (pl.col(high_col) - pl.col(low_col)).alias("hl_raw"),
                ((pl.col(high_col) - pl.col(low_col)) / pl.col(price_col)).alias("range_raw"),
                (pl.col(volume_col) / pl.col(volume_col).shift(1) - 1.0).alias("vchg1_raw"),
                pl.col("time").dt.hour().alias("hour"),
            ]
        )
        .with_columns(
            [
                pl.col("fd_raw").rolling_mean(z_window).alias("fd_ma_raw"),
                pl.col("fd_raw").rolling_std(z_window).alias("fd_sd_raw"),
                (pl.col("ret1_raw").rolling_std(z_window)).alias("vol_raw"),
                (pl.col("hour") * two_pi / 24.0).sin().alias("hour_sin_raw"),
                (pl.col("hour") * two_pi / 24.0).cos().alias("hour_cos_raw"),
            ]
        )
        .with_columns(
            [((pl.col("fd_raw") - pl.col("fd_ma_raw")) / pl.col("fd_sd_raw")).alias("fd_z_raw")]
        )
        .with_columns(
            [
                # Forced shift(1) discipline for every feature:
                pl.col("fd_raw").shift(1).alias("fd"),
                pl.col("fd_z_raw").shift(1).alias("fd_z"),
                pl.col("ret1_raw").shift(1).alias("ret1"),
                pl.col("ret5_raw").shift(1).alias("ret5"),
                pl.col("vol_raw").shift(1).alias("vol"),
                pl.col("hl_raw").shift(1).alias("hl"),
                pl.col("range_raw").shift(1).alias("range"),
                pl.col("vchg1_raw").shift(1).alias("vchg1"),
                pl.col("hour_sin_raw").shift(1).alias("hour_sin"),
                pl.col("hour_cos_raw").shift(1).alias("hour_cos"),
            ]
        )
    )


@dataclass
class LeakageAuditReport:
    summary: pl.DataFrame
    suspicious_count: int
    has_hard_violation: bool


def audit_lookahead_bias(
    features_lf: pl.LazyFrame,
    feature_cols: list[str],
    target_col: str = "target",
    corr_warn_threshold: float = 0.35,
    corr_margin: float = 0.08,
) -> LeakageAuditReport:
    """
    Inquisitor audit:
    - Checks suspicious contemporaneous target-feature correlation.
    - Compares corr(feature_t, target_t) vs corr(feature_t-1, target_t).
    """
    rows: list[dict] = []
    hard_violation = False
    for f in feature_cols:
        c = features_lf.select(
            [
                pl.corr(pl.col(f), pl.col(target_col)).alias("corr_t"),
                pl.corr(pl.col(f).shift(1), pl.col(target_col)).alias("corr_t_minus_1"),
                pl.corr(pl.col(f), pl.col(target_col).shift(-1)).alias("corr_with_future_target"),
            ]
        ).collect()
        corr_t = float(c["corr_t"][0]) if c["corr_t"][0] is not None else 0.0
        corr_tm1 = float(c["corr_t_minus_1"][0]) if c["corr_t_minus_1"][0] is not None else 0.0
        corr_future = (
            float(c["corr_with_future_target"][0]) if c["corr_with_future_target"][0] is not None else 0.0
        )

        suspicious = (abs(corr_t) > corr_warn_threshold) and (abs(corr_t) > abs(corr_tm1) + corr_margin)
        future_suspicious = abs(corr_future) > abs(corr_tm1) + corr_margin
        if future_suspicious:
            hard_violation = True

        rows.append(
            {
                "feature": f,
                "corr_t": corr_t,
                "corr_t_minus_1": corr_tm1,
                "corr_with_future_target": corr_future,
                "suspicious": suspicious,
                "future_suspicious": future_suspicious,
            }
        )

    summary = pl.from_dicts(rows).sort("feature")
    suspicious_count = int(summary.filter(pl.col("suspicious") | pl.col("future_suspicious")).height)
    return LeakageAuditReport(
        summary=summary,
        suspicious_count=suspicious_count,
        has_hard_violation=hard_violation,
    )


def purged_walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    purge: int,
    embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Purged Walk-Forward CV:
    train excludes [test_start-purge, test_end+embargo] neighborhood.
    """
    if n_samples <= train_size + test_size:
        return []
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while True:
        tr_start = start
        tr_end = tr_start + train_size
        te_start = tr_end
        te_end = te_start + test_size
        if te_end > n_samples:
            break

        train_idx = np.arange(tr_start, tr_end)
        test_idx = np.arange(te_start, te_end)

        left_cut = max(tr_start, te_start - max(0, purge))
        right_cut = min(tr_end, te_end + max(0, embargo))
        keep = (train_idx < left_cut) | (train_idx >= right_cut)
        purged_train = train_idx[keep]
        if purged_train.size > 0 and test_idx.size > 0:
            splits.append((purged_train, test_idx))
        start += test_size
    return splits


def tesla_p40_xgb_params() -> dict:
    """
    GPU-first dictionary using XGBoost 3.x syntax (device='cuda').
    Optimized for RTX 2060 / Tesla P40.
    """
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cuda",
        "sampling_method": "gradient_based",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 8,
        "min_child_weight": 8.0,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.08,
        "reg_lambda": 2.5,
        "gamma": 0.0,
        "max_bin": 512,
        "grow_policy": "lossguide",
        "verbosity": 1,
    }


def make_ev_custom_objective(
    avg_win: float,
    avg_loss: float,
    costs: float,
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[np.ndarray, np.ndarray]]:
    """
    EV-aligned custom objective (weighted logistic surrogate).
    """
    pos_w = max(float(avg_win), 1e-9)
    neg_w = max(float(avg_loss) + float(costs), 1e-9)

    def _obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-predt))
        w = np.where(y > 0.5, pos_w, neg_w)
        grad = (p - y) * w
        hess = np.maximum(p * (1.0 - p) * w, 1e-12)
        return grad, hess

    return _obj


def expected_value_score(
    proba: np.ndarray,
    avg_win: np.ndarray,
    avg_loss: np.ndarray,
    costs: np.ndarray,
) -> float:
    return float(np.mean(proba * avg_win - (1.0 - proba) * avg_loss - costs))


def make_optuna_objective(
    features_lf: pl.LazyFrame,
    feature_cols: list[str],
    target_col: str,
    avg_win_col: str,
    avg_loss_col: str,
    base_params: dict,
    train_size: int,
    test_size: int,
    purge: int,
    embargo: int,
    fee_bps_col: str = "fee_bps",
    spread_bps_col: str = "spread_bps",
    slippage_bps_col: str = "slippage_bps",
    slippage_multiplier: float = 2.0,
) -> Callable:
    """
    Optuna objective:
    tunes reg_alpha, reg_lambda, min_child_weight with Purged WFCV.
    """
    if abs(slippage_multiplier - 2.0) > 1e-12:
        raise ValueError("This pipeline enforces hard 2.0x slippage multiplier.")

    cols = feature_cols + [target_col, avg_win_col, avg_loss_col, fee_bps_col, spread_bps_col, slippage_bps_col]
    df = features_lf.select(cols).drop_nulls(cols).collect(streaming=True)
    x = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy().astype(np.float32)
    avg_win = df[avg_win_col].to_numpy().astype(np.float64)
    avg_loss = df[avg_loss_col].to_numpy().astype(np.float64)
    costs = (
        (df[fee_bps_col] + df[spread_bps_col] + (df[slippage_bps_col] * slippage_multiplier)).to_numpy()
        / 1e4
    ).astype(np.float64)

    splits = purged_walk_forward_splits(
        n_samples=len(df),
        train_size=train_size,
        test_size=test_size,
        purge=purge,
        embargo=embargo,
    )
    if not splits:
        raise ValueError("No purged folds available for the chosen train/test sizes.")

    def objective(trial) -> float:
        params = dict(base_params)
        # Regularisation hyperparameters
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True)
        params["min_child_weight"] = trial.suggest_float("min_child_weight", 0.5, 40.0, log=True)
        # Tree structure — broad, shallow trees to prevent overfitting
        params["max_depth"] = trial.suggest_int("max_depth", 3, 7)
        params["gamma"] = trial.suggest_float("gamma", 0.01, 1.0, log=True)
        # Row/column sampling for diversity
        params["subsample"] = trial.suggest_float("subsample", 0.6, 0.9)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 0.9)

        max_rounds = trial.suggest_int("num_boost_round", 200, 1200)

        ev_scores: list[float] = []
        for tr_idx, te_idx in splits:
            dtrain = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
            dtest = xgb.DMatrix(x[te_idx], label=y[te_idx])

            fold_avg_win = float(np.mean(avg_win[tr_idx]))
            fold_avg_loss = float(np.mean(avg_loss[tr_idx]))
            fold_cost = float(np.mean(costs[tr_idx]))
            obj = make_ev_custom_objective(fold_avg_win, fold_avg_loss, fold_cost)

            bst = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=max_rounds,
                obj=obj,
                evals=[(dtest, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            p = bst.predict(dtest)
            ev_scores.append(
                expected_value_score(
                    proba=p,
                    avg_win=avg_win[te_idx],
                    avg_loss=avg_loss[te_idx],
                    costs=costs[te_idx],
                )
            )

        return float(np.mean(ev_scores))

    return objective
