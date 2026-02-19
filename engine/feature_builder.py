"""
Feature builder — 39 leak-safe features using Polars with strict shift(1) discipline.

Original 28 + F4 regime (3) + F2 tick (5) + F8 lead-lag (3) = 39.
HTF expansion: +8 per higher timeframe (H4, D1) = +16 → 55 total (research only).
Copied from trading_prop/ml/features.py (canonical source).
"""
from __future__ import annotations

import math

import polars as pl


FEATURE_COLUMNS = [
    # --- Price returns (3) ---
    "ret1",
    "ret3",
    "ret12",
    # --- Volatility (1) ---
    "vol20",
    # --- Mean-reversion (1) ---
    "z20",
    # --- Range (1) ---
    "range",
    # --- Volume (2) ---
    "vchg1",
    "vratio20",
    # --- Time-of-day (2) ---
    "hour_sin",
    "hour_cos",
    # --- Directional signal (1) ---
    "primary_side",
    # --- Volatility Regime (4) ---
    "parkinson_vol",
    "garman_klass",
    "atr_ratio",
    "vol_of_vol",
    # --- Momentum / Trend (4) ---
    "ret24",
    "ret48",
    "ma_cross",
    "adx_proxy",
    # --- Price Structure (3) ---
    "body_ratio",
    "upper_shadow",
    "lower_shadow",
    # --- Volume Microstructure (3) ---
    "vratio5",
    "volume_trend",
    "vwap_dev",
    # --- Day-of-week (2) ---
    "dow_sin",
    "dow_cos",
    # --- Autocorrelation (1) ---
    "autocorr5",
    # --- F4: Regime Detection (3) ---
    "regime",
    "hurst",
    "regime_duration",
    # --- F2: Tick-level Features (5) ---
    "tick_intensity",
    "bid_ask_imbalance",
    "tick_volatility",
    "large_tick_ratio",
    "trade_flow_toxicity",
    # --- F8: Cross-Asset Lead-Lag (3) ---
    "leader_ret1",
    "leader_ret3",
    "leader_momentum",
]


def build_bar_features(
    bars: pl.DataFrame,
    z_threshold: float,
    vol_winsor_low: float = 0.01,
    vol_winsor_high: float = 0.99,
) -> pl.DataFrame:
    """
    Build leak-safe features with strict shift(1) discipline.
    All predictor features at t are computed only from <= t-1.
    """
    b = bars.sort("time")
    two_pi = 2.0 * math.pi
    _2ln2_minus1 = 2.0 * math.log(2) - 1.0
    _parkinson_denom = 2.0 * math.sqrt(math.log(2))
    hl_guard = pl.col("high") > pl.col("low")

    out = b.with_columns(
        [
            # ---- Existing raw features ----
            (pl.col("close") / pl.col("close").shift(1)).log().alias("ret1_raw"),
            (pl.col("close") / pl.col("close").shift(3) - 1.0).alias("ret3_raw"),
            (pl.col("close") / pl.col("close").shift(12) - 1.0).alias("ret12_raw"),
            pl.col("close").rolling_mean(20).alias("ma20_raw"),
            pl.col("close").rolling_std(20).alias("sd20_raw"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_raw"),
            (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("vchg1_raw"),
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("vratio20_raw"),
            pl.col("time").dt.hour().alias("hour"),
            # ---- Momentum: longer returns ----
            (pl.col("close") / pl.col("close").shift(24) - 1.0).alias("ret24_raw"),
            (pl.col("close") / pl.col("close").shift(48) - 1.0).alias("ret48_raw"),
            # ---- Trend: MA(5) for cross ----
            pl.col("close").rolling_mean(5).alias("ma5_raw"),
            # ---- Price structure (div-by-zero guarded) ----
            pl.when(hl_guard)
            .then((pl.col("close") - pl.col("open")).abs() / (pl.col("high") - pl.col("low")))
            .otherwise(0.0)
            .alias("body_ratio_raw"),
            pl.when(hl_guard)
            .then(
                (pl.col("high") - pl.max_horizontal("open", "close"))
                / (pl.col("high") - pl.col("low"))
            )
            .otherwise(0.0)
            .alias("upper_shadow_raw"),
            pl.when(hl_guard)
            .then(
                (pl.min_horizontal("open", "close") - pl.col("low"))
                / (pl.col("high") - pl.col("low"))
            )
            .otherwise(0.0)
            .alias("lower_shadow_raw"),
            # ---- Volume: short-term ratio ----
            pl.when(pl.col("volume").rolling_mean(5) > 0)
            .then(pl.col("volume") / pl.col("volume").rolling_mean(5))
            .otherwise(1.0)
            .alias("vratio5_raw"),
            # ---- Volume: trend components ----
            pl.col("volume").rolling_mean(5).alias("vol_ma5_raw"),
            pl.col("volume").rolling_mean(20).alias("vol_ma20_raw"),
            # ---- Typical price for VWAP deviation ----
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("tp_raw"),
            # ---- True range components ----
            (pl.col("high") - pl.col("low")).alias("hl_range"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("hpc"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("lpc"),
            # ---- Volatility estimator components ----
            ((pl.col("high") / pl.col("low")).log().pow(2)).alias("parkinson_sq"),
            (
                0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
                - _2ln2_minus1 * (pl.col("close") / pl.col("open")).log().pow(2)
            ).alias("gk_sq"),
            # ---- Directional movement components ----
            pl.max_horizontal(pl.col("high") - pl.col("high").shift(1), pl.lit(0.0)).alias(
                "plus_dm"
            ),
            pl.max_horizontal(pl.col("low").shift(1) - pl.col("low"), pl.lit(0.0)).alias(
                "minus_dm"
            ),
            # ---- Day-of-week ----
            pl.col("time").dt.weekday().alias("dow"),
        ]
    ).with_columns(
        # ---- STAGE 2: first intermediates ----
        [
            # Existing
            (pl.col("ret1_raw").rolling_std(20)).alias("vol20_raw"),
            ((pl.col("close") - pl.col("ma20_raw")) / pl.col("sd20_raw")).alias("z20_raw"),
            (pl.col("hour") * two_pi / 24.0).sin().alias("hour_sin_raw"),
            (pl.col("hour") * two_pi / 24.0).cos().alias("hour_cos_raw"),
            # True range
            pl.max_horizontal("hl_range", "hpc", "lpc").alias("true_range"),
            # Parkinson volatility
            (
                pl.col("parkinson_sq").rolling_mean(20).sqrt() / _parkinson_denom
            ).alias("parkinson_vol_raw"),
            # Garman-Klass volatility (guard negative rolling mean)
            pl.max_horizontal(pl.col("gk_sq").rolling_mean(20), pl.lit(0.0))
            .sqrt()
            .alias("garman_klass_raw"),
            # MA cross
            pl.when(pl.col("ma20_raw") > 0)
            .then(pl.col("ma5_raw") / pl.col("ma20_raw"))
            .otherwise(1.0)
            .alias("ma_cross_raw"),
            # Volume trend
            pl.when(pl.col("vol_ma20_raw") > 0)
            .then(pl.col("vol_ma5_raw") / pl.col("vol_ma20_raw"))
            .otherwise(1.0)
            .alias("volume_trend_raw"),
            # VWAP deviation (price vs rolling typical price; robust to zero-volume bars)
            pl.when(pl.col("tp_raw").rolling_mean(20) > 0)
            .then(pl.col("close") / pl.col("tp_raw").rolling_mean(20))
            .otherwise(1.0)
            .alias("vwap_dev_raw"),
            # Day-of-week cyclic
            (pl.col("dow").cast(pl.Float64) * two_pi / 5.0).sin().alias("dow_sin_raw"),
            (pl.col("dow").cast(pl.Float64) * two_pi / 5.0).cos().alias("dow_cos_raw"),
            # Directional movement difference
            (pl.col("plus_dm") - pl.col("minus_dm")).abs().alias("dm_diff"),
            # Autocorrelation: rolling corr of ret1 with 5-lag
            pl.rolling_corr(
                pl.col("ret1_raw"), pl.col("ret1_raw").shift(5), window_size=20
            ).alias("autocorr5_raw"),
        ]
    ).with_columns(
        # ---- STAGE 2b: second intermediates ----
        [
            pl.col("true_range").rolling_mean(5).alias("atr5_raw"),
            pl.col("true_range").rolling_mean(14).alias("atr14_raw"),
            pl.col("true_range").rolling_mean(20).alias("atr20_raw"),
            # Volatility of volatility
            pl.col("vol20_raw").rolling_std(20).alias("vol_of_vol_raw"),
            # ADX proxy
            pl.when(pl.col("true_range").rolling_mean(14) > 0)
            .then(
                pl.col("dm_diff").rolling_mean(14) / pl.col("true_range").rolling_mean(14)
            )
            .otherwise(0.0)
            .alias("adx_proxy_raw"),
        ]
    ).with_columns(
        # ---- STAGE 2c: third intermediates ----
        [
            pl.when(pl.col("atr20_raw") > 0)
            .then(pl.col("atr5_raw") / pl.col("atr20_raw"))
            .otherwise(1.0)
            .alias("atr_ratio_raw"),
        ]
    ).with_columns(
        # ---- STAGE 3: shift(1) for all features ----
        [
            pl.col("ret1_raw").shift(1).alias("ret1"),
            pl.col("ret3_raw").shift(1).alias("ret3"),
            pl.col("ret12_raw").shift(1).alias("ret12"),
            pl.col("vol20_raw").shift(1).alias("vol20"),
            pl.col("z20_raw").shift(1).alias("z20"),
            pl.col("range_raw").shift(1).alias("range"),
            pl.col("vchg1_raw").shift(1).alias("vchg1"),
            pl.col("vratio20_raw").shift(1).alias("vratio20"),
            pl.col("hour_sin_raw").shift(1).alias("hour_sin"),
            pl.col("hour_cos_raw").shift(1).alias("hour_cos"),
            # New features
            pl.col("parkinson_vol_raw").shift(1).alias("parkinson_vol"),
            pl.col("garman_klass_raw").shift(1).alias("garman_klass"),
            pl.col("atr_ratio_raw").shift(1).alias("atr_ratio"),
            pl.col("vol_of_vol_raw").shift(1).alias("vol_of_vol"),
            pl.col("ret24_raw").shift(1).alias("ret24"),
            pl.col("ret48_raw").shift(1).alias("ret48"),
            pl.col("ma_cross_raw").shift(1).alias("ma_cross"),
            pl.col("adx_proxy_raw").shift(1).alias("adx_proxy"),
            pl.col("body_ratio_raw").shift(1).alias("body_ratio"),
            pl.col("upper_shadow_raw").shift(1).alias("upper_shadow"),
            pl.col("lower_shadow_raw").shift(1).alias("lower_shadow"),
            pl.col("vratio5_raw").shift(1).alias("vratio5"),
            pl.col("volume_trend_raw").shift(1).alias("volume_trend"),
            pl.col("vwap_dev_raw").shift(1).alias("vwap_dev"),
            pl.col("dow_sin_raw").shift(1).alias("dow_sin"),
            pl.col("dow_cos_raw").shift(1).alias("dow_cos"),
            pl.col("autocorr5_raw").shift(1).alias("autocorr5"),
        ]
    ).with_columns(
        # ---- STAGE 4: post-shift derived ----
        [
            pl.when(pl.col("z20") > z_threshold)
            .then(-1)
            .when(pl.col("z20") < -z_threshold)
            .then(1)
            .otherwise(0)
            .alias("primary_side")
        ]
    )

    # ---- F4: Regime Detection (shift(1) applied inside detect()) ----
    try:
        import numpy as np
        from engine.regime_detector import RegimeDetector

        close_arr = out["close"].to_numpy().astype(np.float64)
        vol_arr = out["vol20"].fill_null(0.0).to_numpy().astype(np.float64)
        regimes, hurst_arr, duration_arr = RegimeDetector.detect(close_arr, vol_arr, window=20)

        # Apply shift(1) for leak safety
        out = out.with_columns([
            pl.Series("regime", regimes).shift(1).fill_null(0).alias("regime"),
            pl.Series("hurst", hurst_arr).shift(1).fill_null(0.5).alias("hurst"),
            pl.Series("regime_duration", duration_arr).shift(1).fill_null(1.0).alias("regime_duration"),
        ])
    except Exception:
        # Graceful fallback if regime detection fails
        out = out.with_columns([
            pl.lit(0).alias("regime"),
            pl.lit(0.5).alias("hurst"),
            pl.lit(1.0).alias("regime_duration"),
        ])

    # ---- F2: Tick features are added externally via build_tick_features() ----
    # Fill with defaults if not present (will be overwritten by signal.py when available)
    for col in ["tick_intensity", "bid_ask_imbalance", "tick_volatility",
                "large_tick_ratio", "trade_flow_toxicity"]:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(col))

    # ---- F8: Lead-lag features are added externally via build_lead_lag_features() ----
    for col in ["leader_ret1", "leader_ret3", "leader_momentum"]:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(col))

    # ---- Winsorize heavy-tailed features ----
    winsorize_cols = ["vol20", "parkinson_vol", "garman_klass", "vol_of_vol"]
    q = out.select(
        [
            expr
            for c in winsorize_cols
            for expr in [
                pl.col(c).drop_nulls().quantile(vol_winsor_low).alias(f"{c}_qlo"),
                pl.col(c).drop_nulls().quantile(vol_winsor_high).alias(f"{c}_qhi"),
            ]
        ]
    )
    clip_exprs = []
    for c in winsorize_cols:
        lo = q[f"{c}_qlo"][0]
        hi = q[f"{c}_qhi"][0]
        if lo is not None and hi is not None:
            clip_exprs.append(pl.col(c).clip(lo, hi).alias(c))
    # Hard-clip autocorrelation to valid range
    clip_exprs.append(pl.col("autocorr5").clip(-1.0, 1.0).alias("autocorr5"))
    if clip_exprs:
        out = out.with_columns(clip_exprs)

    return out.select(
        ["time", "open", "high", "low", "close", "volume"] + FEATURE_COLUMNS
    )


# ---------------------------------------------------------------------------
# HTF (Higher-Timeframe) feature expansion — research only, not used in live
# ---------------------------------------------------------------------------

_HTF_NAMES = ["ret1", "ret3", "ma_cross", "vol20", "atr_ratio", "z20", "adx_proxy", "regime"]


def normalize_bar_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize bar data: tick_volume→volume, int64 time→datetime UTC."""
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename({"tick_volume": "volume"})
    elif "volume" in df.columns and "tick_volume" in df.columns:
        if df.select(pl.col("volume").sum()).item() == 0:
            df = df.drop("volume").rename({"tick_volume": "volume"})
    if df["time"].dtype in (pl.Int64, pl.UInt64):
        df = df.with_columns(
            (pl.col("time") * 1_000_000).cast(pl.Datetime("us", "UTC")).alias("time")
        )
    cols = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.select(cols)


def htf_feature_columns(prefixes: list[str] | None = None) -> list[str]:
    """Return HTF feature column names for given prefixes (default: h4, d1)."""
    prefixes = prefixes or ["h4", "d1"]
    return [f"{p}_{name}" for p in prefixes for name in _HTF_NAMES]


def build_htf_features(bars: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Build 8 summary features from a higher-timeframe bar series.

    All features are shift(1) safe: they use only completed bars.
    Returns DataFrame with columns: time, {prefix}_ret1, ..., {prefix}_regime.
    """
    import numpy as np

    b = bars.sort("time")
    two_pi = 2.0 * math.pi
    _2ln2_minus1 = 2.0 * math.log(2) - 1.0

    out = b.with_columns([
        # ret1, ret3
        (pl.col("close") / pl.col("close").shift(1)).log().alias("_ret1_raw"),
        (pl.col("close") / pl.col("close").shift(3) - 1.0).alias("_ret3_raw"),
        # MA cross (MA5 / MA20)
        pl.col("close").rolling_mean(5).alias("_ma5"),
        pl.col("close").rolling_mean(20).alias("_ma20"),
        # vol20
        (pl.col("close") / pl.col("close").shift(1)).log().rolling_std(20).alias("_vol20_raw"),
        # z20
        pl.col("close").rolling_mean(20).alias("_mean20"),
        pl.col("close").rolling_std(20).alias("_std20"),
        # ATR components
        (pl.col("high") - pl.col("low")).alias("_hl"),
        (pl.col("high") - pl.col("close").shift(1)).abs().alias("_hpc"),
        (pl.col("low") - pl.col("close").shift(1)).abs().alias("_lpc"),
        # DM for ADX
        pl.max_horizontal(pl.col("high") - pl.col("high").shift(1), pl.lit(0.0)).alias("_plus_dm"),
        pl.max_horizontal(pl.col("low").shift(1) - pl.col("low"), pl.lit(0.0)).alias("_minus_dm"),
    ]).with_columns([
        # MA cross ratio
        pl.when(pl.col("_ma20") > 0)
        .then(pl.col("_ma5") / pl.col("_ma20"))
        .otherwise(1.0)
        .alias("_ma_cross_raw"),
        # z20
        pl.when(pl.col("_std20") > 0)
        .then((pl.col("close") - pl.col("_mean20")) / pl.col("_std20"))
        .otherwise(0.0)
        .alias("_z20_raw"),
        # True range
        pl.max_horizontal("_hl", "_hpc", "_lpc").alias("_tr"),
        # DM diff
        (pl.col("_plus_dm") - pl.col("_minus_dm")).abs().alias("_dm_diff"),
    ]).with_columns([
        # ATR5, ATR20
        pl.col("_tr").rolling_mean(5).alias("_atr5"),
        pl.col("_tr").rolling_mean(20).alias("_atr20"),
        # ADX proxy
        pl.when(pl.col("_tr").rolling_mean(14) > 0)
        .then(pl.col("_dm_diff").rolling_mean(14) / pl.col("_tr").rolling_mean(14))
        .otherwise(0.0)
        .alias("_adx_proxy_raw"),
    ]).with_columns([
        # ATR ratio
        pl.when(pl.col("_atr20") > 0)
        .then(pl.col("_atr5") / pl.col("_atr20"))
        .otherwise(1.0)
        .alias("_atr_ratio_raw"),
    ]).with_columns([
        # SHIFT(1) — all features use only completed bars
        pl.col("_ret1_raw").shift(1).alias(f"{prefix}_ret1"),
        pl.col("_ret3_raw").shift(1).alias(f"{prefix}_ret3"),
        pl.col("_ma_cross_raw").shift(1).alias(f"{prefix}_ma_cross"),
        pl.col("_vol20_raw").shift(1).alias(f"{prefix}_vol20"),
        pl.col("_atr_ratio_raw").shift(1).alias(f"{prefix}_atr_ratio"),
        pl.col("_z20_raw").shift(1).alias(f"{prefix}_z20"),
        pl.col("_adx_proxy_raw").shift(1).alias(f"{prefix}_adx_proxy"),
    ])

    # Regime detection (shift(1) applied after)
    try:
        from engine.regime_detector import RegimeDetector
        close_arr = out["close"].to_numpy().astype(np.float64)
        vol_arr = out[f"{prefix}_vol20"].fill_null(0.0).to_numpy().astype(np.float64)
        regimes, _, _ = RegimeDetector.detect(close_arr, vol_arr, window=20)
        out = out.with_columns(
            pl.Series(f"{prefix}_regime", regimes).shift(1).fill_null(0).cast(pl.Float64)
        )
    except Exception:
        out = out.with_columns(pl.lit(0.0).alias(f"{prefix}_regime"))

    htf_cols = [f"{prefix}_{n}" for n in _HTF_NAMES]
    return out.select(["time"] + htf_cols)


def merge_htf_features(
    base: pl.DataFrame,
    htf: pl.DataFrame,
    prefix: str,
) -> pl.DataFrame:
    """Merge HTF features into base DataFrame via backward asof join on time.

    For each base row, finds the most recent HTF bar that closed BEFORE it.
    This is inherently leak-safe: only uses already-closed HTF bars.
    Missing values (before first HTF bar) are filled with 0.0.
    """
    htf_cols = [f"{prefix}_{n}" for n in _HTF_NAMES]

    # Ensure both have proper datetime types for asof join
    base_time = base["time"].dtype
    htf_time = htf["time"].dtype

    # Cast to same type if needed
    if base_time != htf_time:
        htf = htf.with_columns(pl.col("time").cast(base_time))

    # Sort both (required for asof join)
    base_sorted = base.sort("time")
    htf_sorted = htf.select(["time"] + htf_cols).sort("time")

    # Asof join: for each base row, get most recent HTF row (backward)
    joined = base_sorted.join_asof(
        htf_sorted,
        on="time",
        strategy="backward",
    )

    # Fill nulls (rows before first HTF bar)
    fill_exprs = [pl.col(c).fill_null(0.0).alias(c) for c in htf_cols]
    return joined.with_columns(fill_exprs)
