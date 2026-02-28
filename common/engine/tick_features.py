"""
F2: Tick-level Features — intra-bar microstructure from tick data.

Aggregates raw tick data per bar to extract:
- tick_intensity: normalized ticks per minute
- bid_ask_imbalance: (bid_volume - ask_volume) / total
- tick_volatility: std of tick returns within bar
- large_tick_ratio: fraction of ticks > 2σ
- trade_flow_toxicity: VPIN-like proxy

All features are shift(1) safe (computed from the previous bar's ticks).
"""
from __future__ import annotations

import numpy as np
import polars as pl


TICK_FEATURE_COLUMNS = [
    "tick_intensity",
    "bid_ask_imbalance",
    "tick_volatility",
    "large_tick_ratio",
    "trade_flow_toxicity",
]


def build_tick_features(ticks: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """
    Add tick-level features to bar DataFrame.

    Parameters
    ----------
    ticks : DataFrame with columns [time, bid, ask, price, size]
    bars : DataFrame with columns [time, open, high, low, close, volume, ...]

    Returns
    -------
    bars DataFrame with 5 new tick feature columns appended (shift(1) applied).
    """
    if ticks is None or ticks.height == 0 or bars.height == 0:
        return _fill_defaults(bars)

    # Ensure time columns are datetime
    ticks_sorted = ticks.sort("time")
    bars_sorted = bars.sort("time")

    bar_times = bars_sorted["time"].to_list()
    if len(bar_times) < 2:
        return _fill_defaults(bars)

    # Compute features per bar window
    tick_intensity = []
    bid_ask_imbalance = []
    tick_volatility = []
    large_tick_ratio = []
    trade_flow_toxicity = []

    # Get tick data as numpy for speed
    tick_times = ticks_sorted["time"].to_numpy()
    tick_prices = ticks_sorted["price"].to_numpy().astype(np.float64)
    tick_bids = ticks_sorted["bid"].to_numpy().astype(np.float64) if "bid" in ticks_sorted.columns else None
    tick_asks = ticks_sorted["ask"].to_numpy().astype(np.float64) if "ask" in ticks_sorted.columns else None
    tick_sizes = ticks_sorted["size"].to_numpy().astype(np.float64) if "size" in ticks_sorted.columns else None

    for i in range(len(bar_times)):
        if i == 0:
            # First bar: no previous bar to compute from
            tick_intensity.append(0.0)
            bid_ask_imbalance.append(0.0)
            tick_volatility.append(0.0)
            large_tick_ratio.append(0.0)
            trade_flow_toxicity.append(0.0)
            continue

        # Window: ticks from previous bar start to current bar start
        bar_start = bar_times[i - 1]
        bar_end = bar_times[i]

        # Convert to naive datetime to avoid np.datetime64 timezone warnings
        bs = bar_start.replace(tzinfo=None) if hasattr(bar_start, 'replace') else bar_start
        be = bar_end.replace(tzinfo=None) if hasattr(bar_end, 'replace') else bar_end
        mask = (tick_times >= np.datetime64(bs)) & (tick_times < np.datetime64(be))
        idx = np.where(mask)[0]

        if len(idx) < 2:
            tick_intensity.append(0.0)
            bid_ask_imbalance.append(0.0)
            tick_volatility.append(0.0)
            large_tick_ratio.append(0.0)
            trade_flow_toxicity.append(0.0)
            continue

        prices = tick_prices[idx]
        n_ticks = len(prices)

        # Duration in minutes
        dt_seconds = (np.datetime64(be) - np.datetime64(bs)) / np.timedelta64(1, 's')
        dt_minutes = max(dt_seconds / 60.0, 1.0)

        # 1. Tick intensity: ticks per minute (normalized)
        intensity = n_ticks / dt_minutes
        tick_intensity.append(float(intensity))

        # 2. Bid-ask imbalance
        if tick_bids is not None and tick_asks is not None:
            bids = tick_bids[idx]
            asks = tick_asks[idx]
            mids = (bids + asks) / 2.0
            # Classify as buy (price >= mid) or sell (price < mid)
            buy_vol = np.sum(prices >= mids)
            sell_vol = np.sum(prices < mids)
            total = buy_vol + sell_vol
            imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0
            bid_ask_imbalance.append(float(imbalance))
        else:
            bid_ask_imbalance.append(0.0)

        # 3. Tick volatility: std of tick returns
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]
        vol = float(np.std(returns)) if len(returns) > 1 else 0.0
        tick_volatility.append(vol)

        # 4. Large tick ratio: fraction of returns > 2σ
        if len(returns) > 5 and vol > 1e-10:
            large = np.sum(np.abs(returns) > 2.0 * vol)
            large_tick_ratio.append(float(large / len(returns)))
        else:
            large_tick_ratio.append(0.0)

        # 5. Trade flow toxicity (VPIN proxy)
        # Split ticks into buckets, measure buy/sell imbalance across buckets
        if tick_sizes is not None and len(idx) > 10:
            sizes = tick_sizes[idx]
            total_volume = np.sum(sizes)
            if total_volume > 0:
                n_buckets = min(10, n_ticks // 5)
                if n_buckets >= 2:
                    bucket_size = n_ticks // n_buckets
                    imbalances = []
                    for b in range(n_buckets):
                        start = b * bucket_size
                        end = start + bucket_size
                        b_prices = prices[start:end]
                        b_sizes = sizes[start:end]
                        if len(b_prices) > 1:
                            b_rets = np.diff(b_prices)
                            buy_v = np.sum(b_sizes[1:][b_rets > 0])
                            sell_v = np.sum(b_sizes[1:][b_rets <= 0])
                            imbalances.append(abs(buy_v - sell_v))
                    toxicity = sum(imbalances) / total_volume if imbalances else 0.0
                    trade_flow_toxicity.append(float(min(toxicity, 1.0)))
                else:
                    trade_flow_toxicity.append(0.0)
            else:
                trade_flow_toxicity.append(0.0)
        else:
            trade_flow_toxicity.append(0.0)

    # Add features to bars with shift(1) — already computed from previous bar's ticks,
    # but add explicit shift(1) for consistency with the feature pipeline
    result = bars_sorted.with_columns([
        pl.Series("tick_intensity_raw", tick_intensity),
        pl.Series("bid_ask_imbalance_raw", bid_ask_imbalance),
        pl.Series("tick_volatility_raw", tick_volatility),
        pl.Series("large_tick_ratio_raw", large_tick_ratio),
        pl.Series("trade_flow_toxicity_raw", trade_flow_toxicity),
    ]).with_columns([
        pl.col("tick_intensity_raw").shift(1).fill_null(0.0).alias("tick_intensity"),
        pl.col("bid_ask_imbalance_raw").shift(1).fill_null(0.0).alias("bid_ask_imbalance"),
        pl.col("tick_volatility_raw").shift(1).fill_null(0.0).alias("tick_volatility"),
        pl.col("large_tick_ratio_raw").shift(1).fill_null(0.0).alias("large_tick_ratio"),
        pl.col("trade_flow_toxicity_raw").shift(1).fill_null(0.0).alias("trade_flow_toxicity"),
    ]).drop([
        "tick_intensity_raw", "bid_ask_imbalance_raw", "tick_volatility_raw",
        "large_tick_ratio_raw", "trade_flow_toxicity_raw",
    ])

    return result


def _fill_defaults(bars: pl.DataFrame) -> pl.DataFrame:
    """Fill tick features with zeros when tick data is unavailable."""
    return bars.with_columns([
        pl.lit(0.0).alias(col) for col in TICK_FEATURE_COLUMNS
    ])
