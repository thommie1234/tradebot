"""
Vectorized trade lifecycle simulator for exit-parameter optimization.

Pure numpy — no Optuna or XGBoost dependency.
Simulates SL / TP / breakeven / trailing-stop exits bar-by-bar.

Usage (standalone smoke test):
    python3 research/exit_simulator.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExitParams:
    """Six tunable exit parameters (all ATR-relative except horizon)."""
    atr_sl_mult: float      # SL distance in ATR multiples
    atr_tp_mult: float      # TP distance in ATR multiples
    breakeven_atr: float    # profit threshold to move SL to entry (ATR mult)
    trail_activation_atr: float  # profit threshold to start trailing (ATR mult)
    trail_distance_atr: float    # trailing distance behind best price (ATR mult)
    horizon: int            # max bars to hold before forced exit


# Exit type codes
EXIT_SL = 0
EXIT_TP = 1
EXIT_BREAKEVEN = 2   # hit SL after it was moved to breakeven
EXIT_TRAIL = 3       # hit trailing stop
EXIT_HORIZON = 4     # max bars reached


def simulate_trades(
    entry_indices: np.ndarray,
    directions: np.ndarray,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    params: ExitParams,
    cost_pct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate trades bar-by-bar with SL/TP/breakeven/trailing exits.

    Parameters
    ----------
    entry_indices : int array — bar indices where entries occur
    directions : int array — +1 for BUY, -1 for SELL (same length)
    open/high/low/close_arr : float arrays — full OHLC series
    atr_arr : float array — ATR(14) per bar
    params : ExitParams — the 6 exit parameters to test
    cost_pct : float — round-trip cost as fraction (e.g. 0.001 = 10 bps)

    Returns
    -------
    pnl : float array — net PnL per trade (fraction of entry price)
    bars_held : int array — bars each trade was held
    exit_types : int array — exit type code per trade
    """
    n_bars = len(close_arr)
    n_entries = len(entry_indices)

    pnl = np.empty(n_entries, dtype=np.float64)
    bars_held = np.empty(n_entries, dtype=np.int32)
    exit_types = np.empty(n_entries, dtype=np.int32)

    trade_count = 0
    position_end = -1  # bar index where current position closes

    for i in range(n_entries):
        idx = int(entry_indices[i])
        dirn = int(directions[i])

        # Skip if we're still in a position (no overlapping trades)
        if idx <= position_end:
            continue

        # Entry at next bar's open (avoid lookahead)
        entry_bar = idx + 1
        if entry_bar >= n_bars:
            continue

        entry_px = open_arr[entry_bar]
        if entry_px <= 0 or not np.isfinite(entry_px):
            continue

        atr = atr_arr[idx]
        if atr <= 0 or not np.isfinite(atr):
            continue

        # Compute absolute distances
        sl_dist = params.atr_sl_mult * atr
        tp_dist = params.atr_tp_mult * atr
        be_threshold = params.breakeven_atr * atr
        trail_threshold = params.trail_activation_atr * atr
        trail_dist = params.trail_distance_atr * atr

        if dirn == 1:  # BUY
            sl_price = entry_px - sl_dist
            tp_price = entry_px + tp_dist
        else:  # SELL
            sl_price = entry_px + sl_dist
            tp_price = entry_px - tp_dist

        breakeven_active = False
        trailing_active = False
        best_price = entry_px  # tracks best favorable price for trailing

        exit_bar = -1
        exit_price = 0.0
        exit_type = EXIT_HORIZON

        max_bar = min(entry_bar + params.horizon, n_bars)

        for j in range(entry_bar + 1, max_bar):
            bar_high = high_arr[j]
            bar_low = low_arr[j]
            bar_close = close_arr[j]

            if dirn == 1:  # BUY position
                # Check SL hit (low touches SL)
                if bar_low <= sl_price:
                    exit_bar = j
                    exit_price = sl_price
                    if trailing_active:
                        exit_type = EXIT_TRAIL
                    elif breakeven_active:
                        exit_type = EXIT_BREAKEVEN
                    else:
                        exit_type = EXIT_SL
                    break

                # Check TP hit
                if bar_high >= tp_price:
                    exit_bar = j
                    exit_price = tp_price
                    exit_type = EXIT_TP
                    break

                # Track best price
                if bar_high > best_price:
                    best_price = bar_high

                # Phase 1: Breakeven activation
                profit = best_price - entry_px
                if not breakeven_active and profit >= be_threshold:
                    breakeven_active = True
                    sl_price = entry_px  # move SL to entry

                # Phase 2: Trailing activation
                if not trailing_active and profit >= trail_threshold:
                    trailing_active = True

                if trailing_active:
                    new_sl = best_price - trail_dist
                    if new_sl > sl_price:
                        sl_price = new_sl  # ratchet up only

            else:  # SELL position
                # Check SL hit (high touches SL)
                if bar_high >= sl_price:
                    exit_bar = j
                    exit_price = sl_price
                    if trailing_active:
                        exit_type = EXIT_TRAIL
                    elif breakeven_active:
                        exit_type = EXIT_BREAKEVEN
                    else:
                        exit_type = EXIT_SL
                    break

                # Check TP hit
                if bar_low <= tp_price:
                    exit_bar = j
                    exit_price = tp_price
                    exit_type = EXIT_TP
                    break

                # Track best price (lowest for sell)
                if bar_low < best_price:
                    best_price = bar_low

                # Phase 1: Breakeven
                profit = entry_px - best_price
                if not breakeven_active and profit >= be_threshold:
                    breakeven_active = True
                    sl_price = entry_px

                # Phase 2: Trailing
                if not trailing_active and profit >= trail_threshold:
                    trailing_active = True

                if trailing_active:
                    new_sl = best_price + trail_dist
                    if new_sl < sl_price:
                        sl_price = new_sl  # ratchet down only

        # Horizon expiry — close at last bar's close
        if exit_bar < 0:
            exit_bar = max_bar - 1
            exit_price = close_arr[exit_bar]
            exit_type = EXIT_HORIZON

        # Calculate PnL
        if dirn == 1:
            raw_pnl = (exit_price - entry_px) / entry_px
        else:
            raw_pnl = (entry_px - exit_price) / entry_px

        net_pnl = raw_pnl - cost_pct

        pnl[trade_count] = net_pnl
        bars_held[trade_count] = exit_bar - entry_bar
        exit_types[trade_count] = exit_type
        trade_count += 1

        position_end = exit_bar

    return pnl[:trade_count], bars_held[:trade_count], exit_types[:trade_count]


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    np.random.seed(42)
    n = 500
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    opn = close + np.random.randn(n) * 0.1
    atr = np.full(n, 1.0)

    entries = np.array([10, 50, 100, 150, 200, 250, 300, 350, 400])
    dirs = np.array([1, -1, 1, -1, 1, 1, -1, 1, -1])

    ep = ExitParams(
        atr_sl_mult=1.5, atr_tp_mult=4.5,
        breakeven_atr=1.0, trail_activation_atr=2.0,
        trail_distance_atr=1.0, horizon=24,
    )
    pnl, held, types = simulate_trades(entries, dirs, opn, high, low, close, atr, ep, 0.001)
    exit_names = {0: "SL", 1: "TP", 2: "BE", 3: "TRAIL", 4: "HORIZON"}
    print(f"Trades: {len(pnl)}")
    for k in range(len(pnl)):
        print(f"  #{k}: PnL={pnl[k]:+.4f}  bars={held[k]}  exit={exit_names[types[k]]}")
    print(f"Mean PnL: {pnl.mean():.4f}")
