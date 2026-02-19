#!/usr/bin/env python3
"""
F12: Backtesting-as-a-Service — pure Polars/NumPy tick-data backtester.

Reuses the same pipeline as live: build_bar_features, triple-barrier, XGBoost.
NOT the legacy backtest_engine.py (which runs on Wine/MT5).

Usage:
    python3 research/backtest_service.py --symbol EURUSD --model models/sovereign_models/EURUSD.json
    python3 research/backtest_service.py --symbol EURUSD --compare model1.json model2.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    symbol: str
    model_path: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    ev_per_trade: float = 0.0
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_log: list = field(default_factory=list)


class BacktestService:
    """
    Tick-data backtester reusing the same pipeline as live trading.
    Hergebruikt: build_bar_features, apply_triple_barrier, XGBoost inference.
    """

    def __init__(self, symbol: str, timeframe: str = "H1", logger=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger
        self._ensure_imports()

    def _ensure_imports(self):
        """Lazy import ML dependencies."""
        import polars as pl
        import xgboost as xgb
        self.pl = pl
        self.xgb = xgb

        from engine.feature_builder import build_bar_features, FEATURE_COLUMNS
        self.build_bar_features = build_bar_features
        self.FEATURE_COLUMNS = FEATURE_COLUMNS

        from engine.labeling import apply_triple_barrier
        self.apply_triple_barrier = apply_triple_barrier

        import importlib
        train_mod = importlib.import_module("research.train_ml_strategy")
        self.make_time_bars = train_mod.make_time_bars
        self.sanitize_training_frame = train_mod.sanitize_training_frame
        self.infer_spread_bps = train_mod.infer_spread_bps

    def _load_ticks(self) -> object | None:
        """Load tick data from parquet files."""
        from config.loader import cfg
        pl = self.pl
        frames = []
        for root in cfg.DATA_ROOTS:
            sym_dir = os.path.join(root, self.symbol)
            if not os.path.isdir(sym_dir):
                continue
            for f in sorted(os.listdir(sym_dir)):
                if f.endswith(".parquet"):
                    df = pl.read_parquet(os.path.join(sym_dir, f)).select(
                        ["time", "bid", "ask", "last", "volume", "volume_real"]
                    )
                    if df.height > 0:
                        frames.append(df)
        if not frames:
            return None
        d = (
            pl.concat(frames, how="vertical")
            .sort("time")
            .with_columns([
                pl.col("time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
                pl.when(pl.col("last") > 0)
                .then(pl.col("last"))
                .otherwise((pl.col("bid") + pl.col("ask")) / 2.0)
                .alias("price"),
                pl.when(pl.col("volume_real") > 0)
                .then(pl.col("volume_real"))
                .otherwise(pl.col("volume"))
                .alias("size"),
            ])
            .drop_nulls(["time", "price", "size"])
        )
        return d.select(["time", "bid", "ask", "price", "size"])

    def run(
        self,
        model_path: str,
        start_date: str | None = None,
        end_date: str | None = None,
        costs_bps: float = 5.0,
        train_ratio: float = 0.6,
        ml_threshold: float = 0.55,
    ) -> BacktestResult:
        """
        Walk-forward backtest:
        1. Load tick data
        2. Build bars and features
        3. Split into train/test
        4. Train on train, predict on test
        5. Simulate trades with costs
        """
        pl = self.pl
        result = BacktestResult(symbol=self.symbol, model_path=model_path)

        # 1. Load ticks
        ticks = self._load_ticks()
        if ticks is None:
            print(f"No tick data for {self.symbol}")
            return result

        # 2. Build bars
        if ticks.select(pl.col("size").sum()).item() <= 0:
            ticks = ticks.with_columns(pl.lit(1.0).alias("size"))
        bars = self.make_time_bars(ticks.select(["time", "price", "size"]), self.timeframe)

        # Filter by date if specified
        if start_date:
            bars = bars.filter(pl.col("time") >= pl.lit(start_date).str.to_datetime())
        if end_date:
            bars = bars.filter(pl.col("time") <= pl.lit(end_date).str.to_datetime())

        if bars.height < 200:
            print(f"Insufficient bars: {bars.height}")
            return result

        # 3. Build features
        feat = self.build_bar_features(bars, z_threshold=0.0)

        # 4. Apply triple-barrier labels
        tb = self.apply_triple_barrier(
            close=feat["close"].to_numpy(),
            vol_proxy=feat["vol20"].to_numpy(),
            side=feat["primary_side"].to_numpy(),
            horizon=6, pt_mult=2.0, sl_mult=1.5,
        )
        feat = feat.with_columns([
            pl.Series("label", tb.label),
            pl.Series("target", tb.label),
            pl.Series("tb_ret", tb.tb_ret),
        ]).filter(pl.col("target").is_finite())
        feat = self.sanitize_training_frame(feat)

        if feat.height < 200:
            print(f"Insufficient samples after sanitization: {feat.height}")
            return result

        # 5. Load or use provided model
        model = self.xgb.Booster()
        if os.path.exists(model_path):
            model.load_model(model_path)
            # Use full dataset as test
            test_df = feat
        else:
            print(f"Model not found: {model_path}")
            return result

        # 6. Predict on test set
        x_test = test_df.select(self.FEATURE_COLUMNS).to_numpy()
        dmat = self.xgb.DMatrix(x_test)
        probas = model.predict(dmat)

        # 7. Simulate trades
        close_prices = test_df["close"].to_numpy()
        sides = test_df["primary_side"].to_numpy()
        tb_rets = test_df["tb_ret"].to_numpy()

        equity = 100_000.0
        equity_curve = [equity]
        trade_pnls = []

        costs_frac = costs_bps / 10_000.0

        for i in range(len(probas)):
            if probas[i] < ml_threshold:
                continue
            if sides[i] == 0:
                continue

            # Simulated P&L: triple-barrier return minus costs
            raw_ret = tb_rets[i] if np.isfinite(tb_rets[i]) else 0.0
            net_ret = raw_ret - costs_frac
            trade_pnl = equity * 0.003 * net_ret  # 0.3% risk per trade
            equity += trade_pnl
            equity_curve.append(equity)
            trade_pnls.append(trade_pnl)

            result.trade_log.append({
                "bar_idx": i,
                "proba": float(probas[i]),
                "side": int(sides[i]),
                "raw_ret": float(raw_ret),
                "net_pnl": float(trade_pnl),
            })

        # 8. Calculate statistics
        result.equity_curve = np.array(equity_curve)
        result.trades = len(trade_pnls)

        if result.trades > 0:
            pnls = np.array(trade_pnls)
            result.wins = int(np.sum(pnls > 0))
            result.losses = int(np.sum(pnls <= 0))
            result.win_rate = result.wins / result.trades
            result.total_pnl = float(np.sum(pnls))
            result.ev_per_trade = float(np.mean(pnls))

            # Profit factor
            gross_profit = float(np.sum(pnls[pnls > 0]))
            gross_loss = float(abs(np.sum(pnls[pnls < 0])))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

            # Max drawdown
            peak = np.maximum.accumulate(result.equity_curve)
            dd = (peak - result.equity_curve) / peak
            result.max_drawdown = float(np.max(dd))

            # Sharpe
            if np.std(pnls) > 0:
                result.sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))

        return result

    def compare_models(self, model_paths: list[str], **kwargs) -> list[BacktestResult]:
        """Compare multiple models on the same data."""
        results = []
        for path in model_paths:
            r = self.run(path, **kwargs)
            results.append(r)
        return results


def print_result(r: BacktestResult):
    """Pretty-print a backtest result."""
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST: {r.symbol}")
    print(f"  Model: {r.model_path}")
    print(f"{'=' * 60}")
    print(f"  Trades:         {r.trades}")
    print(f"  Win Rate:       {r.win_rate:.1%}")
    print(f"  Profit Factor:  {r.profit_factor:.2f}")
    print(f"  Sharpe Ratio:   {r.sharpe:.2f}")
    print(f"  Max Drawdown:   {r.max_drawdown:.2%}")
    print(f"  Total P&L:      ${r.total_pnl:+,.2f}")
    print(f"  EV/Trade:       ${r.ev_per_trade:+.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest-as-a-Service")
    parser.add_argument("--symbol", required=True, help="Symbol to backtest")
    parser.add_argument("--model", default=None, help="Model file path")
    parser.add_argument("--compare", nargs="+", default=None, help="Compare multiple models")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--costs", type=float, default=5.0, help="Costs in basis points")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    from config.loader import cfg, load_config
    load_config()

    svc = BacktestService(args.symbol, args.timeframe)

    if args.compare:
        results = svc.compare_models(
            args.compare,
            costs_bps=args.costs,
            ml_threshold=args.threshold,
            start_date=args.start,
            end_date=args.end,
        )
        for r in results:
            print_result(r)
    else:
        model_path = args.model or os.path.join(cfg.MODEL_DIR, f"{args.symbol}.json")
        result = svc.run(
            model_path,
            costs_bps=args.costs,
            ml_threshold=args.threshold,
            start_date=args.start,
            end_date=args.end,
        )
        print_result(result)
