# Sovereign Trading Bot

A modular algorithmic trading system for MetaTrader 5 (MT5), built for prop firm challenges (FTMO $100K). Runs on Linux via Wine with ML-powered trade filtering, ATR-based risk management, and full audit trail.

## Architecture

```
                    ┌─────────────┐
                    │  MT5 Terminal│  (Wine)
                    │   FTMO Srv4  │
                    └──────┬──────┘
                           │ TCP socket
                    ┌──────┴──────┐
                    │ MT5 Bridge  │  tools/mt5_bridge.py
                    │   Proxy     │  live/mt5_bridge_proxy.py
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
   │   Engine    │  │  Execution │  │    Risk     │
   │             │  │            │  │             │
   │ Features    │  │ OrderRouter│  │ Half-Kelly  │
   │ XGBoost ML  │  │ PosManager │  │ FTMO Guard  │
   │ Signals     │  │ SpreadFilt │  │ Drawdown    │
   │ Multi-TF    │  │ Trailing   │  │ Correlation │
   └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────┴──────┐
                    │ SovereignBot│  live/run_bot.py
                    │  H1 Loop    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────┴───┐ ┌─────┴────┐ ┌─────┴─────┐
       │  Config  │ │  Audit   │ │  Tools    │
       │          │ │          │ │           │
       │ YAML     │ │ SQLite   │ │ Discord   │
       │ Loader   │ │ WAL      │ │ Sentiment │
       │ Configs  │ │ Features │ │ Downloader│
       └──────────┘ └──────────┘ └───────────┘
```

**Core principle**: code that makes money is never mixed with code that learns.

## Directory Structure

```
tradebots/
├── config/          YAML configs + frozen dataclass singleton loader
├── engine/          Feature builder, ML inference, signal generation (no money logic)
├── execution/       MT5 order routing, position manager, spread filter
├── risk/            Position sizing, FTMO compliance, drawdown gates
├── audit/           SQLite WAL audit trail + feature snapshots
├── live/            Bot orchestrator, healthcheck, emergency kill, systemd scripts
├── research/        Backtesting, Optuna optimization, walk-forward analysis
├── models/          Trained models + registry
├── tools/           Data downloader, MT5 bridge, Discord, sentiment engine
├── analysis/        Post-trade forensics (Monte Carlo, bootstrap, journal)
├── data/            FTMO instrument specs per asset class
└── api/             FastAPI web interface with JWT auth
```

## Features

### ML Pipeline
- **28 leak-safe features** built with Polars using strict `shift(1)` — no lookahead bias
- **Dynamic triple-barrier labeling** scaled by rolling volatility
- **Walk-forward XGBoost** with purged cross-validation and embargo gaps
- **Optuna hyperparameter optimization** with multi-GPU support
- **Model decay tracking** — automatic retraining triggers
- **Multi-timeframe scanning** — M30, H1, H4 signals running in parallel

### Risk Management
- **Half-Kelly position sizing** with reinforcement learning adjustment
- **ATR-based stops**: per-symbol SL/TP multipliers calibrated by asset class
- **Trailing stops + breakeven**: automatic management with ATR-scaled activation
- **FTMO compliance**: 5% daily loss limit, 10% max drawdown, Friday close buffer
- **Drawdown recovery mode**: automatic lot reduction at 2% drawdown
- **Sector correlation limits**: prevents concentrated exposure (e.g. max 1.5% in US equities)
- **Spread filter**: rejects trades when spread exceeds asset-class thresholds

### Execution
- **MT5 bridge**: TCP socket proxy between Linux Python and Wine-side MT5 terminal
- **Two-step order placement**: send without SL/TP, then modify (prevents "Invalid stops")
- **Session-aware auto-close**: positions closed 2 minutes before market close
- **Position verification**: every order/modification is verified against broker state
- **Slippage tracking**: monitors fill quality vs. expected price

### Infrastructure
- **systemd managed**: `sovereign-bot.service` with `Restart=always`
- **Discord notifications**: real-time trade alerts, daily P&L summaries
- **Sentiment engine**: market sentiment scoring for trade context
- **Sunday ritual**: weekly automated model retraining + parameter reoptimization
- **Emergency kill**: instant position closure across all symbols
- **Multi-disk tick storage**: NVMe + SATA across 3 mount points

## Traded Instruments

Currently active on 8 symbols across multiple asset classes:

| Symbol | Class | Timeframe | ML Threshold |
|--------|-------|-----------|-------------|
| AMZN   | US Equity  | M30  | 0.55 |
| TSLA   | US Equity  | H1   | 0.55 |
| NVDA   | US Equity  | H1   | 0.51 |
| PFE    | US Equity  | H1   | 0.55 |
| RACE   | EU Equity  | H4   | 0.58 |
| LVMH   | EU Equity  | H1   | 0.56 |
| JP225.cash | Index | H4   | 0.56 |
| margin_leverage | Config | — | — |

## Safety Design

Live trading requires triple opt-in:
1. Config: `execution.trading_enabled: true`
2. CLI flag: `--live`
3. Environment variable: `ENABLE_LIVE_TRADING=1`

All trades pass through 6 guardrails before execution:
1. ML probability threshold (per-symbol)
2. Spread filter (per-asset-class)
3. FTMO daily loss check
4. Drawdown gate
5. Sector correlation limit
6. Session hours check

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Dry run — see what the bot would do
python3 live/run_bot.py --dry-run

# Start live trading (via systemd)
systemctl --user start sovereign-bot

# Monitor
systemctl --user status sovereign-bot
tail -f audit/logs/sovereign_live.nohup.log

# Manual trade (replicates full bot logic)
python3 tools/manual_trade.py AMZN BUY --dry-run
python3 tools/manual_trade.py AMZN SELL --flip

# Download tick data
bash tools/start_download_max.sh

# Run backtest
python3 research/backtest_engine.py
```

## Systemd Services

| Service | Purpose | Auto-restart |
|---------|---------|-------------|
| `sovereign-bot` | Main trading bot (H1 loop) | Yes |
| `mt5-bridge-proxy` | MT5 TCP bridge (Wine) | Yes |
| `sentiment-engine` | Market sentiment scoring | Yes |
| `sunday-ritual.timer` | Weekly retraining (Sun 00:00) | Timer |

## Runtime Environment

- **OS**: Arch Linux (CachyOS kernel 6.12)
- **Python**: 3.14+ (native) + 3.11 (Wine for MT5)
- **Hardware**: Xeon E5-2690v4 (14c/28t), 62GB RAM
- **GPUs**: Tesla P40 + RTX 2060 + GTX 1050
- **Key packages**: polars, xgboost, optuna, numpy, scipy

## Config Example

```yaml
# risk.yaml
kelly:
  fraction: 0.5        # Half-Kelly
  min_trades: 20       # Minimum history before Kelly kicks in
daily:
  budget: 0.035        # 3.5% daily risk budget
  loss_limit: -0.035   # Stop trading at -3.5% daily
max_concurrent_positions: 8

# ftmo.yaml
max_daily_loss_pct: 0.05   # FTMO 5% daily limit
max_total_loss_pct: 0.10   # FTMO 10% total limit
```

## License

Private repository. Not for distribution.
