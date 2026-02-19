# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBots is a modular algorithmic trading platform for MetaTrader 5 (MT5) running via Wine on Linux. It supports live trading, backtesting, and ML-powered trade filtering with FTMO compliance guardrails.

**Clean architecture principle**: code that makes money is NEVER mixed with code that learns.

## Directory Structure

```
tradebots/
├── config/          ← YAML configs + loader.py (frozen dataclass singleton)
├── data/            ← raw data refs, instrument specs, symbols.xlsx
├── engine/          ← feature builder, inference, signal, labeling (knows NOTHING about money)
├── risk/            ← position sizing, FTMO guard, drawdown, correlation (dictator)
├── execution/       ← MT5 broker API, order router, spread filter, position manager
├── audit/           ← SQLite WAL audit trail + feature logging (was: logging/)
├── live/            ← orchestration: run_bot.py, healthcheck, emergency_kill, Wine scripts
├── research/        ← training, optuna, WFA, backtesting (may be destroyed freely)
├── models/          ← approved models + registry.yaml + optuna results
├── tools/           ← data downloader, discord, sentiment, LLM advisor, MT5 bridge
├── analysis/        ← post-trade forensics (never influences live)
├── archive/         ← trading_prop/ + prop/ (legacy, frozen)
└── .venv/           ← Python 3.14 virtualenv
```

## Common Commands

```bash
# Activate venv (Python 3.14)
source /home/tradebot/tradebots/.venv/bin/activate

# Live trading — ALWAYS use systemctl, NEVER start manually (causes duplicates)
systemctl --user restart sovereign-bot
systemctl --user status sovereign-bot
systemctl --user logs -f sovereign-bot      # follow logs

# Dry run (show plan, no trading)
python3 live/run_bot.py --dry-run

# Train models from tick data
python3 live/run_bot.py --train

# Retrain with new Optuna params
python3 live/run_bot.py --retrain

# Native ML training
python3 research/train_ml_strategy.py --symbols EURUSD,GBPUSD --timeframes M15

# Optuna hyperparameter optimization (multi-GPU)
bash research/run_optuna_optimized.sh

# Weekly reoptimization pipeline (Sunday 00:00 CET)
bash live/sunday_ritual.sh

# Download tick data
bash tools/start_download_max.sh

# Bridge proxy
systemctl --user restart mt5-bridge-proxy
```

### IMPORTANT: Bot Process Management

**NEVER** start the bot manually with `nohup python3 run_bot.py &` — systemd manages it.
The `sovereign-bot.service` has `Restart=always`, so killing the process will auto-restart it.
Starting manually on top of that creates duplicate bots that execute trades twice.

Always use:
- `systemctl --user restart sovereign-bot` to restart with new code
- `systemctl --user stop sovereign-bot` to stop (temporarily, will restart on reboot)
- `systemctl --user disable sovereign-bot && systemctl --user stop sovereign-bot` to fully stop

## Architecture

### Component Responsibilities

| Package | Location | Purpose |
|---------|----------|---------|
| **Config** | `config/` | YAML configs loaded into frozen dataclass `cfg` singleton |
| **Engine** | `engine/` | Feature building (28 leak-safe features), XGBoost inference, signal generation, model decay tracking |
| **Risk** | `risk/` | Half-Kelly position sizing, FTMO guardrails, drawdown gates, sector correlation limits |
| **Execution** | `execution/` | MT5 bridge wrapper, order routing with all guardrails, spread filter, trailing stop / breakeven management |
| **Audit** | `audit/` | SQLite WAL + hash-chained audit logging, trade-time feature snapshots to parquet |
| **Live** | `live/` | SovereignBot H1 loop orchestrator, heartbeat monitor, emergency kill |
| **Research** | `research/` | WFO training, Optuna optimization, backtesting, integrated pipeline |
| **Tools** | `tools/` | Data downloader, MT5 bridge proxy (Wine-side), Discord notifier, sentiment engine |

### ML Pipeline

- **`engine/feature_builder.py`** — 28 leak-safe features using Polars with strict `shift(1)` discipline
- **`engine/labeling.py`** — Dynamic triple-barrier labeling scaled by rolling volatility
- **`engine/inference.py`** — SovereignMLFilter: model loading, training, predict(), should_trade()
- **`research/train_ml_strategy.py`** — Walk-forward XGBoost with meta-labeling
- **`research/integrated_pipeline.py`** — Polars lazy pipeline with fractional differentiation, purged WF-CV, Optuna objective

### Communication Style

- **Always express trade values in pips, percentages, or total USD** — never in "per share" or "per unit" terms. Example: say "+$840" or "+0.8% of equity", not "+$1.18/share".

### Verification Discipline

- **Always verify broker state after modifications.** Never trust log output alone — always re-query MT5 (`positions_get`, `symbol_info`) to confirm that changes (SL/TP modifications, order fills, trailing stops) actually took effect on the broker side. Example: after trailing stop logs say SL moved, check `positions_get()` to verify the SL actually changed.
- **Assume nothing works until proven.** If you modify a stop-loss, verify it changed. If you place a trade, verify it filled. If you restart the bot, verify it reconnected and is managing positions.

### Key Conventions

- **Lookahead prevention is critical**: ALL features must use `shift(1)` so only past data is available
- **Purged walk-forward splits**: Embargo/gap bars between train and test sets to prevent data leakage
- **No formatter/linter enforced**: Snake_case functions, PascalCase classes, UPPER_SNAKE_CASE constants, 4-space indentation
- **Config via `cfg` singleton**: `from config.loader import cfg` everywhere — no hardcoded constants

## Safety Design

Live trading is disabled by default. Triple opt-in required:
1. Config: `execution.trading_enabled = True`
2. CLI flag: `--live`
3. Environment: `ENABLE_LIVE_TRADING=1`

## Data Roots

Tick data is stored across multiple disks (configured in `config/paths.yaml`):
```
/home/tradebot/ssd_data_1/tick_data   (NVMe, fast)
/home/tradebot/ssd_data_2/tick_data
/home/tradebot/data_1/tick_data        (SATA, bulk)
```

## Runtime Environment

- Python 3.14+ (system), venv at `.venv/`
- Wine 10+ for MT5 terminal (Python 3.11 under Wine)
- Hardware: Xeon E5-2690v4 (14c/28t), 62GB RAM, Tesla P40 + RTX 2060 + GTX 1050
- Key packages: polars, xgboost, optuna, numpy, scipy, pyyaml, MetaTrader5 (Wine-side)

## Manual Trade Execution

When the user asks to manually execute a trade "like the bot would", replicate the full bot logic:

1. **Connect** via `MT5BridgeClient(port=5056)`
2. **ATR**: fetch H1 bars, calculate ATR(14) for the symbol
3. **SL distance**: `ATR * atr_sl_mult` from `sovereign_configs.json`
4. **TP distance**: `ATR * atr_tp_mult * confidence_scale` where `confidence_scale = max(1.0, min(2.0, proba / 0.55))`
5. **Volume**: `(equity * risk_per_trade) / (sl_distance * contract_size)`, rounded to `volume_step`, clamped to `[volume_min, volume_max]`
6. **Send order** without SL/TP (to avoid "Invalid stops"), then **modify** with SLTP action to set SL and TP
7. **Verify** the position via `positions_get()` after execution
8. Use `magic=2000` (or per-symbol magic from config) so the bot recognizes and manages the position (trailing stop, breakeven)

All per-symbol parameters (atr_sl_mult, atr_tp_mult, risk_per_trade, magic_number) come from `config/sovereign_configs.json`.

## Systemd Services

| Service | ExecStart |
|---------|-----------|
| `sovereign-bot.service` | `live/run_bot.py --live` (Restart=always, MT5_BRIDGE_PORT=5056) |
| `mt5-bridge-proxy.service` | `live/run_wine.sh mt5_bridge_proxy.py` |
| `sunday-ritual.timer` | `live/sunday_ritual.sh` (Sunday 00:00) |
| `sentiment-engine.service` | `tools/sentiment_engine.py` |
