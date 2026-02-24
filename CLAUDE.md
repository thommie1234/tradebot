# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBots is a modular algorithmic trading platform for MetaTrader 5 (MT5) running via Wine on Linux. It supports live trading, backtesting, and ML-powered trade filtering with FTMO compliance guardrails.

**Clean architecture principle**: code that makes money is NEVER mixed with code that learns.

## Multi-Account Setup

This system runs **two accounts simultaneously**: FTMO (port 5056) and BrightFunded (port 5057). They share ML models but have isolated compliance, positions, risk, and audit trails.

**IMPORTANT**: When a task involves trading, deployment, configuration, risk, compliance, or account-specific operations, **ALWAYS ask which account** (FTMO or BrightFunded) the user is referring to before proceeding. Never assume — the accounts have different cost structures, compliance limits, and symbol configs.

| | FTMO 100k | BrightFunded 100k |
|---|---|---|
| Account ID | `ftmo_100k` | `bright_100k` |
| Service | `sovereign-bot-ftmo` | `sovereign-bot-bf` |
| Bridge port | 5056 | 5057 |
| GPU | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` |
| Config | `ftmo/config.json` | `bf/config.json` |
| Models | `ftmo/models/` | `bf/models/` |
| Optuna | `ftmo/optuna/` | `bf/optuna/` |
| Audit DB | `ftmo/audit/sovereign_log.db` | `bf/audit/sovereign_log.db` |
| Logs | `ftmo/logs/sovereign.log` | `bf/logs/sovereign.log` |
| Commission | Per-instrument (CSV specs) | 0 (spread-only) |
| Instrument specs | `common/data/instrument_specs/*.csv` | `common/data/instrument_specs/brightfunded.csv` |
| Risk scale | 1.0 | 0.8 |

## Directory Structure

```
tradebots/
├── common/              ← All shared Python code
│   ├── engine/          ← Feature builder, inference, signal, labeling
│   ├── execution/       ← MT5 broker API, order router, spread filter
│   ├── risk/            ← Position sizing, FTMO guard, drawdown, correlation
│   ├── live/            ← run_bot.py, healthcheck, emergency_kill
│   ├── research/        ← Training, optuna, WFA, backtesting
│   ├── tools/           ← Data downloader, discord, sentiment, MT5 bridge
│   ├── audit/           ← audit_logger.py, feature_logger.py
│   ├── config/          ← YAML configs, loader.py, accounts.yaml, paths.yaml
│   ├── models/          ← registry.yaml, legacy models
│   ├── data/            ← Tick data refs, instrument specs
│   ├── analysis/        ← Post-trade forensics
│   ├── api/             ← FastAPI web dashboard
│   └── tests/           ← Unit and integration tests
│
├── ftmo/                ← FTMO-specific data artifacts
│   ├── config.json      ← Per-symbol configs
│   ├── models/          ← XGBoost models
│   ├── optuna/          ← Optuna results
│   ├── audit/           ← Audit DB
│   └── logs/            ← Service logs
│
├── bf/                  ← BrightFunded-specific data artifacts
│   ├── config.json      ← Per-symbol configs
│   ├── models/          ← XGBoost models
│   ├── optuna/          ← Optuna results
│   ├── audit/           ← Audit DB
│   └── logs/            ← Service logs
│
├── config → common/config   ← Symlink for backward compat
├── models → common/models   ← Symlink for backward compat
├── data → common/data       ← Symlink for backward compat
├── audit → common/audit     ← Symlink for backward compat
├── archive/             ← Legacy, frozen
└── .venv/               ← Python 3.14 virtualenv
```

Python code lives in `common/`. Each account directory (`ftmo/`, `bf/`) contains ONLY data artifacts (configs, models, optuna, audit, logs). Symlinks at root level ensure `REPO_ROOT / "config"` etc. still resolve correctly.

**Import resolution**: A `.pth` file (`.venv/lib/python3.14/site-packages/tradebots.pth`) adds `common/` to `sys.path`, so `from config.loader import cfg` works without any import changes.

## Common Commands

```bash
# Activate venv (Python 3.14)
source /home/tradebot/tradebots/.venv/bin/activate

# Live trading — ALWAYS use systemctl, NEVER start manually (causes duplicates)
# Each account has its own service:
systemctl --user restart sovereign-bot-ftmo
systemctl --user restart sovereign-bot-bf
systemctl --user status sovereign-bot-ftmo
systemctl --user status sovereign-bot-bf
tail -f ftmo/logs/sovereign.log             # FTMO logs
tail -f bf/logs/sovereign.log               # BF logs

# Dry run (show plan, no trading)
python3 common/live/run_bot.py --dry-run

# Train models from tick data
python3 common/live/run_bot.py --train

# Retrain with new Optuna params
python3 common/live/run_bot.py --retrain

# Native ML training
python3 common/research/train_ml_strategy.py --symbols EURUSD,GBPUSD --timeframes M15

# Optuna hyperparameter optimization (multi-GPU)
bash common/research/run_optuna_optimized.sh

# Weekly reoptimization pipeline (Sunday 00:00 CET)
bash common/live/sunday_ritual.sh

# Download tick data
bash common/tools/start_download_max.sh

# Bridge proxy
systemctl --user restart mt5-bridge-proxy
```

### IMPORTANT: Bot Process Management

**NEVER** start the bot manually with `nohup python3 run_bot.py &` — systemd manages it.
Each account runs as a separate service with `Restart=always`. Starting manually creates duplicates.

Always use:
- `systemctl --user restart sovereign-bot-ftmo` to restart FTMO
- `systemctl --user restart sovereign-bot-bf` to restart BrightFunded
- `systemctl --user stop sovereign-bot-ftmo` to stop FTMO (temporarily)
- `systemctl --user disable sovereign-bot-ftmo && systemctl --user stop sovereign-bot-ftmo` to fully stop

The old combined `sovereign-bot.service` is disabled. Do not use it.

## Architecture

### Component Responsibilities

| Package | Location | Purpose |
|---------|----------|---------|
| **Config** | `common/config/` | YAML configs loaded into frozen dataclass `cfg` singleton |
| **Engine** | `common/engine/` | Feature building (28 leak-safe features), XGBoost inference, signal generation, model decay tracking |
| **Risk** | `common/risk/` | Half-Kelly position sizing, FTMO guardrails, drawdown gates, sector correlation limits |
| **Execution** | `common/execution/` | MT5 bridge wrapper, order routing with all guardrails, spread filter, trailing stop / breakeven management |
| **Audit** | `common/audit/` | SQLite WAL + hash-chained audit logging, trade-time feature snapshots to parquet |
| **Live** | `common/live/` | SovereignBot H1 loop orchestrator, heartbeat monitor, emergency kill |
| **Research** | `common/research/` | WFO training, Optuna optimization, backtesting, integrated pipeline |
| **Tools** | `common/tools/` | Data downloader, MT5 bridge proxy (Wine-side), Discord notifier, sentiment engine |

### ML Pipeline

- **`common/engine/feature_builder.py`** — 28 leak-safe features using Polars with strict `shift(1)` discipline
- **`common/engine/labeling.py`** — Dynamic triple-barrier labeling scaled by rolling volatility
- **`common/engine/inference.py`** — SovereignMLFilter: model loading, training, predict(), should_trade()
- **`common/research/train_ml_strategy.py`** — Walk-forward XGBoost with meta-labeling
- **`common/research/integrated_pipeline.py`** — Polars lazy pipeline with fractional differentiation, purged WF-CV, Optuna objective

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

Tick data is stored across multiple disks (configured in `common/config/paths.yaml`):
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

All per-symbol parameters (atr_sl_mult, atr_tp_mult, risk_per_trade, magic_number) come from `ftmo/config.json` (FTMO) or `bf/config.json` (BrightFunded).

## Systemd Services

| Service | ExecStart | Port | GPU | Logs |
|---------|-----------|------|-----|------|
| `sovereign-bot-ftmo.service` | `common/live/run_bot.py --live --account-id ftmo_100k` | 5056 | 0 | `ftmo/logs/sovereign.log` |
| `sovereign-bot-bf.service` | `common/live/run_bot.py --live --account-id bright_100k` | 5057 | 1 | `bf/logs/sovereign.log` |
| `mt5-bridge-proxy.service` | `common/live/run_wine.sh mt5_bridge_proxy.py` | — | — | — |
| `sunday-ritual.timer` | `common/live/sunday_ritual.sh` (Sunday 00:00) | — | — | — |
| `sentiment-engine.service` | `common/tools/sentiment_engine.py` | — | — | — |
| ~~`sovereign-bot.service`~~ | **DISABLED** (old combined service) | — | — | — |
