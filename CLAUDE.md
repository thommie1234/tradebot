# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBots is a modular algorithmic trading platform for MetaTrader 5 (MT5) running natively on Windows Server 2025. It supports live trading, backtesting, and ML-powered trade filtering with FTMO compliance guardrails.

**Clean architecture principle**: code that makes money is NEVER mixed with code that learns.

## Multi-Account Setup

This system runs **multiple accounts simultaneously**: FTMO (port 5056), BrightFunded (port 5057), and TTP (port 5058, paper-only). They share ML code but have isolated models, compliance, positions, risk, and audit trails.

**IMPORTANT**: When a task involves trading, deployment, configuration, risk, compliance, or account-specific operations, **ALWAYS ask which account** (FTMO or BrightFunded) the user is referring to before proceeding. Never assume — the accounts have different cost structures, compliance limits, and symbol configs.

| | FTMO 100k | BrightFunded 100k | TTP Demo |
|---|---|---|---|
| Account ID | `ftmo_100k` | `bright_100k` | `ttp_demo` |
| Service | `sovereign-bot-ftmo` | `sovereign-bot-bf` | `sovereign-bot-ttp` |
| Bridge port | 5056 | 5057 | 5058 |
| GPU | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` | — |
| Config | `ftmo/config.json` | `bf/config.json` | `ttp/config.json` |
| Models | `ftmo/models/` | `bf/models/` | `ttp/models/` |
| Optuna | `ftmo/optuna/` | `bf/optuna/` | — |
| Audit DB | `ftmo/audit/sovereign_log.db` | `bf/audit/sovereign_log.db` | `ttp/audit/` |
| Logs | `ftmo/logs/service.log` | `bf/logs/service.log` | `ttp/logs/service.log` |
| Commission | Per-instrument (CSV specs) | 0 (spread-only) | — |
| Risk scale | 1.0 | 0.8 | 1.0 |
| Mode | Live | Live | Paper only |

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
└── .venv/               ← Python 3.12 virtualenv
```

Python code lives in `common/`. Each account directory (`ftmo/`, `bf/`) contains ONLY data artifacts (configs, models, optuna, audit, logs). Symlinks at root level ensure `REPO_ROOT / "config"` etc. still resolve correctly.

**Import resolution**: A `.pth` file (`.venv/Lib/site-packages/tradebots.pth`) adds `common/` to `sys.path`, so `from config.loader import cfg` works without any import changes.

## Common Commands

```powershell
# Activate venv (Python 3.12)
C:\tradebots\.venv\Scripts\activate

# Live trading — ALWAYS use NSSM, NEVER start manually (causes duplicates)
nssm restart sovereign-bot-ftmo
nssm restart sovereign-bot-bf
nssm restart sovereign-bot-ttp
nssm status sovereign-bot-ftmo
nssm status sovereign-bot-bf
Get-Content ftmo\logs\service.log -Tail 50 -Wait   # FTMO logs
Get-Content bf\logs\service.log -Tail 50 -Wait      # BF logs

# Dry run (show plan, no trading)
python common/live/run_bot.py --dry-run

# Train models from tick data
python common/live/run_bot.py --train

# Retrain with new Optuna params
python common/live/run_bot.py --retrain

# Native ML training
python common/research/train_ml_strategy.py --symbols EURUSD,GBPUSD --timeframes M15

# Optuna hyperparameter optimization (with account-specific costs)
python common/research/optuna_orchestrator.py --account ftmo_100k --all --timeframe H4 --trials 600

# Exit parameter optimization
python common/research/exit_optuna.py --account ftmo_100k --active --trials 400 --workers 12

# WFO validation (unified — replaces old WFO scripts)
python common/research/wfo_validate.py --account ftmo_100k --entry-csv ... --exit-csv ... --out-dir ...
```

### IMPORTANT: Bot Process Management

**NEVER** start the bot manually with `python run_bot.py` — NSSM manages it.
Each account runs as a separate NSSM service with auto-restart. Starting manually creates duplicates.

Always use:
- `nssm restart sovereign-bot-ftmo` to restart FTMO
- `nssm restart sovereign-bot-bf` to restart BrightFunded
- `nssm restart sovereign-bot-ttp` to restart TTP
- `nssm stop sovereign-bot-ftmo` to stop FTMO (temporarily)

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
| **Tools** | `common/tools/` | Data downloader, MT5 native bridge, Discord notifier, sentiment engine |

### ML Pipeline

- **`common/engine/feature_builder.py`** — 39 leak-safe features (28 original + 3 regime + 5 tick + 3 lead-lag) using Polars with strict `shift(1)` discipline
- **`common/engine/labeling.py`** — Dynamic triple-barrier labeling scaled by rolling volatility
- **`common/engine/inference.py`** — SovereignMLFilter: model loading, training, predict(), should_trade()
- **`common/research/train_ml_strategy.py`** — Walk-forward XGBoost with meta-labeling
- **`common/research/integrated_pipeline.py`** — Polars lazy pipeline with fractional differentiation, purged WF-CV, Optuna objective

### IMPORTANT: Tick Data → Time-Based Bars

**All tick data must be converted to time-based OHLCV bars (M5/M15/M30/H1/H4) before feature building.** The pipeline should aggregate raw ticks into proper time-based candles — NOT use tick-count bars or raw tick prices directly. This applies to all data sources (MT5 copy_ticks_from, Alpaca trade ticks, etc.).

### Communication Style

- **Always express trade values in pips, percentages, or total USD** — never in "per share" or "per unit" terms. Example: say "+$840" or "+0.8% of equity", not "+$1.18/share".

### Paper Trade Evaluation

- **Always evaluate paper results at the ticker level, never at portfolio level.** Paper runs ALL symbols × ALL timeframes, so portfolio totals are meaningless. A symbol like US100 M5 can be consistently profitable while the portfolio shows a loss because other symbols drag it down. Always break down by symbol × TF when analyzing paper performance.

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

## MT5 Bridge (tools/mt5_bridge.py)

MT5 is a **singleton** — only one terminal connection per Python process. The `MT5BridgeClient` wraps the native `MetaTrader5` module. Key implications:

- `get_mt5_bridge(name, terminal_path)` creates named instances but they all share one `_mt5` module
- Only the first `initialize(terminal_path=...)` actually connects; subsequent calls are no-ops
- In multi-account mode, `account_context.py` handles MT5 init per account (not `broker_api.py`)
- The `ping()` method checks `_mt5.terminal_info()` as fallback since `_initialized` flag is per-instance

**Account terminals** are defined in `config/accounts.yaml` under `terminal_path`. Each account needs its own MT5 terminal installation.

## Paper Trading (live/paper_bot.py + live/paper_tracker.py)

The paper bot runs independently from the live bot as a separate NSSM service (`paper-bot-bf`). It scans ALL symbols with trained models across M5/M15/M30/H1/H4, simulates trades with real MT5 prices, and tracks positions with SL/TP/trailing/BE.

**Known issues**: Paper bot trades ALL symbols including exotics — position sizing can be broken for exotic pairs (especially JPY crosses where pip value calculation may be incorrect). Always analyze paper results per-symbol, never portfolio-level.

Paper trade databases: `{account}/audit/paper_trades.db` (SQLite).

## Data Roots

Tick data is stored on disk (configured in `common/config/paths.yaml`):
```
C:\tick_data\ssd1    (tick data from sda1)
C:\tick_data\ssd2    (tick data from sdd2)
C:\tick_data\nvme    (tick data from NVMe)
C:\tick_data\bars    (OHLCV bar data)
```

## Runtime Environment

- Windows Server 2025 (Desktop Experience)
- Python 3.12 (native), venv at `.venv/`
- MT5 native (no Wine needed) — separate terminal per account in `C:\MT5\`
- Hardware: Xeon E5-2690v4 (14c/28t), 62GB RAM, Tesla P40 + GTX 1050
- NVIDIA Data Center Driver 582.16, CUDA 13.0
- Key packages: polars, xgboost, optuna, numpy, scipy, pyyaml, MetaTrader5
- NSSM at `C:\tools\nssm.exe` — manages all bot services with auto-restart
- Windows Task Scheduler for ritual timers (Saturday retraining, data downloads)
- Ollama for LLM scan commentary (llama3.1:8b)

## Manual Trade Execution

When the user asks to manually execute a trade "like the bot would", replicate the full bot logic:

1. **Connect** via `get_mt5_bridge(name="manual", terminal_path=...)`
2. **ATR**: fetch H1 bars, calculate ATR(14) for the symbol
3. **SL distance**: `ATR * atr_sl_mult` from `sovereign_configs.json`
4. **TP distance**: `ATR * atr_tp_mult * confidence_scale` where `confidence_scale = max(1.0, min(2.0, proba / 0.55))`
5. **Volume**: `(equity * risk_per_trade) / (sl_distance * contract_size)`, rounded to `volume_step`, clamped to `[volume_min, volume_max]`
6. **Send order** without SL/TP (to avoid "Invalid stops"), then **modify** with SLTP action to set SL and TP
7. **Verify** the position via `positions_get()` after execution
8. Use `magic=2000` (or per-symbol magic from config) so the bot recognizes and manages the position (trailing stop, breakeven)

All per-symbol parameters (atr_sl_mult, atr_tp_mult, risk_per_trade, magic_number) come from `ftmo/config.json` (FTMO) or `bf/config.json` (BrightFunded).

## Systemd Services

| Service (NSSM) | Command | Logs |
|----------------|---------|------|
| `sovereign-bot-ftmo` | `run_bot.py --live --account-id ftmo_100k` | `ftmo/logs/service.log` |
| `sovereign-bot-bf` | `run_bot.py --live --account-id bright_100k` | `bf/logs/service.log` |
| `sovereign-bot-ttp` | `run_bot.py --live --account-id ttp_demo` | `ttp/logs/service.log` |
| `sentiment-engine` | `common/tools/sentiment_engine.py` | — |

Ritual timers run via Windows Task Scheduler (Saturday 00:00 / 12:00).

## Config Locations

There are TWO config directories — this is a common source of bugs:

- `common/config/` — shared configs (accounts.yaml, paths.yaml, sovereign_configs.json)
- `config/` — symlink to `common/config/` (backward compat)
- `{account}/config.json` — per-account symbol configs with enabled/disabled flags, thresholds, risk params

The `config/loader.py` loads the `cfg` singleton. **REPO_ROOT** resolves to `C:\tradebots`. If loader.py is copied elsewhere (e.g., `config/loader.py` at root level), verify REPO_ROOT still resolves correctly.

## Testing

```bash
# Run tests (from repo root with venv activated)
python -m pytest common/engine/tests/
python -m pytest common/execution/tests/
python -m pytest common/risk/tests/

# Single test
python -m pytest common/engine/tests/test_feature_builder.py -v
```

## Multi-Timeframe Scanning

The bot scans multiple timeframes (M5, M15, M30, H1, H4) via `engine/multi_tf_scanner.py`. Each symbol has a primary timeframe defined in its config (`atr_timeframe`/`exit_timeframe`). Models are stored in TF subdirectories: `{account}/models/{TF}/{SYMBOL}.json` (H1 models are in the root `models/` dir).

Scan intervals align with candle closes: M5 every 5min, M15 every 15min, etc.
