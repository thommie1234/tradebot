# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeBots is a modular algorithmic trading platform for MetaTrader 5 (MT5) running natively on Windows Server 2025. It supports live trading, paper trading, ML-powered trade filtering, Telegram signal scraping, and FTMO compliance guardrails.

**Clean architecture principle**: code that makes money is NEVER mixed with code that learns.

## Multi-Account Setup

Two accounts running simultaneously via separate processes (one MT5 terminal per process):

| | FTMO 100k | BrightFunded 100k |
|---|---|---|
| Account ID | `ftmo_100k` | `bright_100k` |
| MT5 Module | `MetaTrader5_FTMO` | `MetaTrader5` (default) |
| Terminal | `C:\Program Files\FTMO Global Markets MT5 Terminal\` | `C:\Program Files\BrightFunded MT5 Terminal\` |
| Config | `ftmo/config.json` | `bf/config.json` |
| Models | `ftmo/models/` | `bf/models/` |
| Audit DB | `ftmo/audit/sovereign_log.db` | `bf/audit/sovereign_log.db` |
| Logs | `ftmo/logs/service.log` | `bf/logs/service.log` |
| Risk | 0.3% (forex), 0.2% (index), 0.5% (NVDA) | 0.1% (survival mode) |
| Symbols | 9 (EURUSD, GBPUSD, USDJPY, GBPCAD, GBPAUD, NZDUSD, FRA40, US100, NVDA) | 4 (EURUSD, USDJPY, US100, FRA40) |

**IMPORTANT**: When a task involves trading or configuration, **ALWAYS ask which account** the user is referring to. They have different risk profiles, symbols, and compliance limits.

## Multi-Terminal MT5

MetaTrader5 DLL is process-locked: one terminal per Python process.

Solution: `MT5_MODULE` environment variable + duplicated package:
- Default (BF): `import MetaTrader5` → connects to BF terminal
- FTMO: `set "MT5_MODULE=MetaTrader5_FTMO"` → loads duplicated package with separate DLL/IPC
- `mt5_bridge.py` handles this via `importlib.import_module()` + `sys.modules` override

## Directory Structure

```
tradebots/
├── common/              ← All shared Python code
│   ├── engine/          ← Feature builder, inference, signal, multi_tf_scanner
│   ├── execution/       ← Order router, position manager, account context, broker API
│   ├── risk/            ← Position sizing, FTMO guard, drawdown, trading schedule
│   ├── live/            ← run_bot.py, paper_bot.py, paper_tracker.py
│   ├── research/        ← Training, optuna, WFA, backtesting
│   ├── tools/           ← MT5 bridge, telegram signals, trade copier, data downloader
│   ├── audit/           ← SQLite WAL audit logger with hash chains
│   ├── config/          ← YAML configs, loader.py, accounts.yaml
│   ├── api/             ← FastAPI dashboard backend (port 8000)
│   └── tests/           ← Unit and integration tests
├── bf/                  ← BF data artifacts (config, models, optuna, audit, logs)
├── ftmo/                ← FTMO data artifacts
├── trade_signals/       ← Cross-account trade copier signals (JSON files)
├── sovereign_launcher.py ← GUI launcher (customtkinter)
├── start_all.bat        ← Master startup for all 10 processes
└── .venv/               ← Python 3.12 virtualenv
```

## Running Processes (10 total via start_all.bat)

1. BF MT5 Terminal (/portable)
2. FTMO MT5 Terminal (/portable)
3. BF Live Bot
4. FTMO Live Bot
5. BF Paper Bot
6. FTMO Paper Bot
7. PredMarket Scheduler (C:\predmarket)
8. Telegram Signal Scraper (FTMO, 3 channels)
9. Trade Copier BF (copies FTMO trades to BF with 0.1% risk)
10. Trade Copier FTMO (copies BF trades to FTMO with 0.3% risk)

## Common Commands

```powershell
# Start everything (opens CMD windows for each process)
C:\tradebots\start_all.bat

# Or use the GUI launcher
pythonw sovereign_launcher.py

# Activate venv
C:\tradebots\.venv\Scripts\activate

# Live bot (manual start — prefer start_all.bat)
set ENABLE_LIVE_TRADING=1
python -u common\live\run_bot.py --live --account-id bright_100k

# FTMO bot (needs MT5_MODULE)
set "MT5_MODULE=MetaTrader5_FTMO"
set "ENABLE_LIVE_TRADING=1"
python -u common\live\run_bot.py --live --account-id ftmo_100k

# Telegram scraper
set "MT5_MODULE=MetaTrader5_FTMO"
python -u common\tools\telegram_signals.py

# Train models
python common/research/train_ml_strategy.py --symbols EURUSD,GBPUSD --timeframes M15
```

## Architecture

### Signal Flow
1. `run_bot.py` main loop every ~5s
2. `multi_tf_scanner.py` fetches bars from MT5, runs ML inference in parallel threads
3. If signal > threshold → `order_router.py` applies guardrails and sends order
4. `position_manager.py` monitors trailing stops in background (0.5s interval)
5. On fill: `order_router.py` writes JSON signal to `trade_signals/` for copier

### Component Responsibilities

| Package | Purpose |
|---------|---------|
| **engine/** | Feature building (39 features), XGBoost inference, multi-TF scanning |
| **execution/** | Order routing with guardrails, position management, account context |
| **risk/** | Position sizing, FTMO compliance, drawdown guard, trading schedule |
| **tools/** | MT5 bridge, telegram scraper, trade copier, data downloader |
| **audit/** | SQLite WAL + hash-chained event/trade logging |

### Optimized Exit Settings (tick-level MFE/MAE analysis)
- SL: 0.30x ATR
- Breakeven: 0.50x ATR
- Trail activation: 0.75x ATR
- Trail distance: 0.38x ATR
- TP: 10.0x ATR (never reached, trail does the work)

## Safety Design

Live trading requires triple opt-in:
1. Config: `execution.trading_enabled = True`
2. CLI flag: `--live`
3. Environment: `ENABLE_LIVE_TRADING=1`

## Key Conventions

- **Lookahead prevention**: ALL features use `shift(1)` — only past data
- **SQLite WAL mode**: All DB connections use WAL + busy_timeout for concurrent access
- **No formatter/linter**: Snake_case functions, PascalCase classes, 4-space indent
- **Config via `cfg` singleton**: `from config.loader import cfg`
- **Always verify broker state**: Re-query MT5 after modifications
- **Evaluate paper results per-symbol, never portfolio-level**
- **Express trade values in USD or % of equity**
- **NO OPTIMIZATION until 500+ trades**: only fix bugs, no parameter tuning on small samples
- **ConfMult is DISABLED**: data shows higher confidence = lower winrate. Fixed at 1.0.
- **TRADE_META logging**: every trade logs ADX, ATR ratio, session, hour, signal flip for post-analysis

## SessionExporter EA (CRITICAL)

**When adding a new symbol to the portfolio**, ALWAYS remind the user to add it to Market Watch in MT5 so the SessionExporter EA can export its trading hours. The EA runs on one chart per terminal and exports ALL Market Watch symbols to `data/sessions/{broker}_sessions.json`.

- EA source: `mt5_scripts/SessionExporter.mq5`
- BF terminal: BrokerTag="bf", outputs `bf_sessions.json`
- FTMO terminal: BrokerTag="ftmo", outputs `ftmo_sessions.json`
- `risk/session_guard.py` reads these files + Finnhub holidays for proactive session close
- Short breaks (<60 min) are ignored — only daily close triggers position close
- Finnhub API key in `.env` as `FINNHUB_API_KEY`

## Config Locations (common source of bugs!)

- `common/config/accounts.yaml` — account definitions (terminal paths, enabled flags)
- `config/accounts.yaml` — SECOND COPY that also needs updating
- `bf/config.json` / `ftmo/config.json` — per-symbol strategy configs (IP — not in git)

## Runtime Environment

- Windows Server 2025, Python 3.12, venv at `.venv/`
- MT5 native (no Wine) — separate terminal per account
- Hardware: Xeon E5-2690v4, 62GB RAM, Tesla P40 + GTX 1050
- Key packages: polars, xgboost, optuna, MetaTrader5, customtkinter, telethon
- GitHub: github.com/thommie1234/tradebot (IP excluded via .gitignore)
