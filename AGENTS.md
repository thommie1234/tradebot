# Repository Guidelines

## Project Structure & Module Organization

This repo hosts a modular MetaTrader 5 trading system. The canonical source is `trading_prop/` (the legacy `prop/` tree is not the primary target).

- `trading_prop/main.py` – main trading loop entry point.
- `trading_prop/strategies/` – signal generation (no execution).
- `trading_prop/risk_manager.py` – risk gates and FTMO compliance.
- `trading_prop/execution.py` – order construction/execution (dry-run by default).
- `trading_prop/ml/` – ML feature engineering, labeling, training, Optuna pipelines.
- `trading_prop/production/` – runnable scripts and orchestration wrappers.
- `trading_prop/tests/` – pytest suite.
- `trading_prop/data/` – symbol lists and historical data inputs.

## Build, Test, and Development Commands

Run commands from `trading_prop/production/` unless noted:

```bash
pytest /home/tradebot/tradebots/trading_prop/tests/
./phase2_runner.sh backtest
WORKERS=6 ./start_download_max.sh
MAX_JOBS=$(nproc) ./run_backtest_all_cores.sh
python3 /home/tradebot/tradebots/trading_prop/ml/train_ml_strategy.py --symbols EURUSD,GBPUSD --timeframes M1,M5,M15 --out-dir ml_boxes
```

Live trading is opt‑in only (see Security & Configuration Tips).

## Coding Style & Naming Conventions

- Python code uses 4‑space indentation and snake_case for functions/variables.
- Classes use PascalCase; constants are UPPER_SNAKE_CASE.
- There is no enforced formatter or linter config in-repo; follow nearby patterns.
- Prefer clear module boundaries: strategy logic in `strategies/`, execution logic in `execution.py`, risk checks in `risk_manager.py`.

## Testing Guidelines

- Framework: pytest (see `trading_prop/tests/`).
- Test files are named `test_*.py`.
- Run the full suite with the pytest command above; add focused tests for new strategy, risk, or ML changes.

## Commit & Pull Request Guidelines

- No strict commit convention is established; use concise, imperative summaries (e.g., “Add cache guard for M1 feeds”).
- PRs should include: purpose, tests run, and any risk impact (especially trading/risk changes).
- Link related issues when available; include logs or outputs if behavior changes.

## Security & Configuration Tips

- Live trading is disabled by default and requires three explicit opt‑ins:
  - `execution.trading_enabled = True` in config
  - `--enable-live-trading` CLI flag
  - `ENABLE_LIVE_TRADING=1` environment variable
- `run_wine.sh` defaults `ENABLE_LIVE_TRADING=0`; do not bypass in development.
