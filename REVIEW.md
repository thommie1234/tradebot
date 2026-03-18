# Code Review Guidelines — Sovereign Trading System

## Critical: Always check
- No lookahead bias: ALL features must use `shift(1)` — only past data in ML features
- SL/TP validation: stop-loss must be on correct side of entry (BUY: SL < entry, SELL: SL > entry)
- No hardcoded leverage assumptions — use broker symbol info or account config
- MT5 calls are NOT thread-safe — verify they only happen on main thread
- Triple opt-in for live trading: config + CLI flag + env var ENABLE_LIVE_TRADING=1
- SQLite connections use WAL mode + busy_timeout for concurrent access
- Position sizing formulas unchanged — any PR touching risk/position_sizing.py needs extra scrutiny
- Guardrail checks (FTMO compliance, drawdown guard, correlation cap) must not be bypassed or reordered

## Security
- No API keys, tokens, or credentials in code (Telegram, Discord, broker credentials)
- No broker account numbers or IP in committed files
- .env files must stay in .gitignore

## Trading-specific
- Verify broker state after any MT5 modification (re-query positions after close/open)
- Trade copier signals must include all fields (account_id, symbol, direction, entry, SL, TP, lot, ticket)
- ATR calculations must use the correct timeframe per symbol (from config, not hardcoded H1)
- Friday close logic must respect per-symbol trading schedules
- Drawdown guard daily reset must be thread-safe (UTC midnight race condition)

## Architecture
- Entry/exit strategy logic must remain 100% identical unless explicitly changing strategy
- MasterOrderGenerator guardrails must match OrderRouter guardrails exactly
- AccountSnapshot is lazy — only created when signals exist, never on every tick
- Config changes to bf/config.json or ftmo/config.json are IP — never commit these

## Skip
- Files under `bf/`, `ftmo/` (account data, gitignored)
- Files under `data/cache/` (ephemeral runtime data)
- `*.db`, `*.db-wal`, `*.db-shm` files (SQLite runtime)
- `*.session` files (Telegram sessions)
- Formatting-only changes (no linter enforced)
