"""Pydantic response models for the API."""
from __future__ import annotations

from pydantic import BaseModel


# --- Auth ---
class LoginRequest(BaseModel):
    api_key: str


class LoginResponse(BaseModel):
    token: str
    expires_in_days: int


# --- Dashboard ---
class FTMOProgress(BaseModel):
    account_size: float
    profit_target_pct: float
    profit_current_pct: float
    daily_loss_used_pct: float
    daily_loss_limit_pct: float
    total_dd_used_pct: float
    total_dd_limit_pct: float


class DashboardResponse(BaseModel):
    balance: float
    equity: float
    daily_pnl: float
    daily_pnl_pct: float
    open_positions: int
    trades_today: int
    win_rate_today: float | None
    bot_running: bool
    ftmo: FTMOProgress
    recent_trades: list[dict]
    equity_sparkline: list[float]


# --- Positions ---
class Position(BaseModel):
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    current_price: float
    pnl: float
    swap: float
    sl: float | None
    tp: float | None
    magic: int
    open_time: str | None
    sector: str | None


class PositionsResponse(BaseModel):
    positions: list[Position]
    total_pnl: float


# --- History ---
class Trade(BaseModel):
    id: int
    timestamp: str
    symbol: str
    direction: str
    entry_price: float | None
    exit_price: float | None
    lot_size: float | None
    pnl: float | None
    sl_price: float | None
    tp_price: float | None
    ml_confidence: float | None
    status: str | None
    exit_timestamp: str | None


class HistoryResponse(BaseModel):
    trades: list[Trade]
    total: int
    page: int
    page_size: int


class HistoryStats(BaseModel):
    total_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    best_trade: float | None
    worst_trade: float | None


# --- Charts ---
class EquityPoint(BaseModel):
    timestamp: str
    equity: float


class DailyPnlBar(BaseModel):
    date: str
    pnl: float


# --- Controls ---
class BotStatus(BaseModel):
    bot_running: bool
    bot_uptime: str | None
    bridge_healthy: bool


class ControlResult(BaseModel):
    success: bool
    message: str


# --- Push ---
class PushRegisterRequest(BaseModel):
    token: str


class PushRegisterResponse(BaseModel):
    success: bool
