"""
Emergency kill — close all positions + Friday auto-close + progressive Friday close.

Extracted from emergency_close_all() + friday_auto_close() in sovereign_bot.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

from config.loader import cfg


def emergency_close_all(logger, mt5, discord=None):
    """Emergency: close all positions."""
    if mt5 is None:
        return
    logger.log('CRITICAL', 'EmergencyKill', 'EMERGENCY_STOP',
               'EMERGENCY STOP — closing all positions')

    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        if pos.magic < 2000:
            continue
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": close_price,
            "deviation": cfg.DEVIATION,
            "magic": pos.magic,
            "comment": "EMERGENCY_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.log('INFO', 'EmergencyKill', 'POS_CLOSED',
                        f'Closed {pos.ticket} ({pos.symbol})')

    if discord:
        discord.send("EMERGENCY STOP", "All Sovereign positions closed", "red")


def profit_close_all(logger, mt5, floating_pnl: float, account_size: float,
                     account_name: str = "", discord=None):
    """Profit take: close all positions when floating P&L hits the target."""
    if mt5 is None:
        return
    pct = floating_pnl / account_size * 100 if account_size > 0 else 0
    msg = f'Floating P&L +${floating_pnl:,.2f} (+{pct:.2f}%) — taking profit, closing all positions'
    logger.log('INFO', 'ProfitGate', 'PROFIT_CLOSE_ALL', msg)

    positions = mt5.positions_get()
    if not positions:
        return

    closed = 0
    for pos in positions:
        if pos.magic < 2000:
            continue
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": close_price,
            "deviation": cfg.DEVIATION,
            "magic": pos.magic,
            "comment": "PROFIT_TAKE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.log('INFO', 'ProfitGate', 'POS_CLOSED',
                       f'Closed {pos.ticket} ({pos.symbol}) PROFIT_TAKE')
            closed += 1

    if discord:
        name = f"[{account_name}] " if account_name else ""
        discord.send(
            f"{name}PROFIT TARGET HIT",
            f"Floating +${floating_pnl:,.2f} (+{pct:.2f}%)\n{closed} positions closed.",
            "green",
        )


def profit_lock_breakeven(logger, mt5, floating_pnl: float, account_size: float,
                          account_name: str = "", discord=None):
    """At soft profit target: move all profitable SLs to breakeven.

    Lets the trailing stop continue managing positions — if the market
    keeps going we capture more than the soft target.
    """
    if mt5 is None:
        return
    pct = floating_pnl / account_size * 100 if account_size > 0 else 0
    positions = mt5.positions_get()
    if not positions:
        return

    moved = 0
    for pos in positions:
        if pos.magic < 2000:
            continue
        entry = pos.price_open
        current_sl = pos.sl or 0
        if pos.type == 0:  # BUY
            if current_sl >= entry:
                continue
            new_sl = entry
        else:  # SELL
            if current_sl != 0 and current_sl <= entry:
                continue
            new_sl = entry

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp,
        })
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.log('INFO', 'ProfitGate', 'SL_TO_BREAKEVEN',
                       f'{pos.symbol} ticket={pos.ticket} SL->{new_sl} (entry={entry})')
            moved += 1

    if moved > 0 and discord:
        name = f"[{account_name}] " if account_name else ""
        discord.send(
            f"{name}PROFIT GATE +{pct:.1f}%",
            f"Floating +${floating_pnl:,.2f}\n{moved} SLs naar breakeven.",
            "green",
        )
    logger.log('INFO', 'ProfitGate', 'BREAKEVEN_LOCK',
               f'Floating +${floating_pnl:,.2f} (+{pct:.2f}%) -- {moved} SLs naar breakeven')


def daily_loss_protect(logger, mt5, account_name: str = "", discord=None):
    """At -2% daily: protect all positions to minimize further loss.

    - Positions in PROFIT: SL to entry (can only win from here)
    - Positions in LOSS: TP to entry (close at 0 when price returns)
    - No new trades (handled by drawdown_guard.check_daily_limits)
    """
    if mt5 is None:
        return
    positions = mt5.positions_get()
    if not positions:
        return

    protected = 0
    for pos in positions:
        if pos.magic < 2000:
            continue

        entry = pos.price_open
        is_buy = pos.type == 0
        in_profit = pos.profit > 0

        if in_profit:
            # Move SL to entry — can only win from here
            current_sl = pos.sl or 0
            if is_buy and current_sl >= entry:
                continue  # already safe
            if not is_buy and current_sl != 0 and current_sl <= entry:
                continue  # already safe

            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": entry,
                "tp": pos.tp,
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.log('INFO', 'DailyLossProtect', 'SL_TO_ENTRY',
                           f'{pos.symbol} ticket={pos.ticket} SL->{entry} (in profit)')
                protected += 1
        else:
            # Set TP to entry — close at 0 when price comes back
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": pos.sl,
                "tp": entry,
            })
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.log('INFO', 'DailyLossProtect', 'TP_TO_ENTRY',
                           f'{pos.symbol} ticket={pos.ticket} TP->{entry} (in loss, close at 0)')
                protected += 1

    if discord:
        name = f"[{account_name}] " if account_name else ""
        discord.send(
            f"{name}DAILY LOSS PROTECT -2%",
            f"{protected} positions protected.\n"
            f"Winners: SL to entry. Losers: TP to entry.\n"
            f"No new trades until tomorrow.",
            "red",
        )
    logger.log('WARNING', 'DailyLossProtect', 'ACTIVATED',
               f'{protected} positions protected at -2% daily loss')


def friday_auto_close(logger, mt5, trading_schedule, running, emergency_stop, discord=None):
    """Close non-crypto positions before weekend."""
    if not running or emergency_stop or mt5 is None:
        return

    positions = mt5.positions_get()
    if not positions:
        return

    closed = []
    for pos in positions:
        if pos.magic < 2000:
            continue

        if not trading_schedule.should_friday_close(pos.symbol):
            continue

        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": cfg.DEVIATION,
            "magic": pos.magic,
            "comment": "Sovereign_FridayClose",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Get real P&L from deal history
            real_pnl = None
            try:
                now_utc = datetime.now(timezone.utc)
                deals = mt5.history_deals_get(now_utc - __import__('datetime').timedelta(days=7), now_utc, position=pos.ticket)
                if deals:
                    real_pnl = sum(d.profit + d.commission + d.swap for d in deals if d.position_id == pos.ticket)
            except Exception:
                pass
            pnl_str = f"${real_pnl:+.2f}" if real_pnl is not None else f"~${pos.profit + pos.swap:+.2f}"
            closed.append(f"{pos.symbol} ({pnl_str})")
            logger.log('INFO', 'EmergencyKill', 'FRIDAY_CLOSE',
                        f'Closed {pos.symbol} ticket {pos.ticket} PnL={pnl_str}')
            # Update trades table with exit info
            if real_pnl is not None and hasattr(logger, 'close_trade'):
                exit_price = close_price
                try:
                    for d in deals:
                        if d.position_id == pos.ticket and hasattr(d, 'entry') and d.entry == 1:
                            exit_price = d.price
                            break
                except Exception:
                    pass
                logger.close_trade(pos.ticket, exit_price, real_pnl)
        else:
            logger.log('ERROR', 'EmergencyKill', 'FRIDAY_CLOSE_FAILED',
                        f'Failed to close {pos.symbol} ticket {pos.ticket}')

    if closed and discord:
        discord.send(
            "FRIDAY AUTO-CLOSE",
            f"Closed {len(closed)} positions before weekend:\n" + ", ".join(closed),
            "blue"
        )


def friday_progressive_close(logger, mt5, trading_schedule, running, emergency_stop, discord=None):
    """F9: Friday close — close ALL non-crypto positions at 23:00 GMT+2 (= 22:00 CET).

    23:00 GMT+2 Friday: close ALL non-crypto positions before weekend.
    Existing friday_auto_close() (~10 min before market close) remains as final safety net.
    """
    if not running or emergency_stop or mt5 is None:
        return

    from risk.ftmo_guard import TradingSchedule
    now = datetime.now(timezone.utc) + TradingSchedule.GMT2_OFFSET
    if now.weekday() != 4:
        return

    hour = now.hour
    if hour < 22:  # Was 23 — too late for EU indices (FRA40 closes 22:49 GMT+2)
        return

    positions = mt5.positions_get()
    if not positions:
        return

    closed = []
    for pos in positions:
        if pos.magic < 2000:
            continue

        # Skip crypto (trades 24/7, no weekend gap risk)
        if trading_schedule.is_crypto(pos.symbol):
            continue

        pnl = pos.profit + pos.swap

        # 22:00+: close everything non-crypto

        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": cfg.DEVIATION,
            "magic": pos.magic,
            "comment": "Sovereign_FridayProgressive",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            phase = "ALL"
            # Get real P&L from deal history
            real_pnl = None
            try:
                deals = mt5.history_deals_get(
                    datetime.now(timezone.utc) - __import__('datetime').timedelta(days=7),
                    datetime.now(timezone.utc), position=pos.ticket)
                if deals:
                    real_pnl = sum(d.profit + d.commission + d.swap for d in deals if d.position_id == pos.ticket)
            except Exception:
                pass
            pnl_str = f"${real_pnl:+.2f}" if real_pnl is not None else f"~${pnl:+.2f}"
            closed.append(f"{pos.symbol} ({pnl_str})")
            logger.log('INFO', 'EmergencyKill', 'FRIDAY_PROGRESSIVE',
                        f'[{phase}] Closed {pos.symbol} {direction} ticket {pos.ticket} '
                        f'PnL={pnl_str}')
            # Update trades table with exit info
            if real_pnl is not None and hasattr(logger, 'close_trade'):
                exit_price = close_price
                try:
                    for d in deals:
                        if d.position_id == pos.ticket and hasattr(d, 'entry') and d.entry == 1:
                            exit_price = d.price
                            break
                except Exception:
                    pass
                logger.close_trade(pos.ticket, exit_price, real_pnl)
        else:
            rc = result.retcode if result else 'None'
            logger.log('ERROR', 'EmergencyKill', 'FRIDAY_PROGRESSIVE_FAILED',
                        f'Failed to close {pos.symbol} ticket {pos.ticket} rc={rc}')

    if closed and discord:
        discord.send(
            "FRIDAY SESSION CLOSE (22:00)",
            f"Closed {len(closed)} positions:\n" + ", ".join(closed),
            "orange"
        )
