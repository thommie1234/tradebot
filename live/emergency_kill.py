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
    """F9: Progressive Friday close — close positions in stages before weekend.

    21:00 GMT+2 Friday: close all non-crypto LOSING positions.
    22:00 GMT+2 Friday: close ALL non-crypto positions (winners too).
    Existing friday_auto_close() (~10 min before market close) remains as final safety net.
    """
    if not running or emergency_stop or mt5 is None:
        return

    from risk.ftmo_guard import TradingSchedule
    now = datetime.now(timezone.utc) + TradingSchedule.GMT2_OFFSET
    if now.weekday() != 4:
        return

    hour = now.hour
    if hour < 21:
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

        # 21:00-21:59: only close losing positions
        if hour == 21 and pnl >= 0:
            continue

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
            phase = "LOSERS" if hour == 21 else "ALL"
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
        else:
            rc = result.retcode if result else 'None'
            logger.log('ERROR', 'EmergencyKill', 'FRIDAY_PROGRESSIVE_FAILED',
                        f'Failed to close {pos.symbol} ticket {pos.ticket} rc={rc}')

    if closed and discord:
        phase = "losers" if hour == 21 else "all non-crypto"
        discord.send(
            f"FRIDAY PROGRESSIVE CLOSE ({phase})",
            f"Closed {len(closed)} positions:\n" + ", ".join(closed),
            "orange"
        )
