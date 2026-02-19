"""
Order router — builds and sends MT5 order requests.

Extracted from execute_trade() order building logic in sovereign_bot.py.
"""
from __future__ import annotations

import os

from config.loader import cfg
from execution.spread_filter import check_spread
from execution.slippage_tracker import log_slippage
from risk.drawdown_guard import DrawdownGuard
from risk.correlation_guard import CorrelationGuard


class OrderRouter:
    """Builds and sends MT5 orders with all guardrails applied."""

    def __init__(self, logger, mt5, position_sizer, trading_schedule,
                 discord=None, ftmo=None):
        self.logger = logger
        self.mt5 = mt5
        self.position_sizer = position_sizer
        self.trading_schedule = trading_schedule
        self.discord = discord
        self.ftmo = ftmo
        self.safe_mode = False
        self.emergency_stop = False
        self.drawdown_guard = DrawdownGuard(logger, discord)
        self.correlation_guard = CorrelationGuard(logger)
        self._cached_sentiment: dict[str, float] = {}
        self.portfolio_optimizer = None  # F5: set by SovereignBot

    def execute_trade(self, symbol: str, direction: str, ml_confidence: float,
                      gpu_trading_pause: bool = False, features_dict: dict | None = None) -> bool:
        """Execute trade with full Sovereign guardrails.

        Sets self.last_reject_reason on failure (readable by callers).
        """
        self.last_reject_reason = None
        mt5 = self.mt5
        if mt5 is None:
            self.logger.log('WARNING', 'OrderRouter', 'MT5_UNAVAILABLE',
                            f'Cannot execute {symbol} {direction}: MT5 not available')
            self.last_reject_reason = 'MT5 unavailable'
            return False
        if self.safe_mode:
            self.logger.log('WARNING', 'OrderRouter', 'SAFE_MODE_ACTIVE',
                            f'Blocked {symbol} {direction}: SAFE MODE')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, 0, ml_confidence,
                                  status='BLOCKED_SAFE_MODE')
            self.last_reject_reason = 'safe mode'
            return False

        # Gebod 6: GPU overheat pause
        if gpu_trading_pause:
            self.logger.log('WARNING', 'OrderRouter', 'GPU_OVERHEAT',
                            f'Blocked {symbol} {direction}: GPU too hot, trading paused')
            self.last_reject_reason = 'GPU overheat'
            return False

        # Gebod 43+48: Daily loss limit and profit lock
        account_info = mt5.account_info()
        allowed, reason = self.drawdown_guard.check_daily_limits(account_info)
        if not allowed:
            self.logger.log('WARNING', 'OrderRouter', 'DAILY_LIMIT',
                            f'Blocked {symbol}: {reason}')
            self.last_reject_reason = f'daily limit: {reason}'
            return False

        # Gebod 58: Max 8 concurrent positions — dynamic replacement
        all_positions = mt5.positions_get()
        our_positions = [p for p in (all_positions or []) if p.magic >= 2000]

        # Anti-hedge check — allow through at high confidence for reversal
        FLIP_THRESHOLD = 0.75
        for p in our_positions:
            if p.symbol == symbol:
                p_dir = 'BUY' if p.type == 0 else 'SELL'
                if p_dir != direction:
                    if ml_confidence >= FLIP_THRESHOLD:
                        self.logger.log('INFO', 'OrderRouter', 'HEDGE_FLIP_ALLOWED',
                                        f'{symbol} {direction}: proba={ml_confidence:.3f} >= {FLIP_THRESHOLD} '
                                        f'— allowing reversal of {p_dir} position')
                        break  # let it through to _handle_existing_position()
                    else:
                        self.logger.log('WARNING', 'OrderRouter', 'HEDGE_BLOCKED',
                                        f'Blocked {symbol} {direction}: opposing position already open '
                                        f'(proba={ml_confidence:.3f} < {FLIP_THRESHOLD})')
                        self.last_reject_reason = 'hedge blocked (tegengestelde positie open)'
                        return False

        # Ultra-high confidence: bypass slot limit
        if ml_confidence > 0.80:
            self.logger.log('INFO', 'OrderRouter', 'ALPHA_OVERRIDE',
                            f'{symbol} {direction}: proba={ml_confidence:.3f} >0.80 — bypassing slot limit '
                            f'({len(our_positions)} positions open)')

        elif len(our_positions) >= 8:
            if not self._try_replace_worst(our_positions, symbol, direction, ml_confidence, mt5):
                self.last_reject_reason = 'max 8 posities bereikt'
                return False

        # Gebod 45: Correlation cap
        if not self.correlation_guard.check_usd_correlation(
                symbol, direction, our_positions, mt5):
            self.last_reject_reason = 'USD correlatie limiet'
            return False

        # Gebod 54: Drawdown recovery check
        dd_recovery = self.drawdown_guard.check_dd_recovery(account_info)

        # Hard safety: live orders require explicit opt-in
        if os.getenv("ENABLE_LIVE_TRADING", "0") != "1":
            self.logger.log('WARNING', 'OrderRouter', 'LIVE_TRADING_DISABLED',
                            f'Blocked {symbol} {direction}: set ENABLE_LIVE_TRADING=1')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, 0, ml_confidence,
                                  status='BLOCKED_LIVE_DISABLED')
            self.last_reject_reason = 'live trading disabled'
            return False

        # Guardrail 0.5: Trading hours check
        is_open, mins_left = self.trading_schedule.is_trading_open(symbol)
        if not is_open:
            self.logger.log('WARNING', 'OrderRouter', 'MARKET_CLOSED',
                            f'{symbol} {direction}: market closed right now')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, 0, ml_confidence,
                                  status='REJECTED_MARKET_CLOSED')
            self.last_reject_reason = 'markt gesloten (feestdag/buiten handelsuren)'
            return False

        # Guardrail 0.6: Friday close
        if self.trading_schedule.should_friday_close(symbol):
            self.logger.log('WARNING', 'OrderRouter', 'FRIDAY_CLOSE_SOON',
                            f'{symbol} {direction}: Friday close approaching')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, 0, ml_confidence,
                                  status='REJECTED_FRIDAY_CLOSE')
            self.last_reject_reason = 'vrijdag sluiting nadert'
            return False

        # Guardrail 1: Spread check
        spread_ok, spread_pct = check_spread(symbol, mt5, self.logger)
        if not spread_ok:
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                  status='REJECTED_SPREAD')
            self.last_reject_reason = f'spread te hoog ({spread_pct:.3f}%)'
            return False

        # Guardrail 2: Blackout period
        if self._is_blackout_period():
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                  status='REJECTED_BLACKOUT')
            self.last_reject_reason = 'blackout periode'
            return False

        # Guardrail 2.5: Sector exposure limit
        from risk.position_sizing import MAX_SECTOR_EXPOSURE
        sym_cfg = cfg.SYMBOLS.get(symbol, {})
        sector = sym_cfg.get('sector', 'unknown')
        sector_limit = MAX_SECTOR_EXPOSURE.get(sector, 0.02)
        if all_positions:
            sector_risk_live = 0.0
            for p in all_positions:
                if p.magic < 2000:
                    continue
                p_cfg = cfg.SYMBOLS.get(p.symbol, {})
                if p_cfg.get('sector', 'unknown') == sector:
                    # Don't count positions at breakeven — their effective risk is ~0
                    if self._is_position_at_breakeven(p, mt5):
                        continue
                    sector_risk_live += p_cfg.get('risk_per_trade', 0.003)
            new_risk = sym_cfg.get('risk_per_trade', 0.003)
            if sector_risk_live + new_risk > sector_limit + 0.0001:
                self.logger.log('WARNING', 'OrderRouter', 'SECTOR_LIMIT',
                                f'{symbol} ({sector}): would exceed sector limit '
                                f'{sector_risk_live:.2%}+{new_risk:.2%} > {sector_limit:.2%}')
                self.logger.log_trade(symbol, direction, 0, 0, 0, 0, 0, ml_confidence,
                                      status='REJECTED_SECTOR_LIMIT')
                self.last_reject_reason = f'sector limiet ({sector})'
                return False

        # Guardrail 3: Anti-hedging with signal reversal close + pyramid check
        positions = mt5.positions_get(symbol=symbol)
        self._is_pyramid = False  # reset flag for position sizing
        if positions and len(positions) > 0:
            # Max 1 pyramid per symbol — block if already 2+ positions
            our_sym_positions = [p for p in positions if p.magic >= 2000]
            if len(our_sym_positions) >= 2:
                self.logger.log('INFO', 'OrderRouter', 'PYRAMID_MAX',
                                f'{symbol}: already {len(our_sym_positions)} positions, no more pyramiding')
                self.last_reject_reason = f'max pyramid bereikt op {symbol}'
                return False
            result = self._handle_existing_position(our_sym_positions[0] if our_sym_positions else positions[0],
                                                     symbol, direction, ml_confidence, spread_pct, mt5)
            if not result:
                return False

        # Get account info and tick
        account_info = mt5.account_info()
        if not account_info:
            self.logger.log('ERROR', 'OrderRouter', 'ACCOUNT_INFO_FAILED',
                            'Could not get account info')
            return False

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            self.logger.log('ERROR', 'OrderRouter', 'TICK_FAILED',
                            f'Could not get tick for {symbol}')
            return False

        # Per-symbol config
        sym_cfg = cfg.SYMBOLS.get(symbol, {})
        sl_mult = sym_cfg.get('atr_sl_mult', 1.5)
        tp_mult = sym_cfg.get('atr_tp_mult', 4.5)
        magic = sym_cfg.get('magic_number', 2000)

        # Calculate ATR
        atr = self._calculate_atr(symbol, mt5, sym_cfg)
        if atr is None:
            self.logger.log('ERROR', 'OrderRouter', 'ATR_FAILED',
                            f'{symbol}: cannot calculate ATR, aborting')
            return False

        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        # ── F8: Confidence-scaled TP ─────────────────────────────────
        # 0.55→1.0×, 0.75→1.36×, 0.85→1.55×, capped at 2.0×
        tp_confidence_factor = max(1.0, min(2.0, ml_confidence / 0.55))
        tp_distance *= tp_confidence_factor

        # ── Cost-aware TP floor ──────────────────────────────────────
        # 1) Spread cost (round-trip: pay on entry, pay on exit)
        spread_price = tick.ask - tick.bid

        # 2) Swap cost — only penalise negative swap, positive swap = benefit
        sym_info_full = mt5.symbol_info(symbol)
        swap_cost_price = 0.0
        swap_benefit_price = 0.0
        estimated_hold_nights = sym_cfg.get('estimated_hold_nights', 2.0)
        if sym_info_full:
            point = sym_info_full.point
            swap_pts = (sym_info_full.swap_long if direction == 'BUY'
                        else sym_info_full.swap_short)
            if swap_pts < 0:
                # Wednesday (weekday=2) or Thursday (weekday=3) → broker charges 3× swap
                from datetime import datetime, timezone
                wd = datetime.now(timezone.utc).weekday()
                swap_mult = 3.0 if wd in (2, 3) else 1.0
                swap_cost_price = abs(swap_pts * point) * swap_mult * estimated_hold_nights
            # F6: positive swap → reduce effective cost (floor at spread so costs never negative)
            swap_benefit_price = 0.0
            if swap_pts > 0:
                from datetime import datetime, timezone
                wd = datetime.now(timezone.utc).weekday()
                swap_mult = 3.0 if wd in (2, 3) else 1.0
                swap_benefit_price = swap_pts * point * swap_mult * estimated_hold_nights

        # 3) Commission estimate (round-trip: entry + exit)
        asset_class = sym_cfg.get('asset_class', 'forex')
        commission_map = {'crypto': 0.0005, 'forex': 0.0002,
                          'commodity': 0.0003, 'index': 0.0002, 'equity': 0.0003}
        commission_pct = sym_cfg.get('commission_pct',
                                     commission_map.get(asset_class, 0.0003))
        entry_price = tick.ask if direction == 'BUY' else tick.bid
        commission_price = entry_price * commission_pct * 2  # round-trip

        # 4) Slippage estimate (entry + exit)
        slippage_map = {'crypto': 0.0005, 'forex': 0.0002,
                        'commodity': 0.0003, 'index': 0.0002, 'equity': 0.0003}
        slippage_pct = sym_cfg.get('slippage_estimate_pct',
                                   slippage_map.get(asset_class, 0.0003))
        slippage_price = entry_price * slippage_pct * 2  # round-trip

        # Total cost in price units (F6: swap benefit reduces cost, floor at spread)
        total_cost = max(spread_price,
                         spread_price + swap_cost_price + commission_price
                         + slippage_price - swap_benefit_price)

        # Minimum TP must be 3× total costs to be worth the trade
        MIN_COST_RATIO = sym_cfg.get('min_cost_ratio', 3.0)
        min_tp = total_cost * MIN_COST_RATIO
        if tp_distance < min_tp:
            cost_pct = (total_cost / tp_distance * 100) if tp_distance > 0 else 999
            self.logger.log('WARNING', 'OrderRouter', 'TP_TOO_SMALL_FOR_COSTS',
                            f'{symbol} {direction}: TP={tp_distance:.5f} < {MIN_COST_RATIO}× costs '
                            f'(spread={spread_price:.5f} swap={swap_cost_price:.5f} '
                            f'comm={commission_price:.5f} slip={slippage_price:.5f} '
                            f'total={total_cost:.5f}, cost={cost_pct:.1f}% of TP)')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                  status='REJECTED_COST_RATIO')
            return False

        # Shift TP out by total entry costs so realized profit at TP is net-positive
        tp_distance += spread_price + (commission_price / 2) + (slippage_price / 2)
        # ─────────────────────────────────────────────────────────────

        # Position sizing — Kelly with config risk as absolute floor
        config_risk = sym_cfg.get('risk_per_trade', 0.003)
        risk_pct = max(self.position_sizer.kelly_risk_pct(symbol), config_risk)

        # ── F4: Confidence-scaled sizing ─────────────────────────────
        # 0.55→0.5×, 0.70→1.0×, 0.85→1.5×, capped at 2.0×
        conf_multiplier = max(0.5, min(2.0, 0.5 + (ml_confidence - 0.55) * 3.33))
        risk_pct *= conf_multiplier

        # ── F3: RL position sizing ──────────────────────────────────
        self._last_rl_arm = None
        self._last_risk_pct = risk_pct
        regime = int(features_dict.get('regime', 0)) if features_dict else 0
        volatility = float(features_dict.get('vol20', 0.0)) if features_dict else 0.0
        initial_bal = getattr(account_info, 'balance', cfg.ACCOUNT_SIZE) or cfg.ACCOUNT_SIZE
        current_dd = (initial_bal - account_info.equity) / initial_bal if initial_bal > 0 else 0.0
        risk_pct, rl_arm = self.position_sizer.rl_adjust_risk(
            risk_pct, ml_confidence, regime, volatility, max(current_dd, 0.0)
        )
        self._last_rl_arm = rl_arm
        self._last_risk_pct = risk_pct

        # ── F5: Portfolio optimization weight ──────────────────────────
        if self.portfolio_optimizer is not None:
            try:
                active_symbols = [s for s in cfg.SYMBOLS if s != symbol]
                active_symbols.append(symbol)
                risk_pct = self.portfolio_optimizer.risk_budget(
                    symbol, risk_pct, active_symbols
                )
            except Exception:
                pass  # Graceful fallback

        # ── Absolute floor: config risk is the minimum after ALL adjustments ──
        risk_pct = max(risk_pct, config_risk)

        from execution.broker_api import get_symbol_info
        sym_info = get_symbol_info(symbol)
        lot_size = self.position_sizer.calculate_lot_size(
            symbol, account_info.equity, risk_pct, sl_distance, sym_info
        )

        # Gebod 54: Drawdown recovery — halve lots
        if dd_recovery:
            lot_size *= 0.5
            # Re-round to volume step after halving
            if sym_info and sym_info.volume_step > 0:
                lot_size = round(lot_size / sym_info.volume_step) * sym_info.volume_step
                lot_size = round(lot_size, 8)  # float precision cleanup
            self.logger.log('WARNING', 'OrderRouter', 'DD_RECOVERY',
                            f'{symbol}: lot halved ({lot_size:.2f}) — drawdown recovery mode')

        # ── Margin clamp: ensure lot_size fits in available margin ──
        if sym_info:
            contract_size = sym_info.get("trade_contract_size", 1.0)
            vol_step = sym_info.get("volume_step", 0.01)
            vol_min = sym_info.get("volume_min", 0.01)
            margin_per_lot = entry_price * contract_size  # notional per lot
            if margin_per_lot > 0:
                effective_leverage = self._estimate_leverage(
                    account_info, mt5, asset_class)
                margin_needed = (lot_size * margin_per_lot) / effective_leverage
                max_margin = account_info.margin_free * 0.80  # 80% safety buffer
                if margin_needed > max_margin > 0:
                    old_lots = lot_size
                    lot_size = (max_margin * effective_leverage) / margin_per_lot
                    # Round to volume_step
                    if vol_step > 0:
                        lot_size = round(lot_size / vol_step) * vol_step
                        lot_size = round(lot_size, 8)
                    lot_size = max(vol_min, lot_size)
                    self.logger.log('WARNING', 'OrderRouter', 'MARGIN_CLAMP',
                        f'{symbol}: lots {old_lots:.0f} → {lot_size:.0f} '
                        f'(margin_free=${account_info.margin_free:.0f}, '
                        f'eff_leverage={effective_leverage:.1f}x)')

        # Pyramid: full risk — existing position is at breakeven so real exposure = only this trade
        if getattr(self, '_is_pyramid', False):
            self.logger.log('INFO', 'OrderRouter', 'PYRAMID_FULL_RISK',
                            f'{symbol}: pyramid position — full risk {lot_size:.2f} lots '
                            f'(existing at BE, effective risk = 1× trade)')

        # Sentiment is now applied pre-threshold in engine/signal.py

        # Calculate entry, TP, SL
        if direction == 'BUY':
            entry_price = tick.ask
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry_price = tick.bid
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            order_type = mt5.ORDER_TYPE_SELL

        self.logger.log('INFO', 'OrderRouter', 'ATR_SLTP',
                        f'{symbol} ATR:{atr:.5f} SL:{sl_distance:.5f} TP:{tp_distance:.5f} '
                        f'Lots:{lot_size:.2f} Risk:{risk_pct:.3%} '
                        f'ConfMult:{conf_multiplier:.2f} TPfact:{tp_confidence_factor:.2f}')

        # Build and send order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": cfg.DEVIATION,
            "magic": magic,
            "comment": f"Sovereign_Pyr_{symbol}" if getattr(self, '_is_pyramid', False) else f"Sovereign_{symbol}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        self.logger.log('INFO', 'OrderRouter', 'ORDER_SEND',
                        f'{symbol} {direction} {lot_size} lots @ {entry_price:.5f} '
                        f'TP:{tp_price:.5f} SL:{sl_price:.5f}')

        result = mt5.order_send(request)

        if result is None:
            self.logger.log('ERROR', 'OrderRouter', 'ORDER_FAILED',
                            'order_send() returned None')
            self.logger.log_trade(symbol, direction, entry_price, tp_price, sl_price,
                                  lot_size, spread_pct, ml_confidence, status='FAILED_NONE')
            self.last_reject_reason = 'order_send failed (None)'
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.log('ERROR', 'OrderRouter', 'ORDER_REJECTED',
                            f'Rejected: {result.retcode} - {result.comment}')
            self.logger.log_trade(symbol, direction, entry_price, tp_price, sl_price,
                                  lot_size, spread_pct, ml_confidence,
                                  status=f'REJECTED_{result.retcode}')
            self.last_reject_reason = f'broker rejected: {result.comment}'
            return False

        # Success — log with slippage
        fill_price = result.price if hasattr(result, 'price') and result.price > 0 else entry_price
        slippage_bps = log_slippage(self.logger, symbol, direction, entry_price,
                                     fill_price, lot_size)

        self.logger.log('SUCCESS', 'OrderRouter', 'ORDER_FILLED',
                        f'Ticket {result.order} | {symbol} {direction} {lot_size} lots '
                        f'| req={entry_price:.5f} fill={fill_price:.5f} slip={slippage_bps:.1f}bps')
        self.logger.log_trade(symbol, direction, fill_price, tp_price, sl_price,
                              lot_size, spread_pct, ml_confidence,
                              ticket=result.order, status='FILLED')

        if self.discord:
            self.discord.trade_entry(
                symbol=symbol, direction=direction, entry_price=fill_price,
                lot_size=lot_size, tp=tp_price, sl=sl_price,
                ticket=result.order, confidence=ml_confidence
            )

        if self.ftmo:
            self.ftmo.increment_trade_count()

        return True

    def _is_blackout_period(self) -> bool:
        now = self.trading_schedule._now_gmt2()
        current_min = now.hour * 60 + now.minute
        if current_min >= 23 * 60 + 50 or current_min <= 10:
            self.logger.log('INFO', 'OrderRouter', 'BLACKOUT_ROLLOVER',
                            'In rollover blackout period (23:50-00:10 GMT+2)')
            return True
        return False

    def _calculate_atr(self, symbol: str, mt5, sym_cfg: dict) -> float | None:
        period = sym_cfg.get('atr_period', 14)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            return None
        tr_values = []
        for i in range(1, len(rates)):
            h, l, pc = rates[i]['high'], rates[i]['low'], rates[i - 1]['close']
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_values.append(tr)
        return sum(tr_values[-period:]) / period

    def _try_replace_worst(self, our_positions, symbol, direction, ml_confidence, mt5) -> bool:
        """Try to replace worst position. Returns True if slot available."""
        worst_pos = None
        worst_score = float('inf')
        for p in our_positions:
            score = p.profit + p.swap
            if score < worst_score:
                worst_score = score
                worst_pos = p

        if worst_pos and ml_confidence > 0.70 and worst_score < 0:
            tick = mt5.symbol_info_tick(worst_pos.symbol)
            if tick:
                close_price = tick.bid if worst_pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                close_type = mt5.ORDER_TYPE_SELL if worst_pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                w_dir = "BUY" if worst_pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": worst_pos.symbol,
                    "volume": worst_pos.volume,
                    "type": close_type,
                    "position": worst_pos.ticket,
                    "price": close_price,
                    "deviation": cfg.DEVIATION,
                    "magic": worst_pos.magic,
                    "comment": "Sovereign_Replaced",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.log('INFO', 'OrderRouter', 'POSITION_REPLACED',
                                    f'Closed {worst_pos.symbol} {w_dir} (PnL=${worst_score:+.2f}) '
                                    f'to make room for {symbol} {direction}')
                    if self.discord:
                        self.discord.send(
                            f"SLOT REPLACE: {worst_pos.symbol} → {symbol}",
                            f"Closed {worst_pos.symbol} {w_dir} (PnL: ${worst_score:+.2f})\n"
                            f"Opening {symbol} {direction} (conf: {ml_confidence:.1%})",
                            "orange",
                        )
                    return True
                else:
                    self.logger.log('WARNING', 'OrderRouter', 'REPLACE_FAILED',
                                    f'Could not close {worst_pos.symbol} for replacement')
                    return False
        else:
            self.logger.log('WARNING', 'OrderRouter', 'MAX_POSITIONS',
                            f'Blocked {symbol} (proba={ml_confidence:.3f}): '
                            f'{len(our_positions)} positions, worst PnL=${worst_score:+.2f}')
            return False

    def _handle_existing_position(self, pos, symbol, direction, ml_confidence, spread_pct, mt5) -> bool:
        """Handle existing position on same symbol. Returns True if should continue to open new."""
        pos_dir = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'

        if pos_dir == direction:
            # ── Pyramid: allow 2nd position if existing is at breakeven + passes normal entry threshold ──
            # Logic: if a signal is good enough for a fresh entry, it's good enough for a pyramid
            # when the existing position is already risk-free (at breakeven).
            # Uses per-symbol threshold from sovereign_configs (same as fresh entry).
            # NOTE: cfg already imported at module level (line 10) — do NOT re-import locally
            # as it causes UnboundLocalError when hedge-flip path uses cfg.DEVIATION before this line.
            pyramid_threshold = cfg.SYMBOLS.get(symbol, {}).get("prob_threshold", cfg.ML_THRESHOLD)
            if ml_confidence >= pyramid_threshold and self._is_position_at_breakeven(pos, mt5):
                self._is_pyramid = True
                self.logger.log('INFO', 'OrderRouter', 'PYRAMID_ALLOWED',
                                f'{symbol}: existing {pos_dir} at breakeven, '
                                f'proba={ml_confidence:.3f} >= {pyramid_threshold} — opening 2nd position')
                if self.discord:
                    self.discord.send(
                        f"PYRAMID: {symbol} {direction}",
                        f"Adding 2nd {direction} position (confidence: {ml_confidence:.0%})\n"
                        f"Existing position at breakeven — real risk only on new trade",
                        "blue",
                    )
                return True  # Continue to open pyramid position
            self.logger.log('INFO', 'OrderRouter', 'ALREADY_IN_MARKET',
                            f'{symbol}: already {pos_dir}, skipping'
                            + (f' (conf={ml_confidence:.2f} < {pyramid_threshold})'
                               if ml_confidence < pyramid_threshold
                               else ' (existing not at breakeven)'))
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                  status='REJECTED_SAME_DIR')
            self.last_reject_reason = f'al {pos_dir} positie open'
            return False

        FLIP_THRESHOLD = 0.75
        if ml_confidence >= FLIP_THRESHOLD:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": cfg.DEVIATION,
                    "magic": pos.magic,
                    "comment": "Sovereign_FlipClose",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    pnl = pos.profit + pos.swap
                    self.logger.log('INFO', 'OrderRouter', 'FLIP_CLOSE',
                                    f'{symbol}: closed {pos_dir} ticket {pos.ticket} '
                                    f'(P&L: ${pnl:+.2f}) → flipping to {direction} '
                                    f'(proba={ml_confidence:.3f})')
                    if self.discord:
                        color = "green" if pnl > 0 else "red"
                        self.discord.send(
                            f"FLIP: {symbol} {pos_dir} → {direction}",
                            f"Closed {pos_dir} (P&L: ${pnl:+.2f})\n"
                            f"Opening {direction} (confidence: {ml_confidence:.0%})",
                            color,
                        )
                    return True  # Continue to open new trade in opposite direction
                else:
                    rc = result.retcode if result else 'None'
                    self.logger.log('ERROR', 'OrderRouter', 'FLIP_CLOSE_FAILED',
                                    f'{symbol}: failed to close ticket {pos.ticket} (rc={rc})')
                    return False
        else:
            self.logger.log('INFO', 'OrderRouter', 'WEAK_REVERSAL',
                            f'{symbol}: opposing signal {direction} but conf={ml_confidence:.2f} < 0.75, '
                            f'keeping {pos_dir}')
            self.logger.log_trade(symbol, direction, 0, 0, 0, 0, spread_pct, ml_confidence,
                                  status='REJECTED_WEAK_REVERSAL')
            return False

    @staticmethod
    def _is_position_at_breakeven(pos, mt5) -> bool:
        """Check if position's SL has been moved to breakeven or better."""
        if pos.sl == 0:
            return False  # No SL set, not at breakeven
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        if is_buy:
            # BUY: breakeven means SL >= entry price
            return pos.sl >= pos.price_open
        else:
            # SELL: breakeven means SL <= entry price
            return pos.sl <= pos.price_open

    def _apply_sentiment(self, symbol, direction, lot_size):
        """Deprecated — sentiment is now applied pre-threshold in engine/signal.py."""
        pass

    def _estimate_leverage(self, account_info, mt5, asset_class: str = 'equity') -> float:
        """Estimate effective leverage from open positions, or fallback per asset class.

        If there are open positions with margin used, calculates:
            sum(notional) / account.margin = effective leverage
        Otherwise falls back to conservative per-asset-class defaults from config.
        """
        try:
            if account_info.margin > 0:
                positions = mt5.positions_get()
                if positions:
                    total_notional = 0.0
                    for p in positions:
                        si = mt5.symbol_info(p.symbol)
                        if si:
                            total_notional += (p.volume * si.trade_contract_size
                                               * p.price_current)
                    if total_notional > 0:
                        return total_notional / account_info.margin
        except Exception:
            pass
        # Fallback: conservative leverage per asset class from config
        defaults = {"equity": 3.5, "forex": 30, "index": 10,
                    "commodity": 10, "crypto": 2}
        leverage_map = cfg.SYMBOLS.get("margin_leverage", defaults)
        return leverage_map.get(asset_class, 3.5)
