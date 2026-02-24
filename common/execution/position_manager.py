"""
Position manager — trailing stop, breakeven, zombie/bleeder cleanup,
and ML-based exit signal.

Extracted from manage_positions() + _process_single_position() +
auto_close_bleeders() in sovereign_bot.py.

Continuous monitoring: start_monitor() spawns a background thread that
polls tick prices every ~500ms and adjusts trailing stops in near-realtime
(vs the main loop's 5s interval).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from config.loader import cfg

# ML exit settings
ML_EXIT_THRESHOLD = 0.42      # Below this proba → count as exit signal
ML_EXIT_STRIKES = 3           # N consecutive exit signals → close
ML_EXIT_COOLDOWN_S = 300      # Check every 5 minutes (not every 60s loop)
ML_EXIT_MIN_HOLD_BARS = 2     # Don't exit within first 2 bars (give trade room)

# Session close — close positions before daily market close
SESSION_CLOSE_BUFFER_MIN = 2  # Close positions 2 minutes before session ends


class PositionManager:
    """Active trade management: trailing stops, breakeven, auto-close."""

    @staticmethod
    def _friday_trail_factor() -> float | None:
        """After 20:00 GMT+2 on Friday: return 0.67 (tighter trail)."""
        from risk.ftmo_guard import TradingSchedule
        now = datetime.now(timezone.utc) + TradingSchedule.GMT2_OFFSET
        if now.weekday() == 4 and now.hour >= 20:
            return 0.67
        return None

    def _session_trail_factor(self, symbol: str) -> float | None:
        """Tighten trailing stop as daily session close approaches.

        Returns a multiplier < 1.0 to reduce trail activation and distance,
        or None if not near session close (or crypto).

        Progressive tightening:
          > 60 min  → None (normal trailing)
          30-60 min → 0.6  (activate earlier, trail tighter)
          15-30 min → 0.4  (aggressive profit lock)
          5-15 min  → 0.25 (very tight, almost at session close)
        """
        if not hasattr(self, '_trading_schedule') or self._trading_schedule is None:
            return None

        if self._trading_schedule.is_crypto(symbol):
            return None

        is_open, minutes_left = self._trading_schedule.is_trading_open(symbol)
        if not is_open or minutes_left is None:
            return None

        if minutes_left > 60:
            return None
        elif minutes_left > 30:
            return 0.6
        elif minutes_left > 15:
            return 0.4
        elif minutes_left > SESSION_CLOSE_BUFFER_MIN:
            return 0.25

        return None  # Within SESSION_CLOSE_BUFFER_MIN — session_close_check handles this

    def __init__(self, logger, mt5, discord=None, account_symbols: dict | None = None,
                 account_name: str = "default"):
        self.logger = logger
        self.mt5 = mt5
        self.discord = discord
        self._account_symbols = account_symbols
        self.account_name = account_name
        self._sym_cfg_cache: dict[str, dict] = {}
        self._atr_cache = {}
        self._trading_schedule = None  # Set by run_bot after init
        self._ftmo = None  # Set by run_bot after init — for SL modification throttle
        # ML exit tracking: ticket → {strikes, last_check, last_proba}
        self._ml_exit_strikes: dict[int, dict] = {}
        self._ml_exit_last_run: float = 0

        # Continuous monitoring thread
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._monitor_interval: float = 0.5  # seconds between cycles
        self._monitor_stats = {"cycles": 0, "sl_moves": 0, "last_cycle_ms": 0.0}

    def _sym_cfg(self, symbol: str) -> dict:
        """Get per-symbol config, checking account-specific symbols first."""
        if self._account_symbols:
            result = self._account_symbols.get(symbol)
            if result is not None:
                return result
        return cfg.SYMBOLS.get(symbol, {})

    # ── Continuous trailing-stop monitor ────────────────────────────

    def start_monitor(self, interval: float = 0.5):
        """Start background thread that polls ticks and adjusts trailing stops.

        Args:
            interval: seconds between polling cycles (default 0.5s).
                      With a ~3ms bridge call and 5 positions, one cycle
                      takes ~15-20ms, so 0.5s gives plenty of headroom.
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return  # Already running

        self._monitor_interval = interval
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True,
            name="TrailingStopMonitor",
        )
        self._monitor_thread.start()
        self.logger.log('INFO', 'PositionManager', 'MONITOR_STARTED',
                        f'Continuous trailing-stop monitor started '
                        f'(interval={interval}s)')

    def stop_monitor(self):
        """Stop the continuous monitoring thread."""
        self._monitor_stop.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        self.logger.log('INFO', 'PositionManager', 'MONITOR_STOPPED',
                        'Continuous trailing-stop monitor stopped')

    def _monitor_loop(self):
        """Background loop: continuously check trailing stops for all positions.

        Runs until stop_monitor() is called.  Each cycle:
          1. positions_get() — one bridge call
          2. For each sovereign position (magic >= 2000):
             - Skip positions clearly in loss (saves bridge calls)
             - Run _process_single_position() (tick fetch + SL logic)
          3. Wait for interval (interruptible via Event)
        """
        self.logger.log('DEBUG', 'PositionManager', 'MONITOR_THREAD',
                        'Monitor thread started')
        while not self._monitor_stop.is_set():
            t0 = time.monotonic()
            try:
                if self.mt5 is None:
                    self._monitor_stop.wait(timeout=5)
                    continue

                positions = self.mt5.positions_get()
                if positions:
                    for pos in positions:
                        if self._monitor_stop.is_set():
                            return
                        if pos.magic < 2000:
                            continue
                        try:
                            self._process_single_position(pos)
                        except Exception as e:
                            self.logger.log('ERROR', 'PositionManager',
                                            'MONITOR_POS_ERROR',
                                            f'{pos.symbol} ticket={pos.ticket}: {e}')

            except Exception as e:
                self.logger.log('ERROR', 'PositionManager', 'MONITOR_LOOP_ERROR',
                                str(e))

            elapsed_ms = (time.monotonic() - t0) * 1000
            self._monitor_stats["cycles"] += 1
            self._monitor_stats["last_cycle_ms"] = elapsed_ms

            # Wait for next cycle (interruptible)
            self._monitor_stop.wait(timeout=self._monitor_interval)

    def _deal_pnl(self, ticket: int) -> float | None:
        """Get realized P&L from deal history (profit + commission + swap)."""
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=7)
            deals = self.mt5.history_deals_get(start, now, position=ticket)
            if not deals:
                return None
            total = sum(d.profit + d.commission + d.swap
                        for d in deals if d.position_id == ticket)
            return total
        except Exception:
            return None

    def manage_positions(self, running: bool, emergency_stop: bool):
        """Active trade management — trailing stop + breakeven."""
        if not running or emergency_stop or self.mt5 is None:
            return

        positions = self.mt5.positions_get()
        if positions:
            for pos in positions:
                if pos.magic < 2000:
                    continue
                try:
                    self._process_single_position(pos)
                except Exception as e:
                    self.logger.log('ERROR', 'PositionManager', 'MANAGE_ERROR',
                                    f'{pos.symbol} ticket={pos.ticket}: {e}')

    def _get_cached_atr(self, symbol: str, period: int = 14) -> float | None:
        cache_key = f"atr_{symbol}_{period}"
        now = time.time()
        if cache_key in self._atr_cache:
            val, ts = self._atr_cache[cache_key]
            if now - ts < 3600:
                return val
        try:
            bars = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_H1, 0, period + 5)
            if bars is None or len(bars) < period + 1:
                return None
            tr_list = []
            for i in range(1, len(bars)):
                hl = bars[i]['high'] - bars[i]['low']
                hpc = abs(bars[i]['high'] - bars[i - 1]['close'])
                lpc = abs(bars[i]['low'] - bars[i - 1]['close'])
                tr_list.append(max(hl, hpc, lpc))
            atr = sum(tr_list[-period:]) / period
            self._atr_cache[cache_key] = (atr, now)
            return atr
        except Exception:
            return None

    def _estimate_costs(self, pos, sym_cfg, atr):
        """Estimate total round-trip costs in price units for a position."""
        sym_info = self.mt5.symbol_info(pos.symbol)
        if not sym_info or sym_info.ask <= 0:
            return 0.2 * atr  # conservative fallback

        spread = sym_info.ask - sym_info.bid
        entry_price = pos.price_open

        # Commission round-trip
        asset_class = sym_cfg.get('asset_class', 'forex')
        comm_map = {'crypto': 0.0005, 'forex': 0.0002,
                    'commodity': 0.0003, 'index': 0.0002, 'equity': 0.0003}
        comm_pct = sym_cfg.get('commission_pct', comm_map.get(asset_class, 0.0003))
        commission = entry_price * comm_pct * 2

        # Slippage round-trip
        slip_map = {'crypto': 0.0005, 'forex': 0.0002,
                    'commodity': 0.0003, 'index': 0.0002, 'equity': 0.0003}
        slip_pct = sym_cfg.get('slippage_estimate_pct', slip_map.get(asset_class, 0.0003))
        slippage = entry_price * slip_pct * 2

        return spread + commission + slippage

    # Asset-class defaults for trailing/breakeven when not in config
    _TRAIL_DEFAULTS = {
        'crypto':    {'breakeven_atr': 1.0, 'trail_activation_atr': 3.0, 'trail_distance_atr': 1.5},
        'forex':     {'breakeven_atr': 0.5, 'trail_activation_atr': 1.5, 'trail_distance_atr': 0.8},
        'commodity': {'breakeven_atr': 0.5, 'trail_activation_atr': 1.5, 'trail_distance_atr': 0.8},
        'index':     {'breakeven_atr': 0.5, 'trail_activation_atr': 1.5, 'trail_distance_atr': 0.8},
        'equity':    {'breakeven_atr': 0.2, 'trail_activation_atr': 0.8, 'trail_distance_atr': 0.4},
    }

    def _process_single_position(self, pos):
        """ATR-based trailing stop + breakeven management."""
        sym_cfg = self._sym_cfg(pos.symbol)
        atr = self._get_cached_atr(pos.symbol)
        if atr is None or atr <= 0:
            return

        # Estimate costs so trailing/breakeven locks in real profit
        total_cost = self._estimate_costs(pos, sym_cfg, atr)

        # Asset-class-aware defaults
        asset_class = sym_cfg.get('asset_class', 'forex')
        defaults = self._TRAIL_DEFAULTS.get(asset_class, self._TRAIL_DEFAULTS['forex'])

        be_trigger = sym_cfg.get('breakeven_atr', defaults['breakeven_atr']) * atr
        # Ensure breakeven trigger is high enough to cover costs + buffer
        be_trigger = max(be_trigger, total_cost * 2.0)

        trail_activation = sym_cfg.get('trail_activation_atr', defaults['trail_activation_atr']) * atr
        trail_distance_base = sym_cfg.get('trail_distance_atr', defaults['trail_distance_atr']) * atr

        # F6: Positive swap → trail activation 30% higher (hold longer for carry)
        sym_info_swap = self.mt5.symbol_info(pos.symbol)
        if sym_info_swap:
            is_buy_swap = pos.type == self.mt5.ORDER_TYPE_BUY
            swap_pts = sym_info_swap.swap_long if is_buy_swap else sym_info_swap.swap_short
            if swap_pts > 0:
                trail_activation *= 1.3

        # F9: Friday evening → tighter trail to protect against weekend gaps
        # Crypto is 24/7 so less aggressive tightening
        friday_factor = self._friday_trail_factor()
        if friday_factor is not None:
            if asset_class == 'crypto':
                friday_factor = 0.85  # Gentler for crypto (was 0.67)
            trail_activation *= friday_factor
            trail_distance_base *= friday_factor

        # Session close approach → progressively tighter trailing
        session_factor = self._session_trail_factor(pos.symbol)
        if session_factor is not None:
            trail_activation *= session_factor
            trail_distance_base *= session_factor

        # Ensure trailing locks at least costs + buffer when it fires
        min_locked_profit = total_cost * 1.5
        if trail_activation - trail_distance_base < min_locked_profit:
            trail_distance_base = trail_activation - min_locked_profit
        # Floor: trail distance can never go negative (would place SL on wrong side)
        trail_distance_base = max(trail_distance_base, atr * 0.05)

        tick = self.mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return

        is_buy = pos.type == self.mt5.ORDER_TYPE_BUY
        current_price = tick.bid if is_buy else tick.ask
        entry = pos.price_open

        price_profit = (current_price - entry) if is_buy else (entry - current_price)

        if price_profit <= 0:
            return

        # Progressive trail: aggressively tighten as profit grows beyond activation.
        # Tiers: (ATR profit beyond activation, fraction of base distance)
        # E.g. with base 0.5 ATR and activation at 1.2 ATR:
        #   1.2 ATR profit → 0.50 ATR trail (full base)
        #   1.5 ATR profit → 0.40 ATR trail
        #   2.0 ATR profit → 0.30 ATR trail
        #   2.5 ATR profit → 0.20 ATR trail
        #   3.0+ ATR profit → 0.15 ATR trail
        _PROG_TIERS = [
            (0.0, 1.00),   # at activation: full base distance
            (0.3, 0.80),   # 0.3 ATR beyond activation
            (0.8, 0.60),
            (1.3, 0.40),
            (1.8, 0.30),   # 1.8+ ATR beyond: floor at 30% of base
        ]
        trail_distance = trail_distance_base
        if price_profit >= trail_activation:
            profit_beyond = (price_profit - trail_activation) / atr
            factor = _PROG_TIERS[-1][1]  # default to floor
            for i in range(len(_PROG_TIERS) - 1):
                p0, f0 = _PROG_TIERS[i]
                p1, f1 = _PROG_TIERS[i + 1]
                if profit_beyond <= p1:
                    t = (profit_beyond - p0) / (p1 - p0) if p1 > p0 else 0
                    factor = f0 + t * (f1 - f0)
                    break
            trail_distance = trail_distance_base * factor

        current_sl = pos.sl
        new_sl = None

        # Phase 2: Trailing stop
        if price_profit >= trail_activation:
            if is_buy:
                candidate_sl = current_price - trail_distance
                if candidate_sl > current_sl:
                    new_sl = candidate_sl
            else:
                candidate_sl = current_price + trail_distance
                if current_sl == 0 or candidate_sl < current_sl:
                    new_sl = candidate_sl

        # Phase 1: Breakeven — SL must cover all costs so we don't
        # lock in a loss when "breaking even"
        elif price_profit >= be_trigger:
            # BE SL = entry + total costs + small ATR buffer
            cost_buf = total_cost + 0.05 * atr
            be_sl = entry + cost_buf if is_buy else entry - cost_buf
            if is_buy and be_sl > current_sl:
                new_sl = be_sl
            elif not is_buy and (current_sl == 0 or be_sl < current_sl):
                new_sl = be_sl

        if new_sl is None:
            return

        sym_info = self.mt5.symbol_info(pos.symbol)
        if sym_info:
            digits = sym_info.digits
            new_sl = round(new_sl, digits)

        if abs(new_sl - current_sl) < (atr * 0.01):
            return

        # FTMO SL modification throttle
        if self._ftmo and not self._ftmo.check_sl_modify_allowed(pos.ticket):
            return

        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": pos.tp,
        }
        result = self.mt5.order_send(request)
        if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
            if self._ftmo:
                self._ftmo.record_sl_modify(pos.ticket)
            self._monitor_stats["sl_moves"] += 1
            mode = "TRAILING" if price_profit >= trail_activation else "BREAKEVEN"
            self.logger.log('INFO', 'TrailingStop', f'SL_{mode}',
                            f'{pos.symbol} SL: {current_sl:.5f} → {new_sl:.5f} '
                            f'(profit={price_profit:.5f}, ATR={atr:.5f})')
            if self.discord and mode == "TRAILING":
                # Only notify Discord when SL moved significantly (>= 0.5× ATR since last notify)
                last_notified = getattr(self, '_trail_last_notified', {})
                prev_sl = last_notified.get(pos.ticket, 0)
                if abs(new_sl - prev_sl) >= atr * 0.5:
                    direction = "BUY" if is_buy else "SELL"
                    acct_tag = f" [{self.account_name}]" if self.account_name != "default" else ""
                    self.discord.send(
                        f"TRAIL{acct_tag}: {pos.symbol}",
                        f"{direction} SL → {new_sl:.5f}\nProfit: {price_profit/atr:.1f}×ATR",
                        "green",
                    )
                    if not hasattr(self, '_trail_last_notified'):
                        self._trail_last_notified = {}
                    self._trail_last_notified[pos.ticket] = new_sl

    # Timeframe → minutes per bar (for horizon exit)
    _TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

    def horizon_exit_check(self, running: bool, emergency_stop: bool):
        """Time-based exit: close positions that exceeded their Optuna-optimized horizon."""
        if not running or emergency_stop or self.mt5 is None:
            return

        try:
            positions = self.mt5.positions_get()
            if not positions:
                return

            now = datetime.now(timezone.utc)

            for pos in positions:
                if pos.magic < 2000:
                    continue

                sym_cfg = self._sym_cfg(pos.symbol)
                horizon = sym_cfg.get('exit_horizon')
                if horizon is None:
                    continue

                tf = sym_cfg.get('exit_timeframe', 'H4')
                bar_minutes = self._TF_MINUTES.get(tf, 240)
                max_hold_secs = horizon * bar_minutes * 60

                open_time = datetime.fromtimestamp(pos.time, tz=timezone.utc)
                hold_secs = (now - open_time).total_seconds()

                if hold_secs < max_hold_secs:
                    continue

                # Horizon exceeded — close at market
                tick = self.mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue

                is_buy = pos.type == self.mt5.ORDER_TYPE_BUY
                close_price = tick.bid if is_buy else tick.ask
                close_type = self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY
                direction = "BUY" if is_buy else "SELL"
                pnl = pos.profit + pos.swap
                hold_hours = hold_secs / 3600

                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": cfg.DEVIATION,
                    "magic": pos.magic,
                    "comment": "Sovereign_HorizonExit",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                }
                result = self.mt5.order_send(request)
                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    real_pnl = self._deal_pnl(pos.ticket)
                    pnl_str = f"${real_pnl:+.2f}" if real_pnl is not None else f"~${pnl:+.2f}"
                    self.logger.log('INFO', 'HorizonExit', 'POSITION_CLOSED',
                                    f'{pos.symbol} {direction} ticket={pos.ticket} '
                                    f'held {hold_hours:.1f}h > {horizon}×{tf} ({horizon * bar_minutes / 60:.0f}h) '
                                    f'PnL={pnl_str}')
                    if self.discord:
                        color = "orange" if (real_pnl if real_pnl is not None else pnl) >= 0 else "red"
                        acct_tag = f" [{self.account_name}]" if self.account_name != "default" else ""
                        self.discord.send(
                            f"HORIZON EXIT{acct_tag}: {pos.symbol}",
                            f"{direction} {pos.volume} lots closed\n"
                            f"Held {hold_hours:.1f}h > max {horizon}×{tf}\n"
                            f"P&L: {pnl_str}",
                            color,
                        )
                else:
                    rc = result.retcode if result else 'None'
                    self.logger.log('ERROR', 'HorizonExit', 'CLOSE_FAILED',
                                    f'{pos.symbol} ticket={pos.ticket} rc={rc}')

        except Exception as e:
            self.logger.log('ERROR', 'HorizonExit', 'ERROR', str(e))

    def auto_close_bleeders(self, running: bool, emergency_stop: bool):
        """Auto-close swap-bleeding and zombie positions."""
        if not running or emergency_stop or self.mt5 is None:
            return

        try:
            positions = self.mt5.positions_get()
            if not positions:
                return

            now = datetime.now(timezone.utc)

            for pos in positions:
                if pos.magic < 2000:
                    continue

                pnl = pos.profit
                swap = pos.swap
                hold_hours = (now - datetime.fromtimestamp(pos.time, tz=timezone.utc)).total_seconds() / 3600
                reason = None

                # Micro-remnant: partial SL/TP fill left a tiny position (e.g. 0.01 lot)
                sym_info = self.mt5.symbol_info(pos.symbol)
                vol_min = sym_info.volume_min if sym_info else 0.01
                if pos.volume <= vol_min and hold_hours > 1:
                    reason = f"MICRO_REMNANT (vol={pos.volume}, {hold_hours:.1f}h, partial fill remnant)"

                if hold_hours > 12 and pnl <= 0 and abs(swap) > 0.50:
                    # F6: positive swap (carry) → don't close as SWAP_BLEEDING
                    if swap <= 0 and (abs(pnl) < 0.50 or (abs(swap) / abs(pnl) > 0.25)):
                        reason = f"SWAP_BLEEDING (pnl=${pnl:+.2f}, swap=${swap:.2f}, {hold_hours:.0f}h)"

                if hold_hours > 48 and abs(pnl + swap) < 5.0:
                    reason = f"ZOMBIE (pnl=${pnl:+.2f}, swap=${swap:+.2f}, {hold_hours:.0f}h held, dead capital)"

                if hold_hours > 24 and (pnl + swap) < -50:
                    reason = f"DEEP_LOSS (pnl=${pnl:+.2f}, {hold_hours:.0f}h, cutting losses)"

                if reason is None:
                    continue

                tick = self.mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue
                close_price = tick.bid if pos.type == self.mt5.ORDER_TYPE_BUY else tick.ask
                close_type = self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
                direction = "BUY" if pos.type == self.mt5.ORDER_TYPE_BUY else "SELL"

                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": cfg.DEVIATION,
                    "magic": pos.magic,
                    "comment": "Sovereign_AutoClose",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                }
                result = self.mt5.order_send(request)
                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    real_pnl = self._deal_pnl(pos.ticket)
                    pnl_str = f"${real_pnl:+.2f}" if real_pnl is not None else f"~${pnl + swap:+.2f}"
                    self.logger.log('INFO', 'AutoClose', 'POSITION_CLOSED',
                                    f'{pos.symbol} {direction} ticket={pos.ticket} PnL={pnl_str} | {reason}')
                    if self.discord:
                        color = "red" if (real_pnl if real_pnl is not None else pnl) < 0 else "orange"
                        acct_tag = f" [{self.account_name}]" if self.account_name != "default" else ""
                        self.discord.send(
                            f"AUTO-CLOSE{acct_tag}: {pos.symbol}",
                            f"{direction} {pos.volume} lots closed\n"
                            f"P&L: {pnl_str}\n"
                            f"Reason: {reason}",
                            color,
                        )
                else:
                    rc = result.retcode if result else 'None'
                    self.logger.log('ERROR', 'AutoClose', 'CLOSE_FAILED',
                                    f'{pos.symbol} ticket={pos.ticket} rc={rc}')

        except Exception as e:
            self.logger.log('ERROR', 'AutoClose', 'ERROR', str(e))

    def ml_exit_check(self, filters: dict, multi_tf_filters: dict | None = None,
                      multi_tf_symbols: dict | None = None):
        """Check open positions against their ML model.

        If the model says 'don't trade' for N consecutive checks, the edge
        is gone and we close the position early.

        Called every ~60s from the main loop, but internally rate-limited
        to ML_EXIT_COOLDOWN_S (300s / 5 min).

        Args:
            filters: H1 SovereignMLFilter dict {symbol: filter}
            multi_tf_filters: Multi-TF filter dict {symbol: filter} (optional)
            multi_tf_symbols: Multi-TF config dict {symbol: {timeframe: ...}} (optional)
        """
        now = time.time()
        if now - self._ml_exit_last_run < ML_EXIT_COOLDOWN_S:
            return
        self._ml_exit_last_run = now

        if self.mt5 is None:
            return

        try:
            positions = self.mt5.positions_get()
            if not positions:
                # Clean up stale strike tracking
                self._ml_exit_strikes.clear()
                return

            active_tickets = set()

            for pos in positions:
                if pos.magic < 2000:
                    continue

                active_tickets.add(pos.ticket)
                symbol = pos.symbol

                # Find the right filter for this symbol
                filt = None
                tf_name = "H1"
                if multi_tf_symbols and symbol in multi_tf_symbols:
                    tf_name = multi_tf_symbols[symbol].get("timeframe", "H1")
                    if multi_tf_filters and symbol in multi_tf_filters:
                        filt = multi_tf_filters[symbol]
                if filt is None and symbol in filters:
                    filt = filters[symbol]

                if filt is None or filt.model is None:
                    continue

                # Skip if position is too young (give trade room)
                hold_secs = now - pos.time
                bar_secs = {"M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}.get(tf_name, 3600)
                if hold_secs < bar_secs * ML_EXIT_MIN_HOLD_BARS:
                    continue

                # Build features for current bar
                features_np, primary_side = self._get_exit_features(symbol, tf_name)
                if features_np is None:
                    continue

                # Run inference
                proba = filt.predict(features_np)

                # Track strikes
                state = self._ml_exit_strikes.setdefault(pos.ticket, {
                    "strikes": 0, "last_proba": 0.5,
                })
                state["last_proba"] = proba

                # Check if model says "don't trade"
                is_buy = pos.type == self.mt5.ORDER_TYPE_BUY
                direction_flipped = (primary_side > 0 and not is_buy) or \
                                    (primary_side < 0 and is_buy)

                if proba < ML_EXIT_THRESHOLD or direction_flipped:
                    state["strikes"] += 1
                    self.logger.log('DEBUG', 'MLExit', 'STRIKE',
                                    f'{symbol} ticket={pos.ticket} '
                                    f'proba={proba:.3f} side={primary_side:+d} '
                                    f'strikes={state["strikes"]}/{ML_EXIT_STRIKES}')
                else:
                    # Reset strikes — model still agrees with trade
                    if state["strikes"] > 0:
                        self.logger.log('DEBUG', 'MLExit', 'RESET',
                                        f'{symbol} proba={proba:.3f} — strikes reset')
                    state["strikes"] = 0

                # Exit if N consecutive strikes
                if state["strikes"] >= ML_EXIT_STRIKES:
                    pnl = pos.profit + pos.swap
                    if self._ml_exit_close(pos, proba, state["strikes"], pnl):
                        state["strikes"] = 0

            # Clean up tracking for closed positions
            stale = [t for t in self._ml_exit_strikes if t not in active_tickets]
            for t in stale:
                del self._ml_exit_strikes[t]

        except Exception as e:
            self.logger.log('ERROR', 'MLExit', 'CHECK_ERROR', str(e))

    def _get_exit_features(self, symbol: str, tf_name: str):
        """Build current features for ML exit check. Lightweight: uses MT5 bars only."""
        try:
            from engine.inference import _ensure_ml_imports
            _ensure_ml_imports()
            from engine.inference import pl, build_bar_features, FEATURE_COLUMNS
            import numpy as np

            # Get the right timeframe
            tf_map = {"M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
                      "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4"}
            mt5_tf = getattr(self.mt5, tf_map.get(tf_name, "TIMEFRAME_H1"), None)
            if mt5_tf is None:
                return None, None

            minutes = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}.get(tf_name, 60)
            bars_needed = max(200, int(200 * 60 / minutes))

            rates = self.mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars_needed)
            if rates is None or len(rates) < 100:
                return None, None

            bars = pl.DataFrame({
                "time": [datetime.fromtimestamp(int(r['time']), tz=timezone.utc) for r in rates],
                "open": [float(r['open']) for r in rates],
                "high": [float(r['high']) for r in rates],
                "low": [float(r['low']) for r in rates],
                "close": [float(r['close']) for r in rates],
                "volume": [float(r['tick_volume']) for r in rates],
            }).with_columns(pl.col("time").cast(pl.Datetime("us", "UTC")))

            feat = build_bar_features(bars, z_threshold=0.0)
            if feat.height < 2:
                return None, None

            last_row = feat.tail(1)
            features_np = last_row.select(FEATURE_COLUMNS).to_numpy()
            if not np.all(np.isfinite(features_np)):
                return None, None

            primary_side = int(last_row["primary_side"][0])
            return features_np, primary_side

        except Exception:
            return None, None

    def session_close_check(self, trading_schedule, running: bool, emergency_stop: bool):
        """Close positions before their daily market session ends.

        Prevents overnight gap risk by closing positions SESSION_CLOSE_BUFFER_MIN
        minutes before the session close time from the FTMO trading schedule.
        Skips crypto (24/7 markets).
        """
        if not running or emergency_stop or self.mt5 is None:
            return

        try:
            positions = self.mt5.positions_get()
            if not positions:
                return

            for pos in positions:
                if pos.magic < 2000:
                    continue

                # Skip crypto — no daily close
                if trading_schedule.is_crypto(pos.symbol):
                    continue

                is_open, minutes_left = trading_schedule.is_trading_open(pos.symbol)

                # Not open or no schedule info → skip
                if not is_open or minutes_left is None:
                    continue

                # Still enough time → skip
                if minutes_left > SESSION_CLOSE_BUFFER_MIN:
                    continue

                # Close the position
                tick = self.mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue

                is_buy = pos.type == self.mt5.ORDER_TYPE_BUY
                close_price = tick.bid if is_buy else tick.ask
                close_type = self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY
                direction = "BUY" if is_buy else "SELL"
                pnl = pos.profit + pos.swap

                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": cfg.DEVIATION,
                    "magic": pos.magic,
                    "comment": "Sovereign_SessionClose",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                }
                result = self.mt5.order_send(request)
                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    real_pnl = self._deal_pnl(pos.ticket)
                    pnl_str = f"${real_pnl:+.2f}" if real_pnl is not None else f"~${pnl:+.2f}"
                    self.logger.log('INFO', 'SessionClose', 'POSITION_CLOSED',
                                    f'{pos.symbol} {direction} ticket={pos.ticket} '
                                    f'{minutes_left} min before session close | PnL={pnl_str}')
                    if self.discord:
                        color = "orange" if (real_pnl if real_pnl is not None else pnl) >= 0 else "red"
                        acct_tag = f" [{self.account_name}]" if self.account_name != "default" else ""
                        self.discord.send(
                            f"SESSION CLOSE{acct_tag}: {pos.symbol}",
                            f"{direction} {pos.volume} lots gesloten\n"
                            f"Markt sluit over {minutes_left} min\n"
                            f"P&L: {pnl_str}",
                            color,
                        )
                else:
                    rc = result.retcode if result else 'None'
                    self.logger.log('ERROR', 'SessionClose', 'CLOSE_FAILED',
                                    f'{pos.symbol} ticket={pos.ticket} rc={rc}')

        except Exception as e:
            self.logger.log('ERROR', 'SessionClose', 'ERROR', str(e))

    def _ml_exit_close(self, pos, proba: float, strikes: int, pnl: float) -> bool:
        """Close a position because ML model no longer supports it. Returns True on success."""
        tick = self.mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return False

        is_buy = pos.type == self.mt5.ORDER_TYPE_BUY
        close_price = tick.bid if is_buy else tick.ask
        close_type = self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY
        direction = "BUY" if is_buy else "SELL"

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": cfg.DEVIATION,
            "magic": pos.magic,
            "comment": "Sovereign_MLExit",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        result = self.mt5.order_send(request)
        if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
            real_pnl = self._deal_pnl(pos.ticket)
            pnl_str = f"${real_pnl:+.2f}" if real_pnl is not None else f"~${pnl:+.2f}"
            self.logger.log('INFO', 'MLExit', 'POSITION_CLOSED',
                            f'{pos.symbol} {direction} ticket={pos.ticket} '
                            f'proba={proba:.3f} strikes={strikes} PnL={pnl_str}')
            if self.discord:
                color = "orange" if (real_pnl if real_pnl is not None else pnl) >= 0 else "red"
                acct_tag = f" [{self.account_name}]" if self.account_name != "default" else ""
                self.discord.send(
                    f"ML EXIT{acct_tag}: {pos.symbol}",
                    f"{direction} {pos.volume} lots closed\n"
                    f"Model proba: {proba:.3f} ({strikes}× under {ML_EXIT_THRESHOLD})\n"
                    f"P&L: {pnl_str}",
                    color,
                )
            return True
        else:
            rc = result.retcode if result else 'None'
            self.logger.log('ERROR', 'MLExit', 'CLOSE_FAILED',
                            f'{pos.symbol} ticket={pos.ticket} rc={rc}')
            return False
