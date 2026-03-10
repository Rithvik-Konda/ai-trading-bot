"""
Risk Management Module — v3.3 (Adaptive + Pyramid Support)
============================================================
Position sizing, trailing stops, exposure limits, circuit breakers,
cooldowns, and hard max loss cap.

Changes in v3.3:
- Adaptive stop loss based on bull/bear regime
- Hard max loss cap (6%) — catches gap-downs past trailing stop
- Supports pyramiding (check_trade allows existing positions)
- 95% max exposure in bull, concentrated positions
- No fixed TP — trailing stop handles exits
"""

import logging
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracks an open position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: int
    entry_time: datetime = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    highest_price: float = 0.0
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.highest_price = max(self.highest_price, current_price)
            # Use adaptive stop loss from config
            sl_pct = getattr(config, 'BULL_STOP_LOSS_PCT', config.STOP_LOSS_PCT)
            self.trailing_stop = self.highest_price * (1 - sl_pct)
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


@dataclass
class RiskCheck:
    """Result of a risk check."""
    approved: bool = True
    reason: str = ""
    adjusted_quantity: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0


class RiskManager:
    """
    Enforces position limits, exposure caps, stop losses, circuit breakers,
    cooldowns, and hard max loss cap.
    """

    def __init__(self, portfolio_value: float = 100000.0):
        self.portfolio_value = portfolio_value
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.last_trade_date: Optional[date] = None
        self.halted = False
        self._cooldowns: Dict[str, datetime] = {}
        self.cooldown_minutes = getattr(config, 'COOLDOWN_MINUTES', 60)

    def update_portfolio_value(self, value: float):
        self.portfolio_value = value

    def _reset_daily_if_needed(self):
        today = date.today()
        if self.last_trade_date != today:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_trade_date = today
            self.halted = False
            self._cooldowns.clear()

    # ── Cooldown Helpers ─────────────────────────────────────────────────

    def _set_cooldown(self, symbol: str):
        until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
        self._cooldowns[symbol] = until
        logger.info(f"Cooldown set: {symbol} blocked until {until.strftime('%H:%M:%S')} ({self.cooldown_minutes}min)")

    def _is_on_cooldown(self, symbol: str) -> bool:
        if symbol not in self._cooldowns:
            return False
        if datetime.now() >= self._cooldowns[symbol]:
            del self._cooldowns[symbol]
            return False
        return True

    # ── Pre-Trade Risk Check ─────────────────────────────────────────────

    def check_trade(
        self, symbol: str, side: str, price: float, requested_qty: int
    ) -> RiskCheck:
        """
        Validate a proposed trade against all risk rules.
        v3.3: Allows pyramiding (no block on existing same-side positions).
        """
        self._reset_daily_if_needed()
        check = RiskCheck()

        # Circuit breaker: daily loss limit
        if self.halted:
            check.approved = False
            check.reason = "Trading halted — daily loss limit breached"
            return check

        # Circuit breaker: max trades
        if self.trades_today >= config.MAX_TRADES_PER_DAY:
            check.approved = False
            check.reason = f"Max trades/day reached ({config.MAX_TRADES_PER_DAY})"
            return check

        # Cooldown check
        if self._is_on_cooldown(symbol):
            remaining = (self._cooldowns[symbol] - datetime.now()).seconds // 60
            check.approved = False
            check.reason = f"Cooldown active on {symbol} ({remaining}min remaining)"
            return check

        # Position size limit — % of portfolio AND hard dollar cap
        max_pos_pct = config.MAX_POSITION_SIZE_PCT
        max_position_value = self.portfolio_value * max_pos_pct
        # Hard dollar cap prevents oversizing as account grows
        dollar_cap = getattr(config, 'MAX_POSITION_DOLLARS', 20000)
        max_position_value = min(max_position_value, dollar_cap)
        max_qty = int(max_position_value / price) if price > 0 else 0
        check.adjusted_quantity = min(requested_qty, max_qty)

        if check.adjusted_quantity <= 0:
            check.approved = False
            check.reason = "Position too small after risk adjustment"
            return check

        # Total exposure limit
        current_exposure = sum(
            pos.entry_price * pos.quantity for pos in self.positions.values()
        )
        new_exposure = current_exposure + (price * check.adjusted_quantity)
        max_exposure = self.portfolio_value * config.MAX_TOTAL_EXPOSURE_PCT

        if new_exposure > max_exposure:
            available = max_exposure - current_exposure
            check.adjusted_quantity = int(available / price) if price > 0 else 0
            if check.adjusted_quantity <= 0:
                check.approved = False
                check.reason = f"Max exposure limit reached ({config.MAX_TOTAL_EXPOSURE_PCT*100:.0f}%)"
                return check

        # Set stop loss and take profit (adaptive)
        sl_pct = config.STOP_LOSS_PCT
        tp_pct = config.TAKE_PROFIT_PCT
        if side == "long":
            check.stop_loss = price * (1 - sl_pct)
            check.take_profit = price * (1 + tp_pct)
        else:
            check.stop_loss = price * (1 + sl_pct)
            check.take_profit = price * (1 - tp_pct)

        check.approved = True
        return check

    # ── Position Tracking ────────────────────────────────────────────────

    def open_position(self, symbol: str, side: str, price: float, quantity: int, risk_check: RiskCheck):
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=risk_check.stop_loss,
            take_profit=risk_check.take_profit,
            highest_price=price,
        )
        self.positions[symbol] = pos
        self.trades_today += 1
        logger.info(f"Opened {side} {quantity}x {symbol} @ ${price:.2f} | SL: ${pos.stop_loss:.2f} TP: ${pos.take_profit:.2f}")

    def close_position(self, symbol: str, exit_price: float, reason: str = ""):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self.daily_pnl += pnl
        self.trades_today += 1
        logger.info(f"Closed {pos.side} {symbol} @ ${exit_price:.2f} | PnL: ${pnl:+,.2f} | {reason}")
        del self.positions[symbol]

        # Cooldown on ALL exits — prevents bot from immediately re-entering after a close
        # Profit exits: shorter cooldown (60min) — maybe momentum is still valid later
        # Stop/loss exits: longer cooldown (120min) — something is wrong, stay away
        if "stop" in reason.lower() or "max loss" in reason.lower():
            original = self.cooldown_minutes
            self.cooldown_minutes = getattr(config, 'COOLDOWN_LOSS_MINUTES', 120)
            self._set_cooldown(symbol)
            self.cooldown_minutes = original
        else:
            original = self.cooldown_minutes
            self.cooldown_minutes = getattr(config, 'COOLDOWN_PROFIT_MINUTES', 60)
            self._set_cooldown(symbol)
            self.cooldown_minutes = original

        # Daily loss circuit breaker
        if self.daily_pnl < -(self.portfolio_value * config.MAX_DAILY_LOSS_PCT):
            self.halted = True
            logger.warning(f"CIRCUIT BREAKER: Daily loss ${self.daily_pnl:+,.2f} exceeds limit. Trading halted.")

    # ── Stop Loss / Take Profit Monitoring ───────────────────────────────

    def check_exits(self, prices: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Check all positions for exit triggers.
        v3.4: per-symbol stop loss + min hold days + trailing stop + hard max loss.
        """
        exits = []
        min_hold_days = getattr(config, 'MIN_HOLD_DAYS', 0)

        for symbol, pos in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue

            pos.update_pnl(price)

            # How long have we held this position?
            hold_days = 0
            if pos.entry_time:
                hold_days = (datetime.now() - pos.entry_time).days

            # Per-symbol stop override — wider for slow cyclicals
            sym_sl_pct = getattr(config, 'SYMBOL_STOP_LOSS_PCT', {}).get(symbol)
            max_loss_pct = getattr(config, 'MAX_LOSS_PCT', 0.06)

            if pos.side == "long":
                # Hard max loss ALWAYS fires — gap-down protection, no min hold exception
                if price <= pos.entry_price * (1 - max_loss_pct):
                    exits.append((symbol, price, f"Hard max loss cap ({max_loss_pct*100:.0f}%)"))
                    continue

                # Everything else respects min hold days
                if hold_days < min_hold_days:
                    continue

                # Use per-symbol stop if configured, else fall back to position's stop
                if sym_sl_pct is not None:
                    sym_trailing = pos.highest_price * (1 - sym_sl_pct) if pos.highest_price > 0 else pos.stop_loss
                    effective_stop = max(pos.stop_loss, sym_trailing)
                else:
                    effective_stop = max(pos.stop_loss, pos.trailing_stop) if pos.trailing_stop > 0 else pos.stop_loss

                if price <= effective_stop:
                    exits.append((symbol, price, "Trailing stop hit"))
                elif price >= pos.take_profit:
                    exits.append((symbol, price, "Take profit hit"))
            else:
                if price >= pos.stop_loss:
                    exits.append((symbol, price, "Stop loss hit"))
                elif price <= pos.take_profit:
                    exits.append((symbol, price, "Take profit hit"))

        return exits

    # ── Reporting ────────────────────────────────────────────────────────

    def summary(self, current_prices: Dict[str, float] = None) -> dict:
        """
        Portfolio summary.
        Pass current_prices dict for accurate exposure and unrealized PnL.
        Falls back to entry price (cost basis) if prices not provided.
        """
        total_unrealized = 0.0
        total_exposure = 0.0
        for sym, p in self.positions.items():
            current = (current_prices or {}).get(sym, p.entry_price)
            total_exposure += current * p.quantity
            if p.side == "long":
                total_unrealized += (current - p.entry_price) * p.quantity
            else:
                total_unrealized += (p.entry_price - current) * p.quantity

        return {
            "portfolio_value": self.portfolio_value,
            "open_positions": len(self.positions),
            "total_exposure": total_exposure,
            "exposure_pct": (total_exposure / self.portfolio_value * 100) if self.portfolio_value > 0 else 0,
            "unrealized_pnl": total_unrealized,
            "daily_pnl": self.daily_pnl,
            "true_day_pnl": self.daily_pnl + total_unrealized,
            "trades_today": self.trades_today,
            "halted": self.halted,
        }