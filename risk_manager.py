from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import config


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_time: str
    stop_pct: float
    initial_stop: float
    highest_price: float
    add_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)

    @property
    def entry_dt(self) -> datetime:
        return _to_naive_utc(datetime.fromisoformat(self.entry_time))

    def current_stop(self) -> float:
        trail_stop = self.highest_price * (1 - self.stop_pct)
        return max(self.initial_stop, trail_stop)

    def update_high(self, price: float) -> None:
        if price > self.highest_price:
            self.highest_price = price

    def age_days(self, now: Optional[datetime] = None) -> int:
        if now is None:
            now = datetime.utcnow()
        now = _to_naive_utc(now)
        return max(0, (now - self.entry_dt).days)


class RiskManager:
    def __init__(self, account_size: float):
        self.account_size = account_size
        self.positions: Dict[str, Position] = {}
        self.daily_realized_pnl = 0.0
        self.trades_today = 0
        self.cooldowns: Dict[str, str] = {}

    def portfolio_market_value(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None:
                total += pos.qty * px
        return total

    def gross_exposure_pct(self, prices: Dict[str, float]) -> float:
        return self.portfolio_market_value(prices) / max(self.account_size, 1.0)

    def on_new_day(self) -> None:
        self.daily_realized_pnl = 0.0
        self.trades_today = 0

    def on_trade_closed(self, pnl: float, symbol: str) -> None:
        self.daily_realized_pnl += pnl
        self.trades_today += 1

        minutes = (
            config.COOLDOWN_AFTER_WIN_MINUTES
            if pnl >= 0
            else config.COOLDOWN_AFTER_LOSS_MINUTES
        )
        cooldown_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.cooldowns[symbol] = cooldown_until.isoformat()

    def in_cooldown(self, symbol: str) -> bool:
        v = self.cooldowns.get(symbol)
        if not v:
            return False
        return datetime.utcnow() < datetime.fromisoformat(v)

    def daily_loss_exceeded(self) -> bool:
        return self.daily_realized_pnl <= -(self.account_size * config.MAX_DAILY_LOSS_PCT)

    def stop_price(self, entry_price: float, stop_pct: float) -> float:
        return entry_price * (1 - stop_pct)

    def risk_position_size(
        self,
        price: float,
        stop_pct: float,
        position_scalar: float = 1.0,
    ) -> int:
        if price <= 0 or stop_pct <= 0:
            return 0

        risk_budget = self.account_size * config.RISK_PER_TRADE * position_scalar
        dollar_risk_per_share = price * stop_pct
        qty_by_risk = int(risk_budget / dollar_risk_per_share)

        max_dollars = min(
            self.account_size * config.MAX_POSITION_WEIGHT * position_scalar,
            config.MAX_POSITION_DOLLARS,
        )
        qty_by_cap = int(max_dollars / price)

        qty = max(0, min(qty_by_risk, qty_by_cap))
        return qty

    def can_open_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        prices: Dict[str, float],
    ) -> tuple[bool, str]:
        if qty <= 0:
            return False, "qty<=0"

        if symbol in self.positions:
            return False, "already_open"

        if self.in_cooldown(symbol):
            return False, "cooldown"

        if self.daily_loss_exceeded():
            return False, "daily_loss_limit"

        if self.trades_today >= config.MAX_TRADES_PER_DAY:
            return False, "trade_limit"

        if len(self.positions) >= config.MAX_POSITIONS:
            return False, "max_positions"

        new_value = qty * price
        new_gross = self.portfolio_market_value(prices) + new_value
        if new_gross / max(self.account_size, 1.0) > config.MAX_TOTAL_EXPOSURE:
            return False, "max_total_exposure"

        return True, "ok"

    def open_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        stop_pct: float,
        entry_time: Optional[datetime] = None,
    ) -> Position:
        if entry_time is None:
            entry_time = datetime.utcnow()

        entry_time = _to_naive_utc(entry_time)

        pos = Position(
            symbol=symbol,
            qty=int(qty),
            entry_price=float(price),
            entry_time=entry_time.isoformat(),
            stop_pct=float(stop_pct),
            initial_stop=float(price * (1 - stop_pct)),
            highest_price=float(price),
            add_count=0,
        )
        self.positions[symbol] = pos
        self.trades_today += 1
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        return self.positions.pop(symbol, None)

    def maybe_update_position_high(self, symbol: str, price: float) -> None:
        pos = self.positions.get(symbol)
        if pos:
            pos.update_high(price)

    def should_exit(self, symbol: str, price: float) -> tuple[bool, str]:
        pos = self.positions.get(symbol)
        if not pos:
            return False, "not_open"

        pos.update_high(price)

        if price <= pos.current_stop():
            return True, "stop"

        if price >= pos.entry_price * (1 + config.TAKE_PROFIT_PCT):
            return True, "take_profit"

        if pos.age_days() >= config.MAX_HOLD_DAYS:
            return True, "max_hold"

        return False, "hold"

    def can_pyramid(self, symbol: str, price: float, prices: Dict[str, float]) -> tuple[bool, str]:
        if not config.ALLOW_PYRAMIDING:
            return False, "pyramiding_disabled"

        pos = self.positions.get(symbol)
        if not pos:
            return False, "not_open"

        if pos.add_count >= config.MAX_PYRAMID_ADDS:
            return False, "max_adds"

        risk_per_share = pos.entry_price - pos.initial_stop
        if risk_per_share <= 0:
            return False, "bad_risk"

        open_profit = price - pos.entry_price
        r_multiple = open_profit / risk_per_share
        if r_multiple < config.PYRAMID_MIN_R_MULTIPLE:
            return False, "not_far_enough"

        add_qty = int(pos.qty * config.PYRAMID_SIZE_FRACTION)
        ok, reason = self.can_open_position(f"{symbol}__addprobe", add_qty, price, prices)
        if not ok and reason not in {"already_open"}:
            return False, reason

        return True, "ok"

    def apply_pyramid(self, symbol: str, add_qty: int, add_price: float) -> None:
        pos = self.positions[symbol]
        new_qty = pos.qty + add_qty
        blended_entry = ((pos.entry_price * pos.qty) + (add_price * add_qty)) / new_qty
        pos.qty = new_qty
        pos.entry_price = blended_entry
        pos.highest_price = max(pos.highest_price, add_price)
        pos.add_count += 1

        new_initial_stop = blended_entry * (1 - pos.stop_pct)
        pos.initial_stop = max(pos.initial_stop, new_initial_stop)

    def to_dict(self) -> dict:
        return {
            "account_size": self.account_size,
            "daily_realized_pnl": self.daily_realized_pnl,
            "trades_today": self.trades_today,
            "cooldowns": self.cooldowns,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskManager":
        rm = cls(account_size=float(d.get("account_size", config.ACCOUNT_SIZE)))
        rm.daily_realized_pnl = float(d.get("daily_realized_pnl", 0.0))
        rm.trades_today = int(d.get("trades_today", 0))
        rm.cooldowns = dict(d.get("cooldowns", {}))
        rm.positions = {
            k: Position.from_dict(v) for k, v in d.get("positions", {}).items()
        }
        return rm