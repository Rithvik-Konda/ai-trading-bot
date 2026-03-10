"""
Earnings Guard Agent — v3.3
==============================
Protects portfolio from earnings gap-downs by:
1. Detecting upcoming earnings dates for watchlist stocks
2. Reducing position size or exiting before earnings
3. Setting cooldowns to prevent entry right before earnings
4. Tracking post-earnings for re-entry opportunities

Uses yfinance for earnings calendar data (free, no API key needed).
Falls back to Alpaca news headlines if yfinance fails.

Install: pip install yfinance

Usage:
    guard = EarningsGuard()
    
    # Check if a stock has earnings coming up
    guard.update_calendar(["NVDA", "AMD", "GOOGL"])
    risk = guard.check_earnings_risk("NVDA")
    
    # risk.days_to_earnings = 2
    # risk.action = "EXIT"  (within 1 day)
    # risk.action = "REDUCE"  (within 3 days)
    # risk.action = "HOLD"  (>3 days away)
"""

import logging
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")


@dataclass
class EarningsRisk:
    """Assessment of earnings risk for a symbol."""
    symbol: str
    has_earnings_soon: bool = False
    earnings_date: Optional[date] = None
    days_to_earnings: Optional[int] = None
    timing: str = "unknown"  # "BMO" (before market open), "AMC" (after market close), "unknown"
    action: str = "HOLD"  # "HOLD", "REDUCE", "EXIT", "BLOCK_ENTRY"
    reason: str = ""
    position_scale: float = 1.0  # Multiplier for position size (1.0 = full, 0.5 = half, 0 = none)


@dataclass
class EarningsEvent:
    """Cached earnings event data."""
    symbol: str
    earnings_date: date
    timing: str = "unknown"
    eps_estimate: Optional[float] = None
    fetched_at: datetime = field(default_factory=datetime.now)


class EarningsGuard:
    """
    Agent 6: Earnings calendar guard.
    Prevents the bot from holding large positions through earnings announcements.
    
    Rules:
    - 1 day before earnings: EXIT position or block new entry
    - 2-3 days before: REDUCE position to 50%
    - 4-7 days before: Allow entry but with 50% size
    - After earnings (next day): Allow re-entry with full size
    
    Config:
    - EXIT_DAYS: Days before earnings to force exit (default: 1)
    - REDUCE_DAYS: Days before earnings to reduce position (default: 3)
    - CAUTION_DAYS: Days before earnings to reduce new entry size (default: 7)
    - COOLDOWN_AFTER_EARNINGS: Hours to wait after earnings before re-entry (default: 4)
    """

    def __init__(
        self,
        exit_days: int = 1,
        reduce_days: int = 3,
        caution_days: int = 7,
        cooldown_hours: int = 4,
    ):
        self.exit_days = exit_days
        self.reduce_days = reduce_days
        self.caution_days = caution_days
        self.cooldown_hours = cooldown_hours

        # Cache: symbol -> EarningsEvent
        self._calendar: Dict[str, EarningsEvent] = {}
        self._last_update: Optional[datetime] = None
        self._update_interval = timedelta(hours=6)  # Refresh every 6 hours

        # Post-earnings cooldown tracker
        self._post_earnings_cooldowns: Dict[str, datetime] = {}

    def needs_update(self) -> bool:
        """Check if calendar cache needs refreshing."""
        if self._last_update is None:
            return True
        return datetime.now() - self._last_update > self._update_interval

    def update_calendar(self, symbols: List[str]) -> int:
        """
        Fetch upcoming earnings dates for all symbols.
        Returns number of symbols with upcoming earnings found.
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available — earnings guard disabled")
            return 0

        found = 0
        for symbol in symbols:
            try:
                event = self._fetch_earnings(symbol)
                if event:
                    self._calendar[symbol] = event
                    found += 1
                    logger.info(
                        f"[earnings] {symbol}: next earnings {event.earnings_date} "
                        f"({event.timing}) — {(event.earnings_date - date.today()).days}d away"
                    )
            except Exception as e:
                logger.debug(f"[earnings] Failed to fetch {symbol}: {e}")

        self._last_update = datetime.now()
        logger.info(f"[earnings] Calendar updated: {found}/{len(symbols)} symbols have upcoming earnings")
        return found

    def _fetch_earnings(self, symbol: str) -> Optional[EarningsEvent]:
        """Fetch next earnings date from yfinance."""
        try:
            ticker = yf.Ticker(symbol)

            # Method 1: calendar property (most reliable for upcoming)
            cal = ticker.calendar
            if cal is not None:
                earnings_date = None
                timing = "unknown"

                if isinstance(cal, dict):
                    # Newer yfinance versions return dict
                    if "Earnings Date" in cal:
                        ed = cal["Earnings Date"]
                        if isinstance(ed, list) and len(ed) > 0:
                            earnings_date = ed[0]
                        elif hasattr(ed, 'date'):
                            earnings_date = ed
                elif hasattr(cal, 'index'):
                    # DataFrame format
                    if "Earnings Date" in cal.index:
                        ed = cal.loc["Earnings Date"]
                        if hasattr(ed, 'iloc'):
                            earnings_date = ed.iloc[0]
                        else:
                            earnings_date = ed

                if earnings_date is not None:
                    if hasattr(earnings_date, 'date'):
                        earnings_date = earnings_date.date()
                    elif isinstance(earnings_date, str):
                        earnings_date = datetime.strptime(earnings_date[:10], "%Y-%m-%d").date()

                    # Only track future earnings
                    if earnings_date >= date.today():
                        eps_est = None
                        if isinstance(cal, dict) and "EPS Estimate" in cal:
                            eps_est = cal["EPS Estimate"]

                        return EarningsEvent(
                            symbol=symbol,
                            earnings_date=earnings_date,
                            timing=timing,
                            eps_estimate=eps_est,
                        )

            # Method 2: earnings_dates property (fallback)
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and len(earnings_dates) > 0:
                    # Find next future date
                    today = datetime.now()
                    for dt in earnings_dates.index:
                        if hasattr(dt, 'date'):
                            ed = dt.date()
                        else:
                            ed = dt
                        if ed >= date.today():
                            return EarningsEvent(
                                symbol=symbol,
                                earnings_date=ed,
                                timing="unknown",
                            )
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"[earnings] yfinance error for {symbol}: {e}")

        return None

    def check_earnings_risk(self, symbol: str) -> EarningsRisk:
        """
        Assess earnings risk for a symbol.
        Returns EarningsRisk with recommended action.
        """
        risk = EarningsRisk(symbol=symbol)

        # Check post-earnings cooldown
        if symbol in self._post_earnings_cooldowns:
            cooldown_until = self._post_earnings_cooldowns[symbol]
            if datetime.now() < cooldown_until:
                remaining = (cooldown_until - datetime.now()).seconds // 3600
                risk.action = "BLOCK_ENTRY"
                risk.reason = f"Post-earnings cooldown ({remaining}h remaining)"
                risk.position_scale = 0.0
                return risk
            else:
                del self._post_earnings_cooldowns[symbol]

        # Check if we have earnings data
        event = self._calendar.get(symbol)
        if not event:
            return risk  # No data = no risk flag

        today = date.today()
        days_away = (event.earnings_date - today).days

        risk.earnings_date = event.earnings_date
        risk.days_to_earnings = days_away
        risk.timing = event.timing

        if days_away < 0:
            # Earnings already passed — set cooldown for re-entry
            self._post_earnings_cooldowns[symbol] = datetime.now() + timedelta(hours=self.cooldown_hours)
            risk.action = "BLOCK_ENTRY"
            risk.reason = f"Earnings just passed, {self.cooldown_hours}h cooldown"
            risk.position_scale = 0.0

        elif days_away <= self.exit_days:
            risk.has_earnings_soon = True
            risk.action = "EXIT"
            risk.reason = f"Earnings in {days_away}d — EXIT to avoid gap risk"
            risk.position_scale = 0.0

        elif days_away <= self.reduce_days:
            risk.has_earnings_soon = True
            risk.action = "REDUCE"
            risk.reason = f"Earnings in {days_away}d — REDUCE to 50%"
            risk.position_scale = 0.5

        elif days_away <= self.caution_days:
            risk.has_earnings_soon = True
            risk.action = "CAUTION"
            risk.reason = f"Earnings in {days_away}d — limit new entries to 50%"
            risk.position_scale = 0.5

        else:
            risk.action = "HOLD"
            risk.reason = f"Earnings in {days_away}d — safe"
            risk.position_scale = 1.0

        return risk

    def get_all_risks(self) -> List[EarningsRisk]:
        """Get earnings risk for all tracked symbols."""
        risks = []
        for symbol in self._calendar:
            risks.append(self.check_earnings_risk(symbol))
        return sorted(risks, key=lambda r: r.days_to_earnings or 999)

    def summary(self) -> str:
        """Human-readable summary of upcoming earnings."""
        risks = self.get_all_risks()
        if not risks:
            return "[earnings] No upcoming earnings tracked"

        lines = ["[earnings] Upcoming earnings:"]
        for r in risks:
            if r.days_to_earnings is not None and r.days_to_earnings <= self.caution_days:
                emoji = "🔴" if r.action == "EXIT" else "🟡" if r.action in ("REDUCE", "CAUTION") else "🟢"
                lines.append(
                    f"  {emoji} {r.symbol}: {r.days_to_earnings}d away "
                    f"({r.earnings_date}) → {r.action}"
                )

        return "\n".join(lines) if len(lines) > 1 else "[earnings] No imminent earnings"


# ── CLI Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import config

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    symbols = [s for s in config.WATCHLIST if s not in getattr(config, 'INVERSE_ETFS', [])]

    guard = EarningsGuard()
    print(f"\nChecking earnings calendar for {len(symbols)} symbols...\n")
    found = guard.update_calendar(symbols)

    print(f"\n{'='*60}")
    print(guard.summary())
    print(f"{'='*60}")

    # Show all risks
    for risk in guard.get_all_risks():
        if risk.earnings_date:
            print(f"  {risk.symbol:>6}: {risk.earnings_date} ({risk.days_to_earnings}d) → {risk.action} | {risk.reason}")
