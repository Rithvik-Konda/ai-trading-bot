"""
Options Earnings Engine v2.0
==============================
Uses alpaca-py (NOT alpaca-trade-api) — the correct SDK for options.

Install: pip install alpaca-py --break-system-packages

Strategy: Buy put spreads before earnings on high-IV stocks.

Why put spreads (not naked puts, not strangles):
  - Paper + live Level 3 allows buying spreads
  - Defined max loss = debit paid. No margin required.
  - Before earnings, IV is elevated → options are expensive → we want to
    buy BEFORE IV drops, then sell when the move happens
  - Buy ATM put + sell OTM put = capture downside move with limited cost
  - Also buy call spreads when momentum is bullish into earnings

How it works:
  1. 2 days before earnings: check if stock has been weak (bearish) or strong (bullish)
  2. If bearish setup: buy put spread (buy ATM put, sell put 5% lower)
  3. If bullish setup: buy call spread (buy ATM call, sell call 5% higher)
  4. Day after earnings: close both legs (take profit or cut loss)
  5. Max loss = premium paid. Typical win = 2-4x premium on a big move.

Real numbers example (NVDA at $800 before earnings):
  - Buy $800 put @ $15, Sell $760 put @ $5 = net debit $10/share = $1,000/contract
  - If NVDA drops to $750 after earnings: spread worth ~$40 → $3,000 profit
  - If NVDA stays flat or rises: lose $1,000 (the debit paid)
  - Risk:reward = 1:3

API: Uses alpaca-py TradingClient and OptionHistoricalDataClient.
     Contract discovery via get_option_contracts() with strike/expiry filters.
     Orders via MarketOrderRequest with the option contract symbol.
"""

import logging
import time
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import config

logger = logging.getLogger("options_engine")

OPTIONS_CONFIG = {
    "max_position_pct":         0.02,    # Max 2% account per spread
    "max_concurrent_plays":     3,
    "days_before_entry":        2,       # Enter 2 days before earnings
    "days_after_exit":          1,       # Exit 1 day after earnings
    "spread_width_pct":         0.05,    # Short leg is 5% away from long leg
    "min_debit":                0.50,    # Min $0.50 net debit to bother
    "max_debit_pct_of_spread":  0.40,    # Don't pay more than 40% of spread width
    "profit_target_pct":        0.70,    # Close at 70% of max profit
    "stop_loss_pct":            0.50,    # Close if spread loses 50% of debit
    "expiry_days_out":          7,       # Buy contracts expiring ~1 week after earnings
}

EARNINGS_CANDIDATES = [
    "NVDA", "AMD", "MU", "AVGO",
    "GOOGL", "MSFT", "META", "AMZN",
    "PLTR", "CRWD", "TSLA", "COIN",
]


@dataclass
class EarningsSpread:
    symbol: str
    spread_type: str            # "put_spread" or "call_spread"
    earnings_date: date
    entry_date: date
    long_contract: str          # The contract we bought (ATM)
    short_contract: str         # The contract we sold (OTM)
    long_strike: float
    short_strike: float
    expiry: date
    net_debit: float            # Per share cost
    contracts: int
    total_cost: float           # net_debit * contracts * 100
    status: str = "open"
    exit_credit: float = 0.0
    pnl: float = 0.0


class OptionsEarningsEngine:
    """
    Buys put spreads / call spreads before earnings.
    Uses alpaca-py SDK — the correct library for options.
    """

    def __init__(self, broker=None):
        self.broker = broker            # AlpacaBroker instance (for stock prices)
        self.active_plays: Dict[str, EarningsSpread] = {}
        self._earnings_calendar: Dict[str, date] = {}
        self._trade_client = None
        self._data_client = None
        self._last_check = datetime.min
        self._init_alpaca_py()

    def _init_alpaca_py(self):
        """Initialize alpaca-py clients — the correct SDK for options."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._trade_client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=True,
            )
            self._data_client = OptionHistoricalDataClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
            )
            logger.info("Options engine: alpaca-py clients initialized")

            # Verify options are enabled
            acct = self._trade_client.get_account()
            options_level = getattr(acct, 'options_trading_level', None)
            logger.info(f"Options engine: options trading level = {options_level}")

        except ImportError:
            logger.warning(
                "Options engine: alpaca-py not installed. "
                "Run: pip install alpaca-py --break-system-packages"
            )
        except Exception as e:
            logger.warning(f"Options engine: init failed: {e}")

    # ── Earnings Calendar ─────────────────────────────────────────────

    def set_earnings_date(self, symbol: str, earnings_date: date):
        """Called by main bot's earnings guard to register upcoming earnings."""
        self._earnings_calendar[symbol] = earnings_date

    def days_to_earnings(self, symbol: str) -> Optional[int]:
        if symbol not in self._earnings_calendar:
            return None
        return (self._earnings_calendar[symbol] - date.today()).days

    # ── Contract Discovery ────────────────────────────────────────────

    def find_spread_contracts(
        self,
        symbol: str,
        spread_type: str,       # "put" or "call"
        stock_price: float,
        expiry_date: date,
    ) -> Optional[Tuple[str, str, float, float]]:
        """
        Find the best ATM + OTM contract pair for a spread.
        Returns (long_symbol, short_symbol, long_strike, short_strike) or None.

        Uses alpaca-py's get_option_contracts() to fetch real contract symbols
        from Alpaca's options chain — no manual symbol construction needed.
        """
        if self._trade_client is None:
            return None

        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import AssetStatus, ExerciseStyle
            from alpaca.data.requests import OptionLatestQuoteRequest

            # For put spread: buy ATM put, sell OTM put (lower strike)
            # For call spread: buy ATM call, sell OTM call (higher strike)
            width = stock_price * OPTIONS_CONFIG["spread_width_pct"]

            if spread_type == "put":
                long_strike_target = stock_price          # ATM
                short_strike_target = stock_price - width  # 5% OTM
                strike_min = short_strike_target * 0.97
                strike_max = long_strike_target * 1.03
            else:  # call
                long_strike_target = stock_price
                short_strike_target = stock_price + width
                strike_min = long_strike_target * 0.97
                strike_max = short_strike_target * 1.03

            # Fetch contracts in our strike range
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=expiry_date - timedelta(days=3),
                expiration_date_lte=expiry_date + timedelta(days=3),
                type=spread_type,
                style=ExerciseStyle.AMERICAN,
                strike_price_gte=str(round(strike_min, 2)),
                strike_price_lte=str(round(strike_max, 2)),
                limit=20,
            )
            result = self._trade_client.get_option_contracts(req)

            if not result.option_contracts:
                logger.info(f"Options: No {spread_type} contracts found for {symbol}")
                return None

            # Find the contract closest to each target strike
            contracts_by_strike = {
                float(c.strike_price): c.symbol
                for c in result.option_contracts
                if c.tradable
            }

            if len(contracts_by_strike) < 2:
                return None

            available_strikes = sorted(contracts_by_strike.keys())

            # Long leg: closest to ATM
            long_strike = min(available_strikes, key=lambda s: abs(s - long_strike_target))
            long_sym = contracts_by_strike[long_strike]

            # Short leg: closest to OTM target, but NOT the same as long
            remaining = [s for s in available_strikes if s != long_strike]
            if not remaining:
                return None
            short_strike = min(remaining, key=lambda s: abs(s - short_strike_target))
            short_sym = contracts_by_strike[short_strike]

            # Get live quotes to check premiums
            quote_req = OptionLatestQuoteRequest(
                symbol_or_symbols=[long_sym, short_sym]
            )
            quotes = self._data_client.get_option_latest_quote(quote_req)

            long_quote = quotes.get(long_sym)
            short_quote = quotes.get(short_sym)

            if not long_quote or not short_quote:
                logger.info(f"Options: Couldn't get quotes for {symbol} {spread_type} spread")
                return None

            long_mid = (float(long_quote.bid_price or 0) + float(long_quote.ask_price or 0)) / 2
            short_mid = (float(short_quote.bid_price or 0) + float(short_quote.ask_price or 0)) / 2

            net_debit = long_mid - short_mid  # We pay this

            if net_debit < OPTIONS_CONFIG["min_debit"]:
                logger.info(f"Options: {symbol} {spread_type} spread debit too low (${net_debit:.2f})")
                return None

            spread_width = abs(long_strike - short_strike)
            if spread_width > 0 and net_debit / spread_width > OPTIONS_CONFIG["max_debit_pct_of_spread"]:
                logger.info(
                    f"Options: {symbol} {spread_type} spread too expensive "
                    f"(${net_debit:.2f} debit / ${spread_width:.0f} width = "
                    f"{net_debit/spread_width:.0%})"
                )
                return None

            logger.info(
                f"Options: Found {symbol} {spread_type} spread | "
                f"Long ${long_strike} ({long_sym}) @ ${long_mid:.2f} | "
                f"Short ${short_strike} ({short_sym}) @ ${short_mid:.2f} | "
                f"Net debit: ${net_debit:.2f}"
            )

            return long_sym, short_sym, long_strike, short_strike, net_debit

        except Exception as e:
            logger.error(f"Options contract search error {symbol}: {e}")
            return None

    # ── Direction Bias ────────────────────────────────────────────────

    def get_direction_bias(self, symbol: str) -> str:
        """
        Check 5-day momentum to decide put spread vs call spread.
        If stock down >3% over 5 days → bearish → put spread.
        If stock up >3% over 5 days → bullish → call spread.
        Otherwise → put spread (default — earnings usually disappoint).
        """
        if self.broker is None:
            return "put"
        try:
            df = self.broker.get_bars(symbol, "1Day", 10)
            if df is None or len(df) < 5:
                return "put"
            ret5 = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]
            if ret5 > 0.03:
                return "call"
            return "put"
        except Exception:
            return "put"

    # ── Entry ─────────────────────────────────────────────────────────

    def check_entries(self, account_equity: float) -> List[EarningsSpread]:
        if self._trade_client is None:
            return []
        if len(self.active_plays) >= OPTIONS_CONFIG["max_concurrent_plays"]:
            return []

        new_plays = []

        for sym in EARNINGS_CANDIDATES:
            if sym in self.active_plays:
                continue

            dte = self.days_to_earnings(sym)
            if dte is None:
                continue

            target = OPTIONS_CONFIG["days_before_entry"]
            if not (target - 1 <= dte <= target + 1):
                continue

            logger.info(f"Options: {sym} earnings in {dte} days — evaluating spread")

            # Get stock price
            try:
                price = self.broker.get_latest_price(sym) if self.broker else None
                if not price:
                    continue
            except Exception:
                continue

            # Target expiry: earnings date + expiry_days_out
            earnings_dt = self._earnings_calendar[sym]
            expiry_target = earnings_dt + timedelta(days=OPTIONS_CONFIG["expiry_days_out"])
            # Round to nearest Friday
            days_to_fri = (4 - expiry_target.weekday()) % 7
            if days_to_fri == 0:
                days_to_fri = 7
            expiry = expiry_target + timedelta(days=days_to_fri)

            # Decide direction
            direction = self.get_direction_bias(sym)
            spread_type = direction  # "put" or "call"

            # Find contracts
            result = self.find_spread_contracts(sym, spread_type, price, expiry)
            if result is None:
                continue

            long_sym, short_sym, long_strike, short_strike, net_debit = result

            # Size the position
            max_spend = account_equity * OPTIONS_CONFIG["max_position_pct"]
            cost_per_contract = net_debit * 100  # 1 contract = 100 shares
            contracts = max(1, int(max_spend / cost_per_contract))
            contracts = min(contracts, 10)
            total_cost = net_debit * contracts * 100

            # Enter the spread
            play = self._enter_spread(
                sym, spread_type, earnings_dt, expiry,
                long_sym, short_sym, long_strike, short_strike,
                net_debit, contracts, total_cost
            )

            if play:
                self.active_plays[sym] = play
                new_plays.append(play)
                time.sleep(1)

        return new_plays

    def _enter_spread(
        self, symbol, spread_type, earnings_date, expiry,
        long_sym, short_sym, long_strike, short_strike,
        net_debit, contracts, total_cost
    ) -> Optional[EarningsSpread]:
        """Submit both legs of the spread."""
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Buy the long leg
            buy_order = self._trade_client.submit_order(
                MarketOrderRequest(
                    symbol=long_sym,
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            time.sleep(0.5)

            # Sell the short leg
            sell_order = self._trade_client.submit_order(
                MarketOrderRequest(
                    symbol=short_sym,
                    qty=contracts,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )

            play = EarningsSpread(
                symbol=symbol,
                spread_type=f"{spread_type}_spread",
                earnings_date=earnings_date,
                entry_date=date.today(),
                long_contract=long_sym,
                short_contract=short_sym,
                long_strike=long_strike,
                short_strike=short_strike,
                expiry=expiry,
                net_debit=net_debit,
                contracts=contracts,
                total_cost=total_cost,
                status="open",
            )

            logger.info(
                f"Options: ENTERED {symbol} {spread_type} spread | "
                f"Long {long_sym} | Short {short_sym} | "
                f"Debit: ${net_debit:.2f}/sh | "
                f"{contracts} contracts | Total cost: ${total_cost:.0f}"
            )
            return play

        except Exception as e:
            logger.error(f"Options entry failed {symbol}: {e}")
            return None

    # ── Exit ──────────────────────────────────────────────────────────

    def check_exits(self):
        if self._trade_client is None or self._data_client is None:
            return

        today = date.today()

        for sym, play in list(self.active_plays.items()):
            if play.status != "open":
                continue

            days_after = (today - play.earnings_date).days
            should_exit = False
            exit_reason = ""

            # Time-based exit: 1 day after earnings
            if days_after >= OPTIONS_CONFIG["days_after_exit"]:
                should_exit = True
                exit_reason = f"Day after earnings ({days_after}d)"

            # Check current value for early exit
            if not should_exit:
                try:
                    from alpaca.data.requests import OptionLatestQuoteRequest
                    req = OptionLatestQuoteRequest(
                        symbol_or_symbols=[play.long_contract, play.short_contract]
                    )
                    quotes = self._data_client.get_option_latest_quote(req)

                    long_q = quotes.get(play.long_contract)
                    short_q = quotes.get(play.short_contract)

                    if long_q and short_q:
                        long_mid = (float(long_q.bid_price or 0) + float(long_q.ask_price or 0)) / 2
                        short_mid = (float(short_q.bid_price or 0) + float(short_q.ask_price or 0)) / 2
                        current_value = long_mid - short_mid

                        spread_width = abs(play.long_strike - play.short_strike)
                        max_profit = spread_width - play.net_debit

                        # Take profit at 70% of max profit
                        if max_profit > 0 and current_value >= play.net_debit + max_profit * OPTIONS_CONFIG["profit_target_pct"]:
                            should_exit = True
                            exit_reason = f"Profit target ({current_value:.2f} vs entry {play.net_debit:.2f})"

                        # Stop loss at 50% of debit paid
                        elif current_value <= play.net_debit * (1 - OPTIONS_CONFIG["stop_loss_pct"]):
                            should_exit = True
                            exit_reason = f"Stop loss (${current_value:.2f})"

                except Exception as e:
                    logger.debug(f"Options exit check error {sym}: {e}")

            if should_exit:
                self._exit_spread(play, exit_reason)

    def _exit_spread(self, play: EarningsSpread, reason: str):
        """Close both legs of the spread."""
        try:
            from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Sell the long leg
            self._trade_client.submit_order(
                MarketOrderRequest(
                    symbol=play.long_contract,
                    qty=play.contracts,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )
            time.sleep(0.5)

            # Buy back the short leg
            self._trade_client.submit_order(
                MarketOrderRequest(
                    symbol=play.short_contract,
                    qty=play.contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )

            # Estimate PnL (approximate — actual fills may differ)
            play.status = "closed"
            logger.info(
                f"Options: CLOSED {play.symbol} {play.spread_type} | "
                f"Reason: {reason} | "
                f"Cost basis: ${play.total_cost:.0f}"
            )

            del self.active_plays[play.symbol]

        except Exception as e:
            logger.error(f"Options exit failed {play.symbol}: {e}")

    # ── Daily Run ─────────────────────────────────────────────────────

    def run_daily_check(self, account_equity: float):
        """Call once per day at market open. Check exits then entries."""
        if self._trade_client is None:
            return

        logger.info("Options: Daily check running...")
        self.check_exits()
        new_plays = self.check_entries(account_equity)

        open_plays = [p for p in self.active_plays.values() if p.status == "open"]
        if open_plays:
            logger.info(f"Options: {len(open_plays)} active spread(s):")
            for p in open_plays:
                dte = (p.earnings_date - date.today()).days
                logger.info(
                    f"  {p.symbol} {p.spread_type} | "
                    f"Earnings in {dte}d | "
                    f"Cost: ${p.total_cost:.0f} | "
                    f"Long ${p.long_strike} Short ${p.short_strike}"
                )
        elif not new_plays:
            logger.info("Options: No active plays, no new entries")

    def summary(self) -> str:
        open_n = sum(1 for p in self.active_plays.values() if p.status == "open")
        return f"Options: {open_n} open spread(s)"