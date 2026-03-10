"""
Broker Interface Module
=======================
Connects to Alpaca for paper trading.
Handles order submission, position queries, and market data.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Interface to Alpaca Trading API for paper trading.
    Wraps alpaca-trade-api for order management and market data.
    """

    def __init__(self):
        self.api = None
        self._connected = False
        self._init_connection()

    def _init_connection(self):
        try:
            import alpaca_trade_api as tradeapi

            self.api = tradeapi.REST(
                key_id=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                base_url=config.ALPACA_BASE_URL,
                api_version="v2",
            )
            # Add retry + timeout to prevent hanging on slow responses
            try:
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
                self.api._session.mount("https://", HTTPAdapter(max_retries=retry))
                # Hard timeout on all requests — prevents 2hr freezes on Alpaca hangs
                import requests
                original_send = self.api._session.send
                def send_with_timeout(*args, **kwargs):
                    kwargs.setdefault("timeout", 30)
                    return original_send(*args, **kwargs)
                self.api._session.send = send_with_timeout
            except Exception:
                pass
            # Test connection
            account = self.api.get_account()
            self._connected = True
            logger.info(f"Connected to Alpaca | Account: {account.id} | Equity: ${float(account.equity):,.2f}")
        except ImportError:
            logger.warning("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            self._connected = False
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            self._connected = False

    @property
    def connected(self):
        return self._connected

    # ── Account Info ─────────────────────────────────────────────────────

    def get_account(self) -> dict:
        if not self._connected:
            return {"error": "Not connected"}
        try:
            acct = self.api.get_account()
            return {
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "daily_pnl": float(acct.equity) - float(acct.last_equity),
            }
        except Exception as e:
            logger.error(f"Account fetch error: {e}")
            return {"error": str(e)}

    def get_positions(self) -> List[dict]:
        if not self._connected:
            return []
        try:
            positions = self.api.list_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "side": p.side,
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
                    "market_value": float(p.market_value),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Position fetch error: {e}")
            return []

    # ── Market Data ──────────────────────────────────────────────────────

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100):
        """
        Fetch OHLCV bars. Returns a pandas DataFrame.
        timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
        """
        if not self._connected:
            return None
        try:
            from alpaca_trade_api.rest import TimeFrame

            # Build TimeFrame safely — handles both old and new library versions
            try:
                from alpaca_trade_api.rest import TimeFrameUnit
                tf_map = {
                    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                    "1Day": TimeFrame(1, TimeFrameUnit.Day),
                }
            except ImportError:
                # Older library version — use class attributes
                tf_map = {
                    "1Min": TimeFrame.Minute,
                    "5Min": TimeFrame.Minute,     # fallback: 1Min if old lib
                    "15Min": TimeFrame.Minute,     # fallback: 1Min if old lib
                    "1Hour": TimeFrame.Hour,
                    "1Day": TimeFrame.Day,
                }

            tf = tf_map.get(timeframe, tf_map["1Day"])
            start = (datetime.now() - timedelta(days=limit * 2)).strftime("%Y-%m-%d")

            bars = self.api.get_bars(symbol, tf, start=start, limit=limit).df
            bars.columns = [c.lower() for c in bars.columns]

            # Ensure required columns
            required = ["open", "high", "low", "close", "volume"]
            if all(c in bars.columns for c in required):
                return bars[required]
            return bars
        except Exception as e:
            logger.error(f"Bar data fetch error for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        if not self._connected:
            return None
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices = {}
        for sym in symbols:
            p = self.get_latest_price(sym)
            if p is not None:
                prices[sym] = p
        return prices

    # ── Order Management ─────────────────────────────────────────────────

    def submit_order(
        self, symbol: str, qty: int, side: str,
        order_type: str = "market", time_in_force: str = "day",
        limit_price: float = None, stop_price: float = None,
    ) -> dict:
        """Submit an order to Alpaca."""
        if not self._connected:
            return {"error": "Not connected"}

        try:
            kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            if limit_price:
                kwargs["limit_price"] = str(limit_price)
            if stop_price:
                kwargs["stop_price"] = str(stop_price)

            order = self.api.submit_order(**kwargs)
            logger.info(f"Order submitted: {side} {qty}x {symbol} ({order_type}) | ID: {order.id}")
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": int(order.qty),
                "side": order.side,
                "type": order.type,
                "status": order.status,
            }
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {"error": str(e)}

    def cancel_order(self, order_id: str):
        if not self._connected:
            return
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
        except Exception as e:
            logger.error(f"Cancel error: {e}")

    def cancel_all_orders(self):
        if not self._connected:
            return
        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Cancel all error: {e}")

    def smart_order(
        self, symbol: str, qty: int, side: str,
        price: float = None, slippage_bps: int = 10,
        max_retries: int = 3, timeout_sec: int = 10,
    ) -> dict:
        """
        Smart order execution — minimizes slippage.
        
        1. Uses limit orders with small buffer (default 10bps = 0.1%)
        2. Checks for fill within timeout
        3. Retries with slightly wider limit if not filled
        4. Falls back to market order on final retry
        
        Args:
            slippage_bps: Max acceptable slippage in basis points (10 = 0.1%)
            max_retries: Number of attempts before market order fallback
            timeout_sec: Seconds to wait for fill per attempt
        """
        import time as _time

        if not self._connected:
            return {"error": "Not connected"}

        # Get current price if not provided
        if price is None:
            price = self.get_latest_price(symbol)
        if not price:
            return {"error": f"No price for {symbol}"}

        for attempt in range(max_retries):
            is_last = (attempt == max_retries - 1)

            if is_last:
                # Final attempt: market order to guarantee fill
                logger.info(f"Smart order: {symbol} attempt {attempt+1}/{max_retries} — market order fallback")
                return self.submit_order(symbol, qty, side, order_type="market")

            # Calculate limit price with buffer
            # Widen buffer on each retry: 10bps, 20bps, 30bps...
            buffer_pct = slippage_bps * (attempt + 1) / 10000
            if side == "buy":
                limit = round(price * (1 + buffer_pct), 2)
            else:
                limit = round(price * (1 - buffer_pct), 2)

            logger.info(
                f"Smart order: {side} {qty}x {symbol} "
                f"limit ${limit:.2f} (attempt {attempt+1}/{max_retries}, "
                f"buffer {slippage_bps * (attempt+1)}bps)"
            )

            result = self.submit_order(
                symbol, qty, side,
                order_type="limit",
                limit_price=limit,
                time_in_force="ioc",  # Immediate-or-cancel
            )

            if "error" in result:
                logger.warning(f"Smart order failed: {result['error']}")
                continue

            order_id = result.get("id")
            if not order_id:
                continue

            # Wait for fill
            _time.sleep(min(timeout_sec, 3))

            try:
                order = self.api.get_order(order_id)
                status = order.status

                if status == "filled":
                    filled_price = float(order.filled_avg_price)
                    slippage = abs(filled_price - price) / price * 10000
                    logger.info(
                        f"Smart order FILLED: {side} {qty}x {symbol} "
                        f"@ ${filled_price:.2f} (slippage: {slippage:.1f}bps)"
                    )
                    return {
                        "id": order.id,
                        "symbol": order.symbol,
                        "qty": int(order.filled_qty),
                        "side": order.side,
                        "type": "limit",
                        "status": "filled",
                        "filled_price": filled_price,
                    }
                elif status == "partially_filled":
                    filled_qty = int(order.filled_qty)
                    logger.info(f"Smart order partial fill: {filled_qty}/{qty} — accepting")
                    return {
                        "id": order.id,
                        "symbol": order.symbol,
                        "qty": filled_qty,
                        "side": order.side,
                        "type": "limit",
                        "status": "filled",
                        "filled_price": float(order.filled_avg_price),
                    }
                else:
                    # Not filled — cancel and retry with wider limit
                    self.cancel_order(order_id)
                    logger.info(f"Smart order not filled ({status}) — retrying wider")
                    # Refresh price for next attempt
                    new_price = self.get_latest_price(symbol)
                    if new_price:
                        price = new_price
            except Exception as e:
                logger.error(f"Smart order status check error: {e}")
                continue

        return {"error": "All smart order attempts failed"}

    # ── Market Status ────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        if not self._connected:
            return False
        try:
            # Hard time check — only trade 9:35 AM to 3:55 PM ET (avoid open/close volatility)
            import pytz
            from datetime import time as dtime
            et = pytz.timezone('America/New_York')
            now_et = __import__('datetime').datetime.now(et)
            market_start = dtime(9, 35)
            market_end = dtime(15, 55)
            if not (market_start <= now_et.time() <= market_end):
                return False
            # Also check Alpaca's clock (handles holidays)
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Clock check error: {e}")
            return False