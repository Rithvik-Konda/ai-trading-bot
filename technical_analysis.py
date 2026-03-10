"""
Technical Analysis Module
=========================
Computes indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, VWAP.
Returns a normalized technical score in [-1, 1].
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class TechnicalSignals:
    """Container for all computed technical indicator values."""
    symbol: str
    sma_fast: float = 0.0
    sma_slow: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    rsi: float = 50.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_pct: float = 0.5
    atr: float = 0.0
    vwap: float = 0.0
    current_price: float = 0.0
    score: float = 0.0  # normalized [-1, 1]
    signals: list = field(default_factory=list)


class TechnicalAnalyzer:
    """
    Runs a full suite of technical analysis on OHLCV data
    and produces a composite score.
    """

    def __init__(self):
        self.indicators = {}

    # ── Core Indicator Calculations ──────────────────────────────────────

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        pct_b = (series - lower) / (upper - lower)
        return upper, middle, lower, pct_b

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_tp_vol / cum_vol

    # ── Full Analysis Pipeline ───────────────────────────────────────────

    def analyze(self, symbol: str, df: pd.DataFrame) -> TechnicalSignals:
        """
        Run full technical analysis on a DataFrame with columns:
        ['open', 'high', 'low', 'close', 'volume']

        Returns TechnicalSignals with a composite score.
        """
        if len(df) < config.SMA_SLOW + 10:
            return TechnicalSignals(symbol=symbol, score=0.0, signals=["Insufficient data"])

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        price = close.iloc[-1]

        sig = TechnicalSignals(symbol=symbol, current_price=price)
        sub_scores = []

        # ── Moving Averages ──────────────────────────────────────────
        sig.sma_fast = self.sma(close, config.SMA_FAST).iloc[-1]
        sig.sma_slow = self.sma(close, config.SMA_SLOW).iloc[-1]
        sig.ema_fast = self.ema(close, config.EMA_FAST).iloc[-1]
        sig.ema_slow = self.ema(close, config.EMA_SLOW).iloc[-1]

        # Golden/death cross
        if sig.sma_fast > sig.sma_slow:
            sub_scores.append(0.6)
            sig.signals.append("SMA Golden Cross (bullish)")
        else:
            sub_scores.append(-0.6)
            sig.signals.append("SMA Death Cross (bearish)")

        # Price vs EMAs
        if price > sig.ema_fast > sig.ema_slow:
            sub_scores.append(0.5)
            sig.signals.append("Price above rising EMAs (bullish)")
        elif price < sig.ema_fast < sig.ema_slow:
            sub_scores.append(-0.5)
            sig.signals.append("Price below falling EMAs (bearish)")
        else:
            sub_scores.append(0.0)

        # ── MACD ─────────────────────────────────────────────────────
        macd_l, macd_s, macd_h = self.macd(
            close, config.EMA_FAST, config.EMA_SLOW, config.EMA_SIGNAL
        )
        sig.macd_line = macd_l.iloc[-1]
        sig.macd_signal = macd_s.iloc[-1]
        sig.macd_histogram = macd_h.iloc[-1]

        if sig.macd_histogram > 0 and macd_h.iloc[-2] <= 0:
            sub_scores.append(0.8)
            sig.signals.append("MACD bullish crossover")
        elif sig.macd_histogram < 0 and macd_h.iloc[-2] >= 0:
            sub_scores.append(-0.8)
            sig.signals.append("MACD bearish crossover")
        elif sig.macd_histogram > 0:
            sub_scores.append(0.3)
        else:
            sub_scores.append(-0.3)

        # ── RSI ──────────────────────────────────────────────────────
        rsi_series = self.rsi(close, config.RSI_PERIOD)
        sig.rsi = rsi_series.iloc[-1]

        if sig.rsi < config.RSI_OVERSOLD:
            sub_scores.append(0.7)
            sig.signals.append(f"RSI oversold ({sig.rsi:.1f})")
        elif sig.rsi > config.RSI_OVERBOUGHT:
            sub_scores.append(-0.7)
            sig.signals.append(f"RSI overbought ({sig.rsi:.1f})")
        else:
            # Linear scale between oversold and overbought
            normalized = (sig.rsi - 50) / 20  # maps 30-70 to roughly -1..1
            sub_scores.append(-normalized * 0.3)

        # ── Bollinger Bands ──────────────────────────────────────────
        bb_u, bb_m, bb_l, bb_pct = self.bollinger_bands(
            close, config.BB_PERIOD, config.BB_STD_DEV
        )
        sig.bb_upper = bb_u.iloc[-1]
        sig.bb_middle = bb_m.iloc[-1]
        sig.bb_lower = bb_l.iloc[-1]
        sig.bb_pct = bb_pct.iloc[-1]

        if sig.bb_pct < 0.0:
            sub_scores.append(0.6)
            sig.signals.append("Price below lower Bollinger Band (oversold)")
        elif sig.bb_pct > 1.0:
            sub_scores.append(-0.6)
            sig.signals.append("Price above upper Bollinger Band (overbought)")
        else:
            sub_scores.append((0.5 - sig.bb_pct) * 0.4)

        # ── ATR (volatility context) ─────────────────────────────────
        atr_series = self.atr(high, low, close, config.ATR_PERIOD)
        sig.atr = atr_series.iloc[-1]

        # ── VWAP ─────────────────────────────────────────────────────
        if config.VWAP_ENABLED:
            vwap_series = self.vwap(high, low, close, volume)
            sig.vwap = vwap_series.iloc[-1]
            if price > sig.vwap:
                sub_scores.append(0.3)
                sig.signals.append("Price above VWAP (bullish)")
            else:
                sub_scores.append(-0.3)
                sig.signals.append("Price below VWAP (bearish)")

        # ── Composite Score ──────────────────────────────────────────
        if sub_scores:
            sig.score = np.clip(np.mean(sub_scores), -1.0, 1.0)

        return sig
