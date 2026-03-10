"""
Volume Analysis Module
======================
Analyzes volume patterns: OBV, MFI, volume spikes, accumulation/distribution.
Returns a normalized volume score in [-1, 1].
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import config


@dataclass
class VolumeSignals:
    """Container for volume analysis results."""
    symbol: str
    current_volume: float = 0.0
    avg_volume: float = 0.0
    volume_ratio: float = 1.0
    obv: float = 0.0
    obv_trend: str = "neutral"
    mfi: float = 50.0
    accumulation_distribution: float = 0.0
    ad_trend: str = "neutral"
    is_volume_spike: bool = False
    score: float = 0.0
    signals: list = field(default_factory=list)


class VolumeAnalyzer:
    """
    Analyzes volume patterns to confirm or contradict price moves.
    Volume precedes price — this module detects smart money activity.
    """

    # ── Indicator Calculations ───────────────────────────────────────────

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume: cumulative volume based on price direction."""
        direction = np.sign(close.diff())
        return (volume * direction).cumsum()

    @staticmethod
    def money_flow_index(
        high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14
    ) -> pd.Series:
        """Money Flow Index: volume-weighted RSI."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        delta = typical_price.diff()

        positive_flow = money_flow.where(delta > 0, 0.0)
        negative_flow = money_flow.where(delta < 0, 0.0)

        pos_sum = positive_flow.rolling(window=period).sum()
        neg_sum = negative_flow.rolling(window=period).sum()

        mfr = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        return mfi

    @staticmethod
    def accumulation_distribution(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Accumulation/Distribution Line: tracks money flow into/out of a stock."""
        hl_range = high - low
        hl_range = hl_range.replace(0, np.nan)
        clv = ((close - low) - (high - close)) / hl_range
        ad = (clv * volume).cumsum()
        return ad

    @staticmethod
    def volume_sma(volume: pd.Series, period: int) -> pd.Series:
        return volume.rolling(window=period).mean()

    @staticmethod
    def detect_volume_spike(volume: pd.Series, avg_volume: pd.Series, threshold: float) -> pd.Series:
        return volume > (avg_volume * threshold)

    @staticmethod
    def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend: measures the strength of price trends with volume."""
        pct_change = close.pct_change()
        vpt = (pct_change * volume).cumsum()
        return vpt

    # ── Full Analysis Pipeline ───────────────────────────────────────────

    def analyze(self, symbol: str, df: pd.DataFrame) -> VolumeSignals:
        """
        Analyze volume patterns from OHLCV DataFrame.
        Returns VolumeSignals with composite score.
        """
        if len(df) < config.VOLUME_SMA_PERIOD + 5:
            return VolumeSignals(symbol=symbol, score=0.0, signals=["Insufficient data"])

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        sig = VolumeSignals(symbol=symbol)
        sub_scores = []

        sig.current_volume = volume.iloc[-1]
        vol_sma = self.volume_sma(volume, config.VOLUME_SMA_PERIOD)
        sig.avg_volume = vol_sma.iloc[-1]
        sig.volume_ratio = sig.current_volume / sig.avg_volume if sig.avg_volume > 0 else 1.0

        # ── Volume Spike Detection ───────────────────────────────────
        sig.is_volume_spike = sig.volume_ratio >= config.VOLUME_SPIKE_THRESHOLD

        if sig.is_volume_spike:
            price_change = close.iloc[-1] - close.iloc[-2]
            if price_change > 0:
                sub_scores.append(0.8)
                sig.signals.append(f"Volume spike ({sig.volume_ratio:.1f}x) on UP move — strong buying")
            else:
                sub_scores.append(-0.8)
                sig.signals.append(f"Volume spike ({sig.volume_ratio:.1f}x) on DOWN move — heavy selling")
        elif sig.volume_ratio > 1.3:
            sub_scores.append(0.2 if close.iloc[-1] > close.iloc[-2] else -0.2)
            sig.signals.append(f"Above-average volume ({sig.volume_ratio:.1f}x)")
        else:
            sub_scores.append(0.0)
            sig.signals.append("Normal volume")

        # ── On-Balance Volume ────────────────────────────────────────
        if config.OBV_ENABLED:
            obv = self.on_balance_volume(close, volume)
            sig.obv = obv.iloc[-1]

            # OBV trend: compare short vs long SMA of OBV
            obv_short = obv.rolling(5).mean().iloc[-1]
            obv_long = obv.rolling(20).mean().iloc[-1]

            if obv_short > obv_long:
                sig.obv_trend = "bullish"
                sub_scores.append(0.5)
                sig.signals.append("OBV trending up (accumulation)")
            elif obv_short < obv_long:
                sig.obv_trend = "bearish"
                sub_scores.append(-0.5)
                sig.signals.append("OBV trending down (distribution)")
            else:
                sig.obv_trend = "neutral"
                sub_scores.append(0.0)

            # OBV divergence from price
            price_up = close.iloc[-1] > close.iloc[-5]
            obv_up = obv.iloc[-1] > obv.iloc[-5]
            if price_up and not obv_up:
                sub_scores.append(-0.6)
                sig.signals.append("Bearish OBV divergence (price up, volume down)")
            elif not price_up and obv_up:
                sub_scores.append(0.6)
                sig.signals.append("Bullish OBV divergence (price down, volume up)")

        # ── Money Flow Index ─────────────────────────────────────────
        mfi = self.money_flow_index(high, low, close, volume, config.MFI_PERIOD)
        sig.mfi = mfi.iloc[-1]

        if sig.mfi < config.MFI_OVERSOLD:
            sub_scores.append(0.7)
            sig.signals.append(f"MFI oversold ({sig.mfi:.1f}) — potential bounce")
        elif sig.mfi > config.MFI_OVERBOUGHT:
            sub_scores.append(-0.7)
            sig.signals.append(f"MFI overbought ({sig.mfi:.1f}) — potential pullback")
        else:
            normalized = (sig.mfi - 50) / 30
            sub_scores.append(-normalized * 0.3)

        # ── Accumulation/Distribution ────────────────────────────────
        ad = self.accumulation_distribution(high, low, close, volume)
        sig.accumulation_distribution = ad.iloc[-1]

        ad_short = ad.rolling(5).mean().iloc[-1]
        ad_long = ad.rolling(20).mean().iloc[-1]
        if ad_short > ad_long:
            sig.ad_trend = "accumulation"
            sub_scores.append(0.4)
            sig.signals.append("A/D line rising (accumulation)")
        elif ad_short < ad_long:
            sig.ad_trend = "distribution"
            sub_scores.append(-0.4)
            sig.signals.append("A/D line falling (distribution)")

        # ── Composite Score ──────────────────────────────────────────
        if sub_scores:
            sig.score = np.clip(np.mean(sub_scores), -1.0, 1.0)

        return sig
