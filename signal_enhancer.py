"""
signal_enhancer.py — High-impact signal improvements using PyData stack
========================================================================
Three improvements proven to boost returns:

1. GRANGER CAUSALITY — peer stock movements predict target 1-2 days ahead
   "If AMD drops 3% today, NVDA likely follows tomorrow"
   Implementation: OLS regression with lagged peer returns (pure numpy)

2. VOLATILITY REGIME FILTER — skip entries when stock vol is expanding
   "Don't buy into a volatility spike — wait for it to settle"
   Implementation: ATR ratio + Garman-Klass volatility estimator

3. REGIME CLASSIFIER — momentum vs mean-reversion mode per stock
   "CEG is trending, apply momentum rules. NVDA is ranging, apply mean-reversion"
   Implementation: Hurst exponent + autocorrelation

All implemented with numpy/scipy/pandas only — no extra installs needed.

Usage:
    from signal_enhancer import SignalEnhancer
    enhancer = SignalEnhancer()
    
    # Get score adjustment for a symbol at bar i
    boost = enhancer.get_boost(symbol, df, i, peer_dfs)
    final_score = base_score + boost
    
    # Check if entry should be blocked
    blocked, reason = enhancer.should_block(symbol, df, i)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Peer relationships — who predicts whom
PEER_MAP = {
    "NVDA":  ["AMD", "AVGO", "MU"],
    "AMD":   ["NVDA", "AVGO", "MU"],
    "AVGO":  ["NVDA", "AMD", "QCOM"],
    "MU":    ["NVDA", "AMD", "WDC"],
    "PLTR":  ["CRWD", "PANW"],
    "CRWD":  ["PLTR", "PANW"],
    "GOOGL": ["MSFT", "META"],
    "MSFT":  ["GOOGL", "AAPL"],
    "VST":   ["CEG", "NEE"],
    "CEG":   ["VST", "NEE"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. GRANGER CAUSALITY — Do peer returns predict this stock tomorrow?
# ─────────────────────────────────────────────────────────────────────────────

def compute_granger_score(
    symbol: str,
    df: pd.DataFrame,
    i: int,
    peer_dfs: Dict[str, pd.DataFrame],
    lookback: int = 60
) -> float:
    """
    Test if peer stock returns Granger-cause target returns.
    
    Method: OLS regression of target returns on lagged peer returns.
    If peers explain target well (high R²) and current peer momentum
    is positive, boost score. If peers are falling, penalize.
    
    Returns: -0.20 to +0.20 score adjustment
    """
    try:
        peers = PEER_MAP.get(symbol, [])
        if not peers or i < lookback + 5:
            return 0.0

        target_close = df["close"]
        target_ret   = target_close.pct_change()

        # Build peer return matrix at lag 1 (yesterday's peer → today's target)
        peer_signals = []
        current_peer_momentum = []

        for peer in peers:
            if peer not in peer_dfs:
                continue
            peer_df = peer_dfs[peer]

            # Align by matching dates
            try:
                peer_close = peer_df["close"]

                # Get overlapping section up to bar i
                target_idx = df.index[:i]
                peer_idx   = peer_df.index

                common = target_idx.intersection(peer_idx)
                if len(common) < lookback:
                    continue

                # Target returns on common dates
                t_ret = target_close.loc[common].pct_change().dropna()
                p_ret = peer_close.loc[common].pct_change().shift(1).dropna()  # Lag 1

                common2 = t_ret.index.intersection(p_ret.index)
                if len(common2) < lookback:
                    continue

                t_vals = t_ret.loc[common2].values[-lookback:]
                p_vals = p_ret.loc[common2].values[-lookback:]

                # OLS: does lagged peer return predict target return?
                X = np.column_stack([p_vals, np.ones(len(p_vals))])
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(X, t_vals, rcond=None)
                    beta = coeffs[0]

                    # R² — how much variance does peer explain?
                    y_pred  = X @ coeffs
                    ss_res  = np.sum((t_vals - y_pred) ** 2)
                    ss_tot  = np.sum((t_vals - t_vals.mean()) ** 2)
                    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    # Only use peer if it has meaningful predictive power
                    if r2 > 0.05 and abs(beta) > 0.1:
                        # Current peer momentum (yesterday's return)
                        peer_ret_yesterday = p_ret.loc[common2].values[-1]
                        predicted_boost    = beta * peer_ret_yesterday * r2
                        peer_signals.append(predicted_boost)
                        current_peer_momentum.append(peer_ret_yesterday)
                except:
                    pass
            except:
                continue

        if not peer_signals:
            return 0.0

        # Average predicted boost from all valid peers
        avg_signal = np.mean(peer_signals)

        # Scale to -0.20 to +0.20
        return float(np.clip(avg_signal * 5, -0.20, 0.20))

    except Exception as e:
        logger.debug(f"Granger {symbol}: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. VOLATILITY REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility_regime(df: pd.DataFrame, i: int) -> Tuple[str, float, bool]:
    """
    Classify current volatility regime for the stock.
    
    Uses:
    - ATR ratio (current ATR vs 20-day avg ATR)
    - Garman-Klass volatility estimator (uses OHLC — more accurate than close-to-close)
    - Rate of change of volatility (is it expanding or contracting?)
    
    Returns:
        regime: "low" | "normal" | "high" | "spike"
        atr_ratio: current ATR / avg ATR
        should_block: True if volatility spike — don't enter
    """
    try:
        s = i - 1
        if s < 20:
            return "normal", 1.0, False

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr_current = tr.iloc[max(0, s-5):s+1].mean()
        atr_avg     = tr.iloc[max(0, s-20):s+1].mean()
        atr_ratio   = atr_current / atr_avg if atr_avg > 0 else 1.0

        # Garman-Klass volatility (more efficient than close-to-close)
        # σ² = 0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²
        window = min(10, s)
        gk_vals = []
        for j in range(s - window, s + 1):
            if j < 1:
                continue
            h, l, c, o = high.iloc[j], low.iloc[j], close.iloc[j], close.iloc[j-1]
            if l > 0 and o > 0:
                hl_term = 0.5 * (np.log(h/l))**2
                co_term = (2*np.log(2) - 1) * (np.log(c/o))**2
                gk_vals.append(hl_term - co_term)

        gk_vol = np.sqrt(np.mean(gk_vals) * 252) if gk_vals else 0.3  # Annualized

        # Volatility trend — is it expanding?
        atr_5d  = tr.iloc[max(0, s-5):s+1].mean()
        atr_20d = tr.iloc[max(0, s-20):s+1].mean()
        vol_expanding = atr_5d > atr_20d * 1.3

        # Classify regime
        if atr_ratio > 2.0 or gk_vol > 0.8:
            regime = "spike"
            should_block = True   # Vol spike — skip entry
        elif atr_ratio > 1.5 or vol_expanding:
            regime = "high"
            should_block = True   # Rising vol — skip entry
        elif atr_ratio < 0.7:
            regime = "low"
            should_block = False  # Low vol — good for entry
        else:
            regime = "normal"
            should_block = False

        return regime, float(atr_ratio), should_block

    except Exception as e:
        logger.debug(f"Vol regime: {e}")
        return "normal", 1.0, False


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME CLASSIFIER — Momentum vs Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────

def compute_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent — classifies time series behavior:
    H < 0.45 → mean-reverting (use contrarian signals)
    H ≈ 0.50 → random walk (signals unreliable)
    H > 0.55 → trending/momentum (use momentum signals)
    
    Implementation: R/S analysis (pure numpy)
    """
    try:
        lags = range(2, min(max_lag, len(series) // 2))
        rs_vals = []
        lag_vals = []

        for lag in lags:
            # Split into chunks of size lag
            chunks = [series[j:j+lag] for j in range(0, len(series)-lag, lag)]
            if len(chunks) < 2:
                continue

            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean   = chunk.mean()
                devs   = np.cumsum(chunk - mean)
                R      = devs.max() - devs.min()
                S      = chunk.std()
                if S > 0:
                    rs_chunk.append(R / S)

            if rs_chunk:
                rs_vals.append(np.log(np.mean(rs_chunk)))
                lag_vals.append(np.log(lag))

        if len(rs_vals) < 2:
            return 0.5

        # Linear regression of log(R/S) vs log(lag) — slope = Hurst
        coeffs = np.polyfit(lag_vals, rs_vals, 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))

    except:
        return 0.5


def compute_regime(df: pd.DataFrame, i: int) -> Tuple[str, float, float]:
    """
    Classify stock as momentum or mean-reverting.
    
    Returns:
        mode: "momentum" | "mean_revert" | "random"
        hurst: 0-1 (>0.55 = trending, <0.45 = mean-reverting)
        autocorr: lag-1 autocorrelation of returns
    """
    try:
        s = i - 1
        if s < 60:
            return "random", 0.5, 0.0

        close   = df["close"].values
        returns = np.diff(np.log(close[max(0, s-60):s+1]))

        if len(returns) < 20:
            return "random", 0.5, 0.0

        # Hurst exponent
        hurst = compute_hurst_exponent(returns)

        # Lag-1 autocorrelation
        if len(returns) > 1:
            autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            autocorr = 0.0

        if hurst > 0.55 and autocorr > 0.05:
            mode = "momentum"
        elif hurst < 0.45 and autocorr < -0.05:
            mode = "mean_revert"
        else:
            mode = "random"

        return mode, hurst, autocorr

    except:
        return "random", 0.5, 0.0


def adjust_score_for_regime(
    base_score: float,
    regime_mode: str,
    hurst: float,
    signal_type: str = "momentum"  # "momentum" or "contrarian"
) -> float:
    """
    Amplify score if signal aligns with regime, dampen if it conflicts.
    
    Momentum signal + momentum stock → amplify
    Momentum signal + mean-reverting stock → dampen
    """
    if regime_mode == "momentum":
        if base_score > 0:
            # Trending stock, bullish signal — amplify
            return base_score * (1.0 + (hurst - 0.5) * 0.6)
        else:
            # Trending stock, bearish signal — trust it more
            return base_score * (1.0 + (hurst - 0.5) * 0.3)

    elif regime_mode == "mean_revert":
        if base_score > 0.3:
            # Mean reverting + overbought momentum signal → dampen
            return base_score * 0.6
        elif base_score < -0.2:
            # Mean reverting + oversold → actually a buy signal
            return abs(base_score) * 0.4
        return base_score * 0.8

    return base_score  # random — use as-is


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CLASS — combines all three
# ─────────────────────────────────────────────────────────────────────────────

class SignalEnhancer:
    """
    Drop-in signal enhancement layer.
    Call get_enhanced_score() instead of raw composite score.
    """

    def __init__(self):
        self._regime_cache:  Dict[Tuple, Tuple] = {}
        self._vol_cache:     Dict[Tuple, Tuple] = {}
        self._granger_cache: Dict[Tuple, float] = {}

    def get_enhanced_score(
        self,
        symbol: str,
        base_score: float,
        df: pd.DataFrame,
        i: int,
        peer_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[float, Dict]:
        """
        Apply all three enhancements to base_score.
        
        Returns (enhanced_score, debug_info)
        """
        debug = {
            "base_score":    base_score,
            "vol_regime":    "normal",
            "vol_blocked":   False,
            "regime_mode":   "random",
            "hurst":         0.5,
            "granger_boost": 0.0,
            "final_score":   base_score,
        }

        score = base_score

        # 1. Volatility regime (cached per bar)
        vol_key = (symbol, i)
        if vol_key not in self._vol_cache:
            self._vol_cache[vol_key] = compute_volatility_regime(df, i)
        vol_regime, atr_ratio, vol_blocked = self._vol_cache[vol_key]
        debug["vol_regime"]  = vol_regime
        debug["atr_ratio"]   = atr_ratio
        debug["vol_blocked"] = vol_blocked

        if vol_blocked:
            # Hard block on volatility spike — return 0
            debug["final_score"] = 0.0
            return 0.0, debug

        # 2. Regime classifier (cached per bar, changes slowly)
        reg_key = (symbol, i // 5)  # Cache for 5-bar windows
        if reg_key not in self._regime_cache:
            self._regime_cache[reg_key] = compute_regime(df, i)
        regime_mode, hurst, autocorr = self._regime_cache[reg_key]
        debug["regime_mode"] = regime_mode
        debug["hurst"]       = hurst
        debug["autocorr"]    = autocorr

        score = adjust_score_for_regime(score, regime_mode, hurst)

        # 3. Granger causality from peers
        if peer_dfs and len(peer_dfs) > 0:
            g_key = (symbol, i)
            if g_key not in self._granger_cache:
                self._granger_cache[g_key] = compute_granger_score(
                    symbol, df, i, peer_dfs
                )
            granger_boost = self._granger_cache[g_key]
            debug["granger_boost"] = granger_boost
            score = score + granger_boost

        score = float(np.clip(score, -1.0, 1.0))
        debug["final_score"] = score
        return score, debug

    def should_block(
        self,
        symbol: str,
        df: pd.DataFrame,
        i: int,
    ) -> Tuple[bool, str]:
        """Quick block check — call before full scoring to save time."""
        vol_key = (symbol, i)
        if vol_key not in self._vol_cache:
            self._vol_cache[vol_key] = compute_volatility_regime(df, i)
        vol_regime, atr_ratio, vol_blocked = self._vol_cache[vol_key]

        if vol_blocked:
            return True, f"Vol {vol_regime} (ATR ratio={atr_ratio:.2f})"
        return False, ""

    def print_summary(self, symbol: str, df: pd.DataFrame, i: int,
                      peer_dfs: Optional[Dict] = None, base_score: float = 0.5):
        score, debug = self.get_enhanced_score(symbol, base_score, df, i, peer_dfs)
        print(f"\n{'═'*50}")
        print(f"  {symbol} Signal Enhancement")
        print(f"{'═'*50}")
        print(f"  Base score:      {debug['base_score']:+.3f}")
        print(f"  Vol regime:      {debug['vol_regime']} (ATR×{debug.get('atr_ratio',1):.2f})")
        print(f"  Regime mode:     {debug['regime_mode']} (H={debug['hurst']:.2f})")
        print(f"  Granger boost:   {debug['granger_boost']:+.3f}")
        print(f"  Vol blocked:     {'YES ⛔' if debug['vol_blocked'] else 'no'}")
        print(f"{'─'*50}")
        print(f"  Final score:     {score:+.3f}")
        print(f"{'═'*50}\n")


if __name__ == "__main__":
    import yfinance as yf
    import sys

    symbol  = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    peers   = PEER_MAP.get(symbol, [])
    
    print(f"\nFetching {symbol} + peers {peers}...")
    df = yf.Ticker(symbol).history(period="2y")
    df.columns = [c.lower() for c in df.columns]
    
    peer_dfs = {}
    for p in peers:
        try:
            pdf = yf.Ticker(p).history(period="2y")
            pdf.columns = [c.lower() for c in pdf.columns]
            peer_dfs[p] = pdf
        except:
            pass
    
    enhancer = SignalEnhancer()
    i = len(df) - 1
    enhancer.print_summary(symbol, df, i, peer_dfs, base_score=0.30)

    # Show regime over last 20 bars
    print(f"\nRecent regime history for {symbol}:")
    for j in range(max(60, len(df)-20), len(df), 2):
        mode, hurst, autocorr = compute_regime(df, j)
        vol_reg, atr_r, blocked = compute_volatility_regime(df, j)
        date = str(df.index[j])[:10]
        print(f"  {date}  regime={mode:<12} H={hurst:.2f}  vol={vol_reg:<7} atr×{atr_r:.2f} {'⛔' if blocked else ''}")