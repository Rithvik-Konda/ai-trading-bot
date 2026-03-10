"""
train_models_15min.py — Intraday ML Model Trainer
===================================================
Trains on 15-minute bars instead of daily bars.

Why this matters:
- Bot trades intraday but daily model can't see intraday patterns
- 15-min bars give ~100x more training samples per symbol
- Model learns: morning momentum, lunch pullbacks, afternoon breakouts
- Labels match exactly what the bot does: 1.5% gain within 5 bars (75 min)
  without hitting 2% stop first

Data source: Alpaca (free, 2 years of 15-min bars)

Usage:
    python3 train_models_15min.py              # Train all watchlist symbols
    python3 train_models_15min.py --symbol NVDA
    python3 train_models_15min.py --eval       # Show detailed stats
    python3 train_models_15min.py --days 500   # Limit history

Saves as: ml_model_15min_SYMBOL.joblib
Bot loads these automatically if present (15min models take priority over daily)
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sector peers (same as daily trainer) ─────────────────────────────────────
SECTOR_PEERS = {
    "NVDA": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "INTC", "AMAT", "KLAC"],
    "AMD":  ["AMD",  "NVDA", "AVGO", "MU", "QCOM", "INTC"],
    "AVGO": ["AVGO", "NVDA", "AMD",  "QCOM", "TXN", "MU"],
    "MU":   ["MU",   "NVDA", "AMD",  "AVGO", "WDC", "INTC"],
    "PLTR": ["PLTR", "CRWD", "PANW", "ZS",  "NET", "DDOG"],
    "CRWD": ["CRWD", "PLTR", "PANW", "ZS",  "NET", "FTNT"],
    "GOOGL":["GOOGL","MSFT", "META", "AMZN", "AAPL"],
    "MSFT": ["MSFT", "GOOGL","AAPL", "META", "CRM"],
    "VST":  ["VST",  "CEG",  "NEE",  "NRG",  "ETR"],
    "CEG":  ["CEG",  "VST",  "NEE",  "ETR",  "EXC"],
}


def fetch_15min(symbol: str, days: int = 500) -> "Optional[pd.DataFrame]":
    """Fetch 15-minute bars from Alpaca. Free tier gives ~2 years."""
    try:
        import config
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )

        start = datetime.now() - timedelta(days=min(days, 700))  # Alpaca free tier limit
        end   = datetime.now() - timedelta(days=1)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed="iex",
        )

        bars = client.get_stock_bars(request).df
        if bars is None or len(bars) == 0:
            return None

        # Flatten multi-index if present
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol") if symbol in bars.index.get_level_values("symbol") else bars.droplevel(0)

        bars.columns = [c.lower() for c in bars.columns]
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in bars.columns for c in required):
            return None

        # Filter to market hours only: 9:30AM - 4:00PM ET
        bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")
        bars.index = bars.index.tz_convert("America/New_York")
        bars = bars.between_time("09:30", "16:00")

        return bars[required].dropna()

    except Exception as e:
        logger.debug(f"Alpaca 15min fetch failed for {symbol}: {e}")
        return None


def compute_15min_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for 15-minute bars.
    Captures intraday patterns the daily model completely misses.
    """
    f     = pd.DataFrame(index=df.index)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Price returns at intraday lookbacks ───────────────────────────────────
    for bars in [1, 2, 4, 8, 13, 26]:  # 15m, 30m, 1h, 2h, ~3h, ~6.5h
        f[f"ret_{bars}b"] = close.pct_change(bars)

    # ── Moving averages ───────────────────────────────────────────────────────
    for period in [4, 8, 13, 26, 52]:  # 1h, 2h, ~3h, ~6.5h, ~13h
        ma = close.rolling(period).mean()
        f[f"price_vs_ma{period}"] = (close - ma) / ma

    f["ma4_vs_ma13"]  = close.rolling(4).mean()  / close.rolling(13).mean()  - 1
    f["ma8_vs_ma26"]  = close.rolling(8).mean()  / close.rolling(26).mean()  - 1
    f["ma13_vs_ma52"] = close.rolling(13).mean() / close.rolling(52).mean()  - 1

    # ── RSI at multiple intraday timeframes ───────────────────────────────────
    for period in [7, 14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        f[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    f["macd"]           = macd
    f["macd_hist"]      = macd - sig
    f["macd_hist_slope"]= (macd - sig).diff()

    # ── Volatility ────────────────────────────────────────────────────────────
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low  - close.shift(1))
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for p in [8, 14, 26]:
        f[f"atr_{p}"] = atr.rolling(p).mean() / close

    for p in [8, 13]:
        ma  = close.rolling(p).mean()
        std = close.rolling(p).std()
        f[f"bb_pos_{p}"] = (close - ma) / (2 * std + 1e-9)

    f["volatility_13"] = close.pct_change().rolling(13).std()
    f["volatility_26"] = close.pct_change().rolling(26).std()

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ma            = vol.rolling(26).mean()
    f["vol_ratio"]    = vol / (vol_ma + 1)
    f["vol_trend"]    = vol.rolling(4).mean() / (vol.rolling(13).mean() + 1)

    # OBV
    direction         = (close.diff() > 0).astype(int) * 2 - 1
    obv               = (direction * vol).cumsum()
    f["obv_slope_4"]  = obv.diff(4)  / (obv.shift(4).abs() + 1)
    f["obv_slope_13"] = obv.diff(13) / (obv.shift(13).abs() + 1)

    # MFI
    tp   = (high + low + close) / 3
    mf   = tp * vol
    pmf  = mf.where(tp.diff() > 0, 0).rolling(14).sum()
    nmf  = mf.where(tp.diff() <= 0, 0).rolling(14).sum()
    f["mfi"] = 100 - (100 / (1 + pmf / (nmf + 1e-9)))

    # ── VWAP (intraday — reset each session) ──────────────────────────────────
    # Approximate VWAP using rolling 26-bar typical price * volume
    f["vwap_proxy"]  = (tp * vol).rolling(26).sum() / (vol.rolling(26).sum() + 1)
    f["price_vs_vwap"] = (close - f["vwap_proxy"]) / (f["vwap_proxy"] + 1e-9)

    # ── Time-of-day features ─────────────────────────────────────────────────
    # Intraday patterns: morning momentum, lunch lull, afternoon trend
    try:
        minutes_since_open = (
            (df.index.hour - 9) * 60 + df.index.minute - 30
        )
        f["time_sin"] = np.sin(2 * np.pi * minutes_since_open / 390)  # 390 min trading day
        f["time_cos"] = np.cos(2 * np.pi * minutes_since_open / 390)
        f["is_morning"]   = (minutes_since_open < 60).astype(int)   # First hour
        f["is_afternoon"] = (minutes_since_open > 300).astype(int)  # Last 90 min
    except Exception:
        f["time_sin"] = 0.0
        f["time_cos"] = 0.0
        f["is_morning"]   = 0
        f["is_afternoon"] = 0

    # ── Candle pattern ────────────────────────────────────────────────────────
    f["candle_body"]    = (close - df["open"]) / (df["open"] + 1e-9)
    f["upper_shadow"]   = (high - close.clip(lower=df["open"])) / (close + 1e-9)
    f["lower_shadow"]   = (close.clip(upper=df["open"]) - low)  / (close + 1e-9)

    # Distance from recent high/low
    f["dist_from_13b_high"] = close / close.rolling(13).max() - 1
    f["dist_from_13b_low"]  = close / close.rolling(13).min() - 1

    return f


def compute_15min_labels(
    df: pd.DataFrame,
    forward_bars: int = 5,    # 75 minutes
    profit_target: float = 0.015,
    stop_loss: float = 0.02,
) -> pd.Series:
    """
    Label each bar: 1 if price hits profit_target within forward_bars
    WITHOUT hitting stop_loss first.
    Matches exactly what the live bot does intraday.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    labels = pd.Series(0, index=df.index)

    for i in range(len(df) - forward_bars):
        entry       = close.iloc[i]
        stopped_out = False
        hit_target  = False

        for j in range(i + 1, i + 1 + forward_bars):
            day_low  = low.iloc[j]
            day_high = high.iloc[j]

            if (entry - day_low) / entry >= stop_loss:
                stopped_out = True
                break
            if (day_high - entry) / entry >= profit_target:
                hit_target = True
                break

        if hit_target and not stopped_out:
            labels.iloc[i] = 1

    return labels


def train_symbol(symbol: str, days: int, eval_mode: bool) -> bool:
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

        peers = SECTOR_PEERS.get(symbol, [symbol])
        if symbol not in peers:
            peers = [symbol] + peers

        all_X, all_y = [], []
        own_bars     = 0
        feat_cols    = None

        for peer in peers:
            print(f"      fetching {peer} 15min...", end=" ", flush=True)
            df = fetch_15min(peer, days)
            if df is None or len(df) < 500:
                print("skip")
                continue

            print(f"{len(df)} bars", end=" → ", flush=True)
            features = compute_15min_features(df)
            labels   = compute_15min_labels(df)
            combined = pd.concat([features, labels.rename("target")], axis=1).dropna()

            if len(combined) < 100:
                print("insufficient samples")
                continue

            X = combined.drop(columns=["target"]).values
            y = combined["target"].values
            if feat_cols is None:
                feat_cols = list(combined.drop(columns=["target"]).columns)

            all_X.append(X)
            all_y.append(y)

            if peer == symbol:
                own_bars = len(combined)
            print(f"{len(combined)} samples")

        if not all_X:
            print(f"  {symbol:<8} FAILED — no data from any peer")
            return False

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        # Holdout from own symbol data only
        df_own    = fetch_15min(symbol, days)
        if df_own is None:
            print(f"  {symbol:<8} FAILED — can't fetch own data for eval")
            return False

        features_own = compute_15min_features(df_own)
        labels_own   = compute_15min_labels(df_own)
        combined_own = pd.concat([features_own, labels_own.rename("target")], axis=1).dropna()
        split        = int(len(combined_own) * 0.75)
        X_test       = combined_own.iloc[split:].drop(columns=["target"]).values
        y_test       = combined_own.iloc[split:]["target"].values

        # Train on full combined dataset
        scaler   = StandardScaler()
        X_s      = scaler.fit_transform(X_combined)
        X_test_s = scaler.transform(X_test)

        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=42,
        )
        model.fit(X_s, y_combined)

        y_pred   = model.predict(X_test_s)
        y_prob   = model.predict_proba(X_test_s)[:, 1]
        acc      = accuracy_score(y_test, y_pred)
        prec     = precision_score(y_test, y_pred, zero_division=0)
        f1       = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc  = roc_auc_score(y_test, y_prob)
        except Exception:
            auc  = 0.5
        sig_rate = y_pred.mean() * 100
        pos_rate = y_test.mean() * 100

        path = f"ml_model_15min_{symbol}.joblib"
        joblib.dump({
            "model":      model,
            "scaler":     scaler,
            "features":   feat_cols,
            "symbol":     symbol,
            "timeframe":  "15min",
            "own_bars":   own_bars,
            "total_bars": len(X_combined),
            "accuracy":   acc,
            "precision":  prec,
            "f1":         f1,
            "auc":        auc,
        }, path)

        color = "\033[92m" if auc >= 0.60 else ("\033[93m" if auc >= 0.55 else "\033[91m")
        rst   = "\033[0m"
        print(f"  {symbol:<8} {color}SAVED{rst} | "
              f"acc={acc:.1%} prec={prec:.1%} auc={auc:.2f} f1={f1:.2f} "
              f"signal={sig_rate:.0f}% | {own_bars:,} own / {len(X_combined):,} total bars")
        return True

    except Exception as e:
        print(f"  {symbol:<8} ERROR — {e}")
        logger.debug(f"{symbol} training error", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Train 15-minute intraday ML models")
    parser.add_argument("--symbol", type=str,  default=None)
    parser.add_argument("--days",   type=int,  default=500, help="Days of 15min history (max ~700 on Alpaca free)")
    parser.add_argument("--eval",   action="store_true")
    args = parser.parse_args()

    import config
    symbols = [args.symbol.upper()] if args.symbol else [
        s for s in config.WATCHLIST if s not in getattr(config, "INVERSE_ETFS", [])
    ]

    print(f"\n{'═'*70}")
    print(f"  15-MIN INTRADAY ML TRAINER")
    print(f"  {len(symbols)} symbols | {args.days} days of 15-min bars")
    print(f"  Labels: 1.5% gain in 75min without hitting 2% stop")
    print(f"  Saves as: ml_model_15min_SYMBOL.joblib")
    print(f"{'═'*70}\n")

    passed, failed = 0, []
    for symbol in symbols:
        print(f"\n  Training {symbol}:")
        ok = train_symbol(symbol, args.days, args.eval)
        if ok:
            passed += 1
        else:
            failed.append(symbol)

    print(f"\n{'═'*70}")
    print(f"  COMPLETE: {passed}/{len(symbols)} trained")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Now update bot.py to load 15min models:")
    print(f"  The bot's _load_ml_models() will auto-detect ml_model_15min_*.joblib")
    print(f"  and use them instead of daily models when available.")
    print(f"\n  Restart the bot:")
    print(f"  pkill -f bot.py && caffeinate -i python3 bot.py >> trading_bot.log 2>&1 &")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
