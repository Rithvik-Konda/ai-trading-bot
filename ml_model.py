"""
ML Signal Model — Gradient Boosted Entry Predictor
====================================================
Trains on historical price/volume features to predict
whether a trade entry will be profitable.

Replaces the rule-based composite score with a trained model
that learns which indicator combinations actually lead to profits.

Usage:
    python3 ml_model.py --symbol NVDA --days 730        # Train & evaluate on one stock
    python3 ml_model.py --all --days 730                 # Train on all watchlist stocks
    python3 ml_model.py --all --days 730 --save          # Save model for live trading

Requirements:
    pip3 install scikit-learn joblib
"""

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import config

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV data.
    Each row = one day's features that would be known at market close.
    """
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Price-based features ──────────────────────────────────
    # Returns at various lookbacks
    for period in [1, 2, 3, 5, 10, 20]:
        f[f"ret_{period}d"] = close.pct_change(period)

    # Moving averages ratios
    for period in [5, 10, 20, 50]:
        ma = close.rolling(period).mean()
        f[f"price_vs_ma{period}"] = (close - ma) / ma

    # MA crossovers
    f["ma5_vs_ma20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    f["ma10_vs_ma50"] = close.rolling(10).mean() / close.rolling(50).mean() - 1
    f["ma50_vs_ma200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1

    # ── Momentum indicators ───────────────────────────────────
    # RSI
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        f[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    f["macd"] = macd
    f["macd_signal"] = signal
    f["macd_hist"] = macd - signal
    f["macd_hist_slope"] = f["macd_hist"].diff()

    # ── Volatility features ───────────────────────────────────
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for period in [5, 10, 20]:
        f[f"atr_{period}"] = true_range.rolling(period).mean() / close

    # Bollinger Band position
    for period in [15, 20]:
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        f[f"bb_position_{period}"] = (close - ma) / (2 * std)

    # Historical volatility
    for period in [10, 20]:
        f[f"volatility_{period}"] = close.pct_change().rolling(period).std()

    # ── Volume features ───────────────────────────────────────
    vol_ma20 = volume.rolling(20).mean()
    f["vol_ratio"] = volume / vol_ma20
    f["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

    # OBV slope
    obv = ((close.diff() > 0).astype(int) * 2 - 1) * volume
    obv_cumsum = obv.cumsum()
    f["obv_slope_5"] = obv_cumsum.diff(5) / obv_cumsum.shift(5).replace(0, np.nan)
    f["obv_slope_10"] = obv_cumsum.diff(10) / obv_cumsum.shift(10).replace(0, np.nan)

    # Money Flow Index
    typical_price = (high + low + close) / 3
    mf = typical_price * volume
    pos_mf = mf.where(typical_price.diff() > 0, 0).rolling(14).sum()
    neg_mf = mf.where(typical_price.diff() <= 0, 0).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    f["mfi"] = 100 - (100 / (1 + mfr))

    # ── Pattern features ──────────────────────────────────────
    f["candle_body"] = (close - df["open"]) / df["open"]
    f["upper_shadow"] = (high - close.clip(lower=df["open"])) / close
    f["lower_shadow"] = (close.clip(upper=df["open"]) - low) / close

    # Consecutive up/down days
    up = (close.diff() > 0).astype(int)
    f["consec_up"] = up.groupby((up != up.shift()).cumsum()).cumcount() * up
    down = (close.diff() < 0).astype(int)
    f["consec_down"] = down.groupby((down != down.shift()).cumsum()).cumcount() * down

    # Distance from recent high/low
    f["dist_from_20d_high"] = close / close.rolling(20).max() - 1
    f["dist_from_20d_low"] = close / close.rolling(20).min() - 1
    f["dist_from_50d_high"] = close / close.rolling(50).max() - 1

    return f


def compute_labels(df: pd.DataFrame, forward_days: int = 5, profit_target: float = 0.015) -> pd.Series:
    """
    Label each day: 1 if price goes up > profit_target% within forward_days
    WITHOUT hitting the 2% stop loss first.
    Matches actual bot strategy: 2% stop, same-day exit, 1.5% realistic target.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    labels = pd.Series(0, index=df.index)
    STOP_LOSS = 0.02  # Must match config.BULL_STOP_LOSS_PCT

    for i in range(len(df) - forward_days):
        entry = close.iloc[i]
        stopped_out = False
        hit_target = False

        for j in range(i+1, i+1+forward_days):
            day_low = low.iloc[j]
            day_high = high.iloc[j]
            loss = (entry - day_low) / entry
            gain = (day_high - entry) / entry

            if loss >= STOP_LOSS:
                stopped_out = True
                break
            if gain >= profit_target:
                hit_target = True
                break

        if hit_target and not stopped_out:
            labels.iloc[i] = 1

    return labels


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_data(symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
    fetch_days = days + 250
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, config.ALPACA_BASE_URL)
        from alpaca_trade_api.rest import TimeFrame
        start = (datetime.now() - timedelta(days=fetch_days + 30)).strftime("%Y-%m-%d")
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        bars = api.get_bars(symbol, TimeFrame.Day, start=start, end=end, limit=fetch_days + 30).df
        if len(bars) > 0:
            bars.columns = [c.lower() for c in bars.columns]
            logger.info(f"Fetched {len(bars)} bars for {symbol}")
            return bars[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.warning(f"Alpaca failed: {e}")

    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=f"{fetch_days}d")
        df.columns = [c.lower() for c in df.columns]
        for col in ["adj close", "dividends", "stock splits"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        logger.info(f"Fetched {len(df)} bars (yfinance) for {symbol}")
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"yfinance failed: {e}")
    return None


# ─── Model Training ──────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    symbol: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importance: Dict[str, float]
    train_size: int
    test_size: int
    signal_rate: float  # % of days the model says BUY
    backtest_return: float
    buy_hold_return: float


def train_and_evaluate(symbol: str, df: pd.DataFrame, test_split: float = 0.3) -> Optional[ModelResult]:
    """Train model on historical data and evaluate on hold-out set."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn not installed. Run: pip3 install scikit-learn")
        return None

    # Build features and labels
    features = compute_features(df)
    labels = compute_labels(df)

    # Align and drop NaN
    combined = pd.concat([features, labels.rename("target")], axis=1).dropna()
    if len(combined) < 100:
        logger.warning(f"Not enough data for {symbol}: {len(combined)} rows")
        return None

    X = combined.drop(columns=["target"])
    y = combined["target"]

    # Time-based split (no lookahead)
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Predict
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Feature importance
    feat_imp = dict(sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:15])

    # Simple backtest on test period
    test_df = df.iloc[split_idx + 200:]  # Offset for NaN features
    if len(test_df) > len(y_pred):
        test_df = test_df.iloc[:len(y_pred)]
    elif len(y_pred) > len(test_df):
        y_pred = y_pred[:len(test_df)]

    # Simulate: buy when model says 1, sell after 10 days
    cash = 100000
    position = None
    for i in range(len(y_pred)):
        if i >= len(test_df):
            break
        price = test_df.iloc[i]["close"]
        if position is not None:
            hold_days = i - position[1]
            gain = (price - position[0]) / position[0]
            if gain >= 0.03 or gain <= -0.04 or hold_days >= 10:
                cash += position[2] * price
                position = None
        if position is None and y_pred[i] == 1:
            qty = int((cash * 0.5) / price) if price > 0 else 0
            if qty > 0:
                cash -= qty * price
                position = (price, i, qty)

    if position:
        cash += position[2] * test_df.iloc[-1]["close"]

    ml_return = ((cash / 100000) - 1) * 100
    bh_return = ((test_df.iloc[-1]["close"] / test_df.iloc[0]["close"]) - 1) * 100 if len(test_df) > 0 else 0

    signal_rate = (y_pred.sum() / len(y_pred)) * 100

    return ModelResult(
        symbol=symbol, accuracy=acc, precision=prec, recall=rec, f1=f1,
        feature_importance=feat_imp, train_size=len(X_train), test_size=len(X_test),
        signal_rate=signal_rate, backtest_return=ml_return, buy_hold_return=bh_return,
    )


def save_model(symbol: str, df: pd.DataFrame, path: str = "ml_signal_model.joblib"):
    """Train on all data and save for live trading."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        logger.error("Install: pip3 install scikit-learn joblib")
        return

    features = compute_features(df)
    labels = compute_labels(df)
    combined = pd.concat([features, labels.rename("target")], axis=1).dropna()

    X = combined.drop(columns=["target"])
    y = combined["target"]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    model.fit(X_s, y)

    joblib.dump({"model": model, "scaler": scaler, "features": list(X.columns)}, path)
    logger.info(f"Model saved to {path}")


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_result(r: ModelResult):
    ac = "\033[92m" if r.backtest_return > r.buy_hold_return else "\033[91m"
    rst = "\033[0m"

    print(f"""
{'═'*60}
  ML MODEL: {r.symbol}
{'═'*60}
  Model Performance
  ─────────────────────────────────────────
  Accuracy:        {r.accuracy:>8.1%}
  Precision:       {r.precision:>8.1%}
  Recall:          {r.recall:>8.1%}
  F1 Score:        {r.f1:>8.1%}
  Signal Rate:     {r.signal_rate:>7.1f}% of days
  Train / Test:    {r.train_size} / {r.test_size}

  Backtest (test period)
  ─────────────────────────────────────────
  ML Strategy:     {ac}{r.backtest_return:>+7.2f}%{rst}
  Buy & Hold:      {r.buy_hold_return:>+7.2f}%
  Alpha:           {ac}{r.backtest_return - r.buy_hold_return:>+7.2f}%{rst}

  Top Features (what the model learned)
  ─────────────────────────────────────────""")
    for feat, imp in list(r.feature_importance.items())[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<25} {imp:>.3f} {bar}")

    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ML Signal Model")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    args = parser.parse_args()

    symbols = config.WATCHLIST if args.all else [args.symbol]
    results = []

    for sym in symbols:
        if sym in config.INVERSE_ETFS:
            continue
        print(f"\n{'▶'*3} Training on {sym}...")
        df = fetch_data(sym, days=args.days)
        if df is None or len(df) < 300:
            continue

        r = train_and_evaluate(sym, df)
        if r:
            results.append(r)
            print_result(r)

            if args.save:
                save_model(sym, df, f"model_{sym.lower()}.joblib")

    if len(results) > 1:
        print(f"\n{'═'*75}")
        print(f"  ML MODEL SUMMARY")
        print(f"{'═'*75}")
        print(f"  {'Symbol':<8} {'Acc':>6} {'Prec':>6} {'F1':>6} {'ML Ret':>8} {'B&H':>8} {'Alpha':>8} {'SigRate':>8}")
        print(f"  {'─'*70}")

        for r in sorted(results, key=lambda x: x.backtest_return - x.buy_hold_return, reverse=True):
            alpha = r.backtest_return - r.buy_hold_return
            c = "\033[92m" if alpha > 0 else "\033[91m"
            rst = "\033[0m"
            print(f"  {r.symbol:<8} {r.accuracy:>5.0%} {r.precision:>5.0%} {r.f1:>5.0%} {c}{r.backtest_return:>+7.1f}%{rst} {r.buy_hold_return:>+7.1f}% {c}{alpha:>+7.1f}%{rst} {r.signal_rate:>7.1f}%")

        print(f"  {'─'*70}")
        avg_alpha = np.mean([r.backtest_return - r.buy_hold_return for r in results])
        pos_alpha = sum(1 for r in results if r.backtest_return > r.buy_hold_return)
        avg_prec = np.mean([r.precision for r in results])
        print(f"  Avg Precision: {avg_prec:.0%} | Avg Alpha: {avg_alpha:+.1f}% | Positive Alpha: {pos_alpha}/{len(results)}")
        print(f"{'═'*75}\n")


if __name__ == "__main__":
    main()