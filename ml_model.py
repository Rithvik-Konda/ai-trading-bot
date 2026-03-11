from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import config

CACHE_DIR = "cache_prices"
os.makedirs(CACHE_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} │ {msg}", flush=True)


@dataclass
class RankerResult:
    horizon: int
    n_rows: int
    train_rows: int
    test_rows: int
    test_rmse: float
    test_corr: float
    top1_avg_return: float
    top3_avg_return: float


def cache_path(symbol: str, days: int) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")


def fetch_data(symbol: str, days: int = 3650, refresh: bool = False) -> Optional[pd.DataFrame]:
    path = cache_path(symbol, days)

    if (not refresh) and os.path.exists(path):
        df = pd.read_csv(path, index_col=0)

        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[~idx.isna()].copy()
        idx = idx[~idx.isna()].tz_convert("UTC").tz_localize(None)
        df.index = idx

        df.columns = [str(c).lower() for c in df.columns]
        log(f"INFO │ Loaded {len(df)} cached bars for {symbol}")

        return df[["open", "high", "low", "close", "volume"]].dropna()

    import yfinance as yf

    df = yf.Ticker(symbol).history(period="10y", interval="1d")

    if df is None or len(df) == 0:
        return None

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df.columns = [str(c).lower() for c in df.columns]

    for c in ["adj close", "dividends", "stock splits"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df[["open", "high", "low", "close", "volume"]].dropna()

    df.to_csv(path)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:

    close = df["close"]

    feat = pd.DataFrame(index=df.index)

    feat["ret_1"] = close.pct_change(1)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)
    feat["ret_20"] = close.pct_change(20)
    feat["ret_60"] = close.pct_change(60)

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    feat["px_vs_sma10"] = close / sma10 - 1
    feat["px_vs_sma20"] = close / sma20 - 1
    feat["px_vs_sma50"] = close / sma50 - 1
    feat["px_vs_sma200"] = close / sma200 - 1

    feat["mom_accel_5_20"] = feat["ret_5"] - feat["ret_20"]

    return feat.replace([np.inf, -np.inf], np.nan)


def compute_forward_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:

    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-horizon)

    return (exit_ / entry) - 1


def build_symbol_store(symbols: List[str], days: int, refresh: bool = False):

    store = {}

    for sym in symbols:

        df = fetch_data(sym, days, refresh)

        if df is None or len(df) < 260:
            continue

        feat = compute_features(df)

        store[sym] = {
            "prices": df,
            "features": feat,
        }

    return store


def build_panel_from_store(store: Dict[str, Dict[str, pd.DataFrame]], horizon: int):

    frames = []

    spy = fetch_data(config.BENCHMARK_SYMBOL)

    spy_feat = compute_features(spy)

    for sym in config.WATCHLIST:

        if sym not in store:
            continue

        df = store[sym]["prices"]
        feat = store[sym]["features"].copy()

        target_raw = compute_forward_return(df, horizon)

        common = feat.index.intersection(spy_feat.index)

        feat.loc[common, "ret_5_vs_spy"] = feat.loc[common, "ret_5"] - spy_feat.loc[common, "ret_5"]
        feat.loc[common, "ret_20_vs_spy"] = feat.loc[common, "ret_20"] - spy_feat.loc[common, "ret_20"]
        feat.loc[common, "ret_60_vs_spy"] = feat.loc[common, "ret_60"] - spy_feat.loc[common, "ret_60"]

        merged = pd.concat([feat, target_raw.rename("target_raw")], axis=1)

        merged["symbol"] = sym
        merged["date"] = merged.index

        frames.append(merged)

    panel = pd.concat(frames)

    panel["target_rank"] = panel.groupby("date")["target_raw"].rank(pct=True)

    feature_cols = [
        c
        for c in panel.columns
        if c not in {"date", "symbol", "target_raw", "target_rank"}
    ]

    for c in feature_cols:
        panel[f"{c}_cs_rank"] = panel.groupby("date")[c].rank(pct=True)

    panel = panel.dropna()

    return panel.reset_index(drop=True)


def train_ranker(panel: pd.DataFrame, horizon: int):

    feature_cols = [
        c
        for c in panel.columns
        if c not in {"date", "symbol", "target_raw", "target_rank"}
    ]

    panel = panel.sort_values(["date", "symbol"])

    unique_dates = sorted(panel["date"].unique())

    split_i = int(len(unique_dates) * 0.7)

    split_date = unique_dates[split_i]

    train_df = panel[panel["date"] < split_date]
    test_df = panel[panel["date"] >= split_date]

    X_train = train_df[feature_cols]
    y_train = train_df["target_rank"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_rank"]

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42 + horizon,
    )

    model.fit(X_train_s, y_train)

    pred_test = model.predict(X_test_s)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

    corr = float(np.corrcoef(pred_test, y_test)[0, 1])

    return model, scaler, feature_cols


def train_and_save_ensemble(symbols: List[str], days: int):

    store = build_symbol_store(symbols, days)

    panel_3 = build_panel_from_store(store, 3)
    panel_5 = build_panel_from_store(store, 5)
    panel_7 = build_panel_from_store(store, 7)

    for horizon, panel in [(3, panel_3), (5, panel_5), (7, panel_7)]:

        model, scaler, features = train_ranker(panel, horizon)

        bundle = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "horizon": horizon,
        }

        joblib.dump(bundle, f"cross_sectional_ranker_{horizon}d.joblib")

    joblib.dump(
        joblib.load("cross_sectional_ranker_5d.joblib"),
        "cross_sectional_ranker.joblib",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--save-ensemble", action="store_true")

    args = parser.parse_args()

    symbols = list(config.WATCHLIST)

    if args.save_ensemble:
        train_and_save_ensemble(symbols, args.days)


if __name__ == "__main__":
    main()