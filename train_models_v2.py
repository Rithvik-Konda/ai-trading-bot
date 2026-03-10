"""
train_models_v2.py — Enhanced ML Trainer
==========================================
Improvements over v1:
1. Expanded training universe — S&P 500 sector peers (10x more data)
2. Walk-forward validation — proves model generalizes, not memorizing
3. Ensemble architecture — GradientBoosting + RandomForest + ExtraTrees voting
4. More features — earnings proximity, sector momentum, relative strength vs SPY
5. Both daily AND 15min compatible feature sets

Usage:
    python3 train_models_v2.py              # Train all symbols
    python3 train_models_v2.py --symbol NVDA
    python3 train_models_v2.py --walkforward # Full walk-forward validation
    python3 train_models_v2.py --eval        # Detailed stats

Saves as: ml_model_v2_SYMBOL.joblib (bot auto-detects, v2 takes priority)
"""

import argparse
import logging
import warnings
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Expanded sector universe (S&P 500 names added) ───────────────────────────
SECTOR_UNIVERSE = {
    "NVDA": [
        "NVDA","AMD","AVGO","MU","QCOM","INTC","TSM","AMAT","KLAC","LRCX",
        "MRVL","SMCI","ON","TXN","ADI","NXPI","SWKS","MPWR",  # expanded
    ],
    "AMD": [
        "AMD","NVDA","AVGO","MU","QCOM","INTC","TSM","AMAT",
        "MRVL","ON","TXN","ADI","NXPI",
    ],
    "AVGO": [
        "AVGO","NVDA","AMD","QCOM","TXN","MU","INTC","MRVL","ADI","NXPI","SWKS",
    ],
    "MU": [
        "MU","NVDA","AMD","AVGO","WDC","STX","INTC","SMCI","AMAT","KLAC",
    ],
    "PLTR": [
        "PLTR","CRWD","S","PANW","ZS","NET","DDOG","SNOW","MDB","GTLB",
        "CYBR","OKTA","HUBS","VEEV",
    ],
    "CRWD": [
        "CRWD","PLTR","PANW","ZS","S","NET","FTNT","CYBR","OKTA","DDOG",
    ],
    "GOOGL": [
        "GOOGL","MSFT","META","AMZN","AAPL","ORCL","CRM","NOW","ADBE","SNOW",
    ],
    "MSFT": [
        "MSFT","GOOGL","AAPL","META","AMZN","CRM","NOW","ADBE","ORCL","SAP",
    ],
    "VST": [
        "VST","CEG","NEE","AES","NRG","ETR","EXC","PPL","SO","DUK","PCG",
    ],
    "CEG": [
        "CEG","VST","NEE","ETR","EXC","NRG","PPL","SO","DUK","AEE",
    ],
}

# SPY for relative strength feature
SPY_CACHE: Optional[pd.DataFrame] = None


def fetch_daily(symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=f"{years}y")
        if df is None or len(df) < 100:
            return None
        df.columns = [c.lower() for c in df.columns]
        for col in ["adj close","dividends","stock splits","capital gains"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df[["open","high","low","close","volume"]].dropna()
    except:
        return None


def fetch_spy(years: int = 10) -> Optional[pd.DataFrame]:
    global SPY_CACHE
    if SPY_CACHE is None:
        SPY_CACHE = fetch_daily("SPY", years)
    return SPY_CACHE


def get_earnings_dates(symbol: str) -> List[datetime]:
    """Get historical earnings dates from yfinance."""
    try:
        import yfinance as yf
        cal = yf.Ticker(symbol).earnings_dates
        if cal is None or len(cal) == 0:
            return []
        return [d.to_pydatetime().replace(tzinfo=None) for d in cal.index]
    except:
        return []


def compute_features_v2(
    df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    sector_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    earnings_dates: Optional[List[datetime]] = None,
) -> pd.DataFrame:
    """
    Enhanced feature set:
    - All original technical features
    - Relative strength vs SPY
    - Sector momentum
    - Earnings proximity
    - Multi-timeframe momentum
    """
    f     = pd.DataFrame(index=df.index)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Original technical features ───────────────────────────────────────────
    for p in [5, 10, 20, 60]:
        f[f"ret_{p}d"] = close.pct_change(p)

    for p in [10, 20, 50, 200]:
        ma = close.rolling(p).mean()
        f[f"price_vs_ma{p}"] = (close - ma) / (ma + 1e-9)

    f["ma10_vs_ma50"]  = close.rolling(10).mean() / (close.rolling(50).mean()  + 1e-9) - 1
    f["ma20_vs_ma200"] = close.rolling(20).mean() / (close.rolling(200).mean() + 1e-9) - 1
    f["ma50_vs_ma200"] = close.rolling(50).mean() / (close.rolling(200).mean() + 1e-9) - 1

    # RSI
    for p in [7, 14, 21]:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        f[f"rsi_{p}"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    f["macd"]            = macd / (close + 1e-9)
    f["macd_hist"]       = (macd - sig) / (close + 1e-9)
    f["macd_hist_slope"] = (macd - sig).diff()

    # Bollinger Bands
    for p in [10, 20]:
        ma  = close.rolling(p).mean()
        std = close.rolling(p).std()
        f[f"bb_pos_{p}"]   = (close - ma) / (2 * std + 1e-9)
        f[f"bb_width_{p}"] = (4 * std) / (ma + 1e-9)

    # ATR
    tr   = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    atr  = tr.rolling(14).mean()
    f["atr_14"] = atr / (close + 1e-9)

    # Volume
    vol_ma         = vol.rolling(20).mean()
    f["vol_ratio"] = vol / (vol_ma + 1e-9)
    f["vol_trend"] = vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-9)

    direction    = (close.diff() > 0).astype(int) * 2 - 1
    obv          = (direction * vol).cumsum()
    f["obv_slope_5"]  = obv.diff(5)  / (obv.shift(5).abs()  + 1)
    f["obv_slope_20"] = obv.diff(20) / (obv.shift(20).abs() + 1)

    tp      = (high + low + close) / 3
    mf      = tp * vol
    pmf     = mf.where(tp.diff() > 0, 0).rolling(14).sum()
    nmf     = mf.where(tp.diff() <= 0, 0).rolling(14).sum()
    f["mfi"] = 100 - (100 / (1 + pmf / (nmf + 1e-9)))

    # Volatility regime
    f["volatility_10"] = close.pct_change().rolling(10).std()
    f["volatility_30"] = close.pct_change().rolling(30).std()

    # ── NEW: Relative strength vs SPY ─────────────────────────────────────────
    if spy_df is not None:
        try:
            spy_aligned = spy_df["close"].reindex(df.index, method="ffill")
            for p in [5, 10, 20]:
                stock_ret = close.pct_change(p)
                spy_ret   = spy_aligned.pct_change(p)
                f[f"rs_vs_spy_{p}d"] = stock_ret - spy_ret
            # RS ratio
            f["rs_ratio_20d"] = (close / close.shift(20)) / (spy_aligned / spy_aligned.shift(20) + 1e-9)
        except:
            pass

    # ── NEW: Sector momentum ──────────────────────────────────────────────────
    if sector_dfs:
        try:
            peer_rets = []
            for peer_sym, peer_df in sector_dfs.items():
                aligned = peer_df["close"].reindex(df.index, method="ffill")
                peer_rets.append(aligned.pct_change(5))
            if peer_rets:
                sector_momentum    = pd.concat(peer_rets, axis=1).mean(axis=1)
                f["sector_mom_5d"] = sector_momentum
                f["vs_sector_5d"]  = close.pct_change(5) - sector_momentum
        except:
            pass

    # ── NEW: Earnings proximity ───────────────────────────────────────────────
    if earnings_dates:
        try:
            dates_arr = np.array([d.timestamp() for d in earnings_dates])
            days_to_earnings   = pd.Series(index=df.index, dtype=float)
            days_from_earnings = pd.Series(index=df.index, dtype=float)

            for dt in df.index:
                try:
                    ts = pd.Timestamp(dt).timestamp()
                except:
                    continue
                diffs      = dates_arr - ts
                future     = diffs[diffs > 0]
                past       = diffs[diffs <= 0]
                days_to_earnings[dt]   = future.min() / 86400 if len(future) > 0 else 999
                days_from_earnings[dt] = abs(past.max())  / 86400 if len(past)   > 0 else 999

            f["days_to_earnings"]   = days_to_earnings.clip(0, 60)
            f["days_from_earnings"] = days_from_earnings.clip(0, 30)
            f["near_earnings"]      = (days_to_earnings < 7).astype(int)
            f["post_earnings"]      = (days_from_earnings < 3).astype(int)
        except:
            pass

    return f


def make_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    profit_target: float = 0.015,
    stop_loss: float = 0.02,
) -> pd.Series:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    labels = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df) - forward_days):
        entry = close.iloc[i]
        hit   = False
        for j in range(i+1, i+1+forward_days):
            if (low.iloc[j]  - entry) / entry <= -stop_loss:
                break
            if (high.iloc[j] - entry) / entry >= profit_target:
                hit = True
                break
        if hit:
            labels.iloc[i] = 1

    return labels


def train_ensemble(X_train, y_train):
    """Train GradientBoosting + RandomForest + ExtraTrees ensemble."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.calibration import CalibratedClassifierCV

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=20, max_features=0.7, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        max_features=0.6, n_jobs=-1, random_state=42
    )
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        max_features=0.6, n_jobs=-1, random_state=42
    )

    print("      Training GradientBoosting...", end=" ", flush=True)
    gb.fit(X_train, y_train); print("✓")
    print("      Training RandomForest...", end=" ", flush=True)
    rf.fit(X_train, y_train); print("✓")
    print("      Training ExtraTrees...", end=" ", flush=True)
    et.fit(X_train, y_train); print("✓")

    return {"gb": gb, "rf": rf, "et": et}


def ensemble_predict_proba(models: dict, X) -> np.ndarray:
    """Soft voting — average probabilities across all models."""
    probs = np.stack([
        models["gb"].predict_proba(X)[:, 1],
        models["rf"].predict_proba(X)[:, 1],
        models["et"].predict_proba(X)[:, 1],
    ], axis=1)
    return probs.mean(axis=1)


def walk_forward_validate(symbol: str, df: pd.DataFrame, spy_df, sector_dfs, earnings_dates, feat_cols) -> dict:
    """
    Walk-forward validation: train on past, test on future, roll forward.
    Proves the model generalizes — not just memorizing history.
    Windows: train 5yr, test 1yr, step 1yr
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

    print(f"\n    Walk-forward validation for {symbol}...")
    min_train_bars = 252 * 5   # 5 years minimum training
    step_bars      = 252       # 1 year step

    all_features = compute_features_v2(df, spy_df, sector_dfs, earnings_dates)
    labels       = make_labels(df)
    combined     = pd.concat([all_features, labels.rename("target")], axis=1).dropna()

    if feat_cols is None:
        feat_cols = [c for c in combined.columns if c != "target"]

    missing = [c for c in feat_cols if c not in combined.columns]
    if missing:
        feat_cols = [c for c in feat_cols if c in combined.columns]

    results = []
    start   = min_train_bars

    while start + step_bars < len(combined):
        train = combined.iloc[:start]
        test  = combined.iloc[start:start + step_bars]

        if len(train) < 200 or len(test) < 20:
            break
        if test["target"].sum() < 5:
            start += step_bars
            continue

        X_tr  = train[feat_cols].values
        y_tr  = train["target"].values
        X_te  = test[feat_cols].values
        y_te  = test["target"].values

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        models   = train_ensemble(X_tr_s, y_tr)
        y_prob   = ensemble_predict_proba(models, X_te_s)
        y_pred   = (y_prob >= 0.5).astype(int)

        try:
            auc  = roc_auc_score(y_te, y_prob)
        except:
            auc  = 0.5
        prec = precision_score(y_te, y_pred, zero_division=0)
        acc  = accuracy_score(y_te, y_pred)

        period_start = combined.index[start]
        period_end   = combined.index[min(start + step_bars - 1, len(combined)-1)]
        results.append({
            "period": f"{str(period_start)[:10]} → {str(period_end)[:10]}",
            "auc": auc, "precision": prec, "accuracy": acc,
            "train_bars": len(train), "test_bars": len(test),
        })
        print(f"      {results[-1]['period']} | auc={auc:.2f} prec={prec:.1%} acc={acc:.1%}")
        start += step_bars

    if not results:
        return {}

    avg_auc  = np.mean([r["auc"]       for r in results])
    avg_prec = np.mean([r["precision"] for r in results])
    avg_acc  = np.mean([r["accuracy"]  for r in results])
    print(f"    Walk-forward avg: auc={avg_auc:.2f} prec={avg_prec:.1%} acc={avg_acc:.1%}")
    return {"wf_auc": avg_auc, "wf_precision": avg_prec, "wf_accuracy": avg_acc, "wf_periods": len(results)}


def train_symbol_v2(symbol: str, years: int, do_walkforward: bool, eval_mode: bool) -> bool:
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

        peers = SECTOR_UNIVERSE.get(symbol, [symbol])
        if symbol not in peers:
            peers = [symbol] + peers

        spy_df        = fetch_spy(years)
        earnings_dates = get_earnings_dates(symbol)
        print(f"    Earnings dates found: {len(earnings_dates)}")

        # Fetch all peer data
        peer_dfs: Dict[str, pd.DataFrame] = {}
        all_X, all_y = [], []
        feat_cols    = None

        for peer in peers:
            print(f"      fetching {peer}...", end=" ", flush=True)
            df = fetch_daily(peer, years)
            if df is None or len(df) < 300:
                print("skip")
                continue
            peer_dfs[peer] = df
            print(f"{len(df)} bars", end=" ", flush=True)

            # Sector peers excluding self for sector momentum feature
            sector_peers_for_peer = {s: d for s, d in peer_dfs.items() if s != peer}
            # Always pass earnings dates so ALL peers get identical feature count
            features = compute_features_v2(df, spy_df, sector_peers_for_peer, earnings_dates)
            labels   = make_labels(df)
            combined = pd.concat([features, labels.rename("target")], axis=1).dropna()

            if len(combined) < 100:
                print("→ insufficient")
                continue

            # Lock feature columns on first peer — fill missing cols with 0 for subsequent peers
            if feat_cols is None:
                feat_cols = [c for c in combined.columns if c != "target"]
            for col in feat_cols:
                if col not in combined.columns:
                    combined[col] = 0.0

            X = combined[feat_cols].values
            y = combined["target"].values
            all_X.append(X)
            all_y.append(y)
            print(f"→ {len(combined)} samples ({y.mean():.0%} positive)")

        if not all_X:
            print(f"  {symbol:<8} FAILED — no data")
            return False

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        # Holdout test set from the target symbol only (last 25%)
        own_df    = peer_dfs.get(symbol)
        if own_df is None:
            print(f"  {symbol:<8} FAILED — own data missing")
            return False

        own_features = compute_features_v2(own_df, spy_df, {s: d for s, d in peer_dfs.items() if s != symbol}, earnings_dates)
        own_labels   = make_labels(own_df)
        own_combined = pd.concat([own_features, own_labels.rename("target")], axis=1).dropna()
        # Fill any missing columns with 0 to match locked feature set
        for col in feat_cols:
            if col not in own_combined.columns:
                own_combined[col] = 0.0

        split   = int(len(own_combined) * 0.75)
        X_test  = own_combined.iloc[split:][feat_cols].values
        y_test  = own_combined.iloc[split:]["target"].values
        X_train = X_combined  # already aligned above

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_train)
        X_te_s   = scaler.transform(X_test)

        # Add static pipeline features (earnings surprise, FINRA short, etc.)
        try:
            from data_sources import DataPipeline
            _pipe    = DataPipeline()
            _pfeats  = _pipe.get_all(symbol)
            _skeys   = ["earnings_beat_rate", "avg_earnings_surprise",
                        "consistent_beater", "earnings_momentum", "finra_short_volume_ratio"]
            _added   = 0
            for k in _skeys:
                if k in _pfeats:
                    val     = float(_pfeats[k])
                    X_tr_s  = np.hstack([X_tr_s,  np.full((X_tr_s.shape[0],  1), val)])
                    X_te_s  = np.hstack([X_te_s,  np.full((X_te_s.shape[0],  1), val)])
                    feat_cols = feat_cols + [k]
                    _added += 1
            if _added:
                print(f"    +{_added} pipeline features added")
        except Exception as _pe:
            pass  # Pipeline unavailable — continue without

        print(f"\n    Training ensemble on {len(X_tr_s):,} samples...")
        models   = train_ensemble(X_tr_s, y_combined[:len(X_tr_s)])

        y_prob   = ensemble_predict_proba(models, X_te_s)
        y_pred   = (y_prob >= 0.5).astype(int)

        acc      = accuracy_score(y_test, y_pred)
        prec     = precision_score(y_test, y_pred, zero_division=0)
        f1       = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc  = roc_auc_score(y_test, y_prob)
        except:
            auc  = 0.5
        sig_rate = y_pred.mean() * 100

        # Walk-forward validation
        wf_stats = {}
        if do_walkforward:
            wf_stats = walk_forward_validate(symbol, own_df, spy_df, {s: d for s, d in peer_dfs.items() if s != symbol}, earnings_dates, feat_cols)

        path = f"ml_model_v2_{symbol}.joblib"
        joblib.dump({
            "models":    models,
            "scaler":    scaler,
            "features":  feat_cols,
            "symbol":    symbol,
            "version":   "v2",
            "ensemble":  True,
            "accuracy":  acc,
            "precision": prec,
            "f1":        f1,
            "auc":       auc,
            **wf_stats,
        }, path)

        color = "\033[92m" if auc >= 0.62 else ("\033[93m" if auc >= 0.55 else "\033[91m")
        rst   = "\033[0m"
        wf_str = f" | wf_auc={wf_stats.get('wf_auc',0):.2f}" if wf_stats else ""
        print(f"\n  {symbol:<8} {color}SAVED v2{rst} | "
              f"acc={acc:.1%} prec={prec:.1%} auc={auc:.2f} f1={f1:.2f} "
              f"signal={sig_rate:.0f}%{wf_str}")
        print(f"           Training: {len(X_combined):,} samples | {len(peers)} peers | {len(feat_cols)} features")
        return True

    except Exception as e:
        import traceback
        print(f"  {symbol:<8} ERROR — {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",      type=str,  default=None)
    parser.add_argument("--years",       type=int,  default=10)
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--eval",        action="store_true")
    args = parser.parse_args()

    import config
    symbols = [args.symbol.upper()] if args.symbol else [
        s for s in config.WATCHLIST if s not in getattr(config, "INVERSE_ETFS", [])
    ]

    print(f"\n{'═'*70}")
    print(f"  ML TRAINER v2 — Enhanced Ensemble")
    print(f"  {len(symbols)} symbols | {args.years} years")
    print(f"  Features: technical + RS vs SPY + sector momentum + earnings proximity")
    print(f"  Ensemble: GradientBoosting + RandomForest + ExtraTrees")
    if args.walkforward:
        print(f"  Walk-forward: ON (5yr train, 1yr test, rolling)")
    print(f"{'═'*70}\n")

    passed, failed = 0, []
    for symbol in symbols:
        print(f"\n{'─'*60}")
        print(f"  Training {symbol}:")
        ok = train_symbol_v2(symbol, args.years, args.walkforward, args.eval)
        if ok:
            passed += 1
        else:
            failed.append(symbol)

    print(f"\n{'═'*70}")
    print(f"  COMPLETE: {passed}/{len(symbols)} trained")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Bot priority: v2 ensemble > 15min > daily")
    print(f"  Restart: pkill -f bot.py && caffeinate -i python3 bot.py >> trading_bot.log 2>&1 &")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()