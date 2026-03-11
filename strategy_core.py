from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import config
from ml_model import compute_features


@dataclass
class SignalSnapshot:
    symbol: str
    rule_score: float
    ml_score: float
    ml_rank_pct: float
    combined_score: float
    trend_bullish: bool
    stop_pct: float
    atr_pct: float


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"missing column: {col}")
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(float)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"OHLCV missing column: {c}")
    out = out[needed].dropna().copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    df = normalize_ohlcv(df)
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    close = _safe_series(df, "close")

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    px = close.iloc[-1]

    if pd.isna(atr) or px <= 0:
        return config.FIXED_STOP_LOSS_PCT

    return float(atr / px)


def trend_bullish(df: pd.DataFrame) -> bool:
    df = normalize_ohlcv(df)
    close = _safe_series(df, "close")

    if len(close) < config.TREND_SMA_SLOW:
        return True

    sma_fast = close.rolling(config.TREND_SMA_FAST).mean()
    sma_slow = close.rolling(config.TREND_SMA_SLOW).mean()
    return bool(sma_fast.iloc[-1] > sma_slow.iloc[-1])


def compute_rule_score(df: pd.DataFrame) -> float:
    df = normalize_ohlcv(df)
    if len(df) < 220:
        return 0.0

    close = _safe_series(df, "close")
    high = _safe_series(df, "high")
    low = _safe_series(df, "low")
    volume = _safe_series(df, "volume")

    s = -1
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    tech = 0.0

    if sma50.iloc[s] > sma200.iloc[s]:
        tech += 0.15

    sma10 = close.rolling(10).mean()
    sma40 = close.rolling(40).mean()
    if sma10.iloc[s] > sma40.iloc[s]:
        tech += 0.10

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    loss_val = loss.iloc[s]
    rs = gain.iloc[s] / loss_val if pd.notna(loss_val) and loss_val != 0 else 1.0
    rsi = 100 - (100 / (1 + rs))

    if 40 < rsi < 68:
        tech += 0.15
    elif rsi > 75:
        tech -= 0.10
    elif rsi < 30:
        tech += 0.05

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    if hist.iloc[s] > 0 and hist.iloc[s] > hist.iloc[s - 1]:
        tech += 0.15
    elif hist.iloc[s] < 0:
        tech -= 0.10

    typical = (high + low + close) / 3
    vwap_proxy = typical.rolling(20).mean()
    if close.iloc[s] > vwap_proxy.iloc[s]:
        tech += 0.10

    vol_score = 0.0
    vol_ma = volume.rolling(20).mean()
    vol_ratio = volume.iloc[s] / vol_ma.iloc[s] if vol_ma.iloc[s] > 0 else 1.0

    if vol_ratio > 1.5:
        vol_score += 0.20
    elif vol_ratio > 1.2:
        vol_score += 0.10
    elif vol_ratio < 0.7:
        vol_score -= 0.10

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv_base = abs(obv.iloc[s - 5]) + 1
    obv_slope = (obv.iloc[s] - obv.iloc[s - 5]) / obv_base
    if obv_slope > 0:
        vol_score += 0.10
    else:
        vol_score -= 0.05

    sent = 0.0
    ret5 = (close.iloc[s] - close.iloc[s - 5]) / close.iloc[s - 5]
    ret20 = (close.iloc[s] - close.iloc[s - 20]) / close.iloc[s - 20]

    if ret5 > 0.02:
        sent += 0.15
    elif ret5 < -0.02:
        sent -= 0.15

    if ret20 > 0.05:
        sent += 0.10
    elif ret20 < -0.05:
        sent -= 0.10

    w = config.WEIGHTS
    score = tech * w["technical"] + vol_score * w["volume"] + sent * w["sentiment"]

    if config.TREND_FILTER_ENABLED:
        if sma50.iloc[s] > sma200.iloc[s] and score < 0:
            score *= 0.5
        elif sma50.iloc[s] <= sma200.iloc[s] and score > 0:
            score *= 0.5

    return float(max(-1.0, min(1.0, score)))


def load_ranker(path: str = "cross_sectional_ranker.joblib") -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("ranker artifact is not a dict")
    for k in ["model", "scaler", "features"]:
        if k not in bundle:
            raise ValueError(f"ranker artifact missing key: {k}")
    return bundle


def load_ranker_ensemble() -> Dict[int, dict]:
    return {
        3: load_ranker("cross_sectional_ranker_3d.joblib"),
        5: load_ranker("cross_sectional_ranker_5d.joblib"),
        7: load_ranker("cross_sectional_ranker_7d.joblib"),
    }


def batch_ml_scores(symbol_to_df: Dict[str, pd.DataFrame], ranker_bundle: dict) -> Dict[str, float]:
    model = ranker_bundle["model"]
    scaler = ranker_bundle["scaler"]
    feat_cols = list(ranker_bundle["features"])

    cs_rank_cols = [c for c in feat_cols if c.endswith("_cs_rank")]
    base_cs_cols = [c[:-8] for c in cs_rank_cols]

    rows = []

    for symbol, df in symbol_to_df.items():
        try:
            daily = normalize_ohlcv(df)
            if len(daily) < 260:
                continue

            feat = compute_features(daily)
            if feat is None or len(feat) == 0:
                continue

            row = feat.iloc[-1].copy()
            row_dict = row.to_dict()
            row_dict["symbol"] = symbol
            rows.append(row_dict)
        except Exception:
            continue

    if not rows:
        return {}

    panel = pd.DataFrame(rows)

    for base_col, rank_col in zip(base_cs_cols, cs_rank_cols):
        if base_col in panel.columns:
            panel[rank_col] = panel[base_col].rank(pct=True)
        else:
            panel[rank_col] = 0.5

    for c in feat_cols:
        if c not in panel.columns:
            panel[c] = 0.0

    panel[feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    preds = model.predict(scaler.transform(panel[feat_cols]))
    panel["pred"] = preds

    return {str(sym): float(pred) for sym, pred in zip(panel["symbol"], panel["pred"])}


def batch_ml_scores_ensemble(
    symbol_to_df: Dict[str, pd.DataFrame],
    ranker_ensemble: Dict[int, dict],
    weights: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    if weights is None:
        weights = {3: 0.30, 5: 0.40, 7: 0.30}

    preds_by_h = {}
    rankpct_by_h = {}

    for h, bundle in ranker_ensemble.items():
        scores = batch_ml_scores(symbol_to_df, bundle)
        if not scores:
            continue
        preds_by_h[h] = scores
        rankpct_by_h[h] = pd.Series(scores, dtype=float).rank(pct=True).to_dict()

    if not preds_by_h:
        return {}

    symbols = set()
    for d in rankpct_by_h.values():
        symbols.update(d.keys())

    final_scores = {}
    for sym in symbols:
        score = 0.0
        wsum = 0.0
        for h, ranks in rankpct_by_h.items():
            if sym in ranks:
                w = float(weights.get(h, 0.0))
                score += w * float(ranks[sym])
                wsum += w
        if wsum > 0:
            final_scores[sym] = score / wsum

    return final_scores


def realized_vol_annualized(df: pd.DataFrame, window: int = 20) -> float:
    df = normalize_ohlcv(df)
    close = _safe_series(df, "close")
    if len(close) < window + 1:
        return 0.0
    return float(close.pct_change().rolling(window).std().iloc[-1] * np.sqrt(252))


def market_regime(spy_df: pd.DataFrame) -> dict:
    spy = normalize_ohlcv(spy_df)
    close = _safe_series(spy, "close")

    if len(close) < 220:
        return {
            "is_bear": False,
            "spy_crash": False,
            "vol_halt": False,
            "position_scalar": 1.0,
        }

    sma200 = close.rolling(200).mean()
    is_bear = bool(close.iloc[-1] < sma200.iloc[-1])

    day_ret = float(close.pct_change().iloc[-1])
    spy_crash = day_ret <= config.SPY_CRASH_HALT_PCT

    vol_ann = realized_vol_annualized(spy, 20)
    vol_halt = vol_ann >= config.REALIZED_VOL_HALT

    return {
        "is_bear": is_bear,
        "spy_crash": spy_crash,
        "vol_halt": vol_halt,
        "position_scalar": config.BEAR_POSITION_SCALAR if is_bear else 1.0,
    }


def stop_pct_for_symbol(df: pd.DataFrame) -> float:
    atr_pct = compute_atr_pct(df, config.ATR_PERIOD)
    if config.USE_ATR_STOPS:
        return float(max(config.FIXED_STOP_LOSS_PCT, atr_pct * config.ATR_STOP_MULTIPLIER))
    return float(config.FIXED_STOP_LOSS_PCT)


def build_signal_snapshots(
    symbol_to_df: Dict[str, pd.DataFrame],
    ml_scores: Dict[str, float],
) -> Dict[str, SignalSnapshot]:
    snapshots: Dict[str, SignalSnapshot] = {}

    if not ml_scores:
        return snapshots

    ml_series = pd.Series(ml_scores, dtype=float)
    ml_rank_pct = ml_series.rank(pct=True)

    for symbol, df in symbol_to_df.items():
        try:
            if symbol not in ml_scores:
                continue

            rule = compute_rule_score(df)
            ml = float(ml_scores[symbol])
            rank_pct = float(ml_rank_pct.get(symbol, 0.0))

            bull = trend_bullish(df)
            atr_pct = compute_atr_pct(df, config.ATR_PERIOD)
            stop_pct = stop_pct_for_symbol(df)

            combined = (0.75 * rank_pct) + (0.15 * ml) + (0.10 * rule)

            snapshots[symbol] = SignalSnapshot(
                symbol=symbol,
                rule_score=float(rule),
                ml_score=float(ml),
                ml_rank_pct=float(rank_pct),
                combined_score=float(combined),
                trend_bullish=bool(bull),
                stop_pct=float(stop_pct),
                atr_pct=float(atr_pct),
            )
        except Exception:
            continue

    return snapshots


def return_corr_matrix(symbol_to_df: Dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    frames = []

    for symbol, df in symbol_to_df.items():
        try:
            daily = normalize_ohlcv(df)
            ret = _safe_series(daily, "close").pct_change().rename(symbol)
            frames.append(ret)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    rets = pd.concat(frames, axis=1).dropna(how="all")
    if len(rets) > lookback:
        rets = rets.tail(lookback)

    return rets.corr()


def bucket_of(symbol: str) -> str:
    return config.CORRELATION_BUCKETS.get(symbol, symbol)


def select_top_candidates(
    snapshots: Dict[str, SignalSnapshot],
    symbol_to_df: Dict[str, pd.DataFrame],
    current_positions: Optional[Dict[str, dict]] = None,
    max_names: Optional[int] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
) -> List[SignalSnapshot]:
    if current_positions is None:
        current_positions = {}
    if max_names is None:
        max_names = config.MAX_POSITIONS

    eligible = [
        s for s in snapshots.values()
        if s.rule_score >= config.RULE_THRESHOLD and s.ml_rank_pct >= config.ML_RANK_MIN_PCT
    ]

    if not eligible:
        return []

    eligible.sort(key=lambda x: x.combined_score, reverse=True)

    corr = corr_matrix if corr_matrix is not None else return_corr_matrix(
        symbol_to_df, config.CORRELATION_LOOKBACK_DAYS
    )

    selected: List[SignalSnapshot] = []
    bucket_weights: Dict[str, float] = {}

    for cand in eligible:
        if len(selected) >= max_names:
            break

        bucket = bucket_of(cand.symbol)
        current_bucket_weight = bucket_weights.get(bucket, 0.0)

        if current_bucket_weight >= config.MAX_CORRELATED_BUCKET_WEIGHT:
            continue

        penalty = 0.0
        if not corr.empty and selected:
            corrs = []
            for s in selected:
                try:
                    corrs.append(float(corr.loc[cand.symbol, s.symbol]))
                except Exception:
                    pass

            if corrs:
                avg_corr = float(np.nanmean(corrs))
                if avg_corr > config.CORRELATION_PENALTY_START:
                    penalty = (
                        avg_corr - config.CORRELATION_PENALTY_START
                    ) * config.CORRELATION_PENALTY_MULT

        adjusted_score = cand.combined_score - penalty

        if adjusted_score < config.COMBINED_SCORE_MIN:
            continue

        selected.append(
            SignalSnapshot(
                symbol=cand.symbol,
                rule_score=cand.rule_score,
                ml_score=cand.ml_score,
                ml_rank_pct=cand.ml_rank_pct,
                combined_score=adjusted_score,
                trend_bullish=cand.trend_bullish,
                stop_pct=cand.stop_pct,
                atr_pct=cand.atr_pct,
            )
        )
        bucket_weights[bucket] = current_bucket_weight + config.MAX_POSITION_WEIGHT

    selected.sort(key=lambda x: x.combined_score, reverse=True)
    return selected


def should_rotate(existing_score: float, new_score: float) -> bool:
    return (new_score - existing_score) >= config.ROTATION_SCORE_GAP