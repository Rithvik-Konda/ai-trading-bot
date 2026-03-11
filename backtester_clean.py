from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

import config
from ml_model import compute_features
from risk_manager import Position
from strategy_core import (
    SignalSnapshot,
    compute_atr_pct,
    compute_rule_score,
    load_ranker_ensemble,
    market_regime,
    normalize_ohlcv,
    select_top_candidates,
    trend_bullish,
)

CACHE_DIR = "cache_prices"
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    reason: str
    ml_rank_pct: float
    rule_score: float
    combined_score: float


def cache_path(symbol: str, days: int) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")


def fetch_history(symbol: str, days: int, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(symbol, days)

    if (not refresh) and os.path.exists(path):
        df = pd.read_csv(path, index_col=0)

        # Robust timezone-safe index parsing
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[~idx.isna()].copy()
        idx = idx[~idx.isna()].tz_convert("UTC").tz_localize(None)
        df.index = idx

        df.columns = [str(c).lower() for c in df.columns]
        df = normalize_ohlcv(df)
        print(f"[cache] {symbol}: {len(df)} rows", flush=True)
        return df

    years = max(2, int(np.ceil(days / 365)))
    print(f"[download] {symbol} for ~{years}y ...", flush=True)

    df = yf.Ticker(symbol).history(period=f"{years}y", interval="1d", auto_adjust=False)
    if df is None or len(df) == 0:
        print(f"[warn] no data for {symbol}", flush=True)
        return pd.DataFrame()

    # Strip timezone before caching
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df.columns = [str(c).lower() for c in df.columns]
    for c in ["adj close", "dividends", "stock splits", "capital gains"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = normalize_ohlcv(df)
    df.to_csv(path)
    print(f"[ok] {symbol}: {len(df)} rows", flush=True)
    return df


def apply_fill_cost(price: float, qty: int, side: str) -> tuple[float, float]:
    slip = price * (config.SLIPPAGE_BPS / 10_000)
    fill = price + slip if side == "buy" else price - slip
    comm = min(qty * config.COMMISSION_PER_SHARE, price * qty * config.COMMISSION_MAX_PCT)
    return float(max(fill, 0.01)), float(comm)


def calc_stats(equity_curve: pd.Series, trades: List[Trade]) -> dict:
    rets = equity_curve.pct_change().dropna()

    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades": len(trades),
            "win_rate": 0.0,
        }

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = max(len(equity_curve) / 252, 1 / 252)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    sharpe = 0.0 if len(rets) == 0 or rets.std() == 0 else (rets.mean() / rets.std()) * np.sqrt(252)

    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1
    max_dd = float(dd.min()) if len(dd) else 0.0

    wins = [t for t in trades if t.pnl > 0]
    win_rate = 0.0 if not trades else len(wins) / len(trades)

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "trades": len(trades),
        "win_rate": float(win_rate),
    }


def stop_pct_for_symbol(df: pd.DataFrame) -> float:
    atr_pct = compute_atr_pct(df, config.ATR_PERIOD)
    if config.USE_ATR_STOPS:
        return float(max(config.FIXED_STOP_LOSS_PCT, atr_pct * config.ATR_STOP_MULTIPLIER))
    return float(config.FIXED_STOP_LOSS_PCT)


def build_panel_for_date(
    date: pd.Timestamp,
    available_symbols: List[str],
    feature_store: Dict[str, pd.DataFrame],
    feat_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for symbol in available_symbols:
        feat_df = feature_store.get(symbol)
        if feat_df is None or len(feat_df) == 0:
            continue

        if date not in feat_df.index:
            sub = feat_df.loc[:date]
            if len(sub) == 0:
                continue
            row = sub.iloc[-1].copy()
        else:
            row = feat_df.loc[date].copy()

        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        row_dict = row.to_dict()
        row_dict["symbol"] = symbol
        rows.append(row_dict)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows)

    cs_rank_cols = [c for c in feat_cols if c.endswith("_cs_rank")]
    base_cs_cols = [c[:-8] for c in cs_rank_cols]

    for base_col, rank_col in zip(base_cs_cols, cs_rank_cols):
        if base_col in panel.columns:
            panel[rank_col] = panel[base_col].rank(pct=True)
        else:
            panel[rank_col] = 0.5

    for c in feat_cols:
        if c not in panel.columns:
            panel[c] = 0.0

    panel[feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel


def batch_ml_scores_from_precomputed(
    panel: pd.DataFrame,
    bundle: dict,
) -> Dict[str, float]:
    if len(panel) == 0:
        return {}

    model = bundle["model"]
    scaler = bundle["scaler"]
    feat_cols = list(bundle["features"])

    for c in feat_cols:
        if c not in panel.columns:
            panel[c] = 0.0

    X = panel[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    preds = model.predict(scaler.transform(X))
    return {str(sym): float(pred) for sym, pred in zip(panel["symbol"], preds)}


def build_fast_snapshots(
    date: pd.Timestamp,
    available_symbols: List[str],
    hist: Dict[str, pd.DataFrame],
    rule_store: Dict[str, pd.Series],
    ml_scores: Dict[str, float],
) -> Dict[str, SignalSnapshot]:
    snapshots: Dict[str, SignalSnapshot] = {}

    if not ml_scores:
        return snapshots

    ml_series = pd.Series(ml_scores, dtype=float)
    ml_rank_pct = ml_series.rank(pct=True)

    for symbol in available_symbols:
        if symbol not in ml_scores:
            continue

        try:
            df = hist[symbol].loc[:date]
            if len(df) < 260:
                continue

            rule = float(rule_store[symbol].get(date, 0.0))
            ml = float(ml_scores[symbol])
            rank_pct = float(ml_rank_pct.get(symbol, 0.0))

            bull = trend_bullish(df)
            atr_pct = compute_atr_pct(df, config.ATR_PERIOD)
            stop_pct = stop_pct_for_symbol(df)

            combined = (0.75 * rank_pct) + (0.15 * ml) + (0.10 * rule)

            snapshots[symbol] = SignalSnapshot(
                symbol=symbol,
                rule_score=rule,
                ml_score=ml,
                ml_rank_pct=rank_pct,
                combined_score=combined,
                trend_bullish=bull,
                stop_pct=stop_pct,
                atr_pct=atr_pct,
            )
        except Exception:
            continue

    return snapshots


def return_corr_matrix_fast(
    date: pd.Timestamp,
    available_symbols: List[str],
    hist: Dict[str, pd.DataFrame],
    lookback: int,
) -> pd.DataFrame:
    frames = []
    for symbol in available_symbols:
        try:
            df = hist[symbol].loc[:date]
            if len(df) < lookback + 2:
                continue
            ret = df["close"].pct_change().tail(lookback).rename(symbol)
            frames.append(ret)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    rets = pd.concat(frames, axis=1).dropna(how="all")
    return rets.corr()


def run_backtest(days: int = 3650, refresh_cache: bool = False):
    print("[start] exact fast ML-ensemble dynamic-universe backtest", flush=True)

    symbols = list(config.WATCHLIST)
    all_symbols = symbols + [config.BENCHMARK_SYMBOL]

    hist: Dict[str, pd.DataFrame] = {}
    for s in all_symbols:
        try:
            hist[s] = fetch_history(s, days, refresh=refresh_cache)
        except Exception as e:
            print(f"[error] failed downloading {s}: {e}", flush=True)
            hist[s] = pd.DataFrame()

    hist = {k: v for k, v in hist.items() if len(v) > 0}
    print(f"[info] loaded symbols: {list(hist.keys())}", flush=True)

    if config.BENCHMARK_SYMBOL not in hist:
        raise RuntimeError("Missing benchmark history")

    spy = hist[config.BENCHMARK_SYMBOL]
    prices_by_symbol = {k: v for k, v in hist.items() if k != config.BENCHMARK_SYMBOL}
    symbols = list(prices_by_symbol.keys())

    if not symbols:
        raise RuntimeError("No tradable histories loaded")

    all_trade_dates = pd.DatetimeIndex([])
    for s in symbols:
        all_trade_dates = all_trade_dates.union(prices_by_symbol[s].index)
    all_trade_dates = all_trade_dates.intersection(spy.index).sort_values()

    print(f"[info] total calendar dates available: {len(all_trade_dates)}", flush=True)
    if len(all_trade_dates) < 300:
        raise RuntimeError(f"Not enough dates: {len(all_trade_dates)}")

    rankers = load_ranker_ensemble()
    feat_cols_union = sorted(set(
        list(rankers[3]["features"]) +
        list(rankers[5]["features"]) +
        list(rankers[7]["features"])
    ))
    print("[ok] ensemble rankers loaded", flush=True)

    print("[prep] precomputing features...", flush=True)
    feature_store: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            f = compute_features(prices_by_symbol[s])
            if f is None or len(f) == 0:
                feature_store[s] = pd.DataFrame()
            else:
                f = f.replace([np.inf, -np.inf], np.nan)
                feature_store[s] = f
        except Exception:
            feature_store[s] = pd.DataFrame()

    print("[prep] precomputing rule scores...", flush=True)
    rule_store: Dict[str, pd.Series] = {}
    for s in symbols:
        df = prices_by_symbol[s]
        scores = {}
        idx = df.index
        for i in range(219, len(idx)):
            d = idx[i]
            try:
                scores[d] = compute_rule_score(df.loc[:d])
            except Exception:
                scores[d] = 0.0
        rule_store[s] = pd.Series(scores, dtype=float)

    cash = config.INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    entry_meta: Dict[str, dict] = {}
    trades: List[Trade] = []
    equity = []

    lookback = 260
    total_steps = len(all_trade_dates) - 1 - lookback

    for step_idx, i in enumerate(range(lookback, len(all_trade_dates) - 1), start=1):
        date = all_trade_dates[i]
        next_date = all_trade_dates[i + 1]

        if (step_idx - 1) % 100 == 0:
            print(f"[progress] step {step_idx}/{total_steps}", flush=True)

        available_symbols = []
        close_prices = {}
        low_prices = {}
        open_next_prices = {}

        for s in symbols:
            full_df = prices_by_symbol[s]

            if date not in full_df.index:
                continue

            df_now = full_df.loc[:date]
            if len(df_now) < lookback:
                continue

            available_symbols.append(s)
            close_prices[s] = float(df_now["close"].iloc[-1])
            low_prices[s] = float(df_now["low"].iloc[-1])

            future_rows = full_df.loc[full_df.index > date]
            if len(future_rows) > 0:
                open_next_prices[s] = float(future_rows.iloc[0]["open"])

        if len(available_symbols) < 2:
            port_val = cash + sum(close_prices.get(s, p.entry_price) * p.qty for s, p in positions.items())
            equity.append((date, port_val))
            continue

        spy_window = spy.loc[:date]
        if len(spy_window) < lookback:
            continue

        regime = market_regime(spy_window)

        for s in list(positions.keys()):
            pos = positions[s]
            current_close = close_prices.get(s)
            current_low = low_prices.get(s)

            if current_close is None or current_low is None:
                continue

            pos.update_high(current_close)
            stop_px = pos.current_stop()

            exit_reason = None
            exit_ref_px = None

            if current_low <= stop_px:
                exit_reason = "stop"
                exit_ref_px = stop_px
            elif current_close >= pos.entry_price * (1 + config.TAKE_PROFIT_PCT):
                exit_reason = "take_profit"
                exit_ref_px = current_close
            elif pos.age_days(pd.Timestamp(date).to_pydatetime()) >= config.MAX_HOLD_DAYS:
                exit_reason = "max_hold"
                exit_ref_px = current_close

            if exit_reason is not None:
                fill, comm = apply_fill_cost(exit_ref_px, pos.qty, "sell")
                pnl = (fill - pos.entry_price) * pos.qty - comm
                cash += fill * pos.qty - comm

                meta = entry_meta.get(s, {})
                trades.append(
                    Trade(
                        symbol=s,
                        entry_date=pos.entry_time,
                        exit_date=str(date.date()),
                        entry_price=pos.entry_price,
                        exit_price=fill,
                        qty=pos.qty,
                        pnl=pnl,
                        reason=exit_reason,
                        ml_rank_pct=float(meta.get("ml_rank_pct", 0.0)),
                        rule_score=float(meta.get("rule_score", 0.0)),
                        combined_score=float(meta.get("combined_score", 0.0)),
                    )
                )
                del positions[s]
                entry_meta.pop(s, None)

        if config.ENABLE_REGIME_FILTER and (regime["spy_crash"] or regime["vol_halt"]):
            port_val = cash + sum(close_prices.get(s, p.entry_price) * p.qty for s, p in positions.items())
            equity.append((date, port_val))
            continue

        base_panel = build_panel_for_date(date, available_symbols, feature_store, feat_cols_union)
        if len(base_panel) == 0:
            port_val = cash + sum(close_prices.get(s, p.entry_price) * p.qty for s, p in positions.items())
            equity.append((date, port_val))
            continue

        scores_3 = batch_ml_scores_from_precomputed(base_panel.copy(), rankers[3])
        scores_5 = batch_ml_scores_from_precomputed(base_panel.copy(), rankers[5])
        scores_7 = batch_ml_scores_from_precomputed(base_panel.copy(), rankers[7])

        rank_3 = pd.Series(scores_3, dtype=float).rank(pct=True) if scores_3 else pd.Series(dtype=float)
        rank_5 = pd.Series(scores_5, dtype=float).rank(pct=True) if scores_5 else pd.Series(dtype=float)
        rank_7 = pd.Series(scores_7, dtype=float).rank(pct=True) if scores_7 else pd.Series(dtype=float)

        all_syms = sorted(set(rank_3.index) | set(rank_5.index) | set(rank_7.index))
        ensemble_scores = {}
        for sym in all_syms:
            vals = []
            wts = []
            if sym in rank_3.index:
                vals.append(float(rank_3[sym])); wts.append(0.25)
            if sym in rank_5.index:
                vals.append(float(rank_5[sym])); wts.append(0.35)
            if sym in rank_7.index:
                vals.append(float(rank_7[sym])); wts.append(0.40)
            if wts:
                ensemble_scores[sym] = float(np.average(vals, weights=wts))

        snapshots = build_fast_snapshots(
            date=date,
            available_symbols=available_symbols,
            hist=prices_by_symbol,
            rule_store=rule_store,
            ml_scores=ensemble_scores,
        )

        symbol_to_df_for_selection = {s: prices_by_symbol[s].loc[:date] for s in available_symbols}
        corr_matrix = return_corr_matrix_fast(
            date=date,
            available_symbols=available_symbols,
            hist=prices_by_symbol,
            lookback=config.CORRELATION_LOOKBACK_DAYS,
        )

        selected = select_top_candidates(
            snapshots=snapshots,
            symbol_to_df=symbol_to_df_for_selection,
            current_positions={},
            max_names=config.MAX_POSITIONS,
            corr_matrix=corr_matrix,
        )

        if step_idx % 200 == 0:
            print(
                f"[debug] {date.date()} available={len(available_symbols)} "
                f"snapshots={len(snapshots)} selected={len(selected)}",
                flush=True
            )

        for snap in selected:
            s = snap.symbol
            if s in positions:
                continue
            if s not in open_next_prices:
                continue

            px = open_next_prices[s]
            stop_pct = snap.stop_pct
            position_scalar = regime["position_scalar"]

            risk_budget = config.INITIAL_CAPITAL * config.RISK_PER_TRADE * position_scalar
            risk_per_share = px * stop_pct
            qty_by_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0

            max_dollars = min(
                config.INITIAL_CAPITAL * config.MAX_POSITION_WEIGHT * position_scalar,
                config.MAX_POSITION_DOLLARS,
                cash,
            )
            qty_by_cap = int(max_dollars / px) if px > 0 else 0
            qty = min(qty_by_risk, qty_by_cap)

            if qty <= 0:
                continue
            if len(positions) >= config.MAX_POSITIONS:
                break

            fill, comm = apply_fill_cost(px, qty, "buy")
            cost = fill * qty + comm
            if cost > cash:
                continue

            cash -= cost
            positions[s] = Position(
                symbol=s,
                qty=qty,
                entry_price=fill,
                entry_time=str(next_date.date()),
                stop_pct=stop_pct,
                initial_stop=fill * (1 - stop_pct),
                highest_price=fill,
                add_count=0,
            )
            entry_meta[s] = {
                "ml_rank_pct": snap.ml_rank_pct,
                "rule_score": snap.rule_score,
                "combined_score": snap.combined_score,
            }

        port_val = cash + sum(close_prices.get(s, p.entry_price) * p.qty for s, p in positions.items())
        equity.append((date, port_val))

    equity_curve = pd.Series(
        data=[v for _, v in equity],
        index=pd.to_datetime([d for d, _ in equity]),
        name="equity",
    )

    stats = calc_stats(equity_curve, trades)

    print("\n=== BACKTEST RESULTS ===", flush=True)
    print(f"Total Return : {stats['total_return']:.2%}", flush=True)
    print(f"CAGR         : {stats['cagr']:.2%}", flush=True)
    print(f"Sharpe       : {stats['sharpe']:.2f}", flush=True)
    print(f"Max Drawdown : {stats['max_drawdown']:.2%}", flush=True)
    print(f"Trades       : {stats['trades']}", flush=True)
    print(f"Win Rate     : {stats['win_rate']:.2%}", flush=True)

    if len(trades) > 0:
        pnl = pd.Series([t.pnl for t in trades])
        trade_df = pd.DataFrame([t.__dict__ for t in trades])
        print(f"Avg Trade PnL: {pnl.mean():.2f}", flush=True)
        print(f"Median PnL   : {pnl.median():.2f}", flush=True)
        print(f"Avg ML Rank% : {trade_df['ml_rank_pct'].mean():.2%}", flush=True)
        print(f"Avg Rule     : {trade_df['rule_score'].mean():.3f}", flush=True)
        print(f"Avg Combined : {trade_df['combined_score'].mean():.3f}", flush=True)
    else:
        print("[warn] no trades were generated", flush=True)

    return equity_curve, trades, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    run_backtest(days=args.days, refresh_cache=args.refresh_cache)


if __name__ == "__main__":
    main()