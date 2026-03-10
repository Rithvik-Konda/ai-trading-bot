"""
optimizer_backtest.py — Find optimal strategy parameters
=========================================================
Tests multiple parameter combinations and ranks by risk-adjusted return.
Tries: position sizing, thresholds, hold periods, stop losses.

Usage:
    python3 optimizer_backtest.py --days 3650
"""

import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import sys
import itertools

logging.basicConfig(level=logging.WARNING)

import config
sys.path.insert(0, '.')

# ── Copy core functions from backtester_v40 ───────────────────────────────────
COMMISSION_PER_SHARE = 0.005
COMMISSION_MAX_PCT   = 0.005
SLIPPAGE_BPS         = 5

def apply_fill_cost(price, qty, side):
    slip = price * (SLIPPAGE_BPS / 10_000)
    fill = price + slip if side == "buy" else price - slip
    comm = min(qty * COMMISSION_PER_SHARE, price * qty * COMMISSION_MAX_PCT)
    return max(fill, 0.01), comm

def fetch_data(symbol, days):
    try:
        import yfinance as yf
        years = max(1, days // 365)
        df = yf.Ticker(symbol).history(period=f"{years}y")
        if df is None or len(df) < 50:
            return None
        df.columns = [c.lower() for c in df.columns]
        for col in ["adj close","dividends","stock splits","capital gains"]:
            if col in df.columns: df = df.drop(columns=[col])
        return df[["open","high","low","close","volume"]].dropna()
    except:
        return None

def compute_rule_score(df, i, sma50, sma200):
    try:
        s = i - 1
        if s < 50: return 0.0
        close  = df["close"]
        volume = df["volume"]
        tech   = 0.0

        if sma50.iloc[s] > sma200.iloc[s]: tech += 0.15
        sma10 = close.rolling(10).mean()
        sma40 = close.rolling(40).mean()
        if sma10.iloc[s] > sma40.iloc[s]: tech += 0.10

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain.iloc[s] / loss.iloc[s] if loss.iloc[s] != 0 else 1
        rsi   = 100 - (100 / (1 + rs))
        if 40 < rsi < 65:  tech += 0.15
        elif rsi > 70:     tech -= 0.15
        elif rsi < 30:     tech += 0.05

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        hist  = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        if hist.iloc[s] > 0 and hist.iloc[s] > hist.iloc[s-1]: tech += 0.15
        elif hist.iloc[s] < 0: tech -= 0.10

        typical    = (df["high"] + df["low"] + close) / 3
        vwap_proxy = typical.rolling(20).mean()
        if close.iloc[s] > vwap_proxy.iloc[s]: tech += 0.10

        vol_ma    = volume.rolling(20).mean()
        vol_ratio = volume.iloc[s] / vol_ma.iloc[s] if vol_ma.iloc[s] > 0 else 1.0
        vol = 0.0
        if vol_ratio > 1.5:   vol += 0.20
        elif vol_ratio > 1.2: vol += 0.10
        elif vol_ratio < 0.7: vol -= 0.10

        direction = (close.diff() > 0).astype(int) * 2 - 1
        obv       = (direction * volume).cumsum()
        obv_slope = (obv.iloc[s] - obv.iloc[s-5]) / (abs(obv.iloc[s-5]) + 1)
        if obv_slope > 0: vol += 0.10
        else:             vol -= 0.05

        ret5  = (close.iloc[s] - close.iloc[s-5])  / close.iloc[s-5]
        ret20 = (close.iloc[s] - close.iloc[s-20]) / close.iloc[s-20]
        sent  = 0.0
        if ret5  >  0.02: sent += 0.15
        elif ret5  < -0.02: sent -= 0.15
        if ret20 >  0.05: sent += 0.10
        elif ret20 < -0.05: sent -= 0.10

        w     = config.WEIGHTS
        score = tech * w["technical"] + vol * w["volume"] + sent * w["sentiment"]
        if sma50.iloc[s] > sma200.iloc[s] and score < 0: score *= 0.5
        elif sma50.iloc[s] <= sma200.iloc[s] and score > 0: score *= 0.5
        return max(-1.0, min(1.0, score))
    except:
        return 0.0

# ── ML models loaded once at startup ─────────────────────────────────────────
_ML_CACHE = {}

def load_ml_models():
    global _ML_CACHE
    try:
        import joblib, glob
        from ml_model import compute_features
        for path in glob.glob("ml_model_*.joblib"):
            sym = path.replace("ml_model_","").replace(".joblib","").upper()
            bundle = joblib.load(path)
            if bundle.get("auc", 0) >= 0.55:
                _ML_CACHE[sym] = bundle
        print(f"  ML models loaded: {list(_ML_CACHE.keys())}")
    except Exception as e:
        print(f"  ML load failed: {e}")

# Pre-compute ML scores for all symbols/bars to avoid repeated disk reads
_ML_SCORES = {}   # (symbol, i) -> float or None
_RULE_SCORES = {} # (symbol, i) -> float

def precompute_ml_scores(dfs):
    """Compute all ML scores upfront — called once before optimization loop."""
    from ml_model import compute_features
    print("  Pre-computing ML scores for all bars...", flush=True)
    for sym, df in dfs.items():
        if sym not in _ML_CACHE:
            continue
        bundle    = _ML_CACHE[sym]
        model     = bundle["model"]
        scaler    = bundle["scaler"]
        feat_cols = bundle["features"]
        hits = 0
        for i in range(220, len(df)):
            try:
                sub_df   = df.iloc[max(0, i-260):i].copy()
                features = compute_features(sub_df)
                if len(features) == 0: continue
                row = features.iloc[-1]
                if any(c not in row.index for c in feat_cols): continue
                vals = row[feat_cols].values.reshape(1,-1)
                if np.any(np.isnan(vals)): continue
                prob = model.predict_proba(scaler.transform(vals))[0][1]
                _ML_SCORES[(sym, i)] = (prob - 0.5) * 2.0
                hits += 1
            except:
                pass
        print(f"    {sym}: {hits} ML scores computed")
    total_cached = len(_ML_SCORES)
    print(f"  Done. {total_cached} total scores cached.\n")

def compute_ml_score(symbol, df, i):
    return _ML_SCORES.get((symbol, i), None)

def precompute_rule_scores(dfs, sma50s, sma200s):
    """Precompute all rule scores upfront — called once before optimization."""
    print("  Pre-computing rule scores for all bars...", flush=True)
    for sym, df in dfs.items():
        hits = 0
        for i in range(220, len(df)):
            score = compute_rule_score(df, i, sma50s[sym], sma200s[sym])
            _RULE_SCORES[(sym, i)] = score
            hits += 1
        print(f"    {sym}: {hits} rule scores computed")
    print(f"  Done. {len(_RULE_SCORES)} rule scores cached.\n")

def compute_spy_gate(spy_df, date):
    try:
        if spy_df is None or date not in spy_df.index: return 0.0, False
        idx  = spy_df.index.get_loc(date)
        if idx <= 0: return 0.0, False
        chg  = (spy_df["close"].iloc[idx] - spy_df["close"].iloc[idx-1]) / spy_df["close"].iloc[idx-1]
        if chg <= -0.02: return 0.0, True
        if chg <= -0.01: return 0.10, False
        return 0.0, False
    except:
        return 0.0, False


def run_simulation(dfs, spy_df, sma50s, sma200s, all_dates, params):
    """Core simulation loop — parameterized."""
    MAX_POS      = params["max_positions"]
    STOP_PCT     = params["stop_loss"]
    THRESHOLD    = params["threshold"]
    MAX_HOLD     = params["max_hold_days"]
    SIZE_MODE    = params["size_mode"]   # "fixed" or "conviction"
    BASE_SIZE    = params["base_size"]   # base position size pct

    cash         = 100_000.0
    positions    = {}   # sym -> dict
    all_trades   = []
    equity_curve = []
    total_costs  = 0.0
    lookback     = 220

    date_list = [d for d in all_dates if all_dates.index(d) >= lookback]

    for date in date_list:
        bar_idxs = {}
        for sym, df in dfs.items():
            if date in df.index:
                bar_idxs[sym] = df.index.get_loc(date)

        if not bar_idxs: continue
        prices = {sym: dfs[sym]["close"].iloc[idx] for sym, idx in bar_idxs.items()}

        # ── STOP LOSS ──
        for sym in list(positions.keys()):
            if sym not in bar_idxs: continue
            pos      = positions[sym]
            idx      = bar_idxs[sym]
            day_low  = dfs[sym]["low"].iloc[idx]
            stop     = pos["entry"] * (1 - STOP_PCT)
            hold_days = idx - pos["entry_idx"]

            if day_low <= stop:
                fp, comm = apply_fill_cost(stop, pos["qty"], "sell")
                total_costs += comm
                cash += fp * pos["qty"] - comm
                pnl  = (fp - pos["entry"]) * pos["qty"] - comm
                all_trades.append({"pnl": pnl, "pnl_pct": (fp-pos["entry"])/pos["entry"]*100, "reason": "stop", "hold": hold_days})
                del positions[sym]

        # ── MAX HOLD EXIT ──
        for sym in list(positions.keys()):
            if sym not in bar_idxs: continue
            pos       = positions[sym]
            idx       = bar_idxs[sym]
            hold_days = idx - pos["entry_idx"]
            if hold_days >= MAX_HOLD:
                price    = prices.get(sym, pos["entry"])
                fp, comm = apply_fill_cost(price, pos["qty"], "sell")
                total_costs += comm
                cash += fp * pos["qty"] - comm
                pnl  = (fp - pos["entry"]) * pos["qty"] - comm
                all_trades.append({"pnl": pnl, "pnl_pct": (fp-pos["entry"])/pos["entry"]*100, "reason": "max_hold", "hold": hold_days})
                del positions[sym]

        # ── EOD CLOSE (if max_hold == 1, this is day trading) ──
        if MAX_HOLD == 1:
            for sym in list(positions.keys()):
                pos      = positions[sym]
                price    = prices.get(sym, pos["entry"])
                fp, comm = apply_fill_cost(price, pos["qty"], "sell")
                total_costs += comm
                cash += fp * pos["qty"] - comm
                pnl  = (fp - pos["entry"]) * pos["qty"] - comm
                all_trades.append({"pnl": pnl, "pnl_pct": (fp-pos["entry"])/pos["entry"]*100, "reason": "eod", "hold": 1})
                del positions[sym]

        # ── SCORES ──
        scores = {}
        for sym, idx in bar_idxs.items():
            if idx < lookback: continue
            rule  = _RULE_SCORES.get((sym, idx), 0.0)
            ml    = compute_ml_score(sym, dfs[sym], idx)
            score = ml * 0.60 + rule * 0.40 if ml is not None else rule
            scores[sym] = score

        # ── SPY GATE ──
        boost, halt = compute_spy_gate(spy_df, date)
        eff_thresh  = THRESHOLD + boost
        if halt:
            pos_val = sum(prices.get(s, p["entry"]) * p["qty"] for s, p in positions.items())
            equity_curve.append(cash + pos_val)
            continue

        # ── ROTATION ──
        if len(positions) >= MAX_POS and scores:
            weakest_sym   = min(positions.keys(), key=lambda s: positions[s]["score"])
            weakest_pos   = positions[weakest_sym]
            weakest_price = prices.get(weakest_sym, weakest_pos["entry"])
            weakest_pnl   = (weakest_price - weakest_pos["entry"]) / weakest_pos["entry"]
            candidates    = [(s, sc) for s, sc in scores.items() if s not in positions and sc >= eff_thresh]
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                best_sym, best_score = candidates[0]
                if best_score - weakest_pos["score"] >= 0.10 and abs(weakest_pnl) <= 0.005:
                    fp, comm = apply_fill_cost(weakest_price, weakest_pos["qty"], "sell")
                    total_costs += comm
                    cash += fp * weakest_pos["qty"] - comm
                    pnl  = (fp - weakest_pos["entry"]) * weakest_pos["qty"] - comm
                    all_trades.append({"pnl": pnl, "pnl_pct": pnl/max(1,weakest_pos["entry"]*weakest_pos["qty"])*100, "reason": "rotation", "hold": 1})
                    del positions[weakest_sym]

        # ── ENTRIES ──
        candidates = [(s, sc) for s, sc in scores.items() if s not in positions and sc >= eff_thresh]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for sym, score in candidates:
            if len(positions) >= MAX_POS: break
            pos_val = sum(prices.get(s, p["entry"]) * p["qty"] for s, p in positions.items())
            equity  = cash + pos_val
            if equity <= 0 or (pos_val / equity) >= 0.90: break

            idx      = bar_idxs[sym]
            next_idx = idx + 1
            if next_idx >= len(dfs[sym]): continue
            open_p   = dfs[sym]["open"].iloc[next_idx]
            fp, comm = apply_fill_cost(open_p, 1, "buy")

            # Conviction-based sizing
            if SIZE_MODE == "conviction":
                if score >= 0.45:   size_pct = min(BASE_SIZE * 1.8, 0.45)
                elif score >= 0.35: size_pct = min(BASE_SIZE * 1.3, 0.35)
                else:               size_pct = BASE_SIZE
            else:
                size_pct = BASE_SIZE

            qty = int(equity * size_pct / fp) if fp > 0 else 0
            if qty <= 0 or qty * fp + comm > cash: continue

            total_costs += comm
            cash -= qty * fp + comm
            positions[sym] = {"entry": fp, "qty": qty, "entry_idx": idx, "score": score}

        pos_val = sum(prices.get(s, p["entry"]) * p["qty"] for s, p in positions.items())
        equity_curve.append(cash + pos_val)

    # Close remaining
    for sym, pos in positions.items():
        price = prices.get(sym, pos["entry"])
        fp, comm = apply_fill_cost(price, pos["qty"], "sell")
        cash += fp * pos["qty"] - comm
        pnl  = (fp - pos["entry"]) * pos["qty"] - comm
        all_trades.append({"pnl": pnl, "pnl_pct": (fp-pos["entry"])/pos["entry"]*100, "reason": "final", "hold": 1})

    if not all_trades or not equity_curve:
        return None

    final    = cash + sum(prices.get(s, p["entry"]) * p["qty"] for s, p in positions.items())
    total_r  = (final / 100_000 - 1) * 100
    winners  = [t for t in all_trades if t["pnl"] > 0]
    losers   = [t for t in all_trades if t["pnl"] <= 0]
    win_rate = len(winners) / len(all_trades) * 100 if all_trades else 0
    avg_win  = np.mean([t["pnl_pct"] for t in winners]) if winners else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losers])  if losers  else 0
    gp       = sum(t["pnl"] for t in winners)
    gl       = abs(sum(t["pnl"] for t in losers))
    pf       = gp / gl if gl > 0 else 99.0

    eq       = np.array(equity_curve)
    dr       = np.diff(eq) / eq[:-1]
    sharpe   = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    peak     = np.maximum.accumulate(eq)
    max_dd   = abs(((eq - peak) / peak).min() * 100)
    annual   = total_r / (len(equity_curve) / 252)

    return {
        "total_return": total_r,
        "annual":       annual,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "win_rate":     win_rate,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "profit_factor":pf,
        "num_trades":   len(all_trades),
        "costs":        total_costs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3650)
    args = parser.parse_args()

    symbols = [s for s in config.WATCHLIST if s not in getattr(config, "INVERSE_ETFS", [])]

    print(f"\n{'═'*70}")
    print(f"  STRATEGY OPTIMIZER — {len(symbols)} symbols, {args.days} days")
    print(f"{'═'*70}\n")

    # Fetch all data once
    print("  Fetching data...")
    dfs = {}
    for sym in symbols:
        df = fetch_data(sym, args.days)
        if df is not None and len(df) > 260:
            dfs[sym] = df
            print(f"    {sym}: {len(df)} bars")

    spy_df = fetch_data("SPY", args.days)

    all_dates = sorted(set.intersection(*[set(df.index) for df in dfs.values()]))
    sma50s  = {sym: df["close"].rolling(50).mean()  for sym, df in dfs.items()}
    sma200s = {sym: df["close"].rolling(200).mean() for sym, df in dfs.items()}

    # ── Parameter grid ────────────────────────────────────────────────────────
    param_grid = {
        "max_positions": [3, 5, 8],
        "stop_loss":     [0.015, 0.02, 0.03],
        "threshold":     [0.20, 0.25, 0.30, 0.35],
        "max_hold_days": [1, 2, 3, 5],
        "size_mode":     ["fixed", "conviction"],
        "base_size":     [0.15, 0.20, 0.25, 0.30],
    }

    combinations = list(itertools.product(*param_grid.values()))
    keys         = list(param_grid.keys())
    total        = len(combinations)

    # Load and precompute all scores once — massive speedup for 1152 combinations
    load_ml_models()
    precompute_ml_scores(dfs)
    precompute_rule_scores(dfs, sma50s, sma200s)

    print(f"\n  Testing {total} parameter combinations...\n")

    results = []
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{total}...", flush=True)

        r = run_simulation(dfs, spy_df, sma50s, sma200s, all_dates, params)
        if r and r["annual"] > 0 and r["num_trades"] > 20:
            r["params"] = params
            results.append(r)

    if not results:
        print("  No profitable combinations found")
        return

    # Sort by a composite score: annual return * sharpe / max_drawdown
    def rank_score(r):
        return r["annual"] * r["sharpe"] / max(r["max_dd"], 1)

    results.sort(key=rank_score, reverse=True)

    print(f"\n{'═'*90}")
    print(f"  TOP 15 PARAMETER COMBINATIONS (ranked by Annual×Sharpe/Drawdown)")
    print(f"{'═'*90}")
    print(f"  {'Annual':>8} {'Total':>8} {'Sharpe':>7} {'MaxDD':>7} {'WR%':>5} {'PF':>5} {'Trades':>7} | Params")
    print(f"  {'─'*85}")

    for r in results[:15]:
        p   = r["params"]
        c   = "\033[92m" if r["annual"] >= 30 else ("\033[93m" if r["annual"] >= 15 else "\033[0m")
        rst = "\033[0m"
        print(f"  {c}{r['annual']:>+7.1f}%{rst} {r['total_return']:>+7.1f}% "
              f"{r['sharpe']:>7.2f} {r['max_dd']:>6.1f}% "
              f"{r['win_rate']:>4.0f}% {r['profit_factor']:>4.1f} {r['num_trades']:>7} | "
              f"pos={p['max_positions']} stop={p['stop_loss']*100:.1f}% "
              f"thresh={p['threshold']} hold={p['max_hold_days']}d "
              f"size={p['size_mode']}({p['base_size']*100:.0f}%)")

    best = results[0]
    print(f"\n{'═'*70}")
    print(f"  BEST COMBINATION:")
    print(f"{'═'*70}")
    for k, v in best["params"].items():
        print(f"    {k:<20} = {v}")
    print(f"\n  Annual Return:   {best['annual']:>+.1f}%")
    print(f"  Total Return:    {best['total_return']:>+.1f}%")
    print(f"  Sharpe:          {best['sharpe']:>.2f}")
    print(f"  Max Drawdown:    {best['max_dd']:>.1f}%")
    print(f"  Win Rate:        {best['win_rate']:>.1f}%")
    print(f"  Profit Factor:   {best['profit_factor']:>.2f}")
    print(f"\n  Apply these to config.py and backtester_v40.py to maximize returns.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()