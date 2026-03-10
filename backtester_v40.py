"""
Backtester v4.0 — Full Strategy Simulation
============================================
Simulates EXACTLY what the live bot now does:

  - ML model scoring (60% weight) + rule-based (40% weight)
  - SPY regime gate: down >1% → raise threshold, down >2% → halt
  - Entry window: 10AM–3:30PM only (simulated as day-of signals)
  - Max hold: 5 days (optimizer proved 53% annual vs 2% for day trading)
  - 5 position max, 85% max exposure
  - Rotation: swap weakest position if better signal appears (score gap >0.10)
  - 2% stop loss, trailing
  - Realistic slippage (5bps) + commission ($0.005/share)
  - No lookahead bias — signals generated at close, filled at next open

Usage:
    python3 backtester_v40.py --days 365          # 1 year, all watchlist
    python3 backtester_v40.py --days 730          # 2 years
    python3 backtester_v40.py --symbol NVDA       # Single symbol
    python3 backtester_v40.py --days 365 --no-ml  # Compare rule-based vs ML
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Execution costs ───────────────────────────────────────────────────────────
SLIPPAGE_BPS        = 5        # 5bps per side = 0.05%
COMMISSION_PER_SHARE = 0.005
COMMISSION_MAX_PCT   = 0.005   # Cap at 0.5% of trade value

MAX_POSITIONS   = 5
MAX_EXPOSURE    = 0.85
STOP_LOSS_PCT   = 0.02
BUY_THRESHOLD   = 0.35           # Raised 0.25→0.35: fewer trades, much higher quality
MAX_HOLD_DAYS   = 5              # Optimizer: 5-day hold = 53% annual vs 2% for 1-day
BASE_SIZE_PCT   = 0.20           # Optimizer: 20% base conviction sizing
ROTATION_GAP    = 0.10           # Score gap required to rotate positions

# ── Signal Enhancer (module-level so it's accessible inside simulation loop) ──
try:
    from signal_enhancer import SignalEnhancer as _SE
    _ENHANCER = _SE()
    print("  Signal enhancer loaded ✓")
except Exception as _enh_err:
    _ENHANCER = None
    print(f"  Signal enhancer not loaded: {_enh_err}")


def apply_fill_cost(price: float, qty: int, side: str):
    slip       = price * (SLIPPAGE_BPS / 10_000)
    fill_price = price + slip if side == "buy" else price - slip
    fill_price = max(fill_price, 0.01)
    commission = min(qty * COMMISSION_PER_SHARE, price * qty * COMMISSION_MAX_PCT)
    return fill_price, commission


def fetch_data(symbol: str, days: int) -> Optional[pd.DataFrame]:
    fetch_days = days + 260
    try:
        import yfinance as yf
        years = max(1, fetch_days // 365)
        df = yf.Ticker(symbol).history(period=f"{years}y")
        if df is None or len(df) < 50:
            return None
        df.columns = [c.lower() for c in df.columns]
        for col in ["adj close", "dividends", "stock splits", "capital gains"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        logger.debug(f"yfinance failed for {symbol}: {e}")
        return None


def compute_rule_score(df: pd.DataFrame, i: int, sma50: pd.Series, sma200: pd.Series) -> float:
    """Rule-based composite score at bar i-1 (signal generated at prior close, no lookahead)."""
    try:
        # Use i-1 — signal is generated at PREVIOUS bar close, filled at current bar open
        s = i - 1
        if s < 50:
            return 0.0
        close  = df["close"]
        volume = df["volume"]

        # ── Technical score ──
        tech = 0.0

        # Trend
        if sma50.iloc[s] > sma200.iloc[s]:
            tech += 0.15

        # SMA crossover
        sma10 = close.rolling(10).mean()
        sma40 = close.rolling(40).mean()
        if sma10.iloc[s] > sma40.iloc[s]:
            tech += 0.10

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain.iloc[s] / loss.iloc[s] if loss.iloc[s] != 0 else 1
        rsi   = 100 - (100 / (1 + rs))
        if 40 < rsi < 65:
            tech += 0.15
        elif rsi > 70:
            tech -= 0.15
        elif rsi < 30:
            tech += 0.05  # Oversold bounce potential

        # MACD histogram slope
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        hist  = macd - sig
        if hist.iloc[s] > 0 and hist.iloc[s] > hist.iloc[s-1]:
            tech += 0.15
        elif hist.iloc[s] < 0:
            tech -= 0.10

        # Price vs VWAP proxy (use 20-day avg price as VWAP proxy)
        typical = (df["high"] + df["low"] + close) / 3
        vwap_proxy = typical.rolling(20).mean()
        if close.iloc[s] > vwap_proxy.iloc[s]:
            tech += 0.10

        # ── Volume score ──
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume.iloc[s] / vol_ma.iloc[s] if vol_ma.iloc[s] > 0 else 1.0

        vol = 0.0
        if vol_ratio > 1.5:
            vol += 0.20
        elif vol_ratio > 1.2:
            vol += 0.10
        elif vol_ratio < 0.7:
            vol -= 0.10

        # OBV slope
        direction = (close.diff() > 0).astype(int) * 2 - 1
        obv = (direction * volume).cumsum()
        obv_slope = (obv.iloc[s] - obv.iloc[s-5]) / (abs(obv.iloc[s-5]) + 1)
        if obv_slope > 0:
            vol += 0.10
        else:
            vol -= 0.05

        # ── Sentiment (momentum proxy) ──
        ret5  = (close.iloc[s] - close.iloc[s-5])  / close.iloc[s-5]
        ret20 = (close.iloc[s] - close.iloc[s-20]) / close.iloc[s-20]
        sent = 0.0
        if ret5 > 0.02:
            sent += 0.15
        elif ret5 < -0.02:
            sent -= 0.15
        if ret20 > 0.05:
            sent += 0.10
        elif ret20 < -0.05:
            sent -= 0.10

        # Weighted composite
        w = config.WEIGHTS
        score = tech * w["technical"] + vol * w["volume"] + sent * w["sentiment"]

        # Trend dampen
        if sma50.iloc[s] > sma200.iloc[s] and score < 0:
            score *= 0.5
        elif sma50.iloc[s] <= sma200.iloc[s] and score > 0:
            score *= 0.5

        return max(-1.0, min(1.0, score))

    except Exception:
        return 0.0


# ── Precomputed score caches (populated once before simulation) ───────────────
_ML_CACHE    = {}   # symbol -> bundle
_ML_SCORES   = {}   # (symbol, i) -> float
_RULE_SCORES_CACHE = {}  # (symbol, i) -> float


def _load_ml_models(watchlist):
    """Load best available model per symbol into _ML_CACHE."""
    import joblib
    from pathlib import Path
    print("  ML models loaded:", end=" ")
    loaded = []
    for sym in watchlist:
        # Priority for DAILY backtester: daily > v2 > 15min
        # 15min models trained on intraday patterns — wrong timeframe for daily backtest
        # The optimizer's 53% result used daily models
        candidates = [
            (f"ml_model_{sym}.joblib",        "daily"),
            (f"ml_model_v2_{sym}.joblib",    "v2"),
            (f"ml_model_15min_{sym}.joblib", "15min"),
        ]
        for path, label in candidates:
            if Path(path).exists():
                try:
                    bundle = joblib.load(path)
                    auc = bundle.get("auc", 0.5)
                    if auc >= 0.55:
                        _ML_CACHE[sym] = (bundle, label)
                        loaded.append(f"{sym}({label},auc={auc:.2f})")
                        break
                except:
                    pass
    print(", ".join(loaded))
    return loaded


def precompute_all_ml_scores(dfs):
    """Compute all ML scores once before simulation — same as optimizer."""
    print("  Pre-computing ML scores for all bars...")
    for sym, df in dfs.items():
        if sym not in _ML_CACHE:
            continue
        bundle, label = _ML_CACHE[sym]

        # Route to correct feature function based on model type
        if label == "15min":
            # 15min models: use daily features as proxy (no intraday data in backtest)
            try:
                from train_models_15min import compute_15min_features as feat_fn
            except:
                try:
                    from ml_model import compute_features as feat_fn
                except:
                    continue
        elif label == "v2":
            try:
                from train_models_v2 import compute_features_v2 as feat_fn
            except:
                try:
                    from ml_model import compute_features as feat_fn
                except:
                    continue
        else:
            try:
                from ml_model import compute_features as feat_fn
            except:
                continue

        # For v2 ensemble models
        is_ensemble = isinstance(bundle.get("model"), list)

        model     = bundle["model"]
        scaler    = bundle["scaler"]
        feat_cols = bundle["features"]
        hits = 0

        for i in range(220, len(df)):
            try:
                sub_df   = df.iloc[max(0, i - 260):i].copy()
                features = feat_fn(sub_df)
                if features is None or len(features) == 0:
                    continue
                row = features.iloc[-1]
                missing = [c for c in feat_cols if c not in row.index]
                if missing:
                    # Fill missing with 0 rather than skip
                    for c in missing:
                        row[c] = 0.0
                vals = row[feat_cols].values.reshape(1, -1)
                if np.any(np.isnan(vals)):
                    continue
                vals_s = scaler.transform(vals)

                if is_ensemble:
                    # Vote across ensemble
                    probs = [m.predict_proba(vals_s)[0][1] for m in model]
                    prob  = float(np.mean(probs))
                else:
                    prob = model.predict_proba(vals_s)[0][1]

                _ML_SCORES[(sym, i)] = (prob - 0.5) * 2.0
                hits += 1
            except:
                pass
        print(f"    {sym}: {hits} ML scores ({label})")
    print(f"  Done. {len(_ML_SCORES)} total scores cached.\n")


def compute_ml_score(symbol: str, df: pd.DataFrame, i: int) -> Optional[float]:
    """Look up precomputed ML score — instant."""
    return _ML_SCORES.get((symbol, i), None)


def compute_spy_adjustment(spy_df: pd.DataFrame, i: int) -> tuple:
    """
    Returns (threshold_boost, halt).
    threshold_boost: extra score needed due to weak market
    halt: True if market down >2%, no entries
    """
    try:
        if spy_df is None or i <= 0:
            return 0.0, False
        prev  = spy_df["close"].iloc[i - 1]
        curr  = spy_df["close"].iloc[i]
        chg   = (curr - prev) / prev
        if chg <= -0.02:
            return 0.0, True    # Halt
        elif chg <= -0.01:
            return 0.10, False  # Raise threshold by 0.10
        return 0.0, False
    except Exception:
        return 0.0, False


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: int
    entry_idx: int
    highest: float
    stop_loss: float
    score: float
    regime: str
    entry_date: str = ""


@dataclass
class Trade:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_date: str
    exit_date: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_days: int
    regime: str


@dataclass
class BacktestResult:
    symbol: str
    total_return_pct: float
    buy_hold_return_pct: float
    alpha: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_hold_days: float
    total_costs: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


class BacktesterV40:
    """Full strategy backtester matching live bot v4.0."""

    def run_portfolio(
        self,
        symbols: List[str],
        spy_df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        use_ml: bool = True,
        days: int = 365,
    ) -> dict:
        """
        Run portfolio-level backtest — all symbols sharing one pool of capital.
        This is more realistic than running each symbol in isolation.
        """
        # Fetch all data
        dfs = {}
        for sym in symbols:
            print(f"  Fetching {sym}...", end=" ", flush=True)
            df = fetch_data(sym, days)
            if df is not None and len(df) > 260:
                dfs[sym] = df
                print(f"{len(df)} bars")
            else:
                print("FAILED")

        if not dfs:
            return {}

        # Align all dataframes to common date index
        all_dates = sorted(set.intersection(*[set(df.index) for df in dfs.values()]))
        if len(all_dates) < 100:
            print("  Not enough overlapping dates")
            return {}

        # Pre-compute SMA50/200 for each symbol
        sma50s  = {}
        sma200s = {}
        for sym, df in dfs.items():
            sma50s[sym]  = df["close"].rolling(50).mean()
            sma200s[sym] = df["close"].rolling(200).mean()

        # ── PRECOMPUTE ALL ML SCORES (like optimizer — fast + accurate) ──
        _load_ml_models(list(dfs.keys()))
        precompute_all_ml_scores(dfs)
        if _ENHANCER is not None:
            print("  Signal enhancer active (Granger + vol regime + Hurst)")

        # Portfolio state
        cash        = initial_capital
        positions: Dict[str, Position] = {}
        all_trades: List[Trade] = []
        equity_curve = []
        total_costs  = 0.0
        lookback     = 220

        date_list = all_dates[lookback:]

        for date in date_list:
            # Get bar index for each symbol
            bar_idxs = {}
            for sym, df in dfs.items():
                if date in df.index:
                    bar_idxs[sym] = df.index.get_loc(date)

            if not bar_idxs:
                continue

            # Current prices
            prices = {}
            for sym, idx in bar_idxs.items():
                prices[sym] = dfs[sym]["close"].iloc[idx]

            # ── STOP LOSS CHECK — must run BEFORE EOD close ──
            # Use intraday low to check if 2% stop was hit during the day
            for sym in list(positions.keys()):
                if sym not in bar_idxs:
                    continue
                pos     = positions[sym]
                idx     = bar_idxs[sym]
                day_low = dfs[sym]["low"].iloc[idx]
                stop    = pos.entry_price * (1 - STOP_LOSS_PCT)

                if day_low <= stop:
                    fill_price, commission = apply_fill_cost(stop, pos.qty, "sell")
                    total_costs += commission
                    pnl     = (fill_price - pos.entry_price) * pos.qty - commission
                    pnl_pct = (fill_price - pos.entry_price) / pos.entry_price * 100
                    cash   += fill_price * pos.qty - commission
                    hold_d  = idx - pos.entry_idx
                    all_trades.append(Trade(
                        symbol=sym,
                        entry_price=pos.entry_price,
                        exit_price=fill_price,
                        quantity=pos.qty,
                        entry_date=pos.entry_date,
                        exit_date=str(date)[:10],
                        pnl=pnl, pnl_pct=pnl_pct,
                        exit_reason="Stop loss (2%)",
                        hold_days=max(1, hold_d),
                        regime=pos.regime,
                    ))
                    del positions[sym]

            # ── MAX HOLD EXIT — exit after MAX_HOLD_DAYS (optimizer: 5 days) ──
            for sym in list(positions.keys()):
                pos    = positions[sym]
                idx    = bar_idxs.get(sym, pos.entry_idx)
                hold_d = idx - pos.entry_idx
                if hold_d < MAX_HOLD_DAYS:
                    continue  # Still within hold window — keep position
                price  = prices.get(sym, pos.entry_price)
                fill_price, commission = apply_fill_cost(price, pos.qty, "sell")
                total_costs += commission
                pnl     = (fill_price - pos.entry_price) * pos.qty - commission
                pnl_pct = (fill_price - pos.entry_price) / pos.entry_price * 100
                cash   += fill_price * pos.qty - commission
                all_trades.append(Trade(
                    symbol=sym,
                    entry_price=pos.entry_price,
                    exit_price=fill_price,
                    quantity=pos.qty,
                    entry_date=pos.entry_date,
                    exit_date=str(date)[:10],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=f"Max hold ({MAX_HOLD_DAYS}d)",
                    hold_days=hold_d,
                    regime=pos.regime,
                ))
                del positions[sym]

            # ── COMPUTE SCORES for all symbols ──
            scores = {}
            for sym, idx in bar_idxs.items():
                if idx < lookback:
                    continue
                rule = compute_rule_score(dfs[sym], idx, sma50s[sym], sma200s[sym])
                if use_ml:
                    ml = compute_ml_score(sym, dfs[sym], idx)
                    score = ml * 0.60 + rule * 0.40 if ml is not None else rule
                else:
                    score = rule

                # ── SIGNAL ENHANCER — Granger + Vol Regime + Hurst ──
                if _ENHANCER is not None:
                    score, _edbg = _ENHANCER.get_enhanced_score(
                        sym, score, dfs[sym], idx, peer_dfs=dfs
                    )

                scores[sym] = score

            # ── SPY REGIME GATE ──
            spy_idx   = spy_df.index.get_loc(date) if date in spy_df.index else None
            threshold_boost, spy_halt = compute_spy_adjustment(spy_df, spy_idx) if spy_idx else (0.0, False)
            effective_threshold = BUY_THRESHOLD + threshold_boost

            # ── ROTATION: swap weakest position for better signal ──
            if not spy_halt and len(positions) >= MAX_POSITIONS and scores:
                weakest_sym   = min(positions.keys(), key=lambda s: positions[s].score)
                weakest_pos   = positions[weakest_sym]
                weakest_score = weakest_pos.score
                weakest_price = prices.get(weakest_sym, weakest_pos.entry_price)
                weakest_pnl   = (weakest_price - weakest_pos.entry_price) / weakest_pos.entry_price

                # Best signal not already held
                candidates = [(s, sc) for s, sc in scores.items()
                              if s not in positions and sc >= effective_threshold]
                candidates.sort(key=lambda x: x[1], reverse=True)

                if candidates:
                    best_sym, best_score = candidates[0]
                    gap     = best_score - weakest_score
                    is_flat = abs(weakest_pnl) <= 0.005

                    if gap >= ROTATION_GAP and is_flat:
                        # Close weakest
                        fill_price, commission = apply_fill_cost(weakest_price, weakest_pos.qty, "sell")
                        total_costs += commission
                        pnl     = (fill_price - weakest_pos.entry_price) * weakest_pos.qty - commission
                        pnl_pct = pnl / (weakest_pos.entry_price * weakest_pos.qty) * 100
                        cash   += fill_price * weakest_pos.qty - commission
                        all_trades.append(Trade(
                            symbol=weakest_sym,
                            entry_price=weakest_pos.entry_price,
                            exit_price=fill_price,
                            quantity=weakest_pos.qty,
                            entry_date=weakest_pos.entry_date,
                            exit_date=str(date)[:10],
                            pnl=pnl, pnl_pct=pnl_pct,
                            exit_reason=f"Rotation out (gap:{gap:.2f})",
                            hold_days=1,
                            regime=weakest_pos.regime,
                        ))
                        del positions[weakest_sym]

            # ── NEW ENTRIES ──
            if not spy_halt:
                # Rank by score
                entry_candidates = [
                    (sym, sc) for sym, sc in scores.items()
                    if sym not in positions and sc >= effective_threshold
                ]
                entry_candidates.sort(key=lambda x: x[1], reverse=True)

                for sym, score in entry_candidates:
                    if len(positions) >= MAX_POSITIONS:
                        break

                    # Exposure check
                    pos_value = sum(prices.get(s, p.entry_price) * p.qty for s, p in positions.items())
                    equity    = cash + pos_value
                    if equity <= 0 or (pos_value / equity) >= MAX_EXPOSURE:
                        break

                    idx        = bar_idxs[sym]
                    # Fill at NEXT bar's open — signal at close, execute next morning
                    next_idx   = idx + 1
                    if next_idx >= len(dfs[sym]):
                        continue
                    open_price = dfs[sym]["open"].iloc[next_idx]
                    fill_price, commission = apply_fill_cost(open_price, 1, "buy")

                    # Conviction sizing — matches optimizer best result (20% base)
                    if score >= 0.45:   size_pct = min(BASE_SIZE_PCT * 1.8, 0.36)
                    elif score >= 0.35: size_pct = min(BASE_SIZE_PCT * 1.3, 0.28)
                    else:               size_pct = BASE_SIZE_PCT
                    max_pos_val = equity * size_pct
                    qty = int(max_pos_val / fill_price) if fill_price > 0 else 0

                    if qty <= 0 or qty * fill_price + commission > cash:
                        continue

                    total_costs += commission
                    cash -= qty * fill_price + commission

                    trend_bull = sma50s[sym].iloc[idx] > sma200s[sym].iloc[idx]
                    positions[sym] = Position(
                        symbol=sym,
                        entry_price=fill_price,
                        qty=qty,
                        entry_idx=idx,
                        highest=fill_price,
                        stop_loss=fill_price * (1 - STOP_LOSS_PCT),
                        score=score,
                        regime="BULL" if trend_bull else "BEAR",
                    )
                    positions[sym].entry_date = str(date)[:10]

            # ── EQUITY CURVE ──
            pos_value  = sum(prices.get(s, p.entry_price) * p.qty for s, p in positions.items())
            equity_curve.append(cash + pos_value)

        # Close any remaining positions at last price
        for sym, pos in positions.items():
            price = prices.get(sym, pos.entry_price)
            fill_price, commission = apply_fill_cost(price, pos.qty, "sell")
            pnl     = (fill_price - pos.entry_price) * pos.qty - commission
            pnl_pct = pnl / (pos.entry_price * pos.qty) * 100
            cash   += fill_price * pos.qty - commission
            all_trades.append(Trade(
                symbol=sym,
                entry_price=pos.entry_price,
                exit_price=fill_price,
                quantity=pos.qty,
                entry_date=pos.entry_date,
                exit_date="final",
                pnl=pnl, pnl_pct=pnl_pct,
                exit_reason="End of backtest",
                hold_days=1,
                regime=pos.regime,
            ))

        return {
            "trades": all_trades,
            "equity_curve": equity_curve,
            "final_equity": cash + sum(prices.get(s, p.entry_price) * p.qty for s, p in positions.items()),
            "total_costs": total_costs,
            "initial_capital": initial_capital,
        }


def print_results(result: dict, days: int, use_ml: bool):
    trades        = result["trades"]
    equity_curve  = result["equity_curve"]
    final_equity  = result["final_equity"]
    initial       = result["initial_capital"]
    total_costs   = result["total_costs"]

    total_return = (final_equity / initial - 1) * 100
    winners      = [t for t in trades if t.pnl > 0]
    losers       = [t for t in trades if t.pnl <= 0]
    win_rate     = len(winners) / len(trades) * 100 if trades else 0
    avg_win      = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss     = np.mean([t.pnl_pct for t in losers])  if losers  else 0
    gross_profit = sum(t.pnl for t in winners)
    gross_loss   = abs(sum(t.pnl for t in losers))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_hold     = np.mean([t.hold_days for t in trades]) if trades else 0

    # Sharpe
    if len(equity_curve) > 1:
        eq      = np.array(equity_curve)
        daily_r = np.diff(eq) / eq[:-1]
        sharpe  = (daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    if equity_curve:
        eq       = np.array(equity_curve)
        peak     = np.maximum.accumulate(eq)
        dd       = (eq - peak) / peak * 100
        max_dd   = abs(dd.min())
    else:
        max_dd = 0

    ml_tag = "ML+Rules" if use_ml else "Rules Only"
    c_g    = "\033[92m"
    c_r    = "\033[91m"
    rst    = "\033[0m"
    color  = c_g if total_return > 0 else c_r

    print(f"\n{'═'*65}")
    print(f"  BACKTEST v4.0 — {ml_tag} — {days} Days")
    print(f"{'═'*65}")
    print(f"  Initial Capital:  ${initial:>12,.2f}")
    print(f"  Final Equity:     ${final_equity:>12,.2f}")
    print(f"  Total Return:     {color}{total_return:>+11.2f}%{rst}")
    print(f"  Annualized:       {color}{total_return / days * 252:>+11.2f}%{rst}")
    print(f"  Sharpe Ratio:     {sharpe:>12.2f}")
    print(f"  Max Drawdown:     {c_r}{max_dd:>11.1f}%{rst}")
    print(f"{'─'*65}")
    print(f"  Total Trades:     {len(trades):>12}")
    print(f"  Win Rate:         {win_rate:>11.1f}%")
    print(f"  Avg Win:          {c_g}{avg_win:>+11.2f}%{rst}")
    print(f"  Avg Loss:         {c_r}{avg_loss:>+11.2f}%{rst}")
    print(f"  Profit Factor:    {pf:>12.2f}  (>1.5 = good, >2.0 = excellent)")
    print(f"  Avg Hold:         {avg_hold:>11.1f} days")
    print(f"  Total Costs:      ${total_costs:>11,.2f}  (slippage + commission)")
    print(f"{'─'*65}")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"  Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"    {reason:<35} {count:>4}x ({pct:.0f}%)")

    # Top 5 / worst 5
    if trades:
        print(f"\n  TOP 5 TRADES:")
        for t in sorted(trades, key=lambda x: x.pnl, reverse=True)[:5]:
            print(f"    {t.symbol:<6} {t.entry_date} → {t.exit_date} | "
                  f"{c_g}${t.pnl:>+8,.0f}{rst} ({t.pnl_pct:>+5.1f}%) | {t.exit_reason}")

        print(f"\n  WORST 5 TRADES:")
        for t in sorted(trades, key=lambda x: x.pnl)[:5]:
            print(f"    {t.symbol:<6} {t.entry_date} → {t.exit_date} | "
                  f"{c_r}${t.pnl:>+8,.0f}{rst} ({t.pnl_pct:>+5.1f}%) | {t.exit_reason}")

    print(f"\n{'═'*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtester v4.0")
    parser.add_argument("--symbol", type=str,   default=None)
    parser.add_argument("--days",   type=int,   default=365)
    parser.add_argument("--no-ml",  action="store_true", help="Disable ML scoring (rule-based only)")
    args = parser.parse_args()

    use_ml  = not args.no_ml
    symbols = [args.symbol.upper()] if args.symbol else [
        s for s in config.WATCHLIST if s not in getattr(config, "INVERSE_ETFS", [])
    ]

    print(f"\n{'═'*65}")
    print(f"  BACKTESTER v4.0 — {'ML + Rules' if use_ml else 'Rules Only'}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {args.days} days")
    print(f"  Stop: {STOP_LOSS_PCT*100:.0f}% | Max positions: {MAX_POSITIONS} | Threshold: {BUY_THRESHOLD}")
    print(f"{'═'*65}\n")

    # Fetch SPY for regime gate
    print("  Fetching SPY for regime gate...")
    spy_df = fetch_data("SPY", args.days)

    bt     = BacktesterV40()
    result = bt.run_portfolio(symbols, spy_df, use_ml=use_ml, days=args.days)

    if not result or not result.get("trades"):
        print("  No results — check data availability")
        return

    print_results(result, args.days, use_ml)

    # Optionally compare ML vs no-ML
    if use_ml:
        print("  Running rules-only for comparison...")
        result_rules = bt.run_portfolio(symbols, spy_df, use_ml=False, days=args.days)
        if result_rules and result_rules.get("trades"):
            ml_return    = (result["final_equity"] / result["initial_capital"] - 1) * 100
            rules_return = (result_rules["final_equity"] / result_rules["initial_capital"] - 1) * 100
            print(f"\n  ML vs Rules comparison:")
            print(f"    ML + Rules:  {ml_return:>+.2f}%")
            print(f"    Rules Only:  {rules_return:>+.2f}%")
            print(f"    ML Edge:     {ml_return - rules_return:>+.2f}%\n")


if __name__ == "__main__":
    main()