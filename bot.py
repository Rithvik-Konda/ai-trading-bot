"""
AI Trading Bot — Main Engine (v3.3 Aggressive + Pyramiding)
=============================================================
Changes in v3.3:
- Concentrated positions: max 5 at a time, pick top signals
- Pyramiding: add to winners that keep scoring high
- Adaptive bull/bear mode: wide stops in bull, tight in bear
- No fixed TP cap: trailing stop lets winners ride
- 35% position size in bull, 95% total exposure
- FinBERT sentiment + LLM gate (from v3.2)
- Cooldown system + anti-churn (from v3.1)

Usage:
    python bot.py                  # Run the bot
    python bot.py --scan-only      # Scan without trading
"""

import sys
import time
import signal as sig_module
import logging
import argparse
from datetime import datetime
from typing import Dict, List
import config

# Data enrichment — analyst ratings, short interest, news, options, insiders
try:
    from data_enrichment import DataEnrichment
    _ENRICHER = DataEnrichment()
    _ENRICHMENT_AVAILABLE = True
except Exception as _enrich_err:
    _ENRICHER = None
    _ENRICHMENT_AVAILABLE = False

# Full data pipeline — FRED macro, Reddit, Google Trends, Alpaca news,
# earnings surprise, FINRA short, Wikipedia, Polygon options
try:
    from data_sources import DataPipeline
    _PIPELINE = DataPipeline()
    _PIPELINE_AVAILABLE = True
except Exception as _pipe_err:
    _PIPELINE = None
    _PIPELINE_AVAILABLE = False

# Signal enhancer — Granger causality, vol regime, Hurst exponent
try:
    from signal_enhancer import SignalEnhancer
    _ENHANCER = SignalEnhancer()
    _ENHANCER_AVAILABLE = True
except Exception as _enh_err:
    _ENHANCER = None
    _ENHANCER_AVAILABLE = False

# ── ML Signal Integration ──────────────────────────────────────────────────
# Load pre-trained models at startup. Falls back to rule-based scoring if
# no model file exists for a symbol.
_ML_MODELS = {}   # symbol -> {model, scaler, features}
_ML_ENABLED = True

def _load_ml_models():
    """Load all saved per-symbol ML models from disk."""
    global _ML_MODELS
    try:
        import joblib, os, glob
        MIN_ML_ACCURACY = 0.53
        # Priority: v2 ensemble > 15min intraday > daily
        daily_files   = {f.replace("ml_model_","").replace(".joblib","").upper(): f
                         for f in glob.glob("ml_model_*.joblib")
                         if "15min" not in f and "v2" not in f}
        intraday_files = {f.replace("ml_model_15min_","").replace(".joblib","").upper(): f
                          for f in glob.glob("ml_model_15min_*.joblib")}
        v2_files       = {f.replace("ml_model_v2_","").replace(".joblib","").upper(): f
                          for f in glob.glob("ml_model_v2_*.joblib")}
        # Merge in priority order — v2 wins over 15min wins over daily
        merged     = {**daily_files, **intraday_files, **v2_files}
        model_files = list(merged.values())
        for path in model_files:
            sym = (path.replace("ml_model_v2_","")
                      .replace("ml_model_15min_","")
                      .replace("ml_model_","")
                      .replace(".joblib","").upper())
            try:
                bundle = joblib.load(path)
                acc = bundle.get("accuracy", 0)
                prec = bundle.get("precision", 0)
                auc = bundle.get("auc", 0.5)
                if auc >= 0.55 and prec >= 0.45:
                    _ML_MODELS[sym] = bundle
                    logger.info(f"ML model loaded: {sym} (auc={auc:.2f} acc={acc:.1%} prec={prec:.1%})")
                else:
                    logger.info(f"ML model REJECTED: {sym} (auc={auc:.2f} acc={acc:.1%} prec={prec:.1%} — below threshold, using rule-based)")
            except Exception as e:
                logger.warning(f"ML model load failed for {sym}: {e}")
        if _ML_MODELS:
            logger.info(f"ML models active: {list(_ML_MODELS.keys())}")
        else:
            logger.info("No qualifying ML models — using rule-based scoring only")
    except ImportError:
        logger.warning("joblib not installed — ML models disabled")

def _ml_score(symbol: str, df) -> float:
    """
    Get ML probability score for a symbol given recent OHLCV bars.
    Returns float in [-1, 1] (mapped from [0,1] probability).
    Returns None if no model available.
    """
    global _ML_MODELS
    if not _ML_ENABLED or symbol not in _ML_MODELS:
        return None
    try:
        import pandas as pd
        bundle       = _ML_MODELS[symbol]
        model        = bundle["model"]
        scaler       = bundle["scaler"]
        feature_cols = bundle["features"]
        version   = bundle.get("version", "v1")
        timeframe = bundle.get("timeframe", "daily")
        is_ensemble = bundle.get("ensemble", False)

        if version == "v2":
            # v2 ensemble — uses enhanced features with SPY RS + sector momentum
            from train_models_v2 import compute_features_v2
            daily = df.resample("1D").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            if len(daily) < 60:
                return None
            features = compute_features_v2(daily)
        elif timeframe == "15min":
            from train_models_15min import compute_15min_features
            if len(df) < 50:
                return None
            features = compute_15min_features(df)
        else:
            from ml_model import compute_features
            daily = df.resample("1D").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            if len(daily) < 220:
                return None
            features = compute_features(daily)
        row = features.iloc[-1][feature_cols].values.reshape(1, -1)

        import numpy as np
        if np.any(np.isnan(row)):
            return None

        row_s = scaler.transform(row)
        if bundle.get("ensemble", False):
            from train_models_v2 import ensemble_predict_proba
            prob = ensemble_predict_proba(bundle["models"], row_s)[0]
        else:
            prob = model.predict_proba(row_s)[0][1]

        # Map [0,1] probability to [-1,1] score
        return (prob - 0.5) * 2.0
    except Exception as e:
        logger.debug(f"ML score failed for {symbol}: {e}")
        return None
from technical_analysis import TechnicalAnalyzer, TechnicalSignals
from volume_analysis import VolumeAnalyzer, VolumeSignals
from news_sentiment import NewsSentimentAnalyzer, SentimentSignals
from risk_manager import RiskManager, RiskCheck
from broker import AlpacaBroker
from llm_gate import LLMTradeGate
from earnings_guard import EarningsGuard

# --- Logging Setup ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


# --- Signal Composite ---

class TradeSignal:
    """Composite signal combining all analysis modules."""

    def __init__(
        self, symbol: str,
        technical: TechnicalSignals,
        volume: VolumeSignals,
        sentiment: SentimentSignals,
        trend_bullish: bool = True,
    ):
        self.symbol = symbol
        self.technical = technical
        self.volume = volume
        self.sentiment = sentiment
        self.trend_bullish = trend_bullish
        self._df = None  # Set externally before composite computed
        self.composite_score = self._compute_composite()
        self.action = self._determine_action()
        self.confidence = abs(self.composite_score)

    def _compute_composite(self) -> float:
        w = config.WEIGHTS
        rule_score = (
            self.technical.score * w["technical"]
            + self.volume.score * w["volume"]
            + self.sentiment.score * w["sentiment"]
        )
        # Trend filter: dampen signals that go against the daily trend
        if config.TREND_FILTER_ENABLED:
            if self.trend_bullish and rule_score < 0:
                rule_score *= 0.5
            elif not self.trend_bullish and rule_score > 0:
                rule_score *= 0.5

        # ML blend: if a trained model exists, weight it 60% vs 40% rules
        # This lets the model dominate while rules act as a sanity check
        ml = _ml_score(self.symbol, self._df) if hasattr(self, '_df') and self._df is not None else None
        if ml is not None:
            score = ml * 0.60 + rule_score * 0.40
            logger.debug(f"{self.symbol} | ML:{ml:+.3f} Rule:{rule_score:+.3f} Blend:{score:+.3f}")
        else:
            score = rule_score

        return max(-1.0, min(1.0, score))

    def _determine_action(self) -> str:
        # Adaptive thresholds based on market regime
        if self.trend_bullish:
            buy_thresh = getattr(config, 'BULL_BUY_THRESHOLD', config.BUY_THRESHOLD)
        else:
            buy_thresh = getattr(config, 'BEAR_BUY_THRESHOLD', config.BUY_THRESHOLD)

        if self.composite_score >= buy_thresh:
            return "BUY"
        return "HOLD"  # No selling/shorting in v3.3

    def summary(self) -> str:
        trend = "BULL" if self.trend_bullish else "BEAR"
        lines = [
            f"{'='*60}",
            f"  {self.symbol} | Action: {self.action} | Score: {self.composite_score:+.3f} | Conf: {self.confidence:.1%} | Trend: {trend}",
            f"{'-'*60}",
            f"  Technical : {self.technical.score:+.3f}  | {', '.join(self.technical.signals[:3])}",
            f"  Volume    : {self.volume.score:+.3f}  | {', '.join(self.volume.signals[:3])}",
            f"  Sentiment : {self.sentiment.score:+.3f}  | {', '.join(self.sentiment.signals[:2])}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# --- Main Bot ---

MAX_POSITIONS = 20  # Soft ceiling — real gate is 85% exposure check below

class TradingBot:
    """Main autopilot trading engine v3.3 — aggressive + pyramiding."""

    def __init__(self, paper_mode: bool = True):
        logger.info("Initializing AI Trading Bot v3.3...")
        self.broker = AlpacaBroker()
        self.tech_analyzer = TechnicalAnalyzer()
        self.vol_analyzer = VolumeAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.risk_manager = RiskManager()
        self.llm_gate = LLMTradeGate()
        self.earnings_guard = EarningsGuard()
        # v3.5: Options earnings engine
        try:
            from options_earnings import OptionsEarningsEngine
            self.options_engine = OptionsEarningsEngine(broker=self.broker)
            logger.info("Options earnings engine loaded")
        except Exception as e:
            self.options_engine = None
            logger.warning(f"Options engine unavailable: {e}")
        self.running = False
        self.scan_only = False
        self.trade_log: List[dict] = []
        self._load_trade_log()
        # Load ML models at startup
        _load_ml_models()

    def _save_trade_log(self):
        """Save trade log to CSV for ML training data."""
        import csv
        filepath = "trade_history.csv"
        if not self.trade_log:
            return
        keys = self.trade_log[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            for row in self.trade_log:
                writer.writerow({k: row.get(k, "") for k in keys})

    def _load_trade_log(self):
        """Load existing trade history on startup."""
        import csv
        filepath = "trade_history.csv"
        try:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                self.trade_log = [row for row in reader]
            logger.info(f"Loaded {len(self.trade_log)} historical trades from {filepath}")
        except FileNotFoundError:
            logger.info("No trade history found — starting fresh")

        # Daily trend cache
        self._daily_trends: Dict[str, bool] = {}
        self._daily_trend_updated: float = 0

        # Market regime (based on SPY trend)
        self._market_bull: bool = True

        # Check if LLM gate is configured
        llm_enabled = getattr(config, 'LLM_GATE_ENABLED', True)
        self.llm_gate.enabled = llm_enabled

        # Update portfolio value from broker
        if self.broker.connected:
            acct = self.broker.get_account()
            if "equity" in acct:
                self.risk_manager.update_portfolio_value(acct["equity"])
                logger.info(f"Portfolio value: ${acct['equity']:,.2f}")

            # Sync existing positions from Alpaca (prevents double-buying on restart)
            broker_positions = self.broker.get_positions()
            for bp in broker_positions:
                sym = bp["symbol"]
                if sym not in self.risk_manager.positions:
                    from risk_manager import Position
                    pos = Position(
                        symbol=sym,
                        side="long" if bp.get("side", "long") == "long" else "short",
                        entry_price=bp["avg_entry"],
                        quantity=bp["qty"],
                        entry_time=datetime.now(),
                        stop_loss=bp["avg_entry"] * (1 - config.BULL_STOP_LOSS_PCT),
                        take_profit=bp["avg_entry"] * (1 + config.BULL_TAKE_PROFIT_PCT),
                        highest_price=bp["current_price"],
                    )
                    self.risk_manager.positions[sym] = pos
                    logger.info(f"Synced position: {sym} {bp['qty']}x @ ${bp['avg_entry']:.2f} (PnL: ${bp['unrealized_pnl']:.2f})")
            if broker_positions:
                logger.info(f"Synced {len(broker_positions)} existing positions from Alpaca")

    # -- Daily Trend Filter --

    def update_daily_trends(self):
        """Fetch daily bars and compute trend direction for each symbol."""
        import time as _time
        if _time.time() - self._daily_trend_updated < 1800:
            return

        logger.info("Updating daily trend filter...")
        for symbol in config.WATCHLIST:
            try:
                df = self.broker.get_bars(symbol, timeframe="1Day", limit=250)
                if df is not None and len(df) >= config.TREND_SMA_SLOW:
                    sma_fast = df["close"].rolling(config.TREND_SMA_FAST).mean().iloc[-1]
                    sma_slow = df["close"].rolling(config.TREND_SMA_SLOW).mean().iloc[-1]
                    self._daily_trends[symbol] = sma_fast > sma_slow
                else:
                    self._daily_trends[symbol] = True
            except Exception as e:
                logger.debug(f"Trend fetch failed for {symbol}: {e}")
                self._daily_trends[symbol] = True

        self._daily_trend_updated = _time.time()
        bulls = sum(1 for v in self._daily_trends.values() if v)
        bears = len(self._daily_trends) - bulls
        logger.info(f"Daily trends: {bulls} bullish, {bears} bearish")

        # Market regime from SPY
        self._market_bull = self._daily_trends.get("SPY", True)
        regime = "BULL" if self._market_bull else "BEAR"
        logger.info(f"Market regime: {regime}")

    # -- Scanning --

    def scan_symbol(self, symbol: str) -> TradeSignal:
        """Run full analysis on a single symbol using intraday bars."""
        df = self.broker.get_bars(
            symbol,
            timeframe=config.INTRADAY_TIMEFRAME,
            limit=config.INTRADAY_BAR_LIMIT,
        )

        if df is not None and len(df) > 0:
            tech_signals = self.tech_analyzer.analyze(symbol, df)
            vol_signals = self.vol_analyzer.analyze(symbol, df)
        else:
            tech_signals = TechnicalSignals(symbol=symbol)
            vol_signals = VolumeSignals(symbol=symbol)
            logger.warning(f"No market data for {symbol}")

        sent_signals = self.news_analyzer.analyze(symbol)
        trend_bull = self._daily_trends.get(symbol, True)

        signal = TradeSignal(symbol, tech_signals, vol_signals, sent_signals, trend_bull)
        signal._df = df  # Pass bars so ML model can compute features
        signal.composite_score = signal._compute_composite()  # Recompute with ML
        signal.action = signal._determine_action()
        return signal

    def scan_watchlist(self) -> List[TradeSignal]:
        """Scan all symbols in the watchlist."""
        signals = []
        for symbol in config.WATCHLIST:
            try:
                sig = self.scan_symbol(symbol)
                signals.append(sig)
                logger.info(sig.summary())
                time.sleep(0.5)  # Rate limit padding — avoid Alpaca 429s
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        return signals

    # -- Adaptive Parameters --

    def get_regime_params(self, symbol: str = None):
        """Return position sizing and stop params based on market regime.
        Optionally pass symbol to apply per-symbol stop loss override (v3.4).
        """
        if self._market_bull:
            base_sl = config.BULL_STOP_LOSS_PCT
            params = {
                "sl_pct": base_sl,
                "tp_pct": config.BULL_TAKE_PROFIT_PCT,
                "max_pos_pct": config.BULL_MAX_POSITION_SIZE_PCT,
            }
        else:
            base_sl = config.BEAR_STOP_LOSS_PCT
            params = {
                "sl_pct": base_sl,
                "tp_pct": config.BEAR_TAKE_PROFIT_PCT,
                "max_pos_pct": config.BEAR_MAX_POSITION_SIZE_PCT,
            }

        # Per-symbol stop override — wider stops for slow cyclicals
        if symbol:
            sym_sl = getattr(config, 'SYMBOL_STOP_LOSS_PCT', {}).get(symbol)
            if sym_sl is not None:
                params["sl_pct"] = sym_sl

        return params

    # -- Pyramiding --

    def try_pyramid(self, symbol: str, signal: TradeSignal, price: float):
        """
        Scale into a position that is moving in our favor.

        Three pyramid tiers — each fires once per position:
          Tier 1: signal score > 0.25 AND price moving right direction → add 50% of initial
          Tier 2: signal score > 0.35 AND price up 1%+ from entry → add 35% of initial
          Tier 3: signal score > 0.45 AND price up 2%+ from entry → add 25% of initial

        Each tier only fires once (tracked via pos._pyramid_tiers set).
        Total max position = initial + 50% + 35% + 25% = 2.1x initial size.
        Stop loss trails up as we add — we never risk more than 2% on the whole position.
        """
        pos = self.risk_manager.positions.get(symbol)
        if not pos or pos.side != "long":
            return

        # Initialize tier tracking on position
        if not hasattr(pos, "_pyramid_tiers"):
            pos._pyramid_tiers = set()

        score = signal.composite_score
        pnl_pct = (price - pos.entry_price) / pos.entry_price

        # Check buying power first
        acct = self.broker.get_account()
        cash = acct.get("cash", 0)
        params = self.get_regime_params(symbol)
        portfolio = self.risk_manager.portfolio_value

        # Determine which tier we can fire
        tier = None
        add_pct = 0.0  # fraction of portfolio to add

        if 1 not in pos._pyramid_tiers and score >= 0.25 and pnl_pct >= -0.005:
            # Tier 1: signal strong, position not deep underwater
            tier = 1
            add_pct = 0.06  # 6% of portfolio
        elif 2 not in pos._pyramid_tiers and score >= 0.35 and pnl_pct >= 0.01:
            # Tier 2: signal very strong, up 1%+
            tier = 2
            add_pct = 0.05  # 5% of portfolio
        elif 3 not in pos._pyramid_tiers and score >= 0.45 and pnl_pct >= 0.02:
            # Tier 3: high conviction, up 2%+
            tier = 3
            add_pct = 0.04  # 4% of portfolio

        if tier is None:
            return

        add_value = portfolio * add_pct
        add_value = min(add_value, cash * 0.85)
        add_qty = int(add_value / price)

        if add_qty <= 0:
            return

        result = self.broker.smart_order(symbol, add_qty, "buy")
        if "error" not in result:
            actual_price = self.broker.get_latest_price(symbol) or price
            total_qty = pos.quantity + add_qty
            avg_entry = (pos.entry_price * pos.quantity + actual_price * add_qty) / total_qty
            pos.quantity = total_qty
            pos.entry_price = avg_entry
            # Trail stop up — never risk more than 2% on full position
            pos.stop_loss = avg_entry * (1 - params["sl_pct"])
            pos.take_profit = avg_entry * (1 + params["tp_pct"])
            pos._pyramid_tiers.add(tier)
            logger.info(
                f"PYRAMID T{tier}: +{add_qty}x {symbol} @ ${actual_price:.2f} "
                f"| Total: {total_qty}x avg ${avg_entry:.2f} "
                f"| Score:{score:.2f} PnL:{pnl_pct:+.1%} "
                f"| Tiers fired: {sorted(pos._pyramid_tiers)}"
            )

    # -- Trade Execution --

    def execute_signal(self, signal: TradeSignal):
        """Execute a trade based on a signal, with adaptive sizing and LLM gate."""
        if signal.action == "HOLD":
            return

        symbol = signal.symbol

        # Block short-selling inverse ETFs
        if signal.action == "SELL" and symbol in getattr(config, 'INVERSE_ETFS', []):
            logger.info(f"Skipped: can't short inverse ETF {symbol}")
            return

        price = self.broker.get_latest_price(symbol)
        if price is None or price <= 0:
            price = signal.technical.current_price
            if price <= 0:
                return

        # Check if we should pyramid an existing position
        if symbol in self.risk_manager.positions:
            self.try_pyramid(symbol, signal, price)
            return  # Don't open duplicate — pyramid handles sizing

        # Earnings guard: check risk once, apply correctly
        earnings_risk = self.earnings_guard.check_earnings_risk(symbol)
        if earnings_risk.action == "BLOCK_ENTRY":
            logger.info(f"Earnings guard BLOCKED: {symbol} — {earnings_risk.reason}")
            return

        # Two gates: max concurrent positions AND max exposure
        _max_pos = getattr(config, 'MAX_CONCURRENT_POSITIONS', 3)
        if len(self.risk_manager.positions) >= _max_pos:
            logger.info(f"Max positions ({_max_pos}) reached — skipping {symbol}")
            return
        _acct_exp = self.broker.get_account()
        _equity_exp = _acct_exp.get("equity", 1)
        _cash_exp = _acct_exp.get("cash", 0)
        _exposure_pct = ((_equity_exp - _cash_exp) / _equity_exp * 100) if _equity_exp > 0 else 0
        _max_exp = getattr(config, 'MAX_TOTAL_EXPOSURE_PCT', 0.85) * 100
        if _exposure_pct >= _max_exp:
            logger.info(f"Exposure {_exposure_pct:.1f}% >= {_max_exp:.0f}% — skipping {symbol}")
            return

        # --- CONVICTION-BASED POSITION SIZING ---
        # Size scales with signal strength, volatility, regime, and available cash.
        params = self.get_regime_params(symbol)
        portfolio = self.risk_manager.portfolio_value
        score = abs(signal.composite_score)

        # Conviction tiers — optimizer proved 20% base with conviction scaling
        # Base 20%, scales up to 36% on highest conviction signals
        _base = getattr(config, 'POSITION_SIZE_BASE', 0.20)
        if score >= 0.45:
            conviction_pct = min(_base * 1.8, 0.36)  # 36% max
        elif score >= 0.35:
            conviction_pct = min(_base * 1.3, 0.28)  # 28% max
        elif score >= 0.25:
            conviction_pct = _base                    # 20% base
        elif score >= 0.15:
            conviction_pct = _base * 0.75             # 15%
        else:
            conviction_pct = _base * 0.5              # 10%

        # Volatility scalar — high beta stocks get smaller size
        vol_scalar = 1.0
        try:
            _df = self.broker.get_bars(symbol, timeframe="15Min", limit=50)
            if _df is not None and len(_df) > 14:
                import numpy as _np
                _atr = _df["close"].diff().abs().rolling(14).mean().iloc[-1]
                _atr_pct = _atr / _df["close"].iloc[-1]
                if _atr_pct > 0.02:
                    vol_scalar = 0.60
                elif _atr_pct > 0.015:
                    vol_scalar = 0.75
                elif _atr_pct > 0.01:
                    vol_scalar = 0.90
        except Exception:
            pass

        # Regime scalar
        regime_scalar = 1.0 if self._market_bull else 0.6

        # Cash scalar — don't overcommit when low on cash
        _acct2 = self.broker.get_account()
        _cash = _acct2.get("cash", portfolio)
        _cash_pct = _cash / portfolio if portfolio > 0 else 1.0
        if _cash_pct < 0.15:
            cash_scalar = 0.5
        elif _cash_pct < 0.30:
            cash_scalar = 0.75
        else:
            cash_scalar = 1.0

        target_pct = conviction_pct * vol_scalar * regime_scalar * cash_scalar
        target_value = portfolio * target_pct
        max_dollars = getattr(config, "MAX_POSITION_DOLLARS", 35000)
        target_value = min(target_value, max_dollars)
        target_value = min(target_value, _cash * 0.90)

        logger.info(
            f"Sizing {symbol}: score={score:.2f} → {conviction_pct:.0%} "
            f"× vol={vol_scalar:.2f} × regime={regime_scalar:.2f} × cash={cash_scalar:.2f} "
            f"= {target_pct:.1%} (${target_value:,.0f})"
        )

        # Apply earnings guard position scale
        if earnings_risk.position_scale < 1.0:
            target_value *= earnings_risk.position_scale
            logger.info(f"Earnings guard: {symbol} size scaled to {earnings_risk.position_scale:.0%} — {earnings_risk.reason}")

        qty = int(target_value / price) if price > 0 else 0

        if qty <= 0:
            return

        side = "buy" if signal.action == "BUY" else "sell"
        risk_side = "long" if side == "buy" else "short"

        # Risk check (includes cooldown)
        check = self.risk_manager.check_trade(symbol, risk_side, price, qty)
        if not check.approved:
            logger.info(f"Trade blocked: {symbol} {side} -- {check.reason}")
            return

        qty = check.adjusted_quantity

        if self.scan_only:
            logger.info(f"[SCAN ONLY] Would {side} {qty}x {symbol} @ ${price:.2f} (${qty*price:,.0f})")
            return

        # --- DATA ENRICHMENT — analyst/news/options/insider signals ---
        if _ENRICHMENT_AVAILABLE and _ENRICHER:
            try:
                blocked, block_reason = _ENRICHER.should_block_entry(symbol)
                if blocked:
                    logger.info(f"ENRICHMENT BLOCK: {symbol} — {block_reason}")
                    return
                enriched = _ENRICHER.enrich_score(symbol, signal.composite_score, weight=0.15)
                if abs(enriched - signal.composite_score) > 0.01:
                    logger.info(f"ENRICHMENT: {symbol} score {signal.composite_score:.3f} → {enriched:.3f} (Δ{enriched-signal.composite_score:+.3f})")
                signal.composite_score = enriched
            except Exception as _ee:
                logger.debug(f"Enrichment failed {symbol}: {_ee}")

        # --- FULL DATA PIPELINE — macro + Reddit + trends + earnings + FINRA ---
        if _PIPELINE_AVAILABLE and _PIPELINE:
            try:
                # Market-wide halt check
                halt, halt_reason = _PIPELINE.should_halt_all_entries()
                if halt:
                    logger.warning(f"PIPELINE HALT: {halt_reason} — blocking all entries")
                    return
                # Symbol-specific boost
                boost = _PIPELINE.get_entry_boost(symbol)
                if abs(boost) > 0.02:
                    old_score = signal.composite_score
                    signal.composite_score = float(np.clip(signal.composite_score + boost, -1, 1))
                    logger.info(
                        f"PIPELINE: {symbol} score {old_score:.3f} → {signal.composite_score:.3f} "
                        f"(boost={boost:+.3f})"
                    )
            except Exception as _pe:
                logger.debug(f"Pipeline failed {symbol}: {_pe}")

        # --- SIGNAL ENHANCER — Granger causality + vol regime + Hurst ---
        if _ENHANCER_AVAILABLE and _ENHANCER:
            try:
                import yfinance as yf
                # Fetch current df for symbol + peers
                _sym_df = yf.Ticker(symbol).history(period="1y")
                _sym_df.columns = [c.lower() for c in _sym_df.columns]
                from signal_enhancer import PEER_MAP
                _peers  = PEER_MAP.get(symbol, [])
                _peer_dfs = {}
                for _p in _peers[:2]:  # Limit to 2 peers for speed
                    try:
                        _pdf = yf.Ticker(_p).history(period="1y")
                        _pdf.columns = [c.lower() for c in _pdf.columns]
                        _peer_dfs[_p] = _pdf
                    except:
                        pass
                _i = len(_sym_df) - 1
                # Block check first
                _blocked, _breason = _ENHANCER.should_block(symbol, _sym_df, _i)
                if _blocked:
                    logger.info(f"ENHANCER BLOCK: {symbol} — {_breason}")
                    return
                # Apply enhancement
                _old = signal.composite_score
                signal.composite_score, _edbg = _ENHANCER.get_enhanced_score(
                    symbol, signal.composite_score, _sym_df, _i, _peer_dfs
                )
                if abs(signal.composite_score - _old) > 0.01:
                    logger.info(
                        f"ENHANCER: {symbol} {_old:.3f}→{signal.composite_score:.3f} "
                        f"regime={_edbg['regime_mode']} vol={_edbg['vol_regime']} "
                        f"granger={_edbg['granger_boost']:+.3f}"
                    )
            except Exception as _ee:
                logger.debug(f"Enhancer failed {symbol}: {_ee}")
        summary = self.risk_manager.summary()
        llm_decision = self.llm_gate.evaluate_trade(
            symbol=symbol,
            action=signal.action,
            price=price,
            qty=qty,
            composite_score=signal.composite_score,
            tech_score=signal.technical.score,
            tech_signals=signal.technical.signals,
            vol_score=signal.volume.score,
            vol_signals=signal.volume.signals,
            sent_score=signal.sentiment.score,
            sent_signals=signal.sentiment.signals,
            trend_bullish=signal.trend_bullish,
            portfolio_value=summary["portfolio_value"],
            open_positions=summary["open_positions"],
            exposure_pct=summary["exposure_pct"],
        )

        if not llm_decision.approved:
            logger.info(f"LLM REJECTED: {symbol} {side} -- {llm_decision.reasoning}")
            return

        # Apply LLM confidence adjustment
        if llm_decision.confidence_adjust != 0 and not llm_decision.passthrough:
            old_qty = qty
            adjusted_confidence = max(0.05, signal.confidence + llm_decision.confidence_adjust)
            target_value = max_value * adjusted_confidence
            qty = int(target_value / price) if price > 0 else 0
            qty = min(qty, check.adjusted_quantity)
            if qty != old_qty:
                logger.info(f"LLM adjusted qty: {old_qty} -> {qty} (conf {llm_decision.confidence_adjust:+.2f})")
            if qty <= 0:
                return

        # Submit order
        result = self.broker.smart_order(symbol, qty, side)
        if "error" not in result:
            actual_price = self.broker.get_latest_price(symbol) or price

            check.stop_loss = actual_price * (1 - params["sl_pct"])
            check.take_profit = actual_price * (1 + params["tp_pct"])

            self.risk_manager.open_position(symbol, risk_side, actual_price, qty, check)
            self.trade_log.append({
                "time": datetime.now().isoformat(),
                "symbol": symbol,
                "action": signal.action,
                "side": "entry",
                "qty": qty,
                "price": actual_price,
                "score": signal.composite_score,
                "value": qty * actual_price,
                "llm_reasoning": llm_decision.reasoning,
            })
            self._save_trade_log()
            logger.info(f"Executed: {side.upper()} {qty}x {symbol} @ ${actual_price:.2f} (${qty*actual_price:,.0f})")
        else:
            logger.error(f"Order failed: {result['error']}")

    # -- Exit Monitoring --

    def check_exits(self):
        """Monitor open positions for stop loss / take profit / max loss / max hold exits."""
        symbols = list(self.risk_manager.positions.keys())
        if not symbols:
            return

        # ── GAP-DOWN PROTECTION ──────────────────────────────────────────────
        # Since we now hold overnight (up to 5 days), check for overnight gaps.
        # If any position gaps down >3% at open vs prior close, cut it immediately
        # before the stop loss fires — prevents riding a bad gap all the way down.
        now_et = datetime.now(pytz.timezone("America/New_York"))
        is_market_open_hour = 9 <= now_et.hour <= 10
        if is_market_open_hour and now_et.minute <= 15:
            for sym in list(symbols):
                pos = self.risk_manager.positions.get(sym)
                if not pos:
                    continue
                price = self.broker.get_latest_price(sym)
                if not price or price <= 0:
                    continue
                gap_pct = (price - pos.entry_price) / pos.entry_price
                # Gap down 3%+ from entry — exit immediately, don't wait for stop
                if gap_pct <= -0.03:
                    side   = "sell" if pos.side == "long" else "buy"
                    result = self.broker.smart_order(sym, pos.quantity, side)
                    if "error" not in result:
                        pnl = (price - pos.entry_price) * pos.quantity
                        self.risk_manager.close_position(sym, price, "Gap-down protection")
                        logger.warning(
                            f"GAP-DOWN EXIT: {sym} {pos.quantity}x @ ${price:.2f} "
                            f"| Gap: {gap_pct:+.1%} | PnL: ${pnl:+,.0f}"
                        )
                        symbols = [s for s in symbols if s != sym]

        # ── MAX HOLD DAYS ────────────────────────────────────────────────────
        # Optimizer: hold up to 5 days, then exit regardless
        max_hold_days = getattr(config, 'SWING_MAX_HOLD_DAYS', 5)
        for sym in list(symbols):
            pos = self.risk_manager.positions.get(sym)
            if pos and hasattr(pos, 'entry_time'):
                hold_days = (datetime.now() - pos.entry_time).total_seconds() / 86400
                if hold_days >= max_hold_days:
                    price = self.broker.get_latest_price(sym)
                    if price and price > 0:
                        side = "sell" if pos.side == "long" else "buy"
                        result = self.broker.smart_order(sym, pos.quantity, side)
                        if "error" not in result:
                            pnl = (price - pos.entry_price) * pos.quantity if pos.side == "long" else (pos.entry_price - price) * pos.quantity
                            self.risk_manager.close_position(sym, price, f"Max hold ({hold_days:.1f}d)")
                            logger.info(f"MAX HOLD EXIT: {sym} {pos.quantity}x @ ${price:.2f} | PnL: ${pnl:+,.0f} | Held {hold_days:.1f}d")
                            symbols = [s for s in symbols if s != sym]

        prices = self.broker.get_latest_prices(symbols)
        exits = self.risk_manager.check_exits(prices)

        for symbol, exit_price, reason in exits:
            pos = self.risk_manager.positions.get(symbol)
            if pos:
                side = "sell" if pos.side == "long" else "buy"
                result = self.broker.smart_order(symbol, pos.quantity, side)
                if "error" not in result:
                    pnl = (exit_price - pos.entry_price) * pos.quantity if pos.side == "long" else (pos.entry_price - exit_price) * pos.quantity
                    self.trade_log.append({
                        "time": datetime.now().isoformat(),
                        "symbol": symbol,
                        "action": reason,
                        "side": "exit",
                        "qty": pos.quantity,
                        "price": exit_price,
                        "entry_price": pos.entry_price,
                        "pnl": pnl,
                        "pnl_pct": (exit_price - pos.entry_price) / pos.entry_price * 100,
                        "hold_time_min": (datetime.now() - pos.entry_time).total_seconds() / 60 if hasattr(pos, 'entry_time') else 0,
                    })
                    self._save_trade_log()
                    self.risk_manager.close_position(symbol, exit_price, reason)
                    logger.info(f"Exit: {symbol} {pos.quantity}x @ ${exit_price:.2f} | PnL: ${pnl:+,.0f} | {reason}")

    # -- Main Loop --

    def run(self, scan_only: bool = False):
        """Main autopilot loop."""
        self.scan_only = scan_only
        self.running = True

        def handle_signal(s, frame):
            logger.info("Shutdown signal received. Stopping...")
            self.running = False
        sig_module.signal(sig_module.SIGINT, handle_signal)
        sig_module.signal(sig_module.SIGTERM, handle_signal)

        mode = "SCAN ONLY" if scan_only else "LIVE PAPER TRADING"
        llm_status = "ON" if self.llm_gate.enabled else "OFF"
        regime = "BULL" if self._market_bull else "BEAR"
        logger.info(f"== AI Trading Bot v3.3 (Aggressive + Pyramid) ==")
        logger.info(f"Mode: {mode}")
        logger.info(f"Watchlist: {len(config.WATCHLIST)} symbols")
        logger.info(f"Position limit: exposure-based (85% cap, no count limit)")
        logger.info(f"Signal timeframe: {config.INTRADAY_TIMEFRAME} bars")
        logger.info(f"Trend filter: {config.TREND_SMA_FAST}/{config.TREND_SMA_SLOW} SMA on daily")
        logger.info(f"BULL mode: SL={config.BULL_STOP_LOSS_PCT*100:.0f}% TP={config.BULL_TAKE_PROFIT_PCT*100:.0f}% Pos={config.BULL_MAX_POSITION_SIZE_PCT*100:.0f}%")
        logger.info(f"BEAR mode: SL={config.BEAR_STOP_LOSS_PCT*100:.0f}% TP={config.BEAR_TAKE_PROFIT_PCT*100:.0f}% Pos={config.BEAR_MAX_POSITION_SIZE_PCT*100:.0f}%")
        logger.info(f"LLM Gate: {llm_status}")
        logger.info(f"Earnings Guard: ON (exit {self.earnings_guard.exit_days}d, reduce {self.earnings_guard.reduce_days}d, caution {self.earnings_guard.caution_days}d)")
        logger.info(f"Scan interval: {config.SCAN_INTERVAL_SECONDS}s")
        logger.info(f"Cooldown: 120min after stops, 60min after profits")

        cycle = 0
        while self.running:
            cycle += 1
            logger.info(f"\n--- Cycle {cycle} -- {datetime.now().strftime('%H:%M:%S')} ---")

            # Always check market status — skip new entries when closed,
            # but still monitor exits (gap-down protection at open).
            try:
                market_open = self.broker.is_market_open()
            except Exception as e:
                logger.error(f"Clock check error: {e}")
                market_open = False

            if not market_open:
                import pytz as _pytz
                _et = _pytz.timezone('America/New_York')
                _now_et = datetime.now(_et)
                _hour = _now_et.hour
                flat_overnight = getattr(config, 'SWING_FLAT_OVERNIGHT', False)

                # v3.5: Flatten all positions at end of day if flat_overnight mode
                if flat_overnight and _hour == 15 and _now_et.minute >= 55 and self.risk_manager.positions:
                    logger.info("=== EOD: Closing all positions (flat overnight mode) ===")
                    # Use actual Alpaca quantities — bot state may be stale
                    _alpaca_qty = {p["symbol"]: abs(p["qty"]) for p in self.broker.get_positions()}
                    for _sym in list(self.risk_manager.positions.keys()):
                        _pos = self.risk_manager.positions.get(_sym)
                        if not _pos:
                            continue
                        _qty = _alpaca_qty.get(_sym, 0)
                        if _qty <= 0:
                            logger.info(f"EOD SKIP: {_sym} — no shares on Alpaca, removing from tracking")
                            self.risk_manager.positions.pop(_sym, None)
                            continue
                        _price = self.broker.get_latest_price(_sym) or _pos.entry_price
                        _side = "sell" if _pos.side == "long" else "buy"
                        # Market is still open at 3:55PM — use normal smart_order
                        try:
                            _res = self.broker.smart_order(_sym, _qty, _side)
                            if "error" not in _res:
                                _pnl = (_price - _pos.entry_price) * _qty if _pos.side == "long" else (_pos.entry_price - _price) * _qty
                                self.risk_manager.close_position(_sym, _price, "EOD flat")
                                logger.info(f"EOD EXIT: {_sym} {_qty}x @ ${_price:.2f} | PnL: ${_pnl:+,.0f}")
                            else:
                                logger.error(f"EOD EXIT FAILED: {_sym} — {_res.get('error')}")
                        except Exception as _eod_err:
                            logger.error(f"EOD EXIT ERROR: {_sym} — {_eod_err}")
                else:
                    try:
                        self.check_exits()
                    except Exception:
                        pass

                positions_str = ", ".join(
                    f"{s}({p.quantity}x)" for s, p in self.risk_manager.positions.items()
                ) or "none"
                # Sleep longer overnight
                _sleep = 600 if (20 <= _hour or _hour < 7) else config.SCAN_INTERVAL_SECONDS
                _flat_tag = " [FLAT-OVERNIGHT]" if flat_overnight else ""
                logger.info(f"Market closed{_flat_tag} | Holding: [{positions_str}] | Next check in {_sleep}s...")
                time.sleep(_sleep)
                continue

            try:
                # 0. Update daily trend filter (cached 30min)
                self.update_daily_trends()

                # 0.5 Update earnings calendar (cached 6hrs)
                # Also feed earnings dates to options engine
                if self.options_engine:
                    try:
                        acct = self.broker.get_account()
                        equity = acct.get("equity", 100000)
                        # Run options check once per day (at 10 AM ET)
                        import pytz as _pytz2
                        _et2 = _pytz2.timezone('America/New_York')
                        _now2 = datetime.now(_et2)
                        if _now2.hour == 10 and _now2.minute < 2:
                            self.options_engine.run_daily_check(equity)
                    except Exception as _oe:
                        logger.debug(f"Options check error: {_oe}")

                if self.earnings_guard.needs_update():
                    watchlist = [s for s in config.WATCHLIST if s not in getattr(config, 'INVERSE_ETFS', [])]
                    self.earnings_guard.update_calendar(watchlist)
                    logger.info(self.earnings_guard.summary())

                # 0.6 Earnings-based forced exits (only if position up >10%)
                for symbol in list(self.risk_manager.positions.keys()):
                    risk = self.earnings_guard.check_earnings_risk(symbol)
                    if risk.action == "EXIT":
                        pos = self.risk_manager.positions[symbol]
                        price = self.broker.get_latest_price(symbol)
                        if price and price > 0:
                            unrealized_pct = (price - pos.entry_price) / pos.entry_price
                            # Only force-exit big winners (priced for perfection)
                            if unrealized_pct > 0.10:
                                side = "sell" if pos.side == "long" else "buy"
                                result = self.broker.smart_order(symbol, pos.quantity, side)
                                if "error" not in result:
                                    pnl = (price - pos.entry_price) * pos.quantity if pos.side == "long" else 0
                                    self.risk_manager.close_position(symbol, price, f"Earnings guard: {risk.reason}")
                                    logger.info(f"EARNINGS EXIT: {symbol} {pos.quantity}x @ ${price:.2f} | PnL: ${pnl:+,.0f} | +{unrealized_pct:.0%} into earnings")
                            else:
                                logger.info(f"EARNINGS HOLD: {symbol} only {unrealized_pct:+.1%} — not enough risk to exit")
                    elif risk.action == "REDUCE" and risk.position_scale < 1.0:
                        pos = self.risk_manager.positions[symbol]
                        price = self.broker.get_latest_price(symbol)
                        if price and price > 0:
                            unrealized_pct = (price - pos.entry_price) / pos.entry_price
                            if unrealized_pct > 0.10:
                                target_qty = int(pos.quantity * risk.position_scale)
                                reduce_qty = pos.quantity - target_qty
                                if reduce_qty > 0:
                                    side = "sell" if pos.side == "long" else "buy"
                                    result = self.broker.smart_order(symbol, reduce_qty, side)
                                    if "error" not in result:
                                        pos.quantity = target_qty
                                        logger.info(f"EARNINGS REDUCE: {symbol} sold {reduce_qty} shares, keeping {target_qty} | {risk.reason}")

                # 1. Check exits on existing positions
                self.check_exits()

                # No force-trim — exposure % gates new entries, stops handle exits

                # 2. Scan watchlist (using intraday bars)
                signals = self.scan_watchlist()

                # 2.5 Circuit breaker — if portfolio dropped >4% today, halt new entries
                prices_now = self.broker.get_latest_prices(list(self.risk_manager.positions.keys()))
                summary_now = self.risk_manager.summary(current_prices=prices_now)
                day_drop_pct = summary_now["true_day_pnl"] / self.risk_manager.portfolio_value if self.risk_manager.portfolio_value > 0 else 0
                max_daily_loss = getattr(config, 'MAX_DAILY_LOSS_PCT', 0.04)
                circuit_open = day_drop_pct <= -max_daily_loss

                if circuit_open:
                    logger.warning(
                        f"⚡ CIRCUIT BREAKER: Portfolio down {day_drop_pct:.1%} today "
                        f"(threshold: -{max_daily_loss:.0%}) — halting new entries, tightening stops"
                    )
                    # Tighten all existing stops to lock in remaining value
                    for sym, pos in self.risk_manager.positions.items():
                        if hasattr(pos, 'highest_price') and pos.highest_price > 0:
                            tighter = pos.highest_price * 0.97
                            pos.stop_loss = max(pos.stop_loss, tighter)

                # 2.6 Live SPY market regime gate
                # If SPY is down on the day, raise entry threshold automatically.
                # ML models trained in a bull market will be over-bullish on red days.
                # Gates: SPY down >1% → need score 0.35+, SPY down >2% → halt all entries
                _spy_gate_threshold = getattr(config, 'BULL_BUY_THRESHOLD', 0.25)
                _spy_halt = False
                try:
                    _spy_bars = self.broker.get_bars("SPY", timeframe="1Day", limit=2)
                    if _spy_bars is not None and len(_spy_bars) >= 2:
                        _spy_prev_close = _spy_bars.iloc[-2]["close"]
                        _spy_now_price  = self.broker.get_latest_price("SPY") or _spy_bars.iloc[-1]["close"]
                        _spy_day_chg    = (_spy_now_price - _spy_prev_close) / _spy_prev_close
                        if _spy_day_chg <= -0.02:
                            _spy_halt = True
                            logger.warning(f"🚨 SPY HALT: Market down {_spy_day_chg:.1%} today — halting all new entries (threshold: -2%)")
                        elif _spy_day_chg <= -0.01:
                            _spy_gate_threshold = max(_spy_gate_threshold, 0.35)
                            logger.info(f"⚠️  SPY CAUTION: Market down {_spy_day_chg:.1%} — raising entry threshold to {_spy_gate_threshold:.2f}")
                        else:
                            logger.debug(f"SPY day change: {_spy_day_chg:+.2%} — normal entry threshold {_spy_gate_threshold:.2f}")
                except Exception as _spy_err:
                    logger.debug(f"SPY regime check failed: {_spy_err} — using default threshold")

                # 3. Rank by score, take only top signals — skip if circuit breaker open
                # Timing gates — no entries too early or too late
                import pytz as _ptz
                _et_now = datetime.now(_ptz.timezone('America/New_York'))
                _no_before_h = getattr(config, 'NO_ENTRY_BEFORE_HOUR', 10)
                _no_before_m = getattr(config, 'NO_ENTRY_BEFORE_MINUTE', 0)
                _no_after_h  = getattr(config, 'NO_ENTRY_AFTER_HOUR', 15)
                _no_after_m  = getattr(config, 'NO_ENTRY_AFTER_MINUTE', 30)
                _too_early = (_et_now.hour < _no_before_h) or (_et_now.hour == _no_before_h and _et_now.minute < _no_before_m)
                _too_late  = (_et_now.hour > _no_after_h) or (_et_now.hour == _no_after_h and _et_now.minute >= _no_after_m)
                if _spy_halt:
                    actionable = []
                elif _too_early:
                    logger.info(f"Entry window not open: {_et_now.strftime('%H:%M')} ET — entries open at {_no_before_h:02d}:{_no_before_m:02d}")
                    actionable = []
                elif _too_late:
                    # After 3:30PM — only allow genuine breakouts (score >= 0.40)
                    # Normal entries blocked, but high conviction momentum plays get through
                    _late_threshold = 0.40
                    _late_signals = [s for s in signals if s.action != "HOLD" and s.composite_score >= _late_threshold]
                    if _late_signals:
                        logger.info(f"Late entry window: {_et_now.strftime('%H:%M')} ET — {len(_late_signals)} breakout signal(s) above {_late_threshold} threshold")
                        actionable = [] if circuit_open else _late_signals
                    else:
                        logger.info(f"Entry cutoff: {_et_now.strftime('%H:%M')} ET — no breakouts above {_late_threshold}, entries closed")
                        actionable = []
                else:
                    # Apply SPY-adjusted threshold — raises to 0.35 when market down 1%+
                    actionable = [] if circuit_open else [
                        s for s in signals
                        if s.action != "HOLD" and s.composite_score >= _spy_gate_threshold
                    ]
                    if _spy_gate_threshold > getattr(config, 'BULL_BUY_THRESHOLD', 0.25):
                        logger.info(f"SPY caution: {len(actionable)} signals above raised threshold {_spy_gate_threshold:.2f}")
                actionable.sort(key=lambda s: s.composite_score, reverse=True)

                # --- ROTATION ENGINE ---
                # If we're at max positions, check if any open position should be
                # swapped out for a higher-scoring signal.
                # Rules:
                #   - Incoming signal must score 0.10+ higher than weakest held position
                #   - Weakest position must be within 0.5% of entry (flat/small loss) — don't sell winners
                #   - Only rotate during trading window (same timing gates apply)
                #   - Max 1 rotation per cycle to avoid churn
                _max_pos = getattr(config, 'MAX_CONCURRENT_POSITIONS', 3)
                if not _too_early and not _too_late and not circuit_open and len(self.risk_manager.positions) >= _max_pos and actionable:
                    # Score all current positions using latest signals
                    _held_scores = {}
                    _signal_map = {s.symbol: s for s in signals}
                    for _sym, _pos in list(self.risk_manager.positions.items()):
                        _sig = _signal_map.get(_sym)
                        _held_scores[_sym] = _sig.composite_score if _sig else 0.0

                    # Find weakest held position
                    _weakest_sym = min(_held_scores, key=_held_scores.get)
                    _weakest_score = _held_scores[_weakest_sym]
                    _weakest_pos = self.risk_manager.positions[_weakest_sym]

                    # Get current price to check if flat
                    _weakest_price = self.broker.get_latest_price(_weakest_sym) or _weakest_pos.entry_price
                    _weakest_pnl_pct = (_weakest_price - _weakest_pos.entry_price) / _weakest_pos.entry_price

                    # Best incoming signal not already held
                    _best_new = next((s for s in actionable if s.symbol not in self.risk_manager.positions), None)

                    if _best_new:
                        _score_gap = _best_new.composite_score - _weakest_score
                        _is_flat = abs(_weakest_pnl_pct) <= 0.005  # within 0.5% of entry

                        if _score_gap >= 0.10 and _is_flat:
                            logger.info(
                                f"ROTATION: Selling {_weakest_sym} (score:{_weakest_score:.3f}, pnl:{_weakest_pnl_pct:+.2%}) "
                                f"→ Buying {_best_new.symbol} (score:{_best_new.composite_score:.3f}, gap:{_score_gap:+.3f})"
                            )
                            # Sell the weak position
                            _sell_qty = _weakest_pos.quantity
                            _sell_side = "sell" if _weakest_pos.side == "long" else "buy"
                            _sell_result = self.broker.smart_order(_weakest_sym, _sell_qty, _sell_side)
                            if "error" not in _sell_result:
                                _exit_price = self.broker.get_latest_price(_weakest_sym) or _weakest_price
                                self.risk_manager.close_position(_weakest_sym, _exit_price, f"Rotation out (score:{_weakest_score:.3f})")
                                # Now execute the better signal
                                self.execute_signal(_best_new)
                        elif _score_gap >= 0.10 and not _is_flat:
                            logger.info(
                                f"ROTATION SKIP: {_weakest_sym} score:{_weakest_score:.3f} but pnl:{_weakest_pnl_pct:+.2%} — "
                                f"not flat enough to rotate (need within 0.5%)"
                            )

                # Only try top signals (max positions worth)
                for sig in actionable[:20]:  # Process top 20 signals per cycle
                    self.execute_signal(sig)

                # 4. Update portfolio value
                acct = self.broker.get_account()
                if "equity" in acct:
                    self.risk_manager.update_portfolio_value(acct["equity"])

                # 5. Log portfolio summary — pass live prices for accurate exposure
                prices_now = self.broker.get_latest_prices(list(self.risk_manager.positions.keys()))
                summary = self.risk_manager.summary(current_prices=prices_now)

                unrealized_pnl = summary["unrealized_pnl"]
                true_day_pnl = summary["true_day_pnl"]

                positions_str = ", ".join(
                    f"{s}({p.quantity}x)" for s, p in self.risk_manager.positions.items()
                ) or "none"
                logger.info(
                    f"Portfolio: ${summary['portfolio_value']:,.2f} | "
                    f"Positions: {summary['open_positions']} [{positions_str}] | "
                    f"Exposure: {summary['exposure_pct']:.1f}% | "
                    f"Day PnL: ${true_day_pnl:+,.2f} (R:${summary['daily_pnl']:+,.0f} U:${unrealized_pnl:+,.0f}) | "
                    f"Trades: {summary['trades_today']}"
                )

            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)

            logger.info(f"Next scan in {config.SCAN_INTERVAL_SECONDS}s...")
            time.sleep(config.SCAN_INTERVAL_SECONDS)

        logger.info("Bot stopped.")


# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trading Bot v3.3")
    parser.add_argument("--scan-only", action="store_true", help="Scan without executing trades")
    args = parser.parse_args()

    bot = TradingBot()
    bot.run(scan_only=args.scan_only)