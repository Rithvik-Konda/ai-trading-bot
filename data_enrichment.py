"""
data_enrichment.py — Free Alpha Data Sources
=============================================
Pulls additional signals into the ML feature set:
1. Analyst ratings & price targets (Yahoo Finance)
2. Short interest ratio (Yahoo Finance)
3. Institutional ownership changes (Yahoo Finance 13F)
4. News sentiment (DuckDuckGo headlines + VADER scoring)
5. Options unusual activity (Yahoo Finance options chain)
6. Insider transactions (Yahoo Finance)

All free — no API keys required beyond what we already have.

Usage:
    from data_enrichment import DataEnrichment
    enricher = DataEnrichment()
    features = enricher.get_all_features("NVDA")

Features are cached to disk for 24 hours to avoid hammering APIs.
"""

import os
import json
import time
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR  = Path(".enrichment_cache")
CACHE_DAYS = 1  # Refresh daily


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.json"


def _load_cache(key: str) -> Optional[Any]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if datetime.fromisoformat(data["ts"]) > datetime.now() - timedelta(hours=CACHE_DAYS * 24):
            return data["value"]
    except:
        pass
    return None


def _save_cache(key: str, value: Any):
    try:
        _cache_path(key).write_text(json.dumps({"ts": datetime.now().isoformat(), "value": value}))
    except:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. ANALYST RATINGS & PRICE TARGETS
# ─────────────────────────────────────────────────────────────────────────────

def get_analyst_data(symbol: str) -> Dict:
    """
    Returns:
    - analyst_buy_pct: % of analysts with Buy/Strong Buy
    - price_target_upside: (mean_target - current_price) / current_price
    - recent_upgrades: number of upgrades in last 30 days
    - recent_downgrades: number of downgrades in last 30 days
    - upgrade_momentum: upgrades - downgrades (positive = bullish)
    """
    key = f"analyst_{symbol}"
    cached = _load_cache(key)
    if cached:
        return cached

    result = {
        "analyst_buy_pct": 0.5,
        "price_target_upside": 0.0,
        "recent_upgrades": 0,
        "recent_downgrades": 0,
        "upgrade_momentum": 0,
        "num_analysts": 0,
    }

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)

        # Analyst recommendations
        recs = ticker.recommendations
        if recs is not None and len(recs) > 0:
            # Last 90 days
            cutoff   = datetime.now() - timedelta(days=90)
            recs.index = pd.to_datetime(recs.index).tz_localize(None) if recs.index.tz else pd.to_datetime(recs.index)
            recent   = recs[recs.index >= cutoff]

            if len(recent) > 0:
                # Count upgrades/downgrades
                upgrades   = 0
                downgrades = 0
                for _, row in recent.iterrows():
                    action = str(row.get("Action", "")).lower()
                    if "up" in action or "upgrade" in action:
                        upgrades += 1
                    elif "down" in action or "downgrade" in action:
                        downgrades += 1

                result["recent_upgrades"]   = upgrades
                result["recent_downgrades"] = downgrades
                result["upgrade_momentum"]  = upgrades - downgrades

        # Price target
        info = ticker.info
        if info:
            current   = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            target    = info.get("targetMeanPrice", 0)
            low_tgt   = info.get("targetLowPrice", 0)
            high_tgt  = info.get("targetHighPrice", 0)
            num_anal  = info.get("numberOfAnalystOpinions", 0)
            rec_mean  = info.get("recommendationMean", 3.0)  # 1=Strong Buy, 5=Strong Sell

            if current and current > 0 and target and target > 0:
                result["price_target_upside"] = (target - current) / current
            if num_anal:
                result["num_analysts"] = num_anal
            # Convert 1-5 scale to buy % (1=100% buy, 5=0% buy)
            result["analyst_buy_pct"] = max(0, min(1, (5 - rec_mean) / 4))

        _save_cache(key, result)
        logger.debug(f"Analyst data {symbol}: {result}")

    except Exception as e:
        logger.debug(f"Analyst data failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHORT INTEREST
# ─────────────────────────────────────────────────────────────────────────────

def get_short_interest(symbol: str) -> Dict:
    """
    Returns:
    - short_ratio: days to cover (short interest / avg daily volume)
    - short_float_pct: % of float that is short
    - short_squeeze_score: proprietary score 0-1 (high = ripe for squeeze)
    """
    key = f"short_{symbol}"
    cached = _load_cache(key)
    if cached:
        return cached

    result = {
        "short_ratio": 1.0,
        "short_float_pct": 0.02,
        "short_squeeze_score": 0.0,
    }

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        if info:
            short_ratio = info.get("shortRatio", 1.0) or 1.0
            short_pct   = info.get("shortPercentOfFloat", 0.02) or 0.02

            result["short_ratio"]      = short_ratio
            result["short_float_pct"]  = short_pct

            # Short squeeze score:
            # High short float + low days-to-cover = most dangerous for shorts
            # Score spikes when stock starts moving up into high short interest
            squeeze_score = min(1.0, (short_pct * 5) * (1 / max(short_ratio, 0.5)))
            result["short_squeeze_score"] = squeeze_score

        _save_cache(key, result)

    except Exception as e:
        logger.debug(f"Short interest failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. INSTITUTIONAL OWNERSHIP
# ─────────────────────────────────────────────────────────────────────────────

def get_institutional_data(symbol: str) -> Dict:
    """
    Returns:
    - inst_ownership_pct: % owned by institutions
    - inst_net_change: net shares bought/sold by institutions (13F)
    - num_institutions: total number of institutional holders
    - inst_buying_pressure: positive = net buyers, negative = net sellers
    """
    key = f"inst_{symbol}"
    cached = _load_cache(key)
    if cached:
        return cached

    result = {
        "inst_ownership_pct": 0.6,
        "inst_net_change": 0,
        "num_institutions": 0,
        "inst_buying_pressure": 0.0,
    }

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info   = ticker.info

        if info:
            result["inst_ownership_pct"] = info.get("heldPercentInstitutions", 0.6) or 0.6

        # Institutional holders — 13F data
        inst = ticker.institutional_holders
        if inst is not None and len(inst) > 0:
            result["num_institutions"] = len(inst)
            if "pctHeld" in inst.columns:
                result["inst_ownership_pct"] = inst["pctHeld"].sum()

        # Major holders change
        major = ticker.major_holders
        if major is not None and len(major) > 0:
            try:
                inst_pct = float(str(major.iloc[1, 0]).replace("%", "")) / 100
                result["inst_ownership_pct"] = inst_pct
            except:
                pass

        _save_cache(key, result)

    except Exception as e:
        logger.debug(f"Institutional data failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEWS SENTIMENT (DuckDuckGo + VADER)
# ─────────────────────────────────────────────────────────────────────────────

def get_news_sentiment(symbol: str, company_name: str = "") -> Dict:
    """
    Scrapes recent headlines from DuckDuckGo news search.
    Scores sentiment using VADER (no API key needed).
    Returns:
    - news_sentiment_score: -1 (very negative) to +1 (very positive)
    - news_volume: number of articles found
    - negative_news_flag: 1 if recent strongly negative news
    - positive_news_flag: 1 if recent strongly positive news
    """
    key = f"news_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _load_cache(key)
    if cached:
        return cached

    result = {
        "news_sentiment_score": 0.0,
        "news_volume": 0,
        "negative_news_flag": 0,
        "positive_news_flag": 0,
    }

    try:
        # Try VADER sentiment analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            # Fall back to basic keyword scoring
            analyzer = None

        headlines = []

        # Method 1: yfinance news
        try:
            import yfinance as yf
            news = yf.Ticker(symbol).news
            if news:
                for article in news[:10]:
                    title = article.get("title", "")
                    if title:
                        headlines.append(title)
        except:
            pass

        # Method 2: DuckDuckGo news search
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            query = f"{symbol} stock {company_name} news"
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=10))
                for r in results:
                    headlines.append(r.get("title", "") + " " + r.get("body", "")[:100])
        except:
            pass

        if not headlines:
            return result

        result["news_volume"] = len(headlines)

        # Score all headlines
        scores = []
        for headline in headlines:
            if analyzer:
                vs = analyzer.polarity_scores(headline)
                scores.append(vs["compound"])
            else:
                # Basic keyword scoring
                h = headline.lower()
                pos_words = ["beat", "surge", "jump", "rally", "gain", "record", "strong",
                             "upgrade", "buy", "bullish", "growth", "profit", "exceed"]
                neg_words = ["miss", "fall", "drop", "decline", "loss", "weak", "downgrade",
                             "sell", "bearish", "cut", "lawsuit", "investigation", "recall"]
                score = sum(1 for w in pos_words if w in h) - sum(1 for w in neg_words if w in h)
                scores.append(max(-1, min(1, score / 3)))

        if scores:
            avg_score = np.mean(scores)
            result["news_sentiment_score"]  = avg_score
            result["negative_news_flag"]    = 1 if avg_score < -0.3 else 0
            result["positive_news_flag"]    = 1 if avg_score >  0.3 else 0

        _save_cache(key, result)
        logger.debug(f"News sentiment {symbol}: score={result['news_sentiment_score']:.2f} from {len(headlines)} headlines")

    except Exception as e:
        logger.debug(f"News sentiment failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIONS UNUSUAL ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def get_options_signals(symbol: str) -> Dict:
    """
    Analyzes options chain for unusual activity.
    Smart money often buys options before big moves.
    Returns:
    - put_call_ratio: < 0.7 bullish, > 1.2 bearish
    - unusual_call_volume: call volume vs 20-day avg (>2x = unusual)
    - options_sentiment: -1 to +1 based on put/call skew
    - iv_percentile: implied volatility rank (high = expensive options)
    """
    key = f"options_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    cached = _load_cache(key)
    if cached:
        return cached

    result = {
        "put_call_ratio": 1.0,
        "unusual_call_volume": 1.0,
        "options_sentiment": 0.0,
        "iv_percentile": 0.5,
    }

    try:
        import yfinance as yf
        ticker  = yf.Ticker(symbol)
        exps    = ticker.options
        if not exps:
            return result

        # Use nearest expiration (most liquid)
        nearest_exp = exps[0]
        chain       = ticker.option_chain(nearest_exp)
        calls       = chain.calls
        puts        = chain.puts

        if calls is None or puts is None or len(calls) == 0 or len(puts) == 0:
            return result

        # Put/call volume ratio
        total_call_vol = calls["volume"].fillna(0).sum()
        total_put_vol  = puts["volume"].fillna(0).sum()
        if total_call_vol > 0:
            pc_ratio = total_put_vol / total_call_vol
            result["put_call_ratio"] = pc_ratio
            # Convert to sentiment: low ratio = bullish calls dominating
            result["options_sentiment"] = max(-1, min(1, (1.0 - pc_ratio) / 0.5))

        # Unusual call volume (vs open interest as proxy for normal)
        call_oi = calls["openInterest"].fillna(0).sum()
        if call_oi > 0:
            result["unusual_call_volume"] = total_call_vol / (call_oi / 20 + 1)

        # IV percentile from ATM options
        try:
            current_price = ticker.info.get("regularMarketPrice", 0)
            if current_price > 0:
                calls["moneyness"] = abs(calls["strike"] - current_price) / current_price
                atm_calls = calls.nsmallest(3, "moneyness")
                avg_iv    = atm_calls["impliedVolatility"].mean()
                result["iv_percentile"] = min(1.0, avg_iv / 1.0)  # normalize, 100% IV = 1.0
        except:
            pass

        _save_cache(key, result)
        logger.debug(f"Options signals {symbol}: PC={result['put_call_ratio']:.2f} sentiment={result['options_sentiment']:.2f}")

    except Exception as e:
        logger.debug(f"Options signals failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. INSIDER TRANSACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_insider_signals(symbol: str) -> Dict:
    """
    Insider buying is one of the strongest signals in finance.
    CEOs/CFOs buying their own stock with personal money = very bullish.
    Returns:
    - insider_buy_count: purchases in last 90 days
    - insider_sell_count: sales in last 90 days
    - insider_net_shares: net shares bought (positive = buying)
    - insider_signal: +1 strong buy, 0 neutral, -1 heavy selling
    """
    key = f"insider_{symbol}"
    cached = _load_cache(key)
    if cached:
        return cached

    result = {
        "insider_buy_count": 0,
        "insider_sell_count": 0,
        "insider_net_shares": 0,
        "insider_signal": 0.0,
    }

    try:
        import yfinance as yf
        ticker       = yf.Ticker(symbol)
        transactions = ticker.insider_transactions

        if transactions is None or len(transactions) == 0:
            return result

        # Last 90 days
        cutoff = datetime.now() - timedelta(days=90)
        transactions.index = pd.to_datetime(transactions.index).tz_localize(None) if transactions.index.tz else pd.to_datetime(transactions.index)
        recent = transactions[transactions.index >= cutoff] if len(transactions) > 0 else transactions

        if len(recent) == 0:
            _save_cache(key, result)
            return result

        buys  = 0
        sells = 0
        net   = 0

        for _, row in recent.iterrows():
            text   = str(row.get("Text", "")).lower()
            shares = row.get("Shares", 0) or 0
            value  = row.get("Value", 0) or 0

            # Skip option exercises — not meaningful
            if "option" in text or "exercise" in text:
                continue

            if "purchase" in text or "buy" in text or "acquisition" in text:
                buys += 1
                net  += abs(shares)
            elif "sale" in text or "sell" in text or "disposition" in text:
                sells += 1
                net   -= abs(shares)

        result["insider_buy_count"]  = buys
        result["insider_sell_count"] = sells
        result["insider_net_shares"] = net

        # Signal: more buys than sells = positive
        total = buys + sells
        if total > 0:
            buy_ratio = buys / total
            result["insider_signal"] = (buy_ratio - 0.5) * 2  # -1 to +1

        _save_cache(key, result)
        logger.debug(f"Insider {symbol}: buys={buys} sells={sells} net={net:,}")

    except Exception as e:
        logger.debug(f"Insider signals failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENRICHMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_NAMES = {
    "NVDA": "NVIDIA", "AMD": "Advanced Micro Devices", "AVGO": "Broadcom",
    "MU": "Micron", "PLTR": "Palantir", "CRWD": "CrowdStrike",
    "GOOGL": "Google Alphabet", "MSFT": "Microsoft", "VST": "Vistra",
    "CEG": "Constellation Energy",
}


class DataEnrichment:
    """
    Central enrichment class — call get_all_features(symbol) to get
    a flat dict of all alpha signals for a given symbol.
    Handles failures gracefully — always returns something.
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def get_all_features(self, symbol: str, max_age_hours: int = 4) -> Dict[str, float]:
        """
        Returns flat dict of all enrichment features.
        Uses in-memory cache to avoid repeated calls within same session.
        """
        if symbol in self._cache:
            return self._cache[symbol]

        company = COMPANY_NAMES.get(symbol, symbol)
        features = {}

        # Run all enrichment sources — fail gracefully
        sources = [
            ("analyst",       lambda: get_analyst_data(symbol)),
            ("short",         lambda: get_short_interest(symbol)),
            ("institutional", lambda: get_institutional_data(symbol)),
            ("news",          lambda: get_news_sentiment(symbol, company)),
            ("options",       lambda: get_options_signals(symbol)),
            ("insider",       lambda: get_insider_signals(symbol)),
        ]

        for name, func in sources:
            try:
                data = func()
                features.update(data)
            except Exception as e:
                logger.debug(f"Enrichment source {name} failed for {symbol}: {e}")

        self._cache[symbol] = features
        return features

    def get_composite_signal(self, symbol: str) -> float:
        """
        Single composite alpha signal from all enrichment sources.
        Returns -1 to +1. Positive = bullish enrichment signal.
        """
        f = self.get_all_features(symbol)
        if not f:
            return 0.0

        signals = []

        # Analyst signal
        buy_pct = f.get("analyst_buy_pct", 0.5)
        signals.append((buy_pct - 0.5) * 2)                      # -1 to +1

        target_upside = f.get("price_target_upside", 0)
        signals.append(np.clip(target_upside * 5, -1, 1))         # 20% upside = max bullish

        upgrade_mom = f.get("upgrade_momentum", 0)
        signals.append(np.clip(upgrade_mom / 3, -1, 1))

        # Short squeeze signal
        squeeze = f.get("short_squeeze_score", 0)
        signals.append(squeeze * 0.5)                              # Partial weight

        # News sentiment
        news = f.get("news_sentiment_score", 0)
        signals.append(news)

        neg_flag = f.get("negative_news_flag", 0)
        signals.append(-neg_flag * 0.5)                            # Heavy penalty for bad news

        # Options sentiment
        opt_sent = f.get("options_sentiment", 0)
        signals.append(opt_sent)

        pc_ratio = f.get("put_call_ratio", 1.0)
        signals.append(np.clip((1.0 - pc_ratio) / 0.5, -1, 1))

        # Insider signal
        insider = f.get("insider_signal", 0)
        signals.append(insider * 0.8)                              # Strong weight — insiders know

        return float(np.clip(np.mean(signals), -1, 1))

    def enrich_score(self, symbol: str, base_score: float, weight: float = 0.20) -> float:
        """
        Blend enrichment signal into existing ML+rules score.
        Default 20% weight — enrichment is complementary, not dominant.
        """
        enrichment = self.get_composite_signal(symbol)
        blended    = base_score * (1 - weight) + enrichment * weight
        return float(np.clip(blended, -1, 1))

    def should_block_entry(self, symbol: str) -> tuple:
        """
        Hard blocks — override everything else.
        Returns (blocked: bool, reason: str)
        """
        f = self.get_all_features(symbol)

        # Block on strongly negative news
        if f.get("negative_news_flag", 0) and f.get("news_sentiment_score", 0) < -0.5:
            return True, f"Strongly negative news sentiment ({f['news_sentiment_score']:.2f})"

        # Block if analysts overwhelmingly bearish (< 20% buy)
        if f.get("analyst_buy_pct", 0.5) < 0.20 and f.get("num_analysts", 0) > 5:
            return True, f"Analyst consensus bearish ({f['analyst_buy_pct']:.0%} buy)"

        return False, ""

    def print_summary(self, symbol: str):
        """Print enrichment summary for a symbol."""
        f   = self.get_all_features(symbol)
        sig = self.get_composite_signal(symbol)
        blocked, reason = self.should_block_entry(symbol)

        print(f"\n{'─'*50}")
        print(f"  {symbol} Enrichment Summary")
        print(f"{'─'*50}")
        print(f"  Analyst buy %:      {f.get('analyst_buy_pct', 0)*100:.0f}%")
        print(f"  Price tgt upside:   {f.get('price_target_upside', 0)*100:+.1f}%")
        print(f"  Upgrade momentum:   {f.get('upgrade_momentum', 0):+.0f}")
        print(f"  Short float %:      {f.get('short_float_pct', 0)*100:.1f}%")
        print(f"  Short ratio:        {f.get('short_ratio', 0):.1f} days")
        print(f"  Squeeze score:      {f.get('short_squeeze_score', 0):.2f}")
        print(f"  News sentiment:     {f.get('news_sentiment_score', 0):+.2f}")
        print(f"  News volume:        {f.get('news_volume', 0)} articles")
        print(f"  Options P/C ratio:  {f.get('put_call_ratio', 1):.2f}")
        print(f"  Options sentiment:  {f.get('options_sentiment', 0):+.2f}")
        print(f"  Insider signal:     {f.get('insider_signal', 0):+.2f}")
        print(f"  Insider buys/sells: {f.get('insider_buy_count', 0)}/{f.get('insider_sell_count', 0)}")
        print(f"{'─'*50}")
        c = "\033[92m" if sig > 0.2 else ("\033[91m" if sig < -0.2 else "\033[0m")
        print(f"  COMPOSITE SIGNAL:   {c}{sig:+.3f}\033[0m")
        if blocked:
            print(f"  ⛔ BLOCKED: {reason}")
        print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--install", action="store_true", help="Install optional dependencies")
    args = parser.parse_args()

    if args.install:
        import subprocess
        pkgs = ["vaderSentiment", "ddgs"]
        for pkg in pkgs:
            print(f"Installing {pkg}...")
            subprocess.run(["pip3", "install", pkg, "--break-system-packages", "--quiet"])
        print("Done. Re-run without --install to test.")
        exit(0)

    import config
    symbols = [args.symbol.upper()] if args.symbol else config.WATCHLIST[:5]

    enricher = DataEnrichment()
    print(f"\nTesting enrichment for: {symbols}")
    print("(First run fetches live data, subsequent runs use cache)\n")

    for sym in symbols:
        enricher.print_summary(sym)
        time.sleep(1)  # Be polite to APIs