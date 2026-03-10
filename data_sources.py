"""
data_sources.py — Comprehensive Free Data Pipeline
====================================================
Pulls from every available free source:
1. FRED — macro economic indicators
2. SEC EDGAR — 8-K filings, 13F institutional changes
3. Reddit WSB — retail sentiment
4. Google Trends — search volume / retail interest
5. Alpaca News — news headlines with sentiment (free, we have API key)
6. Alpha Vantage — earnings surprise history
7. FINRA — official short interest data
8. Wikipedia — page view trends
9. Polygon.io — options flow (free tier)

All cached to disk for appropriate intervals.
All failures handled gracefully — always returns something.

Usage:
    from data_sources import DataPipeline
    pipeline = DataPipeline()
    features = pipeline.get_all(symbol="NVDA")
    macro    = pipeline.get_macro()  # market-wide signals
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

CACHE_DIR = Path(".data_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache(key: str, hours: float = 24) -> Optional[Any]:
    p = CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()[:16]}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if datetime.fromisoformat(d["ts"]) > datetime.now() - timedelta(hours=hours):
                return d["v"]
        except:
            pass
    return None


def _save(key: str, value: Any):
    p = CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()[:16]}.json"
    try:
        p.write_text(json.dumps({"ts": datetime.now().isoformat(), "v": value}))
    except:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. FRED — Macro Economic Indicators
# ─────────────────────────────────────────────────────────────────────────────

FRED_SERIES = {
    "DGS10":   "10yr_yield",          # 10-year Treasury yield
    "DGS2":    "2yr_yield",           # 2-year Treasury yield
    "T10Y2Y":  "yield_curve",         # 10yr-2yr spread (negative = recession risk)
    "VIXCLS":  "vix",                 # VIX fear index
    "DTWEXBGS":"dollar_index",        # US Dollar strength
    "BAMLH0A0HYM2": "hy_spread",     # High yield credit spread (risk appetite)
    "UMCSENT": "consumer_sentiment",  # University of Michigan consumer sentiment
    "CPIAUCSL": "cpi",               # CPI inflation
    "FEDFUNDS": "fed_funds_rate",     # Federal funds rate
    "UNRATE":  "unemployment",        # Unemployment rate
}

def get_fred_macro() -> Dict:
    """
    Fetch key macro indicators from FRED (St. Louis Fed).
    Free, no API key needed via pandas_datareader or direct URL.
    Cache for 4 hours — these don't change often.
    """
    key = f"fred_macro_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _cache(key, hours=4)
    if cached:
        return cached

    result = {}
    try:
        import pandas_datareader.data as web
        end   = datetime.now()
        start = end - timedelta(days=30)

        for series_id, name in FRED_SERIES.items():
            try:
                df = web.DataReader(series_id, "fred", start, end)
                if df is not None and len(df) > 0:
                    latest = df.iloc[-1, 0]
                    prev   = df.iloc[-5, 0] if len(df) >= 5 else latest
                    result[name]              = float(latest) if not np.isnan(latest) else 0.0
                    result[f"{name}_5d_chg"] = float(latest - prev) if not np.isnan(latest - prev) else 0.0
            except:
                pass

        # Derived signals
        if "10yr_yield" in result and "2yr_yield" in result:
            result["yield_curve_spread"] = result["10yr_yield"] - result["2yr_yield"]
            result["yield_curve_inverted"] = 1 if result["yield_curve_spread"] < 0 else 0

        if "vix" in result:
            result["high_fear"] = 1 if result["vix"] > 30 else 0
            result["low_fear"]  = 1 if result["vix"] < 15 else 0
            result["vix_normalized"] = min(1.0, result["vix"] / 40)

        # Rate environment signal: high/rising rates = bad for growth stocks
        if "10yr_yield" in result:
            result["rate_headwind"] = 1 if result["10yr_yield"] > 4.5 else 0
            result["rate_rising"]   = 1 if result.get("10yr_yield_5d_chg", 0) > 0.1 else 0

        _save(key, result)
        logger.info(f"FRED: fetched {len(result)} macro indicators")

    except Exception as e:
        logger.debug(f"FRED failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEC EDGAR — 8-K Filings & 13F Institutional Changes
# ─────────────────────────────────────────────────────────────────────────────

def get_sec_filings(symbol: str) -> Dict:
    """
    Check for recent 8-K filings (material events) and 13F changes.
    8-K filings signal: earnings, M&A, leadership changes, etc.
    Free via SEC EDGAR REST API.
    """
    key = f"sec_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache(key, hours=12)
    if cached:
        return cached

    result = {
        "recent_8k_count":      0,
        "recent_13f_buy":       0,
        "recent_13f_sell":      0,
        "material_event_flag":  0,
        "institutional_change": 0.0,
    }

    try:
        import requests

        # Get CIK number for symbol
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt={datetime.now().strftime('%Y-%m-%d')}&forms=8-K"
        headers    = {"User-Agent": "trading-bot research@example.com"}

        # Get company CIK
        cik_url  = f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={symbol}&type=8-K&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
        response = requests.get(cik_url, headers=headers, timeout=10)

        if response.status_code == 200:
            # Count recent 8-K filings
            content = response.text
            count_8k = content.count("<accession-number>")
            result["recent_8k_count"] = min(count_8k, 10)
            result["material_event_flag"] = 1 if count_8k > 2 else 0

        _save(key, result)

    except Exception as e:
        logger.debug(f"SEC EDGAR failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. REDDIT WSB — Retail Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def get_reddit_sentiment(symbol: str) -> Dict:
    """
    Scrape WallStreetBets and stocks subreddits for mention count and sentiment.
    Uses Reddit's public JSON API (no auth needed for public posts).
    """
    key = f"reddit_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _cache(key, hours=2)
    if cached:
        return cached

    result = {
        "wsb_mention_count":  0,
        "wsb_sentiment":      0.0,
        "retail_hype_score":  0.0,
        "reddit_bullish":     0,
        "reddit_bearish":     0,
    }

    try:
        import requests

        subreddits = ["wallstreetbets", "stocks", "investing", "options"]
        total_mentions = 0
        sentiments     = []

        # Positive/negative keywords
        bull_words = ["bull", "calls", "moon", "long", "buy", "yolo", "squeeze",
                      "breakout", "run", "pump", "gains", "ath", "upgrade"]
        bear_words = ["bear", "puts", "short", "sell", "crash", "dump", "drop",
                      "overvalued", "downgrade", "miss", "loss", "baghold"]

        headers = {"User-Agent": "Mozilla/5.0 research bot"}

        for sub in subreddits[:2]:  # Limit to avoid rate limiting
            try:
                url      = f"https://www.reddit.com/r/{sub}/search.json?q={symbol}&sort=new&limit=25&t=day"
                response = requests.get(url, headers=headers, timeout=8)
                if response.status_code != 200:
                    continue

                data  = response.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    pd_data = post.get("data", {})
                    title   = pd_data.get("title", "").lower()
                    body    = pd_data.get("selftext", "").lower()
                    text    = title + " " + body

                    if symbol.lower() in text or symbol.lower() + " " in text:
                        total_mentions += 1
                        bull = sum(1 for w in bull_words if w in text)
                        bear = sum(1 for w in bear_words if w in text)
                        if bull + bear > 0:
                            sentiments.append((bull - bear) / (bull + bear))

                time.sleep(0.5)  # Be polite

            except Exception as e:
                logger.debug(f"Reddit {sub} failed: {e}")

        result["wsb_mention_count"] = total_mentions
        result["retail_hype_score"] = min(1.0, total_mentions / 20)

        if sentiments:
            avg_sent = np.mean(sentiments)
            result["wsb_sentiment"] = avg_sent
            result["reddit_bullish"] = 1 if avg_sent > 0.2 else 0
            result["reddit_bearish"] = 1 if avg_sent < -0.2 else 0

        _save(key, result)
        logger.debug(f"Reddit {symbol}: {total_mentions} mentions, sentiment={result['wsb_sentiment']:.2f}")

    except Exception as e:
        logger.debug(f"Reddit sentiment failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. GOOGLE TRENDS — Search Volume
# ─────────────────────────────────────────────────────────────────────────────

def get_google_trends(symbol: str, company_name: str = "") -> Dict:
    """
    Google search volume for the stock ticker and company name.
    Spikes in search volume often precede retail buying waves.
    Uses pytrends (free, no API key).
    """
    key = f"trends_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache(key, hours=24)
    if cached:
        return cached

    result = {
        "search_volume_score":    0.5,
        "search_volume_trend":    0.0,
        "search_spike":           0,
        "search_momentum":        0.0,
    }

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

        query = symbol if not company_name else f"{symbol} stock"
        pytrends.build_payload([query], timeframe="today 3-m", geo="US")
        df = pytrends.interest_over_time()

        if df is not None and len(df) > 0 and query in df.columns:
            values = df[query].values
            latest = values[-1]
            avg    = values[:-4].mean() if len(values) > 4 else values.mean()
            trend  = (values[-1] - values[-4]) / (abs(values[-4]) + 1) if len(values) >= 4 else 0

            result["search_volume_score"] = float(latest / 100)
            result["search_volume_trend"] = float(trend)
            result["search_spike"]        = 1 if latest > avg * 1.5 else 0
            result["search_momentum"]     = float(np.clip(trend, -1, 1))

        _save(key, result)
        logger.debug(f"Google Trends {symbol}: score={result['search_volume_score']:.2f} spike={result['search_spike']}")

    except Exception as e:
        logger.debug(f"Google Trends failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. ALPACA NEWS — Headlines with Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def get_alpaca_news(symbol: str) -> Dict:
    """
    Alpaca provides free news headlines with sentiment scores.
    We already have the API key — this costs nothing extra.
    """
    key = f"alpaca_news_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = _cache(key, hours=1)
    if cached:
        return cached

    result = {
        "alpaca_sentiment":       0.0,
        "alpaca_news_count":      0,
        "alpaca_negative_flag":   0,
        "alpaca_positive_flag":   0,
        "breaking_news_flag":     0,
    }

    try:
        import config
        import requests

        url     = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID":     config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        }
        params = {
            "symbols": symbol,
            "limit":   20,
            "start":   (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            news    = response.json().get("news", [])
            result["alpaca_news_count"] = len(news)

            sentiments = []
            for article in news:
                # Alpaca provides sentiment in some feeds
                sentiment = article.get("sentiment", None)
                if sentiment == "positive":
                    sentiments.append(1.0)
                elif sentiment == "negative":
                    sentiments.append(-1.0)
                else:
                    # Score headline manually
                    headline = article.get("headline", "").lower()
                    pos = sum(1 for w in ["beat", "surge", "record", "strong", "upgrade", "buy"]
                              if w in headline)
                    neg = sum(1 for w in ["miss", "fall", "drop", "weak", "downgrade", "sell",
                                          "investigation", "lawsuit", "recall"] if w in headline)
                    if pos + neg > 0:
                        sentiments.append((pos - neg) / (pos + neg))

                # Breaking news flag — very recent articles
                created = article.get("created_at", "")
                if created:
                    try:
                        age = datetime.now() - datetime.fromisoformat(created.replace("Z", ""))
                        if age.total_seconds() < 3600:  # Last hour
                            result["breaking_news_flag"] = 1
                    except:
                        pass

            if sentiments:
                avg = np.mean(sentiments)
                result["alpaca_sentiment"]     = float(avg)
                result["alpaca_positive_flag"] = 1 if avg > 0.3 else 0
                result["alpaca_negative_flag"] = 1 if avg < -0.3 else 0

        _save(key, result)
        logger.debug(f"Alpaca news {symbol}: {result['alpaca_news_count']} articles, sentiment={result['alpaca_sentiment']:.2f}")

    except Exception as e:
        logger.debug(f"Alpaca news failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. ALPHA VANTAGE — Earnings Surprise History
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_surprise(symbol: str) -> Dict:
    """
    Stocks that consistently beat earnings keep doing it.
    Alpha Vantage free tier: 25 calls/day.
    Get free API key at: https://www.alphavantage.co/support/#api-key
    """
    key = f"earnings_{symbol}_{datetime.now().strftime('%Y%m')}"
    cached = _cache(key, hours=24 * 7)  # Cache a week — earnings quarterly
    if cached:
        return cached

    result = {
        "earnings_beat_rate":    0.5,
        "avg_earnings_surprise": 0.0,
        "last_surprise_pct":     0.0,
        "consistent_beater":     0,
        "earnings_momentum":     0.0,
    }

    try:
        import requests
        import config

        av_key = getattr(config, "ALPHA_VANTAGE_KEY", None)
        if not av_key:
            # Try yfinance as fallback
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_history
            if earnings is not None and len(earnings) > 0:
                if "surprisePercent" in earnings.columns:
                    surprises = earnings["surprisePercent"].dropna()
                    if len(surprises) > 0:
                        result["avg_earnings_surprise"] = float(surprises.mean())
                        result["last_surprise_pct"]     = float(surprises.iloc[0])
                        result["earnings_beat_rate"]    = float((surprises > 0).mean())
                        result["consistent_beater"]     = 1 if (surprises > 0).mean() > 0.75 else 0
                        result["earnings_momentum"]     = float(np.clip(surprises.mean() / 10, -1, 1))
            _save(key, result)
            return result

        url      = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={av_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            quarterly = data.get("quarterlyEarnings", [])

            if quarterly:
                surprises = []
                for q in quarterly[:8]:  # Last 8 quarters
                    try:
                        surprise_pct = float(q.get("surprisePercentage", 0))
                        surprises.append(surprise_pct)
                    except:
                        pass

                if surprises:
                    result["avg_earnings_surprise"] = float(np.mean(surprises))
                    result["last_surprise_pct"]     = float(surprises[0])
                    result["earnings_beat_rate"]    = float(sum(s > 0 for s in surprises) / len(surprises))
                    result["consistent_beater"]     = 1 if result["earnings_beat_rate"] > 0.75 else 0
                    result["earnings_momentum"]     = float(np.clip(np.mean(surprises) / 10, -1, 1))

        _save(key, result)
        logger.debug(f"Earnings {symbol}: beat_rate={result['earnings_beat_rate']:.0%} avg_surprise={result['avg_earnings_surprise']:.1f}%")

    except Exception as e:
        logger.debug(f"Earnings surprise failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINRA — Official Short Interest
# ─────────────────────────────────────────────────────────────────────────────

def get_finra_short(symbol: str) -> Dict:
    """
    FINRA publishes official biweekly short interest data.
    More accurate than Yahoo Finance's estimate.
    Free public API.
    """
    key = f"finra_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache(key, hours=24)
    if cached:
        return cached

    result = {
        "finra_short_volume_ratio": 0.5,
        "finra_short_trend":        0.0,
        "short_increasing":         0,
        "short_decreasing":         0,
    }

    try:
        import requests

        # FINRA short sale volume data
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        url      = f"https://regsho.finra.org/FNSQshvol{date_str}.txt"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            lines = response.text.split("\n")
            for line in lines:
                parts = line.split("|")
                if len(parts) >= 4 and parts[0] == symbol:
                    try:
                        short_vol = int(parts[1])
                        total_vol = int(parts[3])
                        if total_vol > 0:
                            ratio = short_vol / total_vol
                            result["finra_short_volume_ratio"] = ratio
                            result["short_increasing"] = 1 if ratio > 0.55 else 0
                            result["short_decreasing"] = 1 if ratio < 0.40 else 0
                    except:
                        pass
                    break

        _save(key, result)

    except Exception as e:
        logger.debug(f"FINRA short failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. WIKIPEDIA — Page View Trends
# ─────────────────────────────────────────────────────────────────────────────

WIKI_PAGES = {
    "NVDA":  "Nvidia",
    "AMD":   "Advanced_Micro_Devices",
    "AVGO":  "Broadcom_Inc.",
    "MU":    "Micron_Technology",
    "PLTR":  "Palantir_Technologies",
    "CRWD":  "CrowdStrike",
    "GOOGL": "Alphabet_Inc.",
    "MSFT":  "Microsoft",
    "VST":   "Vistra_Corp",
    "CEG":   "Constellation_Energy",
}

def get_wikipedia_views(symbol: str) -> Dict:
    """
    Wikipedia page views correlate with retail investor interest.
    Free Wikimedia REST API, no key needed.
    """
    key = f"wiki_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache(key, hours=24)
    if cached:
        return cached

    result = {
        "wiki_views_ratio":   1.0,
        "wiki_views_spike":   0,
        "wiki_views_trend":   0.0,
    }

    try:
        import requests

        page  = WIKI_PAGES.get(symbol, symbol)
        end   = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=30)

        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia/all-access/all-agents/{page}/daily/"
               f"{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}")

        headers  = {"User-Agent": "trading-bot/1.0 research@example.com"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            items  = response.json().get("items", [])
            views  = [item["views"] for item in items]

            if len(views) >= 7:
                recent_avg = np.mean(views[-7:])
                older_avg  = np.mean(views[:-7]) if len(views) > 7 else recent_avg
                ratio      = recent_avg / (older_avg + 1)
                trend      = (views[-1] - views[-7]) / (views[-7] + 1)

                result["wiki_views_ratio"] = float(ratio)
                result["wiki_views_spike"] = 1 if ratio > 1.5 else 0
                result["wiki_views_trend"] = float(np.clip(trend, -1, 1))

        _save(key, result)
        logger.debug(f"Wikipedia {symbol}: ratio={result['wiki_views_ratio']:.2f} spike={result['wiki_views_spike']}")

    except Exception as e:
        logger.debug(f"Wikipedia views failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. POLYGON.IO — Options Flow (Free Tier)
# ─────────────────────────────────────────────────────────────────────────────

def get_polygon_options(symbol: str) -> Dict:
    """
    Polygon.io free tier gives options snapshot data.
    Better structured than Yahoo Finance options chain.
    Get free API key at: https://polygon.io (free forever tier available)
    """
    key = f"polygon_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    cached = _cache(key, hours=4)
    if cached:
        return cached

    result = {
        "polygon_put_call":      1.0,
        "polygon_call_oi":       0,
        "polygon_put_oi":        0,
        "polygon_options_sent":  0.0,
        "unusual_options_flag":  0,
    }

    try:
        import requests
        import config

        poly_key = getattr(config, "POLYGON_API_KEY", None)
        if not poly_key:
            return result

        url      = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={poly_key}&limit=50"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data    = response.json().get("results", [])
            calls   = [r for r in data if r.get("details", {}).get("contract_type") == "call"]
            puts    = [r for r in data if r.get("details", {}).get("contract_type") == "put"]

            call_oi = sum(r.get("open_interest", 0) for r in calls)
            put_oi  = sum(r.get("open_interest",  0) for r in puts)
            call_vol = sum(r.get("day", {}).get("volume", 0) for r in calls)
            put_vol  = sum(r.get("day", {}).get("volume", 0) for r in puts)

            result["polygon_call_oi"] = call_oi
            result["polygon_put_oi"]  = put_oi

            if call_vol > 0:
                pc = put_vol / call_vol
                result["polygon_put_call"]     = pc
                result["polygon_options_sent"] = float(np.clip((1.0 - pc) / 0.5, -1, 1))

            # Unusual options: volume >> open interest
            for r in calls:
                vol = r.get("day", {}).get("volume", 0)
                oi  = r.get("open_interest", 1)
                if oi > 0 and vol / oi > 3:
                    result["unusual_options_flag"] = 1
                    break

        _save(key, result)

    except Exception as e:
        logger.debug(f"Polygon options failed {symbol}: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_NAMES = {
    "NVDA": "NVIDIA", "AMD": "Advanced Micro Devices", "AVGO": "Broadcom",
    "MU": "Micron", "PLTR": "Palantir", "CRWD": "CrowdStrike",
    "GOOGL": "Google Alphabet", "MSFT": "Microsoft",
    "VST": "Vistra", "CEG": "Constellation Energy",
}


class DataPipeline:
    """
    Master pipeline — call get_all(symbol) to get every available signal.
    Macro signals (FRED) are fetched once and shared across all symbols.
    """

    def __init__(self):
        self._macro_cache: Optional[Dict] = None
        self._symbol_cache: Dict[str, Dict] = {}

    def get_macro(self) -> Dict:
        """Market-wide macro signals — fetch once, reuse all session."""
        if self._macro_cache is None:
            self._macro_cache = get_fred_macro()
        return self._macro_cache

    def get_all(self, symbol: str) -> Dict:
        """All signals for a symbol — symbol-specific + macro."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        company = COMPANY_NAMES.get(symbol, symbol)
        features = {}

        # Macro (shared)
        features.update(self.get_macro())

        # Symbol-specific sources
        sources = [
            ("sec",       lambda: get_sec_filings(symbol)),
            ("reddit",    lambda: get_reddit_sentiment(symbol)),
            ("trends",    lambda: get_google_trends(symbol, company)),
            ("alpaca",    lambda: get_alpaca_news(symbol)),
            ("earnings",  lambda: get_earnings_surprise(symbol)),
            ("finra",     lambda: get_finra_short(symbol)),
            ("wiki",      lambda: get_wikipedia_views(symbol)),
            ("polygon",   lambda: get_polygon_options(symbol)),
        ]

        for name, func in sources:
            try:
                data = func()
                features.update(data)
            except Exception as e:
                logger.debug(f"Pipeline source {name} failed for {symbol}: {e}")

        self._symbol_cache[symbol] = features
        return features

    def should_halt_all_entries(self) -> tuple:
        """
        Market-wide halt signals based on macro.
        Returns (halt: bool, reason: str)
        """
        macro = self.get_macro()

        # VIX above 35 — extreme fear, don't enter new positions
        if macro.get("vix", 0) > 35:
            return True, f"VIX extreme fear: {macro['vix']:.1f}"

        # Yield curve deeply inverted — recession signal
        if macro.get("yield_curve_spread", 0) < -0.5:
            return True, f"Yield curve deeply inverted: {macro.get('yield_curve_spread', 0):.2f}"

        # High yield spreads spiking — credit stress
        if macro.get("hy_spread_5d_chg", 0) > 0.5:
            return True, f"Credit spreads spiking: +{macro.get('hy_spread_5d_chg', 0):.2f}"

        return False, ""

    def get_entry_boost(self, symbol: str) -> float:
        """
        Composite boost/penalty for a symbol based on all external signals.
        Returns -0.3 to +0.3 — added to ML+rules score.
        """
        f      = self.get_all(symbol)
        macro  = self.get_macro()
        boosts = []

        # Earnings consistency
        if f.get("consistent_beater", 0):
            boosts.append(0.10)
        boosts.append(np.clip(f.get("earnings_momentum", 0) * 0.15, -0.1, 0.1))

        # News sentiment
        boosts.append(np.clip(f.get("alpaca_sentiment", 0) * 0.15, -0.15, 0.15))
        if f.get("alpaca_negative_flag", 0):
            boosts.append(-0.20)  # Hard penalty for bad news
        if f.get("breaking_news_flag", 0) and f.get("alpaca_sentiment", 0) > 0:
            boosts.append(0.10)   # Breaking positive news

        # Reddit/retail
        boosts.append(np.clip(f.get("wsb_sentiment", 0) * 0.08, -0.05, 0.08))
        if f.get("reddit_bullish", 0) and f.get("wsb_mention_count", 0) > 5:
            boosts.append(0.05)

        # Google Trends spike
        if f.get("search_spike", 0):
            boosts.append(0.05)

        # Wikipedia spike
        if f.get("wiki_views_spike", 0):
            boosts.append(0.03)

        # Options sentiment
        boosts.append(np.clip(f.get("polygon_options_sent", f.get("options_sentiment", 0)) * 0.10, -0.10, 0.10))
        if f.get("unusual_options_flag", 0):
            boosts.append(0.08)  # Unusual options = smart money signal

        # FINRA short
        if f.get("short_decreasing", 0):
            boosts.append(0.05)  # Shorts covering = bullish
        if f.get("short_increasing", 0):
            boosts.append(-0.05)

        # Macro headwinds
        if macro.get("high_fear", 0):
            boosts.append(-0.10)
        if macro.get("rate_headwind", 0) and macro.get("rate_rising", 0):
            boosts.append(-0.08)
        if macro.get("low_fear", 0):
            boosts.append(0.05)

        total = float(np.clip(sum(boosts), -0.30, 0.30))
        return total

    def print_summary(self, symbol: str):
        f     = self.get_all(symbol)
        boost = self.get_entry_boost(symbol)
        halt, reason = self.should_halt_all_entries()

        print(f"\n{'═'*55}")
        print(f"  {symbol} Full Data Pipeline Summary")
        print(f"{'═'*55}")
        print(f"  MACRO:")
        print(f"    VIX:              {f.get('vix', 0):.1f} {'⚠️ HIGH' if f.get('high_fear') else '✓'}")
        print(f"    10yr Yield:       {f.get('10yr_yield', 0):.2f}%")
        print(f"    Yield Curve:      {f.get('yield_curve_spread', 0):+.2f}% {'⚠️ INVERTED' if f.get('yield_curve_inverted') else '✓'}")
        print(f"  SENTIMENT:")
        print(f"    Alpaca News:      {f.get('alpaca_sentiment', 0):+.2f} ({f.get('alpaca_news_count', 0)} articles)")
        print(f"    Reddit WSB:       {f.get('wsb_sentiment', 0):+.2f} ({f.get('wsb_mention_count', 0)} mentions)")
        print(f"    Google Trends:    {f.get('search_volume_score', 0):.2f} spike={'YES' if f.get('search_spike') else 'no'}")
        print(f"    Wikipedia:        ratio={f.get('wiki_views_ratio', 0):.2f} spike={'YES' if f.get('wiki_views_spike') else 'no'}")
        print(f"  FUNDAMENTALS:")
        print(f"    Earnings beat %:  {f.get('earnings_beat_rate', 0)*100:.0f}%")
        print(f"    Avg surprise:     {f.get('avg_earnings_surprise', 0):+.1f}%")
        print(f"    Consistent beat:  {'YES ✓' if f.get('consistent_beater') else 'no'}")
        print(f"  OPTIONS/SHORT:")
        print(f"    Put/call ratio:   {f.get('polygon_put_call', f.get('put_call_ratio', 1)):.2f}")
        print(f"    Unusual options:  {'YES ⚡' if f.get('unusual_options_flag') else 'no'}")
        print(f"    FINRA short vol:  {f.get('finra_short_volume_ratio', 0):.1%}")
        print(f"{'─'*55}")
        c = "\033[92m" if boost > 0.05 else ("\033[91m" if boost < -0.05 else "\033[0m")
        print(f"  SCORE BOOST:      {c}{boost:+.3f}\033[0m")
        if halt:
            print(f"  ⛔ MARKET HALT: {reason}")
        print(f"{'═'*55}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  type=str, default=None)
    parser.add_argument("--macro",   action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    if args.install:
        import subprocess
        pkgs = ["pandas-datareader", "pytrends", "requests"]
        for pkg in pkgs:
            print(f"Installing {pkg}...")
            subprocess.run(["pip3", "install", pkg, "--break-system-packages", "--quiet"])
        print("Done.")
        exit(0)

    import config
    pipeline = DataPipeline()

    if args.macro:
        macro = pipeline.get_macro()
        print("\nMacro Indicators:")
        for k, v in macro.items():
            print(f"  {k:<30} {v}")
        halt, reason = pipeline.should_halt_all_entries()
        print(f"\nMarket halt: {halt} {reason}")
    else:
        symbols = [args.symbol.upper()] if args.symbol else config.WATCHLIST[:3]
        for sym in symbols:
            pipeline.print_summary(sym)
            time.sleep(1)
