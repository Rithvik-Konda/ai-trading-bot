"""
News Sentiment Analysis Module — v3.2 (FinBERT + Alpaca News)
==============================================================
Two-tier sentiment system:
1. Alpaca News API (free, no rate limit) for news fetching
2. FinBERT transformer model for sentiment scoring (if available)
3. Falls back to keyword scoring if FinBERT not installed

Changes from v3.1:
- Replaced NewsAPI with Alpaca's built-in news endpoint (no rate limits!)
- Added FinBERT transformer sentiment (pip install transformers torch)
- Keyword scorer kept as fallback
- 1-hour cache per symbol

Install for FinBERT:
    pip install transformers torch --break-system-packages
    # First run will download ~400MB model, then it's cached
"""

import re
import logging
import time as _time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import config

logger = logging.getLogger(__name__)

# Try to load FinBERT
_finbert_pipeline = None
_finbert_loaded = False

def _load_finbert():
    """Lazy-load FinBERT model on first use."""
    global _finbert_pipeline, _finbert_loaded
    if _finbert_loaded:
        return _finbert_pipeline
    _finbert_loaded = True
    try:
        from transformers import pipeline
        logger.info("Loading FinBERT model (first time may take a minute)...")
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,  # CPU only
            top_k=None,  # Return all scores
        )
        logger.info("FinBERT loaded successfully")
    except ImportError:
        logger.warning("transformers not installed — using keyword fallback. "
                       "Install with: pip install transformers torch")
    except Exception as e:
        logger.warning(f"FinBERT load failed: {e} — using keyword fallback")
    return _finbert_pipeline


# --- Keyword Fallback Lexicon ---
BULLISH_WORDS = {
    "surge": 2.0, "soar": 2.0, "skyrocket": 2.0, "breakout": 1.8,
    "rally": 1.8, "boom": 1.7, "blowout": 1.8,
    "beat": 1.3, "exceed": 1.3, "upgrade": 1.5, "outperform": 1.5,
    "growth": 1.2, "profit": 1.2, "bullish": 1.5, "upside": 1.3,
    "gain": 1.1, "rise": 1.0, "positive": 1.0, "strong": 1.1,
    "record": 1.3, "momentum": 1.1, "expansion": 1.2,
    "optimistic": 1.3, "recovery": 1.2, "demand": 1.0,
    "dividend": 1.0, "buyback": 1.3, "approval": 1.3,
    "breakthrough": 1.5, "buy": 1.2,
}

BEARISH_WORDS = {
    "crash": -2.0, "plunge": -2.0, "collapse": -2.0, "tank": -1.8,
    "freefall": -2.0, "selloff": -1.8, "capitulation": -1.8,
    "miss": -1.3, "downgrade": -1.5, "underperform": -1.5, "decline": -1.2,
    "loss": -1.2, "bearish": -1.5, "downside": -1.3, "weak": -1.1,
    "drop": -1.0, "fall": -1.0, "negative": -1.0, "risk": -0.8,
    "recession": -1.5, "inflation": -0.9, "layoff": -1.3, "lawsuit": -1.2,
    "investigation": -1.1, "scandal": -1.5, "fraud": -1.8, "bankruptcy": -2.0,
    "debt": -0.8, "concern": -0.7, "warning": -1.0, "sell": -1.2,
    "volatility": -0.6, "uncertainty": -0.8, "default": -1.5, "tariff": -0.9,
}


@dataclass
class NewsArticle:
    """Single news article with sentiment."""
    title: str
    source: str
    published_at: str
    url: str
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    matched_keywords: list = field(default_factory=list)


@dataclass
class SentimentSignals:
    """Container for sentiment analysis results."""
    symbol: str
    article_count: int = 0
    avg_sentiment: float = 0.0
    weighted_sentiment: float = 0.0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    top_articles: List[NewsArticle] = field(default_factory=list)
    score: float = 0.0
    signals: list = field(default_factory=list)
    method: str = "none"  # "finbert", "keyword", or "none"


class NewsSentimentAnalyzer:
    """
    Fetches news via Alpaca API (free, unlimited) and scores with FinBERT.
    Falls back to keyword scoring if FinBERT is not installed.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.NEWS_API_KEY
        self._cache: Dict[str, tuple] = {}  # {symbol: (timestamp, articles)}
        self._cache_ttl = 3600  # 1 hour
        self._alpaca_api = None
        self._init_alpaca()

    def _init_alpaca(self):
        """Initialize Alpaca API for news fetching."""
        try:
            import alpaca_trade_api as tradeapi
            self._alpaca_api = tradeapi.REST(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                config.ALPACA_BASE_URL,
            )
            logger.debug("Alpaca news API ready")
        except Exception as e:
            logger.warning(f"Alpaca news init failed: {e}")

    # --- Sentiment Scoring ---

    def score_text_finbert(self, text: str) -> tuple:
        """Score text using FinBERT transformer model.
        Returns (score in [-1,1], label)."""
        pipe = _load_finbert()
        if pipe is None:
            return self.score_text_keywords(text)

        try:
            # FinBERT has max 512 tokens, truncate
            result = pipe(text[:512])
            # result is list of list of dicts: [[{label, score}, ...]]
            scores = result[0] if isinstance(result[0], list) else result

            pos = 0.0
            neg = 0.0
            neu = 0.0
            for item in scores:
                if item["label"] == "positive":
                    pos = item["score"]
                elif item["label"] == "negative":
                    neg = item["score"]
                else:
                    neu = item["score"]

            # Convert to [-1, 1] scale
            sentiment = pos - neg
            if pos > neg and pos > neu:
                label = "bullish"
            elif neg > pos and neg > neu:
                label = "bearish"
            else:
                label = "neutral"

            return sentiment, label
        except Exception as e:
            logger.debug(f"FinBERT error: {e}")
            return self.score_text_keywords(text)

    def score_text_keywords(self, text: str) -> tuple:
        """Fallback keyword-based scoring. Returns (score, label)."""
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', text_lower)
        total_score = 0.0
        matches = 0

        for word in words:
            if word in BULLISH_WORDS:
                total_score += BULLISH_WORDS[word]
                matches += 1
            elif word in BEARISH_WORDS:
                total_score += BEARISH_WORDS[word]
                matches += 1

        num_scored = max(matches, 1)
        normalized = total_score / (num_scored ** 0.5)
        clamped = max(-1.0, min(1.0, normalized / 3.0))

        if clamped >= config.SENTIMENT_BULLISH_THRESHOLD:
            label = "bullish"
        elif clamped <= config.SENTIMENT_BEARISH_THRESHOLD:
            label = "bearish"
        else:
            label = "neutral"

        return clamped, label

    # --- News Fetching ---

    def fetch_news_alpaca(self, symbol: str) -> List[Dict]:
        """Fetch news via Alpaca API — free, no rate limits."""
        if self._alpaca_api is None:
            return []

        # Check cache
        if symbol in self._cache:
            cached_time, cached_articles = self._cache[symbol]
            if _time.time() - cached_time < self._cache_ttl:
                return cached_articles

        try:
            news = self._alpaca_api.get_news(symbol=symbol, limit=10)
            articles = []
            for item in news:
                articles.append({
                    "title": item.headline or "",
                    "description": item.summary or "",
                    "source": item.source or "Unknown",
                    "publishedAt": str(item.created_at) if item.created_at else "",
                    "url": item.url or "",
                })
            self._cache[symbol] = (_time.time(), articles)
            return articles
        except Exception as e:
            logger.debug(f"Alpaca news fetch failed for {symbol}: {e}")
            # Return stale cache if available
            if symbol in self._cache:
                return self._cache[symbol][1]
            return []

    def fetch_news_newsapi(self, symbol: str) -> List[Dict]:
        """Fallback: NewsAPI (rate limited on free tier)."""
        # Check cache
        if symbol in self._cache:
            cached_time, cached_articles = self._cache[symbol]
            if _time.time() - cached_time < self._cache_ttl:
                return cached_articles

        try:
            import requests
        except ImportError:
            return []

        from_date = (
            datetime.utcnow() - timedelta(hours=config.NEWS_LOOKBACK_HOURS)
        ).strftime("%Y-%m-%dT%H:%M:%S")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 10,
            "apiKey": self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                logger.debug(f"NewsAPI rate limited for {symbol}")
                if symbol in self._cache:
                    return self._cache[symbol][1]
                return []
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            # Normalize format
            normalized = []
            for a in articles:
                normalized.append({
                    "title": a.get("title", "") or "",
                    "description": a.get("description", "") or "",
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "publishedAt": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                })
            self._cache[symbol] = (_time.time(), normalized)
            return normalized
        except Exception as e:
            logger.debug(f"NewsAPI failed for {symbol}: {e}")
            if symbol in self._cache:
                return self._cache[symbol][1]
            return []

    def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news — tries Alpaca first, falls back to NewsAPI."""
        articles = self.fetch_news_alpaca(symbol)
        if not articles:
            articles = self.fetch_news_newsapi(symbol)
        return articles

    # --- Full Analysis Pipeline ---

    def analyze(self, symbol: str, articles: List[Dict] = None) -> SentimentSignals:
        """Analyze news sentiment for a symbol."""
        if articles is None:
            articles = self.fetch_news(symbol)

        sig = SentimentSignals(symbol=symbol)
        sig.article_count = len(articles)

        if not articles:
            sig.signals.append("No recent news found")
            return sig

        # Determine scoring method
        pipe = _load_finbert()
        use_finbert = pipe is not None
        sig.method = "finbert" if use_finbert else "keyword"

        scored_articles = []
        total_weighted = 0.0
        total_weight = 0.0

        for i, article in enumerate(articles):
            title = article.get("title", "") or ""
            description = article.get("description", "") or ""
            source = article.get("source", "Unknown")
            published = article.get("publishedAt", "")
            url = article.get("url", "")

            # Score using best available method
            text = f"{title}. {description}" if description else title

            if use_finbert:
                combined_score, label = self.score_text_finbert(text)
            else:
                combined_score, label = self.score_text_keywords(text)

            news_item = NewsArticle(
                title=title,
                source=source,
                published_at=published,
                url=url,
                sentiment_score=combined_score,
                sentiment_label=label,
            )
            scored_articles.append(news_item)

            # Recency weighting
            recency_weight = 1.0 / (i + 1) ** 0.3
            total_weighted += combined_score * recency_weight
            total_weight += recency_weight

            if label == "bullish":
                sig.bullish_count += 1
            elif label == "bearish":
                sig.bearish_count += 1
            else:
                sig.neutral_count += 1

        scored_articles.sort(key=lambda a: abs(a.sentiment_score), reverse=True)
        sig.top_articles = scored_articles[:5]

        sig.avg_sentiment = sum(a.sentiment_score for a in scored_articles) / len(scored_articles)
        sig.weighted_sentiment = total_weighted / total_weight if total_weight > 0 else 0.0
        sig.score = max(-1.0, min(1.0, sig.weighted_sentiment))

        # Signal descriptions
        method_tag = f"[{sig.method}]"
        if sig.bullish_count > sig.bearish_count * 2:
            sig.signals.append(f"{method_tag} Strongly bullish ({sig.bullish_count}B/{sig.bearish_count}b)")
        elif sig.bearish_count > sig.bullish_count * 2:
            sig.signals.append(f"{method_tag} Strongly bearish ({sig.bearish_count}b/{sig.bullish_count}B)")
        elif sig.bullish_count > sig.bearish_count:
            sig.signals.append(f"{method_tag} Moderately bullish ({sig.bullish_count}B/{sig.bearish_count}b)")
        elif sig.bearish_count > sig.bullish_count:
            sig.signals.append(f"{method_tag} Moderately bearish ({sig.bearish_count}b/{sig.bullish_count}B)")
        else:
            sig.signals.append(f"{method_tag} Mixed/neutral sentiment")

        return sig