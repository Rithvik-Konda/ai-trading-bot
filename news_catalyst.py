"""
News Catalyst Engine v1.0
==========================
Monitors Alpaca news feed every 30 seconds.
When a strong bullish/bearish headline drops, enters within 60 seconds
before the move fully prices in.

This is a real retail edge: institutions move slowly (compliance, committees).
A fast algo reading FinBERT scores can enter before the bulk of the move.

Strategy:
  - Poll Alpaca news every 30s for all watched symbols
  - Score each NEW headline with FinBERT (or keyword fallback)
  - If score > CATALYST_THRESHOLD: enter immediately, set tight stop
  - Catalyst trades use tighter stops (0.4%) and faster exits (15 min max)
  - Tracks which articles have already been acted on (by URL hash)

Used by: scalper.py imports this and calls check_catalysts() each scan
"""

import logging
import time
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import config

logger = logging.getLogger("news_catalyst")

# Catalyst config
CATALYST_CONFIG = {
    "poll_interval_sec":      30,      # Check news every 30s
    "entry_window_sec":       90,      # Only act on news < 90s old
    "strong_bull_threshold":  0.55,    # FinBERT positive score to trigger long
    "strong_bear_threshold":  0.55,    # FinBERT negative score to trigger short
    "take_profit_pct":        0.008,   # 0.8% TP — news moves are bigger
    "stop_loss_pct":          0.004,   # 0.4% SL — tight, news fades fast
    "max_hold_minutes":       20,      # Exit after 20 min regardless
    "cooldown_minutes":       30,      # 30 min cooldown per symbol after catalyst trade
    "max_catalyst_positions": 3,       # Max 3 catalyst trades open at once
}


@dataclass
class CatalystSignal:
    symbol: str
    direction: str          # "long" or "short"
    score: float            # 0.0 - 1.0
    headline: str
    source: str
    published_at: datetime
    article_id: str         # hash of URL for deduplication


class NewsCatalystEngine:
    """
    Watches news in real-time and generates catalyst trade signals.
    Integrated into scalper.py — called every scan cycle.
    """

    def __init__(self, broker=None):
        self.broker = broker
        self._seen_articles: Set[str] = set()   # Never re-act on same article
        self._symbol_cooldowns: Dict[str, datetime] = {}
        self._finbert = None
        self._finbert_loaded = False
        self._alpaca_api = None
        self._init_apis()

    def _init_apis(self):
        try:
            import alpaca_trade_api as tradeapi
            self._alpaca_api = tradeapi.REST(
                key_id=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                base_url=config.ALPACA_BASE_URL,
                api_version="v2",
            )
            logger.info("News catalyst: Alpaca news API ready")
        except Exception as e:
            logger.warning(f"News catalyst: Alpaca init failed: {e}")

    def _load_finbert(self):
        if self._finbert_loaded:
            return self._finbert
        self._finbert_loaded = True
        try:
            from transformers import pipeline
            self._finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,
                top_k=None,
            )
            logger.info("News catalyst: FinBERT loaded")
        except Exception as e:
            logger.warning(f"News catalyst: FinBERT unavailable ({e}) — using keyword scoring")
        return self._finbert

    def _score_with_finbert(self, text: str) -> float:
        """Returns score: +1.0 = very bullish, -1.0 = very bearish."""
        model = self._load_finbert()
        if model is None:
            return self._keyword_score(text)
        try:
            results = model(text[:512])
            scores = {r["label"].lower(): r["score"] for r in results[0]}
            positive = scores.get("positive", 0)
            negative = scores.get("negative", 0)
            neutral = scores.get("neutral", 0)
            # Weighted: positive - negative, amplified when neutral is low
            raw = positive - negative
            conviction = 1 - neutral
            return raw * (0.5 + 0.5 * conviction)
        except Exception as e:
            logger.debug(f"FinBERT scoring error: {e}")
            return self._keyword_score(text)

    def _keyword_score(self, text: str) -> float:
        """Fast keyword fallback when FinBERT unavailable."""
        text_lower = text.lower()
        BULLISH = {
            "surge": 2.0, "soar": 2.0, "beat": 1.5, "exceed": 1.5,
            "upgrade": 1.8, "outperform": 1.5, "breakout": 1.8,
            "record": 1.5, "strong": 1.2, "bullish": 1.5, "rally": 1.5,
            "blowout": 2.0, "contract": 1.3, "win": 1.3, "partnership": 1.2,
            "raised guidance": 2.0, "raised outlook": 2.0, "buyback": 1.5,
        }
        BEARISH = {
            "miss": -1.5, "downgrade": -1.8, "underperform": -1.5,
            "crash": -2.0, "plunge": -2.0, "layoff": -1.5, "cut": -1.2,
            "loss": -1.2, "bearish": -1.5, "warning": -1.2, "recall": -1.5,
            "investigation": -1.5, "lawsuit": -1.3, "tariff": -1.0,
            "lowered guidance": -2.0, "lowered outlook": -2.0,
        }
        score = 0.0
        count = 0
        for word, weight in {**BULLISH, **BEARISH}.items():
            if word in text_lower:
                score += weight
                count += 1
        if count == 0:
            return 0.0
        return max(-1.0, min(1.0, score / (count * 2)))

    def _article_id(self, url: str, headline: str) -> str:
        return hashlib.md5(f"{url}{headline}".encode()).hexdigest()[:12]

    def is_on_cooldown(self, symbol: str) -> bool:
        return symbol in self._symbol_cooldowns and \
               datetime.now() < self._symbol_cooldowns[symbol]

    def set_cooldown(self, symbol: str):
        self._symbol_cooldowns[symbol] = datetime.now() + \
            timedelta(minutes=CATALYST_CONFIG["cooldown_minutes"])

    def fetch_recent_news(self, symbols: List[str]) -> List[dict]:
        """Fetch news from last 3 minutes for all symbols."""
        if self._alpaca_api is None:
            return []

        all_articles = []
        cutoff = datetime.utcnow() - timedelta(seconds=CATALYST_CONFIG["entry_window_sec"])

        # Batch fetch — Alpaca supports multi-symbol news
        try:
            # Fetch news for up to 10 symbols at once
            for sym in symbols[:15]:
                try:
                    news = self._alpaca_api.get_news(symbol=sym, limit=5)
                    for item in news:
                        try:
                            pub = item.created_at
                            if hasattr(pub, 'replace'):
                                pub_naive = pub.replace(tzinfo=None) if pub.tzinfo else pub
                            else:
                                from datetime import timezone
                                pub_naive = pub.astimezone(timezone.utc).replace(tzinfo=None)

                            if pub_naive < cutoff:
                                continue

                            all_articles.append({
                                "symbol": sym,
                                "headline": item.headline or "",
                                "summary": item.summary or "",
                                "source": item.source or "Unknown",
                                "published_at": pub_naive,
                                "url": item.url or "",
                            })
                        except Exception:
                            continue
                    time.sleep(0.15)
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"News fetch error: {e}")

        return all_articles

    def scan_for_catalysts(
        self,
        symbols: List[str],
        swing_longs: set,
        existing_positions: set,
        regime: str = "MIXED",
    ) -> List[CatalystSignal]:
        """
        Main entry point. Call this every scan cycle.
        Returns list of CatalystSignal to act on immediately.
        """
        signals = []

        try:
            articles = self.fetch_recent_news(symbols)
        except Exception as e:
            logger.debug(f"Catalyst scan error: {e}")
            return []

        for article in articles:
            sym = article["symbol"]
            art_id = self._article_id(article["url"], article["headline"])

            # Skip if already acted on
            if art_id in self._seen_articles:
                continue
            self._seen_articles.add(art_id)

            # Skip if on cooldown or already in position
            if self.is_on_cooldown(sym) or sym in existing_positions:
                continue

            # Score the headline
            text = f"{article['headline']} {article['summary']}"
            score = self._score_with_finbert(text)

            direction = None
            abs_score = abs(score)

            if score > CATALYST_CONFIG["strong_bull_threshold"]:
                if regime != "BEAR":   # Don't go long in bear regime
                    direction = "long"
            elif score < -CATALYST_CONFIG["strong_bear_threshold"]:
                if sym not in swing_longs:   # Don't short swing longs
                    if regime != "BULL":     # Dampen shorts in bull regime (still allow)
                        direction = "short"
                    elif abs_score > 0.75:   # Only very strong bear signals override bull regime
                        direction = "short"

            if direction is None:
                continue

            # Fresh and strong enough — fire
            age_sec = (datetime.utcnow() - article["published_at"]).total_seconds()
            logger.info(
                f"  📰 CATALYST {sym} {direction.upper()} | "
                f"Score: {score:+.2f} | Age: {age_sec:.0f}s | "
                f"\"{article['headline'][:60]}...\""
            )

            signals.append(CatalystSignal(
                symbol=sym,
                direction=direction,
                score=abs_score,
                headline=article["headline"],
                source=article["source"],
                published_at=article["published_at"],
                article_id=art_id,
            ))

        # Clean up seen articles older than 10 minutes to prevent memory growth
        if len(self._seen_articles) > 500:
            self._seen_articles = set(list(self._seen_articles)[-200:])

        return signals
