"""
LLM Trade Reasoning Gate — v4.0 (Claude)
=========================================
Uses Claude claude-haiku-4-5-20251001 as a final sanity check before trades.
Haiku is fast (~1s), cheap (~$0.25/1M tokens), and more reliable than Gemini free tier.

At ~5 trades/day, cost is negligible (fractions of a cent per decision).

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in config.py or as environment variable.

If the API is unavailable, the gate defaults to APPROVE (passthrough mode).
"""

import os
import json
import logging
import time as _time
from dataclasses import dataclass
import config

logger = logging.getLogger(__name__)

_client = None
_client_loaded = False


def _load_client():
    """Lazy-load Anthropic client."""
    global _client, _client_loaded
    if _client_loaded:
        return _client
    _client_loaded = True

    api_key = getattr(config, 'ANTHROPIC_API_KEY', None) or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.info("LLM Gate: No ANTHROPIC_API_KEY — running in passthrough mode")
        return None

    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key)
        logger.info("LLM Gate: Claude connected (claude-haiku-4-5-20251001)")
        return _client
    except ImportError:
        logger.warning("LLM Gate: anthropic package not installed. Run: pip install anthropic")
    except Exception as e:
        logger.warning(f"LLM Gate: Failed to init Claude: {e}")

    return None


@dataclass
class LLMDecision:
    """Result from the LLM reasoning gate."""
    approved: bool = True
    confidence_adjust: float = 0.0
    reasoning: str = ""
    passthrough: bool = False


SYSTEM_PROMPT = """You are a senior quantitative trader reviewing a proposed trade.
You will be given technical analysis, volume analysis, news sentiment, and market context.

Your job is to decide: should this trade be executed?

RESPOND WITH ONLY VALID JSON (no markdown, no backticks, no explanation outside JSON):
{
  "approved": true/false,
  "confidence_adjust": float between -0.3 and +0.3,
  "reasoning": "one sentence explanation"
}

Rules:
- REJECT if technicals and volume disagree strongly (divergence = danger)
- REJECT if entering against the daily trend without very strong conviction (score > 0.40)
- REJECT if news sentiment is strongly negative for a long entry
- REJECT if the stock is already up >5% today (chasing)
- REDUCE confidence (negative adjust) if volume is weak or declining
- INCREASE confidence (positive adjust) if volume confirms price AND news is aligned
- APPROVE with high confidence only when technicals, volume, AND trend all align
- Be conservative: when in doubt, reject. Capital preservation > returns.
- Avoid entries in last 30 min before close unless score > 0.40
- Watch for overbought RSI (>70) on buys"""


class LLMTradeGate:
    """
    Passes trade context to Claude for a go/no-go decision.
    Falls back to passthrough if API unavailable.
    """

    def __init__(self):
        self.enabled = getattr(config, 'LLM_GATE_ENABLED', True)
        self._call_times = []  # Track call timestamps for rate limiting
        self._max_calls_per_min = 30  # Claude Haiku is generous — no real concern

    def _rate_limit_ok(self) -> bool:
        now = _time.time()
        # Keep only calls from last 60 seconds
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self._max_calls_per_min:
            return False
        self._call_times.append(now)
        return True

    def evaluate_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        qty: int,
        composite_score: float,
        tech_score: float,
        tech_signals: list,
        vol_score: float,
        vol_signals: list,
        sent_score: float,
        sent_signals: list,
        trend_bullish: bool,
        portfolio_value: float,
        open_positions: int,
        exposure_pct: float,
    ) -> LLMDecision:
        """Ask Claude whether this trade should be executed."""
        if not self.enabled:
            return LLMDecision(approved=True, passthrough=True, reasoning="LLM gate disabled")

        client = _load_client()
        if client is None:
            return LLMDecision(approved=True, passthrough=True, reasoning="No API key")

        if not self._rate_limit_ok():
            return LLMDecision(approved=True, passthrough=True, reasoning="Rate limited")

        trade_value = price * qty
        position_pct = (trade_value / portfolio_value * 100) if portfolio_value > 0 else 0

        from datetime import datetime
        now = datetime.now()
        near_close = now.hour >= 15 and now.minute >= 30

        prompt = f"""PROPOSED TRADE:
Symbol: {symbol}
Action: {action}
Price: ${price:.2f}
Quantity: {qty}
Trade Value: ${trade_value:,.2f} ({position_pct:.1f}% of portfolio)
Composite Score: {composite_score:+.3f}

TECHNICAL ANALYSIS (score: {tech_score:+.3f}):
{chr(10).join('- ' + s for s in tech_signals[:5])}

VOLUME ANALYSIS (score: {vol_score:+.3f}):
{chr(10).join('- ' + s for s in vol_signals[:5])}

NEWS SENTIMENT (score: {sent_score:+.3f}):
{chr(10).join('- ' + s for s in sent_signals[:3])}

MARKET CONTEXT:
Daily Trend: {"BULLISH (50MA > 200MA)" if trend_bullish else "BEARISH (50MA < 200MA)"}
Portfolio: ${portfolio_value:,.2f}
Open Positions: {open_positions}
Current Exposure: {exposure_pct:.1f}%
Time: {now.strftime('%H:%M ET')} {"⚠️ NEAR CLOSE" if near_close else "(market open)"}

Respond with ONLY valid JSON:"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                temperature=0.1,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            data = json.loads(text)

            decision = LLMDecision(
                approved=data.get("approved", True),
                confidence_adjust=max(-0.3, min(0.3, data.get("confidence_adjust", 0.0))),
                reasoning=data.get("reasoning", ""),
                passthrough=False,
            )

            status = "✅ APPROVED" if decision.approved else "❌ REJECTED"
            logger.info(f"LLM Gate [{symbol}]: {status} | adj={decision.confidence_adjust:+.2f} | {decision.reasoning}")
            return decision

        except json.JSONDecodeError as e:
            logger.warning(f"LLM Gate: JSON parse error: {e}")
            return LLMDecision(approved=True, passthrough=True, reasoning="Parse error — passthrough")
        except Exception as e:
            logger.warning(f"LLM Gate: API error: {e}")
            return LLMDecision(approved=True, passthrough=True, reasoning=f"API error — passthrough")