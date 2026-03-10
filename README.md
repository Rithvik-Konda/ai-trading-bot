# AI Trading Bot — Autopilot Mode

An automated trading system that combines **technical indicators**, **volume analysis**, and **news sentiment** to generate and execute trades on autopilot via Alpaca paper trading.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MAIN ENGINE (bot.py)               │
│                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│   │  Technical    │  │   Volume     │  │  News    │ │
│   │  Analysis     │  │   Analysis   │  │ Sentiment│ │
│   │              │  │              │  │          │ │
│   │ • SMA/EMA    │  │ • OBV        │  │ • Keyword│ │
│   │ • MACD       │  │ • MFI        │  │   Scoring│ │
│   │ • RSI        │  │ • Vol Spikes │  │ • Recency│ │
│   │ • Bollinger  │  │ • A/D Line   │  │   Weight │ │
│   │ • ATR        │  │ • VPT        │  │ • NewsAPI│ │
│   │ • VWAP       │  │              │  │          │ │
│   └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │
│          │                 │               │        │
│          └────────┬────────┴───────────────┘        │
│                   ▼                                  │
│       ┌───────────────────────┐                      │
│       │  Composite Signal     │                      │
│       │  Score: [-1, +1]      │                      │
│       │  45% Tech + 30% Vol   │                      │
│       │  + 25% Sentiment      │                      │
│       └───────────┬───────────┘                      │
│                   ▼                                  │
│       ┌───────────────────────┐                      │
│       │  Risk Manager         │                      │
│       │  • Position sizing    │                      │
│       │  • Stop loss / TP     │                      │
│       │  • Exposure limits    │                      │
│       │  • Circuit breakers   │                      │
│       └───────────┬───────────┘                      │
│                   ▼                                  │
│       ┌───────────────────────┐                      │
│       │  Alpaca Broker        │                      │
│       │  (Paper Trading)      │                      │
│       └───────────────────────┘                      │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config.py`:

```python
ALPACA_API_KEY = "your-key-here"
ALPACA_SECRET_KEY = "your-secret-here"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading

NEWS_API_KEY = "your-newsapi-key"  # From newsapi.org
```

### 3. Run the Bot

```bash
# Scan only (no trades executed)
python bot.py --scan-only

# Full autopilot (paper trading)
python bot.py
```

---

## Signal Scoring

Each module produces a score from **-1** (strongly bearish) to **+1** (strongly bullish).

| Module     | Weight | What it measures                        |
|------------|--------|-----------------------------------------|
| Technical  | 45%    | SMA/EMA crossovers, MACD, RSI, BB, VWAP |
| Volume     | 30%    | OBV, MFI, volume spikes, A/D line       |
| Sentiment  | 25%    | News keyword scoring, recency-weighted   |

**Composite score** = weighted average → BUY if > 0.60, SELL if < -0.40.

---

## Risk Management

| Rule                   | Default    | Description                          |
|------------------------|------------|--------------------------------------|
| Max position size      | 5%         | Max % of portfolio per trade         |
| Max total exposure     | 80%        | Max % of portfolio invested          |
| Stop loss              | 3%         | Trailing stop loss per position      |
| Take profit            | 8%         | Auto-close at profit target          |
| Max daily loss         | 2%         | Circuit breaker — halts trading      |
| Max trades/day         | 20         | Prevents overtrading                 |

---

## Files

| File                    | Purpose                                    |
|-------------------------|--------------------------------------------|
| `bot.py`                | Main engine & autopilot loop               |
| `config.py`             | All tunable parameters                     |
| `technical_analysis.py` | SMA, EMA, MACD, RSI, BB, ATR, VWAP        |
| `volume_analysis.py`    | OBV, MFI, A/D line, volume spikes          |
| `news_sentiment.py`     | News fetching & keyword sentiment scoring  |
| `risk_manager.py`       | Position sizing, stops, circuit breakers   |
| `broker.py`             | Alpaca API interface                       |

---

## Important Notes

- **This uses PAPER TRADING by default** — no real money is at risk
- To switch to live trading, change `ALPACA_BASE_URL` to `https://api.alpaca.markets` (proceed with extreme caution)
- The news sentiment module uses a keyword-based approach; for production, consider upgrading to FinBERT
- Always test thoroughly in paper mode before considering live trading
- Past performance does not guarantee future results

---

## Customization

Adjust signal weights, thresholds, indicators, and risk parameters in `config.py`. The modular architecture lets you swap or add analysis modules easily.
