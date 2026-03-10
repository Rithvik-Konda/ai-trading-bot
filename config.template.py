# config.template.py — Copy this to config.py and fill in your keys
# NEVER commit config.py — it's in .gitignore

# ── Alpaca ────────────────────────────────────────────────────
ALPACA_API_KEY    = "YOUR_ALPACA_KEY_HERE"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_HERE"
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"  # paper trading
# ALPACA_BASE_URL = "https://api.alpaca.markets"        # live trading

# ── Anthropic (Claude LLM gate) ───────────────────────────────
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_KEY_HERE"

# ── Optional — improves data pipeline ─────────────────────────
ALPHA_VANTAGE_KEY = ""   # Free at alphavantage.co — earnings surprise data
POLYGON_API_KEY   = ""   # Free tier at polygon.io — options flow

# ── Strategy Parameters (optimizer-proven) ────────────────────
WATCHLIST = ["NVDA", "AVGO", "AMD", "MU", "VST", "CEG", "GOOGL", "MSFT", "PLTR", "CRWD"]

SWING_STOP_LOSS_PCT      = 0.02
SWING_MAX_HOLD_DAYS      = 5
SWING_FLAT_OVERNIGHT     = False
MAX_POSITION_SIZE_PCT    = 0.20
BULL_MAX_POSITION_SIZE_PCT = 0.20
POSITION_SIZE_MODE       = "conviction"
POSITION_SIZE_BASE       = 0.20

WEIGHTS = {
    "technical":  0.50,
    "volume":     0.30,
    "sentiment":  0.20,
}
