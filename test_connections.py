"""
test_connections.py — Verify all bot components are working
============================================================
Run this before market open to confirm everything is connected.

Usage:
    python3 test_connections.py
"""

import sys
import os

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"

results = []

def test(name, fn):
    try:
        msg = fn()
        print(f"  {PASS} {name}: {msg}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False))

print("\n" + "═"*60)
print("  CONNECTION TEST — AI Trading Bot v4.0")
print("═"*60 + "\n")

# ── 1. Config ────────────────────────────────────────────────
print("  [ Config ]")

def check_config():
    import config
    keys = ["ALPACA_API_KEY", "ANTHROPIC_API_KEY", "WATCHLIST"]
    missing = [k for k in keys if not getattr(config, k, None)]
    if missing:
        raise Exception(f"Missing: {', '.join(missing)}")
    return f"{len(config.WATCHLIST)} symbols, threshold={config.BULL_BUY_THRESHOLD}"

test("Config loaded", check_config)

# ── 2. Alpaca ────────────────────────────────────────────────
print("\n  [ Alpaca Broker ]")

def check_alpaca_account():
    from broker import AlpacaBroker
    b = AlpacaBroker()
    acct = b.get_account()
    if "error" in acct:
        raise Exception(acct["error"])
    return f"Equity=${acct['equity']:,.2f} | Cash=${acct['cash']:,.2f}"

def check_alpaca_prices():
    from broker import AlpacaBroker
    import config
    b = AlpacaBroker()
    prices = b.get_latest_prices(config.WATCHLIST[:3])
    if not prices:
        raise Exception("No prices returned")
    pairs = ", ".join(f"{s}=${p:.2f}" for s, p in list(prices.items())[:3])
    return pairs

def check_alpaca_positions():
    from broker import AlpacaBroker
    b = AlpacaBroker()
    positions = b.get_positions()
    return f"{len(positions)} open positions"

test("Account connected", check_alpaca_account)
test("Live prices", check_alpaca_prices)
test("Positions", check_alpaca_positions)

# ── 3. Claude LLM Gate ──────────────────────────────────────
print("\n  [ Claude LLM Gate ]")

def check_claude():
    import config
    import anthropic
    key = getattr(config, 'ANTHROPIC_API_KEY', None)
    if not key or key == "YOUR_ANTHROPIC_API_KEY_HERE":
        raise Exception("API key not set in config.py")
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with just: CONNECTED"}],
    )
    text = response.content[0].text.strip()
    if "CONNECTED" not in text.upper():
        raise Exception(f"Unexpected response: {text}")
    return f"claude-haiku-4-5-20251001 responding"

test("Claude API", check_claude)

# ── 4. ML Models ─────────────────────────────────────────────
print("\n  [ ML Models ]")

def check_ml_models():
    import glob, joblib
    files = glob.glob("ml_model_*.joblib")
    if not files:
        raise Exception("No model files found — run train_models.py first")
    loaded = []
    rejected = []
    for f in files:
        sym = f.replace("ml_model_", "").replace(".joblib", "").upper()
        bundle = joblib.load(f)
        auc = bundle.get("auc", 0.5)
        acc = bundle.get("accuracy", 0)
        if auc >= 0.55:
            loaded.append(f"{sym}(auc={auc:.2f})")
        else:
            rejected.append(f"{sym}(auc={auc:.2f})")
    if not loaded:
        raise Exception(f"All models rejected (AUC too low): {rejected}")
    msg = f"{len(loaded)} active: {', '.join(loaded)}"
    if rejected:
        msg += f" | {len(rejected)} rejected: {', '.join(rejected)}"
    return msg

test("ML models", check_ml_models)

def check_ml_prediction():
    import glob, joblib
    import numpy as np
    files = glob.glob("ml_model_*.joblib")
    if not files:
        raise Exception("No models found")
    bundle = joblib.load(files[0])
    sym = files[0].replace("ml_model_", "").replace(".joblib", "").upper()
    auc = bundle.get("auc", 0)
    if auc < 0.55:
        raise Exception(f"Best model {sym} has AUC {auc:.2f} — too low")
    # Try a dummy prediction
    model = bundle["model"]
    scaler = bundle["scaler"]
    feat_cols = bundle["features"]
    dummy = np.zeros((1, len(feat_cols)))
    dummy_s = scaler.transform(dummy)
    prob = model.predict_proba(dummy_s)[0][1]
    return f"{sym} prediction working | dummy prob={prob:.3f}"

test("ML prediction", check_ml_prediction)

# ── 5. Technical Analysis ────────────────────────────────────
print("\n  [ Analysis Modules ]")

def check_technical():
    from technical_analysis import TechnicalAnalyzer
    ta = TechnicalAnalyzer()
    return "TechnicalAnalyzer loaded"

def check_volume():
    from volume_analysis import VolumeAnalyzer
    va = VolumeAnalyzer()
    return "VolumeAnalyzer loaded"

def check_sentiment():
    from news_sentiment import NewsSentimentAnalyzer
    ns = NewsSentimentAnalyzer()
    return "NewsSentimentAnalyzer loaded"

test("Technical analysis", check_technical)
test("Volume analysis", check_volume)
test("News sentiment", check_sentiment)

# ── 6. SPY Data ──────────────────────────────────────────────
print("\n  [ Market Data ]")

def check_spy():
    from broker import AlpacaBroker
    b = AlpacaBroker()
    price = b.get_latest_price("SPY")
    if not price or price <= 0:
        raise Exception("No SPY price returned")
    return f"SPY=${price:.2f}"

test("SPY live price", check_spy)

# ── Summary ──────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'═'*60}")
print(f"  RESULTS: {passed}/{total} passed", end="")
if failed == 0:
    print(f" \033[92m— ALL SYSTEMS GO ✅\033[0m")
else:
    print(f" \033[91m— {failed} FAILED ❌\033[0m")
    print(f"\n  Failed checks:")
    for name, ok in results:
        if not ok:
            print(f"    • {name}")
print("═"*60 + "\n")

sys.exit(0 if failed == 0 else 1)
