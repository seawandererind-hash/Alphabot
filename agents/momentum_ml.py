"""Agent 4 — Momentum (placeholder for the ML agent; swap in XGBoost later)."""
from .base import Vote, ema, rsi


def analyze(df, ctx=None):
    if len(df) < 30:
        return Vote("ML/Momentum", "HOLD", 50, "warming up")
    e9, e21 = ema(df["close"], 9), ema(df["close"], 21)
    r = rsi(df["close"]).iloc[-1]
    up = e9.iloc[-1] > e21.iloc[-1]
    gap = abs(e9.iloc[-1] - e21.iloc[-1]) / df["close"].iloc[-1]
    conf = int(min(90, 58 + gap * 4000))
    if up and r < 72:
        return Vote("ML/Momentum", "BUY", conf, f"EMA9>EMA21, RSI {r:.0f}")
    if not up and r > 28:
        return Vote("ML/Momentum", "SELL", conf, f"EMA9<EMA21, RSI {r:.0f}")
    return Vote("ML/Momentum", "HOLD", 55, f"flat, RSI {r:.0f}")
