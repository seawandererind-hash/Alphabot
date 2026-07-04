"""Agent 5 — Trend Following (50/200 EMA bias, enter on pullback)."""
from .base import Vote, ema


def analyze(df, ctx=None):
    if len(df) < 210:
        return Vote("Trend Following", "HOLD", 52, "need 200+ candles")
    e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
    price = df["close"].iloc[-1]
    if e50.iloc[-1] > e200.iloc[-1]:
        pull = price <= e50.iloc[-1] * 1.001
        return Vote("Trend Following", "BUY", 76 if pull else 66,
                    "uptrend" + (" + pullback" if pull else ""))
    if e50.iloc[-1] < e200.iloc[-1]:
        pull = price >= e50.iloc[-1] * 0.999
        return Vote("Trend Following", "SELL", 76 if pull else 66,
                    "downtrend" + (" + pullback" if pull else ""))
    return Vote("Trend Following", "HOLD", 54, "no clear trend")
