"""Agent 7 — Statistical Arbitrage: EUR/USD vs GBP/USD spread z-score."""
import numpy as np
from .base import Vote


def analyze(df, ctx=None):
    ctx = ctx or {}
    other = ctx.get("gbp")
    if other is None or len(other) < 60:
        return Vote("Stat Arbitrage", "HOLD", 53, "pair data not loaded")
    n = min(len(df), len(other), 120)
    a = df["close"].iloc[-n:].reset_index(drop=True)
    b = other["close"].iloc[-n:].reset_index(drop=True)
    spread = (a / a.iloc[0]) - (b / b.iloc[0])
    z = (spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-12)
    if z > 1.5:
        return Vote("Stat Arbitrage", "SELL", 70, f"EUR rich vs GBP (z={z:.1f})")
    if z < -1.5:
        return Vote("Stat Arbitrage", "BUY", 70, f"EUR cheap vs GBP (z={z:.1f})")
    return Vote("Stat Arbitrage", "HOLD", 55, f"in line (z={z:.1f})")
