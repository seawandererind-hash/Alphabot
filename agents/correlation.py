"""Agent 6 — Correlation: DXY inverse to EUR/GBP; confirms direction."""
from .base import Vote


def analyze(df, ctx=None):
    ctx = ctx or {}
    dxy = ctx.get("dxy")
    if dxy is None or len(dxy) < 10:
        return Vote("Correlation", "HOLD", 53, "DXY data not loaded")
    chg = dxy["close"].iloc[-1] - dxy["close"].iloc[-10]
    base = ctx.get("symbol", "EURUSD=X")
    usd_quote = base in ("EURUSD=X", "GBPUSD=X", "GC=F")  # USD in denominator -> inverse to DXY
    if usd_quote:
        if chg > 0:
            return Vote("Correlation", "SELL", 68, "DXY rising -> USD strong")
        if chg < 0:
            return Vote("Correlation", "BUY", 68, "DXY falling -> USD weak")
    return Vote("Correlation", "HOLD", 55, "no strong DXY signal")
