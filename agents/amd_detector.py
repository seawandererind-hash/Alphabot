"""Agent 3 — AMD: Accumulation / Manipulation / Distribution phase."""
from .base import Vote, atr


def analyze(df, ctx=None):
    if len(df) < 40:
        return Vote("AMD Detector", "HOLD", 50, "warming up")
    a = atr(df, 14)
    cur, avg = a.iloc[-1], a.iloc[-30:].mean()
    recent = df.iloc[-6:]
    body = recent["close"].iloc[-1] - recent["open"].iloc[0]
    rng = recent["high"].max() - recent["low"].min()
    upper_wick = recent["high"].max() - max(recent["close"].iloc[-1], recent["open"].iloc[0])
    lower_wick = min(recent["close"].iloc[-1], recent["open"].iloc[0]) - recent["low"].min()

    if cur < avg * 0.8:
        return Vote("AMD Detector", "HOLD", 60, "Accumulation (low volatility)")
    # manipulation: big wick relative to body = stop hunt, wait
    if max(upper_wick, lower_wick) > abs(body) * 1.5 and rng > avg:
        return Vote("AMD Detector", "HOLD", 62, "Manipulation (stop hunt) — wait")
    # distribution: strong directional expansion
    if cur > avg * 1.1 and abs(body) > rng * 0.5:
        return Vote("AMD Detector", "BUY" if body > 0 else "SELL", 78,
                    "Distribution (trend expansion)")
    return Vote("AMD Detector", "HOLD", 56, "transition")
