"""Agent 1 — Market Structure: swings, BOS/CHoCH, support/resistance."""
import pandas as pd
from .base import Vote


def _swings(df, left=2, right=2):
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    for i in range(left, len(df) - right):
        wh, wl = h[i-left:i+right+1], l[i-left:i+right+1]
        if h[i] == wh.max() and wh.argmax() == left:
            highs.append((i, h[i]))
        if l[i] == wl.min() and wl.argmin() == left:
            lows.append((i, l[i]))
    return highs, lows


def analyze(df, ctx=None):
    if len(df) < 30:
        return Vote("Market Structure", "HOLD", 50, "not enough candles")
    highs, lows = _swings(df)
    if len(highs) < 2 or len(lows) < 2:
        return Vote("Market Structure", "HOLD", 52, "structure unclear")
    hh = [p for _, p in highs[-2:]]
    ll = [p for _, p in lows[-2:]]
    price = df["close"].iloc[-1]
    higher_high, higher_low = hh[-1] > hh[-2], ll[-1] > ll[-2]
    lower_high, lower_low = hh[-1] < hh[-2], ll[-1] < ll[-2]
    res, sup = max(hh), min(ll)
    if higher_high and higher_low:
        sig, conf, why = "BUY", 74, "BOS up (HH+HL)"
    elif lower_high and lower_low:
        sig, conf, why = "SELL", 74, "BOS down (LH+LL)"
    elif higher_high and lower_low:
        sig, conf, why = "HOLD", 55, "expanding range (CHoCH risk)"
    else:
        sig, conf, why = "HOLD", 57, "range / inside structure"
    span = max(res - sup, 1e-9)
    pos = (price - sup) / span
    if sig == "BUY" and pos < 0.5:
        conf += 6
    if sig == "SELL" and pos > 0.5:
        conf += 6
    return Vote("Market Structure", sig, int(min(96, conf)), why,
                {"support": round(float(sup), 5), "resistance": round(float(res), 5)})
