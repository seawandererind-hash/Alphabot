"""Agents 12-14 — classic strategies used by well-known traders worldwide.

- RSI-2 Reversion (Larry Connors): high-win-rate mean reversion with a 200-EMA
  trend filter. Buy panic dips in uptrends, sell spikes in downtrends.
- London Breakout: institutional classic — trade the break of the Asian-session
  range in the London morning window.
- Bollinger Fade: fade 2.2-sigma stretches back toward the mean.

Each abstains (HOLD) outside its setup so it never distorts the vote.
"""
import numpy as np

from .base import Vote, ema, rsi


def _utc_index(df):
    idx = df.index
    try:
        idx = idx.tz_convert("UTC")
    except Exception:
        try:
            idx = idx.tz_localize("UTC")
        except Exception:
            pass
    return idx


def rsi2_reversion(df, ctx=None):
    if len(df) < 210:
        return Vote("RSI-2 Reversion", "HOLD", 50, "warming up")
    c = df["close"]
    r2 = float(rsi(c, 2).iloc[-1])
    uptrend = float(c.iloc[-1]) > float(ema(c, 200).iloc[-1])
    if uptrend and r2 < 10:
        return Vote("RSI-2 Reversion", "BUY", 72, f"RSI2 {r2:.0f} oversold in uptrend")
    if not uptrend and r2 > 90:
        return Vote("RSI-2 Reversion", "SELL", 72, f"RSI2 {r2:.0f} overbought in downtrend")
    return Vote("RSI-2 Reversion", "HOLD", 55, f"no extreme (RSI2 {r2:.0f})")


def london_breakout(df, ctx=None):
    if len(df) < 30:
        return Vote("London Breakout", "HOLD", 50, "warming up")
    idx = _utc_index(df)
    h = int(idx[-1].hour)
    if not (7 <= h < 12):
        return Vote("London Breakout", "HOLD", 54, "outside London window")
    today = idx[-1].date()
    day = np.asarray(idx.date) == today
    asian = day & (np.asarray(idx.hour) < 7)
    if int(asian.sum()) < 3:
        return Vote("London Breakout", "HOLD", 54, "no Asian range yet")
    hi = float(df["high"].values[asian].max())
    lo = float(df["low"].values[asian].min())
    px = float(df["close"].iloc[-1])
    if px > hi:
        return Vote("London Breakout", "BUY", 74, "broke above Asian range")
    if px < lo:
        return Vote("London Breakout", "SELL", 74, "broke below Asian range")
    return Vote("London Breakout", "HOLD", 55, "inside Asian range")


def bollinger_fade(df, ctx=None):
    if len(df) < 60:
        return Vote("Bollinger Fade", "HOLD", 50, "warming up")
    c = df["close"]
    m = float(c.rolling(20).mean().iloc[-1])
    sd = float(c.rolling(20).std().iloc[-1])
    z = (float(c.iloc[-1]) - m) / (sd + 1e-12)
    if z > 2.2:
        return Vote("Bollinger Fade", "SELL", 68, f"stretched +{z:.1f} sigma")
    if z < -2.2:
        return Vote("Bollinger Fade", "BUY", 68, f"stretched {z:.1f} sigma")
    return Vote("Bollinger Fade", "HOLD", 55, f"inside bands (z {z:.1f})")
