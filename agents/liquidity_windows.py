"""Agent — Liquidity Windows (replaces the old Session Timer).

Trades the two windows where institutional flow is most predictable:
- NY open drive (12:30-14:30 UTC): follow the momentum that develops as US
  desks come in — continuation of the young move.
- Post-London-fix fade (16:00-17:00 UTC): benchmark-fix flows around 4pm
  London often overshoot; fade the move made into the fix.

Abstains everywhere else — no more voting in dead hours (the old agent's
biggest leak, -2162 pips solo over 2y).
"""
from .base import Vote


def _utc_hour(df):
    ts = df.index[-1]
    try:
        ts = ts.tz_convert("UTC")
    except Exception:
        try:
            ts = ts.tz_localize("UTC")
        except Exception:
            pass
    return ts.hour + ts.minute / 60.0


def analyze(df, ctx=None):
    if len(df) < 8:
        return Vote("Liquidity Windows", "HOLD", 50, "warming up")
    h = _utc_hour(df)
    c = df["close"]

    if 12.5 <= h < 14.5:                     # NY open drive: ride the push
        mom = float(c.iloc[-1]) - float(c.iloc[-4])
        if mom > 0:
            return Vote("Liquidity Windows", "BUY", 70, "NY open drive up")
        if mom < 0:
            return Vote("Liquidity Windows", "SELL", 70, "NY open drive down")
        return Vote("Liquidity Windows", "HOLD", 55, "NY open, no drive")

    if 16.0 <= h < 17.0:                     # post-fix: fade the fix move
        move = float(c.iloc[-1]) - float(c.iloc[-3])
        if move > 0:
            return Vote("Liquidity Windows", "SELL", 66, "fading post-fix push up")
        if move < 0:
            return Vote("Liquidity Windows", "BUY", 66, "fading post-fix push down")
        return Vote("Liquidity Windows", "HOLD", 55, "fix window, flat")

    return Vote("Liquidity Windows", "HOLD", 54, "outside liquidity windows")
