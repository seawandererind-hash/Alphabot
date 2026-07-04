"""Agent 2 — Session Timer: kill-zone windows (London/NY overlap = best)."""
from .base import Vote, ema


def _ist_hour(df):
    ts = df.index[-1]
    try:
        ts = ts.tz_convert("Asia/Kolkata")
    except Exception:
        try:
            ts = ts.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            pass
    return ts.hour + ts.minute / 60


def analyze(df, ctx=None):
    if len(df) < 25:
        return Vote("Session Timer", "HOLD", 50, "warming up")
    h = _ist_hour(df)
    killzone = 13 <= h < 23
    overlap = 18.5 <= h < 21.5
    mom = df["close"].iloc[-1] - ema(df["close"], 20).iloc[-1]
    lean = "BUY" if mom > 0 else "SELL"
    if overlap:
        return Vote("Session Timer", lean, 80, f"London-NY overlap (IST {h:.1f})")
    if killzone:
        return Vote("Session Timer", lean, 66, f"active session (IST {h:.1f})")
    return Vote("Session Timer", "HOLD", 54, f"quiet / off-hours (IST {h:.1f})")
