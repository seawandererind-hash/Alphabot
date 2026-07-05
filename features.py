"""
features.py — one shared feature set for both ML training and the live ML agent.

feature_frame(df)  -> full DataFrame of features (vectorised, for training)
make_features(df)  -> the latest row as a plain dict (for the live agent)

Keeping them in one place guarantees the model is scored on exactly the
features it was trained on.
"""
import numpy as np
import pandas as pd

from agents.base import ema, rsi, atr

FEATURE_COLS = ["ret1", "ret3", "ret6", "ema_gap", "ema_slope",
                "rsi", "atr_pct", "range_pos", "vol_z", "hour_sin", "hour_cos"]


def feature_frame(df):
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change()
    f["ret3"] = c.pct_change(3)
    f["ret6"] = c.pct_change(6)
    e9, e21 = ema(c, 9), ema(c, 21)
    f["ema_gap"] = (e9 - e21) / c
    f["ema_slope"] = e9.diff() / c
    f["rsi"] = rsi(c, 14) / 100.0
    f["atr_pct"] = atr(df, 14) / c
    lo = df["low"].rolling(20).min()
    hi = df["high"].rolling(20).max()
    f["range_pos"] = (c - lo) / (hi - lo + 1e-12)
    v = df["volume"].astype(float)
    f["vol_z"] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-9)
    try:
        hrs = df.index.tz_convert("UTC").hour
    except Exception:
        hrs = df.index.hour
    hrs = np.asarray(hrs, dtype=float)
    f["hour_sin"] = np.sin(2 * np.pi * hrs / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hrs / 24)
    return f[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)


def make_features(df):
    """Latest row as a dict; df should include enough trailing history (>=60)."""
    row = feature_frame(df.tail(80)).iloc[-1]
    return {k: (0.0 if pd.isna(row[k]) else float(row[k])) for k in FEATURE_COLS}
