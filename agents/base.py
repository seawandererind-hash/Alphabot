"""base.py — shared types + indicator helpers used by all agents."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Vote:
    agent: str
    signal: str              # "BUY" | "SELL" | "HOLD" | "BLOCK"
    confidence: int          # 0-100
    reason: str
    meta: dict = field(default_factory=dict)


# ---- indicators -------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()
