"""
data_loader.py — AlphaBot market data
Fetches OHLC candles via Yahoo Finance (free, no key) on your machine.
Falls back to per-symbol synthetic candles if Yahoo isn't reachable,
so the whole bot still runs and multi-symbol agents stay meaningful.
"""
import os
import time
import pickle
import pandas as pd
import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

_START = {"EURUSD=X": 1.0850, "GBPUSD=X": 1.2650, "JPY=X": 156.20,
          "GC=F": 2350.0, "DX-Y.NYB": 104.3}


def get_candles(symbol="EURUSD=X", period="60d", interval="1h"):
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError("empty download")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        print(f"[data] {symbol}: {len(df)} real candles (Yahoo)")
        return df
    except Exception as e:
        print(f"[data] {symbol}: Yahoo unavailable ({e}); synthetic")
        return _synthetic(symbol, interval)


def _synthetic(symbol="EURUSD=X", interval="1h", n=700):
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    price = _START.get(symbol, 1.0)
    scale = price * 0.0006
    rows, t = [], pd.Timestamp("2026-01-01", tz="UTC")
    step = pd.Timedelta(hours=1)
    trend = 0.0
    for i in range(n):
        if i % 55 == 0:
            trend = rng.uniform(-0.1, 0.1) * scale
        o = price
        c = o + trend + rng.normal(0, scale)
        hi = max(o, c) + abs(rng.normal(0, scale * 0.6))
        lo = min(o, c) - abs(rng.normal(0, scale * 0.6))
        rows.append((t, o, hi, lo, c, int(abs(rng.normal(1500, 400)))))
        price = c
        t += step
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")


# --- long history with disk cache (for 2-year backtests / learning) ----------
def _safe(symbol):
    return symbol.replace("=", "_").replace("^", "").replace("/", "_")


def get_history(symbol="EURUSD=X", days=730, interval="1h", max_age_h=12):
    """Up to ~2 years of candles, cached to disk so we download once.
    Yahoo caps 1h data at ~730 days. Re-downloads only if cache is stale."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{_safe(symbol)}_{interval}_{days}.pkl")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_h * 3600:
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    try:
        import yfinance as yf
        df = yf.download(symbol, period=f"{days}d", interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError("empty download")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        print(f"[history] {symbol}: {len(df)} real candles ({interval}, {days}d) -> cached")
        with open(path, "wb") as fh:
            pickle.dump(df, fh)
        return df
    except Exception as e:
        print(f"[history] {symbol}: Yahoo unavailable ({e}); synthetic")
        return _synthetic(symbol, interval, n=min(days * 24, 12000))


def _align(base, other):
    """Reindex `other` onto base's timestamps (forward-fill last known value)
    so positional slicing across symbols stays time-correct."""
    other = other[~other.index.duplicated(keep="last")].sort_index()
    return other.reindex(base.index, method="ffill").bfill()


def load_aligned(symbol="EURUSD=X", days=730, interval="1h"):
    """Return (df, dxy, gbp) all sharing the main symbol's timestamp index."""
    df = get_history(symbol, days, interval)
    dxy = _align(df, get_history("DX-Y.NYB", days, interval))
    gbp = _align(df, get_history("GBPUSD=X", days, interval))
    return df, dxy, gbp


if __name__ == "__main__":
    print(get_candles().tail())


# --- fast price (for managing open trades every few seconds) -----------------
_TD_MAP = {"EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY",
           "GC=F": "XAU/USD", "DX-Y.NYB": "DXY"}


def get_price(symbol="EURUSD=X"):
    """Latest price for fast SL/TP checks.
    Uses Twelve Data streaming-grade REST if TWELVEDATA_KEY is set (real-time,
    works in India), else yfinance quick price (delayed), else None."""
    import os
    key = os.environ.get("TWELVEDATA_KEY")
    if key:
        try:
            import urllib.request, json
            sym = _TD_MAP.get(symbol, symbol)
            url = f"https://api.twelvedata.com/price?symbol={sym}&apikey={key}"
            with urllib.request.urlopen(url, timeout=5) as r:
                return float(json.load(r)["price"])
        except Exception as e:
            print(f"[price] twelvedata failed ({e})")
    try:
        import yfinance as yf
        return float(yf.Ticker(symbol).fast_info["last_price"])
    except Exception:
        return None
