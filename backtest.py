"""backtest.py — walk-forward paper backtest over real history.

Now uses engine.precompute + engine.simulate (same core the optimizer uses) and
supports a start/end date range, so you can replay any window of the last ~2
years. No future data leak: each candle only sees data up to itself. Paper only.
"""
from data_loader import load_aligned
import engine
import master_brain as mb


def _cap_days(days, interval):
    """Yahoo caps intraday history: 15m/30m ~60d, 1h ~730d."""
    if interval in ("5m", "15m", "30m"):
        return min(days, 59)
    return days


def available_range(symbol="EURUSD=X", days=730, interval="1h"):
    days = _cap_days(days, interval)
    df, _, _ = load_aligned(symbol, days, interval)
    return {"symbol": symbol, "candles": len(df), "interval": interval,
            "start": str(df.index[0])[:10], "end": str(df.index[-1])[:10]}


def _pos(df, date, default):
    if not date:
        return default
    try:
        mask = df.index >= date
        if mask.any():
            return int(mask.argmax())
    except Exception:
        pass
    return default


def run(symbol="EURUSD=X", start=None, end=None, days=730, interval="1h",
        params=None, pip=0.0001, warmup=210, lookback=250):
    days = _cap_days(days, interval)
    df, dxy, gbp = load_aligned(symbol, days, interval)
    n = len(df)
    i0 = max(_pos(df, start, warmup), warmup)
    # include the WHOLE end day (compare against its last minute)
    i1 = _pos(df, f"{end} 23:59", n) if end else n
    if i1 <= i0:
        i1 = n

    pre = engine.precompute(df, dxy, gbp, symbol, i0=i0, i1=i1,
                            lookback=lookback, warmup=warmup)
    params = params or mb.load_config(interval)
    res = engine.simulate(pre, params, pip)

    res.update({
        "symbol": symbol, "interval": interval,
        "candles": i1 - i0,
        "from": str(df.index[i0])[:16], "to": str(df.index[i1 - 1])[:16],
        "used_config": params is not mb.DEFAULT_PARAMS,
    })
    return res


if __name__ == "__main__":
    r = run()
    print("\n===== BACKTEST RESULT =====")
    print(f"Symbol   : {r['symbol']}  ({r['from']} -> {r['to']})")
    print(f"Candles  : {r['candles']}")
    print(f"Trades   : {r['trades']}   Win rate: {r['win_rate']}%")
    print(f"Net pips : {r['net_pips']}")
    for t in r["log"][-5:]:
        print(f"  {t['ts']}  {t['dir']:<4} entry {t['entry']}  "
              f"pnl {t['pnl']:+.0f}p  cons {t['consensus']}")
