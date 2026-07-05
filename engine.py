"""
engine.py — fast backtest core shared by the interactive backtester and the
optimizer.

precompute(): run every agent once per candle (trailing window) and store their
raw signal/confidence in a matrix. This is the expensive step (~all candles).

simulate(): given a weight vector + thresholds + SL/TP, aggregate the stored
votes into decisions and walk the price path to score trades. This is cheap, so
the optimizer can evaluate thousands of parameter sets over one precompute.
"""
import numpy as np

import master_brain as mb

AGENT_NAMES = [name for name, _ in mb.AGENTS]
_ML_NAME = "Learned ML"


def _fill_ml_column(sig, conf, df, i0, i1, col):
    """Fill the Learned-ML column vectorised (one predict for the whole range)
    instead of running the model per candle — orders of magnitude faster."""
    import os
    import pickle
    from agents.learned_ml import _MODEL_PATH
    if not os.path.exists(_MODEL_PATH):
        return
    try:
        with open(_MODEL_PATH, "rb") as fh:
            b = pickle.load(fh)
        from features import feature_frame, FEATURE_COLS
        f = feature_frame(df).iloc[i0:i1][FEATURE_COLS].fillna(0.0).values
        X = b["scaler"].transform(f) if b.get("scaler") is not None else f
        p_up = b["model"].predict_proba(X)[:, 1]
        buy_th, sell_th = b.get("buy_th", 0.55), b.get("sell_th", 0.45)
        s = np.where(p_up >= buy_th, 1, np.where(p_up <= sell_th, -1, 0)).astype(np.int8)
        c = np.clip((np.abs(p_up - 0.5) * 200).astype(int), 50, 95)
        sig[:, col] = s
        conf[:, col] = np.where(s == 0, 55, c)
    except Exception:
        pass


def precompute(df, dxy, gbp, symbol, i0=None, i1=None, lookback=250, warmup=210,
               progress=None):
    """Vote matrix for candles in [i0, i1). Each agent -> signal (-1/0/+1) and
    confidence, unweighted, so any weight vector can be applied later.
    Rule agents run per-candle (trailing window); the ML agent is vectorised."""
    n = len(df)
    i0 = warmup if i0 is None else max(i0, warmup)
    i1 = n if i1 is None else min(i1, n)
    A = len(mb.AGENTS)
    m = max(0, i1 - i0)

    sig = np.zeros((m, A), dtype=np.int8)
    conf = np.zeros((m, A), dtype=np.int16)
    conf[:] = 55
    close = df["close"].values[i0:i1].astype(float)
    high = df["high"].values[i0:i1].astype(float)
    low = df["low"].values[i0:i1].astype(float)
    ts = [str(t)[:16] for t in df.index[i0:i1]]

    rule = [(a, fn) for a, (name, fn) in enumerate(mb.AGENTS) if name != _ML_NAME]
    for k, i in enumerate(range(i0, i1)):
        lo = max(0, i - lookback)
        w = df.iloc[lo:i + 1]
        ctx = {"symbol": symbol, "dxy": dxy.iloc[lo:i + 1], "gbp": gbp.iloc[lo:i + 1]}
        for a, fn in rule:
            v = fn(w, ctx)
            sig[k, a] = 1 if v.signal == "BUY" else (-1 if v.signal == "SELL" else 0)
            conf[k, a] = v.confidence
        if progress and (k % 500 == 0):
            progress(k, m)

    if _ML_NAME in AGENT_NAMES:
        _fill_ml_column(sig, conf, df, i0, i1, AGENT_NAMES.index(_ML_NAME))
    if progress:
        progress(m, m)
    return {"names": AGENT_NAMES, "sig": sig, "conf": conf,
            "close": close, "high": high, "low": low, "ts": ts}


def sub(pre, a, b):
    """View of a precompute over candle rows [a, b)."""
    out = {"names": pre["names"]}
    for k in ("sig", "conf", "close", "high", "low"):
        out[k] = pre[k][a:b]
    out["ts"] = pre["ts"][a:b]
    return out


def _weight_vector(params):
    return np.array([mb._weight_for(name, params) for name in AGENT_NAMES], dtype=float)


def simulate(pre, params, pip=0.0001):
    """Walk the precomputed votes -> decisions -> paper trades. SL checked before
    TP within a candle (conservative). Returns metrics + trade log + equity."""
    w = _weight_vector(params)
    thr_full = params.get("thr_full", 0.70)
    thr_half = params.get("thr_half", 0.60)
    min_conv = params.get("min_conviction", 4.0)
    sl = params.get("sl", 15)
    tp = params.get("tp", 30)
    spread = params.get("spread", 0.0)   # pips cost per round-trip trade
    # smart exits (0 = off). be_trigger: move stop to entry after +N pips of
    # favorable excursion. trail: keep stop N pips behind the best price seen.
    # Stops only ratchet AFTER a candle survives — no intra-candle look-ahead.
    be_trigger = params.get("be_trigger", 0) or 0
    trail = params.get("trail", 0) or 0

    sig, close, high, low, ts = pre["sig"], pre["close"], pre["high"], pre["low"], pre["ts"]
    buyw = ((sig > 0) * w).sum(axis=1)
    sellw = ((sig < 0) * w).sum(axis=1)

    # optional regime router: weights_r used when the market is RANGING
    # (Kaufman efficiency ratio below regime_thr), main weights when TRENDING.
    if params.get("weights_r"):
        wr_vec = np.array([float(params["weights_r"].get(name,
                          mb._weight_for(name, params))) for name in AGENT_NAMES])
        n_er = int(params.get("regime_n", 48))
        thr_er = float(params.get("regime_thr", 0.3))
        c = pre["close"]
        net = np.abs(c - np.roll(c, n_er))
        step = np.abs(np.diff(c, prepend=c[0]))
        noise = np.convolve(step, np.ones(n_er), mode="full")[:len(c)]
        er = np.divide(net, noise, out=np.zeros_like(net), where=noise > 0)
        er[:n_er] = 0.0                     # warmup -> treat as ranging
        trending = er >= thr_er
        buyw_r = ((sig > 0) * wr_vec).sum(axis=1)
        sellw_r = ((sig < 0) * wr_vec).sum(axis=1)
        buyw = np.where(trending, buyw, buyw_r)
        sellw = np.where(trending, sellw, sellw_r)

    directional = buyw + sellw
    aligned = np.maximum(buyw, sellw)
    consensus = np.divide(aligned, directional, out=np.zeros_like(aligned),
                          where=directional > 0)
    is_buy = buyw > sellw
    is_sell = sellw > buyw

    trades, equity, cum = [], [], 0.0
    wins = 0
    open_t = None
    m = len(close)

    for k in range(m):
        price = close[k]
        if open_t:
            d = 1 if open_t["dir"] == "BUY" else -1
            entry = open_t["entry"]
            # current stop distance (pips, relative to entry; negative = locked profit)
            stop_dist = open_t["stop_dist"]
            adverse = (entry - low[k]) / pip if d == 1 else (high[k] - entry) / pip
            favor = (high[k] - entry) / pip if d == 1 else (entry - low[k]) / pip
            hit = None
            if adverse >= stop_dist:
                hit = -stop_dist                 # exit at the (possibly moved) stop
            elif favor >= tp:
                hit = tp
            if hit is not None:
                pnl = (hit - spread) * open_t["size"]
                cum += pnl
                wins += 1 if pnl > 0 else 0
                t = {kk: v for kk, v in open_t.items() if kk not in ("stop_dist", "best")}
                trades.append({**t, "exit": round(price, 5), "pnl": round(pnl, 1)})
                equity.append(round(cum, 1))
                open_t = None
            else:
                # candle survived -> ratchet the stop for FUTURE candles
                open_t["best"] = max(open_t["best"], favor)
                if be_trigger and open_t["best"] >= be_trigger:
                    open_t["stop_dist"] = min(open_t["stop_dist"], 0.0)
                if trail:
                    open_t["stop_dist"] = min(open_t["stop_dist"],
                                              trail - open_t["best"])
        if open_t:
            continue
        if directional[k] >= min_conv and consensus[k] >= thr_half:
            size = 1.0 if consensus[k] >= thr_full else 0.5
            direction = "BUY" if is_buy[k] else ("SELL" if is_sell[k] else None)
            if direction:
                open_t = {"dir": direction, "entry": round(price, 5), "size": size,
                          "consensus": round(float(consensus[k]), 2), "ts": ts[k],
                          "stop_dist": float(sl), "best": 0.0}

    n = len(trades)
    wr = (wins / n * 100) if n else 0.0
    return {"trades": n, "wins": wins, "win_rate": round(wr, 1),
            "net_pips": round(cum, 1), "log": trades, "equity": equity}
