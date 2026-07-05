"""
dashboard.py — AlphaBot local web dashboard.

Run:  python dashboard.py
Then open:  http://127.0.0.1:5000

Shows every agent's live vote + the Master Brain decision in a clean UI.
Pulls real Yahoo data (synthetic fallback), same engine as run_bot.py.
No real orders — paper only.
"""
import time
import threading

from flask import Flask, jsonify, render_template, request

from data_loader import get_candles
import master_brain as mb
import backtest as bt
from news_feed import get_news

app = Flask(__name__)

SYMBOLS = ["EURUSD=X", "GBPUSD=X", "JPY=X", "GC=F"]

# --- tiny candle cache so we don't hammer Yahoo on every refresh -------------
_CACHE = {}
_CACHE_TTL = 60          # seconds
_LOCK = threading.Lock()


def _cached_candles(symbol, period="60d", interval="1h"):
    key = (symbol, period, interval)
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]
    df = get_candles(symbol, period=period, interval=interval)
    with _LOCK:
        _CACHE[key] = (now, df)
    return df


def _run_decision(symbol):
    df = _cached_candles(symbol)
    ctx = {
        "symbol": symbol,
        "dxy": _cached_candles("DX-Y.NYB"),
        "gbp": _cached_candles("GBPUSD=X"),
    }
    account = {"day_pnl_pct": 0, "open_trades": 0, "wins": 0, "trades": 0}
    out = mb.decide(df, ctx, account)

    votes = [{
        "agent": v.agent,
        "signal": v.signal,
        "confidence": v.confidence,
        "reason": v.reason,
        "weight": v.meta.get("weight", 1.0),
    } for v in out["votes"]]

    closes = [round(float(x), 5) for x in df["close"].tail(120).tolist()]

    return {
        "symbol": symbol,
        "price": round(float(df["close"].iloc[-1]), 5),
        "decision": out["decision"],
        "direction": out["direction"],
        "consensus": out["consensus"],
        "avg_conf": out["avg_conf"],
        "votes": votes,
        "risk": {"signal": out["risk"].signal, "reason": out["risk"].reason},
        "spark": closes,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.route("/")
def index():
    return render_template("dashboard.html", symbols=SYMBOLS)


def _clean_symbol(symbol):
    return symbol if symbol in SYMBOLS + ["DX-Y.NYB", "GBPUSD=X"] else "EURUSD=X"


@app.route("/api/decision")
def api_decision():
    symbol = _clean_symbol(request.args.get("symbol", "EURUSD=X"))
    try:
        return jsonify(_run_decision(symbol))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news")
def api_news():
    symbol = _clean_symbol(request.args.get("symbol", "EURUSD=X"))
    try:
        return jsonify(get_news(symbol))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _clean_interval(iv):
    return iv if iv in ("1h", "15m") else "1h"


@app.route("/api/range")
def api_range():
    symbol = _clean_symbol(request.args.get("symbol", "EURUSD=X"))
    interval = _clean_interval(request.args.get("interval", "1h"))
    try:
        return jsonify(bt.available_range(symbol, interval=interval))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# backtest cache keyed by the full query (symbol + dates + config choice)
_BT_CACHE = {}
_BT_TTL = 600


@app.route("/api/backtest")
def api_backtest():
    symbol = _clean_symbol(request.args.get("symbol", "EURUSD=X"))
    start = request.args.get("start") or None
    end = request.args.get("end") or None
    which = request.args.get("config", "learned")   # learned | default
    interval = _clean_interval(request.args.get("interval", "1h"))
    key = (symbol, start, end, which, interval)
    now = time.time()
    with _LOCK:
        hit = _BT_CACHE.get(key)
        if hit and now - hit[0] < _BT_TTL and not request.args.get("fresh"):
            return jsonify(hit[1])
    try:
        params = mb.DEFAULT_PARAMS if which == "default" else mb.load_config(interval)
        r = bt.run(symbol, start=start, end=end, params=params, interval=interval)
        with _LOCK:
            _BT_CACHE[key] = (now, r)
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- learning job (background thread + progress) -----------------------------
_LEARN = {"running": False, "frac": 0.0, "msg": "idle", "report": None, "error": None}


def _learn_worker(symbol, n_iter):
    import learn
    try:
        def prog(msg, frac):
            _LEARN["msg"] = msg
            _LEARN["frac"] = round(float(frac), 3)
        rep = learn.learn_all(symbol, n_iter=n_iter, progress=prog)
        _LEARN["report"] = rep
        _LEARN["msg"] = "done"
        _LEARN["frac"] = 1.0
    except Exception as e:
        _LEARN["error"] = str(e)
        _LEARN["msg"] = "error"
    finally:
        _LEARN["running"] = False
        _BT_CACHE.clear()          # learned config changed -> invalidate backtests


@app.route("/api/learn/start", methods=["POST", "GET"])
def api_learn_start():
    symbol = _clean_symbol(request.args.get("symbol", "EURUSD=X"))
    n_iter = int(request.args.get("iters", 700))
    if _LEARN["running"]:
        return jsonify({"running": True, "msg": "already running"})
    _LEARN.update({"running": True, "frac": 0.0, "msg": "starting",
                   "report": None, "error": None})
    threading.Thread(target=_learn_worker, args=(symbol, n_iter), daemon=True).start()
    return jsonify({"running": True, "msg": "started"})


@app.route("/api/learn/status")
def api_learn_status():
    return jsonify(_LEARN)


@app.route("/api/learn/last")
def api_learn_last():
    import learn
    return jsonify({"report": learn.load_report()})


if __name__ == "__main__":
    print("AlphaBot dashboard -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
