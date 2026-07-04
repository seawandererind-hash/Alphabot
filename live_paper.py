"""
live_paper.py — LIVE paper trading loop (runs on YOUR machine)
--------------------------------------------------------------
Two speeds, like a real bot:
  * PRICE_SEC (default 10s): fetch latest PRICE and manage the open trade's
    SL/TP. Fast reaction for exits.
  * Full agent analysis runs only when a NEW candle closes (no point re-running
    a 1H strategy every 10 seconds), or at most every ANALYZE_MIN minutes.

Real-time price needs a streaming source. Set env TWELVEDATA_KEY (free tier,
works in India) for real-time quotes. Without it, prices are Yahoo-delayed
(~15 min) — OK to learn the plumbing, not for real scalping.

Run:            python live_paper.py
Stop:           Ctrl + C
Quick test:     ALPHABOT_TEST=1 python live_paper.py

Paper only — places NO real orders. Logs to paper_trades.csv.
"""
import os, csv, time, datetime
from data_loader import get_candles, get_price
import master_brain as mb

SYMBOL     = "EURUSD=X"
PIP        = 0.0001
SL_PIPS    = 15
TP_PIPS    = 30
PRICE_SEC  = 10        # how often to check price for SL/TP (seconds)
ANALYZE_MIN = 15       # re-run agents at most this often (minutes) for new entries
LOGFILE    = "paper_trades.csv"

state = {"open": None, "wins": 0, "trades": 0, "pips": 0.0,
         "df": None, "last_candle_ts": None, "last_analyze": 0}


def _log(row):
    new = not os.path.exists(LOGFILE)
    with open(LOGFILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "dir", "price", "pnl_pips",
                        "consensus", "net_pips", "win_rate"])
        w.writerow(row)


def _refresh_candles():
    df = get_candles(SYMBOL, period="60d", interval="1h")
    state["df"] = df
    return df


def _price():
    p = get_price(SYMBOL)
    if p is None and state["df"] is not None:      # fallback to last close
        p = float(state["df"]["close"].iloc[-1])
    return p


def _manage_open(price):
    t = state["open"]
    move = (price - t["entry"]) / PIP * (1 if t["dir"] == "BUY" else -1)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if move >= TP_PIPS or move <= -SL_PIPS:
        pnl = TP_PIPS if move >= TP_PIPS else -SL_PIPS
        state["pips"] += pnl; state["trades"] += 1
        state["wins"] += 1 if pnl > 0 else 0
        wr = round(state["wins"] / state["trades"] * 100, 1)
        _log([now, "CLOSE", t["dir"], f"{price:.5f}", f"{pnl:+.0f}",
              t["consensus"], round(state["pips"], 1), wr])
        print(f"{now}  CLOSE {t['dir']} @ {price:.5f}  {pnl:+.0f}p | "
              f"net {state['pips']:+.0f}p  win {wr}%")
        state["open"] = None
    else:
        print(f"{now}  watching {t['dir']} {move:+.1f}p", end="\r")


def _maybe_enter():
    df = state["df"]
    ctx = {"symbol": SYMBOL,
           "dxy": get_candles("DX-Y.NYB", period="60d", interval="1h"),
           "gbp": get_candles("GBPUSD=X", period="60d", interval="1h")}
    account = {"day_pnl_pct": 0, "open_trades": 0,
               "wins": state["wins"], "trades": state["trades"]}
    out = mb.decide(df, ctx, account)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if out["decision"] in ("BUY", "SELL"):
        price = float(df["close"].iloc[-1])
        state["open"] = {"dir": out["decision"], "entry": price,
                         "consensus": out["consensus"]}
        _log([now, "OPEN", out["decision"], f"{price:.5f}", "",
              out["consensus"], round(state["pips"], 1), ""])
        print(f"{now}  OPEN {out['decision']} @ {price:.5f}  "
              f"consensus {out['consensus']} conf {out['avg_conf']}%")
    else:
        print(f"{now}  no entry ({out['decision']}, consensus {out['consensus']})")


def _tick(test=False):
    price = _price()
    if price is None:
        print("no price yet"); return

    if state["open"]:                     # fast path: manage exit
        _manage_open(price)
        return

    # entry path: only re-analyse on a new candle or every ANALYZE_MIN
    now = time.time()
    df = state["df"]
    fresh_candle = df is not None and df.index[-1] != state["last_candle_ts"]
    due = now - state["last_analyze"] >= ANALYZE_MIN * 60
    if df is None or fresh_candle or due or test:
        _refresh_candles()
        state["last_candle_ts"] = state["df"].index[-1]
        state["last_analyze"] = now
        _maybe_enter()


def main():
    test = os.environ.get("ALPHABOT_TEST") == "1"
    rt = "REAL-TIME (Twelve Data)" if os.environ.get("TWELVEDATA_KEY") else "DELAYED (Yahoo)"
    print(f"AlphaBot LIVE paper — {SYMBOL} — price every {PRICE_SEC}s, "
          f"analysis on candle close — {rt}\nPaper only, no real orders.\n")
    _refresh_candles(); state["last_candle_ts"] = state["df"].index[-1]
    loops = 0
    while True:
        try:
            _tick(test=test)
        except Exception as e:
            print("tick error:", e)
        loops += 1
        if test and loops >= 3:
            print("\n[test mode] done."); break
        time.sleep(1 if test else PRICE_SEC)


if __name__ == "__main__":
    main()
