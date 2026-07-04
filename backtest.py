"""backtest.py — walk-forward: agents -> master brain -> risk -> simulated trades.
No future data leak: each step only sees candles up to that point.
Pure paper simulation. Places no real orders."""
from data_loader import get_candles
from agents.base import atr
import master_brain as mb


def run(symbol="EURUSD=X", pip=0.0001, sl_pips=15, tp_pips=30, warmup=210):
    df = get_candles(symbol, period="60d", interval="1h")
    dxy = get_candles("DX-Y.NYB", period="60d", interval="1h")
    gbp = get_candles("GBPUSD=X", period="60d", interval="1h")

    trades, open_t = [], None
    wins = 0
    equity_pips = 0.0

    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1]
        price = window["close"].iloc[-1]

        # manage open trade
        if open_t:
            move = (price - open_t["entry"]) / pip * (1 if open_t["dir"] == "BUY" else -1)
            if move >= tp_pips or move <= -sl_pips:
                pnl = tp_pips if move >= tp_pips else -sl_pips
                equity_pips += pnl
                wins += 1 if pnl > 0 else 0
                trades.append({**open_t, "exit": price, "pnl": pnl})
                open_t = None

        if open_t:
            continue

        ctx = {"symbol": symbol, "dxy": dxy.iloc[:i + 1], "gbp": gbp.iloc[:i + 1]}
        account = {"day_pnl_pct": 0, "open_trades": 0,
                   "wins": wins, "trades": len(trades)}
        out = mb.decide(window, ctx, account)
        d = out["decision"]
        if d in ("BUY", "SELL"):
            open_t = {"dir": d, "entry": price, "consensus": out["consensus"],
                      "conf": out["avg_conf"], "ts": str(window.index[-1])[:16]}

    n = len(trades)
    wr = (wins / n * 100) if n else 0
    return {"symbol": symbol, "candles": len(df), "trades": n,
            "wins": wins, "win_rate": round(wr, 1),
            "net_pips": round(equity_pips, 1), "log": trades}


if __name__ == "__main__":
    r = run()
    print("\n===== BACKTEST RESULT =====")
    print(f"Symbol     : {r['symbol']}")
    print(f"Candles    : {r['candles']}")
    print(f"Trades     : {r['trades']}")
    print(f"Win rate   : {r['win_rate']}%  ({r['wins']}/{r['trades']})")
    print(f"Net pips   : {r['net_pips']}")
    print("\nLast 5 trades:")
    for t in r["log"][-5:]:
        print(f"  {t['ts']}  {t['dir']:<4} entry {t['entry']:.5f}  "
              f"pnl {t['pnl']:+.0f}p  consensus {t['consensus']}")
