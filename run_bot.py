"""run_bot.py — one live decision snapshot: every agent votes, master brain decides."""
from data_loader import get_candles
import master_brain as mb

SYMBOL = "EURUSD=X"

def main():
    df = get_candles(SYMBOL, period="60d", interval="1h")
    ctx = {"symbol": SYMBOL,
           "dxy": get_candles("DX-Y.NYB", period="60d", interval="1h"),
           "gbp": get_candles("GBPUSD=X", period="60d", interval="1h")}
    account = {"day_pnl_pct": 0, "open_trades": 0, "wins": 0, "trades": 0}
    out = mb.decide(df, ctx, account)

    print("\n================  ALPHABOT DECISION  ================")
    print(f"{SYMBOL}   price {df['close'].iloc[-1]:.5f}\n")
    for v in out["votes"]:
        w = v.meta.get("weight", 1.0)
        print(f"  {v.agent:<18} {v.signal:<6} {v.confidence:>3}%  x{w}   {v.reason}")
    print(f"\n  {out['risk'].agent:<18} {out['risk'].signal:<6}      {out['risk'].reason}")
    print("-" * 52)
    print(f"  MASTER DECISION : {out['decision']}")
    print(f"  consensus       : {out['consensus']}   avg confidence {out['avg_conf']}%")
    print("====================================================")

if __name__ == "__main__":
    main()
