"""
week_test.py — replay last week on 15-minute candles, all agents auto-trading,
then update the agents from what worked.

Honest split: the optimizer only sees data BEFORE the test week (train), and the
week itself is scored out-of-sample. The tuned 15m config is saved to
config_15m.json (the 1h live config is untouched).

Usage:  python week_test.py [SYMBOL] [WEEK_START] [WEEK_END]
        python week_test.py EURUSD=X 2026-06-29 2026-07-04
"""
import sys
import json
from collections import defaultdict

import numpy as np

import engine
import learn
import master_brain as mb
from data_loader import load_aligned

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "EURUSD=X"
WEEK_START = sys.argv[2] if len(sys.argv) > 2 else "2026-06-29"
WEEK_END = sys.argv[3] if len(sys.argv) > 3 else "2026-07-04"
INTERVAL, DAYS, PIP = "15m", 59, 0.0001
WARMUP = 210
_ABSTAIN = {"News & Calendar", "Social Sentiment"}


def show(tag, r):
    print(f"  {tag:<26} trades {r['trades']:>4}   win {r['win_rate']:>5}%   "
          f"net {r['net_pips']:>+9.1f} pips")


def main():
    print(f"== {SYMBOL} {INTERVAL} · test week {WEEK_START} -> {WEEK_END} ==")
    df, dxy, gbp = load_aligned(SYMBOL, DAYS, INTERVAL)
    print(f"data: {len(df)} candles  {str(df.index[0])[:16]} -> {str(df.index[-1])[:16]}")

    pre = engine.precompute(df, dxy, gbp, SYMBOL, warmup=WARMUP,
                            progress=lambda k, m: print(f"  scoring {k}/{m}", end="\r"))
    print()
    idx = df.index[WARMUP:]
    cut = int((idx >= WEEK_START).argmax())
    endmask = idx >= WEEK_END
    end = int(endmask.argmax()) if endmask.any() else len(idx)
    train = engine.sub(pre, 0, cut)
    week = engine.sub(pre, cut, end)
    print(f"train: {cut} candles (before week) | week: {end - cut} candles\n")

    print("-- last week, as-is (agents' current knowledge) --")
    res_def = engine.simulate(week, mb.DEFAULT_PARAMS, PIP)
    show("default config", res_def)
    res_1h = engine.simulate(week, mb.load_config("1h"), PIP)
    show("learned 1h config", res_1h)

    # ---- tune on the 7 weeks BEFORE the test week (week never seen) ----------
    print("\n-- tuning on prior weeks only (900 candidates) --")
    rng = np.random.default_rng(1)
    best, best_score = None, -1e18
    for it in range(900):
        w = {n: float(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
             for n in engine.AGENT_NAMES if n not in _ABSTAIN}
        tf = float(rng.uniform(0.60, 0.85))
        p = {"weights": w, "thr_full": round(tf, 3),
             "thr_half": round(float(rng.uniform(0.50, tf)), 3),
             "min_conviction": float(rng.integers(2, 8)),
             "sl": int(rng.integers(5, 21)), "tp": int(rng.integers(8, 41))}
        r = engine.simulate(train, p, PIP)
        score = r["net_pips"] if r["trades"] >= 30 else -1e17
        if score > best_score:
            best_score, best = score, p

    res_train = engine.simulate(train, best, PIP)
    res_week = engine.simulate(week, best, PIP)
    show("tuned  (train, in-sample)", res_train)
    show("tuned  (WEEK, out-of-sample)", res_week)

    learn.save_config(best, "15m")
    print("\nsaved -> config_15m.json (15m backtests use it; 1h live config untouched)")
    print("weights:", json.dumps({k: v for k, v in best["weights"].items() if v > 0}))
    print(f"thr_full {best['thr_full']} · thr_half {best['thr_half']} · "
          f"min_conv {best['min_conviction']} · SL {best['sl']} · TP {best['tp']}")

    # ---- day-by-day trade log of the tuned week run --------------------------
    print("\n-- tuned config: last week trade log (day by day) --")
    per_day = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in res_week["log"]:
        d = t["ts"][:10]
        per_day[d]["n"] += 1
        per_day[d]["pnl"] += t["pnl"]
    for d in sorted(per_day):
        v = per_day[d]
        print(f"  {d}   {v['n']:>3} trades   {v['pnl']:>+8.1f} pips")
    print(f"\n  last trades:")
    for t in res_week["log"][-12:]:
        print(f"  {t['ts']}  {t['dir']:<4} entry {t['entry']}  exit {t['exit']}  "
              f"{t['pnl']:+.1f}p  cons {t['consensus']}")


if __name__ == "__main__":
    main()
