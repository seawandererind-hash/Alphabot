"""
upgrade_search.py — bolt smart exits + a regime router onto the trained
champion and test honestly.

Selection happens on the 3 CV segments (mean - 0.5*std). The holdout (last
15%) is used ONLY as a final gate: the upgrade is adopted only if it beats the
current champion's +94.5p there. Prints everything; writes config.json only on
a win.
"""
import json
import pickle

import numpy as np

import engine
import master_brain as mb
import auto_train

PIP, SPREAD = 0.0001, 1.0

pre = pickle.load(open("models/pre_1h.pkl", "rb"))
segs, hold = auto_train.segments(pre)

champ = dict(mb.load_config("1h"))
champ["spread"] = SPREAD


def cv(p):
    return auto_train.score(segs, p)


def on_hold(p):
    return engine.simulate(hold, p, PIP)


base_cv = cv(champ)
base_h = on_hold(champ)
print(f"BASELINE champion: cv {base_cv['score']:.1f} ({base_cv['win_rate']}% win) | "
      f"holdout {base_h['net_pips']:+.1f}p ({base_h['win_rate']}% win, {base_h['trades']} tr)")

results = []

# ---- stage 1: exhaustive smart-exit grid on the champion ---------------------
print("\n[1] smart-exit grid (be_trigger x trail)...")
for be in (0, 4, 6, 8, 10, 12, 15):
    for tr in (0, 6, 8, 10, 12, 15, 20, 25, 30):
        if be == 0 and tr == 0:
            continue
        p = dict(champ)
        p["be_trigger"], p["trail"] = be, tr
        r = cv(p)
        if r:
            results.append(("exit", f"be{be}/tr{tr}", r, p))
best_exit = max((x for x in results if x[0] == "exit"),
                key=lambda x: x[2]["score"], default=None)
if best_exit:
    _, tag, r, _ = best_exit
    print(f"    best exit combo: {tag}  cv {r['score']:.1f} ({r['win_rate']}% win)")

# ---- stage 2: regime router — trend-team weights for trending candles --------
print("[2] regime router search (champion weights = ranging side)...")
rng = np.random.default_rng(7)
TREND_POOL = ["Trend Following", "Market Structure", "London Breakout",
              "Learned ML", "ML/Momentum", "Stat Arbitrage"]
for it in range(6000):
    p = json.loads(json.dumps(champ))
    p["weights_r"] = dict(p["weights"])            # ranging side = champion
    tw = {n: 0.0 for n in p["weights"]}
    for n in TREND_POOL:
        tw[n] = float(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    p["weights"] = tw                              # trending side = new team
    p["regime_n"] = int(rng.choice([36, 48, 64]))
    p["regime_thr"] = float(rng.choice([0.25, 0.30, 0.35, 0.40]))
    if best_exit and rng.random() < 0.5:           # half the time keep best exits
        p["be_trigger"] = best_exit[3]["be_trigger"]
        p["trail"] = best_exit[3]["trail"]
    r = cv(p)
    if r:
        results.append(("regime", f"thr{p['regime_thr']}", r, p))

# ---- rank by CV, judge top-5 on holdout --------------------------------------
results.sort(key=lambda x: -x[2]["score"])
print(f"\n[3] holdout gate — top 5 by CV vs champion {base_h['net_pips']:+.1f}p:")
winner = None
for kind, tag, r, p in results[:5]:
    h = on_hold(p)
    beat = h["net_pips"] > base_h["net_pips"] and h["win_rate"] >= 65
    print(f"    {kind:<6} {tag:<12} cv {r['score']:>7.1f} ({r['win_rate']}%) | "
          f"holdout {h['net_pips']:+7.1f}p ({h['win_rate']}% win, {h['trades']} tr) "
          f"{'<<< BEATS CHAMPION' if beat else ''}")
    if beat and winner is None:
        winner = (kind, tag, p, h)

if winner:
    kind, tag, p, h = winner
    cfg = {k: v for k, v in p.items() if k != "spread"}
    with open(mb.config_path("1h"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nADOPTED: {kind} {tag} -> config.json updated "
          f"(holdout {h['net_pips']:+.1f}p, {h['win_rate']}% win)")
else:
    print("\nNo upgrade beat the champion on holdout with >=65% win — config UNCHANGED.")
