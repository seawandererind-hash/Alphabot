"""
retrain2.py — re-optimize with the 15-agent roster (Liquidity Windows + COT).

Search: 40% mutations of the current adopted config (regime router), 60%
fresh random candidates (incl. regime variants). Scored on the 3 CV segments;
holdout (last 15%) is ONLY the final gate — adopt only if it beats the
current +155.0p at >=65% win. Honest by construction.
"""
import json
import pickle

import numpy as np

import engine
import master_brain as mb
import auto_train

PIP, SPREAD = 0.0001, 1.0
N_ITER = 9000

pre = pickle.load(open("models/pre_1h.pkl", "rb"))
segs, hold = auto_train.segments(pre)

base = dict(mb.load_config("1h"))
base["spread"] = SPREAD
base_h = engine.simulate(hold, base, PIP)
base_cv = auto_train.score(segs, base)
print(f"BASELINE (regime champion): cv {base_cv['score'] if base_cv else None} | "
      f"holdout {base_h['net_pips']:+.1f}p ({base_h['win_rate']}% win)")

rng = np.random.default_rng(21)
NEW = ["Liquidity Windows", "COT Institutional"]


def mutate_full(p0):
    p = json.loads(json.dumps(p0))
    for wkey in ("weights", "weights_r"):
        if not p.get(wkey):
            continue
        for n in engine.AGENT_NAMES:
            cur = p[wkey].get(n, 0.0)
            if n in NEW and rng.random() < 0.5:      # push the new agents in
                p[wkey][n] = float(rng.choice([0.5, 1.0, 1.5, 2.0, 2.5]))
            elif rng.random() < 0.25:
                p[wkey][n] = float(max(0.0, min(2.5, cur + rng.choice([-0.5, 0.5]))))
    p["thr_full"] = round(min(0.92, max(0.55, p["thr_full"] + float(rng.uniform(-0.03, 0.03)))), 3)
    p["thr_half"] = round(min(p["thr_full"], max(0.45, p["thr_half"] + float(rng.uniform(-0.03, 0.03)))), 3)
    p["sl"] = int(max(5, min(45, p["sl"] + rng.integers(-3, 4))))
    p["tp"] = int(max(5, min(55, p["tp"] + rng.integers(-3, 4))))
    if rng.random() < 0.3:
        p["regime_thr"] = float(rng.choice([0.25, 0.30, 0.35, 0.40]))
    p["spread"] = SPREAD
    return p


results = []
for it in range(N_ITER):
    if rng.random() < 0.4:
        p = mutate_full(base)
    else:
        p = auto_train.sample(rng, "1h")
        if rng.random() < 0.5:                       # random regime variant
            p["weights_r"] = {n: float(rng.choice([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
                              for n in p["weights"]}
            p["regime_n"] = int(rng.choice([36, 48, 64]))
            p["regime_thr"] = float(rng.choice([0.25, 0.30, 0.35, 0.40]))
    r = auto_train.score(segs, p)
    if r:
        results.append((r, p))
    if it % 1500 == 0:
        print(f"  {it}/{N_ITER} searched, {len(results)} valid", flush=True)

results.sort(key=lambda x: -x[0]["score"])
print(f"\nHOLDOUT GATE — top 5 by CV vs baseline {base_h['net_pips']:+.1f}p:")
winner = None
for r, p in results[:5]:
    h = engine.simulate(hold, p, PIP)
    used_new = {n: p["weights"].get(n, 0) for n in NEW}
    used_new_r = {n: (p.get("weights_r") or {}).get(n, 0) for n in NEW}
    beat = h["net_pips"] > base_h["net_pips"] and h["win_rate"] >= 65
    print(f"  cv {r['score']:>7.1f} ({r['win_rate']}%) | holdout {h['net_pips']:+7.1f}p "
          f"({h['win_rate']}% win, {h['trades']} tr) | LW/COT w {used_new}/{used_new_r} "
          f"{'<<< BEATS' if beat else ''}")
    if beat and winner is None:
        winner = (p, h)

if winner:
    p, h = winner
    cfg = {k: v for k, v in p.items() if k != "spread"}
    with open(mb.config_path("1h"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nADOPTED -> config.json (holdout {h['net_pips']:+.1f}p, {h['win_rate']}% win)")
else:
    print("\nNo candidate beat the regime champion on holdout — config UNCHANGED.")
