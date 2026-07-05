"""
auto_train.py — checkpointed long-run trainer for AlphaBot.

Each invocation runs ONE chunk (2-6 min) and exits, saving progress to
models/train_state.json so the next invocation resumes where it left off.
Relaunch it repeatedly for a multi-hour training session; it stops itself
at the deadline (default 6h from first run) and writes final configs.

Honesty rules baked in:
- every simulated trade pays 1 pip spread
- configs are scored mean - 0.5*std of net pips across 3 sequential time
  segments (must be consistent across regimes, not lucky in one)
- the final 15% of history (holdout) is NEVER seen during search; the end
  report scores champion vs current configs on it
- the ML model is only adopted if its edge on an untouched test tail is > 0

Phases: diag -> ml -> opt1h -> opt15m -> final
"""
import os
import sys
import json
import time
import pickle

import numpy as np

import engine
import master_brain as mb
from data_loader import load_aligned

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
STATE_F = os.path.join(MODELS, "train_state.json")
LOG_F = os.path.join(ROOT, "training_log.md")

PIP = 0.0001
SPREAD = 1.0                    # pips per round-trip, both timeframes
HOURS = 6.0
ABSTAIN = {"News & Calendar", "Social Sentiment"}
DATA = {"1h": ("EURUSD=X", 730), "15m": ("EURUSD=X", 59)}


def log(msg):
    line = f"- {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_F, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state():
    if os.path.exists(STATE_F):
        with open(STATE_F) as fh:
            return json.load(fh)
    return None


def save_state(st):
    os.makedirs(MODELS, exist_ok=True)
    with open(STATE_F, "w") as fh:
        json.dump(st, fh, indent=1)


# ------------------------------------------------------------- vote cache ----
def get_pre(interval):
    """Precompute the agent-vote matrix once per timeframe, cache to disk."""
    symbol, days = DATA[interval]
    path = os.path.join(MODELS, f"pre_{interval}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    df, dxy, gbp = load_aligned(symbol, days, interval)
    t = time.time()
    pre = engine.precompute(df, dxy, gbp, symbol)
    log(f"precomputed {interval}: {len(pre['close'])} candles in {time.time()-t:.0f}s")
    with open(path, "wb") as fh:
        pickle.dump(pre, fh)
    return pre


def refresh_ml_column(interval):
    """After retraining the ML model, refill its column in the cached votes."""
    symbol, days = DATA[interval]
    path = os.path.join(MODELS, f"pre_{interval}.pkl")
    if not os.path.exists(path):
        return
    with open(path, "rb") as fh:
        pre = pickle.load(fh)
    df, _, _ = load_aligned(symbol, days, interval)
    i0 = 210
    i1 = i0 + len(pre["close"])
    engine._fill_ml_column(pre["sig"], pre["conf"], df, i0, i1,
                           pre["names"].index("Learned ML"))
    with open(path, "wb") as fh:
        pickle.dump(pre, fh)


# ------------------------------------------------------------- scoring -------
def segments(pre, k=3, holdout=0.15):
    m = len(pre["close"])
    cut = int(m * (1 - holdout))
    step = cut // k
    segs = [engine.sub(pre, i * step, (i + 1) * step if i < k - 1 else cut)
            for i in range(k)]
    return segs, engine.sub(pre, cut, m)


def sample(rng, interval):
    w = {n: float(rng.choice([0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
         for n in engine.AGENT_NAMES if n not in ABSTAIN}
    tf = float(rng.uniform(0.60, 0.88))
    # wide SL/TP ranges INCLUDING tp < sl (scalp shapes win more often per
    # trade — the only route to very high win rates — but must still net
    # positive pips after spread to be accepted)
    sl_lo, sl_hi, tp_lo, tp_hi = (8, 40, 6, 50) if interval == "1h" else (5, 30, 4, 40)
    return {"weights": w, "thr_full": round(tf, 3),
            "thr_half": round(float(rng.uniform(0.50, tf)), 3),
            "min_conviction": float(rng.integers(2, 9)),
            "sl": int(rng.integers(sl_lo, sl_hi)),
            "tp": int(rng.integers(tp_lo, tp_hi)),
            "spread": SPREAD}


def score(segs, p):
    """Consistency score + aggregate stats across the CV segments."""
    pips, trades, wins = [], 0, 0
    for s in segs:
        r = engine.simulate(s, p, PIP)
        if r["trades"] < 4:
            return None
        pips.append(r["net_pips"])
        trades += r["trades"]
        wins += r["wins"]
    if trades < 30:
        return None
    return {"score": float(np.mean(pips) - 0.5 * np.std(pips)),
            "trades": trades,
            "win_rate": round(wins / trades * 100, 1),
            "total_pips": round(float(np.sum(pips)), 1)}


def mutate(rng, base, interval):
    """Small random perturbation of a champion — local hill-climbing."""
    p = json.loads(json.dumps(base))            # deep copy
    w = p["weights"]
    for n in list(w):
        if rng.random() < 0.3:
            w[n] = float(max(0.0, min(2.5, w[n] + rng.choice([-0.5, 0.5]))))
    p["thr_full"] = round(min(0.92, max(0.55, p["thr_full"] + float(rng.uniform(-0.03, 0.03)))), 3)
    p["thr_half"] = round(min(p["thr_full"], max(0.45, p["thr_half"] + float(rng.uniform(-0.03, 0.03)))), 3)
    p["min_conviction"] = float(max(1.5, min(9.0, p["min_conviction"] + rng.choice([-0.5, 0, 0.5]))))
    p["sl"] = int(max(4, min(45, p["sl"] + rng.integers(-3, 4))))
    p["tp"] = int(max(4, min(55, p["tp"] + rng.integers(-3, 4))))
    p["spread"] = SPREAD
    return p


def opt_chunk(st, interval, n_iter):
    """Track TWO champions per timeframe:
    best_*   = highest consistency score (max profit)
    best65_* = highest score among configs with win rate >= 65% AND
               positive total pips after spread (the user's target).
    35% of candidates are mutations of the champions (local refinement),
    the rest pure random exploration."""
    pre = get_pre(interval)
    segs, _ = segments(pre)
    seed = 1000 * (1 if interval == "1h" else 2) + st["chunk"]
    rng = np.random.default_rng(seed)
    kb, k65 = f"best_{interval}", f"best65_{interval}"
    best, best65 = st.get(kb), st.get(k65)
    hi_win = st.get(f"hiwin_{interval}", 0.0)   # best profitable win rate seen
    found = 0
    for _ in range(n_iter):
        roll = rng.random()
        if roll < 0.20 and best is not None:
            p = mutate(rng, best["params"], interval)
        elif roll < 0.35 and best65 is not None:
            p = mutate(rng, best65["params"], interval)
        else:
            p = sample(rng, interval)
        r = score(segs, p)
        if r is None:
            continue
        rec = {"score": round(r["score"], 1), "trades": r["trades"],
               "win_rate": r["win_rate"], "total_pips": r["total_pips"],
               "params": p}
        if best is None or r["score"] > best["score"]:
            best = rec
            found += 1
        if r["total_pips"] > 0 and r["win_rate"] > hi_win:
            hi_win = r["win_rate"]
        if (r["win_rate"] >= 65.0 and r["total_pips"] > 0
                and (best65 is None or r["score"] > best65["score"])):
            best65 = rec
            found += 1
    st[kb], st[k65] = best, best65
    st[f"hiwin_{interval}"] = hi_win
    return best, best65, hi_win, found


# ------------------------------------------------------------- diagnosis -----
def agent_quality(pre, horizon=6):
    close = pre["close"]
    fwd = np.zeros(len(close))
    fwd[:-horizon] = close[horizon:] - close[:-horizon]
    out = {}
    for a, name in enumerate(pre["names"]):
        s = pre["sig"][:-horizon, a]
        f = fwd[:-horizon]
        idx = s != 0
        n = int(idx.sum())
        if n < 10:
            out[name] = {"votes": n}
            continue
        out[name] = {
            "votes": n,
            "dir_acc": round(float((np.sign(f[idx]) == s[idx]).mean()), 3),
            "avg_fwd_pips": round(float((f[idx] * s[idx]).mean() / PIP), 2),
        }
    return out


def diag_phase(st):
    report = {}
    for interval in ("1h", "15m"):
        pre = get_pre(interval)
        segs, hold = segments(pre)
        cur = {}
        for tag, params in (("default", dict(mb.DEFAULT_PARAMS)),
                            ("learned", mb.load_config(interval))):
            p = dict(params)
            p["spread"] = SPREAD
            r = engine.simulate(engine.sub(pre, 0, len(pre["close"])), p, PIP)
            cur[tag] = {k: r[k] for k in ("trades", "win_rate", "net_pips")}
            log(f"diag {interval} {tag} WITH 1p spread: {r['trades']} trades, "
                f"{r['win_rate']}% win, {r['net_pips']:+.1f} pips")
        report[interval] = {"configs_with_spread": cur,
                           "agent_quality": agent_quality(pre)}
    with open(os.path.join(MODELS, "diag.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    aq = report["1h"]["agent_quality"]
    ranked = sorted([x for x in aq.items() if "dir_acc" in x[1]],
                    key=lambda x: -x[1]["dir_acc"])
    log("agent ranking (1h dir_acc): " +
        ", ".join(f"{n} {v['dir_acc']}" for n, v in ranked))
    st["diag"] = True


# ------------------------------------------------------------- ML phase ------
def ml_phase(st):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from features import feature_frame, FEATURE_COLS

    df, _, _ = load_aligned("EURUSD=X", 730, "1h")
    f = feature_frame(df)
    close, high, low = (df[c].values for c in ("close", "high", "low"))
    sl_d, tp_d, H = 15 * PIP, 30 * PIP, 24

    n = len(df)
    y = np.full(n, -1, dtype=int)          # 1 = TP-before-SL going long
    for i in range(210, n - H - 1):
        e = close[i]
        for j in range(i + 1, i + 1 + H):
            if low[j] <= e - sl_d:
                y[i] = 0
                break
            if high[j] >= e + tp_d:
                y[i] = 1
                break

    X = f.values
    ok = (y >= 0) & ~np.isnan(X).any(axis=1)
    X, y = X[ok], y[ok]
    a, b = int(len(X) * 0.70), int(len(X) * 0.85)
    Xtr, ytr, Xva, yva, Xte, yte = X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]

    scaler = StandardScaler().fit(Xtr)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.85,
                                       random_state=0)
    model.fit(scaler.transform(Xtr), ytr)

    def edge(Xs, ys, bt, stt):
        p = model.predict_proba(scaler.transform(Xs))[:, 1]
        pips = 0.0
        cnt = 0
        lo = p >= bt
        pips += (ys[lo] * 30 - (1 - ys[lo]) * 15 - SPREAD).sum()
        cnt += int(lo.sum())
        sh = p <= stt
        pips += ((1 - ys[sh]) * 30 - ys[sh] * 15 - SPREAD).sum()
        cnt += int(sh.sum())
        return (pips / cnt if cnt else -99.0), cnt

    best = None
    for bt in (0.55, 0.60, 0.65, 0.70):
        for stt in (0.45, 0.40, 0.35, 0.30):
            e, c = edge(Xva, yva, bt, stt)
            if c >= 25 and (best is None or e > best[0]):
                best = (e, c, bt, stt)

    if best is None:
        log("ML: no threshold pair had enough validation signals — model NOT adopted")
        st["ml"] = {"adopted": False}
        return
    e_va, c_va, bt, stt = best
    e_te, c_te = edge(Xte, yte, bt, stt)
    log(f"ML tp-before-sl label: val edge {e_va:+.2f} p/trade ({c_va} sig), "
        f"TEST edge {e_te:+.2f} p/trade ({c_te} sig), thr {bt}/{stt}")

    if e_te > 0 and c_te >= 15:
        with open(os.path.join(MODELS, "model.pkl"), "wb") as fh:
            pickle.dump({"model": model, "scaler": scaler, "buy_th": bt,
                         "sell_th": stt, "horizon": H,
                         "features": FEATURE_COLS, "label": "tp_before_sl"}, fh)
        for iv in ("1h", "15m"):
            refresh_ml_column(iv)
        log("ML model ADOPTED (positive test edge) — vote caches refreshed")
        st["ml"] = {"adopted": True, "test_edge": round(float(e_te), 2),
                    "signals": c_te, "buy_th": bt, "sell_th": stt}
    else:
        log("ML model NOT adopted (test edge <= 0) — keeping agents rule-based")
        st["ml"] = {"adopted": False, "test_edge": round(float(e_te), 2)}


# ------------------------------------------------------------- final ---------
def final_phase(st):
    summary = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "intervals": {}}
    for interval in ("1h", "15m"):
        pre = get_pre(interval)
        _, hold = segments(pre)
        cur = dict(mb.load_config(interval))
        cur["spread"] = SPREAD
        r_cur = engine.simulate(hold, cur, PIP)
        row = {"holdout_current": {k: r_cur[k] for k in ("trades", "win_rate", "net_pips")}}
        log(f"HOLDOUT {interval} current config: {r_cur['net_pips']:+.1f}p "
            f"({r_cur['trades']} trades, {r_cur['win_rate']}% win)")

        # score BOTH champions (max-profit and 65%-win-club) on the holdout
        candidates = []
        for tag in ("best", "best65"):
            b = st.get(f"{tag}_{interval}")
            if not b:
                continue
            r = engine.simulate(hold, b["params"], PIP)
            row[f"holdout_{tag}"] = {k: r[k] for k in ("trades", "win_rate", "net_pips")}
            row[f"cv_{tag}"] = {k: b[k] for k in ("score", "win_rate", "total_pips")}
            log(f"HOLDOUT {interval} {tag}: {r['net_pips']:+.1f}p "
                f"({r['trades']} trades, {r['win_rate']}% win)")
            candidates.append((r["net_pips"], r, b, tag))

        adopted = None
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            top_pips, r_new, b, tag = candidates[0]
            if top_pips > r_cur["net_pips"] and top_pips > 0:
                cfg = dict(b["params"])
                cfg.pop("spread", None)
                with open(mb.config_path(interval), "w") as fh:
                    json.dump(cfg, fh, indent=2)
                adopted = tag
                log(f"config_{interval} UPDATED with '{tag}' champion "
                    f"(holdout {top_pips:+.1f}p, {r_new['win_rate']}% win)")
        if adopted is None:
            log(f"{interval}: no champion beat current config on holdout — config kept")
        row["adopted"] = adopted
        summary["intervals"][interval] = row
    summary["ml"] = st.get("ml")
    with open(os.path.join(MODELS, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log("=== TRAINING SESSION COMPLETE — models/train_summary.json written ===")


# ------------------------------------------------------------- driver --------
def main():
    os.makedirs(MODELS, exist_ok=True)
    st = load_state()
    if st is None:
        st = {"start": time.time(), "deadline": time.time() + HOURS * 3600,
              "phase": "diag", "chunk": 0}
        save_state(st)
        log(f"=== TRAINING SESSION START ({HOURS:.0f}h budget) ===")

    if st["phase"] == "done":
        print("training already complete")
        return

    now = time.time()
    left = st["deadline"] - now
    frac_used = (now - st["start"]) / (HOURS * 3600)
    if left < 420 and st["phase"] not in ("final",):
        st["phase"] = "final"

    ph = st["phase"]
    if ph == "diag":
        diag_phase(st)
        st["phase"] = "ml"
    elif ph == "ml":
        ml_phase(st)
        st["phase"] = "opt1h"
    elif ph == "opt1h":
        best, best65, hi_win, found = opt_chunk(st, "1h", 3000)
        st["chunk"] += 1
        log(f"opt1h chunk {st['chunk']}: best score "
            f"{best['score'] if best else None} (win {best['win_rate'] if best else 0}%) | "
            f"65%-club: {'YES score ' + str(best65['score']) if best65 else 'not yet'} | "
            f"max profitable win rate so far {hi_win}%")
        if frac_used > 0.55:
            st["phase"], st["chunk"] = "opt15m", 0
    elif ph == "opt15m":
        best, best65, hi_win, found = opt_chunk(st, "15m", 5000)
        st["chunk"] += 1
        log(f"opt15m chunk {st['chunk']}: best score "
            f"{best['score'] if best else None} (win {best['win_rate'] if best else 0}%) | "
            f"65%-club: {'YES score ' + str(best65['score']) if best65 else 'not yet'} | "
            f"max profitable win rate so far {hi_win}%")
        if frac_used > 0.90:
            st["phase"] = "final"
    elif ph == "final":
        final_phase(st)
        st["phase"] = "done"

    save_state(st)
    hrs = max(0.0, (st["deadline"] - time.time()) / 3600)
    print(f"[chunk done] phase={st['phase']} chunk={st.get('chunk')} "
          f"time_left={hrs:.1f}h", flush=True)


if __name__ == "__main__":
    main()
