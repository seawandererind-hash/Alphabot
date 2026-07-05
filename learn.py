"""
learn.py — how the agents "learn" from 2 years of real history.

Two independent methods, both split TRAIN (older 70%) / TEST (newer 30%) by time
so the reported numbers are out-of-sample and honest — no future data leaks into
what we score.

  1. optimize()  — random search over per-agent weights + consensus/conviction
                   thresholds + SL/TP. Best on TRAIN, then measured on TEST.
                   Winner is written to config.json (the live bot reads it).
  2. train_ml()  — a gradient-boosting model learns to predict whether price is
                   higher `horizon` candles ahead, from the shared feature set.
                   Saved to models/model.pkl and voted by the Learned ML agent.

learn_all() runs ML first (so its vote exists), then precomputes votes once, then
tunes weights over everything.
"""
import os
import json
import pickle

import numpy as np

import engine
import master_brain as mb
from features import feature_frame, FEATURE_COLS

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_ROOT, "models")
_CFG = os.path.join(_ROOT, "config.json")

# agents that always abstain — no point spending search budget on their weight
_ABSTAIN = {"News & Calendar", "Social Sentiment"}
_MIN_TRADES = 12          # ignore param sets that trade too rarely to trust


# ---------------------------------------------------------------- ML model ----
def train_ml(df, horizon=5, frac=0.7):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    f = feature_frame(df)
    close = df["close"]
    label = (close.shift(-horizon) > close).astype(int)
    data = f.copy()
    data["y"] = label
    data = data.dropna()

    X = data[FEATURE_COLS].values
    y = data["y"].values.astype(int)
    cut = int(len(X) * frac)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    scaler = StandardScaler().fit(Xtr)
    model = GradientBoostingClassifier(n_estimators=160, max_depth=3,
                                       learning_rate=0.05, subsample=0.85,
                                       random_state=0)
    model.fit(scaler.transform(Xtr), ytr)
    acc_tr = float(model.score(scaler.transform(Xtr), ytr))
    acc_te = float(model.score(scaler.transform(Xte), yte))
    baseline = float(max(yte.mean(), 1 - yte.mean()))  # always-guess-majority

    os.makedirs(_MODELS, exist_ok=True)
    with open(os.path.join(_MODELS, "model.pkl"), "wb") as fh:
        pickle.dump({"model": model, "scaler": scaler, "buy_th": 0.55,
                     "sell_th": 0.45, "horizon": horizon, "features": FEATURE_COLS}, fh)

    return {"train_acc": round(acc_tr, 3), "test_acc": round(acc_te, 3),
            "baseline": round(baseline, 3), "rows": int(len(X)), "horizon": horizon,
            "edge": round(acc_te - baseline, 3)}


# ---------------------------------------------------------------- optimizer ---
def _sample(rng):
    weights = {}
    for name in engine.AGENT_NAMES:
        if name in _ABSTAIN:
            continue
        weights[name] = float(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    thr_full = float(rng.uniform(0.60, 0.85))
    thr_half = float(rng.uniform(0.50, thr_full))
    return {
        "weights": weights,
        "thr_full": round(thr_full, 3),
        "thr_half": round(thr_half, 3),
        "min_conviction": float(rng.integers(2, 8)),
        "sl": int(rng.integers(10, 26)),
        "tp": int(rng.integers(15, 46)),
    }


def optimize(pre, n_iter=700, seed=0, frac=0.7, pip=0.0001, progress=None):
    rng = np.random.default_rng(seed)
    m = len(pre["close"])
    cut = int(m * frac)
    train, test = engine.sub(pre, 0, cut), engine.sub(pre, cut, m)

    base_params = mb.load_config()
    base_test = engine.simulate(test, base_params, pip)

    best, best_score = None, -1e18
    for it in range(n_iter):
        p = _sample(rng)
        r = engine.simulate(train, p, pip)
        score = r["net_pips"] if r["trades"] >= _MIN_TRADES else -1e17
        if score > best_score:
            best_score, best = score, p
        if progress and it % 50 == 0:
            progress(it, n_iter)
    if progress:
        progress(n_iter, n_iter)

    best_train = engine.simulate(train, best, pip)
    best_test = engine.simulate(test, best, pip)
    return {
        "best": best,
        "baseline_test": {k: base_test[k] for k in ("trades", "win_rate", "net_pips")},
        "tuned_train": {k: best_train[k] for k in ("trades", "win_rate", "net_pips")},
        "tuned_test": {k: best_test[k] for k in ("trades", "win_rate", "net_pips")},
    }


def save_config(best, interval="1h"):
    with open(mb.config_path(interval), "w") as fh:
        json.dump(best, fh, indent=2)


_REPORT = os.path.join(_ROOT, "models", "last_report.json")


def save_report(rep):
    os.makedirs(_MODELS, exist_ok=True)
    with open(_REPORT, "w") as fh:
        json.dump(rep, fh, indent=2)


def load_report():
    if os.path.exists(_REPORT):
        try:
            with open(_REPORT) as fh:
                return json.load(fh)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------- full run ----
def learn_all(symbol="EURUSD=X", days=730, interval="1h", n_iter=700,
              lookback=250, warmup=210, progress=None):
    """ML -> precompute votes -> tune weights. Returns a full report and writes
    models/model.pkl + config.json."""
    from data_loader import load_aligned

    def stage(msg, frac):
        if progress:
            progress(msg, frac)

    stage("loading 2y data", 0.05)
    df, dxy, gbp = load_aligned(symbol, days, interval)

    stage("training ML model", 0.15)
    ml = train_ml(df)

    stage("scoring agents across history", 0.25)

    def pp(k, total):
        if progress and total:
            stage(f"scoring candles {k}/{total}", 0.25 + 0.5 * (k / total))
    pre = engine.precompute(df, dxy, gbp, symbol, warmup=warmup,
                            lookback=lookback, progress=pp)

    stage("optimising weights & thresholds", 0.80)

    def op(it, total):
        if progress and total:
            stage(f"search {it}/{total}", 0.80 + 0.18 * (it / total))
    opt = optimize(pre, n_iter=n_iter, progress=op)

    save_config(opt["best"])
    report = {"symbol": symbol, "days": days, "interval": interval,
              "candles": int(len(df)), "ml": ml, "opt": opt,
              "range": [str(df.index[0])[:10], str(df.index[-1])[:10]]}
    save_report(report)
    stage("done", 1.0)
    return report


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "EURUSD=X"
    rep = learn_all(sym, n_iter=400,
                    progress=lambda m, f: print(f"[{f*100:4.0f}%] {m}"))
    print(json.dumps(rep, indent=2))
