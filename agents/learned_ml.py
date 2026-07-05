"""
Agent 11 — Learned ML.
Loads a model trained by learn.py (models/model.pkl). Until one exists it
abstains (HOLD) so it never distorts voting. Once trained it votes BUY/SELL
from the model's probability that price rises over the training horizon.
"""
import os
import pickle

from .base import Vote

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "model.pkl")
_cache = {"mtime": None, "bundle": None}


def _load():
    if not os.path.exists(_MODEL_PATH):
        return None
    mt = os.path.getmtime(_MODEL_PATH)
    if _cache["mtime"] != mt:
        try:
            with open(_MODEL_PATH, "rb") as fh:
                _cache["bundle"] = pickle.load(fh)
            _cache["mtime"] = mt
        except Exception:
            return None
    return _cache["bundle"]


def analyze(df, ctx=None):
    bundle = _load()
    if bundle is None:
        return Vote("Learned ML", "HOLD", 55, "no model trained yet")
    try:
        from features import make_features, FEATURE_COLS
        model, scaler = bundle["model"], bundle["scaler"]
        x = [[make_features(df)[k] for k in FEATURE_COLS]]
        if scaler is not None:
            x = scaler.transform(x)
        p_up = float(model.predict_proba(x)[0][1])
        buy_th = bundle.get("buy_th", 0.55)
        sell_th = bundle.get("sell_th", 0.45)
        conf = int(abs(p_up - 0.5) * 200)          # 0..100
        conf = max(50, min(95, conf))
        if p_up >= buy_th:
            return Vote("Learned ML", "BUY", conf, f"model p(up)={p_up:.2f}")
        if p_up <= sell_th:
            return Vote("Learned ML", "SELL", conf, f"model p(up)={p_up:.2f}")
        return Vote("Learned ML", "HOLD", 55, f"model neutral p={p_up:.2f}")
    except Exception as e:
        return Vote("Learned ML", "HOLD", 55, f"model error: {e}")
