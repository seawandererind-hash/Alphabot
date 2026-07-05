"""master_brain.py — collect agent votes, weighted consensus, final decision.

Tunable: decide() takes an optional `params` dict (per-agent weights + the
consensus/conviction thresholds). If none is passed it loads config.json (written
by learn.py's optimizer), else falls back to sensible defaults. This is what lets
the agents "learn" — the optimizer just rewrites config.json.
"""
import os
import json

from agents import (market_structure, session_timer, amd_detector, momentum_ml,
                    trend_following, correlation, stat_arbitrage, standby,
                    risk_manager, learned_ml, strategy_pack)

# name -> analyze fn.  Order is stable so precompute/backtest stay reproducible.
AGENTS = [
    ("Market Structure", market_structure.analyze),
    ("Session Timer",    session_timer.analyze),
    ("AMD Detector",     amd_detector.analyze),
    ("ML/Momentum",      momentum_ml.analyze),
    ("Trend Following",  trend_following.analyze),
    ("News & Calendar",  standby.news),
    ("Social Sentiment", standby.sentiment),
    ("NLP Tone",         standby.nlp_tone),
    ("Correlation",      correlation.analyze),
    ("Stat Arbitrage",   stat_arbitrage.analyze),
    ("RSI-2 Reversion",  strategy_pack.rsi2_reversion),   # Larry Connors classic
    ("London Breakout",  strategy_pack.london_breakout),  # Asian-range break
    ("Bollinger Fade",   strategy_pack.bollinger_fade),   # sigma-stretch fade
    ("Learned ML",       learned_ml.analyze),      # HOLD unless a model is trained
]

# default weights: the 3 "power" agents + the learned model count 1.5x
_POWER = {"NLP Tone", "Correlation", "Stat Arbitrage", "Learned ML"}

DEFAULT_PARAMS = {
    "weights": {},          # per-agent override, e.g. {"Trend Following": 2.0}
    "min_conviction": 4.0,  # weighted directional votes required to trade
    "thr_full": 0.70,       # consensus for a full position
    "thr_half": 0.60,       # consensus for a half position
    "sl": 15, "tp": 30,     # pips — used by the backtester
}

_ROOT = os.path.dirname(os.path.abspath(__file__))


def config_path(interval="1h"):
    """One tuned config per timeframe: config.json (1h), config_15m.json, ..."""
    suffix = "" if interval in ("1h", None) else f"_{interval}"
    return os.path.join(_ROOT, f"config{suffix}.json")


def load_config(interval="1h"):
    """Merge the saved config for this timeframe (if any) over the defaults."""
    p = dict(DEFAULT_PARAMS)
    path = config_path(interval)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                saved = json.load(fh)
            p.update({k: v for k, v in saved.items() if k in DEFAULT_PARAMS})
            p["weights"] = {**DEFAULT_PARAMS["weights"], **saved.get("weights", {})}
        except Exception:
            pass
    return p


def _weight_for(name, params):
    w = params.get("weights", {}).get(name)
    if w is not None:
        return float(w)
    return 1.5 if name in _POWER else 1.0


def collect_votes(df, ctx, params=None):
    params = params or DEFAULT_PARAMS
    votes = []
    for name, fn in AGENTS:
        v = fn(df, ctx)
        v.meta["weight"] = _weight_for(v.agent, params)
        votes.append(v)
    return votes


def decide(df, ctx, account, params=None):
    if params is None:
        params = load_config()
    votes = collect_votes(df, ctx, params)
    risk = risk_manager.evaluate(account, ctx)

    buy = sum(v.meta["weight"] for v in votes if v.signal == "BUY")
    sell = sum(v.meta["weight"] for v in votes if v.signal == "SELL")
    directional = buy + sell

    if buy == sell:
        direction, aligned = "HOLD", 0.0
    else:
        direction = "BUY" if buy > sell else "SELL"
        aligned = buy if direction == "BUY" else sell
    consensus = (aligned / directional) if directional else 0.0

    directional_votes = [v for v in votes if v.signal in ("BUY", "SELL")]
    avg_conf = (int(sum(v.confidence for v in directional_votes) / len(directional_votes))
                if directional_votes else 0)

    min_conv = params.get("min_conviction", 4.0)
    thr_full = params.get("thr_full", 0.70)
    thr_half = params.get("thr_half", 0.60)

    if risk.signal == "BLOCK":
        decision = "BLOCK"
    elif direction == "HOLD":
        decision = "HOLD"
    elif directional >= min_conv and consensus >= thr_full:
        decision = direction
    elif directional >= min_conv and consensus >= thr_half:
        decision = "HALF_" + direction
    else:
        decision = "HOLD"

    return {"decision": decision, "direction": direction, "consensus": round(consensus, 2),
            "avg_conf": avg_conf, "votes": votes, "risk": risk}
