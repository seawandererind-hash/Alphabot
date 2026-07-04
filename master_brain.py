"""master_brain.py — collect agent votes, weighted consensus, final decision."""
from agents import (market_structure, session_timer, amd_detector, momentum_ml,
                    trend_following, correlation, stat_arbitrage, standby, risk_manager)

CORE = [market_structure.analyze, session_timer.analyze, amd_detector.analyze,
        momentum_ml.analyze, trend_following.analyze, standby.news, standby.sentiment]
POWER = [standby.nlp_tone, correlation.analyze, stat_arbitrage.analyze]


def collect_votes(df, ctx):
    votes = [f(df, ctx) for f in CORE]
    for f in POWER:
        v = f(df, ctx); v.meta["weight"] = 1.5
        votes.append(v)
    for v in votes:
        v.meta.setdefault("weight", 1.0)
    return votes


def decide(df, ctx, account):
    votes = collect_votes(df, ctx)
    risk = risk_manager.evaluate(account, ctx)

    buy = sum(v.meta["weight"] for v in votes if v.signal == "BUY")
    sell = sum(v.meta["weight"] for v in votes if v.signal == "SELL")
    directional = buy + sell

    if buy == sell:
        direction = "HOLD"
        aligned = 0.0
    else:
        direction = "BUY" if buy > sell else "SELL"
        aligned = buy if direction == "BUY" else sell
    consensus = (aligned / directional) if directional else 0.0

    directional_votes = [v for v in votes if v.signal in ("BUY", "SELL")]
    avg_conf = (int(sum(v.confidence for v in directional_votes) / len(directional_votes))
                if directional_votes else 0)

    MIN_CONVICTION = 4.0
    if risk.signal == "BLOCK":
        decision = "BLOCK"
    elif direction == "HOLD":
        decision = "HOLD"
    elif directional >= MIN_CONVICTION and consensus >= 0.70:
        decision = direction
    elif directional >= MIN_CONVICTION and consensus >= 0.60:
        decision = "HALF_" + direction
    else:
        decision = "HOLD"

    return {"decision": decision, "direction": direction, "consensus": round(consensus, 2),
            "avg_conf": avg_conf, "votes": votes, "risk": risk}
