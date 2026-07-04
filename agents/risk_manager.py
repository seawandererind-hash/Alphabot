"""Risk Manager — position sizing + HARD BLOCK. Runs last, can veto everything."""
from .base import Vote


def evaluate(account, ctx=None):
    """account: dict with day_pnl_pct, open_trades, wins, trades, killzone."""
    if account.get("day_pnl_pct", 0) <= -3:
        return Vote("Risk Manager", "BLOCK", 100, "daily loss limit -3% hit")
    if account.get("open_trades", 0) >= 5:
        return Vote("Risk Manager", "BLOCK", 100, "max 5 open trades")
    tr = account.get("trades", 0)
    if tr >= 6 and account.get("wins", 0) / max(tr, 1) < 0.30:
        return Vote("Risk Manager", "BLOCK", 100, "drawdown guard (win rate < 30%)")
    return Vote("Risk Manager", "OK", 100, "within limits")
