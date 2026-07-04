# AlphaBot — Multi-Agent Trading System (full starter)

A complete, runnable multi-agent decision engine. Ten agents each analyse price
from their own angle, vote, and a Master Brain trades only on strong consensus.
A Risk Manager can hard-block anything. **No broker/API key needed to run** — it
reads free Yahoo Finance data and places **no real orders** (paper simulation).

---

## Setup (one time)
1. Install Python 3.11 (python.org — tick "Add Python to PATH").
2. In this folder:  `pip install -r requirements.txt`

## Run
```
python run_bot.py     # one decision snapshot: every agent votes + master brain
python backtest.py    # walk-forward paper backtest, prints win rate + net pips
python live_paper.py  # LIVE paper loop: polls Yahoo every 15m, logs paper_trades.csv
```

`live_paper.py` runs two speeds like a real bot: it checks PRICE every 10s to
manage an open trade's SL/TP, and re-runs the agents only when a new candle
closes (re-running a 1H strategy every 10s is pointless).

Real-time price: Yahoo is ~15 min delayed and rate-limited, so true 5-10s data
needs a streaming source. Set `TWELVEDATA_KEY` (free tier, works in India) for
real-time quotes — the loop uses it automatically. Forex is also closed on
weekends. For scalping later, switch to M1/M5 candles + a streaming broker/API.
On your machine these pull REAL EUR/USD, DXY, GBP/USD data from Yahoo. If Yahoo
is blocked, it auto-falls back to synthetic candles so it still runs.

---

## The 10 agents
| # | Agent | File | Status |
|---|-------|------|--------|
| 1 | Market Structure | agents/market_structure.py | live |
| 2 | Session Timer | agents/session_timer.py | live |
| 3 | AMD Detector | agents/amd_detector.py | live |
| 4 | ML / Momentum | agents/momentum_ml.py | live (swap in XGBoost later) |
| 5 | Trend Following | agents/trend_following.py | live |
| 6 | Correlation (DXY) | agents/correlation.py | live |
| 7 | Stat Arbitrage | agents/stat_arbitrage.py | live |
| 8 | News & Calendar | agents/standby.py | standby — add NewsAPI key |
| 9 | Social Sentiment | agents/standby.py | standby — add Reddit key |
| 10| NLP Tone | agents/standby.py | standby — add Claude key |

Standby agents return neutral HOLD (they abstain) until you wire their API keys —
they don't distort voting in the meantime.

## How a decision is made
1. All agents vote BUY / SELL / HOLD with confidence (`master_brain.collect_votes`).
2. HOLD = abstain. Among directional voters, need weighted agreement ≥ 70% AND
   enough conviction to trade; 60–70% = half position; else HOLD.
3. Risk Manager runs last and can **BLOCK** (daily loss limit, max trades, drawdown).
4. `backtest.py` opens/closes paper trades with SL/TP and reports stats.

## Honest note on results
On synthetic (random) data there is no edge, so a 2:1 stop/target loses — that's
expected and correct. Real results come only from real data + months of paper
trading. Anyone showing guaranteed profit is not being straight with you.

## Files
```
alphabot/
├── data_loader.py         free OHLC (Yahoo), synthetic fallback
├── master_brain.py        voting + consensus + decision
├── backtest.py            walk-forward paper simulation
├── run_bot.py             single decision snapshot
├── agents/                the 10 agents + shared base (Vote, EMA/RSI/ATR)
├── requirements.txt
└── README.md
```

## Next steps
- Run both scripts on real data, read the votes, get a feel for it.
- Wire NewsAPI / Reddit / Claude keys to activate agents 8–10.
- Replace the momentum agent with a trained XGBoost model.
- Add a broker for real paper execution — Alpaca works from India.
Build and test one change at a time.
