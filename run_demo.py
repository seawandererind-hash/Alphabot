"""
run_demo.py — see Agent 1 in action
-----------------------------------
Loads candles and shows the Market Structure agent's vote at several points,
walk-forward style (only past candles visible each step — no future peeking).

Run:  python run_demo.py
"""

from data_loader import get_candles
from agents.market_structure import analyze


def main():
    df = get_candles(symbol="EURUSD=X", period="60d", interval="1h")
    print(f"\nLoaded {len(df)} candles. Walking forward...\n")
    print(f"{'time':<20}{'price':<10}{'vote':<7}{'conf':<7}reason")
    print("-" * 70)

    # step through the last ~15 readings, each seeing only candles up to that point
    start = max(30, len(df) - 15)
    for i in range(start, len(df)):
        window = df.iloc[:i + 1]              # past-only slice
        v = analyze(window)
        ts = str(window.index[-1])[:16]
        px = round(float(window["close"].iloc[-1]), 5)
        print(f"{ts:<20}{px:<10}{v.signal:<7}{v.confidence:<7}{v.reason}")

    print("\nDone. This is ONE agent. In the full bot, 10 agents vote and the")
    print("Master Brain only trades when 7+ agree and the Risk agent approves.")


if __name__ == "__main__":
    main()
