"""
news_feed.py — free live financial headlines (no API key).

Pulls Yahoo Finance RSS for a symbol and tags each headline with a naive
bullish/bearish tone from keyword matching. Display-only helper for the
dashboard; the News agent voting logic is unchanged (still standby) so
backtests stay honest.
"""
import time
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape

_BULL = ["rise", "rises", "rally", "gains", "surge", "jumps", "up ", "higher",
         "strong", "boost", "beats", "optimism", "hawkish", "growth", "recovery",
         "bullish", "soars", "climbs", "rebound"]
_BEAR = ["fall", "falls", "drop", "drops", "slump", "plunge", "sinks", "down ",
         "lower", "weak", "misses", "fears", "dovish", "recession", "cuts",
         "bearish", "tumbles", "slides", "selloff", "crisis"]

_CACHE = {}
_TTL = 120


def _tone(title):
    t = title.lower()
    b = sum(1 for w in _BULL if w in t)
    s = sum(1 for w in _BEAR if w in t)
    if b > s:
        return "bull"
    if s > b:
        return "bear"
    return "neutral"


def get_news(symbol="EURUSD=X", limit=12):
    now = time.time()
    hit = _CACHE.get(symbol)
    if hit and now - hit[0] < _TTL:
        return hit[1]

    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?"
           f"s={symbol}&region=US&lang=en-US")
    result = {"symbol": symbol, "items": [], "bull": 0, "bear": 0, "neutral": 0,
              "tone": "neutral", "error": None}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            root = ET.fromstring(r.read())
        for it in root.findall(".//item")[:limit]:
            title = unescape(it.findtext("title") or "").strip()
            if not title:
                continue
            tone = _tone(title)
            result[tone] += 1
            pub = (it.findtext("pubDate") or "").strip()
            result["items"].append({
                "title": title,
                "link": (it.findtext("link") or "").strip(),
                "time": pub[:22],
                "tone": tone,
            })
        if result["bull"] > result["bear"]:
            result["tone"] = "bull"
        elif result["bear"] > result["bull"]:
            result["tone"] = "bear"
    except Exception as e:
        result["error"] = str(e)

    _CACHE[symbol] = (now, result)
    return result


if __name__ == "__main__":
    r = get_news()
    print(f"{r['symbol']}  tone={r['tone']}  "
          f"(bull {r['bull']} / bear {r['bear']} / neutral {r['neutral']})")
    for it in r["items"][:6]:
        print(f"  [{it['tone']:<7}] {it['title']}")
