"""Agent — COT Institutional Positioning (free CFTC data, no key needed).

Every Friday the CFTC publishes what hedge funds / large speculators hold in
EUR futures (data as of the preceding Tuesday). This is the one free window
into institutional positioning. We vote WITH the big-money trend:

- specs net long AND adding  -> BUY
- specs net short AND adding -> SELL
- otherwise abstain.

No look-ahead: a report dated Tuesday is only used from the following Friday
(release day). Data cached to models/cot_eur.pkl for 3 days; on any fetch
failure the agent abstains.
"""
import os
import json
import time
import pickle
import datetime
import urllib.request

from .base import Vote

_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "cot_eur.pkl")
_URL = ("https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        "?$select=report_date_as_yyyy_mm_dd,noncomm_positions_long_all,"
        "noncomm_positions_short_all"
        "&$where=market_and_exchange_names='EURO FX - CHICAGO MERCANTILE EXCHANGE'"
        "&$order=report_date_as_yyyy_mm_dd&$limit=5000")
_mem = {"rows": None, "loaded": 0}


def _load():
    """[(available_from_date, net_position), ...] sorted by date."""
    now = time.time()
    if _mem["rows"] is not None and now - _mem["loaded"] < 3600:
        return _mem["rows"]
    rows = None
    try:
        if os.path.exists(_CACHE_PATH) and now - os.path.getmtime(_CACHE_PATH) < 3 * 86400:
            with open(_CACHE_PATH, "rb") as fh:
                rows = pickle.load(fh)
    except Exception:
        rows = None
    if rows is None:
        try:
            req = urllib.request.Request(_URL.replace(" ", "%20").replace("'", "%27"),
                                         headers={"User-Agent": "Mozilla/5.0"})
            raw = json.load(urllib.request.urlopen(req, timeout=15))
            rows = []
            for r in raw:
                d = datetime.date.fromisoformat(r["report_date_as_yyyy_mm_dd"][:10])
                net = int(float(r["noncomm_positions_long_all"])) - \
                      int(float(r["noncomm_positions_short_all"]))
                rows.append((d + datetime.timedelta(days=3), net))  # Tue data -> Fri release
            rows.sort()
            os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
            with open(_CACHE_PATH, "wb") as fh:
                pickle.dump(rows, fh)
        except Exception:
            rows = []
    _mem["rows"] = rows
    _mem["loaded"] = now
    return rows


def analyze(df, ctx=None):
    sym = (ctx or {}).get("symbol", "EURUSD=X")
    if not sym.startswith("EURUSD"):
        return Vote("COT Institutional", "HOLD", 53, "COT wired for EURUSD only")
    rows = _load()
    if not rows:
        return Vote("COT Institutional", "HOLD", 53, "COT data unavailable")

    ts = df.index[-1]
    try:
        today = ts.tz_convert("UTC").date()
    except Exception:
        today = ts.date() if hasattr(ts, "date") else datetime.date.today()

    hist = [n for d, n in rows if d <= today]
    if len(hist) < 5:
        return Vote("COT Institutional", "HOLD", 53, "COT history too short")
    net, prev = hist[-1], hist[-4]           # ~3 weeks back
    building = net - prev
    if net > 0 and building > 0:
        return Vote("COT Institutional", "BUY", 64,
                    f"specs net long {net/1000:.0f}k & adding")
    if net < 0 and building < 0:
        return Vote("COT Institutional", "SELL", 64,
                    f"specs net short {net/1000:.0f}k & adding")
    return Vote("COT Institutional", "HOLD", 55,
                f"specs mixed (net {net/1000:.0f}k)")
