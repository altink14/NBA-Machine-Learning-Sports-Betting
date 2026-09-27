"""
odds_board.py
=============
Read side of the odds archive: the Line Shop board and every book's line
movement, built from odds_snapshots.

WHY THIS EXISTS (2026-09-27). The recorders (odds_api_client.write_snapshot_rows
and main_api.snapshot_odds) write a row ONLY when a book's numbers change. The
Line Shop used to keep a quote only if its row was captured in the last 7 days,
so a price that simply had not moved for a week vanished: opening night was
missing and the board showed 17 of the ~41 games the archive holds. Its "as of
6d ago" stamp really meant "last changed 6 days ago". Line Movement had the
same blind spot from the other side: a 48-hour capture window, FanDuel only,
and "open" meant the first change inside that window.

The recorders are right to write on change (it keeps the archive small and
every row is a real move), so the fix is here, on the read side:

  * A book's latest row IS its current price until the next row replaces it,
    however old that row is. It stays on the board while the game is upcoming.
  * Each market carries "unchanged since": the first capture of the trailing
    run of identical numbers. That is the honest label, not "as of".
  * "Open" is the book's first capture of that market for that game, over the
    whole archive, never a window.

WHAT DROPS, and the one thing the archive cannot see:

  * games whose start time has passed (history, not a shopping board);
  * legacy rows with no start time (the old scraper; they were WNBA games
    mis-tagged NBA);
  * a book whose latest row carries no price at all (it took every market
    down while still listing the game - the recorder writes that as a change);
  * a single market whose latest value is empty (e.g. the book pulled its
    moneyline but kept the spread).
  A book that stops listing a game ENTIRELY writes nothing, so its last price
  would stay on the board. The archive cannot tell that apart from a price
  that has not moved; the page says so rather than pretending otherwise.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.Utils import Parlay as parlay
from src.Utils import devig

#: The numbers that make up each market, in the order they are compared.
MARKETS: Dict[str, Tuple[str, ...]] = {
    "ml": ("home_ml", "away_ml"),
    "spread": ("spread_home", "spread_home_price", "spread_away_price"),
    "total": ("ou_line", "ou_over_price", "ou_under_price"),
}
_ALL_FIELDS = [f for fields in MARKETS.values() for f in fields]


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """
    ISO string -> aware UTC datetime. The archive holds both '...+00:00'
    (The Odds API recorder) and naive UTC (the legacy scraper's utcnow()), and
    'Z' starts; string comparison across those formats is not safe.
    """
    if not value:
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def _get(row, field):
    try:
        return row[field]
    except (IndexError, KeyError):
        return None  # a pre-migration archive without the spread/price columns


def _values(row, market: str) -> Tuple:
    return tuple(_get(row, f) for f in MARKETS[market])


def market_history(rows: List[Any], market: str) -> Dict[str, Any]:
    """
    One book, one game, one market, rows oldest first. Returns the current
    numbers, when they were first seen unchanged, the first numbers ever
    captured, and how many times they changed. Empty = None, never 0.
    """
    fields = MARKETS[market]
    now = _values(rows[-1], market)
    current = None if all(v is None for v in now) else dict(zip(fields, now))

    since = None
    if current is not None:
        # Walk back over the trailing run of identical numbers. A row written
        # because a DIFFERENT market moved still carries this market's numbers,
        # so the run spans it and "since" stays the date this price appeared.
        since = rows[-1]["captured_at"]
        for r in reversed(rows[:-1]):
            if _values(r, market) != now:
                break
            since = r["captured_at"]

    opened = None
    for r in rows:
        v = _values(r, market)
        if any(x is not None for x in v):
            opened = {**dict(zip(fields, v)), "captured_at": r["captured_at"]}
            break

    changes = 0
    prev = None
    for r in rows:
        v = _values(r, market)
        if all(x is None for x in v):
            continue
        if prev is not None and v != prev:
            changes += 1
        prev = v

    return {"current": current, "since": since, "open": opened, "changes": changes}


def _fair(quotes: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Median of per-book Shin de-vigs of the moneyline pair."""
    probs = []
    for q in quotes.values():
        hm, am = q["home_ml"], q["away_ml"]
        if hm is None or am is None:
            continue
        try:
            dec = [parlay.american_to_true_decimal(float(hm)),
                   parlay.american_to_true_decimal(float(am))]
            probs.append(devig.fair_probs(dec)[0])
        except (ValueError, TypeError):
            continue
    if not probs:
        return None
    probs.sort()
    n = len(probs)
    median = probs[n // 2] if n % 2 else (probs[n // 2 - 1] + probs[n // 2]) / 2
    return {
        "home_prob": round(median, 4),
        "away_prob": round(1 - median, 4),
        "home_ml_fair": _american(1 / median),
        "away_ml_fair": _american(1 / (1 - median)),
        "books_used": n,
    }


def _best(quotes: Dict[str, Dict[str, Any]], side: str, fair: Optional[Dict[str, Any]]):
    candidates = [(book, q[side]) for book, q in quotes.items() if q[side] is not None]
    if not candidates:
        return None
    book, price = max(candidates, key=lambda x: x[1])  # higher American = better payout
    entry: Dict[str, Any] = {"book": book, "price": price}
    if fair:
        prob = fair["home_prob"] if side == "home_ml" else fair["away_prob"]
        dec = parlay.american_to_true_decimal(float(price))
        entry["ev_pct_at_fair"] = round((prob * dec - 1) * 100, 2)
    return entry


def build_board(conn, sport: str = "NBA", now: Optional[datetime] = None,
                sportsbook: Optional[str] = None) -> Dict[str, Any]:
    """
    Every upcoming game with each book's current quote, its "unchanged since"
    dates, and its open (first capture) per market, plus the de-vigged fair
    moneyline and the best price on each side.
    """
    now = now or datetime.now(timezone.utc)
    params: List[Any] = [sport]
    where = "sport = ? AND game_start_time_utc IS NOT NULL"
    if sportsbook:
        where += " AND sportsbook = ?"
        params.append(sportsbook)
    rows = conn.execute(f"SELECT * FROM odds_snapshots WHERE {where}", params).fetchall()

    archive_since = latest_change = None
    grouped: Dict[str, Dict[str, List[Any]]] = {}
    for r in rows:
        ts = parse_ts(r["captured_at"])
        if ts is None:
            continue
        archive_since = ts if archive_since is None or ts < archive_since else archive_since
        latest_change = ts if latest_change is None or ts > latest_change else latest_change
        grouped.setdefault(r["game_key"], {}).setdefault(r["sportsbook"], []).append(r)

    dropped = {"started": 0, "no_price": 0}
    games = []
    for game_key, by_book in grouped.items():
        for book_rows in by_book.values():
            book_rows.sort(key=lambda r: (parse_ts(r["captured_at"]), r["id"]))
        # The start time from the most recent row of any book: a rescheduled
        # game is re-stamped only by books whose prices moved afterwards.
        newest = max((br[-1] for br in by_book.values()), key=lambda r: parse_ts(r["captured_at"]))
        start = parse_ts(newest["game_start_time_utc"])
        if start is None or start <= now:
            dropped["started"] += 1
            continue

        quotes: Dict[str, Dict[str, Any]] = {}
        last_change = None
        for book, br in by_book.items():
            latest = br[-1]
            hist = {m: market_history(br, m) for m in MARKETS}
            if all(hist[m]["current"] is None for m in MARKETS):
                dropped["no_price"] += 1
                continue
            q: Dict[str, Any] = {f: None for f in _ALL_FIELDS}
            for m in MARKETS:
                if hist[m]["current"]:
                    q.update(hist[m]["current"])
            q.update({
                # When ANY of this book's numbers for the game last changed.
                "last_change_at": latest["captured_at"],
                "since": {m: hist[m]["since"] for m in MARKETS},
                "open": {m: hist[m]["open"] for m in MARKETS},
                "changes": {m: hist[m]["changes"] for m in MARKETS},
                "first_seen_at": br[0]["captured_at"],
                "provenance": _get(latest, "provenance") or "observed",
            })
            quotes[book] = q
            ts = parse_ts(latest["captured_at"])
            last_change = ts if last_change is None or ts > last_change else last_change
        if not quotes:
            continue

        fair = _fair(quotes)
        games.append({
            "game_key": game_key,
            "home_team": newest["home_team"],
            "away_team": newest["away_team"],
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_change_at": last_change.isoformat() if last_change else None,
            "books": quotes,
            "fair": fair,
            "best": {"home_ml": _best(quotes, "home_ml", fair),
                     "away_ml": _best(quotes, "away_ml", fair)},
        })

    games.sort(key=lambda g: (g["start"], g["game_key"]))
    return {
        "sport": sport,
        "games": games,
        "books": sorted({bk for g in games for bk in g["books"]}),
        "devig_method": devig.ACTIVE_METHOD,
        "archive_since": archive_since.isoformat() if archive_since else None,
        "latest_change_at": latest_change.isoformat() if latest_change else None,
        "dropped": dropped,
        "generated_at": now.isoformat(),
    }
