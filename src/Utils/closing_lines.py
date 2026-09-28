"""
closing_lines.py
================
Closing prices for a MEMBER's own bets, read from our odds archive
(odds_snapshots). Backs POST /api/closing-lines/lookup, which the /bets page
calls to put closing line value next to each bet a member logged.

WHAT THIS ANSWERS. "The book you bet at: what was its last price on your side
before tip-off, and did you beat it?" Nothing else. It never looks at our
model, and a member's CLV is a fact about their prices, not a profit figure.

WHOSE CLOSE. The SAME book the member names, never a better or worse price
from a book they did not bet at. A member at DraftKings compared with
FanDuel's close would be measuring the gap between two books, not whether
they beat the market they were in. A book the recorder does not carry gets
"book_not_recorded" and the list of books that do have a close for that game,
so the page can say so instead of quietly swapping one in.

HOW CLOSE IS "CLOSING". The recorders write a row only when a book's numbers
change (see odds_board.py), so the latest row before tip is that book's price
at tip only if nobody could have seen it change afterwards. What we can prove
travels with the number as `confirmed_at` / `minutes_before_tip`:

  * the heartbeat (odds_seen) saw this market listed after the row was written
    and at or before tip -> confirmed at that sighting;
  * it saw it listed AFTER tip and the book's first post-tip row (if any) left
    this market unchanged -> the price held through tip, confirmed at tip;
  * otherwise -> confirmed only when the row was written (captured_at).

A price last confirmed eight hours before tip is not a closing price, and the
page must say how far out it was rather than call it one. This module does
not decide what is "close enough"; it reports the distance.

SPREADS AND TOTALS. Price CLV is only meaningful at the same number: -110 on
-3.5 and -110 on -2.5 are different bets. When the member's number matches
the close, CLV is the price comparison; when it does not, the result is
"line_moved" with the difference in points (positive = the member's number
was better) and no percentage.

ONLY AFTER TIP. A game that has not started has no close, whatever the newest
row says. Unknown is None, never 0.
"""

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.Utils.odds_board import MARKETS, market_history, parse_ts

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - no tz database on this machine
    _ET = None

#: A member's market -> the board's market key. Props, parlays and "other"
#: have no archived close (the recorder captures game lines only).
_MARKET_KEY = {"moneyline": "ml", "spread": "spread", "total": "total"}

#: Which side of which market reads which columns: (line column, price column).
#: The away spread is the home spread negated -- one number per game.
_SIDE_FIELDS = {
    ("ml", "home"): (None, "home_ml"),
    ("ml", "away"): (None, "away_ml"),
    ("spread", "home"): ("spread_home", "spread_home_price"),
    ("spread", "away"): ("spread_home", "spread_away_price"),
    ("total", "over"): ("ou_line", "ou_over_price"),
    ("total", "under"): ("ou_line", "ou_under_price"),
}

#: Names members type -> the recorder's book key (after _norm_book). The
#: recorder stores The Odds API's keys; a few books go by other names on a
#: slip. Anything not listed is compared as typed (lowercase, letters/digits).
_BOOK_ALIASES = {
    "dk": "draftkings",
    "fd": "fanduel",
    "mgm": "betmgm",
    "caesarssportsbook": "caesars",
    "williamhill": "caesars",
    "williamhillus": "caesars",
    "betonline": "betonlineag",
    "lowvigag": "lowvig",
    "betriversny": "betrivers",
    "betrivers": "betrivers",
    "espn": "espnbet",
    "fanaticssportsbook": "fanatics",
    "wynnbet": "wynn",
    "hardrock": "hardrockbet",
}

MAX_ITEMS = 500


def _norm_book(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if not key:
        return None
    return _BOOK_ALIASES.get(key, key)


def _norm_team(name: Optional[str]) -> str:
    # One Clippers spelling: the prediction path writes "LA Clippers", The
    # Odds API "Los Angeles Clippers" (same rule as grade_predictions.price_clv).
    s = re.sub(r"\s+", " ", str(name or "")).strip().casefold()
    return "la clippers" if s == "los angeles clippers" else s


def american_to_decimal(price: Any) -> Optional[float]:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if -100.0 < p < 100.0:
        return None
    return 1.0 + (p / 100.0 if p > 0 else 100.0 / -p)


def _et_date(ts: datetime) -> str:
    if _ET is not None:
        return ts.astimezone(_ET).date().isoformat()
    return (ts - timedelta(hours=5)).date().isoformat()  # EST; off only for late spring tips


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _heartbeat_last_seen(conn) -> Dict[Tuple[str, str, str], datetime]:
    """(game_key, book, market) -> latest sighting, or {} before the heartbeat existed."""
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = 'odds_seen'")}
    if not have:
        return {}
    out: Dict[Tuple[str, str, str], datetime] = {}
    for game_key, book, market, last in conn.execute(
            "SELECT game_key, sportsbook, market, last_seen_at FROM odds_seen WHERE sport = 'NBA'"):
        ts = parse_ts(last)
        if ts is None:
            continue
        k = (game_key, book, market)
        if k not in out or ts > out[k]:
            out[k] = ts
    return out


def _provenance_rank(row) -> int:
    # Where two rows share a timestamp the one we watched must be the latest,
    # so it sorts last (same rule as grade_predictions.price_clv).
    try:
        prov = row["provenance"] or "observed"
    except (IndexError, KeyError):
        prov = "observed"
    return 1 if prov == "observed" else 0


class _Archive:
    """The NBA rows of odds_snapshots, grouped once per request."""

    def __init__(self, conn):
        self.games: Dict[str, Dict[str, Any]] = {}
        first_tip: Optional[datetime] = None
        for r in conn.execute(
                "SELECT * FROM odds_snapshots WHERE sport = 'NBA' "
                "AND game_start_time_utc IS NOT NULL"):
            ts = parse_ts(r["captured_at"])
            if ts is None:
                continue
            g = self.games.setdefault(r["game_key"], {"rows": [], "books": {}})
            g["rows"].append(r)
            g["books"].setdefault(r["sportsbook"], []).append(r)
        self.by_teams: Dict[frozenset, List[str]] = {}
        for key, g in self.games.items():
            for rows in g["books"].values():
                rows.sort(key=lambda r: (parse_ts(r["captured_at"]), _provenance_rank(r), r["id"]))
            # Tip from the newest row of any book: a rescheduled game is
            # re-stamped only by books whose prices moved afterwards.
            newest = max(g["rows"], key=lambda r: (parse_ts(r["captured_at"]), r["id"]))
            g["tip"] = parse_ts(newest["game_start_time_utc"])
            g["home_team"], g["away_team"] = newest["home_team"], newest["away_team"]
            if g["tip"] is not None:
                first_tip = g["tip"] if first_tip is None or g["tip"] < first_tip else first_tip
            teams = frozenset((_norm_team(newest["home_team"]), _norm_team(newest["away_team"])))
            self.by_teams.setdefault(teams, []).append(key)
        self.first_tip = first_tip
        self.last_seen = _heartbeat_last_seen(conn)

    def find_game(self, game_date: str, team_a: str, team_b: str) -> Optional[str]:
        """The game between these two teams on this US Eastern date, either orientation."""
        teams = frozenset((_norm_team(team_a), _norm_team(team_b)))
        for key in self.by_teams.get(teams, []):
            tip = self.games[key]["tip"]
            if tip is not None and _et_date(tip) == game_date:
                return key
        return None


def _confirmed_at(archive: _Archive, game_key: str, book: str, market: str,
                  pre: List[Any], post: List[Any], tip: datetime) -> Tuple[datetime, str]:
    """When we last KNEW this book's pre-tip price for this market still stood."""
    recorded = parse_ts(pre[-1]["captured_at"])
    seen = archive.last_seen.get((game_key, book, market))
    if seen is not None and recorded <= seen <= tip:
        return seen, "heartbeat"
    if seen is not None and seen > tip:
        fields = MARKETS[market]
        before = tuple(pre[-1][f] for f in fields)
        if not post or tuple(post[0][f] for f in fields) == before:
            return tip, "heartbeat"
    return recorded, "captured"


def lookup_one(archive: _Archive, item: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": item.get("id"), "status": None, "close": None,
                           "clv_pct": None, "line_diff": None, "game": None,
                           "books_with_close": []}

    market = _MARKET_KEY.get(str(item.get("market") or "").lower())
    if market is None:
        out["status"] = "unsupported_market"
        return out
    game_date = str(item.get("game_date") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", game_date) or not item.get("home_team") \
            or not item.get("away_team"):
        out["status"] = "game_unknown"
        return out

    key = archive.find_game(game_date, item["home_team"], item["away_team"])
    if key is None:
        out["status"] = "no_game"
        out["archive_first_tip"] = archive.first_tip.isoformat() if archive.first_tip else None
        return out
    g = archive.games[key]
    tip = g["tip"]
    out["game"] = {"game_key": key, "home_team": g["home_team"], "away_team": g["away_team"],
                   "tip_utc": tip.isoformat()}
    if tip > now:
        out["status"] = "not_started"
        return out

    # Books that closed this market with a price, for the "we don't carry your
    # book" message. Listed, never substituted.
    closed_books = []
    for b, rows in g["books"].items():
        pre = [r for r in rows if parse_ts(r["captured_at"]) < tip]
        if pre and market_history(pre, market)["current"] is not None:
            closed_books.append(b)
    out["books_with_close"] = sorted(closed_books)

    book = _norm_book(item.get("sportsbook"))
    if book is None:
        out["status"] = "no_book"
        return out
    rows = next((rs for b, rs in g["books"].items() if _norm_book(b) == book), None)
    pre = [r for r in rows or [] if parse_ts(r["captured_at"]) < tip]
    if not pre:
        out["status"] = "book_not_recorded"
        return out
    current = market_history(pre, market)["current"]
    if current is None:
        out["status"] = "no_market"
        return out

    side = str(item.get("side") or "").lower()
    if side in ("home", "away") and _norm_team(item["home_team"]) != _norm_team(g["home_team"]):
        # The member has the teams the other way round. Their side names a
        # TEAM, so it follows the team to wherever the archive has it.
        side = "away" if side == "home" else "home"
    fields = _SIDE_FIELDS.get((market, side))
    if fields is None:
        out["status"] = "side_unknown"
        return out
    line_col, price_col = fields
    close_price = _num(current.get(price_col))
    close_line = _num(current.get(line_col)) if line_col else None
    if line_col == "spread_home" and side == "away" and close_line is not None:
        close_line = -close_line
    if close_price is None or (line_col and close_line is None):
        out["status"] = "no_market"
        return out

    post = [r for r in rows if parse_ts(r["captured_at"]) >= tip]
    confirmed, how = _confirmed_at(archive, key, pre[-1]["sportsbook"], market, pre, post, tip)
    hist = market_history(pre, market)
    try:
        provenance = pre[-1]["provenance"] or "observed"
    except (IndexError, KeyError):
        provenance = "observed"
    out["close"] = {
        "book": pre[-1]["sportsbook"],
        "price": int(round(close_price)),
        "line": close_line,
        "since": hist["since"],
        "confirmed_at": confirmed.isoformat(),
        "confirmed_by": how,
        "minutes_before_tip": round((tip - confirmed).total_seconds() / 60.0, 1),
        "provenance": provenance,
    }

    taken = american_to_decimal(item.get("odds_american"))
    closing = american_to_decimal(close_price)
    if taken is None or closing is None:
        out["status"] = "bad_odds"
        return out

    if line_col:
        member_line = _num(item.get("line"))
        if member_line is None:
            out["status"] = "line_unknown"
            return out
        if abs(member_line - close_line) > 1e-9:
            # Points, signed so positive = the member's number was better:
            # more points on a spread or an under, fewer on an over.
            diff = member_line - close_line if side != "over" else close_line - member_line
            out["line_diff"] = round(diff, 1)
            out["status"] = "line_moved"
            return out

    out["clv_pct"] = round((taken / closing - 1.0) * 100.0, 2)
    out["status"] = "priced"
    return out


def lookup(conn, items: Iterable[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    items = list(items)[:MAX_ITEMS]
    archive = _Archive(conn)
    return {
        "results": [lookup_one(archive, it if isinstance(it, dict) else {}, now) for it in items],
        "archive_first_tip": archive.first_tip.isoformat() if archive.first_tip else None,
        "heartbeat": bool(archive.last_seen),
    }
