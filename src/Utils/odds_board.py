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
    moneyline but kept the spread);
  * a market the book STOPPED LISTING (the heartbeat, below).

THE HEARTBEAT (2026-09-27). A book that stops listing a game writes no
snapshot row, so until now its last price stayed on the board forever. Each
poll now records what it saw (odds_polls / odds_seen, written by
odds_api_client.record_poll). For each quoted market the board asks: has a
successful WHOLE-BOARD poll that could have listed this book and market run
since we last knew it was offered (its latest row or its last sighting,
whichever is later), without listing it? If so the market leaves the board
and fair/best, and is reported under the game's "withdrawn" with its last
sighting and the first poll that missed it. Otherwise it stays, labelled:

  "confirmed"    a poll has seen it priced; last_seen_at says when.
  "unconfirmed"  no poll has looked since this price was recorded: rows from
                 before the heartbeat existed, or a book only the SBR scraper
                 reads. Today's rule applies (the latest row is the price),
                 and the label says it has not been re-checked.

A failed or empty poll is never evidence (record_poll refuses to write an 'ok'
poll with no sightings), and the SBR scraper's polls never are either: it
reads one book for one date and reports a failed request as "no games".
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


class _Heartbeat:
    """What the recorders' polls saw, loaded once per board."""

    def __init__(self, conn, sport: str):
        # (time, source, books asked for or None = all, markets) per ok whole-board poll
        self.polls: List[Tuple[datetime, str, Optional[frozenset], frozenset]] = []
        self.seen: Dict[Tuple[str, str, str], datetime] = {}
        self.books_by_source: Dict[str, set] = {}
        self.first_poll_at: Optional[datetime] = None
        self.last_ok_at: Optional[datetime] = None
        self.last_poll: Optional[Dict[str, Any]] = None
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('odds_polls', 'odds_seen')")}
        if have != {"odds_polls", "odds_seen"}:
            return  # an archive from before the heartbeat: every quote unconfirmed
        for polled_at, source, status, covers, books, markets in conn.execute(
                "SELECT polled_at, source, status, covers_board, books, markets "
                "FROM odds_polls WHERE sport = ?", (sport,)):
            ts = parse_ts(polled_at)
            if ts is None:
                continue
            if self.first_poll_at is None or ts < self.first_poll_at:
                self.first_poll_at = ts
            if self.last_poll is None or ts > self.last_poll["_ts"]:
                self.last_poll = {"_ts": ts, "at": polled_at, "status": status, "source": source}
            if status != "ok":
                continue
            if self.last_ok_at is None or ts > self.last_ok_at:
                self.last_ok_at = ts
            if covers:
                self.polls.append((
                    ts, source,
                    frozenset(b.strip() for b in books.split(",")) if books else None,
                    frozenset(m.strip() for m in (markets or "").split(","))))
        self.polls.sort(key=lambda p: p[0])
        for source, game_key, book, market, last in conn.execute(
                "SELECT source, game_key, sportsbook, market, last_seen_at FROM odds_seen "
                "WHERE sport = ?", (sport,)):
            ts = parse_ts(last)
            if ts is None:
                continue
            self.books_by_source.setdefault(source, set()).add(book)
            key = (game_key, book, market)
            if key not in self.seen or ts > self.seen[key]:
                self.seen[key] = ts

    def _covers(self, poll, book: str, market: str) -> bool:
        _, source, books, markets = poll
        if market not in markets:
            return False
        if books is not None:
            return book in books
        # "Every book the source carries" = the books it has ever shown us. A
        # name only another feed uses (SBR's 'caesars' is the Odds API's
        # 'williamhill_us') is never covered, so a feed that could not have
        # listed a book never pulls it.
        return book in self.books_by_source.get(source, ())

    def check(self, game_key: str, book: str, market: str, recorded_at: datetime):
        """(status, last_seen_at, missing_since) for one quoted market."""
        last_seen = self.seen.get((game_key, book, market))
        known = max(recorded_at, last_seen) if last_seen else recorded_at
        for poll in self.polls:  # oldest first, so this is the FIRST poll that missed it
            if poll[0] > known and self._covers(poll, book, market):
                return "withdrawn", last_seen, poll[0]
        return ("confirmed" if last_seen else "unconfirmed"), last_seen, None

    def summary(self) -> Dict[str, Any]:
        last = None
        if self.last_poll:
            last = {k: v for k, v in self.last_poll.items() if k != "_ts"}
        return {
            "since": _iso(self.first_poll_at),
            "last_ok_poll_at": _iso(self.last_ok_at),
            "last_poll": last,
        }


def _iso(d: Optional[datetime]) -> Optional[str]:
    return d.isoformat() if d else None


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

    beat = _Heartbeat(conn, sport)
    dropped = {"started": 0, "no_price": 0, "withdrawn": 0, "withdrawn_markets": 0}
    games = []
    withdrawn_games = []
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
        withdrawn: Dict[str, Dict[str, Any]] = {}
        last_change = None
        for book, br in by_book.items():
            latest = br[-1]
            hist = {m: market_history(br, m) for m in MARKETS}
            if all(hist[m]["current"] is None for m in MARKETS):
                dropped["no_price"] += 1
                continue
            recorded_at = parse_ts(latest["captured_at"])
            feed: Dict[str, Optional[str]] = {m: None for m in MARKETS}
            last_seen: Dict[str, Optional[str]] = {m: None for m in MARKETS}
            for m in MARKETS:
                if hist[m]["current"] is None:
                    continue
                status, seen_at, missing = beat.check(game_key, book, m, recorded_at)
                if status == "withdrawn":
                    # The book stopped listing it: not a price anyone can bet.
                    withdrawn.setdefault(book, {})[m] = {
                        **hist[m]["current"],
                        "last_seen_at": _iso(seen_at) or latest["captured_at"],
                        "missing_since": _iso(missing),
                    }
                    dropped["withdrawn_markets"] += 1
                    hist[m] = {**hist[m], "current": None, "since": None}
                    continue
                feed[m], last_seen[m] = status, _iso(seen_at)
            if all(hist[m]["current"] is None for m in MARKETS):
                dropped["withdrawn"] += 1
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
                # Heartbeat per market: 'confirmed' (a poll saw it; last_seen_at
                # says when) or 'unconfirmed' (no poll has looked since this
                # price was recorded). None where the market has no price.
                "feed": feed,
                "last_seen_at": last_seen,
            })
            quotes[book] = q
            ts = parse_ts(latest["captured_at"])
            last_change = ts if last_change is None or ts > last_change else last_change
        if not quotes:
            if withdrawn:
                # Every book stopped listing it (postponed, or pulled): say so
                # rather than letting the game vanish without a trace.
                withdrawn_games.append({"game_key": game_key, "home_team": newest["home_team"],
                                        "away_team": newest["away_team"],
                                        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        "withdrawn": withdrawn})
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
            # Markets a book stopped listing; kept out of fair and best.
            "withdrawn": withdrawn,
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
        "heartbeat": beat.summary(),
        "withdrawn_games": withdrawn_games,
        "generated_at": now.isoformat(),
    }
