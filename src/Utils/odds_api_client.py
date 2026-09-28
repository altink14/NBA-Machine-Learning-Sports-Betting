"""
odds_api_client.py
==================
The Odds API (the-odds-api.com) -> odds_snapshots archive.

WHY THIS EXISTS. Every blocked money-layer feature (CLV Time Machine, Line
Shop, Book Softness) needs a season's worth of odds snapshots across multiple
books, and closing lines are only observable live - this data cannot be
bought back later. This module is the tape recorder: one official API call
per run captures moneylines, spreads and totals for every NBA game on the
board, across every US book the API carries, into the same odds_snapshots
table the line-movement endpoint already reads.

SCHEMA. odds_snapshots gains five additive columns on first write (spread and
the prices on both sides of spread/total); every existing reader keeps
working because the original columns are untouched.

QUOTA. The API is credit-metered (free tier: 500/month). One call with three
markets in one region costs 3 credits, so the daily 9 AM snapshot costs ~90
credits/month - safely inside the free tier. The in-season cadence (every
30-60 min, snapshot_odds_api.py --loop) needs the paid tier; the quota
headers are logged on every call so drift is visible in daily_update.log.

KEY. Set ODDS_API_KEY in the environment. No key -> callers get a clear
error, never a silent no-op that leaves the archive empty while looking fine.

HEARTBEAT (2026-09-27). odds_snapshots is written on CHANGE, so a book that
stops listing a game writes nothing and its last price looked current on the
Line Shop forever. Every poll now also leaves a heartbeat, written separately
from the snapshot rows (which are unchanged):

  odds_polls  one row per poll, success OR failure: when, which source, which
              books and markets it asked for, and whether it saw the source's
              WHOLE board (covers_board). Only an 'ok' whole-board poll is
              evidence that something missing from it is gone. A failed or
              empty poll proves nothing, so it can never pull a quote: that is
              the difference between "we could not look" and "the book took
              it down" (the silent-fallback hazard, again).
  odds_seen   per (source, game, book, market): first and last time a poll
              saw that market priced. Upserted, one row per key.

The poll row and its odds_seen upserts land in ONE transaction, so a poll can
never exist without the sightings that go with it (that would read as "every
book pulled every game"). The read rule lives in src/Utils/odds_board.py.
"""

import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_REGIONS = "us"

_EXTRA_COLUMNS = [
    ("spread_home", "REAL"),          # home side handicap (negative = home favored)
    ("spread_home_price", "REAL"),    # American juice on the home spread
    ("spread_away_price", "REAL"),
    ("ou_over_price", "REAL"),        # American juice on the over/under
    ("ou_under_price", "REAL"),
    # How we came to know this price. 'observed' = this recorder read it off
    # the live board at captured_at; 'reconstructed' = pulled out of The Odds
    # API's historical archive after a missed run. captured_at means "when the
    # price was true" either way, and provenance is the difference between
    # "we watched this" and "a vendor says it existed".
    #
    # The NFL side has carried this since it was built. The NBA -- which is
    # the product -- did not, so a repaired snapshot would have been
    # indistinguishable from a watched one in exactly the table the public
    # track record's CLV is computed from. Added 2026-09-21, before the
    # season, while there is nothing to get wrong.
    ("provenance", "TEXT NOT NULL DEFAULT 'observed'"),
]

#: Every row already in the table was written by the live recorder, so
#: 'observed' is the truth for all of them, not a guess. Only rows written
#: before the column existed are touched: the NBA repair path
#: (src/Sports/repair_odds.py --sport nba, 2026-09-23) always writes
#: 'reconstructed' explicitly, so it never reaches this backfill.
_PROVENANCE_BACKFILL = (
    "UPDATE odds_snapshots SET provenance = 'observed' WHERE provenance IS NULL")

#: Refuse anything outside the taxonomy. NULL is checked explicitly because
#: `NULL NOT IN (...)` is NULL, not true, and would sail through.
_PROVENANCE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS odds_snapshots_provenance_valid
BEFORE INSERT ON odds_snapshots
WHEN NEW.provenance IS NULL
  OR NEW.provenance NOT IN ('observed', 'reconstructed', 'third_party')
BEGIN
  SELECT RAISE(ABORT, 'odds_snapshots.provenance must be observed, reconstructed or third_party');
END;
"""


#: The board's market names (odds_board.MARKETS) and the row fields that
#: price each one. A market counts as "seen" when any of its fields is priced.
HEARTBEAT_MARKETS = {
    "ml": ("home_ml", "away_ml"),
    "spread": ("spread_home", "spread_home_price", "spread_away_price"),
    "total": ("ou_line", "ou_over_price", "ou_under_price"),
}
#: The Odds API's market keys -> the board's.
_API_TO_BOARD_MARKET = {"h2h": "ml", "spreads": "spread", "totals": "total"}

SOURCE_ODDS_API = "odds_api"

_HEARTBEAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_polls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    polled_at     TEXT NOT NULL,     -- UTC ISO; equals captured_at of rows this poll wrote
    sport         TEXT NOT NULL,
    source        TEXT NOT NULL,     -- 'odds_api' (whole board) | 'sbr' (one book, one date)
    status        TEXT NOT NULL CHECK (status IN ('ok', 'empty', 'failed')),
    covers_board  INTEGER NOT NULL CHECK (covers_board IN (0, 1)),
    books         TEXT,              -- comma list asked for; NULL = every book the source carries
    markets       TEXT NOT NULL,     -- comma list of board markets asked for (ml,spread,total)
    events        INTEGER,
    book_rows     INTEGER,
    error         TEXT               -- why a failed poll failed; NULL otherwise
);
CREATE INDEX IF NOT EXISTS idx_odds_polls_time ON odds_polls (sport, polled_at);
CREATE TABLE IF NOT EXISTS odds_seen (
    source         TEXT NOT NULL,
    sport          TEXT NOT NULL,
    game_key       TEXT NOT NULL,
    sportsbook     TEXT NOT NULL,
    market         TEXT NOT NULL,    -- ml | spread | total
    first_seen_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL,
    last_poll_id   INTEGER NOT NULL REFERENCES odds_polls(id),
    PRIMARY KEY (source, sport, game_key, sportsbook, market)
);
"""


def ensure_heartbeat_schema(conn: sqlite3.Connection) -> None:
    """Create odds_polls / odds_seen if missing. Additive; touches no other table."""
    conn.executescript(_HEARTBEAT_SCHEMA)


def record_poll(conn: sqlite3.Connection, *, polled_at: str, sport: str, source: str,
                status: str, covers_board: bool, markets: str,
                books: Optional[str] = None, events: Optional[int] = None,
                book_rows: Optional[int] = None, rows: Optional[List[Dict[str, Any]]] = None,
                error: Optional[str] = None) -> int:
    """
    Write one poll and, for an 'ok' poll, every (game, book, market) it saw
    priced, in one transaction. Returns the poll id.

    A poll that saw nothing is recorded as 'empty', never as 'ok': an empty
    answer from a feed that normally lists the whole slate is far more often
    a hiccup than every book pulling every game at once, and an 'ok' poll with
    no sightings would drop the entire board.
    """
    error = redact(error) if error else error
    if status == "ok" and not rows:
        status = "empty"
    ensure_heartbeat_schema(conn)
    with conn:  # one transaction: the poll and its sightings, or neither
        cur = conn.execute(
            "INSERT INTO odds_polls (polled_at, sport, source, status, covers_board, books, "
            "markets, events, book_rows, error) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (polled_at, sport, source, status, 1 if covers_board else 0, books, markets,
             events, book_rows, error))
        poll_id = cur.lastrowid
        if status == "ok":
            seen = []
            for row in rows or []:
                for market, fields in HEARTBEAT_MARKETS.items():
                    if any(row.get(f) is not None for f in fields):
                        seen.append((source, sport, row["game_key"], row["sportsbook"], market,
                                     polled_at, polled_at, poll_id))
            conn.executemany(
                "INSERT INTO odds_seen (source, sport, game_key, sportsbook, market, "
                "first_seen_at, last_seen_at, last_poll_id) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT (source, sport, game_key, sportsbook, market) DO UPDATE SET "
                "last_seen_at = excluded.last_seen_at, last_poll_id = excluded.last_poll_id",
                seen)
    return poll_id


def _board_markets(api_markets: str) -> str:
    return ",".join(_API_TO_BOARD_MARKET[m] for m in api_markets.split(",")
                    if m in _API_TO_BOARD_MARKET)


class OddsApiError(RuntimeError):
    pass


_KEY_IN_TEXT = re.compile(r"(apiKey=)[^&\s'\")]+", re.IGNORECASE)


def redact(text: Any) -> str:
    """The text with any apiKey=... value replaced.

    requests puts the full URL, query string included, into its exception
    messages, so an unreachable host or a 5xx used to write ODDS_API_KEY
    into the logs and into odds_polls.error, which ledger_sync copies to
    the public server (found 2026-09-28). Everything that leaves this module
    as an error goes through here."""
    return _KEY_IN_TEXT.sub(r"\1REDACTED", str(text))


def _get(url: str, params: Dict[str, Any], timeout: int) -> "requests.Response":
    """requests.get whose failures never carry the key: a transport error is
    re-raised as OddsApiError with the URL's key redacted, and without the
    original exception chained (its message holds the raw URL)."""
    try:
        return requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise OddsApiError(f"The Odds API could not be reached: {redact(exc)}") from None


def get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise OddsApiError(
            "ODDS_API_KEY is not set. Create a free key at the-odds-api.com and "
            "set it in the environment (or the scheduled task's environment)."
        )
    return key


def fetch_nba_events(
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """The schedule, for free.

    /events returns every upcoming event with its commence_time and costs
    NOTHING -- the API's own docs say it "does not count against the usage
    quota", and a live check on 2026-09-19 confirmed it: x-requests-last was 0
    and remaining did not move.

    That is what makes a schedule-aware recorder possible. We can ask "is a
    game about to tip?" as often as we like, and spend credits only when the
    answer is yes. The alternative, and what this module did before, is a blind
    loop that pays full price to discover there is nothing to watch.
    """
    resp = _get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events",
        params={"apiKey": api_key or get_api_key()},
        timeout=timeout,
    )
    quota = {
        "remaining": resp.headers.get("x-requests-remaining"),
        "used": resp.headers.get("x-requests-used"),
        "last": resp.headers.get("x-requests-last"),
    }
    if resp.status_code == 401:
        raise OddsApiError("The Odds API rejected the key (401). Check ODDS_API_KEY.")
    if resp.status_code == 429:
        raise OddsApiError(f"The Odds API quota is exhausted (429). Remaining={quota['remaining']}.")
    if not resp.ok:
        # raise_for_status() would put the URL, key and all, in the message.
        raise OddsApiError(f"The Odds API returned {resp.status_code} for /events: {redact(resp.text[:200])}")
    return resp.json(), quota


def fetch_nba_odds(
    api_key: Optional[str] = None,
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    bookmakers: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """
    One board snapshot: every upcoming NBA event with every carried book's
    h2h/spreads/totals. Returns (events, quota) where quota holds the API's
    x-requests-remaining/used headers.
    """
    params = {
        "apiKey": api_key or get_api_key(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    resp = _get(f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds", params=params, timeout=timeout)
    quota = {
        "remaining": resp.headers.get("x-requests-remaining"),
        "used": resp.headers.get("x-requests-used"),
    }
    if resp.status_code == 401:
        raise OddsApiError("The Odds API rejected the key (401). Check ODDS_API_KEY.")
    if resp.status_code == 429:
        raise OddsApiError(f"The Odds API quota is exhausted (429). Remaining={quota['remaining']}.")
    if not resp.ok:
        raise OddsApiError(f"The Odds API returned {resp.status_code}: {redact(resp.text[:200])}")
    events = resp.json()
    if not isinstance(events, list):
        raise OddsApiError(f"Unexpected response shape: {type(events)}")
    logger.info(
        "The Odds API: %d NBA events on the board; quota used=%s remaining=%s",
        len(events), quota["used"], quota["remaining"],
    )
    return events, quota


def events_to_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten API events into one row per (game, bookmaker), in the archive's
    vocabulary: game_key 'Home:Away' with full team names, American prices.
    """
    rows: List[Dict[str, Any]] = []
    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue
        start = ev.get("commence_time")  # ISO8601 Zulu
        for book in ev.get("bookmakers") or []:
            row: Dict[str, Any] = {
                "sportsbook": book.get("key"),
                "game_key": f"{home}:{away}",
                "home_team": home,
                "away_team": away,
                "game_start_time_utc": start,
                "home_ml": None, "away_ml": None,
                "spread_home": None, "spread_home_price": None, "spread_away_price": None,
                "ou_line": None, "ou_over_price": None, "ou_under_price": None,
            }
            for market in book.get("markets") or []:
                mkey = market.get("key")
                outcomes = market.get("outcomes") or []
                if mkey == "h2h":
                    for o in outcomes:
                        if o.get("name") == home:
                            row["home_ml"] = o.get("price")
                        elif o.get("name") == away:
                            row["away_ml"] = o.get("price")
                elif mkey == "spreads":
                    for o in outcomes:
                        if o.get("name") == home:
                            row["spread_home"] = o.get("point")
                            row["spread_home_price"] = o.get("price")
                        elif o.get("name") == away:
                            row["spread_away_price"] = o.get("price")
                elif mkey == "totals":
                    for o in outcomes:
                        if o.get("name") == "Over":
                            row["ou_line"] = o.get("point")
                            row["ou_over_price"] = o.get("price")
                        elif o.get("name") == "Under":
                            row["ou_under_price"] = o.get("price")
            if row["sportsbook"]:
                rows.append(row)
    return rows


def ensure_snapshot_schema(conn: sqlite3.Connection) -> None:
    """Additive migration: the price/spread columns and provenance, if missing."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(odds_snapshots)")}
    if not existing:
        return                      # no table yet; the creator will make it
    for col, ctype in _EXTRA_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE odds_snapshots ADD COLUMN {col} {ctype}")
            logger.info("odds_snapshots: added column %s", col)
    conn.execute(_PROVENANCE_BACKFILL)
    conn.execute(_PROVENANCE_TRIGGER)
    conn.commit()


_CHANGE_FIELDS = [
    "home_ml", "away_ml", "ou_line",
    "spread_home", "spread_home_price", "spread_away_price",
    "ou_over_price", "ou_under_price",
]


def write_snapshot_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]],
                        sport: str = "NBA", captured_at: Optional[str] = None) -> Dict[str, int]:
    """
    Change-detected insert, same contract as the legacy snapshot writer: a row
    is written only when any tracked number moved since that book's last
    snapshot of that game. Returns {written, unchanged}. `captured_at` lets the
    caller stamp the rows with the same instant as the poll's heartbeat.
    """
    ensure_snapshot_schema(conn)
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    written = unchanged = 0
    for row in rows:
        last = conn.execute(
            "SELECT * FROM odds_snapshots WHERE sportsbook=? AND sport=? AND game_key=? "
            "ORDER BY captured_at DESC LIMIT 1",
            (row["sportsbook"], sport, row["game_key"]),
        ).fetchone()
        if last is not None and all(last[f] == row[f] for f in _CHANGE_FIELDS):
            unchanged += 1
            continue
        conn.execute(
            "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, home_team, "
            "away_team, home_ml, away_ml, ou_line, game_start_time_utc, spread_home, "
            "spread_home_price, spread_away_price, ou_over_price, ou_under_price, "
            "provenance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'observed')",
            (
                captured_at, sport, row["sportsbook"], row["game_key"], row["home_team"],
                row["away_team"], row["home_ml"], row["away_ml"], row["ou_line"],
                row["game_start_time_utc"], row["spread_home"], row["spread_home_price"],
                row["spread_away_price"], row["ou_over_price"], row["ou_under_price"],
            ),
        )
        written += 1
    conn.commit()
    return {"written": written, "unchanged": unchanged}


def snapshot_nba_board(db_path: str, bookmakers: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch the board once and archive it. The one-call entry point.

    Every call leaves an odds_polls row: 'ok' (with odds_seen sightings) when
    the board was fetched AND archived, 'empty' when the feed listed nothing,
    'failed' when the fetch or the archive write raised (then re-raised, so
    callers still see the failure). The heartbeat is written after the
    snapshot rows commit, in its own transaction: if it fails, the archive is
    intact and the board simply has no evidence from this poll, which can only
    make it keep a quote, never drop one.
    """
    polled_at = datetime.now(timezone.utc).isoformat()
    heartbeat = {"polled_at": polled_at, "sport": "NBA", "source": SOURCE_ODDS_API,
                 # /odds returns every upcoming event the API lists, so absence
                 # from an 'ok' poll is evidence. A --bookmakers run is still a
                 # whole board for the books it named; `books` records which.
                 "covers_board": True, "books": bookmakers,
                 "markets": _board_markets(DEFAULT_MARKETS)}
    try:
        events, quota = fetch_nba_odds(bookmakers=bookmakers)
    except Exception as exc:
        _record_poll_quietly(db_path, status="failed", error=str(exc)[:500], **heartbeat)
        raise
    rows = events_to_rows(events)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 15000")
        try:
            result = write_snapshot_rows(conn, rows, captured_at=polled_at)
        except Exception as exc:
            # We saw the board but could not archive it. Recording sightings
            # now would label the OLD price as confirmed, so the poll is failed.
            conn.rollback()
            _record_poll_quietly(db_path, status="failed", events=len(events),
                                 book_rows=len(rows), error=f"archive write: {exc}"[:500],
                                 **heartbeat)
            raise
        try:
            record_poll(conn, status="ok", events=len(events), book_rows=len(rows),
                        rows=rows, **heartbeat)
        except Exception as exc:
            logger.error("Odds heartbeat NOT recorded (snapshot rows are saved): %s", exc)
    finally:
        conn.close()
    summary = {
        "events": len(events),
        "book_rows": len(rows),
        **result,
        "quota_remaining": quota["remaining"],
        "quota_used": quota["used"],
    }
    logger.info("Odds snapshot: %s", summary)
    return summary


def _record_poll_quietly(db_path: str, **kw: Any) -> None:
    """Write a failed/empty poll row; never mask the original error with a new one."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA busy_timeout = 15000")
            record_poll(conn, **kw)
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Could not record the failed odds poll: %s", exc)
