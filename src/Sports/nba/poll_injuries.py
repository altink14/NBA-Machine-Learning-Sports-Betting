"""
poll_injuries.py (NBA)
======================
The NBA injury recorder: the same idea as src/Sports/nfl/poll_injuries.py,
for the sport that is actually the product.

THE PROBLEM. The NBA injury report reaches us only through ESPN's public feed
(src/Utils/espn_injuries.py), and only at request time. ESPN overwrites the
current state and keeps no history, so until this existed nobody could say
afterwards who was reported Out before a game, when a Day-To-Day tag appeared,
or what was knowable when a line was available. Every hour this does not run
is an hour of history that cannot be recovered later at any price.

THE FIX, which only works going forward. Poll the report on a schedule and
stamp every observation with OUR OWN capture time.

WHERE IT LIVES. Data/TeamData.sqlite, the NBA's own database, the way the NFL
recorder writes to Data/NflData.sqlite (see core_schema.py: one database per
sport). Two tables, both append-only, enforced by triggers:

  nba_injury_polls         one row per run, success OR failure. This is what
                           makes a gap in the observation log readable: an
                           hour with no observations and an 'ok' poll means
                           nothing changed; an hour with a 'failed' poll means
                           we could not look; an hour with no poll at all
                           means the recorder was not running.
  nba_injury_observations  one row per player per change, same shape as
                           nfl_injury_observations minus the NFL-only columns
                           (week, practice_status) plus our own player and
                           team ids so a row joins to the archive.

WHAT GETS WRITTEN. Only changes, following the NFL precedent: a state hash per
player, and a row when it differs from that player's last observation. A quiet
day costs almost nothing and the table reads as a diff log. --snapshot writes
every row, as on the NFL side.

TWO DELIBERATE DIFFERENCES FROM THE NFL RECORDER, both about honesty:

1. A FAILED FETCH IS RECORDED AS A FAILURE AND THE RUN EXITS 1. The NFL
   recorder logs a failed source and still exits 0, so the hourly job reads
   "ok" on an hour it could not see. Here the poll row says 'failed', with the
   reason, and no observation is written: nothing about an unread feed is
   allowed to look like "nobody changed", and nothing is ever inferred as
   "cleared" from a feed we did not read.

2. A PLAYER WHO LEAVES THE REPORT GETS A 'cleared' ROW. A change-only log that
   never records removals answers "what was his status before tip?" with his
   last status, forever: a player who came back from injury would read as Out
   for the rest of the season. That is exactly the stale-value-looks-fine bug
   shape this project keeps finding. So when a successful poll no longer lists
   someone who was listed, a row with report_status NULL and
   change_kind='cleared' is written. (The NFL recorder has the same hole; it is
   noted in the report that added this file rather than changed there.)

   The corollary guard: a feed that parses to ZERO entries while our last state
   has players listed is treated as a failure, not as the whole league getting
   healthy in one hour. --allow-empty overrides, for the day that is true.

WHAT IS IN THE HASH. Status, injury type/location/detail/side, projected
return, team, and ESPN's own entry date. The NFL hash leaves the date out; the
NBA report is re-issued game by game, and a re-dated "Out" is a new report for
a new game, which is the difference between a stale designation and a
reaffirmed one. The free-text comment is stored but not hashed, because ESPN
retouches wording without the status moving.

USER AGENT. None is overridden. ESPN's edge accepts stock clients and 403s
custom and spoofed-browser strings (measured on the NFL side, 2026-09-19);
see the note in src/Sports/nfl/poll_injuries.py before "fixing" this.

Usage:
    venv/Scripts/python.exe src/Sports/nba/poll_injuries.py
    venv/Scripts/python.exe src/Sports/nba/poll_injuries.py --snapshot
    venv/Scripts/python.exe src/Sports/nba/poll_injuries.py --db some/other.sqlite

Cadence: hourly, from the "hourly" job in src/Sports/run_scheduled.py. The
hours before tip are the ones that matter; NBA designations move through the
afternoon of game day.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests  # noqa: E402

from src.Sports.core_schema import INGEST_VERSION  # noqa: E402
from src.Utils.espn_injuries import ESPN_INJURIES_URL, normalize_name  # noqa: E402

logger = logging.getLogger("nba.poll_injuries")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
SOURCE = "espn"
FETCH_TIMEOUT_SECONDS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS nba_injury_polls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       TEXT NOT NULL,
    finished_at      TEXT NOT NULL,
    source           TEXT NOT NULL,
    endpoint         TEXT,
    status           TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    http_status      INTEGER,
    feed_timestamp   TEXT,            -- ESPN's own top-level timestamp, when sent
    entries_seen     INTEGER,
    rows_written     INTEGER,
    n_new            INTEGER,
    n_changed        INTEGER,
    n_cleared        INTEGER,
    n_unmatched      INTEGER,         -- entries we could not tie to players.player_id
    error            TEXT,            -- why a failed poll failed; NULL when ok
    ingest_version   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_nba_inj_polls_time ON nba_injury_polls(started_at);

CREATE TABLE IF NOT EXISTS nba_injury_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at      TEXT NOT NULL,   -- OUR capture time, UTC. The point of the table.
    poll_id          INTEGER NOT NULL REFERENCES nba_injury_polls(id),
    source           TEXT NOT NULL,   -- espn
    season           TEXT,            -- '2026-27', from the feed when it says
    team             TEXT,            -- ESPN's team display name
    team_abbr        TEXT,            -- our team_metadata abbreviation, NULL if unresolved
    player_key       TEXT NOT NULL,   -- ESPN athlete id, or 'name:<normalized>' without one
    player_id        TEXT,            -- ESPN athlete id
    nba_player_id    INTEGER,         -- our players.player_id by normalized name, NULL if unmatched
    player_name      TEXT,
    position         TEXT,
    report_status    TEXT,            -- Out | Day-To-Day | ... ; NULL on a 'cleared' row
    primary_injury   TEXT,            -- ESPN details.type, e.g. Knee
    secondary_injury TEXT,            -- ESPN details.location, e.g. Leg
    injury_detail    TEXT,            -- ESPN details.detail, e.g. Soreness
    side             TEXT,
    source_date      TEXT,            -- ESPN's own entry timestamp
    return_date      TEXT,            -- ESPN projected return, when given
    comment          TEXT,
    state_hash       TEXT NOT NULL,
    change_kind      TEXT NOT NULL CHECK (change_kind IN ('new', 'changed', 'cleared', 'snapshot')),
    prev_status      TEXT,            -- what it was before, for changed and cleared rows
    ingest_version   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_nba_inj_obs_player   ON nba_injury_observations(source, player_key, id);
CREATE INDEX IF NOT EXISTS idx_nba_inj_obs_observed ON nba_injury_observations(observed_at);
CREATE INDEX IF NOT EXISTS idx_nba_inj_obs_nba_id   ON nba_injury_observations(nba_player_id, observed_at);

-- Append-only, enforced here rather than by convention: an observation that
-- can be edited afterwards is not evidence of anything, and neither is a
-- failure record that can be quietly turned into a success.
CREATE TRIGGER IF NOT EXISTS nba_injury_obs_no_update
BEFORE UPDATE ON nba_injury_observations
BEGIN SELECT RAISE(ABORT, 'nba_injury_observations is append-only'); END;

CREATE TRIGGER IF NOT EXISTS nba_injury_obs_no_delete
BEFORE DELETE ON nba_injury_observations
BEGIN SELECT RAISE(ABORT, 'nba_injury_observations is append-only'); END;

CREATE TRIGGER IF NOT EXISTS nba_injury_polls_no_update
BEFORE UPDATE ON nba_injury_polls
BEGIN SELECT RAISE(ABORT, 'nba_injury_polls is append-only'); END;

CREATE TRIGGER IF NOT EXISTS nba_injury_polls_no_delete
BEFORE DELETE ON nba_injury_polls
BEGIN SELECT RAISE(ABORT, 'nba_injury_polls is append-only'); END;
"""

_CLEARED_HASH = "cleared"


class FeedError(RuntimeError):
    """We could not read the report. Never to be recorded as 'nobody changed'."""

    def __init__(self, message: str, http_status: Optional[int] = None):
        super().__init__(message)
        self.http_status = http_status


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _hash(*parts: Any) -> str:
    return hashlib.sha1("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def season_for(when: datetime) -> str:
    """'2026-27' for anything from 1 July 2026 to 30 June 2027 (the league year).

    Used only when the feed does not name its season itself.
    """
    start = when.year if when.month >= 7 else when.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


# ---------------------------------------------------------------------------
# Fetch and parse
# ---------------------------------------------------------------------------

def fetch_payload(url: str = ESPN_INJURIES_URL,
                  timeout: int = FETCH_TIMEOUT_SECONDS) -> Tuple[Dict[str, Any], int]:
    """GET the feed. Raises FeedError on anything but a 200 carrying JSON."""
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise FeedError(f"request failed: {exc}") from exc
    if r.status_code != 200:
        raise FeedError(f"HTTP {r.status_code} from {url}", http_status=r.status_code)
    try:
        return r.json(), r.status_code
    except ValueError as exc:
        raise FeedError(f"HTTP 200 but the body is not JSON: {exc}", http_status=200) from exc


def _athlete_id(ath: Dict[str, Any]) -> Optional[str]:
    """ESPN's athlete id. It is not a field on the NBA feed (verified
    2026-09-23: 0 of 75 entries carry athlete.id); it is in the player-card
    link, as on the NFL feed. `id` is still tried first in case that changes."""
    if _s(ath.get("id")):
        return _s(ath.get("id"))
    for link in ath.get("links") or []:
        m = re.search(r"/id/(\d+)(?:/|$)", (link or {}).get("href") or "")
        if m:
            return m.group(1)
    return None


def parse_payload(payload: Any, now: Optional[datetime] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """The feed, flattened to one dict per player. Returns (rows, feed_timestamp).

    Strict on shape, because the alternative is worse: a payload whose
    structure has changed would otherwise parse to an empty list, and an empty
    list is indistinguishable from a healthy league. Anything that is not the
    shape we know raises FeedError.
    """
    if not isinstance(payload, dict):
        raise FeedError(f"payload is {type(payload).__name__}, not an object")
    status = payload.get("status")
    if status is not None and str(status).lower() != "success":
        raise FeedError(f"feed reports status={status!r}")
    teams = payload.get("injuries")
    if not isinstance(teams, list):
        raise FeedError("payload has no 'injuries' list; the feed's shape has changed")

    feed_season = payload.get("season")
    season = _s(feed_season.get("displayName")) if isinstance(feed_season, dict) else None
    season = season or season_for(now or datetime.now(timezone.utc))

    rows: Dict[str, Dict[str, Any]] = {}
    for team in teams:
        if not isinstance(team, dict):
            raise FeedError("an 'injuries' entry is not an object")
        entries = team.get("injuries")
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise FeedError(f"team {team.get('displayName')!r} has a non-list 'injuries'")
        team_name = _s(team.get("displayName"))
        for e in entries:
            if not isinstance(e, dict):
                continue
            ath = e.get("athlete") or {}
            name = _s(ath.get("displayName")) or _s(ath.get("shortName"))
            pid = _athlete_id(ath)
            key = pid or (f"name:{normalize_name(name)}" if name else None)
            if not key:
                logger.warning("skipping an entry with neither an athlete id nor a name (team %s)",
                               team_name)
                continue
            det = e.get("details") or {}
            row = {
                "season": season,
                "team": team_name or _s((ath.get("team") or {}).get("displayName")),
                "player_key": key, "player_id": pid, "player_name": name,
                "position": _s((ath.get("position") or {}).get("abbreviation")),
                "report_status": _s(e.get("status")),
                "primary_injury": _s(det.get("type")),
                "secondary_injury": _s(det.get("location")),
                "injury_detail": _s(det.get("detail")),
                "side": _s(det.get("side")),
                "source_date": _s(e.get("date")),
                "return_date": _s(det.get("returnDate")),
                "comment": ((_s(e.get("shortComment")) or _s(e.get("longComment")) or "")[:400]
                            or None),
            }
            # One row per player. If ESPN ever lists a player twice, keep the
            # most recently dated entry rather than letting the two alternate
            # as "changes" on every poll.
            prev = rows.get(key)
            if prev is None or (row["source_date"] or "") > (prev["source_date"] or ""):
                if prev is not None:
                    logger.warning("%s is listed twice; keeping the entry dated %s",
                                   name, row["source_date"])
                rows[key] = row
    return list(rows.values()), _s(payload.get("timestamp"))


# ---------------------------------------------------------------------------
# Joining to our archive
# ---------------------------------------------------------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def player_index(conn: sqlite3.Connection) -> Dict[str, int]:
    """normalized full name -> players.player_id; active players win clashes.
    Same rule as espn_injuries._load_player_index, read from this connection."""
    if not _table_exists(conn, "players"):
        return {}
    out: Dict[str, int] = {}
    for pid, full_name, _active in conn.execute(
            "SELECT player_id, full_name, is_active FROM players ORDER BY is_active ASC"):
        k = normalize_name(full_name)
        if k:
            out[k] = pid
    return out


def team_index(conn: sqlite3.Connection) -> Dict[str, str]:
    """normalized team name -> our abbreviation. ESPN says 'LA Clippers'."""
    if not _table_exists(conn, "team_metadata"):
        return {}
    out: Dict[str, str] = {}
    for full_name, nickname, abbr in conn.execute(
            "SELECT full_name, nickname, abbreviation FROM team_metadata"):
        for n in (full_name, nickname, abbr):
            k = normalize_name(n)
            if k:
                out[k] = abbr
    if "los angeles clippers" in out:
        out.setdefault("la clippers", out["los angeles clippers"])
    return out


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------

def _state_hash(r: Dict[str, Any]) -> str:
    return _hash(r["report_status"], r["primary_injury"], r["secondary_injury"],
                 r["injury_detail"], r["side"], r["return_date"], r["source_date"], r["team"])


def last_state(conn: sqlite3.Connection, source: str = SOURCE) -> Dict[str, sqlite3.Row]:
    """The latest observation per player for this source."""
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    out = {}
    for r in cur.execute(
            """SELECT * FROM nba_injury_observations WHERE source = ? AND id IN (
                 SELECT MAX(id) FROM nba_injury_observations WHERE source = ?
                 GROUP BY player_key)""", (source, source)):
        out[r["player_key"]] = r
    return out


def diff(rows: List[Dict[str, Any]], last: Dict[str, Any], snapshot: bool,
         players: Dict[str, int], teams: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """What to write, given what the feed says now and what we last recorded."""
    out: List[Dict[str, Any]] = []
    counts = {"new": 0, "changed": 0, "cleared": 0, "unmatched": 0}
    seen = set()
    for r in rows:
        key = r["player_key"]
        seen.add(key)
        r = dict(r)
        r["nba_player_id"] = players.get(normalize_name(r["player_name"] or ""))
        if r["nba_player_id"] is None:
            counts["unmatched"] += 1
        r["team_abbr"] = teams.get(normalize_name(r["team"] or ""))
        h = _state_hash(r)
        prev = last.get(key)
        if prev is None or prev["change_kind"] == "cleared":
            kind, prev_status = "new", None
            counts["new"] += 1
        elif prev["state_hash"] != h:
            kind, prev_status = "changed", prev["report_status"]
            counts["changed"] += 1
        elif snapshot:
            kind, prev_status = "snapshot", prev["report_status"]
        else:
            continue
        r.update(state_hash=h, change_kind=kind, prev_status=prev_status)
        out.append(r)

    # Anyone listed last time and absent now has left the report. Only ever
    # reached with a feed we actually read -- see run().
    for key, prev in last.items():
        if key in seen or prev["change_kind"] == "cleared":
            continue
        counts["cleared"] += 1
        out.append({
            "season": prev["season"], "team": prev["team"], "team_abbr": prev["team_abbr"],
            "player_key": key, "player_id": prev["player_id"],
            "nba_player_id": prev["nba_player_id"], "player_name": prev["player_name"],
            "position": prev["position"], "report_status": None, "primary_injury": None,
            "secondary_injury": None, "injury_detail": None, "side": None,
            "source_date": None, "return_date": None, "comment": None,
            "state_hash": _CLEARED_HASH, "change_kind": "cleared",
            "prev_status": prev["report_status"],
        })
    return out, counts


_OBS_COLS = ("observed_at", "poll_id", "source", "season", "team", "team_abbr", "player_key",
             "player_id", "nba_player_id", "player_name", "position", "report_status",
             "primary_injury", "secondary_injury", "injury_detail", "side", "source_date",
             "return_date", "comment", "state_hash", "change_kind", "prev_status",
             "ingest_version")


def _insert_poll(conn: sqlite3.Connection, **kw: Any) -> int:
    cols = ("started_at", "finished_at", "source", "endpoint", "status", "http_status",
            "feed_timestamp", "entries_seen", "rows_written", "n_new", "n_changed",
            "n_cleared", "n_unmatched", "error", "ingest_version")
    vals = [kw.get(c) for c in cols]
    vals[-1] = INGEST_VERSION
    cur = conn.execute(f"INSERT INTO nba_injury_polls ({', '.join(cols)}) "
                       f"VALUES ({', '.join('?' * len(cols))})", vals)
    return cur.lastrowid


def record_failure(conn: sqlite3.Connection, started_at: str, error: str,
                   http_status: Optional[int] = None, endpoint: str = ESPN_INJURIES_URL) -> None:
    _insert_poll(conn, started_at=started_at, finished_at=_now(), source=SOURCE,
                 endpoint=endpoint, status="failed", http_status=http_status,
                 error=error[:500])
    conn.commit()


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run(conn: sqlite3.Connection,
        fetch: Callable[[], Tuple[Any, int]] = fetch_payload,
        snapshot: bool = False, allow_empty: bool = False,
        endpoint: str = ESPN_INJURIES_URL) -> Dict[str, Any]:
    """Poll once and record the outcome, success or failure. Never raises for
    a feed problem: it records it and returns status 'failed'."""
    ensure_schema(conn)
    # Read our own state before touching the network, so a database problem
    # surfaces as itself and is never filed as a feed failure.
    last = last_state(conn)
    started_at = _now()
    http_status: Optional[int] = None
    try:
        payload, http_status = fetch()
        observed_at = _now()
        rows, feed_ts = parse_payload(payload)
        listed_before = sum(1 for r in last.values() if r["change_kind"] != "cleared")
        if not rows and listed_before and not allow_empty:
            raise FeedError(
                f"the feed parsed to zero entries while {listed_before} player(s) were listed "
                f"at the last poll. The whole league does not get healthy in an hour; this is "
                f"far more likely a broken feed than a true report, so nothing was recorded as "
                f"cleared. Re-run with --allow-empty if it is genuinely true.",
                http_status=http_status)
    except Exception as exc:
        # FeedError is the expected kind; anything else (a parser surprise on
        # a reshaped payload) is recorded the same way rather than escaping
        # with no poll row, which would read later as "the recorder was off".
        if isinstance(exc, FeedError):
            http_status = exc.http_status or http_status
            why = str(exc)
        else:
            why = f"unexpected {type(exc).__name__} reading the feed: {exc}"
        logger.error("NBA injury poll FAILED: %s", why)
        record_failure(conn, started_at, why, http_status, endpoint)
        return {"status": "failed", "error": why, "rows_written": 0}

    players, teams = player_index(conn), team_index(conn)
    # The join to our ids is best effort and the raw ESPN fields are kept
    # either way, but a join that silently stops working would leave every
    # row unlinked while each poll reads 'ok'. Say so.
    if not players or not teams:
        logger.warning("players/team_metadata missing or empty in this database: every "
                       "observation will be written without our player and team ids")
    to_write, counts = diff(rows, last, snapshot, players, teams)
    if rows and counts["unmatched"] > 0.2 * len(rows):
        logger.warning("%d of %d listed players did not match players.player_id by name; "
                       "is the player directory current?", counts["unmatched"], len(rows))
    # One transaction: the poll row and its observations land together or
    # not at all, so an observation can never exist without its poll.
    try:
        poll_id = _insert_poll(
            conn, started_at=started_at, finished_at=_now(), source=SOURCE, endpoint=endpoint,
            status="ok", http_status=http_status, feed_timestamp=feed_ts,
            entries_seen=len(rows), rows_written=len(to_write), n_new=counts["new"],
            n_changed=counts["changed"], n_cleared=counts["cleared"],
            n_unmatched=counts["unmatched"])
        conn.executemany(
            f"INSERT INTO nba_injury_observations ({', '.join(_OBS_COLS)}) "
            f"VALUES ({', '.join('?' * len(_OBS_COLS))})",
            [tuple({**r, "observed_at": observed_at, "poll_id": poll_id, "source": SOURCE,
                    "ingest_version": INGEST_VERSION}.get(c) for c in _OBS_COLS)
             for r in to_write])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"status": "ok", "entries_seen": len(rows), "rows_written": len(to_write),
            **counts, "poll_id": poll_id, "feed_timestamp": feed_ts}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Record the NBA injury report with our own timestamps.")
    ap.add_argument("--snapshot", action="store_true",
                    help="Write every row, not only changes (use sparingly; it inflates the log).")
    ap.add_argument("--allow-empty", action="store_true",
                    help="Accept a feed with zero entries as true even though players were "
                         "listed last time, and record them all as cleared.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    conn = sqlite3.connect(args.db, timeout=120)
    try:
        result = run(conn, snapshot=args.snapshot, allow_empty=args.allow_empty)
        if result["status"] != "ok":
            logger.error("SUMMARY poll=failed (recorded as a failed poll, not as an empty report)")
            return 1
        polls, obs, first = conn.execute(
            "SELECT (SELECT COUNT(*) FROM nba_injury_polls), "
            "(SELECT COUNT(*) FROM nba_injury_observations), "
            "(SELECT MIN(observed_at) FROM nba_injury_observations)").fetchone()
        logger.info("SUMMARY %d listed -> %d written (%d new, %d changed, %d cleared; "
                    "%d not matched to our players) | archive %d observations over %d polls "
                    "since %s", result["entries_seen"], result["rows_written"], result["new"],
                    result["changed"], result["cleared"], result["unmatched"], obs, polls, first)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
