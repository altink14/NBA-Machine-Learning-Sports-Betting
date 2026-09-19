"""
poll_injuries.py (NFL)
======================
The forward-looking injury recorder. It exists because of a hole we cannot
backfill.

THE PROBLEM. `ingest_injuries.py` loads nflverse's weekly injury file, which
holds one row per player per week carrying the FINAL status, and which stopped
publishing `date_modified` in 2025. So for the seasons a live model would
actually run on, we can neither see how a designation changed through the week
nor prove that any row predates kickoff. A model feature built on it would be
reading the future, and it would not look like a bug; it would look like a
model that is unusually good.

THE FIX, which only works going forward. Poll the report ourselves on a
schedule and stamp every observation with OUR OWN capture time. Run often
enough and the sequence of observations IS the history the file lacks: when a
player first appeared, when Questionable became Out, and, crucially, what was
knowable at the moment a line was available.

Every hour we do not run this is an hour of history that cannot be recovered
later at any price.

SOURCES, both polled every run:

  nflverse  the weekly file (about 50 KB). Authoritative GSIS player ids that
            join to the rest of our archive, plus the week number. No
            timestamp of its own since 2025.
  ESPN      the live injuries feed (about 8.8 MB, 800 entries). Carries ESPN's
            own per-entry `date` saying when the status was set, the body part,
            a projected return date and a short comment. No GSIS ids, so it
            joins by name until a crosswalk is built.

APPEND-ONLY, ENFORCED BY THE DATABASE. `nfl_injury_observations` has triggers
that abort any UPDATE or DELETE. An observation is a record of what we saw at a
moment; editing it later would make the timestamps worthless, which is the same
reason the prediction ledger is append-only. Corrections are new rows.

WHAT GETS WRITTEN. Only changes. On each run a state hash is computed per
player per source; a row is written when that hash differs from the last
observation for that player, or when the player is new. A quiet week therefore
costs almost no rows, and the table reads as a clean diff log.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/poll_injuries.py
    venv/Scripts/python.exe src/Sports/nfl/poll_injuries.py --snapshot   # write every row
    venv/Scripts/python.exe src/Sports/nfl/poll_injuries.py --source espn

Cadence: hourly through the season is plenty, and the two hours before kickoff
are the ones that matter most. See docs/sports/nfl/RUNBOOK.md for scheduling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.poll_injuries")

DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
# User agents, per host, and the reason is not obvious.
#
# ESPN's edge accepts recognisable stock clients (curl/*, Python-urllib/*) and
# answers 403 to BOTH a custom application name and a spoofed browser string.
# Measured 2026-09-19: no header 200, "curl/8.0" 200, "Python-urllib/3.13" 200,
# "BettingBuddy/1.0" 403, a full Chrome string 403. So for ESPN we send no
# override and let urllib identify itself truthfully, which is what we are.
# Do not "fix" this by adding a friendly UA; it will start 403ing.
#
# GitHub accepts the honest name, so nflverse downloads carry it.
UA_GITHUB = {"User-Agent": "BettingBuddy/1.0 (injury recorder)"}
UA_ESPN: dict = {}

SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_injury_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at      TEXT NOT NULL,   -- OUR capture time, UTC. The point of the table.
    source           TEXT NOT NULL,   -- nflverse | espn
    season           TEXT,
    week             INTEGER,
    team             TEXT,
    player_id        TEXT,            -- GSIS id (nflverse) or ESPN athlete id
    player_name      TEXT,
    position         TEXT,
    report_status    TEXT,
    practice_status  TEXT,
    primary_injury   TEXT,
    secondary_injury TEXT,
    source_date      TEXT,            -- the source's own timestamp, when it has one
    return_date      TEXT,            -- ESPN projected return, when given
    comment          TEXT,
    state_hash       TEXT NOT NULL,
    change_kind      TEXT NOT NULL,   -- new | changed | snapshot
    prev_status      TEXT,            -- what it was before, for a changed row
    ingest_version   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_obs_player   ON nfl_injury_observations(source, player_id, week, id);
CREATE INDEX IF NOT EXISTS idx_obs_observed ON nfl_injury_observations(observed_at);
CREATE INDEX IF NOT EXISTS idx_obs_season   ON nfl_injury_observations(season, week);

-- Append-only, enforced here rather than by convention: an observation that
-- can be edited afterwards is not evidence of anything.
CREATE TRIGGER IF NOT EXISTS nfl_injury_obs_no_update
BEFORE UPDATE ON nfl_injury_observations
BEGIN SELECT RAISE(ABORT, 'nfl_injury_observations is append-only'); END;

CREATE TRIGGER IF NOT EXISTS nfl_injury_obs_no_delete
BEFORE DELETE ON nfl_injury_observations
BEGIN SELECT RAISE(ABORT, 'nfl_injury_observations is append-only'); END;
"""


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _hash(*parts: Any) -> str:
    return hashlib.sha1("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()[:16]


def _get(url: str, timeout: int = 180) -> bytes:
    headers = UA_ESPN if "espn.com" in url else UA_GITHUB
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def current_season_and_week(conn: sqlite3.Connection) -> tuple[Optional[str], Optional[int]]:
    """The week whose games surround today, from our own schedule table."""
    now = datetime.now(timezone.utc)
    row = conn.execute(
        """SELECT season, week FROM games
           WHERE local_date BETWEEN ? AND ? ORDER BY local_date LIMIT 1""",
        ((now - timedelta(days=3)).date().isoformat(), (now + timedelta(days=10)).date().isoformat()),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def fetch_nflverse(season: str) -> List[Dict[str, Any]]:
    raw = _get(NFLVERSE_URL.format(season=season)).decode("utf-8", errors="replace")
    out = []
    for r in csv.DictReader(io.StringIO(raw)):
        gsis = _s(r.get("gsis_id"))
        if not gsis:
            continue
        try:
            wk = int(float(r.get("week") or 0)) or None
        except (TypeError, ValueError):
            wk = None
        out.append({
            "season": _s(r.get("season")) or season, "week": wk, "team": _s(r.get("team")),
            "player_id": gsis, "player_name": _s(r.get("full_name")), "position": _s(r.get("position")),
            "report_status": _s(r.get("report_status")), "practice_status": _s(r.get("practice_status")),
            "primary_injury": _s(r.get("report_primary_injury")) or _s(r.get("practice_primary_injury")),
            "secondary_injury": _s(r.get("report_secondary_injury")) or _s(r.get("practice_secondary_injury")),
            "source_date": _s(r.get("date_modified")), "return_date": None, "comment": None,
        })
    return out


def fetch_espn(season: Optional[str], week: Optional[int]) -> List[Dict[str, Any]]:
    d = json.loads(_get(ESPN_URL).decode("utf-8", errors="replace"))
    out = []
    for team in d.get("injuries", []):
        abbr = None
        for e in team.get("injuries", []):
            ath = e.get("athlete") or {}
            abbr = abbr or ((ath.get("team") or {}).get("abbreviation"))
            # The athlete id is not a field; it is in the player-card link.
            pid = None
            for link in ath.get("links") or []:
                m = re.search(r"/id/(\d+)/", link.get("href") or "")
                if m:
                    pid = m.group(1)
                    break
            det = e.get("details") or {}
            out.append({
                "season": season, "week": week,
                "team": abbr or _s(team.get("displayName")),
                "player_id": pid, "player_name": _s(ath.get("displayName")),
                "position": _s((ath.get("position") or {}).get("abbreviation")),
                "report_status": _s(e.get("status")), "practice_status": None,
                "primary_injury": _s(det.get("type")), "secondary_injury": _s(det.get("location")),
                "source_date": _s(e.get("date")), "return_date": _s(det.get("returnDate")),
                "comment": (_s(e.get("shortComment")) or "")[:400] or None,
            })
    return out


def poll(conn: sqlite3.Connection, source: str, rows: List[Dict[str, Any]],
         snapshot: bool) -> tuple[int, int, int]:
    observed_at = datetime.now(timezone.utc).isoformat()
    # Last state we recorded per player PER WEEK, for this source only. The
    # week matters: the nflverse file carries every week of the season in one
    # download, so keying on the player alone makes week 1's row look like a
    # change from week 2's on the very next poll, and the log fills with
    # churn that never happened.
    last: Dict[tuple, tuple] = {}
    for r in conn.execute(
        """SELECT player_id, week, state_hash, report_status FROM nfl_injury_observations
           WHERE source = ? AND id IN (
             SELECT MAX(id) FROM nfl_injury_observations WHERE source = ?
             GROUP BY player_id, week)""",
        (source, source),
    ):
        last[(r[0], r[1])] = (r[2], r[3])

    to_write, n_new, n_chg = [], 0, 0
    for r in rows:
        pid = r["player_id"]
        h = _hash(r["report_status"], r["practice_status"], r["primary_injury"],
                  r["secondary_injury"], r["return_date"], r["week"])
        prev = last.get((pid, r["week"]))
        if prev is None:
            kind, prev_status = "new", None
            n_new += 1
        elif prev[0] != h:
            kind, prev_status = "changed", prev[1]
            n_chg += 1
        elif snapshot:
            kind, prev_status = "snapshot", prev[1]
        else:
            continue
        to_write.append((
            observed_at, source, r["season"], r["week"], r["team"], pid, r["player_name"],
            r["position"], r["report_status"], r["practice_status"], r["primary_injury"],
            r["secondary_injury"], r["source_date"], r["return_date"], r["comment"],
            h, kind, prev_status, INGEST_VERSION,
        ))

    conn.executemany(
        "INSERT INTO nfl_injury_observations (observed_at, source, season, week, team, player_id, "
        "player_name, position, report_status, practice_status, primary_injury, secondary_injury, "
        "source_date, return_date, comment, state_hash, change_kind, prev_status, ingest_version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", to_write)
    conn.commit()
    return len(to_write), n_new, n_chg


def main() -> int:
    ap = argparse.ArgumentParser(description="Record NFL injury report state with our own timestamps.")
    ap.add_argument("--snapshot", action="store_true",
                    help="Write every row, not only changes (use sparingly; it inflates the log).")
    ap.add_argument("--source", choices=["nflverse", "espn", "both"], default="both")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=120)
    ensure_core_schema(conn)
    conn.executescript(SCHEMA)
    conn.commit()

    season, week = current_season_and_week(conn)
    if not season:
        season = str(max(int(r[0]) for r in conn.execute("SELECT DISTINCT season FROM games")))
    logger.info("polling for season %s week %s", season, week)

    started_at = datetime.now(timezone.utc).isoformat()
    total = 0
    for source, fetch in (("nflverse", lambda: fetch_nflverse(season)),
                          ("espn", lambda: fetch_espn(season, week))):
        if args.source not in ("both", source):
            continue
        try:
            rows = fetch()
        except Exception as exc:
            logger.error("%s fetch failed: %s", source, str(exc)[:200])
            continue
        written, n_new, n_chg = poll(conn, source, rows, args.snapshot)
        total += written
        logger.info("%-9s %4d entries seen -> %3d written (%d new, %d changed)",
                    source, len(rows), written, n_new, n_chg)

    record_run(conn, "nfl_injury_observations", "nflverse+espn", "poll",
               started_at, datetime.now(timezone.utc).isoformat(), total,
               notes=f"season={season} week={week} snapshot={args.snapshot}")

    grand = conn.execute("SELECT COUNT(*) FROM nfl_injury_observations").fetchone()[0]
    first = conn.execute("SELECT MIN(observed_at) FROM nfl_injury_observations").fetchone()[0]
    logger.info("SUMMARY written=%d archive=%d recording_since=%s", total, grand, first)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
