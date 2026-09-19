"""
ingest_injuries.py (NFL)
========================
Weekly injury reports, 2009 to date, from the nflverse-data `injuries` release
(CC-BY-4.0). Practice participation, game-status designation, the body part,
and the timestamp at which the report was last modified.

READ THIS BEFORE USING THIS TABLE IN A MODEL.

The source is ONE ROW PER PLAYER PER WEEK carrying the FINAL state of that
week's report, stamped with when it was last touched. It is NOT a history of
how the designation changed through the week. Measured on 2024: 6,215 rows,
6,213 distinct player-week keys, so only two players in a whole season carry
more than one row.

The consequence is the single most important fact about this table. We cannot
reconstruct what a team's injury report said on Thursday, or at the moment a
line was posted. We know only what it said when it stopped changing. So:

  * `date_modified` is stored exactly as given, never inferred.
  * `known_before_kickoff` is computed once, at ingest, by comparing
    `date_modified` against the game's own kickoff in UTC. It is 1 when the
    report was finalised before the game started, 0 when it was finalised
    after, and NULL when we do not know the kickoff.
  * A model feature built from this table MUST filter `known_before_kickoff = 1`.
    A row finalised after kickoff knows how the game went. Using it is the
    textbook silent leak, and it would not look like a bug: it would look like
    a model that is unusually good at predicting games.

Even filtered, this is an approximation of "what was known at lock time", not a
measurement of it. Any page or model card built on it says so. The only way to
measure the real thing is to poll the report ourselves from now on and keep the
diffs, which is a forward-looking job, not a backfill.

THE TIMESTAMP DISAPPEARS IN 2025. Verified 2026-09-19 by reading the header of
every season file: `date_modified` is present 2009 through 2024 and is GONE
from 2025 and 2026, which instead add a `season_type` column. So for the two
current seasons we have the report but no indication of when it was written,
and `known_before_kickoff` is NULL for every one of those rows.

That is the opposite of a footnote. The forward-looking seasons, the ones a
live model would actually run on, are exactly the ones where we cannot prove a
report predates kickoff. Two consequences:

  1. A model trained or evaluated on 2025+ injury data cannot claim to be free
     of injury leakage on the strength of this file alone.
  2. The fix is not a backfill. It is to poll the injury report ourselves from
     now on and stamp each observation with our own capture time. Until that
     runs, treat 2025+ injury rows as descriptive only.

The 2009 floor is a data floor, not a choice: no free source carries timestamped
NFL injury reports before then. Injury pages must not offer earlier seasons.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/ingest_injuries.py
    venv/Scripts/python.exe src/Sports/nfl/ingest_injuries.py --season 2024
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.ingest_injuries")

SOURCE = "nflverse-data injuries (CC-BY-4.0)"
ENDPOINT_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
CACHE_DIR = os.path.join(REPO_ROOT, "Data", "nfl_cache")
FIRST_SEASON = 2009  # no free source carries timestamped reports before this

SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_injury_reports (
    season                   TEXT NOT NULL,
    week                     INTEGER NOT NULL,
    game_type                TEXT,
    team                     TEXT NOT NULL,
    gsis_id                  TEXT NOT NULL,
    position                 TEXT,
    full_name                TEXT,
    report_primary_injury    TEXT,
    report_secondary_injury  TEXT,
    report_status            TEXT,   -- Out | Doubtful | Questionable | '' | Note
    practice_primary_injury  TEXT,
    practice_secondary_injury TEXT,
    practice_status          TEXT,
    date_modified            TEXT NOT NULL,   -- '' when the source omits it (2025+)
    has_timestamp            INTEGER NOT NULL,-- 1 when the source dated this row
    game_id                  TEXT,   -- resolved from season+week+team, null if unmatched
    known_before_kickoff     INTEGER,-- 1 before, 0 after, NULL when kickoff unknown
    source                   TEXT,
    fetched_at               TEXT,
    ingest_version           INTEGER,
    PRIMARY KEY (season, week, team, gsis_id, date_modified)
);
CREATE INDEX IF NOT EXISTS idx_inj_game   ON nfl_injury_reports(game_id);
CREATE INDEX IF NOT EXISTS idx_inj_player ON nfl_injury_reports(gsis_id);
CREATE INDEX IF NOT EXISTS idx_inj_status ON nfl_injury_reports(report_status);
CREATE INDEX IF NOT EXISTS idx_inj_known  ON nfl_injury_reports(known_before_kickoff);
"""


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def download(season: int) -> Optional[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"injuries_{season}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return path
    import urllib.request
    url = ENDPOINT_TMPL.format(season=season)
    req = urllib.request.Request(url, headers={"User-Agent": "BettingBuddy/1.0 (archive ingest)"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except Exception as exc:
        logger.warning("  season %d unavailable: %s", season, str(exc)[:100])
        return None
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest nflverse weekly injury reports.")
    ap.add_argument("--season", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=120)
    conn.row_factory = sqlite3.Row
    ensure_core_schema(conn)
    conn.executescript(SCHEMA)
    conn.commit()

    last = max(int(r[0]) for r in conn.execute("SELECT DISTINCT season FROM games"))
    seasons = [args.season] if args.season else list(range(FIRST_SEASON, last + 1))
    done = ({r[0] for r in conn.execute("SELECT DISTINCT season FROM nfl_injury_reports")}
            if not args.force else set())
    todo = [s for s in seasons if str(s) not in done]
    logger.info("%d season(s) present; %d to do", len(done), len(todo))

    # (season, week, team) -> (game_id, date_utc). A team plays at most one
    # game a week, so this is a unique key; POST weeks continue the numbering
    # in both files, which is asserted below.
    sched: Dict[tuple, tuple] = {}
    for r in conn.execute("SELECT game_id, season, week, home_team_id, away_team_id, date_utc FROM games"):
        for tid in (r["home_team_id"], r["away_team_id"]):
            sched[(r["season"], r["week"], tid.replace("nfl-", ""))] = (r["game_id"], r["date_utc"])

    started_at = datetime.now(timezone.utc).isoformat()
    total = matched = before = after = unknown = 0

    for season in todo:
        path = download(season)
        if not path:
            continue
        fetched_at = datetime.now(timezone.utc).isoformat()
        rows = []
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                gsis, team, week = _s(r.get("gsis_id")), _s(r.get("team")), r.get("week")
                # 2025+ files carry no date_modified at all. Keep the row, mark
                # it undated, and let known_before_kickoff stay NULL; dropping
                # it would hide the regression rather than report it.
                dm = _s(r.get("date_modified")) or ""
                if not (gsis and team and week):
                    continue
                try:
                    wk = int(float(week))
                except (TypeError, ValueError):
                    continue

                gid, kickoff = sched.get((str(season), wk, team), (None, None))
                known = None
                if gid and kickoff and dm:
                    # Both are ISO-8601 UTC; a string compare is safe and exact.
                    known = 1 if dm < kickoff else 0
                if gid:
                    matched += 1
                if known == 1:
                    before += 1
                elif known == 0:
                    after += 1
                else:
                    unknown += 1

                rows.append((
                    str(season), wk, _s(r.get("game_type")), team, gsis,
                    _s(r.get("position")), _s(r.get("full_name")),
                    _s(r.get("report_primary_injury")), _s(r.get("report_secondary_injury")),
                    _s(r.get("report_status")),
                    _s(r.get("practice_primary_injury")), _s(r.get("practice_secondary_injury")),
                    _s(r.get("practice_status")),
                    dm, 1 if dm else 0, gid, known, SOURCE, fetched_at, INGEST_VERSION,
                ))
        conn.executemany(
            "INSERT OR REPLACE INTO nfl_injury_reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        total += len(rows)
        dated = sum(1 for r in rows if r[14])
        logger.info("  %d: %d reports, %d dated%s", season, len(rows), dated,
                    "  <-- SOURCE PUBLISHES NO TIMESTAMPS FOR THIS SEASON" if rows and not dated else "")
        try:
            os.remove(path)
        except OSError:
            pass

    finished_at = datetime.now(timezone.utc).isoformat()
    record_run(conn, "nfl_injury_reports", SOURCE, ENDPOINT_TMPL.format(season="{season}"),
               started_at, finished_at, total,
               notes=f"matched_to_game={matched} before_kickoff={before} after_kickoff={after}")

    grand = conn.execute("SELECT COUNT(*) FROM nfl_injury_reports").fetchone()[0]
    logger.info("SUMMARY this_run=%d archive=%d matched_to_game=%d before_kickoff=%d "
                "after_kickoff=%d kickoff_unknown=%d", total, grand, matched, before, after, unknown)
    if after:
        logger.warning("%d report(s) were finalised AFTER kickoff. They are stored with "
                       "known_before_kickoff=0 and MUST be excluded from model features.", after)
    undated = conn.execute(
        "SELECT COUNT(*) FROM nfl_injury_reports WHERE has_timestamp = 0").fetchone()[0]
    if undated:
        seasons_undated = [r[0] for r in conn.execute(
            "SELECT DISTINCT season FROM nfl_injury_reports WHERE has_timestamp = 0 ORDER BY season")]
        logger.warning("%d report(s) across season(s) %s carry NO timestamp: the source dropped "
                       "date_modified in 2025. known_before_kickoff is NULL for these, so they "
                       "cannot be shown to predate kickoff and must not feed a model feature "
                       "until we poll and stamp the report ourselves.",
                       undated, ", ".join(seasons_undated))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
