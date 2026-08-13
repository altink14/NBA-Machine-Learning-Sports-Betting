"""
backfill_passing.py
===================
Pre-warm the passing wheel: fetch player->player pass tracking for every
team-season so no visitor ever pays the cold-start cost.

One team-season is ~18 outbound requests (one PlayerDashPtPass per rostered
player), so the full 30 teams x 11 seasons is roughly 6,000 requests. At the
client's 2s backfill rate that is several hours — run it overnight, and run it
in chunks with --teams / --seasons if you would rather not do it in one sitting.

Everything is resumable: a team-season already recorded in
`team_passing_fetch_log` is skipped, and the stats client keeps a permanent disk
cache in Data/nba_cache, so an interrupted run costs nothing to restart.

Usage
-----
    # everything from 2015-16 to the current season
    venv/Scripts/python.exe backfill_passing.py

    # one team, one season
    venv/Scripts/python.exe backfill_passing.py --teams BOS --seasons 2024-25

    # playoffs too
    venv/Scripts/python.exe backfill_passing.py --season-types "Regular Season" Playoffs

    # see what is missing without fetching anything
    venv/Scripts/python.exe backfill_passing.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from typing import List

# main_api owns the fetch/store logic; importing it keeps exactly one
# implementation of the passing schema and the roster fallback.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_api import _ensure_team_passing, CURRENT_SEASON  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "TeamData.sqlite")

# SecondSpectrum tracking starts in 2013-14; we publish from 2015-16.
FIRST_SEASON_END = 2016

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_passing")


def season_list() -> List[str]:
    end_now = int(CURRENT_SEASON[:4]) + 1
    return [f"{e - 1}-{str(e % 100).zfill(2)}" for e in range(FIRST_SEASON_END, end_now + 1)]


def team_list(conn) -> List[tuple]:
    rows = conn.execute(
        "SELECT team_id, abbreviation FROM team_metadata ORDER BY abbreviation"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def already_done(conn, team_id: int, season: str, season_type: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM team_passing_fetch_log
        WHERE team_id=? AND season=? AND season_type=?
        """,
        (team_id, season, season_type),
    ).fetchone()
    return row is not None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill NBA passing-wheel data.")
    ap.add_argument("--teams", nargs="*", help="Team abbreviations (default: all 30).")
    ap.add_argument("--seasons", nargs="*", help="Seasons like 2024-25 (default: 2015-16 onward).")
    ap.add_argument(
        "--season-types", nargs="*", default=["Regular Season"],
        help='Default "Regular Season". Pass Playoffs to add postseason wheels.',
    )
    ap.add_argument("--dry-run", action="store_true", help="List missing team-seasons and exit.")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # The API writes to the same file whenever someone opens an uncached
    # team-season, so wait for the lock rather than dying mid-run.
    conn.execute("PRAGMA busy_timeout = 30000")

    # The fetch log may not exist yet on a fresh database; one ensure call
    # against a throwaway pair creates both tables.
    teams = team_list(conn)
    if not teams:
        logger.error("No rows in team_metadata — run the main pipeline first.")
        return 1

    if args.teams:
        wanted = {t.upper() for t in args.teams}
        teams = [t for t in teams if t[1].upper() in wanted]
        if not teams:
            logger.error("None of %s matched a team abbreviation.", args.teams)
            return 1

    seasons = args.seasons or season_list()
    season_types = args.season_types

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_passing_fetch_log (
            team_id INTEGER, season TEXT, season_type TEXT, fetched_at TEXT,
            players_queried INTEGER, edges_stored INTEGER,
            PRIMARY KEY (team_id, season, season_type)
        )
        """
    )
    conn.commit()

    jobs = [
        (tid, abbr, season, stype)
        for stype in season_types
        for season in seasons
        for tid, abbr in teams
        if not already_done(conn, tid, season, stype)
    ]

    total = len(teams) * len(seasons) * len(season_types)
    logger.info("%d of %d team-seasons still to fetch.", len(jobs), total)

    if args.dry_run:
        for _, abbr, season, stype in jobs:
            logger.info("  MISSING  %s  %s  %s", abbr, season, stype)
        conn.close()
        return 0

    if not jobs:
        logger.info("Nothing to do.")
        conn.close()
        return 0

    started = time.time()
    failures = 0

    for i, (tid, abbr, season, stype) in enumerate(jobs, 1):
        logger.info("[%d/%d] %s %s (%s)", i, len(jobs), abbr, season, stype)
        try:
            _ensure_team_passing(conn, tid, season, stype)
            edges = conn.execute(
                """
                SELECT edges_stored FROM team_passing_fetch_log
                WHERE team_id=? AND season=? AND season_type=?
                """,
                (tid, season, stype),
            ).fetchone()
            n = edges["edges_stored"] if edges else 0
            if n == 0:
                logger.warning("    no tracking data (expected for pre-2013-14 seasons)")
            else:
                logger.info("    %d edges", n)
        except KeyboardInterrupt:
            logger.warning("Interrupted — progress is saved, rerun to resume.")
            break
        except Exception as exc:
            # A team-season that blows up should not end the night's run.
            failures += 1
            logger.error("    FAILED %s %s: %s", abbr, season, exc)

    elapsed = time.time() - started
    logger.info("Done in %.1f min. %d failure(s).", elapsed / 60, failures)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
