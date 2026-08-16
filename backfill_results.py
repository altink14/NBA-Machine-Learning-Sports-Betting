"""
backfill_results.py
===================
Every NBA final score, 1946-47 to today, into `game_results`.

WHY THIS EXISTS SEPARATELY
box_scores only reaches back to 2022-23 and pbp_events to the same. Scorigami -
"this final score has never happened" - is only an honest claim against the full
history, so it needs its own source. The league game log gives a whole season in
one request, so all ~80 seasons cost about 160 calls.

One row per TEAM per game (the shape the endpoint returns), so a game is two
rows and the winner/loser split is derived rather than baked in.

Usage
-----
    venv/Scripts/python.exe backfill_results.py
    venv/Scripts/python.exe backfill_results.py --from-season 2020-21
    venv/Scripts/python.exe backfill_results.py --dry-run

Resumable: a season already present is skipped unless --refresh is passed. The
current season should be re-pulled as it progresses, which is what --refresh is
for; the daily job can call it with --from-season <current>.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "TeamData.sqlite")

# The BAA's first season. Everything the NBA counts as its own history starts here.
FIRST_SEASON_START = 1946

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("backfill_results")

SCHEMA = """
CREATE TABLE IF NOT EXISTS game_results (
    game_id     TEXT NOT NULL,
    team_id     INTEGER NOT NULL,
    season      TEXT,
    season_type TEXT,
    game_date   TEXT,
    team_abbr   TEXT,
    team_name   TEXT,
    matchup     TEXT,
    wl          TEXT,
    pts         INTEGER,
    PRIMARY KEY (game_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_results_season ON game_results(season);
CREATE INDEX IF NOT EXISTS idx_results_date ON game_results(game_date);
"""

INSERT = """
INSERT OR REPLACE INTO game_results
    (game_id, team_id, season, season_type, game_date, team_abbr, team_name, matchup, wl, pts)
VALUES (?,?,?,?,?,?,?,?,?,?)
"""


def season_labels(start_year: int, end_year: int) -> List[str]:
    return [f"{y}-{str((y + 1) % 100).zfill(2)}" for y in range(start_year, end_year + 1)]


def current_season_start() -> int:
    from datetime import date
    t = date.today()
    # An NBA season is labelled by the calendar year it opens in.
    return t.year if t.month >= 8 else t.year - 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill every NBA final score.")
    ap.add_argument("--from-season", help="Earliest season to fetch, e.g. 2020-21.")
    ap.add_argument("--season-types", nargs="*", default=["Regular Season", "Playoffs"])
    ap.add_argument("--refresh", action="store_true", help="Re-fetch seasons already stored.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript(SCHEMA)
    conn.commit()

    start = FIRST_SEASON_START
    if args.from_season:
        start = int(args.from_season[:4])
    seasons = season_labels(start, current_season_start())

    have = {
        (r["season"], r["season_type"])
        for r in conn.execute("SELECT DISTINCT season, season_type FROM game_results")
    }

    jobs = [
        (s, st) for st in args.season_types for s in seasons
        if args.refresh or (s, st) not in have
    ]
    logger.info("%d season/type combinations to fetch (of %d).", len(jobs), len(seasons) * len(args.season_types))

    if args.dry_run:
        for s, st in jobs[:20]:
            logger.info("  MISSING %s %s", s, st)
        if len(jobs) > 20:
            logger.info("  ... and %d more", len(jobs) - 20)
        conn.close()
        return 0
    if not jobs:
        logger.info("Nothing to do.")
        conn.close()
        return 0

    from src.Utils.nba_stats_client import get_client
    client = get_client()

    started = time.time()
    stored = failures = 0

    for i, (season, stype) in enumerate(jobs, 1):
        try:
            rows = client.league_game_log(season=season, season_type=stype, player_or_team="T")
        except KeyboardInterrupt:
            logger.warning("Interrupted - progress is saved, rerun to resume.")
            break
        except Exception as exc:
            failures += 1
            logger.error("[%d/%d] %s %s FAILED: %s", i, len(jobs), season, stype, str(exc)[:110])
            continue

        n = 0
        for r in rows:
            gid, tid, pts = r.get("GAME_ID"), r.get("TEAM_ID"), r.get("PTS")
            if not gid or not tid or pts is None:
                continue
            conn.execute(INSERT, (
                gid, tid, season, stype, r.get("GAME_DATE"),
                r.get("TEAM_ABBREVIATION"), r.get("TEAM_NAME"),
                r.get("MATCHUP"), r.get("WL"), pts,
            ))
            n += 1
        conn.commit()
        stored += n
        # Early seasons legitimately have no playoff log; say which, so an empty
        # result is never mistaken for a failed request.
        if n == 0:
            logger.info("[%d/%d] %s %s - no games recorded", i, len(jobs), season, stype)
        elif i % 10 == 0 or i == len(jobs):
            logger.info("[%d/%d] %s %s - %d rows (%s total)", i, len(jobs), season, stype, n, format(stored, ","))

    games = conn.execute("SELECT COUNT(DISTINCT game_id) FROM game_results").fetchone()[0]
    logger.info("Done in %.1f min. %s team-rows across %s games. %d failure(s).",
                (time.time() - started) / 60, format(stored, ","), format(games, ","), failures)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
