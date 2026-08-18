"""
ingest_player_bios_bulk.py
==========================
Fills position, height, weight, college, country, jersey and last team for every
player in the directory, so it can be browsed rather than only searched.

The cheap way in. `playerindex` partitions on TO_YEAR - it returns the players
whose LAST season was the one requested, not that year's rosters - so every player
appears in exactly one season bucket. Walking 1946-47 forward covers all 5,210
players in about eighty requests, where commonplayerinfo would have taken 5,210.

That distinction is the whole reason this is feasible, and it is not documented
anywhere: asking for 2010-11 returns 74 players, all of whom retired after that
season, not the ~450 who played in it.

Columns are added to `players` rather than a side table, because the directory
query then filters and sorts in one place with no join.

Run standalone, or let refresh_registry.py call it weekly:
    venv/Scripts/python.exe ingest_player_bios_bulk.py [--from-season 1946]
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("player_bios_bulk")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
FIRST_SEASON = 1946
CURRENT_SEASON_START = 2025

COLUMNS = {
    "position": "TEXT",
    "height": "TEXT",
    "weight": "TEXT",
    "college": "TEXT",
    "country": "TEXT",
    "jersey": "TEXT",
    "last_team": "TEXT",
    "last_team_id": "INTEGER",
    "bio_fetched_at": "TEXT",
}


def _ensure_columns(conn) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(players)")}
    for col, typ in COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE players ADD COLUMN {col} {typ}")
            logger.info("Added players.%s", col)


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season", type=int, default=FIRST_SEASON)
    ap.add_argument("--only-current", action="store_true",
                    help="Refresh just the current season's bucket (active players).")
    args = ap.parse_args()

    from src.Utils.nba_stats_client import get_client

    client = get_client()
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_columns(conn)
        before = conn.execute(
            "SELECT COUNT(*) FROM players WHERE position IS NOT NULL AND position != ''"
        ).fetchone()[0]

        seasons = (
            [CURRENT_SEASON_START]
            if args.only_current
            else list(range(args.from_season, CURRENT_SEASON_START + 1))
        )
        now = datetime.now(timezone.utc).isoformat()
        total_rows = 0
        failed = []

        for start in seasons:
            season = f"{start}-{str(start + 1)[2:]}"
            try:
                rows = client.player_index(season)
            except Exception as exc:
                logger.warning("  %s: %s", season, str(exc)[:90])
                failed.append(season)
                continue
            if not rows:
                continue

            payload = []
            for r in rows:
                pid = _int(r.get("PERSON_ID"))
                if pid is None:
                    continue
                payload.append((
                    r.get("POSITION") or None,
                    r.get("HEIGHT") or None,
                    str(r.get("WEIGHT")) if r.get("WEIGHT") else None,
                    r.get("COLLEGE") or None,
                    r.get("COUNTRY") or None,
                    str(r.get("JERSEY_NUMBER")) if r.get("JERSEY_NUMBER") else None,
                    r.get("TEAM_ABBREVIATION") or None,
                    _int(r.get("TEAM_ID")),
                    now,
                    pid,
                ))

            # UPDATE, not upsert: the directory's membership comes from
            # ingest_players.py and this only decorates rows that already exist.
            # A player in the index but not in `players` would be a contradiction
            # worth noticing rather than silently inserting.
            conn.executemany(
                """
                UPDATE players SET
                    position = COALESCE(?, position),
                    height = COALESCE(?, height),
                    weight = COALESCE(?, weight),
                    college = COALESCE(?, college),
                    country = COALESCE(?, country),
                    jersey = COALESCE(?, jersey),
                    last_team = COALESCE(?, last_team),
                    last_team_id = COALESCE(?, last_team_id),
                    bio_fetched_at = ?
                WHERE player_id = ?
                """,
                payload,
            )
            conn.commit()
            total_rows += len(payload)
            if start % 10 == 0 or start >= CURRENT_SEASON_START - 1:
                logger.info("  %s: %d players (%d total)", season, len(payload), total_rows)

        after, total_players = conn.execute(
            "SELECT SUM(CASE WHEN position IS NOT NULL AND position != '' THEN 1 ELSE 0 END), "
            "COUNT(*) FROM players"
        ).fetchone()
        logger.info(
            "players with a position: %d of %d (was %d). %d index rows applied, %d seasons failed.",
            after or 0, total_players, before, total_rows, len(failed),
        )
        if failed:
            logger.info("failed seasons (retried on the next run): %s", ", ".join(failed))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
