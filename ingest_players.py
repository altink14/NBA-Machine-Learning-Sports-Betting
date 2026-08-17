"""
ingest_players.py
=================
Builds the player directory: every player in NBA history, with their real roster
status and the span of their career.

This supersedes ingest_roster_status.py, which corrected `is_active` on the 891
players our box scores happened to contain. Same upstream request, so there is no
reason to make it twice - and the directory itself was the bigger problem. It
covered players appearing in ingested box scores, 2022-23 onward, which meant
searching the "player encyclopedia" for Michael Jordan returned nothing.

Two columns are added to `players` so a retired player reads as one:
  from_year / to_year - the first and last season the league has them playing.
Without them a 1960s guard and a rookie look identical in a search result.

Honest limits, both surfaced by the frontend rather than hidden:
  - Most of these players have no stats in our archive. Their page resolves and
    shows career totals pulled live from the league, but the parts of the profile
    built on our own game logs - splits, career highs, recent games - are empty
    for anyone who stopped playing before 2022-23.
  - ROSTERSTATUS lags reality. A player who announced his retirement this week
    can still read active until the league updates its index.

Run standalone, or let refresh_registry.py call it daily:
    venv/Scripts/python.exe ingest_players.py
"""

import logging
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("players_ingest")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
CURRENT_SEASON = "2025-26"

# A response much smaller than the league's real history means something is wrong
# upstream, and rewriting the directory from it would delete most of the game.
MIN_EXPECTED = 4000


def _ensure_columns(conn) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(players)")}
    for col in ("from_year", "to_year"):
        if col not in have:
            conn.execute(f"ALTER TABLE players ADD COLUMN {col} INTEGER")
            logger.info("Added players.%s", col)


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    from src.Utils.nba_stats_client import get_client

    try:
        index = get_client().common_all_players(
            season=CURRENT_SEASON, is_only_current_season=0
        )
    except Exception as exc:
        logger.error("Could not fetch the player index: %s", exc, exc_info=True)
        return 1

    if len(index) < MIN_EXPECTED:
        logger.error(
            "Player index returned only %d rows, expected %d+. Refusing to rewrite the "
            "directory from a partial response.", len(index), MIN_EXPECTED
        )
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_columns(conn)
        before = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]

        rows = []
        for r in index:
            pid = _int(r.get("PERSON_ID"))
            full = (r.get("DISPLAY_FIRST_LAST") or "").strip()
            if pid is None or not full:
                continue
            # The index gives "Last, First"; split the display name instead so
            # first/last match what the rest of the app already stores.
            parts = full.split(" ", 1)
            first = parts[0]
            last = parts[1] if len(parts) > 1 else ""
            active = 1 if r.get("ROSTERSTATUS") in (1, "1", "Active") else 0
            rows.append((pid, full, first, last, active,
                         _int(r.get("FROM_YEAR")), _int(r.get("TO_YEAR"))))

        # Names, status and career span come from the index and are authoritative.
        # Nothing else in the row is touched, so a player already present keeps
        # whatever else the app has attached to them.
        conn.executemany(
            """
            INSERT INTO players (player_id, full_name, first_name, last_name,
                                 is_active, from_year, to_year)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                full_name=excluded.full_name,
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                is_active=excluded.is_active,
                from_year=excluded.from_year,
                to_year=excluded.to_year
            """,
            rows,
        )
        conn.commit()

        after, active_now = conn.execute(
            "SELECT COUNT(*), SUM(is_active) FROM players"
        ).fetchone()
        logger.info(
            "players: %d rows (was %d), %d currently on a roster. Added %d.",
            after, before, active_now or 0, after - before,
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
