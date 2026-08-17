"""
ingest_roster_status.py
=======================
Makes `players.is_active` mean what it says.

The column was never a roster flag. `players` is filled as a side effect of
box-score ingestion - nba_pipeline inserts every player it sees in a processed
game with is_active hardcoded to 1 - so it meant "we have seen this player in a
game we ingested", and every one of the 891 rows read Active. The player
directory printed an Active badge on all of them, including players who have
since retired, which is worse than printing nothing.

The real flag is ROSTERSTATUS on the league's all-time player index: 1 for
players currently on a roster, 0 for everyone else. This reads that and writes it
back, so the badge distinguishes a current player from a retired one.

Note what this does NOT do: it does not add players to the directory. The
directory still covers players who appear in our ingested box scores, which is
2022-23 onward, so Michael Jordan is still not searchable. Expanding it to all
5,205 players in league history is a separate decision, because most of them
would have no stats behind their page.

Run standalone, or let refresh_registry.py call it daily:
    venv/Scripts/python.exe ingest_roster_status.py
"""

import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("roster_status")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
CURRENT_SEASON = "2025-26"


def main() -> int:
    from src.Utils.nba_stats_client import get_client

    try:
        index = get_client().common_all_players(
            season=CURRENT_SEASON, is_only_current_season=0
        )
    except Exception as exc:
        logger.error("Could not fetch the player index: %s", exc, exc_info=True)
        return 1

    if len(index) < 4000:
        logger.error(
            "Player index returned only %d rows, expected 5,000+. Refusing to rewrite "
            "roster status from a partial response.", len(index)
        )
        return 1

    # ROSTERSTATUS is 1 for players currently on a roster, 0 otherwise.
    status = {}
    for r in index:
        pid = r.get("PERSON_ID")
        if pid is None:
            continue
        raw = r.get("ROSTERSTATUS")
        status[int(pid)] = 1 if raw in (1, "1", "Active") else 0
    active_in_league = sum(status.values())
    logger.info(
        "Index: %d players, %d currently on a roster.", len(status), active_in_league
    )

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT player_id, is_active FROM players").fetchall()
        before_active = sum(1 for _, a in rows if a == 1)

        changed = 0
        unknown = 0
        for pid, current in rows:
            want = status.get(pid)
            if want is None:
                # In our box scores but not in the league index. Left alone rather
                # than guessed at - flipping an unknown to Inactive would be
                # asserting a retirement we cannot see.
                unknown += 1
                continue
            if current != want:
                conn.execute(
                    "UPDATE players SET is_active = ? WHERE player_id = ?", (want, pid)
                )
                changed += 1
        conn.commit()

        after = conn.execute(
            "SELECT SUM(is_active), COUNT(*) FROM players"
        ).fetchone()
        logger.info(
            "players: %d of %d now marked active (was %d of %d). %d rows updated, "
            "%d not found in the index and left as they were.",
            after[0] or 0, after[1], before_active, len(rows), changed, unknown,
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
