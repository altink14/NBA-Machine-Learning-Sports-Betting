"""
backfill_officials.py
=====================
Ingest the officiating crew for archived games.

WHY THIS EXISTS. The referee page on the frontend was deleted because the
figures on it were invented and no source was known. There is a source: the
Officials result set of boxscoresummaryv2 returns the three-person crew, with
stable OFFICIAL_IDs, for every game from about 2003-04 onward — including the
current season, since the endpoint's documented 2025-04-10 data problems
affect other result sets and not this one. Verified against 1996-97 (empty),
2001-02 (empty), 2003-04 (three officials) and 2025-26 (three officials).

WHAT IT WRITES
  officials         one row per official: id, name, jersey number
  game_officials    one row per (game, official)
  officials_fetch   one row per game we have ASKED about, whether or not it
                    returned a crew. This is what makes the job resumable and
                    stops us re-asking about pre-2003 games forever.

MANNERS. One request per game through the shared stats client (disk cache,
retries, rate limiting all come from there), newest season first so the most
useful data lands earliest, and a commit every 25 games so a kill loses
almost nothing.

    python src/Process-Data/backfill_officials.py --seasons 2025-26 2024-25
    python src/Process-Data/backfill_officials.py --from-season 2018-19
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Utils.nba_stats_client import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data", "TeamData.sqlite"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS officials (
    official_id INTEGER PRIMARY KEY,
    first_name  TEXT,
    last_name   TEXT,
    jersey_num  TEXT
);
CREATE TABLE IF NOT EXISTS game_officials (
    game_id     TEXT NOT NULL,
    official_id INTEGER NOT NULL,
    PRIMARY KEY (game_id, official_id)
);
CREATE INDEX IF NOT EXISTS idx_game_officials_official ON game_officials(official_id);
CREATE TABLE IF NOT EXISTS officials_fetch (
    game_id      TEXT PRIMARY KEY,
    fetched_at   TEXT,
    n_officials  INTEGER
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def seasons_to_do(conn: sqlite3.Connection, args) -> list:
    rows = conn.execute(
        "SELECT DISTINCT season FROM box_scores ORDER BY season DESC"
    ).fetchall()
    all_seasons = [r[0] for r in rows]
    if args.seasons:
        return [s for s in all_seasons if s in set(args.seasons)]
    if args.from_season:
        return [s for s in all_seasons if s >= args.from_season]
    return all_seasons


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="*", help="Explicit seasons, e.g. 2025-26 2024-25")
    p.add_argument("--from-season", help="Every archived season >= this one")
    p.add_argument("--limit", type=int, default=0, help="Stop after N fetches (0 = no cap)")
    p.add_argument("--db", default=DB_PATH)
    args = p.parse_args()

    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    targets = seasons_to_do(conn, args)
    if not targets:
        logger.error("No matching seasons in box_scores.")
        return 1
    logger.info("Seasons queued (newest first): %s", ", ".join(targets))

    client = get_client(backfill_mode=True)
    fetched = crews = empty = errors = 0

    for season in targets:
        games = conn.execute(
            """
            SELECT b.game_id
            FROM box_scores b
            LEFT JOIN officials_fetch f ON f.game_id = b.game_id
            WHERE b.season = ? AND f.game_id IS NULL
            ORDER BY b.game_date DESC
            """,
            (season,),
        ).fetchall()
        if not games:
            logger.info("%s: already complete", season)
            continue
        logger.info("%s: %d games to fetch", season, len(games))

        for i, row in enumerate(games, 1):
            gid = row["game_id"]
            try:
                data = client.boxscore_summary(gid)
                crew = data.get("Officials") or []
            except Exception as e:  # a single bad game must not end the run
                errors += 1
                logger.warning("game %s failed: %s", gid, str(e)[:90])
                continue

            now = datetime.now(timezone.utc).isoformat()
            for off in crew:
                oid = off.get("OFFICIAL_ID")
                if not oid:
                    continue
                conn.execute(
                    """INSERT INTO officials (official_id, first_name, last_name, jersey_num)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(official_id) DO UPDATE SET
                         first_name = excluded.first_name,
                         last_name  = excluded.last_name,
                         jersey_num = excluded.jersey_num""",
                    (oid, (off.get("FIRST_NAME") or "").strip(),
                     (off.get("LAST_NAME") or "").strip(),
                     (str(off.get("JERSEY_NUM") or "")).strip()),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO game_officials (game_id, official_id) VALUES (?, ?)",
                    (gid, oid),
                )
            conn.execute(
                "INSERT OR REPLACE INTO officials_fetch (game_id, fetched_at, n_officials) VALUES (?, ?, ?)",
                (gid, now, len(crew)),
            )

            fetched += 1
            if crew:
                crews += 1
            else:
                empty += 1
            if fetched % 25 == 0:
                conn.commit()
                logger.info("%s: %d/%d (crews %d, empty %d, errors %d)",
                            season, i, len(games), crews, empty, errors)
            if args.limit and fetched >= args.limit:
                conn.commit()
                logger.info("Hit --limit %d, stopping.", args.limit)
                _summary(conn, fetched, crews, empty, errors)
                return 0

        conn.commit()
        logger.info("%s: done", season)

    conn.commit()
    _summary(conn, fetched, crews, empty, errors)
    return 0


def _summary(conn: sqlite3.Connection, fetched, crews, empty, errors) -> None:
    n_off = conn.execute("SELECT COUNT(*) FROM officials").fetchone()[0]
    n_link = conn.execute("SELECT COUNT(*) FROM game_officials").fetchone()[0]
    n_games = conn.execute("SELECT COUNT(DISTINCT game_id) FROM game_officials").fetchone()[0]
    logger.info(
        "SUMMARY fetched=%d withCrew=%d empty=%d errors=%d | officials=%d links=%d games=%d",
        fetched, crews, empty, errors, n_off, n_link, n_games,
    )


if __name__ == "__main__":
    raise SystemExit(main())
