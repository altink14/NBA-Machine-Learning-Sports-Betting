"""
ingest_career_totals.py
=======================
Builds career totals for every player who appeared between 1996-97 and now,
for the Pick Value curve (what a draft slot has actually been worth).

The window is the whole point and also the limitation. A career is only complete
in this table if it started in 1996-97 or later; someone drafted in 1990 has the
first half of his career outside the window and would look worse than he was.
`first_season_in_window` is stored so a caller can exclude anyone whose career
may be clipped, rather than quietly comparing a truncated total to a full one.

WHERE THE NUMBERS COME FROM (changed 2026-09-27). This used to sum one
leaguedashplayerstats request per season from stats.nba.com, skip any season
whose request failed, and write anyway if 80% of seasons answered. On
2026-08-18 the 2011-12 request failed and the guard let it through, so every
career in the table was missing the lockout season: Kobe Bryant 1,288 games
(he played 1,346), LeBron James 62 short, Stephen Curry 26 short, and the 21
players whose only season was 2011-12 counted as never having played.

It now sums the archive's own per-season table, player_season_totals (one row
per player per team per season, built from our box scores), with no network
at all. Those are the same totals the player pages show. They can differ from
nba.com's own career rows by a game or two where nba.com's box score for a
game does not exist (four permanent holes, e.g. the 1996-97 Sonics) - the
archive-wide caveat, not a new one. Kobe's career sums to 1,346 games and
33,643 points, nba.com's figures exactly.

THE GUARD. No season may be missing: the build refuses unless every season in
the window has box scores AND player rows whose appearances fit a real season
(8-16 player-games per team-game; every archived season sits at 10-11). A
missing or half-built season stops the build instead of shrinking every
career. The table is replaced in one transaction, after the old rows are
copied to Data/backups/.

    venv/Scripts/python.exe ingest_career_totals.py [--from 1996] [--to 2025] [--dry-run]
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("career_totals")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
FIRST = 1996
# Appearances per team-game a real season produces. Every archived season is
# 10.3-10.8; a season with only some of its box scores built lands far below.
MIN_APPEARANCES_PER_TEAM_GAME = 8.0
MAX_APPEARANCES_PER_TEAM_GAME = 16.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_career_span (
    player_id INTEGER PRIMARY KEY,
    player_name TEXT,
    seasons INTEGER,
    gp INTEGER,
    min REAL,
    pts INTEGER,
    reb INTEGER,
    ast INTEGER,
    first_season_in_window INTEGER,
    last_season_in_window INTEGER,
    window_first INTEGER,
    window_last INTEGER,
    fetched_at TEXT
)
"""


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[2:]}"


def latest_archived_year(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MAX(season) FROM player_season_totals WHERE season_type = 'Regular Season'"
    ).fetchone()
    return int(row[0][:4])


def window_problems(conn: sqlite3.Connection, start: int, end: int) -> List[str]:
    """Every reason the window cannot be summed; empty when it can."""
    problems = []
    for year in range(start, end + 1):
        season = season_label(year)
        games = conn.execute(
            "SELECT COUNT(*) FROM box_scores WHERE season = ? AND season_type = 'Regular Season'", (season,)
        ).fetchone()[0]
        appearances, players = conn.execute(
            "SELECT COALESCE(SUM(gp), 0), COUNT(DISTINCT player_id) FROM player_season_totals "
            "WHERE season = ? AND season_type = 'Regular Season'", (season,)
        ).fetchone()
        if not games:
            problems.append(f"{season}: no regular-season box scores")
            continue
        if not players:
            problems.append(f"{season}: no player_season_totals rows")
            continue
        rate = appearances / (2 * games)
        if not (MIN_APPEARANCES_PER_TEAM_GAME <= rate <= MAX_APPEARANCES_PER_TEAM_GAME):
            problems.append(f"{season}: {rate:.1f} player-games per team-game (expected "
                            f"{MIN_APPEARANCES_PER_TEAM_GAME:.0f}-{MAX_APPEARANCES_PER_TEAM_GAME:.0f})")
    return problems


def career_rows(conn: sqlite3.Connection, start: int, end: int, now: str) -> list:
    """One row per player: his regular seasons in the window, stints summed."""
    return conn.execute(
        """
        SELECT t.player_id,
               COALESCE(p.full_name, CAST(t.player_id AS TEXT)),
               COUNT(DISTINCT t.season), SUM(t.gp), ROUND(SUM(t.min), 1),
               SUM(t.pts), SUM(t.reb), SUM(t.ast),
               MIN(CAST(substr(t.season, 1, 4) AS INTEGER)),
               MAX(CAST(substr(t.season, 1, 4) AS INTEGER)),
               ?, ?, ?
        FROM player_season_totals t
        LEFT JOIN players p ON p.player_id = t.player_id
        WHERE t.season_type = 'Regular Season' AND t.season BETWEEN ? AND ?
        GROUP BY t.player_id
        """,
        (start, end, now, season_label(start), season_label(end)),
    ).fetchall()


def build(conn: sqlite3.Connection, start: int, end: Optional[int] = None, dry_run: bool = False,
          backup_dir: Optional[str] = None) -> int:
    end = latest_archived_year(conn) if end is None else end
    problems = window_problems(conn, start, end)
    if problems:
        for p in problems:
            logger.error("  %s", p)
        logger.error("Refusing to write a career table with a missing or partial season (%d-%d).", start, end)
        return 1

    now = datetime.now(timezone.utc).isoformat()
    rows = career_rows(conn, start, end, now)
    logger.info("%d players from %d seasons (%s to %s).", len(rows), end - start + 1,
                season_label(start), season_label(end))
    if dry_run:
        return 0

    conn.execute(SCHEMA)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        path = os.path.join(backup_dir, f"player_career_span_before_{datetime.now():%Y%m%d_%H%M%S}.sqlite")
        b = sqlite3.connect(path)
        try:
            b.execute(SCHEMA)
            b.executemany("INSERT INTO player_career_span VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          conn.execute("SELECT * FROM player_career_span").fetchall())
            b.commit()
        finally:
            b.close()
        logger.info("Old table copied to %s", path)
    with conn:
        # Replaced whole, not upserted: a player the new build no longer
        # finds must not keep a stale career.
        conn.execute("DELETE FROM player_career_span")
        conn.executemany(
            """
            INSERT INTO player_career_span (
                player_id, player_name, seasons, gp, min, pts, reb, ast,
                first_season_in_window, last_season_in_window,
                window_first, window_last, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
    total = conn.execute("SELECT COUNT(*) FROM player_career_span").fetchone()[0]
    logger.info("player_career_span: %d players written.", total)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=FIRST)
    ap.add_argument("--to", dest="end", type=int, default=None,
                    help="last season's start year (default: the latest archived season)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(DB_PATH)
    try:
        return build(conn, args.start, args.end, args.dry_run,
                     backup_dir=os.path.join(REPO_ROOT, "Data", "backups"))
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
