"""
backfill_pbp.py
===============
Download play-by-play for every archived game into a queryable `pbp_events`
table.

WHY A TABLE AND NOT THE JSON BLOB
`box_scores.pbp_json` already exists and the single-game endpoint fills it
lazily, which is right for displaying one game. It is the wrong shape for
everything else: measured, the blobs average 239 KB, so all 5,255 games would
add ~1.26 GB, and answering "how often does a team down 15 in the 4th win?"
would mean loading and parsing every one of them. Normalising to one row per
action costs roughly a quarter of the space and turns those questions into SQL.

This does not touch `pbp_json`. The two coexist: blob for one game, table for
league-wide work.

WHAT IT UNLOCKS
Win-probability curves, an excitement index, Scorigami, run detection,
comeback-probability tables, clutch possession lists, and a per-game passing
wheel (assists are parseable from the action description).

Usage
-----
    venv/Scripts/python.exe backfill_pbp.py                  # everything missing
    venv/Scripts/python.exe backfill_pbp.py --limit 25       # a taste
    venv/Scripts/python.exe backfill_pbp.py --season 2025-26
    venv/Scripts/python.exe backfill_pbp.py --dry-run

Resumable: a game already present in `pbp_events` is skipped, and the stats
client keeps a permanent disk cache, so an interrupted run costs nothing to
restart.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "TeamData.sqlite")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_pbp")

REGULATION_PERIOD = 720  # 12-minute quarters, in seconds
OT_PERIOD = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS pbp_events (
    game_id         TEXT NOT NULL,
    action_number   INTEGER NOT NULL,
    period          INTEGER,
    clock_seconds   REAL,      -- remaining in the period
    elapsed_seconds REAL,      -- since tip-off, so games are comparable
    team_id         INTEGER,
    team_tricode    TEXT,
    person_id       INTEGER,
    player_name     TEXT,
    action_type     TEXT,
    sub_type        TEXT,
    description     TEXT,
    loc_x           REAL,
    loc_y           REAL,
    shot_distance   REAL,
    shot_value      INTEGER,
    shot_result     TEXT,
    is_field_goal   INTEGER,
    score_home      INTEGER,
    score_away      INTEGER,
    assist_hint     TEXT,      -- surname lifted from "(Poole 1 AST)"
    PRIMARY KEY (game_id, action_number)
);
CREATE INDEX IF NOT EXISTS idx_pbp_game ON pbp_events(game_id);
CREATE INDEX IF NOT EXISTS idx_pbp_person ON pbp_events(person_id);
CREATE INDEX IF NOT EXISTS idx_pbp_type ON pbp_events(action_type);
"""

# "Queen 4' Driving Layup (2 PTS) (Poole 1 AST)" -> "Poole"
_ASSIST_RE = re.compile(r"\(([^)]+?)\s+\d+\s+AST\)")
# "PT11M08.00S"
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def parse_clock(clock: Optional[str]) -> Optional[float]:
    if not clock:
        return None
    m = _CLOCK_RE.match(clock)
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def elapsed(period: Optional[int], clock_seconds: Optional[float]) -> Optional[float]:
    """Seconds since tip-off, so events from different games line up."""
    if not period or clock_seconds is None:
        return None
    if period <= 4:
        before = (period - 1) * REGULATION_PERIOD
        length = REGULATION_PERIOD
    else:
        before = 4 * REGULATION_PERIOD + (period - 5) * OT_PERIOD
        length = OT_PERIOD
    return before + (length - clock_seconds)


def parse_assist(description: Optional[str]) -> Optional[str]:
    if not description:
        return None
    m = _ASSIST_RE.search(description)
    return m.group(1).strip() if m else None


def to_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def rows_for(game_id: str, actions: List[Dict]) -> List[tuple]:
    out = []
    for a in actions:
        num = a.get("actionNumber")
        if num is None:
            continue
        period = a.get("period")
        cs = parse_clock(a.get("clock"))
        out.append((
            game_id, num, period, cs, elapsed(period, cs),
            a.get("teamId") or None, a.get("teamTricode") or None,
            a.get("personId") or None, a.get("playerName") or None,
            a.get("actionType"), a.get("subType"), a.get("description"),
            a.get("xLegacy"), a.get("yLegacy"), a.get("shotDistance"),
            a.get("shotValue"), a.get("shotResult") or None,
            a.get("isFieldGoal"),
            to_int(a.get("scoreHome")), to_int(a.get("scoreAway")),
            parse_assist(a.get("description")),
        ))
    return out


INSERT = """
INSERT OR REPLACE INTO pbp_events (
    game_id, action_number, period, clock_seconds, elapsed_seconds,
    team_id, team_tricode, person_id, player_name,
    action_type, sub_type, description,
    loc_x, loc_y, shot_distance, shot_value, shot_result, is_field_goal,
    score_home, score_away, assist_hint
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill NBA play-by-play into pbp_events.")
    ap.add_argument("--season", help="Only this season, e.g. 2025-26.")
    ap.add_argument("--limit", type=int, help="Stop after this many games.")
    ap.add_argument("--dry-run", action="store_true", help="Report what is missing and exit.")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # The API writes to this file too; wait for the lock rather than dying.
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript(SCHEMA)
    conn.commit()

    where = "WHERE b.game_id NOT IN (SELECT DISTINCT game_id FROM pbp_events)"
    params: List[Any] = []
    if args.season:
        where += " AND b.season = ?"
        params.append(args.season)

    # Newest first: partial progress is then immediately useful for the
    # seasons anyone actually looks at.
    todo = conn.execute(
        f"SELECT b.game_id, b.season, b.season_type, b.game_date FROM box_scores b {where} "
        "ORDER BY b.game_date DESC",
        params,
    ).fetchall()

    total_games = conn.execute("SELECT COUNT(*) FROM box_scores").fetchone()[0]
    done = total_games - len(todo)
    logger.info("%d of %d games already have play-by-play; %d to fetch.",
                done, total_games, len(todo))

    if args.dry_run:
        by_season: Dict[str, int] = {}
        for r in todo:
            by_season[r["season"]] = by_season.get(r["season"], 0) + 1
        for s in sorted(by_season, reverse=True):
            logger.info("  MISSING %s: %d games", s, by_season[s])
        conn.close()
        return 0

    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        logger.info("Nothing to do.")
        conn.close()
        return 0

    from src.Utils.nba_stats_client import get_client
    client = get_client()

    started = time.time()
    failures = 0
    events_written = 0

    for i, r in enumerate(todo, 1):
        gid = r["game_id"]
        try:
            actions = client.play_by_play(gid)
        except KeyboardInterrupt:
            logger.warning("Interrupted - progress is saved, rerun to resume.")
            break
        except Exception as exc:
            failures += 1
            logger.error("[%d/%d] %s FAILED: %s", i, len(todo), gid, exc)
            continue

        rows = rows_for(gid, actions or [])
        if not rows:
            # Recorded as a failure rather than skipped silently: an empty
            # play-by-play for an archived game means something is wrong, and
            # leaving no row means the next run retries it.
            failures += 1
            logger.warning("[%d/%d] %s returned no actions", i, len(todo), gid)
            continue

        conn.executemany(INSERT, rows)
        conn.commit()
        events_written += len(rows)

        if i % 25 == 0 or i == len(todo):
            rate = i / max(time.time() - started, 1)
            eta = (len(todo) - i) / rate / 60 if rate else 0
            logger.info("[%d/%d] %s %s - %s events so far, ~%.0f min left",
                        i, len(todo), r["season"], gid, format(events_written, ","), eta)

    logger.info("Done in %.1f min. %s events written. %d failure(s).",
                (time.time() - started) / 60, format(events_written, ","), failures)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
