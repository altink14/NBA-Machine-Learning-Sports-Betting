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
    venv/Scripts/python.exe backfill_pbp.py --rebuild        # re-derive every held game from the disk cache

Resumable: a game already present in `pbp_events` is skipped, and the stats
client keeps a permanent disk cache, so an interrupted run costs nothing to
restart.

THE KEY IS THE FEED'S actionId, NOT actionNumber (fixed 2026-09-27)
playbyplayv3 splits one play into two rows that share an actionNumber: a
missed shot and the BLOCK that caused it, a turnover and the STEAL that
forced it. Until 2026-09-27 the table was keyed on (game_id, action_number)
with INSERT OR REPLACE, so the second row silently overwrote the first:
86,077 blocked misses and 137,078 stolen turnovers were missing across
2019-20..2025-26, every PBP FG% read high and PBP turnovers read about half
of the box score. `actionId` is unique in every cached game and runs in
feed (chronological) order - actionNumber does not: a late-entered event
keeps its high number - so it is the key, and readers that care about order
sort by it. Rows go in with a plain INSERT: if two rows ever collided
again, the game fails loudly instead of losing one.
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
    action_number   INTEGER NOT NULL,  -- the feed's number: NOT unique (a block/steal shares it), NOT time order
    action_id       INTEGER NOT NULL,  -- the feed's actionId: unique per game, chronological
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
    PRIMARY KEY (game_id, action_id)
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
    # actionId equals the list position (1..n) in every cached game (checked
    # on all 8,885 held games, 2026-09-27). Should a feed ever omit it, the
    # position is that same sequence - used for the whole game, so the two
    # numberings can never mix inside one game.
    use_position = any(a.get("actionId") is None for a in actions)
    out = []
    for pos, a in enumerate(actions, 1):
        num = a.get("actionNumber")
        if num is None:
            continue
        aid = pos if use_position else a.get("actionId")
        period = a.get("period")
        cs = parse_clock(a.get("clock"))
        out.append((
            game_id, num, aid, period, cs, elapsed(period, cs),
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


# Plain INSERT on purpose (it was INSERT OR REPLACE, which is how blocks
# erased misses). Games are skipped whole once present, so nothing is ever
# re-inserted; a key collision is a feed problem and must fail the game.
COLUMNS = (
    "game_id, action_number, action_id, period, clock_seconds, elapsed_seconds, "
    "team_id, team_tricode, person_id, player_name, "
    "action_type, sub_type, description, "
    "loc_x, loc_y, shot_distance, shot_value, shot_result, is_field_goal, "
    "score_home, score_away, assist_hint"
)


def insert_sql(table: str = "pbp_events") -> str:
    return f"INSERT INTO {table} ({COLUMNS}) VALUES ({','.join(['?'] * 22)})"


INSERT = insert_sql()


def has_action_id(conn: sqlite3.Connection, table: str = "pbp_events") -> bool:
    """True when the table is on the fixed key (or does not exist yet)."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    return not cols or "action_id" in cols


def write_game(conn: sqlite3.Connection, rows: List[tuple], table: str = "pbp_events") -> None:
    """One game's rows in one transaction: all of them land, or none do."""
    try:
        conn.executemany(insert_sql(table), rows)
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        raise


BACKUP_PREFIX = "pbp_events_pre_"


def rebuild_from_cache(conn: sqlite3.Connection) -> int:
    """Re-derive every game pbp_events holds from the disk cache, never the network.

    Built side by side in `pbp_events_rebuild`, 250 games per transaction so
    other writers to this file are never locked out for long, then swapped
    in with one short transaction. The old table is kept as
    `pbp_events_pre_<date>` until it is dropped after verifying. If any held
    game has no cache entry, the run stops before changing anything and
    lists them: a rebuild must not quietly shrink the table, and a network
    fetch is a different job (the normal backfill).
    """
    from src.Utils.nba_stats_client import _cache_path, cache_candidates, load_cache_file

    def cached(gid):
        return cache_candidates(_cache_path("playbyplayv3", {"game_id": gid}))

    games = [r[0] for r in conn.execute("SELECT DISTINCT game_id FROM pbp_events ORDER BY game_id")]
    uncached = [g for g in games if not cached(g)]
    if uncached:
        logger.error("%d held game(s) have no cached play-by-play; nothing changed: %s",
                     len(uncached), ", ".join(uncached[:50]))
        return 1
    logger.info("Rebuilding %d games from the disk cache.", len(games))

    table_ddl = SCHEMA.split(";")[0].replace("IF NOT EXISTS pbp_events", "pbp_events_rebuild")
    conn.execute("DROP TABLE IF EXISTS pbp_events_rebuild")
    conn.execute(table_ddl)
    conn.commit()

    failures, written = [], 0
    for i, gid in enumerate(games, 1):
        try:
            actions = load_cache_file(cached(gid)[0][0]).get("game", {}).get("actions", []) or []
            rows = rows_for(gid, actions)
            if not rows:
                raise ValueError("no actions in the cached payload")
            conn.executemany(insert_sql("pbp_events_rebuild"), rows)
            written += len(rows)
        except Exception as exc:
            failures.append(gid)
            logger.error("%s FAILED: %s", gid, exc)
        if i % 250 == 0 or i == len(games):
            conn.commit()
            logger.info("[%d/%d] %s rows so far", i, len(games), format(written, ","))
    conn.commit()
    if failures:
        logger.error("%d game(s) failed; pbp_events is untouched: %s", len(failures), ", ".join(failures[:50]))
        return 1

    backup = BACKUP_PREFIX + time.strftime("%Y%m%d")
    started = time.time()
    # An explicit BEGIN: Python's sqlite3 opens transactions only for DML, so
    # without it each DDL statement below would commit on its own and a
    # failure halfway could leave no pbp_events at all.
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Index names are global in SQLite: the old table's are dropped with
        # the swap and the canonical names are recreated on the new one.
        for ix in ("idx_pbp_game", "idx_pbp_person", "idx_pbp_type"):
            conn.execute(f"DROP INDEX IF EXISTS {ix}")
        conn.execute(f"DROP TABLE IF EXISTS {backup}")
        conn.execute(f"ALTER TABLE pbp_events RENAME TO {backup}")
        conn.execute("ALTER TABLE pbp_events_rebuild RENAME TO pbp_events")
        for stmt in SCHEMA.split(";")[1:]:
            if stmt.strip():
                conn.execute(stmt)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    logger.info("Swapped in %s rows for %d games (%.1f s). Old table kept as %s - drop it once verified.",
                format(written, ","), len(games), time.time() - started, backup)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill NBA play-by-play into pbp_events.")
    ap.add_argument("--season", help="Only this season, e.g. 2025-26.")
    ap.add_argument("--limit", type=int, help="Stop after this many games.")
    ap.add_argument("--dry-run", action="store_true", help="Report what is missing and exit.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-derive every held game from the disk cache (no network) and swap the table in.")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # The API writes to this file too; wait for the lock rather than dying.
    conn.execute("PRAGMA busy_timeout = 30000")

    if args.rebuild:
        if args.season or args.limit:
            # The swap is whole-table; a partial rebuild would drop every
            # game it skipped.
            logger.error("--rebuild covers every held game; it takes no --season or --limit.")
            conn.close()
            return 1
        rc = rebuild_from_cache(conn)
        conn.close()
        return rc

    # A table from before the 2026-09-27 key fix has no action_id and lacks
    # every blocked miss and stolen turnover. Appending to it would mix
    # complete and incomplete games, so stop and say what to run.
    if not has_action_id(conn):
        logger.error("pbp_events predates the action_id key fix; run backfill_pbp.py --rebuild first.")
        conn.close()
        return 1
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

        try:
            write_game(conn, rows)
        except sqlite3.IntegrityError as exc:
            failures += 1
            logger.error("[%d/%d] %s FAILED: two actions share a key (%s)", i, len(todo), gid, exc)
            continue
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
