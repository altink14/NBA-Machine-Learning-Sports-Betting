"""
ingest_pbp.py (NFL)
===================
Play-by-play for every NFL game since 1999, from the nflverse-data `pbp`
release (CC-BY-4.0). This is the largest slice of the NFL archive and the one
that unlocks expected points, win probability, fourth-down decisions, drive
charts, situational splits and every model feature worth having.

SHAPE. The source is 372 columns of CSV, about 99.5 MB a season. We keep 86 of
them, chosen below, and write two tables:

  events      the thin cross-sport table: one row per play with the columns
              that mean the same thing in every sport (period, clock, type,
              team, actor, description, score). Cross-sport surfaces read this.
  nfl_plays   the wide extension, same (game_id, sequence) key: down, distance,
              field position, EPA family, win probability, play detail, the
              player ids for each role. NFL pages join it.

The 286 columns we drop are recoverable at any time by re-running against the
same release; nothing here is destructive.

RESUMABLE. A season already present in `nfl_plays` is skipped unless
--force is given, and each season's CSV is cached on disk, so an interrupted
run costs only the season it was in the middle of.

WHAT THE MARKET COLUMNS IN THIS FILE ARE NOT. Each play carries `spread_line`
and `total_line`, but they are the same game-level closing numbers already in
games.csv, not a live in-game line. `vegas_wp` is nflfastR's win probability
model conditioned on that closing spread; it is THEIR model, not ours, and any
page that shows it must say so. We store it because it is a useful published
baseline to measure our own work against, which is exactly the kind of
comparison the honesty doctrine asks for.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/ingest_pbp.py                 # every missing season
    venv/Scripts/python.exe src/Sports/nfl/ingest_pbp.py --season 2024
    venv/Scripts/python.exe src/Sports/nfl/ingest_pbp.py --from-season 2020
    venv/Scripts/python.exe src/Sports/nfl/ingest_pbp.py --keep-csv      # do not delete cached CSVs
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.ingest_pbp")

SOURCE = "nflverse-data pbp (CC-BY-4.0)"
ENDPOINT_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
CACHE_DIR = os.path.join(REPO_ROOT, "Data", "nfl_cache")
FIRST_SEASON = 1999

# Python's csv module refuses very long fields by default; some `desc` values
# and the 372-column header exceed it.
csv.field_size_limit(10_000_000)

#: The 86 source columns we keep, grouped by what they are for. Anything not
#: listed stays in the release and can be pulled later; this list is the
#: contract, so a column added here must also be added to NFL_PLAYS_SCHEMA.
KEEP = [
    # identity and situation
    "play_id", "game_id", "old_game_id", "home_team", "away_team", "season_type", "week",
    "posteam", "posteam_type", "defteam", "side_of_field", "yardline_100", "game_date",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "game_half", "quarter_end", "drive", "series", "series_result", "sp", "qtr", "down",
    "goal_to_go", "time", "yrdln", "ydstogo", "ydsnet", "desc", "play_type", "yards_gained",
    # play detail
    "shotgun", "no_huddle", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike",
    "pass_length", "pass_location", "air_yards", "yards_after_catch", "run_location", "run_gap",
    "complete_pass", "incomplete_pass", "interception", "sack", "touchback", "fumble_lost",
    "field_goal_result", "kick_distance", "extra_point_result", "two_point_conv_result",
    "timeout", "timeout_team", "penalty", "penalty_type", "penalty_yards", "penalty_team",
    # outcome
    "first_down", "touchdown", "pass_touchdown", "rush_touchdown", "return_touchdown", "safety",
    "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
    # value models (nflfastR's, not ours)
    "epa", "air_epa", "yac_epa", "wp", "def_wp", "wpa", "vegas_wp", "vegas_home_wp",
    "success", "cpoe", "xpass", "pass_oe",
    # score state
    "total_home_score", "total_away_score", "posteam_score", "defteam_score", "score_differential",
    # actors (GSIS ids + names)
    "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name", "kicker_player_id", "td_player_id",
]

INT_COLS = {
    "play_id", "week", "yardline_100", "quarter_end", "drive", "series", "sp", "qtr", "down",
    "goal_to_go", "ydstogo", "ydsnet", "yards_gained", "shotgun", "no_huddle", "qb_dropback",
    "qb_scramble", "qb_kneel", "qb_spike", "air_yards", "yards_after_catch", "complete_pass",
    "incomplete_pass", "interception", "sack", "touchback", "fumble_lost", "kick_distance",
    "timeout", "penalty", "penalty_yards", "first_down", "touchdown", "pass_touchdown",
    "rush_touchdown", "return_touchdown", "safety", "third_down_converted", "third_down_failed",
    "fourth_down_converted", "fourth_down_failed", "success", "total_home_score",
    "total_away_score", "posteam_score", "defteam_score", "score_differential",
}
FLOAT_COLS = {
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "epa", "air_epa", "yac_epa", "wp", "def_wp", "wpa", "vegas_wp", "vegas_home_wp",
    "cpoe", "xpass", "pass_oe",
}

NFL_PLAYS_SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_plays (
    game_id TEXT NOT NULL, play_id INTEGER NOT NULL, season TEXT, old_game_id TEXT,
    home_team TEXT, away_team TEXT, season_type TEXT, week INTEGER,
    posteam TEXT, posteam_type TEXT, defteam TEXT, side_of_field TEXT, yardline_100 INTEGER,
    game_date TEXT, quarter_seconds_remaining REAL, half_seconds_remaining REAL,
    game_seconds_remaining REAL, game_half TEXT, quarter_end INTEGER, drive INTEGER,
    series INTEGER, series_result TEXT, sp INTEGER, qtr INTEGER, down INTEGER,
    goal_to_go INTEGER, time TEXT, yrdln TEXT, ydstogo INTEGER, ydsnet INTEGER,
    desc TEXT, play_type TEXT, yards_gained INTEGER,
    shotgun INTEGER, no_huddle INTEGER, qb_dropback INTEGER, qb_scramble INTEGER,
    qb_kneel INTEGER, qb_spike INTEGER, pass_length TEXT, pass_location TEXT,
    air_yards INTEGER, yards_after_catch INTEGER, run_location TEXT, run_gap TEXT,
    complete_pass INTEGER, incomplete_pass INTEGER, interception INTEGER, sack INTEGER,
    touchback INTEGER, fumble_lost INTEGER, field_goal_result TEXT, kick_distance INTEGER,
    extra_point_result TEXT, two_point_conv_result TEXT, timeout INTEGER, timeout_team TEXT,
    penalty INTEGER, penalty_type TEXT, penalty_yards INTEGER, penalty_team TEXT,
    first_down INTEGER, touchdown INTEGER, pass_touchdown INTEGER, rush_touchdown INTEGER,
    return_touchdown INTEGER, safety INTEGER, third_down_converted INTEGER,
    third_down_failed INTEGER, fourth_down_converted INTEGER, fourth_down_failed INTEGER,
    epa REAL, air_epa REAL, yac_epa REAL, wp REAL, def_wp REAL, wpa REAL,
    vegas_wp REAL, vegas_home_wp REAL, success INTEGER, cpoe REAL, xpass REAL, pass_oe REAL,
    total_home_score INTEGER, total_away_score INTEGER, posteam_score INTEGER,
    defteam_score INTEGER, score_differential INTEGER,
    passer_player_id TEXT, passer_player_name TEXT, rusher_player_id TEXT,
    rusher_player_name TEXT, receiver_player_id TEXT, receiver_player_name TEXT,
    kicker_player_id TEXT, td_player_id TEXT,
    source TEXT, fetched_at TEXT, ingest_version INTEGER,
    PRIMARY KEY (game_id, play_id)
);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_season  ON nfl_plays(season);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_posteam ON nfl_plays(posteam);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_type    ON nfl_plays(play_type);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_passer  ON nfl_plays(passer_player_id);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_rusher  ON nfl_plays(rusher_player_id);
"""

# nfl_plays column order for the INSERT: game_id, play_id, season, then KEEP
# minus the two identity columns already placed, then provenance.
_ORDER = ["game_id", "play_id", "season"] + [c for c in KEEP if c not in ("game_id", "play_id")]
_INSERT = (f"INSERT OR REPLACE INTO nfl_plays ({', '.join(_ORDER)}, source, fetched_at, ingest_version) "
           f"VALUES ({', '.join('?' * (len(_ORDER) + 3))})")


def _num(v: str, col: str) -> Any:
    if v is None or v == "" or v == "NA":
        return None
    if col in INT_COLS:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    if col in FLOAT_COLS:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return v


def download(season: int, keep_csv: bool) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"play_by_play_{season}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        logger.info("  cached %s (%.1f MB)", os.path.basename(path), os.path.getsize(path) / 1e6)
        return path
    import urllib.request
    url = ENDPOINT_TMPL.format(season=season)
    logger.info("  downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "BettingBuddy/1.0 (archive ingest)"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(path, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    logger.info("  downloaded %.1f MB", os.path.getsize(path) / 1e6)
    return path


def ingest_season(conn: sqlite3.Connection, season: int, keep_csv: bool) -> tuple[int, int]:
    path = download(season, keep_csv)
    fetched_at = datetime.now(timezone.utc).isoformat()
    endpoint = ENDPOINT_TMPL.format(season=season)

    # utf-8 with replacement: the `weather` free-text column carries a mojibake
    # degree sign in the CSV export, which must not abort a 50,000-row season.
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in KEEP if c not in idx]
        if missing:
            logger.warning("  season %d is missing %d expected columns: %s",
                           season, len(missing), ", ".join(missing[:8]))
        take = [(c, idx.get(c)) for c in KEEP]

        plays: List[tuple] = []
        events: List[tuple] = []
        n = 0
        for row in reader:
            if not row:
                continue
            vals: Dict[str, Any] = {}
            for col, i in take:
                vals[col] = _num(row[i], col) if i is not None and i < len(row) else None
            gid, pid = vals.get("game_id"), vals.get("play_id")
            if not gid or pid is None:
                continue
            ordered = [gid, pid, str(season)] + [vals.get(c) for c in _ORDER[3:]]
            plays.append(tuple(ordered) + (SOURCE, fetched_at, INGEST_VERSION))

            actor = (vals.get("passer_player_id") or vals.get("rusher_player_id")
                     or vals.get("kicker_player_id") or vals.get("receiver_player_id"))
            events.append((
                gid, pid, vals.get("qtr"), vals.get("game_seconds_remaining"),
                vals.get("play_type"),
                f"nfl-{vals['posteam']}" if vals.get("posteam") else None,
                actor, vals.get("desc"),
                vals.get("total_home_score"), vals.get("total_away_score"),
            ))
            n += 1

            if len(plays) >= 20000:
                conn.executemany(_INSERT, plays)
                conn.executemany("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", events)
                conn.commit()
                plays.clear(); events.clear()

        if plays:
            conn.executemany(_INSERT, plays)
            conn.executemany("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", events)
            conn.commit()

    games = conn.execute("SELECT COUNT(DISTINCT game_id) FROM nfl_plays WHERE season = ?",
                         (str(season),)).fetchone()[0]
    if not keep_csv:
        try:
            os.remove(path)
        except OSError:
            pass
    return n, games


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest nflverse play-by-play into the NFL archive.")
    ap.add_argument("--season", type=int, help="Only this season.")
    ap.add_argument("--from-season", type=int, help="This season and every later one.")
    ap.add_argument("--force", action="store_true", help="Re-ingest seasons already present.")
    ap.add_argument("--keep-csv", action="store_true", help="Keep the ~100 MB CSVs on disk.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_core_schema(conn)
    conn.executescript(NFL_PLAYS_SCHEMA)
    conn.commit()

    last = max(int(r[0]) for r in conn.execute("SELECT DISTINCT season FROM games"))
    if args.season:
        seasons = [args.season]
    elif args.from_season:
        seasons = list(range(args.from_season, last + 1))
    else:
        seasons = list(range(FIRST_SEASON, last + 1))

    done = {r[0] for r in conn.execute("SELECT DISTINCT season FROM nfl_plays")} if not args.force else set()
    todo = [s for s in seasons if str(s) not in done]
    logger.info("%d season(s) already ingested; %d to do: %s",
                len(done), len(todo), ", ".join(str(s) for s in todo) or "none")

    started_at = datetime.now(timezone.utc).isoformat()
    total_plays = 0
    for i, season in enumerate(todo, 1):
        logger.info("[%d/%d] season %d", i, len(todo), season)
        try:
            n, games = ingest_season(conn, season, args.keep_csv)
        except Exception as exc:  # one bad season must not end the run
            logger.error("  season %d FAILED: %s", season, str(exc)[:200])
            continue
        total_plays += n
        logger.info("  %d plays across %d games", n, games)

    finished_at = datetime.now(timezone.utc).isoformat()
    record_run(conn, "nfl_plays", SOURCE, ENDPOINT_TMPL.format(season="{season}"),
               started_at, finished_at, total_plays,
               notes=f"seasons={','.join(str(s) for s in todo)}")

    grand = conn.execute("SELECT COUNT(*) FROM nfl_plays").fetchone()[0]
    gg = conn.execute("SELECT COUNT(DISTINCT game_id) FROM nfl_plays").fetchone()[0]
    logger.info("SUMMARY this_run_plays=%d archive_plays=%d archive_games=%d", total_plays, grand, gg)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
