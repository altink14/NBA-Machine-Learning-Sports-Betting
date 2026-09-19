"""
ingest_player_stats.py (NFL)
============================
Per-player, per-week statistics from the nflverse-data `stats_player` release
(CC-BY-4.0), 1999 to date: passing, rushing, receiving, defence, kicking,
punting and returns, plus the expected-points and share metrics nflverse
derives from its own play-by-play.

The source carries 150 columns. We keep 74, listed in KEEP below, which is
every counting stat a box score or a leaderboard needs plus the derived
columns that are genuinely useful (EPA by phase, target share, air-yards
share, completion percentage over expected). Dropped: the headshot URL, the
fantasy-scoring columns, the field-goal distance list columns, and the
long-tail special-teams breakdowns. Nothing is lost permanently; the release
is one re-run away.

DERIVED COLUMNS ARE NFLVERSE'S, NOT OURS. `passing_epa`, `racr`, `wopr`,
`target_share` and their neighbours come out of nflfastR's models and
conventions. They are good and widely used, and they are somebody else's
method. Any page that shows them says so, exactly as the play-by-play pages
must say that `vegas_wp` is nflfastR's win probability rather than ours. Our
own numbers are the ones we compute from `nfl_plays`.

The table is keyed (game_id, player_id) rather than (season, week, player_id)
so it joins to everything else in the archive without a three-column dance,
and so a player traded mid-week cannot collide with himself.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/ingest_player_stats.py
    venv/Scripts/python.exe src/Sports/nfl/ingest_player_stats.py --season 2024
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.ingest_player_stats")

SOURCE = "nflverse-data stats_player (CC-BY-4.0)"
ENDPOINT_TMPL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                 "stats_player/stats_player_week_{season}.csv")
DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
UA = {"User-Agent": "BettingBuddy/1.0 (archive ingest)"}
FIRST_SEASON = 1999

csv.field_size_limit(10_000_000)

KEEP = [
    "player_id", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "game_id", "team", "opponent_team",
    # passing
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "sack_yards_lost", "passing_air_yards", "passing_yards_after_catch",
    "passing_first_downs", "passing_epa", "passing_cpoe", "passing_2pt_conversions",
    # rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions",
    # receiving
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_fumbles_lost", "receiving_air_yards", "receiving_yards_after_catch",
    "receiving_first_downs", "receiving_epa", "receiving_2pt_conversions",
    "racr", "target_share", "air_yards_share", "wopr",
    # defence
    "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss", "def_fumbles_forced",
    "def_sacks", "def_sack_yards", "def_qb_hits", "def_interceptions",
    "def_interception_yards", "def_pass_defended", "def_tds", "def_safeties",
    # kicking and punting
    "fg_made", "fg_att", "fg_missed", "fg_blocked", "fg_long", "fg_pct",
    "pat_made", "pat_att", "gwfg_made", "gwfg_att",
    "pt_att", "pt_yards", "pt_net_yards", "pt_long", "pt_inside_20", "pt_touchback",
    # returns and misc
    "punt_returns", "punt_return_yards", "kickoff_returns", "kickoff_return_yards",
    "special_teams_tds", "penalties", "penalty_yards", "fumbles_total", "fumbles_lost_total",
]

TEXT_COLS = {"player_id", "player_display_name", "position", "position_group", "season",
             "season_type", "game_id", "team", "opponent_team"}
FLOAT_COLS = {"passing_epa", "passing_cpoe", "rushing_epa", "receiving_epa", "racr",
              "target_share", "air_yards_share", "wopr", "fg_pct"}

_COLS = ["game_id", "player_id", "season", "week"] + [
    c for c in KEEP if c not in ("game_id", "player_id", "season", "week")]
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS nfl_player_stats_week (\n"
    + ",\n".join(
        f"    {c} " + ("TEXT" if c in TEXT_COLS else "REAL" if c in FLOAT_COLS else
                       "INTEGER" if c == "week" else "REAL")
        for c in _COLS)
    + ",\n    source TEXT,\n    fetched_at TEXT,\n    ingest_version INTEGER,\n"
      "    PRIMARY KEY (game_id, player_id)\n);\n"
      "CREATE INDEX IF NOT EXISTS idx_pstats_player ON nfl_player_stats_week(player_id);\n"
      "CREATE INDEX IF NOT EXISTS idx_pstats_season ON nfl_player_stats_week(season, week);\n"
      "CREATE INDEX IF NOT EXISTS idx_pstats_team   ON nfl_player_stats_week(team);\n"
)
_INSERT = (f"INSERT OR REPLACE INTO nfl_player_stats_week ({', '.join(_COLS)}, source, "
           f"fetched_at, ingest_version) VALUES ({', '.join('?' * (len(_COLS) + 3))})")


def _val(v: str, col: str) -> Any:
    if v is None or v == "" or v == "NA":
        return None
    if col in TEXT_COLS:
        return v.strip() or None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest NFL per-player weekly statistics.")
    ap.add_argument("--season", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=120)
    conn.row_factory = sqlite3.Row
    ensure_core_schema(conn)
    conn.executescript(SCHEMA)
    conn.commit()

    last = max(int(r[0]) for r in conn.execute("SELECT DISTINCT season FROM games"))
    seasons = [args.season] if args.season else list(range(FIRST_SEASON, last + 1))
    done = ({r[0] for r in conn.execute("SELECT DISTINCT season FROM nfl_player_stats_week")}
            if not args.force else set())
    todo = [s for s in seasons if str(s) not in done]
    logger.info("%d season(s) present; %d to do", len(done), len(todo))

    known_games = {r[0] for r in conn.execute("SELECT game_id FROM games")}
    started_at = datetime.now(timezone.utc).isoformat()
    total = 0
    persons: Dict[str, tuple] = {}

    for season in todo:
        url = ENDPOINT_TMPL.format(season=season)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("  %d unavailable: %s", season, str(exc)[:120])
            continue

        fetched_at = datetime.now(timezone.utc).isoformat()
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if not header:
            continue
        idx = {n: i for i, n in enumerate(header)}
        missing = [c for c in KEEP if c not in idx]
        if missing:
            logger.warning("  %d lacks %d expected column(s): %s", season, len(missing),
                           ", ".join(missing[:6]))

        rows, orphan = [], 0
        for raw in reader:
            if not raw:
                continue
            vals = {c: (_val(raw[idx[c]], c) if c in idx and idx[c] < len(raw) else None)
                    for c in KEEP}
            gid, pid = vals.get("game_id"), vals.get("player_id")
            if not gid or not pid:
                continue
            if gid not in known_games:
                orphan += 1
                continue
            wk = vals.get("week")
            ordered = [gid, pid, vals.get("season") or str(season),
                       int(wk) if wk is not None else None]
            ordered += [vals.get(c) for c in _COLS[4:]]
            rows.append(tuple(ordered) + (SOURCE, fetched_at, INGEST_VERSION))
            if pid not in persons:
                nm = vals.get("player_display_name")
                persons[pid] = (pid, "football", "player", nm,
                                (nm or "").split(" ")[0] or None,
                                " ".join((nm or "").split(" ")[1:]) or None,
                                None, None, json.dumps({"gsis_id": pid}),
                                SOURCE, url, fetched_at, INGEST_VERSION)

        conn.executemany(_INSERT, rows)
        conn.commit()
        total += len(rows)
        logger.info("  %d: %d player-games%s", season, len(rows),
                    f", {orphan} skipped (game not in schedule)" if orphan else "")

    if persons:
        conn.executemany("INSERT OR REPLACE INTO persons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         list(persons.values()))
        conn.commit()

    record_run(conn, "nfl_player_stats_week", SOURCE, ENDPOINT_TMPL.format(season="{season}"),
               started_at, datetime.now(timezone.utc).isoformat(), total,
               notes=f"seasons={','.join(str(s) for s in todo)} players={len(persons)}")

    grand = conn.execute("SELECT COUNT(*) FROM nfl_player_stats_week").fetchone()[0]
    nplayers = conn.execute("SELECT COUNT(DISTINCT player_id) FROM nfl_player_stats_week").fetchone()[0]
    logger.info("SUMMARY this_run=%d archive=%d distinct_players=%d", total, grand, nplayers)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
