"""
team_stats_from_archive.py
==========================
The team-stats snapshot the model predicts from, rebuilt from our own box
scores instead of fetched from stats.nba.com.

WHY (2026-09-28)
refresh_team_stats.py asks nba.com's league dashboard (leaguedashteamstats)
for season-to-date team averages every morning. With this PC blocked by
nba.com's edge and the outbound breaker open, that request is refused, and
the newest snapshot would silently age through opening night. Every number in
that table is arithmetic on box scores we already hold, so it can be rebuilt
here with no network at all.

HOW EXACT (measured 2026-09-28, no network: every stored nba.com snapshot,
4,015 non-empty tables from 2007-08 to 2025-26, 6.24 million cells)
  - 99.966% of cells identical; 3,547 of 4,015 tables identical in every cell;
    13 of the 18 seasons identical in every cell of every table.
  - The other 2,152 cells are 0.1 in a per-game average, 0.001 in a
    percentage, or a rank one or two places off beside them: one unit in one
    game. 2,007 of them are in 2022-23 and 2023-24, the seasons whose
    snapshots were fetched day by day as the games happened, and they are
    nearly all gone by the season's final table (8 cells remain across all 18
    final tables). That is what a later nba.com stat correction looks like:
    the dashboard was read before the correction, the box score we hold after
    it. The 8 that remain are nba.com's dashboard disagreeing with nba.com's
    own box scores by one unit over a season (e.g. PHI 2025-26 defensive
    rebounds: the box scores sum to 2,612, the dashboard's 31.8 a game means
    at most 2,611). No rebuild from box scores can reproduce those.
  - Three rules make that possible, each checked against the stored tables:
    minutes come from the advanced box score's team clock (a '239:60' clock is
    a fraction of a second short of 240, which is how nba.com ranks it);
    ranks are taken on unrounded values, ties sharing the best rank; values
    are rounded half away from zero (nba.com shows -5.25 as -5.3).

The sealed evaluation (backtest_model.build_snapshots) used the same
arithmetic with plain half-up rounding and the traditional box score's clock;
the two differ only in those two details (99.932% vs 99.966% of cells).

PROVENANCE
The table written is shaped exactly like nba.com's (the column contract is
read from the newest existing snapshot, as refresh_team_stats does), so the
model cannot tell them apart; `team_stats_snapshot_source` records which
source wrote each table, how many games it covers and how many of those came
from ESPN rather than nba.com (src/Utils/espn_boxscore.py). A table with no
row there was written by refresh_team_stats.py from nba.com.

It never replaces a snapshot nba.com wrote. daily_update.py calls it only
when refresh_team_stats.py failed.

Usage:
    venv/Scripts/python.exe team_stats_from_archive.py                 # today's table
    venv/Scripts/python.exe team_stats_from_archive.py --as-of 2026-04-01 --dry-run
    venv/Scripts/python.exe team_stats_from_archive.py --compare 2024-04-29
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.Utils import espn_boxscore  # noqa: E402

logger = logging.getLogger("team_stats_from_archive")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
SOURCE_TABLE = "team_stats_snapshot_source"
SOURCE_ARCHIVE = "archive rebuild"
_SNAPSHOT_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEASON = re.compile(r"^\d{4}-\d{2}$")

_bm = None


def _harness():
    """backtest_model.py, loaded once: its box-score reader and column lists
    are the ones the sealed evaluation used, so they are reused, not copied."""
    global _bm
    if _bm is None:
        spec = importlib.util.spec_from_file_location("bb_backtest_model_ts",
                                                      os.path.join(REPO_ROOT, "backtest_model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _bm = mod
    return _bm


def current_season(today: date) -> str:
    """Same rule as refresh_team_stats: October opens a new season."""
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def round_half_away(x, nd: int):
    """nba.com's display rounding: half away from zero (-5.25 -> -5.3)."""
    f = 10.0 ** nd
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) * f + 0.5) / f


def _clock_minutes(clock: str) -> float:
    """'240:00' -> 240.0. A ':60' clock is a fraction of a second short of the
    next minute (nba.com rounds seconds for display), so it sorts just below."""
    mm, ss = str(clock).split(":")[:2]
    v = float(mm) + float(ss) / 60.0
    return v - 1e-6 if ss.strip() == "60" else v


def load_season_team_games(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """One row per (game, team) for `season`: nba.com box scores read by the
    sealed harness's own reader, with team minutes taken from the advanced box
    score's clock; then any usable ESPN games the archive lacks. Carries a
    `source` column ('nba.com' / 'espn')."""
    if not _SEASON.match(season or ""):
        raise ValueError(f"not a season label: {season!r}")
    bm = _harness()
    # The harness reads `box_scores` whole. A TEMP view of that name shadows
    # the table on this connection only, so the same reader sees one season.
    conn.execute("DROP VIEW IF EXISTS temp.box_scores")
    conn.execute(f"CREATE TEMP VIEW box_scores AS SELECT * FROM main.box_scores WHERE season = '{season}'")
    try:
        if conn.execute("SELECT COUNT(*) FROM box_scores").fetchone()[0]:
            tg = bm.load_team_games(conn)
        else:   # a season nba.com has supplied nothing for (yet)
            tg = pd.DataFrame(columns=list(espn_boxscore.TEAM_GAME_COLUMNS) + ["W", "L", "PLUS_MINUS"])
        clocks = {}
        for gid, aj in conn.execute("SELECT game_id, advanced_json FROM box_scores"):
            a = json.loads(aj)["boxScoreAdvanced"]
            for side in ("homeTeam", "awayTeam"):
                clocks[(gid, int(a[side]["teamId"]))] = _clock_minutes(a[side]["statistics"]["minutes"])
    finally:
        conn.execute("DROP VIEW IF EXISTS temp.box_scores")
    if len(tg):
        tg["MIN"] = [clocks.get((g, int(t)), m) for g, t, m in zip(tg.game_id, tg.team_id, tg.MIN)]
    tg["source"] = "nba.com"
    return espn_boxscore.merge_team_games(conn, tg, season=season)


def build_snapshot(tg: pd.DataFrame, season: str, as_of: str) -> Optional[pd.DataFrame]:
    """The 52 model columns per team (index TEAM_ID) from `season`'s
    regular-season games played strictly before `as_of` (the table is named
    for the day it is used and holds games through the day before, exactly as
    refresh_team_stats asks nba.com for DateTo = yesterday). None if no games."""
    bm = _harness()
    reg = tg[(tg.season == season) & (tg.season_type == "Regular Season") & (tg.game_date < as_of)]
    if reg.empty:
        return None
    g = reg.groupby("team_id")[bm.SUM_COLS].sum()
    g["GP"] = reg.groupby("team_id").size()
    raw = pd.DataFrame(index=g.index)
    raw["GP"] = g.GP.astype(float)
    raw["W"] = g.W.astype(float)
    raw["L"] = g.L.astype(float)
    raw["W_PCT"] = g.W / g.GP
    raw["MIN"] = g.MIN / 5.0 / g.GP          # team minutes -> nba.com's 48.x per game
    for c in bm.PER_GAME:
        raw[c] = g[c] / g.GP
    raw["FG_PCT"] = g.FGM / g.FGA
    raw["FG3_PCT"] = g.FG3M / g.FG3A
    raw["FT_PCT"] = g.FTM / g.FTA
    raw = raw[bm.BASE]
    out = pd.DataFrame(index=raw.index)
    for c in bm.BASE:
        # Ranks on the unrounded values; ties share the best rank.
        out[c + "_RANK"] = raw[c].rank(ascending=(c in bm.RANK_ASCENDING), method="min").astype(int)
    for c in bm.BASE:
        nd = 3 if c.endswith("_PCT") else (0 if c in ("GP", "W", "L") else 1)
        out[c] = round_half_away(raw[c], nd)
    for c in ("GP", "W", "L"):
        out[c] = out[c].astype(int)
    out.index.name = "TEAM_ID"
    return out[bm.TEAM_BLOCK]


def newest_snapshot(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name GLOB "
                       "'[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]' ORDER BY name DESC LIMIT 1").fetchone()
    return row[0] if row else None


def column_contract(conn: sqlite3.Connection, table: str) -> List[tuple]:
    """[(name, declared type)] of an existing nba.com snapshot, in order."""
    return [(r[1], r[2]) for r in conn.execute(f'PRAGMA table_info("{table}")')]


def team_names(conn: sqlite3.Connection, reference: str) -> Dict[int, str]:
    """TEAM_NAME exactly as nba.com's dashboard spells it ('LA Clippers'),
    from the newest snapshot; team_metadata only for a team it lacks."""
    names = {int(t): n for t, n in conn.execute(f'SELECT TEAM_ID, TEAM_NAME FROM "{reference}"')}
    for t, n in conn.execute("SELECT team_id, full_name FROM team_metadata"):
        names.setdefault(int(t), n)
    return names


def snapshot_rows(snap: pd.DataFrame, names: Dict[int, str], table_name: str,
                  contract: List[tuple]) -> pd.DataFrame:
    """The frame in the contract's column order, rows ordered by TEAM_NAME
    (as nba.com returns them), `index` 0..29 and `Date` = the table's name."""
    df = snap.copy()
    df.insert(0, "TEAM_NAME", [names.get(int(t)) for t in df.index])
    if df.TEAM_NAME.isna().any():
        raise RuntimeError(f"no team name for team id(s) {list(df.index[df.TEAM_NAME.isna()])}")
    df = df.reset_index().sort_values("TEAM_NAME").reset_index(drop=True)
    df["Date"] = table_name
    df["index"] = range(len(df))
    cols = [c for c, _ in contract]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"the rebuild has no column(s) {missing} that the models expect; refusing to write")
    return df[cols]


def ensure_source_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
            table_name    TEXT PRIMARY KEY,   -- the dated snapshot table
            source        TEXT NOT NULL,      -- 'archive rebuild' (no row = nba.com via refresh_team_stats)
            season        TEXT NOT NULL,
            games_through TEXT,               -- newest game date counted
            n_games       INTEGER NOT NULL,   -- regular-season games counted
            n_games_espn  INTEGER NOT NULL,   -- of which box scores came from ESPN, not nba.com
            built_at      TEXT NOT NULL       -- UTC ISO
        )""")


def snapshot_source(conn: sqlite3.Connection, table: str) -> Dict[str, object]:
    """Where a dated snapshot came from."""
    try:
        r = conn.execute(f"SELECT source, season, games_through, n_games, n_games_espn, built_at "
                         f"FROM {SOURCE_TABLE} WHERE table_name=?", (table,)).fetchone()
    except sqlite3.OperationalError:
        r = None
    if not r:
        return {"table": table, "source": "nba.com leaguedashteamstats (refresh_team_stats.py)"}
    return {"table": table, "source": r[0], "season": r[1], "games_through": r[2],
            "n_games": r[3], "n_games_espn": r[4], "built_at": r[5]}


def refresh(as_of: Optional[date] = None, season: Optional[str] = None,
            db_path: str = DB_PATH, dry_run: bool = False) -> Optional[str]:
    """Write the dated snapshot for `as_of` (default today) from the archive.

    Returns the table name, or None when the season has no completed
    regular-season game yet (not a failure, same as refresh_team_stats).
    Never overwrites a table nba.com wrote; an earlier archive rebuild of the
    same day is replaced (it may be missing games that have landed since).
    """
    as_of = as_of or date.today()
    season = season or current_season(as_of)
    table_name = as_of.strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        reference = newest_snapshot(conn)
        if reference is None:
            raise RuntimeError("No existing team-stats snapshot to take the column contract from; "
                               "refusing to invent a schema the models were not trained on.")
        existing = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                (table_name,)).fetchone()
        if existing and snapshot_source(conn, table_name)["source"] != SOURCE_ARCHIVE:
            logger.info("Snapshot %s was written from nba.com; the archive rebuild leaves it alone.", table_name)
            return table_name
        if existing and reference == table_name:
            # The contract must come from an nba.com table, never from a rebuild.
            reference = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB "
                "'[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]' AND name < ? ORDER BY name DESC LIMIT 1",
                (table_name,)).fetchone()[0]
        contract = column_contract(conn, reference)

        tg = load_season_team_games(conn, season)
        snap = build_snapshot(tg, season, table_name)
        if snap is None:
            logger.info("No %s regular-season games before %s in the archive. Nothing written.",
                        season, table_name)
            return None
        reg = tg[(tg.season == season) & (tg.season_type == "Regular Season") & (tg.game_date < table_name)]
        n_games = int(reg.game_id.nunique())
        n_espn = int(reg[reg.source == espn_boxscore.SOURCE].game_id.nunique())
        through = str(reg.game_date.max())
        if len(snap) != 30:
            logger.warning("Archive snapshot %s covers %d team(s), not 30.", table_name, len(snap))
        df = snapshot_rows(snap, team_names(conn, reference), table_name, contract)
        if dry_run:
            logger.info("Dry run: %s would hold %d teams from %d %s games through %s (%d from ESPN).",
                        table_name, len(df), n_games, season, through, n_espn)
            return table_name

        cols_sql = ", ".join(f'"{c}" {t}' for c, t in contract)
        placeholders = ", ".join("?" * len(contract))
        values = [[None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row]
                  for row in df.itertuples(index=False, name=None)]
        conn.execute("BEGIN IMMEDIATE")
        try:
            ensure_source_table(conn)
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            conn.execute(f'CREATE TABLE "{table_name}" ({cols_sql})')
            conn.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)
            conn.execute(f"INSERT OR REPLACE INTO {SOURCE_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (table_name, SOURCE_ARCHIVE, season, through, n_games, n_espn,
                          datetime.now(timezone.utc).isoformat()))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        logger.warning("Wrote team-stats snapshot '%s' FROM THE ARCHIVE (not nba.com): %d teams, "
                       "%d %s games through %s, %d of them from ESPN box scores.",
                       table_name, len(df), n_games, season, through, n_espn)
        return table_name
    finally:
        conn.close()


def compare(table: str, db_path: str = DB_PATH, season: Optional[str] = None) -> Dict[str, object]:
    """Rebuild a stored nba.com snapshot from the archive and diff it, cell by cell."""
    if not _SNAPSHOT_NAME.match(table):
        raise ValueError(f"not a snapshot table name: {table!r}")
    bm = _harness()
    conn = sqlite3.connect("file:" + db_path.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        stored = pd.read_sql_query(f'SELECT * FROM "{table}"', conn).set_index("TEAM_ID")
        if season is None:
            starts = dict(conn.execute("SELECT season, MIN(game_date) FROM box_scores "
                                       "WHERE season_type='Regular Season' GROUP BY season").fetchall())
            season = max((s for s, d in starts.items() if d < table), default=None)
        tg = load_season_team_games(conn, season)
    finally:
        conn.close()
    rebuilt = build_snapshot(tg, season, table)
    if rebuilt is None or stored.empty:
        return {"table": table, "season": season, "stored_rows": len(stored),
                "rebuilt_rows": 0 if rebuilt is None else len(rebuilt), "cells": 0, "mismatches": []}
    common = rebuilt.index.intersection(stored.index)
    mism = []
    for c in bm.TEAM_BLOCK:
        a = stored.loc[common, c].astype(float).values
        b = rebuilt.loc[common, c].astype(float).values
        for i in np.where(np.abs(a - b) >= 1e-9)[0]:
            mism.append({"team_id": int(common[i]), "column": c, "stored": float(a[i]), "rebuilt": float(b[i])})
    n = len(common) * len(bm.TEAM_BLOCK)
    return {"table": table, "season": season, "teams": len(common),
            "team_sets_equal": set(common) == set(stored.index) == set(rebuilt.index),
            "cells": n, "exact": n - len(mism), "mismatches": mism}


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Rebuild the team-stats snapshot from the box-score archive.")
    ap.add_argument("--as-of", help="YYYY-MM-DD: the day the table is for (games through the day before)")
    ap.add_argument("--season", help="YYYY-YY (default: from --as-of)")
    ap.add_argument("--dry-run", action="store_true", help="build and report, write nothing")
    ap.add_argument("--compare", metavar="TABLE", help="diff a stored nba.com snapshot against its rebuild")
    args = ap.parse_args(argv)
    if args.compare:
        res = compare(args.compare, season=args.season)
        print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1))
        for m in res["mismatches"][:40]:
            print("  ", m)
        return 0
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
    try:
        refresh(as_of=as_of, season=args.season, dry_run=args.dry_run)
    except Exception as exc:
        logger.error("Archive team-stats rebuild failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
