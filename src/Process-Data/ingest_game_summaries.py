"""
ingest_game_summaries.py
========================
Line scores, inactive players and game info from the boxscoresummaryv2
responses ALREADY ON DISK. No network calls: this reads the permanent cache
that backfill_officials.py filled (one JSON per game, 2003-04 onward) and
writes three tables. Run it any time after the officials backfill; it is
idempotent (INSERT OR REPLACE) and takes a few minutes for ~29,000 files.

Tables
------
game_line_scores   one row per team per game: points by quarter and up to
                   ten overtimes, plus the final. This is nba.com's own line
                   score, so it also covers games with no play-by-play.
game_inactives     one row per inactive player per game (the "Inactive:" line
                   on a box score). Empty for most games before 2005-06; the
                   feed simply did not carry it then. Missing means unknown,
                   not "everyone played".
game_info          attendance, game duration ("2:13" = 2 h 13 min), national
                   TV broadcaster, home and visitor ids.

Usage:
    python src/Process-Data/ingest_game_summaries.py            # everything cached
    python src/Process-Data/ingest_game_summaries.py --limit 500
"""

import argparse
import glob
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Utils.nba_stats_client import NBAStatsClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_game_summaries")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
CACHE_GLOB = os.path.join(REPO_ROOT, "Data", "nba_cache", "boxscoresummaryv2_game_id=*.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS game_line_scores (
    game_id    TEXT NOT NULL,
    team_id    INTEGER NOT NULL,
    team_abbr  TEXT,
    q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER,
    ot1 INTEGER, ot2 INTEGER, ot3 INTEGER, ot4 INTEGER, ot5 INTEGER,
    ot6 INTEGER, ot7 INTEGER, ot8 INTEGER, ot9 INTEGER, ot10 INTEGER,
    pts INTEGER,
    PRIMARY KEY (game_id, team_id)
);
CREATE TABLE IF NOT EXISTS game_inactives (
    game_id    TEXT NOT NULL,
    team_id    INTEGER,
    team_abbr  TEXT,
    player_id  INTEGER NOT NULL,
    first_name TEXT,
    last_name  TEXT,
    jersey_num TEXT,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_game_inactives_player ON game_inactives(player_id);
CREATE TABLE IF NOT EXISTS game_info (
    game_id          TEXT PRIMARY KEY,
    attendance       INTEGER,
    game_time        TEXT,
    natl_tv          TEXT,
    home_team_id     INTEGER,
    visitor_team_id  INTEGER,
    ingested_at      TEXT
);
"""

OT_COLS = [f"PTS_OT{i}" for i in range(1, 11)]


def _int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="Stop after N files (0 = all)")
    p.add_argument("--db", default=DB_PATH)
    args = p.parse_args()

    files = sorted(glob.glob(CACHE_GLOB))
    if args.limit:
        files = files[: args.limit]
    logger.info("%d cached summaries to read", len(files))

    conn = sqlite3.connect(args.db, timeout=60)
    conn.executescript(SCHEMA)
    known = {r[0] for r in conn.execute("SELECT game_id FROM box_scores")}

    now = datetime.now(timezone.utc).isoformat()
    n_files = n_lines = n_inactive = n_info = n_skipped = n_bad = 0
    batch_lines, batch_inact, batch_info = [], [], []

    def flush():
        if batch_lines:
            conn.executemany(
                "INSERT OR REPLACE INTO game_line_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch_lines)
        if batch_inact:
            conn.executemany("INSERT OR REPLACE INTO game_inactives VALUES (?,?,?,?,?,?,?)", batch_inact)
        if batch_info:
            conn.executemany("INSERT OR REPLACE INTO game_info VALUES (?,?,?,?,?,?,?)", batch_info)
        conn.commit()
        batch_lines.clear(); batch_inact.clear(); batch_info.clear()

    for path in files:
        gid = os.path.basename(path)[len("boxscoresummaryv2_game_id="):-len(".json")]
        if gid not in known:
            n_skipped += 1
            continue
        try:
            raw = json.load(open(path, encoding="utf-8"))
            parsed = NBAStatsClient._parse_all_result_sets(raw)
        except Exception as exc:  # a corrupt cache file must not end the run
            n_bad += 1
            logger.warning("unreadable cache for %s: %s", gid, str(exc)[:80])
            continue
        n_files += 1

        for r in parsed.get("LineScore") or []:
            if _int(r.get("TEAM_ID")) is None:
                continue  # a handful of rows carry no team id; nothing to attach them to
            batch_lines.append((
                gid, _int(r.get("TEAM_ID")), r.get("TEAM_ABBREVIATION"),
                _int(r.get("PTS_QTR1")), _int(r.get("PTS_QTR2")), _int(r.get("PTS_QTR3")), _int(r.get("PTS_QTR4")),
                *[_int(r.get(c)) for c in OT_COLS],
                _int(r.get("PTS")),
            ))
            n_lines += 1

        for r in parsed.get("InactivePlayers") or []:
            pid = _int(r.get("PLAYER_ID"))
            if pid is None:
                continue
            batch_inact.append((
                gid, _int(r.get("TEAM_ID")), r.get("TEAM_ABBREVIATION"), pid,
                r.get("FIRST_NAME"), r.get("LAST_NAME"),
                (r.get("JERSEY_NUM") or "").strip() or None,
            ))
            n_inactive += 1

        info = (parsed.get("GameInfo") or [{}])[0]
        summ = (parsed.get("GameSummary") or [{}])[0]
        if info or summ:
            batch_info.append((
                gid, _int(info.get("ATTENDANCE")), (info.get("GAME_TIME") or "").strip() or None,
                summ.get("NATL_TV_BROADCASTER_ABBREVIATION"),
                _int(summ.get("HOME_TEAM_ID")), _int(summ.get("VISITOR_TEAM_ID")), now,
            ))
            n_info += 1

        if len(batch_lines) >= 2000:
            flush()
            logger.info("%d files read, %d line rows, %d inactives, %d info rows", n_files, n_lines, n_inactive, n_info)

    flush()
    logger.info("SUMMARY files=%d line_rows=%d inactives=%d info=%d skipped_not_in_archive=%d unreadable=%d",
                n_files, n_lines, n_inactive, n_info, n_skipped, n_bad)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
