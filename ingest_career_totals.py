"""
ingest_career_totals.py
=======================
Builds career totals for every player who appeared between 1996-97 and now, by
summing one bulk request per season instead of one request per player.

Thirty requests instead of five thousand. leaguedashplayerstats returns every
player's season totals in a single call, so summing across seasons per player id
gives career games, minutes and points - enough to ask what a draft slot has
actually been worth.

The window is the whole point and also the limitation. A career is only complete
in this table if it started in 1996-97 or later; someone drafted in 1990 has the
first half of his career outside the window and would look worse than he was.
`first_season_in_window` is stored so a caller can exclude anyone whose career
may be clipped, rather than quietly comparing a truncated total to a full one.

    venv/Scripts/python.exe ingest_career_totals.py [--from 1996]
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("career_totals")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
FIRST = 1996
LAST = 2025

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=FIRST)
    ap.add_argument("--to", dest="end", type=int, default=LAST)
    args = ap.parse_args()

    from nba_api.stats.endpoints import leaguedashplayerstats

    agg: dict = {}
    seasons_done = 0
    for year in range(args.start, args.end + 1):
        season = f"{year}-{str(year + 1)[2:]}"
        try:
            d = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="Totals",
                timeout=90,
            ).get_dict()
        except Exception as exc:
            logger.warning("  %s: %s", season, str(exc)[:90])
            continue
        rs = d["resultSets"][0]
        idx = {h: i for i, h in enumerate(rs["headers"])}
        for r in rs["rowSet"]:
            pid = r[idx["PLAYER_ID"]]
            if pid is None:
                continue
            a = agg.setdefault(pid, {
                "name": r[idx["PLAYER_NAME"]], "seasons": 0, "gp": 0, "min": 0.0,
                "pts": 0, "reb": 0, "ast": 0, "first": year, "last": year,
            })
            a["seasons"] += 1
            a["gp"] += r[idx["GP"]] or 0
            a["min"] += r[idx["MIN"]] or 0.0
            a["pts"] += r[idx["PTS"]] or 0
            a["reb"] += r[idx["REB"]] or 0
            a["ast"] += r[idx["AST"]] or 0
            a["first"] = min(a["first"], year)
            a["last"] = max(a["last"], year)
        seasons_done += 1
        if year % 5 == 0 or year == args.end:
            logger.info("  %s done (%d players so far)", season, len(agg))

    if seasons_done < (args.end - args.start) * 0.8:
        logger.error(
            "Only %d of %d seasons returned data. Refusing to write a career table "
            "built on a partial window.", seasons_done, args.end - args.start + 1
        )
        return 1

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(SCHEMA)
        conn.executemany(
            """
            INSERT INTO player_career_span (
                player_id, player_name, seasons, gp, min, pts, reb, ast,
                first_season_in_window, last_season_in_window,
                window_first, window_last, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(player_id) DO UPDATE SET
                player_name=excluded.player_name, seasons=excluded.seasons,
                gp=excluded.gp, min=excluded.min, pts=excluded.pts,
                reb=excluded.reb, ast=excluded.ast,
                first_season_in_window=excluded.first_season_in_window,
                last_season_in_window=excluded.last_season_in_window,
                window_first=excluded.window_first, window_last=excluded.window_last,
                fetched_at=excluded.fetched_at
            """,
            [
                (pid, a["name"], a["seasons"], a["gp"], round(a["min"], 1), a["pts"],
                 a["reb"], a["ast"], a["first"], a["last"], args.start, args.end, now)
                for pid, a in agg.items()
            ],
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM player_career_span").fetchone()[0]
        logger.info(
            "player_career_span: %d players from %d seasons (%d-%d).",
            total, seasons_done, args.start, args.end,
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
