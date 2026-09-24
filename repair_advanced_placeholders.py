"""Repair player_season_advanced rows that hold placeholder zeros.

Found 2026-09-23. The backfill created every player_season_advanced row with
0.0 for usage, the ratings, AST%/REB% and pace, then filled them from
nba.com's leaguedashplayerstats matched on team. nba.com gives a traded player
ONE row for the whole season under his latest team, so:
  - his other stints kept 0.0s that read as real values (1,988 rows);
  - his latest stint was given whole-season numbers, labelled as that stint.
The backfill now writes NULL for anything unknown and fills those fields only
for single-team players (src/Process-Data/backfill.py). This repairs the rows
already written:
  1. a traded player's stints: the season-only fields become NULL (the API
     builds his season line from nba.com's whole-season row instead);
  2. any other row that was never filled (pace = 0, which no player who
     played has): the season-only fields become NULL;
  3. TS%, eFG% and TOV% stored as 0.0 for a stint with no attempts become NULL.

Dry run by default. --apply writes, after saving every row it changes to
Data/backups/advanced_placeholders_<timestamp>.json so it can be undone.
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "Data", "TeamData.sqlite")
SEASON_ONLY = ("usg_pct", "off_rating", "def_rating", "net_rating", "ast_pct", "reb_pct", "pace")

MULTI_TEAM = """
    (SELECT COUNT(*) FROM player_season_totals t
     WHERE t.player_id = a.player_id AND t.season = a.season AND t.season_type = a.season_type) > 1
"""
TRADED_WHERE = MULTI_TEAM + " AND (" + " OR ".join(f"a.{c} IS NOT NULL" for c in SEASON_ONLY) + ")"
UNFILLED_WHERE = "NOT " + MULTI_TEAM + " AND a.pace = 0 AND a.off_rating = 0 AND a.def_rating = 0"
NO_ATTEMPTS = {
    "ts_pct": "(t.fga + 0.44 * t.fta) = 0",
    "efg_pct": "t.fga = 0",
    "tov_pct": "(t.fga + 0.44 * t.fta + t.tov) = 0",
}


def _rows(conn, where):
    return [dict(r) for r in conn.execute(f"SELECT a.* FROM player_season_advanced a WHERE {where}")]


def _no_attempt_rows(conn, col, cond):
    return [dict(r) for r in conn.execute(
        f"""SELECT a.* FROM player_season_advanced a
            JOIN player_season_totals t ON t.player_id = a.player_id AND t.season = a.season
             AND t.season_type = a.season_type AND t.team_id = a.team_id
            WHERE a.{col} = 0 AND {cond}""")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the repair (default: dry run)")
    ap.add_argument("--db", default=DB)
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    traded = _rows(conn, TRADED_WHERE)
    unfilled = _rows(conn, UNFILLED_WHERE)
    no_att = {c: _no_attempt_rows(conn, c, cond) for c, cond in NO_ATTEMPTS.items()}
    print(f"traded players' stints with season-only values: {len(traded)}")
    print(f"single-team rows never filled (pace 0):         {len(unfilled)}")
    for c, rows in no_att.items():
        print(f"{c} stored as 0.0 with no attempts:          {len(rows)}")
    if not args.apply:
        print("dry run: nothing written. Re-run with --apply.")
        return 0

    touched = {r["id"]: r for r in traded + unfilled + [r for rows in no_att.values() for r in rows]}
    os.makedirs(os.path.join(os.path.dirname(args.db), "backups"), exist_ok=True)
    backup = os.path.join(os.path.dirname(args.db), "backups",
                          f"advanced_placeholders_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(list(touched.values()), fh)
    print(f"backed up {len(touched)} rows to {backup}")

    nulls = ", ".join(f"{c} = NULL" for c in SEASON_ONLY)
    with conn:
        conn.execute(f"UPDATE player_season_advanced SET {nulls} WHERE id IN "
                     f"(SELECT a.id FROM player_season_advanced a WHERE {TRADED_WHERE})")
        conn.execute(f"UPDATE player_season_advanced SET {nulls} WHERE id IN "
                     f"(SELECT a.id FROM player_season_advanced a WHERE {UNFILLED_WHERE})")
        for c, rows in no_att.items():
            conn.executemany(f"UPDATE player_season_advanced SET {c} = NULL WHERE id = ?", [(r["id"],) for r in rows])
    print("applied. Re-run without --apply to confirm zero remaining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
