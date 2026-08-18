"""
ingest_scoring_runs.py
======================
Detects every unanswered scoring run in the archive and stores them, so the run
detector queries a small table instead of walking 679,000 scoring events per
request.

Definition, kept deliberately narrow: a run is consecutive points by one team with
the opponent scoring NOTHING in between. That is what "an 8-0 run" means and it
needs no parameters. The looser kind - "outscored 14-2 over five minutes" - needs
an arbitrary window and an arbitrary allowance, and two people would pick
different ones and get different leaderboards.

Runs are detected from score_home/score_away deltas rather than by interpreting
action types, because the score is the ground truth and free throws, and-ones and
corrected scores all land in it without special cases.

A run may cross a period boundary, which is real - a team closing the third and
opening the fourth without reply is one run, not two - so start and end period are
both recorded rather than truncating at the break.

    venv/Scripts/python.exe ingest_scoring_runs.py [--min-points 6]
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
logger = logging.getLogger("scoring_runs")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    game_date TEXT,
    season TEXT,
    season_type TEXT,
    team_id INTEGER,
    team_tricode TEXT,
    opp_team_id INTEGER,
    opp_tricode TEXT,
    points INTEGER NOT NULL,
    scoring_plays INTEGER,
    start_period INTEGER,
    end_period INTEGER,
    start_elapsed REAL,
    end_elapsed REAL,
    score_before_team INTEGER,
    score_before_opp INTEGER,
    fetched_at TEXT
)
"""
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_points ON scoring_runs(points DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_team ON scoring_runs(team_tricode)",
    "CREATE INDEX IF NOT EXISTS idx_runs_opp ON scoring_runs(opp_tricode)",
    "CREATE INDEX IF NOT EXISTS idx_runs_game ON scoring_runs(game_id)",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-points", type=int, default=6,
                    help="Smallest run to store. The endpoint filters upward from here.")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA)
        for ix in INDEXES:
            conn.execute(ix)

        # Which team is home, so a score delta can be attributed to a tricode.
        meta = {
            r["game_id"]: r for r in conn.execute(
                """
                SELECT b.game_id, b.game_date, b.season, b.season_type,
                       b.home_team_id, b.away_team_id,
                       hm.abbreviation AS home_tri, am.abbreviation AS away_tri
                FROM box_scores b
                LEFT JOIN team_metadata hm ON hm.team_id = b.home_team_id
                LEFT JOIN team_metadata am ON am.team_id = b.away_team_id
                """
            )
        }
        logger.info("Games with metadata: %d", len(meta))

        rows = conn.execute(
            """
            SELECT game_id, period, elapsed_seconds, action_number,
                   score_home, score_away
            FROM pbp_events
            WHERE score_home IS NOT NULL AND score_away IS NOT NULL
            ORDER BY game_id, period, elapsed_seconds, action_number
            """
        ).fetchall()
        logger.info("Scoring-capable events: %d", len(rows))

        now = datetime.now(timezone.utc).isoformat()
        out = []
        cur_game = None
        prev_h = prev_a = 0
        run_side = None          # 'home' | 'away'
        run_pts = 0
        run_plays = 0
        run_start_period = None
        run_start_elapsed = None
        run_before_h = run_before_a = 0

        def close_run():
            """Store the finished run if it clears the floor."""
            if run_side is None or run_pts < args.min_points:
                return
            m = meta.get(cur_game)
            if not m:
                return
            if run_side == "home":
                tid, tri = m["home_team_id"], m["home_tri"]
                oid, otri = m["away_team_id"], m["away_tri"]
                before_team, before_opp = run_before_h, run_before_a
            else:
                tid, tri = m["away_team_id"], m["away_tri"]
                oid, otri = m["home_team_id"], m["home_tri"]
                before_team, before_opp = run_before_a, run_before_h
            out.append((
                cur_game, m["game_date"], m["season"], m["season_type"],
                tid, tri, oid, otri, run_pts, run_plays,
                run_start_period, last_period, run_start_elapsed, last_elapsed,
                before_team, before_opp, now,
            ))

        last_period = last_elapsed = None
        for r in rows:
            g = r["game_id"]
            if g != cur_game:
                close_run()
                cur_game = g
                prev_h = prev_a = 0
                run_side, run_pts, run_plays = None, 0, 0
                run_start_period = run_start_elapsed = None

            h, a = r["score_home"] or 0, r["score_away"] or 0
            dh, da = h - prev_h, a - prev_a
            prev_h, prev_a = h, a
            if dh <= 0 and da <= 0:
                continue

            # A single event that moves both scores should not happen; if it does,
            # it is a correction rather than a play, so the run is simply ended.
            if dh > 0 and da > 0:
                close_run()
                run_side, run_pts, run_plays = None, 0, 0
                continue

            side = "home" if dh > 0 else "away"
            pts = dh if dh > 0 else da
            if side != run_side:
                close_run()
                run_side = side
                run_pts = 0
                run_plays = 0
                run_start_period = r["period"]
                run_start_elapsed = r["elapsed_seconds"]
                run_before_h = h - dh
                run_before_a = a - da
            run_pts += pts
            run_plays += 1
            last_period = r["period"]
            last_elapsed = r["elapsed_seconds"]
        close_run()

        conn.execute("DELETE FROM scoring_runs")
        conn.executemany(
            """
            INSERT INTO scoring_runs (
                game_id, game_date, season, season_type, team_id, team_tricode,
                opp_team_id, opp_tricode, points, scoring_plays,
                start_period, end_period, start_elapsed, end_elapsed,
                score_before_team, score_before_opp, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            out,
        )
        conn.commit()
        biggest = conn.execute(
            "SELECT points, team_tricode, opp_tricode, game_date FROM scoring_runs "
            "ORDER BY points DESC LIMIT 1"
        ).fetchone()
        logger.info(
            "scoring_runs: %d runs of %d+ points stored. Biggest: %d-0 by %s over %s on %s.",
            len(out), args.min_points, biggest["points"], biggest["team_tricode"],
            biggest["opp_tricode"], (biggest["game_date"] or "")[:10],
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
