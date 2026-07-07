"""
grade_predictions.py
====================
Fills in actual results for logged model predictions (predictions_log in
Data/OddsData.sqlite) using final scores from the stats database
(Data/TeamData.sqlite, kept fresh by daily_update.py).

A prediction is graded once the game's box score is in team_game_advanced:
  actual_winner = team with more points, actual_total = combined score.

Run standalone or via daily_update.py. Idempotent — only ungraded rows
with a game date in the past are touched.
"""

import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("grade_predictions")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
TEAM_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")


def _team_name_to_id(team_conn: sqlite3.Connection) -> dict:
    rows = team_conn.execute("SELECT team_id, full_name FROM team_metadata").fetchall()
    mapping = {}
    for team_id, full_name in rows:
        mapping[full_name.strip().lower()] = team_id
    # sbrscrape uses "LA Clippers" for the Clippers
    la = mapping.get("los angeles clippers")
    if la:
        mapping["la clippers"] = la
    return mapping


def _find_result(team_conn, home_id: int, away_id: int, around_date: str):
    """
    Find the home team's game against the away team within +/-1 day of the
    logged date (game dates can straddle midnight UTC vs local).
    Returns (home_pts, away_pts) or None.
    """
    row = team_conn.execute(
        """
        SELECT pts, opp_pts, game_date FROM team_game_advanced
        WHERE team_id = ? AND opp_team_id = ?
          AND game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
        ORDER BY game_date ASC LIMIT 1
        """,
        (home_id, away_id, around_date, around_date),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1]


def grade() -> int:
    if not os.path.exists(ODDS_DB) or not os.path.exists(TEAM_DB):
        logger.warning("Databases not found; nothing to grade.")
        return 0

    odds_conn = sqlite3.connect(ODDS_DB)
    odds_conn.row_factory = sqlite3.Row
    team_conn = sqlite3.connect(TEAM_DB)
    graded = 0
    try:
        try:
            ungraded = odds_conn.execute(
                """
                SELECT id, log_date, home_team, away_team, game_start_time_utc
                FROM predictions_log
                WHERE actual_winner IS NULL AND log_date < ?
                """,
                (datetime.utcnow().strftime("%Y-%m-%d"),),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.info("predictions_log table does not exist yet; nothing to grade.")
            return 0

        if not ungraded:
            logger.info("No ungraded predictions.")
            return 0

        name_to_id = _team_name_to_id(team_conn)
        for row in ungraded:
            home_id = name_to_id.get(row["home_team"].strip().lower())
            away_id = name_to_id.get(row["away_team"].strip().lower())
            if not home_id or not away_id:
                continue  # non-NBA game (e.g. WNBA fallback) — not gradeable from this DB

            game_date = (row["game_start_time_utc"] or row["log_date"])[:10]
            result = _find_result(team_conn, home_id, away_id, game_date)
            if result is None:
                continue  # box score not ingested yet

            home_pts, away_pts = result
            winner = row["home_team"] if home_pts > away_pts else row["away_team"]
            odds_conn.execute(
                "UPDATE predictions_log SET actual_winner = ?, actual_total = ? WHERE id = ?",
                (winner, home_pts + away_pts, row["id"]),
            )
            graded += 1

        odds_conn.commit()
        logger.info("Graded %d of %d ungraded predictions.", graded, len(ungraded))
        return graded
    finally:
        odds_conn.close()
        team_conn.close()


if __name__ == "__main__":
    grade()
    sys.exit(0)
