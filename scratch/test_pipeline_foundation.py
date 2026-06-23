"""
test_pipeline_foundation.py
===========================
Automated verification script for Phase 1.
1. Creates a test database and runs the idempotent schema migration.
2. Fetches and processes a completed game (0022400001: Knicks @ Celtics, 2024-10-22).
3. Compares computed Dean Oliver derivatives against official NBA advanced box scores.
4. Verifies database storage and logs verification results.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Utils.nba_db_schema import ensure_schema, get_connection
from src.Utils.nba_pipeline import process_game

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_DB = "Data/test_nba_pipeline.sqlite"
TEST_GAME_ID = "0022400061"  # Knicks @ Celtics, Oct 22, 2024 (Season opener)

def run_test():
    # Remove test DB if exists to start fresh
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
            logger.info("Removed existing test database %s", TEST_DB)
        except OSError as e:
            logger.warning("Could not remove test database: %s", e)

    # 1. Run Schema Migration
    logger.info("Step 1: Running schema migration...")
    ensure_schema(TEST_DB)
    
    # Verify tables exist
    conn = get_connection(TEST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row["name"] for row in cursor.fetchall()]
    logger.info("Created tables: %s", tables)
    
    required_tables = [
        "box_scores", "team_game_advanced", "team_season_advanced", "raw_scrape_log",
        "players", "player_game_log", "player_season_totals", "player_season_advanced", "player_splits"
    ]
    for t in required_tables:
        assert t in tables, f"Missing required table {t}"
    logger.info("Schema verification PASSED.")

    # 2. Run Single Game ETL
    logger.info("Step 2: Processing game %s...", TEST_GAME_ID)
    res = process_game(
        game_id=TEST_GAME_ID,
        season="2024-25",
        season_type="Regular Season",
        db_path=TEST_DB,
        overwrite=True
    )
    logger.info("Pipeline run output: %s", res)
    assert res["status"] == "processed", "Pipeline status should be 'processed'"

    # 3. Verify Database Storage & Validation Logs
    logger.info("Step 3: Verifying DB records and validation outputs...")
    
    # Check box_scores table
    cursor.execute("SELECT game_id, season, game_date FROM box_scores WHERE game_id = ?", (TEST_GAME_ID,))
    box_row = cursor.fetchone()
    assert box_row is not None, "box_scores row was not saved"
    logger.info("Saved raw box score: game_id=%s, season=%s, date=%s", 
                box_row["game_id"], box_row["season"], box_row["game_date"])

    # Check team_game_advanced table
    cursor.execute(
        """
        SELECT team_id, opp_team_id, poss_estimated, pace, off_rating, def_rating, net_rating, efg_pct
        FROM team_game_advanced WHERE game_id = ?
        """,
        (TEST_GAME_ID,)
    )
    rows = cursor.fetchall()
    assert len(rows) == 2, f"Should have 2 rows in team_game_advanced, got {len(rows)}"
    
    for r in rows:
        logger.info(
            "Computed team %d: poss=%.2f, pace=%.2f, ORtg=%.2f, DRtg=%.2f, NetRtg=%.2f, eFG%%=%.3f",
            r["team_id"], r["poss_estimated"], r["pace"], r["off_rating"],
            r["def_rating"], r["net_rating"], r["efg_pct"]
        )

    # Check raw_scrape_log table for validation status
    cursor.execute("SELECT status, message, metric, diff FROM raw_scrape_log WHERE game_id = ?", (TEST_GAME_ID,))
    log_rows = cursor.fetchall()
    
    warnings = [l for l in log_rows if l["status"] == "warning"]
    successes = [l for l in log_rows if l["status"] == "ok"]
    
    logger.info("Validation logs: %d warnings, %d successes", len(warnings), len(successes))
    
    # Assert validation rate rules
    for w in warnings:
        logger.warning("Validation discrepancy: %s (diff=%s)", w["message"], w["diff"])
        # Ensure discrepancy is indeed minimal or at least explainable
        assert w["diff"] < 2.0, f"Discrepancy too large: {w['message']}"

    assert len(successes) > 0 or len(warnings) < 3, "Too many validation failures!"
    
    # Assert players and player game log exist
    cursor.execute("SELECT COUNT(*) as count FROM players")
    player_count = cursor.fetchone()["count"]
    logger.info("Found %d players in the players table.", player_count)
    assert player_count > 0, "No players were saved in players table"
    
    cursor.execute("SELECT COUNT(*) as count FROM player_game_log")
    log_count = cursor.fetchone()["count"]
    logger.info("Found %d player game logs in the player_game_log table.", log_count)
    assert log_count > 0, "No player game logs were saved"
    
    # Query a sample player game log to verify starter status
    cursor.execute("SELECT player_id, starter, min FROM player_game_log LIMIT 5")
    sample_logs = cursor.fetchall()
    for row in sample_logs:
        logger.info("Player game log sample: player_id=%d, starter=%d, min=%.2f", row["player_id"], row["starter"], row["min"])

    logger.info("Database verification PASSED.")

    # 4. Cleanup
    conn.close()
    logger.info("All tests PASSED successfully!")

if __name__ == "__main__":
    run_test()
