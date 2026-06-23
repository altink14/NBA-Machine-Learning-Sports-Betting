import logging
import sqlite3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Utils.nba_db_schema import ensure_schema, get_connection
from src.Utils.nba_pipeline import process_game

# We need to import compute_and_save_player_season_aggregates. Since 'Process-Data' contains a dash, we can use importlib or sys.path
import importlib
backfill_mod = importlib.import_module("src.Process-Data.backfill")
compute_and_save_player_season_aggregates = backfill_mod.compute_and_save_player_season_aggregates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DB = "Data/test_backfill_aggregates.sqlite"

def main():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
        
    ensure_schema(TEST_DB)
    
    # Process game NYK @ BOS
    res = process_game("0022400061", season="2024-25", season_type="Regular Season", db_path=TEST_DB, overwrite=True)
    logger.info("Process game response: %s", res)
    
    # Run the backfill aggregates computation
    compute_and_save_player_season_aggregates("2024-25", "Regular Season", TEST_DB)
    
    # Verify the tables
    conn = get_connection(TEST_DB)
    cursor = conn.cursor()
    
    # Totals
    cursor.execute("SELECT COUNT(*) as count FROM player_season_totals")
    totals_cnt = cursor.fetchone()["count"]
    logger.info("player_season_totals count: %d", totals_cnt)
    assert totals_cnt > 0, "No records in player_season_totals"
    
    # Query sample totals
    cursor.execute("SELECT player_id, gp, min, pts FROM player_season_totals LIMIT 3")
    for r in cursor.fetchall():
        logger.info("Total sample: player_id=%d, gp=%d, min=%.2f, pts=%d", r["player_id"], r["gp"], r["min"], r["pts"])
        
    # Advanced
    cursor.execute("SELECT COUNT(*) as count FROM player_season_advanced")
    adv_cnt = cursor.fetchone()["count"]
    logger.info("player_season_advanced count: %d", adv_cnt)
    assert adv_cnt > 0, "No records in player_season_advanced"
    
    cursor.execute("SELECT player_id, ts_pct, efg_pct, tov_pct FROM player_season_advanced LIMIT 3")
    for r in cursor.fetchall():
        logger.info("Advanced sample: player_id=%d, ts=%.4f, efg=%.4f, tov=%.4f", r["player_id"], r["ts_pct"], r["efg_pct"], r["tov_pct"])
        
    # Splits
    cursor.execute("SELECT COUNT(*) as count FROM player_splits")
    splits_cnt = cursor.fetchone()["count"]
    logger.info("player_splits count: %d", splits_cnt)
    assert splits_cnt > 0, "No records in player_splits"
    
    cursor.execute("SELECT player_id, split_type, split_value, gp, pts FROM player_splits LIMIT 5")
    for r in cursor.fetchall():
        logger.info("Splits sample: player_id=%d, type=%s, val=%s, gp=%d, pts=%d", r["player_id"], r["split_type"], r["split_value"], r["gp"], r["pts"])
        
    conn.close()
    logger.info("ALL BACKFILL AGGREGATE TESTS PASSED!")

if __name__ == "__main__":
    main()
