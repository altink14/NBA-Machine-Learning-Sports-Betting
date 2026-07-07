"""
daily_update.py
===============
Daily data refresh for the BettingBuddy stats backend. Intended to run once a
morning (e.g. 9 AM via Windows Task Scheduler):

1. Works out the current NBA season from today's date.
2. Runs the incremental backfill (already-processed games are cached and skip
   instantly, so during the season this only ingests yesterday's games).
3. Recomputes season aggregates, SRS, and player stats (inside backfill).
4. Best-effort: pings the local API's /predictions to snapshot today's odds
   for line-movement tracking.

Register with:
  schtasks /create /tn "BettingBuddy Daily Data Update" ^
    /tr "<venv python> <this file>" /sc daily /st 09:00
"""

import logging
import os
import subprocess
import sys
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_update.log")),
    ],
)
logger = logging.getLogger("daily_update")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def current_season(today: date) -> str:
    """
    NBA seasons run October-June and are labeled by their span, e.g. 2025-26.
    July-September (offseason) maps to the season that just ended.
    """
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def run_backfill(season: str) -> bool:
    python = sys.executable
    script = os.path.join(REPO_ROOT, "src", "Process-Data", "backfill.py")
    cmd = [python, script, "--season", season, "--season-type", "Regular Season"]
    logger.info("Running backfill: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        logger.error("Backfill exited with code %s", result.returncode)
        return False
    logger.info("Backfill completed successfully.")
    return True


def snapshot_todays_odds() -> None:
    """Ping the local API so today's lines get snapshotted (line-movement tracking)."""
    try:
        import requests

        resp = requests.get("http://localhost:8000/predictions?sportsbook=fanduel", timeout=120)
        logger.info("Odds snapshot ping: HTTP %s", resp.status_code)
    except Exception as exc:
        logger.warning("Odds snapshot ping skipped (API not running?): %s", exc)


def grade_logged_predictions() -> None:
    """Fill in final scores for yesterday's logged predictions."""
    try:
        from grade_predictions import grade
        grade()
    except Exception as exc:
        logger.warning("Prediction grading failed (non-fatal): %s", exc)


def main() -> int:
    season = current_season(date.today())
    logger.info("=== Daily update starting for season %s ===", season)
    ok = run_backfill(season)
    grade_logged_predictions()
    snapshot_todays_odds()
    logger.info("=== Daily update finished (%s) ===", "OK" if ok else "WITH ERRORS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
