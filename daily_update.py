"""
daily_update.py
===============
Daily data refresh for the BettingBuddy stats backend. Intended to run once a
morning (e.g. 9 AM via Windows Task Scheduler):

1. Works out the current NBA season from today's date.
2. Runs the incremental backfill (already-processed games are cached and skip
   instantly, so during the season this only ingests yesterday's games).
3. Recomputes season aggregates, SRS, and player stats (inside backfill).
4. Refreshes the team-stats snapshot the prediction model reads.
5. Grades yesterday's logged predictions against final scores.
6. Runs today's predictions and logs them PRE-GAME to predictions_log, which is
   the evidence behind the public track record.
7. Re-runs any periodic ingest that has come due (refresh_registry.py) - the
   datasets with no live feed, like the Hall of Fame register.

Step 6 used to be an HTTP ping at localhost:8000, on the assumption that the API
was running. It is a dev server that nobody starts, so the ping was refused every
morning from 2026-07-28 on, the warning was swallowed, and the run still reported
OK - predictions_log sat empty for weeks with nothing surfacing it. Predictions
now run in this process, and a failure during the season fails the task.

Register with:
  schtasks /create /tn "BettingBuddy Daily Data Update" ^
    /tr "<venv python> <this file>" /sc daily /st 09:00
"""

import logging
import os
import subprocess
import sys
from datetime import date
from typing import Optional

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
# Task Scheduler may launch this from any working directory; main_api and
# refresh_team_stats are imported below and both live at the repo root.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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


def refresh_team_stats_snapshot() -> bool:
    """Write today's season-to-date team stats, which the model predicts from."""
    try:
        from refresh_team_stats import refresh

        written = refresh()
        if written:
            logger.info("Team-stats snapshot refreshed: %s", written)
        else:
            logger.info("No team-stats snapshot written (no completed games yet this season).")
        return True
    except Exception as exc:
        logger.error("Team-stats refresh failed: %s", exc, exc_info=True)
        return False


def log_todays_predictions() -> str:
    """Run today's predictions in-process and log them before tip-off.

    Returns one of:
      "logged"    - predictions were recorded
      "offseason" - the odds provider fell back off NBA, or there are no games
      "failed"    - the prediction path errored, or produced nothing for NBA games

    Odds snapshots for line-movement tracking happen inside run_predictions(), so
    they keep working through this path too.
    """
    try:
        from main_api import PredictionRunner, log_predictions

        runner = PredictionRunner(sportsbook="fanduel", kelly_criterion=True, sport="NBA")
        resolved = getattr(runner, "resolved_sport", "NBA") or "NBA"
        result = runner.run_predictions()
    except Exception as exc:
        logger.error("Prediction run failed: %s", exc, exc_info=True)
        return "failed"

    predictions = result.get("predictions") or []
    if resolved != "NBA":
        logger.info(
            "Odds provider resolved to %s (NBA offseason). Nothing logged - the track "
            "record covers NBA model predictions only.", resolved
        )
        return "offseason"
    if not predictions:
        logger.warning(
            "No NBA predictions produced%s. Nothing logged.",
            f": {result['error']}" if result.get("error") else ""
        )
        return "failed" if _nba_games_expected() else "offseason"

    try:
        log_predictions(result, "fanduel", resolved)
    except Exception as exc:
        logger.error("Writing predictions_log failed: %s", exc, exc_info=True)
        return "failed"

    logger.info("Logged %d prediction(s) to predictions_log.", len(predictions))
    return "logged"


def refresh_periodic_ingests() -> bool:
    """Re-run any dataset that is not read live from a feed and has come due.

    Registered in refresh_registry.py. Nothing here needs to be remembered or
    run by hand - a job that has not run for its interval runs itself, and a job
    that failed retries the next morning instead of waiting out its interval.
    """
    try:
        from refresh_registry import run_due
        return run_due()
    except Exception as exc:
        logger.error("Periodic refresh failed: %s", exc, exc_info=True)
        return False


def _nba_games_expected(today: Optional[date] = None) -> bool:
    """Whether the NBA is in season today (October-June).

    Used only to decide whether an empty prediction run is a failure worth
    failing the task over, or just July.
    """
    today = today or date.today()
    return today.month >= 10 or today.month <= 6


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

    backfill_ok = run_backfill(season)
    stats_ok = refresh_team_stats_snapshot()
    grade_logged_predictions()
    prediction_status = log_todays_predictions()
    ingests_ok = refresh_periodic_ingests()

    failures = []
    if not backfill_ok:
        failures.append("backfill")
    if not stats_ok:
        failures.append("team-stats refresh")
    if prediction_status == "failed":
        failures.append("prediction logging")
    if not ingests_ok:
        failures.append("periodic ingests")

    if failures:
        logger.error("=== Daily update finished WITH ERRORS: %s ===", ", ".join(failures))
        return 1
    logger.info("=== Daily update finished OK (predictions: %s) ===", prediction_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
