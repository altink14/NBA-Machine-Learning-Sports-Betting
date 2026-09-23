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
import re
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

# Load the repo's .env explicitly, by path. Task Scheduler and a bare terminal
# both start this with an environment that has no ODDS_API_KEY in it; without
# this line the odds step only worked when main_api happened to be imported
# first (it calls load_dotenv), which is an accident, not a design.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO_ROOT, ".env"))


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
        counts = log_predictions(result, "fanduel", resolved) or {}
    except Exception as exc:
        logger.error("Writing predictions_log failed: %s", exc, exc_info=True)
        return "failed"

    # Report what was WRITTEN, not what the model produced. This line used to
    # print len(predictions) -- so on 2026-09-22, with the odds provider
    # supplying no tip-off time and the ledger correctly refusing every row,
    # it would still have said "Logged 12 prediction(s)" on opening night
    # while the public track record stayed empty.
    written = counts.get("written", 0)
    already = counts.get("already_present", 0)
    no_tip = counts.get("no_tipoff", 0)
    late = counts.get("late", 0)
    logger.info(
        "predictions_log: %d written, %d already logged earlier today, %d refused "
        "(no tip-off time), %d refused (already under way) -- of %d produced.",
        written, already, no_tip, late, len(predictions))

    # A missing tip-off is always a defect in the feed, never a timing accident,
    # and those picks can never be logged later: the table's CHECK constraint
    # forbids a row written after tip-off. So any of them fails the run.
    if no_tip:
        logger.error(
            "%d pick(s) had no tip-off time and were NOT logged. They are gone for "
            "good. Check the odds provider's start-time field.", no_tip)
        return "failed"
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
    """Fill in final scores for yesterday's logged predictions, then price them.

    Grading says whether the pick was right. CLV says whether the price was
    good, which is a different question and answerable much sooner -- return
    on investment needs hundreds of settled bets, closing line value says
    something after a few dozen. Both are post-hoc enrichment of a row that was
    frozen before tip-off, so they run together.
    """
    try:
        from grade_predictions import grade
        grade()
    except Exception as exc:
        logger.warning("Prediction grading failed (non-fatal): %s", exc)
    try:
        from grade_predictions import price_clv
        price_clv()
    except Exception as exc:
        logger.warning("Closing line value failed (non-fatal): %s", exc)


def snapshot_odds_board() -> str:
    """
    Archive today's odds board from The Odds API (moneylines, spreads, totals,
    every US book) into odds_snapshots. One run costs ~3 API credits, so the
    daily cadence fits the free tier; the in-season high-frequency cadence is
    snapshot_odds_api.py --loop (see that file's quota math).

    Returns 'ok', 'skipped' (no key configured), or 'failed'. A missing key is
    logged loudly every day rather than silently: an empty archive that LOOKS
    fine is how the predictions_log bug lived for weeks.
    """
    if not os.environ.get("ODDS_API_KEY", "").strip():
        logger.warning(
            "ODDS_API_KEY is not set - odds board NOT archived. CLV/line-shop "
            "features starve without this. Create a key at the-odds-api.com."
        )
        return "skipped"
    try:
        from src.Utils.odds_api_client import snapshot_nba_board
        db_path = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
        summary = snapshot_nba_board(db_path)
        logger.info(
            "Odds board archived: %s events, %s rows written, quota remaining %s",
            summary["events"], summary["written"], summary["quota_remaining"],
        )
        return "ok"
    except Exception as exc:
        logger.error("Odds board snapshot failed: %s", exc, exc_info=True)
        return "failed"


def run_preflight() -> int:
    """Check the configuration opening night depends on, and say what is wrong.

    WHY THIS RUNS EVERY MORNING. The sealed candidate model stopped loading on
    22 August 2026 and nobody noticed for a month, because the only symptom was
    a fallback to the old model that logged once per process. A check nobody
    runs is a check that does not exist, and the one thing guaranteed to run
    daily is this job.

    NON-FATAL BY DESIGN. It reports; it does not decide. Most of what it looks
    at is upstream of grading and ingest, so failing the whole task on a
    preflight finding would hide a working night's work behind a warning. The
    count goes in the log either way, and a non-zero "wrong" is loud.

    Returns the number of wrong checks, or -1 if the preflight itself failed.
    """
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "preflight_opening_night.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
            encoding="utf-8", errors="replace")
        tail = [ln for ln in (r.stdout or "").strip().split("\n") if ln.strip()]
        # The counts line ("18 pass, 4 not yet, 0 wrong"), not the prose line
        # after it -- both contain the word "wrong".
        summary = next((ln.strip() for ln in reversed(tail)
                        if re.match(r"^\d+ pass,", ln.strip())), "")
        if r.returncode == 0:
            logger.info("Preflight: %s", summary or "clean")
            return 0
        logger.error("Preflight found problems — %s", summary or "see below")
        for line in tail:
            if line.strip().startswith("WRONG"):
                logger.error("  %s", line.strip())
        return sum(1 for ln in tail if ln.strip().startswith("WRONG")) or 1
    except Exception as exc:
        logger.error("Preflight could not run: %s", exc, exc_info=True)
        return -1


def main() -> int:
    season = current_season(date.today())
    logger.info("=== Daily update starting for season %s ===", season)

    backfill_ok = run_backfill(season)
    stats_ok = refresh_team_stats_snapshot()
    grade_logged_predictions()
    prediction_status = log_todays_predictions()
    odds_status = snapshot_odds_board()
    ingests_ok = refresh_periodic_ingests()
    # Last, so it sees the state this run leaves behind rather than the state
    # it started from.
    preflight_wrong = run_preflight()

    failures = []
    if not backfill_ok:
        failures.append("backfill")
    if not stats_ok:
        failures.append("team-stats refresh")
    if prediction_status == "failed":
        failures.append("prediction logging")
    # A failed snapshot only fails the task in season: July has no board, and
    # a missing key is a loud warning rather than a red run.
    if odds_status == "failed" and _nba_games_expected():
        failures.append("odds board snapshot")
    if not ingests_ok:
        failures.append("periodic ingests")

    if failures:
        logger.error("=== Daily update finished WITH ERRORS: %s ===", ", ".join(failures))
        return 1
    logger.info("=== Daily update finished OK (predictions: %s, preflight: %s) ===",
                prediction_status,
                "clean" if preflight_wrong == 0
                else "COULD NOT RUN" if preflight_wrong < 0
                else f"{preflight_wrong} WRONG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
