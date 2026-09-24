"""
daily_update.py
===============
Daily data refresh for the BettingBuddy stats backend. Intended to run once a
morning (e.g. 9 AM via Windows Task Scheduler):

1. Works out the current NBA season from today's date.
2. Runs the incremental backfill (already-processed games are cached and skip
   instantly, so during the season this only ingests yesterday's games), plus
   the play-in in April-May and the playoffs in April-June. A game that fails
   to land fails the task; the known permanent holes do not.
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

# Every child process this job starts writes UTF-8. Scheduled tasks run
# without PYTHONIOENCODING, so children printed in the console code page; the
# capture below decoded that as UTF-8, a dash became U+FFFD, and the log file
# (also code-page encoded) could not write it: the logging module dropped the
# line. That is how the 2026-09-23 9am run logged "preflight: 2 WRONG" without
# saying which two checks failed.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_update.log"),
                            encoding="utf-8", errors="backslashreplace"),
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


#: What backfill.py's exit codes mean (see its module docstring).
_BACKFILL_EXIT_MEANING = {
    1: "a stage crashed",
    2: "bad command line",
    3: "one or more games failed to ingest -- the backfill's summary names them",
    4: "the league game log listed no games while games were expected",
}


def _regular_season_games_expected(today: Optional[date] = None) -> bool:
    """Whether the current season's regular-season game log must be non-empty.

    True from two days after opening night through the following September
    (the log of a finished season stays full all summer). False from
    1 October until then, because the season label rolls over on 1 October
    and the new season's log is legitimately empty until games are played --
    a red run every morning for three weeks is how people learn to ignore red.
    """
    today = today or date.today()
    if today.month != 10:
        return True
    try:
        from preflight_opening_night import OPENING_NIGHT
        opening = OPENING_NIGHT if OPENING_NIGHT.year == today.year else None
    except Exception:
        opening = None
    if opening is None:
        # A stale constant from a previous year must not turn every October
        # morning red; the league opens in the third or fourth week.
        return today.day >= 26
    return (today - opening).days >= 2


def run_backfill(season: str, season_type: str = "Regular Season",
                 expect_games: bool = False) -> bool:
    """Run backfill.py for one season type. False on any non-zero exit.

    backfill.py used to exit 0 however many games failed, so this reported
    "completed successfully" on mornings when games were missing. It now exits
    non-zero when a game genuinely failed (known permanent holes do not count),
    and main() puts "backfill" in the failures list.
    """
    python = sys.executable
    script = os.path.join(REPO_ROOT, "src", "Process-Data", "backfill.py")
    cmd = [python, script, "--season", season, "--season-type", season_type]
    if expect_games:
        cmd.append("--expect-games")
    logger.info("Running backfill: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        logger.error("Backfill %s %s exited with code %s (%s)", season, season_type,
                     result.returncode,
                     _BACKFILL_EXIT_MEANING.get(result.returncode, "unexpected exit code"))
        return False
    logger.info("Backfill %s %s completed successfully.", season, season_type)
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

    # The runner now says what kind of result this is (main_api PRED_STATUS_*).
    # "failed" is a pipeline that should have worked; "market_only" means no
    # team had a stats row, so every row is a market-implied placeholder the
    # ledger refuses. Either way nothing real can be logged, and before this
    # check a failed run in the offseason reported "offseason" and a
    # market-only run reported "logged" with 0 rows written.
    status = result.get("status")
    if status in ("failed", "market_only"):
        logger.error("Prediction run returned status=%s: %s", status,
                     result.get("error") or result.get("message") or "no detail")
        return "failed"

    predictions = result.get("predictions") or []
    if resolved != "NBA":
        # In season, this means the NBA scrape failed on every day it tried
        # and the provider fell back to another league. Reporting that as
        # "offseason" made opening night with a broken scraper a GREEN run
        # with no picks logged, and those picks can never be logged later.
        if _nba_games_expected():
            logger.error(
                "Odds provider resolved to %s during the NBA season. No NBA pick "
                "was logged, and none can be after tip-off.", resolved)
            return "failed"
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
    preseason = counts.get("preseason", 0)
    logger.info(
        "predictions_log: %d written, %d already logged earlier today, %d refused "
        "(no tip-off time), %d refused (already under way), %d skipped (preseason) "
        "-- of %d produced.",
        written, already, no_tip, late, preseason, len(predictions))

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
    """Whether NBA games are expected today (opening night through June).

    Used to decide whether an empty or non-NBA prediction run is a failure
    worth failing the task over.

    This used to be `month >= 10`, which is true from 1 October -- three
    weeks before the season starts. Every morning from the 1st to the 19th
    would have gone red for having no NBA slate, which is how people learn
    to ignore red, and on the 20th the one red that mattered would have
    looked like the rest. October now waits for opening night, read from
    the same constant the preflight uses so there is only one date to bump.
    """
    today = today or date.today()
    if today.month in (7, 8, 9):
        return False
    if today.month == 10:
        try:
            from preflight_opening_night import OPENING_NIGHT
            return today >= OPENING_NIGHT
        except Exception:
            return today.day >= 20   # the league opens in the third week
    return True


def refresh_play_by_play(season: str) -> bool:
    """Fetch play-by-play for this season's new games, then rebuild the runs.

    Added 2026-09-24. Play-by-play was in no scheduled job: every season was
    backfilled by hand (backfill_pbp.py --season). The box-score backfill
    does not ingest it, and on/off, clutch, comebacks, win probability, the
    run detector and injury-impact pricing all read pbp_events, so from
    opening night those pages would have stopped at the last game someone
    remembered to fetch. backfill_pbp skips games it already holds, so this
    costs one request per new game. It runs after the box-score backfill in
    the same process: one stats.nba.com job at a time.

    backfill_pbp exits 0 even when games fail, so its own summary line is
    read. Outside the season there is nothing to fetch, which is not an error.
    """
    if not _nba_games_expected():
        logger.info("Play-by-play: no NBA games expected; skipped.")
        return True
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "backfill_pbp.py"), "--season", season],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600,
            encoding="utf-8", errors="replace")
        log = (r.stdout or "") + (r.stderr or "")
        fetch = re.search(r"(\d+) to fetch", log)
        fails = re.search(r"(\d+) failure\(s\)", log)
        n_fetch = int(fetch.group(1)) if fetch else 0
        n_fail = int(fails.group(1)) if fails else 0
        if r.returncode != 0:
            logger.error("Play-by-play backfill exited %s: %s", r.returncode, log.strip()[-500:])
            return False
        logger.info("Play-by-play %s: %d new game(s), %d failure(s).", season, n_fetch, n_fail)
        if n_fetch > n_fail:
            rr = subprocess.run(
                [sys.executable, os.path.join(REPO_ROOT, "ingest_scoring_runs.py")],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
                encoding="utf-8", errors="replace")
            if rr.returncode != 0:
                logger.error("Scoring-runs rebuild failed: %s", ((rr.stdout or "") + (rr.stderr or "")).strip()[-500:])
                return False
            logger.info("Scoring runs rebuilt.")
        return n_fail == 0
    except Exception as exc:
        logger.error("Play-by-play refresh could not run: %s", exc, exc_info=True)
        return False


def grade_logged_predictions() -> bool:
    """Fill in final scores for yesterday's logged predictions, then price them.

    Grading says whether the pick was right. CLV says whether the price was
    good, which is a different question and answerable much sooner -- return
    on investment needs hundreds of settled bets, closing line value says
    something after a few dozen. Both are post-hoc enrichment of a row that was
    frozen before tip-off, so they run together.
    """
    # Both steps used to catch every exception, log it as a WARNING marked
    # "non-fatal", and return None -- which main() never looked at. So NBA
    # grading could fail every single morning with the task reporting success.
    # They still do not stop the rest of the job (a broken grader must not
    # prevent tonight's predictions being logged), but a failure now reaches
    # the failures list and turns the run red.
    ok = True
    try:
        from grade_predictions import grade
        grade()
    except Exception as exc:
        logger.error("Prediction grading FAILED: %s", exc, exc_info=True)
        ok = False
    try:
        from grade_predictions import price_clv
        price_clv()
    except Exception as exc:
        logger.error("Closing line value FAILED: %s", exc, exc_info=True)
        ok = False
    return ok


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


def run_integrity_audit(season: str) -> int:
    """Check this season's archive against itself (audit_archive.py).

    Added 2026-09-24. The bugs found the day before were each invisible table
    by table and plain across tables: a team's points not equal to its
    players', season totals not equal to the game log, placeholder zeros.
    The daily backfill is what writes this season, so this is where a new one
    would appear. Non-fatal for the same reason as the preflight: it reports,
    the log says it loudly, and a night's grading is not held hostage to it.

    Returns the number of failed checks, or -1 if the audit could not run.
    """
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "audit_archive.py"), "--season", season],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
            encoding="utf-8", errors="replace")
        lines = [ln.rstrip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        if r.returncode == 0:
            logger.info("Integrity audit (%s): all checks pass", season)
            return 0
        failed = [ln.strip() for ln in lines if ln.strip().startswith(("FAIL", "ERROR"))]
        logger.error("Integrity audit (%s) found problems:", season)
        for ln in lines:
            if ln.strip().startswith(("FAIL", "ERROR")) or ln.startswith("           {"):
                logger.error("  %s", ln.strip())
        return len(failed) or 1
    except Exception as exc:
        logger.error("Integrity audit could not run: %s", exc, exc_info=True)
        return -1


def publish_ledger() -> str:
    """Mirror the ledger to the public server. 'published' | 'skipped' | 'failed'.

    The home PC is the only machine that writes the record (DEPLOY.md 3a);
    this is how the public page gets it. 'skipped' means no server is
    configured yet, which is not an error. A refusal is: it means the public
    copy and this one disagree about the past, and push_ledger.py prints why.
    """
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "push_ledger.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        last = lines[-1] if lines else (r.stderr or "").strip()[-300:]
        if r.returncode == 0 and last.startswith("SKIPPED"):
            logger.info("Ledger publish: %s", last)
            return "skipped"
        if r.returncode == 0:
            logger.info("Ledger publish: %s", last)
            return "published"
        for ln in lines:
            logger.error("Ledger publish: %s", ln)
        if not lines:
            logger.error("Ledger publish failed: %s", last)
        return "failed"
    except Exception as exc:
        logger.error("Ledger publish could not run: %s", exc, exc_info=True)
        return "failed"


def main() -> int:
    season = current_season(date.today())
    logger.info("=== Daily update starting for season %s ===", season)

    backfill_ok = run_backfill(season, expect_games=_regular_season_games_expected())
    # Play-in box scores. The play-in (2020-21 on, game ids 005...) is its own
    # season type on stats.nba.com, so neither the regular-season nor the
    # playoff run ever fetched it: the odds feed carries those games, the
    # ledger logs picks on them, and grade() would never find a box score.
    # April AND May, because the calendar moves: the 2020-21 play-in was
    # played 18-21 May 2021. Empty outside the tournament, which is not a
    # failure (no --expect-games).
    if date.today().month in (4, 5):
        backfill_ok = run_backfill(season, "PlayIn") and backfill_ok
    # Playoff box scores. backfill.py does one season type per run and this
    # only ever asked for the regular season, so from the play-in onward no
    # box score would land, grade() would find nothing, and every playoff
    # pick would sit ungraded for good -- while log_predictions kept writing
    # them, because the odds feed carries those games. The same shape as the
    # NFL results gap fixed on 2026-09-21, one calendar season later.
    # April-June only, so the rest of the year costs nothing.
    if date.today().month in (4, 5, 6):
        backfill_ok = run_backfill(season, "Playoffs") and backfill_ok
    pbp_ok = refresh_play_by_play(season)
    stats_ok = refresh_team_stats_snapshot()
    grading_ok = grade_logged_predictions()
    prediction_status = log_todays_predictions()
    odds_status = snapshot_odds_board()
    ingests_ok = refresh_periodic_ingests()
    # After everything that writes OddsData (grading, logging, the board
    # snapshot), so the public copy gets this run's picks and grades.
    ledger_status = publish_ledger()
    # Last, so they see the state this run leaves behind rather than the
    # state it started from.
    audit_failed = run_integrity_audit(season)
    preflight_wrong = run_preflight()

    failures = []
    if not grading_ok:
        failures.append("grading / CLV")
    if not backfill_ok:
        failures.append("backfill")
    if not stats_ok:
        failures.append("team-stats refresh")
    if not pbp_ok:
        failures.append("play-by-play")
    if prediction_status == "failed":
        failures.append("prediction logging")
    # A failed snapshot only fails the task in season: July has no board, and
    # a missing key is a loud warning rather than a red run.
    if odds_status == "failed" and _nba_games_expected():
        failures.append("odds board snapshot")
    if not ingests_ok:
        failures.append("periodic ingests")
    if ledger_status == "failed":
        failures.append("ledger publish")

    if failures:
        logger.error("=== Daily update finished WITH ERRORS: %s ===", ", ".join(failures))
        return 1
    logger.info("=== Daily update finished OK (predictions: %s, preflight: %s, audit: %s) ===",
                prediction_status,
                "clean" if preflight_wrong == 0
                else "COULD NOT RUN" if preflight_wrong < 0
                else f"{preflight_wrong} WRONG",
                "clean" if audit_failed == 0
                else "COULD NOT RUN" if audit_failed < 0
                else f"{audit_failed} FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
