"""
run_scheduled.py (cross-sport)
==============================
One entry point for everything that must run on a clock, so the Task Scheduler
definitions stay short and the cadences live here in readable form.

THE THREE JOBS AND WHY THEIR CADENCES DIFFER

  frequent   every 15 minutes. The odds recorder, which reads our own schedule
             and calls The Odds API only as a kickoff approaches. Almost every
             run does nothing and costs nothing; the runs that matter are the
             ones inside thirty minutes of kickoff, and those are the closing
             lines. Running it rarely would defeat the point.

  hourly     the injury recorder and the predictor. Injuries move on a scale of
             hours, and pulling ESPN's 9 MB feed every fifteen minutes would be
             rude to them and pointless to us. The predictor is idempotent (a
             UNIQUE key per game and model) so an hourly run simply catches any
             fixture that has come into the horizon.

  daily      grading, repair, sealing, then closing line value. Yesterday's
             games are final by the morning, which is also when it is clear
             whether a capture was missed and worth buying back. CLV runs last
             because it reads the closing lines the seal has just written.

WHAT A MISSED RUN ACTUALLY COSTS, WHICH DIFFERS BY JOB.

  hourly and daily are UNRECOVERABLE. An injury status we did not observe is
  gone: the sources overwrite the current state and keep no history, so there
  is nowhere to buy it back from. A prediction we did not write before kickoff
  can never be written at all -- the ledger's CHECK constraint refuses it, and
  that refusal is the product's main argument, not an inconvenience.

  frequent is REPAIRABLE, at a price. The Odds API serves historical snapshots
  back to 2020-06-06 (5-minute resolution from September 2022) at 10 credits
  per market per region, so a slate we slept through can be reconstructed
  rather than mourned. A missed NFL Sunday is roughly 900 credits to repair.
  See docs/sources/odds-providers.md in the frontend repo.

  But repaired is not the same as observed. A backfilled price is one we read
  out of a vendor's archive afterwards, not one we recorded while it was live,
  and anything built on it -- closing line value above all -- must say so. Keep
  the recorder running; the repair path is for accidents, not a substitute.

THE DAILY JOB SPENDS MONEY, CAREFULLY. `repair_odds.py --unattended` runs here
every night. Unattended spending earns its guardrails: a three-day window, a
90-credit cap (three kickoff times), a backlog worked down a slice a night
rather than bought in one gulp, and a hard refusal to spend if it would leave
the live recorder short -- a price we can still watch beats one we would have
to buy back. It writes nothing on the current free tier, where the historical
endpoint returns 401, and reports that as a warning rather than a failure so
this job does not go red every night for a reason nobody can act on tonight.

Nothing here raises on failure of a single job: one dead source must not stop
the other two. Failures are logged loudly and the exit code reflects them, so
Task Scheduler's "last result" column is meaningful.

Usage:
    python src/Sports/run_scheduled.py frequent
    python src/Sports/run_scheduled.py hourly
    python src/Sports/run_scheduled.py daily
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(REPO_ROOT, "venv", "Scripts", "python.exe")
LOG_DIR = os.path.join(REPO_ROOT, "logs")


def _nfl_season(today=None) -> int:
    """The season nflverse would call today's games.

    A season is named for the year it kicked off in, and it runs into
    February, so January and February belong to the year before them.
    """
    d = today or datetime.now(timezone.utc).date()
    return d.year - 1 if d.month <= 2 else d.year


JOBS = {
    "frequent": [
        ("odds recorder (NFL)", ["src/Sports/odds_recorder.py", "--sport", "nfl"]),
        # NBA added 2026-09-20, a month before opening night, so the machinery
        # is proven boring before it matters. Nothing scheduled ran the NBA
        # recorder at all: daily_update took one board snapshot a morning,
        # which is an archive entry, not a closing line. On opening night that
        # would have meant no closing price for any game.
        #
        # Safe to add this early because the recorder checks the schedule
        # first and that check is free. Through the offseason every run costs
        # 0 credits and logs "nearest tip is N hours away"; it starts spending
        # only when a game is genuinely close.
        ("odds recorder (NBA)", ["snapshot_odds_api.py"]),
    ],
    "hourly": [
        ("injury recorder (NFL)", ["src/Sports/nfl/poll_injuries.py"]),
        ("predictions (NFL)", ["src/Sports/nfl/predict.py"]),
        # Last in the job, so the public copy gets this hour's picks. Says
        # SKIPPED (exit 0) until a public server is configured.
        ("publish ledger", ["push_ledger.py"]),
    ],
    "daily": [
        # FIRST, because everything below it grades, seals or prices against
        # a final score, and until 2026-09-21 nothing in any scheduled job
        # fetched one. The ledger had fourteen finished games sitting at
        # `pending_past_kickoff` and would have sat there all season: the
        # grader was working perfectly and being handed a table in which no
        # 2026 game had ever finished. The rehearsal could not catch this --
        # it grades archive games, which already have their scores.
        #
        # nflverse refreshes the schedules release daily and the ingest is
        # idempotent (INSERT OR REPLACE on a natural key), so re-running it
        # every morning costs one download and converges on the truth.
        #
        # The season is derived, not typed. An NFL season carries the year it
        # kicked off in, so January and February belong to the season before
        # them; hardcoding "2026" would have quietly ingested the wrong year
        # from 1 January and nobody would have noticed until the ledger
        # stopped grading again.
        ("ingest results (NFL)", ["src/Sports/nfl/ingest_games.py",
                                  "--season", str(_nfl_season())]),
        # The same gap one layer down, found by the validator the moment the
        # step above started marking games final: week 2 had play-by-play for
        # one game out of fifteen. Nothing was keeping the current season's
        # plays current either.
        #
        # --force is needed because the ingest skips a season it already
        # holds, and the current season is always one we already hold and
        # always incomplete. It is safe: every write is INSERT OR REPLACE on
        # (game_id, sequence), nothing is deleted, so a half-finished run
        # simply leaves the rest for tomorrow. The whole season re-ingests in
        # about 4 seconds off a 10 MB file in September, growing to ~100 MB
        # by January, which is still cheaper than tracking which games are
        # new.
        ("ingest play-by-play (NFL)", ["src/Sports/nfl/ingest_pbp.py",
                                       "--season", str(_nfl_season()), "--force"]),
        ("grade ledger (NFL)", ["src/Sports/nfl/predict.py", "--grade"]),
        # Repair before the seal, so anything rebuilt tonight is sealed by the
        # step after it as well as by the repair's own seal.
        ("repair missed odds (NFL)", ["src/Sports/repair_odds.py", "--sport", "nfl",
                                      "--unattended", "--apply"]),
        ("odds seal (NFL)", ["src/Sports/odds_recorder.py", "--sport", "nfl", "--seal"]),
        # Last, because it reads the closing lines the two steps above produce.
        ("closing line value (NFL)", ["src/Sports/nfl/predict.py", "--clv"]),
        ("publish ledger", ["push_ledger.py"]),
    ],
}


def _tail(result: subprocess.CompletedProcess, lines: int = 2) -> list:
    """The last couple of lines a job said, wherever it said them.

    This used to read `stdout` only, and every one of these jobs reports
    through `logging`, which writes to STDERR. So the tail was always empty
    and every run logged a bare "ok". On 2026-09-21 that hid a real failure
    for a full day: the grader was returning `still_pending: 31` with
    fourteen finished games waiting, and the daily log said "grade ledger
    (NFL): ok" exactly as it does on a good morning.

    A monitoring line that cannot distinguish success from doing nothing is
    worse than no monitoring line, because it is trusted. Read both streams,
    prefer whichever carried the substance, and drop the timestamp prefix so
    the message survives the wrapper's own formatting.
    """
    for stream in (result.stderr, result.stdout):
        got = [ln.strip() for ln in (stream or "").strip().split("\n") if ln.strip()]
        if got:
            return [_strip_stamp(ln) for ln in got[-lines:]]
    return []


def _strip_stamp(line: str) -> str:
    """Drop a leading '2026-09-21 09:30:02,081 - INFO - ' if there is one."""
    for sep in (" - INFO - ", " - WARNING - ", " - ERROR - "):
        if sep in line:
            return line.split(sep, 1)[1]
    return line


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in JOBS:
        print(f"usage: run_scheduled.py {{{'|'.join(JOBS)}}}")
        return 2
    mode = sys.argv[1]

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"scheduled_{mode}.log"),
                                      encoding="utf-8"),
                  logging.StreamHandler()],
    )
    logger = logging.getLogger(f"scheduled.{mode}")
    logger.info("=== %s run starting (%s) ===", mode, datetime.now(timezone.utc).isoformat())

    failures = []
    for name, argv in JOBS[mode]:
        try:
            r = subprocess.run([PY] + argv, cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=900, encoding="utf-8", errors="replace")
            tail = _tail(r)
            if r.returncode == 0:
                logger.info("%s: ok%s", name, (" | " + " | ".join(tail)) if tail else "")
            else:
                failures.append(name)
                logger.error("%s: exit %d | %s", name, r.returncode,
                             (r.stderr or "").strip()[-300:])
        except subprocess.TimeoutExpired:
            failures.append(name)
            logger.error("%s: timed out after 15 minutes", name)
        except Exception as exc:                       # one dead job, not three
            failures.append(name)
            logger.error("%s: %s", name, str(exc)[:300])

    if failures:
        logger.error("=== %s run finished with %d failure(s): %s ===",
                     mode, len(failures), ", ".join(failures))
        return 1
    logger.info("=== %s run finished clean ===", mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
