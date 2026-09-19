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

  daily      grading, repair, sealing. Yesterday's games are final by the
             morning, which is also when it is clear whether a capture was
             missed and worth buying back.

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

JOBS = {
    "frequent": [
        ("odds recorder (NFL)", ["src/Sports/odds_recorder.py", "--sport", "nfl"]),
    ],
    "hourly": [
        ("injury recorder (NFL)", ["src/Sports/nfl/poll_injuries.py"]),
        ("predictions (NFL)", ["src/Sports/nfl/predict.py"]),
    ],
    "daily": [
        ("grade ledger (NFL)", ["src/Sports/nfl/predict.py", "--grade"]),
        # Repair before the seal, so anything rebuilt tonight is sealed by the
        # step after it as well as by the repair's own seal.
        ("repair missed odds (NFL)", ["src/Sports/repair_odds.py", "--sport", "nfl",
                                      "--unattended", "--apply"]),
        ("odds seal (NFL)", ["src/Sports/odds_recorder.py", "--sport", "nfl", "--seal"]),
    ],
}


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
            tail = [ln for ln in (r.stdout or "").strip().split("\n") if ln][-2:]
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
