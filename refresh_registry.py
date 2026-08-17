"""
refresh_registry.py
===================
Keeps every dataset that is NOT read live from a feed up to date on its own
cadence, so nobody has to remember to re-run an ingest.

Most of the site is already self-updating, and this exists only for the parts
that cannot be:

  live on request      - schedule, NBA Cup, news, transactions, player headshots,
                         live scores. Fetched per request with a short cache, so
                         they are as fresh as their upstream feed.
  daily via backfill   - box scores, derived ratings, team-stats snapshot,
                         predictions log. Handled by daily_update.py directly.
  periodic ingests     - everything registered below. These pull from sources
                         with no API (a scrape) or are expensive enough that
                         per-request fetching would be rude to the upstream.

A job runs when it has not run for `interval_days`. Outcomes are recorded in
`ingest_runs`, which is what makes this honest: every page that displays ingested
data also displays its fetch time, so a job that starts failing shows up on the
site as a stale date rather than as silently old numbers.

Adding a dataset: append one Job below. Nothing else needs changing - it will be
picked up on the next daily run.
"""

import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("refresh_registry")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_runs (
    name TEXT PRIMARY KEY,
    last_run_at TEXT,
    last_status TEXT,
    last_detail TEXT,
    consecutive_failures INTEGER DEFAULT 0
)
"""


@dataclass
class Job:
    name: str
    interval_days: int
    run: Callable[[], str]
    why: str


def _hall_of_fame() -> str:
    """The Naismith register. A new class is enshrined each September."""
    from ingest_hall_of_fame import main as run
    if run() != 0:
        raise RuntimeError("ingest_hall_of_fame reported failure")
    conn = sqlite3.connect(DB_PATH)
    try:
        n = conn.execute("SELECT COUNT(*) FROM hof_inductees").fetchone()[0]
    finally:
        conn.close()
    return f"{n} inductees"


def _hof_careers() -> str:
    """Career totals for inducted players. Cheap on repeat - already-stored
    players are skipped, so a weekly run only fetches a new class."""
    from ingest_hof_careers import main as run
    if run() != 0:
        raise RuntimeError("ingest_hof_careers reported failure")
    conn = sqlite3.connect(DB_PATH)
    try:
        n = conn.execute("SELECT COUNT(*) FROM hof_career_totals").fetchone()[0]
    finally:
        conn.close()
    return f"{n} careers"


def _draft_history() -> str:
    """Draft classes. A one-time helper in main_api populated this and then
    returned early forever, so the 2026 class never appeared - the table sat on
    2025 while the league had all 60 picks. Re-checking the recent classes
    weekly costs two requests and closes that hole for good."""
    from ingest_draft import main as run
    if run() != 0:
        raise RuntimeError("ingest_draft reported failure")
    conn = sqlite3.connect(DB_PATH)
    try:
        n, newest = conn.execute(
            "SELECT COUNT(*), MAX(season) FROM draft_history"
        ).fetchone()
    finally:
        conn.close()
    return f"{n} picks, newest {newest}"


# Cadences are set by how fast the underlying truth moves, not by habit.
# The Hall inducts once a year; weekly means a new class appears within days
# without hammering a site that changes eleven times a decade.
JOBS: List[Job] = [
    Job("hall_of_fame", 7, _hall_of_fame, "New class enshrined each September"),
    Job("hof_careers", 7, _hof_careers, "Career totals for any newly inducted players"),
    Job("draft_history", 7, _draft_history, "New draft class each June, plus late pick corrections"),
]


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def due(job: Job, now: datetime, conn) -> bool:
    row = conn.execute(
        "SELECT last_run_at, last_status FROM ingest_runs WHERE name = ?", (job.name,)
    ).fetchone()
    if not row or not row["last_run_at"]:
        return True
    try:
        last = datetime.fromisoformat(row["last_run_at"])
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    # A job that failed last time retries on the next run rather than waiting out
    # its full interval - a week-long gap after a transient network error would
    # leave the site stale for no reason.
    if row["last_status"] != "ok":
        return True
    return now - last >= timedelta(days=job.interval_days)


def run_due(force: Optional[str] = None) -> bool:
    """Run every job that is due. Returns False if any job failed."""
    now = datetime.now(timezone.utc)
    conn = _conn()
    ok = True
    try:
        for job in JOBS:
            if force and force != job.name:
                continue
            if not force and not due(job, now, conn):
                row = conn.execute(
                    "SELECT last_run_at FROM ingest_runs WHERE name = ?", (job.name,)
                ).fetchone()
                logger.info(
                    "%s: up to date (last run %s, every %dd)",
                    job.name, (row["last_run_at"] or "?")[:10], job.interval_days,
                )
                continue

            logger.info("%s: running - %s", job.name, job.why)
            try:
                # The ingests are also standalone scripts and parse sys.argv in
                # main(). Called from here they would inherit ours and argparse
                # would reject the job name, so argv is neutralised for the call
                # - centrally, because every job added later has the same trap.
                saved_argv = sys.argv
                sys.argv = [job.name]
                try:
                    detail = job.run()
                finally:
                    sys.argv = saved_argv
                conn.execute(
                    """
                    INSERT INTO ingest_runs (name, last_run_at, last_status, last_detail,
                                             consecutive_failures)
                    VALUES (?, ?, 'ok', ?, 0)
                    ON CONFLICT(name) DO UPDATE SET
                        last_run_at=excluded.last_run_at, last_status='ok',
                        last_detail=excluded.last_detail, consecutive_failures=0
                    """,
                    (job.name, now.isoformat(), detail),
                )
                conn.commit()
                logger.info("%s: ok - %s", job.name, detail)
            except Exception as exc:
                ok = False
                conn.execute(
                    """
                    INSERT INTO ingest_runs (name, last_run_at, last_status, last_detail,
                                             consecutive_failures)
                    VALUES (?, ?, 'failed', ?, 1)
                    ON CONFLICT(name) DO UPDATE SET
                        last_run_at=excluded.last_run_at, last_status='failed',
                        last_detail=excluded.last_detail,
                        consecutive_failures=consecutive_failures + 1
                    """,
                    (job.name, now.isoformat(), str(exc)[:400]),
                )
                conn.commit()
                logger.error("%s: FAILED - %s", job.name, exc, exc_info=True)
    finally:
        conn.close()
    return ok


def status() -> List[dict]:
    conn = _conn()
    try:
        rows = {r["name"]: dict(r) for r in conn.execute("SELECT * FROM ingest_runs")}
    finally:
        conn.close()
    return [
        {
            "name": j.name,
            "interval_days": j.interval_days,
            "why": j.why,
            **(rows.get(j.name) or {"last_run_at": None, "last_status": "never run"}),
        }
        for j in JOBS
    ]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    force = sys.argv[1] if len(sys.argv) > 1 else None
    if force == "--status":
        for s in status():
            print(f"{s['name']:<16} every {s['interval_days']:>2}d  "
                  f"last {(s.get('last_run_at') or 'never')[:19]:<19} "
                  f"{s.get('last_status')}  {s.get('last_detail') or ''}")
        return 0
    return 0 if run_due(force) else 1


if __name__ == "__main__":
    sys.exit(main())
