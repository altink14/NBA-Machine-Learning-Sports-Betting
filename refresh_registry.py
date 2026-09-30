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


#: How much earlier than its interval a job may run. Less than a day, so a
#: daily job still cannot run twice on one morning's retries.
DUE_SLACK = timedelta(hours=3)


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


def _player_directory() -> str:
    """Every player in league history, with roster status and career span.

    Supersedes the roster-status job: same upstream request, and the directory was
    the bigger problem. It used to hold only players from our ingested box scores,
    so the player encyclopedia could not find Michael Jordan. Daily because roster
    status moves constantly in season, and because the league's own flag lags a
    retirement announcement by days."""
    from ingest_players import main as run
    if run() != 0:
        raise RuntimeError("ingest_players reported failure")
    conn = sqlite3.connect(DB_PATH)
    try:
        total, active = conn.execute(
            "SELECT COUNT(*), SUM(is_active) FROM players"
        ).fetchone()
    finally:
        conn.close()
    return f"{total} players, {active or 0} active"


def _draft_bios() -> str:
    """Position, height, weight and country for recent draft classes, which the
    draft board shows alongside each pick. Only players missing a bio are
    fetched, so this costs nothing once a class is filled - and it keeps trying
    the handful whose bio the league has not published yet."""
    from ingest_draft_bios import main as run
    if run() != 0:
        raise RuntimeError("ingest_draft_bios reported failure")
    conn = sqlite3.connect(DB_PATH)
    try:
        filled = conn.execute(
            "SELECT COUNT(*) FROM draft_history d JOIN player_bio b "
            "ON b.player_id = d.person_id WHERE d.season = (SELECT MAX(season) FROM draft_history)"
        ).fetchone()[0]
    finally:
        conn.close()
    return f"{filled} bios in the newest class"


def _player_bios() -> str:
    """Position, height, college and last team for the directory.

    Only the current season's bucket is refreshed on a schedule. playerindex
    partitions on TO_YEAR, so historical buckets are closed sets that never
    change - re-walking eighty of them weekly would be eighty requests to learn
    nothing. A player who retires moves into a past bucket, and the current-season
    call is where that shows up.

    Run ingest_player_bios_bulk.py by hand with no flag to rebuild all of history.
    """
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "ingest_player_bios_bulk.py"), "--only-current"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "ingest_player_bios_bulk failed")[-300:])
    conn = sqlite3.connect(DB_PATH)
    try:
        withbio, total = conn.execute(
            "SELECT SUM(CASE WHEN position IS NOT NULL AND position != '' THEN 1 ELSE 0 END), "
            "COUNT(*) FROM players"
        ).fetchone()
    finally:
        conn.close()
    return f"{withbio or 0} of {total} with a bio"


def _officials() -> str:
    """Officiating crews for the newest archived season.

    Nothing ran this on a schedule until 2026-09-22; it had only ever been run
    by hand, which is how 1,212 of 1,315 games in 2025-26 went a season with no
    crew. Daily, because the season adds games daily. --retry-empty re-asks any
    game still recorded crewless (the newest games can lag), and since that
    goes to boxscoresummaryv3 it is one request per such game, a handful a day.
    """
    import subprocess
    conn = sqlite3.connect(DB_PATH)
    try:
        season = conn.execute("SELECT MAX(season) FROM box_scores").fetchone()[0]
    finally:
        conn.close()
    r = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "src", "Process-Data", "backfill_officials.py"),
         "--seasons", season, "--retry-empty"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "backfill_officials failed")[-300:])
    conn = sqlite3.connect(DB_PATH)
    try:
        with_crew, games = conn.execute(
            "SELECT SUM(f.n_officials > 0), COUNT(*) FROM "
            "(SELECT DISTINCT game_id FROM box_scores WHERE season = ?) b "
            "LEFT JOIN officials_fetch f USING (game_id)", (season,)).fetchone()
    finally:
        conn.close()
    return f"{with_crew or 0} of {games} {season} games with a crew"


def _game_summaries() -> str:
    """Line scores, attendance, duration, national TV and inactive lists for
    the newest season, from the game summaries already on disk.

    No network: it reads the cache only. The officials job just before it is
    what puts each new game's boxscoresummaryv3 there (v2 has been an empty
    shell since 2025-04-10, so every crew comes from v3). Until 2026-09-29
    this ingest had only ever been run by hand, which is how 1,203 of
    2025-26's games had no attendance, line score or inactive list. Fails
    only if the ingest itself fails; games still empty are counted, since the
    newest night's summaries can lag a day.
    """
    import subprocess
    conn = sqlite3.connect(DB_PATH)
    try:
        season = conn.execute("SELECT MAX(season) FROM box_scores").fetchone()[0]
    finally:
        conn.close()
    r = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "src", "Process-Data", "ingest_game_summaries.py"),
         "--season", season],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "ingest_game_summaries failed")[-300:])
    return _summary_coverage(DB_PATH, season)


def _summary_coverage(db_path: str, season: str) -> str:
    """'<season>: N of M games with a line score (K from v3), E empty'."""
    conn = sqlite3.connect(db_path)
    try:
        games, with_line, from_v3 = conn.execute(
            """
            SELECT COUNT(*),
                   SUM(EXISTS (SELECT 1 FROM game_line_scores l
                               WHERE l.game_id = b.game_id AND l.pts IS NOT NULL)),
                   SUM(EXISTS (SELECT 1 FROM game_info i
                               WHERE i.game_id = b.game_id AND i.source = 'boxscoresummaryv3'))
            FROM box_scores b WHERE b.season = ?
            """, (season,)).fetchone()
    finally:
        conn.close()
    games, with_line, from_v3 = games or 0, with_line or 0, from_v3 or 0
    return (f"{season}: {with_line} of {games} games with a line score "
            f"({from_v3} from v3), {games - with_line} empty")


def _server_mirror() -> str:
    """The public server's copy of the league-wide stats.nba.com tables.

    The server cannot reach stats.nba.com (it refuses cloud IPs) and is given
    only this database, so the schedule, play types, tracking tables, hustle,
    lineups and zone averages it serves come from `nba_response_mirror`,
    which this fills (src/Utils/nba_mirror.py). The current season only, disk
    cache first; a season-type stored after it ended is never asked for again,
    so the offseason costs no requests. Fails only when every request failed.

    Off until NBA_MIRROR_REFRESH=on is in this repo's .env (daily_update.py
    loads it): added on 2026-09-28 while another job owned stats.nba.com, and
    its first run asks for ~60 tables, so it waits to be switched on.
    """
    if (os.environ.get("NBA_MIRROR_REFRESH") or "").strip().lower() not in ("on", "1", "true", "yes"):
        return "skipped: switch on with NBA_MIRROR_REFRESH=on in .env"
    from src.Utils import nba_mirror
    from datetime import date
    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        newest = conn.execute("SELECT MAX(season) FROM box_scores").fetchone()[0]
        counts = nba_mirror.refresh_live(
            conn, newest, nba_mirror.schedule_season_for(date.today(), newest))
    finally:
        conn.close()
    if counts["failed"] and not counts["copied"]:
        raise RuntimeError(f"no table copied: {counts}")
    return (f"{counts['copied']} copied, {counts['already_final']} final, "
            f"{counts['failed']} failed, {counts['no_games_yet']} not started")


# Cadences are set by how fast the underlying truth moves, not by habit.
# The Hall inducts once a year; weekly means a new class appears within days
# without hammering a site that changes eleven times a decade.
JOBS: List[Job] = [
    Job("hall_of_fame", 7, _hall_of_fame, "New class enshrined each September"),
    Job("hof_careers", 7, _hof_careers, "Career totals for any newly inducted players"),
    Job("player_directory", 1, _player_directory, "Roster status moves daily; new players arrive mid-season"),
    Job("draft_history", 7, _draft_history, "New draft class each June, plus late pick corrections"),
    Job("draft_bios", 7, _draft_bios, "Bios for the newest picks appear over the weeks after the draft"),
    Job("player_bios", 7, _player_bios, "Current-season bios; historical buckets never change"),
    Job("officials", 1, _officials, "New games need their crews; the newest can lag a day"),
    # After officials: that job is what caches each new game's v3 summary.
    Job("game_summaries", 1, _game_summaries, "New games need line scores, attendance and inactive lists"),
    Job("server_mirror", 1, _server_mirror, "The public server cannot reach stats.nba.com; it reads this copy"),
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
    # A few hours of slack. The daily task starts at 9:00 but each job is
    # stamped when it FINISHES, a few minutes later; the next morning's check
    # then came 7 minutes short of 24 hours and skipped. Found 2026-09-23:
    # player_directory (every 1d) was running every other day, and every
    # weekly job was slipping to 8 days.
    return now - last >= timedelta(days=job.interval_days) - DUE_SLACK


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
