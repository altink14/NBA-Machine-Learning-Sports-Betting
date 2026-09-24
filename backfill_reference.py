"""
backfill_reference.py
=====================
Fills the three reference tables that were only ever filled on demand, so the
site and the chat can answer league-wide questions from them:

  awards    player_awards: every player's official award history
            (playerawards). Before this, 15 players had one, so "who won MVP
            in 2011" had no answer unless someone had happened to open that
            player's page.
  coaches   team_coaches: every archived team-season's staff
            (commonteamroster). Before this, 33 team-seasons.
  careers   player_career_official: every player's official season-by-season
            and career totals (playercareerstats). Before this, 75 players,
            so there was no all-time career leaderboard.

It reuses main_api's own fetchers (_ensure_player_awards,
_ensure_team_coaches, _ensure_career_official), so the rows are exactly what
the lazy path writes, and runs them through the shared stats client in
backfill mode (2 s between requests). About 12,000 requests in all: roughly
8-10 hours. One stats.nba.com job at a time: do not run this alongside
another backfill.

RESUMABLE. reference_fetch records every key asked about (a player id or a
team-season), including the ones nba.com answered with nothing, so a rerun
skips them; a kill loses at most the request in flight. --refresh ignores
the log.

MANNERS. Players with the longest careers go first, so the award winners
and career leaders land in the first hour. After 20 failures in a row the
job pauses for ten minutes (nba.com throttling looks like that), and after
three such pauses it stops and says so.

    venv/Scripts/python.exe backfill_reference.py                 # all three, in order
    venv/Scripts/python.exe backfill_reference.py --parts awards  # one part
    venv/Scripts/python.exe backfill_reference.py --limit 20      # a quick trial
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from src.Utils.nba_stats_client import get_client

# Backfill pacing must be chosen before anything creates the shared client.
get_client(backfill_mode=True)

import main_api  # noqa: E402  (after the client, deliberately)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("backfill_reference")

PARTS = ("awards", "coaches", "careers")
MAX_FAILS_IN_A_ROW = 20
PAUSE_SECONDS = 600
MAX_PAUSES = 3


def ensure_log(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reference_fetch (
            kind TEXT, key TEXT, fetched_at TEXT, n_rows INTEGER, error TEXT,
            PRIMARY KEY (kind, key)
        )
        """
    )
    conn.commit()


def done_keys(conn, kind: str) -> set:
    return {r[0] for r in conn.execute(
        "SELECT key FROM reference_fetch WHERE kind = ? AND error IS NULL", (kind,)
    )}


def record(conn, kind: str, key: str, n_rows, error=None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO reference_fetch (kind, key, fetched_at, n_rows, error) VALUES (?, ?, ?, ?, ?)",
        (kind, key, datetime.utcnow().isoformat(), n_rows, error),
    )
    conn.commit()


def players_by_career(conn) -> list:
    # Longest careers first: that is where the awards and the career leaders are.
    return [r[0] for r in conn.execute(
        """
        SELECT player_id FROM players
        ORDER BY (COALESCE(to_year, from_year, 0) - COALESCE(from_year, to_year, 0)) DESC,
                 COALESCE(to_year, 0) DESC, player_id
        """
    )]


def team_seasons(conn) -> list:
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT DISTINCT team_id, season FROM team_game_advanced ORDER BY season DESC, team_id"
    )]


class Guard:
    """Counts failures in a row; pauses, then gives up, when nba.com pushes back."""

    def __init__(self):
        self.fails = 0
        self.pauses = 0

    def ok(self):
        self.fails = 0

    def failed(self) -> bool:
        """True when the job should stop."""
        self.fails += 1
        if self.fails < MAX_FAILS_IN_A_ROW:
            return False
        self.pauses += 1
        if self.pauses > MAX_PAUSES:
            log.error("Stopping: %d failures in a row, %d pauses already. Rerun later; it resumes.", self.fails, MAX_PAUSES)
            return True
        log.warning("%d failures in a row: pausing %d s (pause %d of %d)", self.fails, PAUSE_SECONDS, self.pauses, MAX_PAUSES)
        time.sleep(PAUSE_SECONDS)
        self.fails = 0
        return False


def run_awards(conn, guard, limit, refresh) -> bool:
    done = set() if refresh else done_keys(conn, "awards")
    todo = [p for p in players_by_career(conn) if str(p) not in done][: limit or None]
    log.info("awards: %d players to ask (%d already done)", len(todo), len(done))
    for i, pid in enumerate(todo, 1):
        try:
            # Force a fetch: the lazy path would skip anything seen this week.
            conn.execute("DELETE FROM player_awards WHERE player_id = ?", (pid,)) if refresh else None
            main_api._ensure_player_awards(conn, pid)
            n = conn.execute("SELECT COUNT(*) FROM player_awards WHERE player_id = ?", (pid,)).fetchone()[0]
            record(conn, "awards", str(pid), n)
            guard.ok()
        except Exception as exc:  # one bad id must not end the run
            record(conn, "awards", str(pid), None, str(exc)[:300])
            log.warning("awards %s failed: %s", pid, exc)
            if guard.failed():
                return False
        if i % 100 == 0:
            log.info("awards: %d / %d", i, len(todo))
    return True


def run_coaches(conn, guard, limit, refresh) -> bool:
    done = set() if refresh else done_keys(conn, "coaches")
    todo = [(t, s) for t, s in team_seasons(conn) if f"{t}|{s}" not in done][: limit or None]
    log.info("coaches: %d team-seasons to ask (%d already done)", len(todo), len(done))
    for i, (team_id, season) in enumerate(todo, 1):
        try:
            n = main_api._ensure_team_coaches(conn, team_id, season, max_age_days=0 if refresh else 7)
            record(conn, "coaches", f"{team_id}|{season}", n)
            guard.ok()
        except Exception as exc:
            record(conn, "coaches", f"{team_id}|{season}", None, str(exc)[:300])
            log.warning("coaches %s %s failed: %s", team_id, season, exc)
            if guard.failed():
                return False
        if i % 50 == 0:
            log.info("coaches: %d / %d", i, len(todo))
    return True


def run_careers(conn, guard, limit, refresh) -> bool:
    done = set() if refresh else done_keys(conn, "careers")
    todo = [p for p in players_by_career(conn) if str(p) not in done][: limit or None]
    log.info("careers: %d players to ask (%d already done)", len(todo), len(done))
    for i, pid in enumerate(todo, 1):
        try:
            if refresh:
                conn.execute("DELETE FROM player_career_official WHERE player_id = ?", (pid,))
            main_api._ensure_career_official(conn, pid)
            n = conn.execute(
                "SELECT COUNT(*) FROM player_career_official WHERE player_id = ? AND season != '__EMPTY__'", (pid,)
            ).fetchone()[0]
            record(conn, "careers", str(pid), n)
            guard.ok()
        except Exception as exc:
            record(conn, "careers", str(pid), None, str(exc)[:300])
            log.warning("careers %s failed: %s", pid, exc)
            if guard.failed():
                return False
        if i % 100 == 0:
            log.info("careers: %d / %d", i, len(todo))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default=",".join(PARTS), help="comma list of: " + ", ".join(PARTS))
    ap.add_argument("--limit", type=int, default=0, help="at most this many keys per part (a trial run)")
    ap.add_argument("--refresh", action="store_true", help="ignore the log and ask again")
    args = ap.parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        ap.error(f"unknown part(s): {', '.join(bad)}")

    conn = main_api.get_db_conn()
    try:
        ensure_log(conn)
        guard = Guard()
        runners = {"awards": run_awards, "coaches": run_coaches, "careers": run_careers}
        for part in parts:
            if not runners[part](conn, guard, args.limit, args.refresh):
                return 1
        log.info("done: %s", ", ".join(parts))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
