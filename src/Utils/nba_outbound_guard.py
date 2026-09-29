"""
A hard ceiling on stats.nba.com traffic from this machine, shared by every
process on it (the API server, the daily job, any backfill).

WHY (2026-09-28)
A reference backfill sent about 12,000 requests over roughly ten hours on
2026-09-24/25, paced one at a time. nba.com's edge (Akamai) answered by
blocking the home PC's address on both stats.nba.com and cdn.nba.com
("Access Denied"). Pacing is not enough: the edge judges volume and pattern.
The home PC is the only machine that can ingest the season, so a block on it
is the one failure the product cannot route around. This module makes that
volume impossible to reach by accident, whoever starts the job:

  - a DAILY budget (NBA_DAILY_REQUEST_BUDGET, default 600) and an HOURLY one
    (NBA_HOURLY_REQUEST_BUDGET, default 200), counted per calendar day/hour
    in UTC across all processes;
  - a CIRCUIT BREAKER: NBA_BREAKER_FAILURES (default 10) failures in a row
    (a timeout, a connection error or a non-200 answer) stop every request
    for NBA_BREAKER_HOURS (default 72). When the cooldown ends, ONE request is
    let through as a probe: success closes the breaker, failure re-opens it
    for twice as long (at most a week). A block is never "retried through".

A refused request raises OutboundRefused, a LiveFetchDisabled, so every
caller that already degrades when live fetching is off (the API's 503
"live-only", the cache and the mirror) degrades the same way here. Nothing
waits and nothing touches the network when refused.

The state lives in Data/nba_outbound.sqlite (its own small file, so it never
contends with the archive's writers). An owner-approved large job may raise
the budgets for that run only, through the environment; that is the one way
past the ceiling, and it is deliberate.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
MAX_COOLDOWN_HOURS = 24 * 7


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def daily_budget() -> int:
    return _int_env("NBA_DAILY_REQUEST_BUDGET", 600)


def hourly_budget() -> int:
    return _int_env("NBA_HOURLY_REQUEST_BUDGET", 200)


def breaker_failures() -> int:
    return max(1, _int_env("NBA_BREAKER_FAILURES", 10))


def breaker_hours() -> int:
    return max(1, _int_env("NBA_BREAKER_HOURS", 72))


def db_path() -> Path:
    env = os.environ.get("NBA_OUTBOUND_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "Data" / "nba_outbound.sqlite"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS request_counts (
            bucket TEXT PRIMARY KEY,          -- 'd:YYYY-MM-DD' or 'h:YYYY-MM-DDTHH'
            n      INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS breaker (
            id               INTEGER PRIMARY KEY CHECK (id = 1),
            failure_streak   INTEGER NOT NULL DEFAULT 0,
            open_until       TEXT,              -- UTC ISO; NULL = closed
            cooldown_hours   INTEGER NOT NULL DEFAULT 0,
            probing          INTEGER NOT NULL DEFAULT 0,
            last_failure     TEXT,
            last_success     TEXT,
            last_reason      TEXT
        );
        INSERT OR IGNORE INTO breaker (id) VALUES (1);
        """
    )
    return conn


class OutboundRefused(Exception):
    """Placeholder until nba_stats_client binds it to LiveFetchDisabled."""


def _refused(message: str) -> Exception:
    # Bound at import time by nba_stats_client so OutboundRefused IS a
    # LiveFetchDisabled and every existing handler treats it the same way.
    return OutboundRefused(message)


def _now(now: Optional[datetime]) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def before_request(endpoint: str, now: Optional[datetime] = None) -> None:
    """Claim one request, or raise OutboundRefused without touching the network."""
    t = _now(now)
    day, hour = f"d:{t:%Y-%m-%d}", f"h:{t:%Y-%m-%dT%H}"
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            b = conn.execute(
                "SELECT failure_streak, open_until, cooldown_hours, probing FROM breaker WHERE id = 1").fetchone()
            open_until = datetime.fromisoformat(b[1]) if b[1] else None
            if open_until and t < open_until:
                conn.execute("ROLLBACK")
                raise _refused(
                    f"stats.nba.com requests are paused until {open_until:%Y-%m-%d %H:%M} UTC "
                    f"after {b[0]} failures in a row ('{endpoint}' not sent)")
            if open_until and b[3]:
                # The one probe after a cooldown is already in flight.
                conn.execute("ROLLBACK")
                raise _refused(f"a probe of stats.nba.com is already in flight ('{endpoint}' not sent)")
            used_day = (conn.execute("SELECT n FROM request_counts WHERE bucket = ?", (day,)).fetchone() or [0])[0]
            used_hour = (conn.execute("SELECT n FROM request_counts WHERE bucket = ?", (hour,)).fetchone() or [0])[0]
            if used_day >= daily_budget():
                conn.execute("ROLLBACK")
                raise _refused(f"today's stats.nba.com budget is spent ({used_day}/{daily_budget()}; '{endpoint}' not sent)")
            if used_hour >= hourly_budget():
                conn.execute("ROLLBACK")
                raise _refused(f"this hour's stats.nba.com budget is spent ({used_hour}/{hourly_budget()}; '{endpoint}' not sent)")
            for bucket in (day, hour):
                conn.execute("INSERT INTO request_counts (bucket, n) VALUES (?, 1) "
                             "ON CONFLICT(bucket) DO UPDATE SET n = n + 1", (bucket,))
            if open_until:  # cooldown over: this request is the probe
                conn.execute("UPDATE breaker SET probing = 1 WHERE id = 1")
            conn.execute("COMMIT")
        finally:
            conn.close()


def record_result(ok: bool, reason: str = "", now: Optional[datetime] = None) -> None:
    """After a request: success closes the breaker; enough failures open it."""
    t = _now(now)
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            streak, open_until, cooldown, probing = conn.execute(
                "SELECT failure_streak, open_until, cooldown_hours, probing FROM breaker WHERE id = 1").fetchone()
            if ok:
                conn.execute("UPDATE breaker SET failure_streak = 0, open_until = NULL, cooldown_hours = 0, "
                             "probing = 0, last_success = ? WHERE id = 1", (t.isoformat(),))
            else:
                streak += 1
                if probing:
                    cooldown = min(MAX_COOLDOWN_HOURS, max(breaker_hours(), cooldown) * 2)
                    opened = t + timedelta(hours=cooldown)
                elif streak >= breaker_failures():
                    cooldown = breaker_hours()
                    opened = t + timedelta(hours=cooldown)
                else:
                    opened = None
                conn.execute(
                    "UPDATE breaker SET failure_streak = ?, open_until = COALESCE(?, open_until), "
                    "cooldown_hours = ?, probing = 0, last_failure = ?, last_reason = ? WHERE id = 1",
                    (streak, opened.isoformat() if opened else None, cooldown, t.isoformat(), reason[:300]))
            conn.execute("COMMIT")
        finally:
            conn.close()


def status(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Read-only summary for the health check and the owner."""
    t = _now(now)
    conn = _connect()
    try:
        streak, open_until, cooldown, probing, last_f, last_s, reason = conn.execute(
            "SELECT failure_streak, open_until, cooldown_hours, probing, last_failure, last_success, last_reason "
            "FROM breaker WHERE id = 1").fetchone()
        used_day = (conn.execute("SELECT n FROM request_counts WHERE bucket = ?", (f"d:{t:%Y-%m-%d}",)).fetchone() or [0])[0]
        used_hour = (conn.execute("SELECT n FROM request_counts WHERE bucket = ?", (f"h:{t:%Y-%m-%dT%H}",)).fetchone() or [0])[0]
    finally:
        conn.close()
    is_open = bool(open_until) and t < datetime.fromisoformat(open_until)
    return {
        "breaker": "open" if is_open else ("probe_next" if open_until else "closed"),
        "open_until": open_until, "failure_streak": streak, "cooldown_hours": cooldown,
        "last_failure": last_f, "last_success": last_s, "last_reason": reason,
        "used_today": used_day, "daily_budget": daily_budget(),
        "used_this_hour": used_hour, "hourly_budget": hourly_budget(),
    }


def response_ok(response: Any) -> bool:
    """nba_api returns non-200 answers (Akamai's 403 page) without raising."""
    code = getattr(response, "_status_code", None)
    return code is None or code == 200


if __name__ == "__main__":  # python -m src.Utils.nba_outbound_guard
    import json
    print(json.dumps(status(), indent=2))
