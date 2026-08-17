"""
ingest_draft.py
===============
Keeps `draft_history` current.

The table was populated by a one-time helper in main_api that returns early once
the table has any rows at all, so it never picked up a new class. On 2026-08-17
the table stopped at the 2025 draft while stats.nba.com had all 60 picks of the
2026 draft, AJ Dybantsa first to Washington. Nothing would ever have noticed.

This ingest is cheap to repeat, which is what lets it run on a schedule:
  - the full history is fetched only if the table is empty
  - otherwise it re-fetches the recent classes only, and upserts

Recent classes are re-fetched rather than skipped because a draft is not
immutable for long: picks get traded and recorded late, and second-round picks
and withdrawals get corrected in the days after the draft.

Run standalone, or let refresh_registry.py call it:
    venv/Scripts/python.exe ingest_draft.py [--full]
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("draft_ingest")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS draft_history (
    person_id INTEGER,
    player_name TEXT,
    season INTEGER,
    round_number INTEGER,
    round_pick INTEGER,
    overall_pick INTEGER,
    team_id INTEGER,
    team_city TEXT,
    team_name TEXT,
    team_abbreviation TEXT,
    organization TEXT,
    organization_type TEXT,
    fetched_at TEXT,
    PRIMARY KEY (season, overall_pick, person_id)
)
"""

COLUMNS = [
    "PERSON_ID", "PLAYER_NAME", "SEASON", "ROUND_NUMBER", "ROUND_PICK",
    "OVERALL_PICK", "TEAM_ID", "TEAM_CITY", "TEAM_NAME", "TEAM_ABBREVIATION",
    "ORGANIZATION", "ORGANIZATION_TYPE",
]

# How many recent classes to re-check on every run. Two covers a draft that has
# just happened plus the one before it, which is where late corrections land.
RECENT_YEARS = 2


def _rows(season: str | None):
    from nba_api.stats.endpoints import drafthistory

    kwargs = {"league_id": "00", "timeout": 90}
    if season:
        kwargs["season_year_nullable"] = season
    data = drafthistory.DraftHistory(**kwargs).get_dict()
    rs = data["resultSets"][0]
    idx = {h: i for i, h in enumerate(rs["headers"])}
    missing = [c for c in COLUMNS if c not in idx]
    if missing:
        raise RuntimeError(f"drafthistory response is missing columns: {missing}")
    return [tuple(r[idx[c]] for c in COLUMNS) for r in rs["rowSet"]]


def _store(conn, rows) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT OR REPLACE INTO draft_history (
            person_id, player_name, season, round_number, round_pick,
            overall_pick, team_id, team_city, team_name, team_abbreviation,
            organization, organization_type, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [r + (now,) for r in rows],
    )
    conn.commit()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Re-fetch the entire draft history.")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(SCHEMA)
        before = conn.execute("SELECT COUNT(*) FROM draft_history").fetchone()[0]
        newest_before = conn.execute("SELECT MAX(season) FROM draft_history").fetchone()[0]

        if args.full or before == 0:
            logger.info("Fetching the complete draft history%s...",
                        " (table is empty)" if before == 0 else " (--full)")
            rows = _rows(None)
            if len(rows) < 5000:
                raise RuntimeError(
                    f"Full history returned only {len(rows)} picks, expected 8,000+. "
                    "Refusing to overwrite the table."
                )
            _store(conn, rows)
        else:
            # A draft happens in June, so the "current" draft year is this
            # calendar year once we are past the spring.
            year = datetime.now(timezone.utc).year
            years = [str(year - i) for i in range(RECENT_YEARS)]
            total = 0
            for y in years:
                try:
                    rows = _rows(y)
                except Exception as exc:
                    logger.warning("  %s: fetch failed - %s", y, exc)
                    continue
                if not rows:
                    logger.info("  %s: no picks published yet", y)
                    continue
                total += _store(conn, rows)
                logger.info("  %s: %d picks stored", y, len(rows))
            if total == 0:
                logger.info("No recent draft classes returned anything to store.")

        after = conn.execute("SELECT COUNT(*) FROM draft_history").fetchone()[0]
        newest = conn.execute("SELECT MAX(season) FROM draft_history").fetchone()[0]
        logger.info(
            "draft_history: %d picks (was %d), newest class %s (was %s).",
            after, before, newest, newest_before,
        )
        return 0
    except Exception as exc:
        logger.error("Draft ingest failed: %s", exc, exc_info=True)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
