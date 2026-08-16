"""
refresh_team_stats.py
=====================
Writes today's season-to-date team-stats snapshot into TeamData.sqlite, as a
table named for today's date (`YYYY-MM-DD`) — the format PredictionRunner
._load_team_stats() reads, which always picks the newest such table.

Why this exists: those snapshot tables were produced by the original project's
src/Process-Data/Get_Data.py, a one-shot backfill script that nobody runs any
more. The newest table in the database is 2024-04-29, so live predictions were
being made from team stats frozen at the end of the 2023-24 season. The modern
pipeline (backfill.py) writes box scores and derived ratings, but never these,
because the model consumes the raw NBA.com league-dashboard shape.

Two conventions inherited from Get_Data.py, both load-bearing:

1. The table is named for the day the stats are used, and holds games through
   the *previous* day (`DateTo` = yesterday). A game can therefore never
   contribute to the stats used to predict it.
2. The column set and order must match the tables the models were trained on,
   exactly. The schema contract is read from the newest existing snapshot
   rather than hardcoded here, and a mismatch aborts without writing — a
   silently reordered frame would produce confident nonsense, since XGBoost
   sees positions, not names.

Run standalone, or via daily_update.py which calls refresh() before predicting.
"""

import logging
import os
import sqlite3
import sys
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.Utils.nba_stats_client import get_client  # noqa: E402

logger = logging.getLogger("refresh_team_stats")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")

# Columns that exist in the snapshot but do not come from the API response.
_INDEX_COL = "index"
_DATE_COL = "Date"


def current_season(today: date) -> str:
    """NBA seasons run October-June, labeled by span (e.g. 2025-26)."""
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def newest_snapshot(conn: sqlite3.Connection) -> Optional[str]:
    """Name of the most recent `YYYY-MM-DD` team-stats table, or None."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' "
        "ORDER BY name DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def reference_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """The column contract, taken from an existing snapshot."""
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def refresh(as_of: Optional[date] = None, season: Optional[str] = None) -> Optional[str]:
    """Fetch and store today's snapshot. Returns the table name written, or None.

    None means "nothing to write" (no games played yet this season) rather than
    failure; failures raise.
    """
    as_of = as_of or date.today()
    season = season or current_season(as_of)
    table_name = as_of.strftime("%Y-%m-%d")
    date_to = as_of - timedelta(days=1)
    season_start_year = int(season[:4])

    conn = sqlite3.connect(DB_PATH)
    try:
        reference = newest_snapshot(conn)
        if reference is None:
            raise RuntimeError(
                "No existing team-stats snapshot to take the column contract from; "
                "refusing to invent a schema the models were not trained on."
            )
        expected = reference_columns(conn, reference)

        client = get_client()
        rows = client.league_dash_team_stats(
            season=season,
            season_type="Regular Season",
            per_mode="PerGame",
            measure_type="Base",
            date_from=f"10/01/{season_start_year}",
            date_to=date_to.strftime("%m/%d/%Y"),
        )
        if not rows:
            logger.info(
                "No team stats returned for %s through %s - no games played yet. "
                "Nothing written.", season, date_to
            )
            return None

        df = pd.DataFrame(rows)
        if df.empty or ("GP" in df.columns and df["GP"].fillna(0).sum() == 0):
            logger.info("Season %s has no completed games through %s. Nothing written.",
                        season, date_to)
            return None

        df[_DATE_COL] = table_name

        api_cols = set(df.columns)
        needed = [c for c in expected if c != _INDEX_COL]
        missing = [c for c in needed if c not in api_cols]
        if missing:
            raise RuntimeError(
                f"stats.nba.com response is missing {len(missing)} column(s) the models "
                f"expect: {missing}. Refusing to write a snapshot that would shift every "
                f"feature position. Compare against table '{reference}'."
            )
        extra = sorted(api_cols - set(needed))
        if extra:
            logger.warning(
                "Dropping %d column(s) the API added since the models were trained: %s",
                len(extra), extra
            )

        # Reindex to the exact trained order, then write with the integer index
        # materialised as the `index` column the loader reads back.
        df = df[needed]
        df.to_sql(table_name, conn, if_exists="replace", index=True, index_label=_INDEX_COL)
        conn.commit()
        logger.info(
            "Wrote team-stats snapshot '%s': %d teams, %s season through %s.",
            table_name, len(df), season, date_to
        )
        return table_name
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        written = refresh()
    except Exception as exc:
        logger.error("Team-stats refresh failed: %s", exc, exc_info=True)
        return 1
    return 0 if written else 0


if __name__ == "__main__":
    sys.exit(main())
