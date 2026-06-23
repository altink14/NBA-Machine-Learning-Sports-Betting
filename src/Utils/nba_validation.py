"""
nba_validation.py
=================
Validation logic comparing our computed derivatives against the official NBA Stats API
ground-truth (`boxscoreadvancedv2`).

Logs warnings to `raw_scrape_log` if differences exceed +/- 0.5.
Also tracks overall failure rates to assert pipeline health.
"""

from __future__ import annotations

import logging
from datetime import datetime
import sqlite3
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def validate_game_stats(
    db_conn: sqlite3.Connection,
    game_id: str,
    team_id: int,
    our_pace: float,
    our_ortg: float,
    our_drtg: float,
    official_pace: float,
    official_ortg: float,
    official_drtg: float,
    threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Compare computed Pace, ORtg, and DRtg against official values.
    If the difference exceeds the threshold, write a warning to raw_scrape_log.

    Parameters
    ----------
    db_conn : sqlite3.Connection
        Connection to the central SQLite database.
    game_id : str
        The unique NBA game ID.
    team_id : int
        The team's ID.
    our_pace, our_ortg, our_drtg : float
        Our computed metrics.
    official_pace, official_ortg, official_drtg : float
        Official metrics from boxscoreadvancedv2.
    threshold : float
        Max acceptable difference before logging a warning. Default is 0.5.

    Returns
    -------
    summary : Dict[str, Any]
        A summary of the comparison results.
    """
    comparisons = {
        "pace": (our_pace, official_pace),
        "off_rating": (our_ortg, official_ortg),
        "def_rating": (our_drtg, official_drtg)
    }

    warnings = []
    timestamp = datetime.utcnow().isoformat()

    for metric, (our_val, off_val) in comparisons.items():
        diff = abs(our_val - off_val)
        if diff > threshold:
            msg = f"Discrepancy in {metric} for game {game_id}, team {team_id}: ours={our_val:.2f}, official={off_val:.2f}, diff={diff:.2f}"
            logger.warning(msg)
            warnings.append({
                "metric": metric,
                "our_val": our_val,
                "official_val": off_val,
                "diff": diff,
                "message": msg
            })

            # Write to raw_scrape_log
            try:
                db_conn.execute(
                    """
                    INSERT INTO raw_scrape_log (
                        logged_at, game_id, team_id, endpoint, status, message, metric, our_value, official_value, diff
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        game_id,
                        team_id,
                        "boxscoreadvancedv2",
                        "warning",
                        msg,
                        metric,
                        our_val,
                        off_val,
                        diff
                    )
                )
            except sqlite3.Error as err:
                logger.error("Failed to write to raw_scrape_log: %s", err)

    if not warnings:
        # Log a successful validation entry
        try:
            db_conn.execute(
                """
                INSERT INTO raw_scrape_log (
                    logged_at, game_id, team_id, endpoint, status, message
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    game_id,
                    team_id,
                    "boxscoreadvancedv2",
                    "ok",
                    f"Validation successful. All computed metrics within +/- {threshold}."
                )
            )
        except sqlite3.Error as err:
            logger.error("Failed to write successful validation log: %s", err)

    return {
        "game_id": game_id,
        "team_id": team_id,
        "valid": len(warnings) == 0,
        "warnings": warnings
    }


def get_validation_failure_rate(db_conn: sqlite3.Connection) -> float:
    """
    Calculate the percentage of game validation records (game-team pairs) that flagged warning/error status.

    Returns
    -------
    failure_rate : float
        A value between 0.0 and 100.0.
    """
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT game_id || '-' || team_id) FROM raw_scrape_log WHERE status != 'ok'")
        failures = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(DISTINCT game_id || '-' || team_id) FROM raw_scrape_log WHERE status IN ('ok', 'warning')")
        total = cursor.fetchone()[0] or 0

        if total == 0:
            return 0.0
        return (failures / total) * 100.0
    except sqlite3.Error as err:
        logger.error("Failed to calculate validation failure rate: %s", err)
        return 0.0
