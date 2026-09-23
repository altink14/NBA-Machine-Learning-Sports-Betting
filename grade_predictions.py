"""
grade_predictions.py
====================
Fills in actual results for logged model predictions (predictions_log in
Data/OddsData.sqlite) using final scores from the stats database
(Data/TeamData.sqlite, kept fresh by daily_update.py).

A prediction is graded once the game's box score is in team_game_advanced:
  actual_winner = team with more points, actual_total = combined score.

Run standalone or via daily_update.py. Idempotent — only ungraded rows
with a game date in the past are touched.
"""

import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("grade_predictions")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
TEAM_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")


def _team_name_to_id(team_conn: sqlite3.Connection) -> dict:
    rows = team_conn.execute("SELECT team_id, full_name FROM team_metadata").fetchall()
    mapping = {}
    for team_id, full_name in rows:
        mapping[full_name.strip().lower()] = team_id
    # sbrscrape uses "LA Clippers" for the Clippers
    la = mapping.get("los angeles clippers")
    if la:
        mapping["la clippers"] = la
    return mapping


def _et_date(value) -> str:
    """US Eastern calendar date of a tip-off, as 'YYYY-MM-DD'.

    `team_game_advanced.game_date` is the NBA's own (Eastern) game date. A
    tip-off is stored in UTC, and anything after 20:00 ET is already the next
    day in UTC, so comparing the raw UTC date against the Eastern game date is
    off by one for every late game.
    """
    s = str(value or "")
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return s[:10]
        return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return s[:10]


def _find_result(team_conn, home_id: int, away_id: int, around_date: str):
    """
    Find the home team's game against the away team, preferring the exact
    Eastern date and allowing +/-1 day only as a fallback.

    WHY NOT `ORDER BY game_date ASC`. That took the EARLIEST game in a
    three-day window, so if the same home team hosted the same opponent on
    consecutive days, a prediction could be graded against the previous
    night's result. And a wrong grade here is permanent: the
    `predictions_log_no_regrade` trigger forbids correcting it. Nearest date
    wins, so the exact day always beats a neighbour.

    Returns (home_pts, away_pts) or None.
    """
    row = team_conn.execute(
        """
        SELECT pts, opp_pts, game_date FROM team_game_advanced
        WHERE team_id = ? AND opp_team_id = ?
          AND game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
        ORDER BY abs(julianday(game_date) - julianday(?)) ASC, game_date ASC
        LIMIT 1
        """,
        (home_id, away_id, around_date, around_date, around_date),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1]


#: Columns added 2026-09-19 for closing line value. Additive only.
_CLV_COLUMNS = [
    ("closing_home_ml", "REAL"),
    ("closing_away_ml", "REAL"),
    ("closing_captured_at", "TEXT"),
    ("closing_minutes_before_tip", "REAL"),
    # Whether the close we settled against is a price we watched or one
    # rebuilt from a vendor archive. Pooling the two and publishing the
    # average is the mistake this column exists to make impossible.
    ("closing_provenance", "TEXT"),
    ("clv", "REAL"),
]


def _ensure_clv_columns(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions_log)")}
    for col, decl in _CLV_COLUMNS:
        if have and col not in have:
            conn.execute(f"ALTER TABLE predictions_log ADD COLUMN {col} {decl}")
            logger.info("predictions_log: added column %s", col)


def _american_to_decimal(price):
    if price is None:
        return None
    p = float(price)
    if -1.0 < p < 1.0:
        return None
    return 1.0 + (p / 100.0 if p > 0 else 100.0 / -p)


def price_clv(odds_db: str = None) -> dict:
    """Attach closing line value to logged predictions, from our own snapshots.

    WHAT THIS ASKS, AND WHY IT IS NOT ROI. Did we take a better price than the
    market's last one before tip? Return on investment needs hundreds of
    settled bets to say anything; closing line value says something after a few
    dozen, and it does not depend on who won. The NFL ledger has had this since
    it was built; the NBA, which is the actual product, has not.

        clv = (decimal price we logged / decimal price at the close) - 1

    so +0.02 means the price we had was 2% better on the same side.

    HOW CLOSE IS "CLOSING". This is the honest part and the NBA-specific
    problem. The recorder is schedule-aware as of today, but the archive still
    contains snapshots taken hours before tip from when it was not, and a price
    from eight hours out is not a closing price. So every row records
    `closing_minutes_before_tip` next to the number. Nothing here decides what
    is close enough -- that is a judgement for whoever reports it -- but the
    distance travels with the figure instead of being lost, and a CLV computed
    against a snapshot from the morning must be described that way.

    Only predictions whose game has actually started are priced: before tip
    there is no close, whatever the newest snapshot says.

    WHICH PRICE, AND WHOSE. The close is the same book's last price before
    tip -- never a better one from a book we did not log -- and each row
    records `closing_provenance` so a CLV settled against a rebuilt price is
    never pooled with one settled against a price we watched. Report the two
    apart or not at all.
    """
    odds_db = odds_db or ODDS_DB
    if not os.path.exists(odds_db):
        return {"priced": 0, "no_close": 0, "not_started": 0}
    conn = sqlite3.connect(odds_db)
    conn.row_factory = sqlite3.Row
    counts = {"priced": 0, "no_close": 0, "not_started": 0, "no_price": 0}
    try:
        _ensure_clv_columns(conn)
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            "SELECT id, sportsbook, game_key, game_start_time_utc, home_team, "
            "predicted_winner, home_ml, away_ml FROM predictions_log WHERE clv IS NULL"
        ).fetchall()
        for r in rows:
            tip = r["game_start_time_utc"]
            if not tip or tip > now:
                counts["not_started"] += 1
                continue
            # Latest price before tip, and where two share a timestamp the
            # one we watched wins. Ordering by captured_at alone was fine
            # while the live recorder was the only writer; a repair inserts
            # reconstructed rows today carrying last week's captured_at, and
            # an arbitrary tiebreak would let a rebuilt price outrank a
            # watched one. Same rule as odds_recorder.seal().
            close = conn.execute(
                "SELECT captured_at, home_ml, away_ml, "
                "COALESCE(provenance, 'observed') AS provenance FROM odds_snapshots "
                # Both sides go through one Clippers spelling. The prediction
                # path (SbrOddsProvider) rewrites the team to "LA Clippers" and
                # the closing-line recorder stores The Odds API's own name, so an
                # exact game_key match could never find a Clippers close and every
                # Clippers game would quietly count as no_close all season. The
                # REPLACE is a no-op for the other 29 teams.
                "WHERE REPLACE(game_key, 'Los Angeles Clippers', 'LA Clippers') = "
                "      REPLACE(?, 'Los Angeles Clippers', 'LA Clippers') "
                "AND sportsbook = ? AND captured_at < ? "
                "AND home_ml IS NOT NULL AND away_ml IS NOT NULL "
                "ORDER BY captured_at DESC, "
                "         CASE COALESCE(provenance, 'observed') "
                "              WHEN 'observed' THEN 0 ELSE 1 END, "
                "         id DESC "
                "LIMIT 1",
                (r["game_key"], r["sportsbook"], tip),
            ).fetchone()
            if not close:
                counts["no_close"] += 1
                continue
            on_home = (r["predicted_winner"] or "") == (r["home_team"] or "")
            ours = r["home_ml"] if on_home else r["away_ml"]
            theirs = close["home_ml"] if on_home else close["away_ml"]
            d_ours, d_theirs = _american_to_decimal(ours), _american_to_decimal(theirs)
            if d_ours is None or d_theirs is None or d_theirs <= 1.0:
                counts["no_price"] += 1
                continue
            try:
                gap = (datetime.fromisoformat(tip.replace("Z", "+00:00"))
                       - datetime.fromisoformat(str(close["captured_at"]).replace("Z", "+00:00")
                                                )).total_seconds() / 60.0
            except ValueError:
                gap = None
            conn.execute(
                "UPDATE predictions_log SET closing_home_ml=?, closing_away_ml=?, "
                "closing_captured_at=?, closing_minutes_before_tip=?, "
                "closing_provenance=?, clv=? WHERE id=?",
                (close["home_ml"], close["away_ml"], close["captured_at"], gap,
                 close["provenance"], (d_ours / d_theirs) - 1.0, r["id"]),
            )
            counts["priced"] += 1
        conn.commit()
    finally:
        conn.close()
    logger.info("closing line value: %s", counts)
    return counts


def grade(odds_db: str = None, team_db: str = None) -> int:
    """Grade ungraded predictions. Defaults to the real databases.

    The two paths are arguments rather than constants so rehearse_ledger.py can
    run this exact function against a scratch ledger and the real archive. The
    thing that has to work on opening night should be the thing that gets
    rehearsed, not a copy of it that can drift.
    """
    odds_db = odds_db or ODDS_DB
    team_db = team_db or TEAM_DB
    if not os.path.exists(odds_db) or not os.path.exists(team_db):
        logger.warning("Databases not found; nothing to grade.")
        return 0

    odds_conn = sqlite3.connect(odds_db)
    odds_conn.row_factory = sqlite3.Row
    team_conn = sqlite3.connect(f"file:{team_db}?mode=ro", uri=True)
    graded = 0
    try:
        try:
            ungraded = odds_conn.execute(
                """
                SELECT id, log_date, home_team, away_team, game_start_time_utc
                FROM predictions_log
                WHERE actual_winner IS NULL AND log_date < ?
                """,
                (datetime.utcnow().strftime("%Y-%m-%d"),),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.info("predictions_log table does not exist yet; nothing to grade.")
            return 0

        if not ungraded:
            logger.info("No ungraded predictions.")
            return 0

        name_to_id = _team_name_to_id(team_conn)
        unmatched_names, no_box_score = [], 0
        for row in ungraded:
            home_id = name_to_id.get(row["home_team"].strip().lower())
            away_id = name_to_id.get(row["away_team"].strip().lower())
            if not home_id or not away_id:
                # Usually a non-NBA row (WNBA fallback). But a team name the
                # mapper misses looks identical, and that game would sit
                # ungraded forever -- quietly leaving the denominator of the
                # published record. So count them and say which.
                unmatched_names.append(f"{row['away_team']} @ {row['home_team']}")
                continue

            # The EASTERN date of the tip-off, not the UTC one: a 10:30pm ET
            # game is the next day in UTC and would otherwise be looked up a
            # day late.
            game_date = (_et_date(row["game_start_time_utc"])
                         if row["game_start_time_utc"] else row["log_date"][:10])
            result = _find_result(team_conn, home_id, away_id, game_date)
            if result is None:
                no_box_score += 1
                continue  # box score not ingested yet

            home_pts, away_pts = result
            winner = row["home_team"] if home_pts > away_pts else row["away_team"]
            odds_conn.execute(
                "UPDATE predictions_log SET actual_winner = ?, actual_total = ? WHERE id = ?",
                (winner, home_pts + away_pts, row["id"]),
            )
            graded += 1

        odds_conn.commit()
        logger.info("Graded %d of %d ungraded predictions (%d still waiting on a box "
                    "score, %d with a team name the archive does not know).",
                    graded, len(ungraded), no_box_score, len(unmatched_names))
        if unmatched_names:
            logger.warning("Ungradeable team names (fine if these are WNBA; a real NBA "
                           "team here will never be graded): %s",
                           ", ".join(sorted(set(unmatched_names))[:10]))
        return graded
    finally:
        odds_conn.close()
        team_conn.close()


if __name__ == "__main__":
    grade()
    sys.exit(0)
