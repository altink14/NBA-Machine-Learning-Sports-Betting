"""
grade_predictions.py
====================
Fills in actual results for logged model predictions (predictions_log in
Data/OddsData.sqlite) using final scores from the stats database
(Data/TeamData.sqlite, kept fresh by daily_update.py).

A prediction is graded once the game's box score is in team_game_advanced:
  actual_winner = team with more points, actual_total = combined score.
When nba.com's box score has not landed (it blocked this PC in September
2026), ESPN's final score from espn_box_scores grades it instead; the row
records which (result_source) and is checked against nba.com's box score
when that arrives (confirm_espn_grades).

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
    found = _find_nba_result(team_conn, home_id, away_id, around_date)
    return None if found is None else (found[0], found[1])


#: Where each grade's final score came from (added 2026-09-29). NULL on a row
#: graded before these columns existed, which was always nba.com.
GRADE_COLUMNS = [
    ("result_source", "TEXT"),          # 'nba.com' | 'espn'
    ("result_source_ref", "TEXT"),      # nba.com game id or ESPN event id
    ("result_confirmed_at", "TEXT"),    # UTC ISO: nba.com's later box score agreed with an ESPN grade
    ("result_conflict", "TEXT"),        # nba.com's later box score DISAGREED; says how
]


def ensure_grade_columns(conn: sqlite3.Connection) -> None:
    """Add the grade-source columns. Only ever ADDED, which ledger_sync carries
    to the public copy by itself; none is a pick column, so the immutability
    trigger and the pick commitments (ledger_commit.CANONICAL_FIELDS) never
    see them."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions_log)")}
    if not have:
        return
    for col, decl in GRADE_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE predictions_log ADD COLUMN {col} {decl}")
            logger.info("predictions_log: added column %s", col)


def _find_nba_result(team_conn, home_id: int, away_id: int, around_date: str):
    """(home_pts, away_pts, game_id) from nba.com's box scores, or None."""
    row = team_conn.execute(
        """
        SELECT pts, opp_pts, game_id FROM team_game_advanced
        WHERE team_id = ? AND opp_team_id = ?
          AND game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
        ORDER BY abs(julianday(game_date) - julianday(?)) ASC, game_date ASC
        LIMIT 1
        """,
        (home_id, away_id, around_date, around_date, around_date),
    ).fetchone()
    return None if row is None else (row[0], row[1], row[2])


def _find_espn_result(team_conn, home_id: int, away_id: int, around_date: str):
    """(home_pts, away_pts, espn_event_id) from ESPN's final score, or None.

    WHY (2026-09-29). nba.com blocked this PC after a large backfill, and the
    outbound guard keeps it paused. The picks already survive that
    (espn_boxscore.py feeds the model), but grading read nba.com's box scores
    only, so every pick would have sat ungraded for as long as the block
    lasted. ESPN's final scores matched nba.com's in all 292 team-games
    measured on 2026-09-28, and espn_box_scores holds only games ESPN marked
    STATUS_FINAL.

    Every ESPN row counts here, including ones held out of the MODEL's inputs
    (usable = 0: e.g. the NBA Cup final, which nba.com's team stats leave
    out): whether a game counts toward the standings has nothing to do with
    who won it. Home and away are matched either way round, because a
    neutral-site game (Mexico City, Paris) may be filed with the other team at
    home, and the points are then read by team, not by side.
    """
    try:
        rows = team_conn.execute(
            """
            SELECT espn_event_id, home_team_id, home_pts, away_pts FROM espn_box_scores
            WHERE ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
              AND game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
            ORDER BY abs(julianday(game_date) - julianday(?)) ASC, game_date ASC
            LIMIT 1
            """,
            (home_id, away_id, away_id, home_id, around_date, around_date, around_date),
        ).fetchall()
    except sqlite3.OperationalError:
        return None  # no ESPN tables: the fallback has never had to run
    if not rows:
        return None
    event_id, espn_home, h_pts, a_pts = rows[0]
    if int(espn_home) == int(home_id):
        return h_pts, a_pts, str(event_id)
    return a_pts, h_pts, str(event_id)


def confirm_espn_grades(odds_conn, team_conn, name_to_id: dict, now=None) -> list:
    """Check each ESPN grade against nba.com's box score once it lands.

    A grade can never be changed (predictions_log_no_regrade), so an ESPN
    grade is final the moment it is written. This is the check on it: when
    nba.com's box score for that game arrives, the two results are compared.
    Agreement stamps result_confirmed_at. Disagreement (a different winner or
    total) is written to result_conflict, stated on the track record, and
    returned so the daily job turns red: a person has to look, because the
    ledger's own rules forbid a quiet fix. A conflict is reported on the
    morning it is found and kept on the row after that; it is not re-raised
    every day, which would teach everyone to ignore red.

    Returns the conflicts found this run, as text.
    """
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    try:
        rows = odds_conn.execute(
            "SELECT id, home_team, away_team, game_start_time_utc, log_date, actual_winner, "
            "actual_total FROM predictions_log WHERE result_source = 'espn' "
            "AND result_confirmed_at IS NULL AND result_conflict IS NULL").fetchall()
    except sqlite3.OperationalError:
        return []
    conflicts = []
    for row in rows:
        home_id = name_to_id.get((row["home_team"] or "").strip().lower())
        away_id = name_to_id.get((row["away_team"] or "").strip().lower())
        if not home_id or not away_id:
            continue
        game_date = (_et_date(row["game_start_time_utc"])
                     if row["game_start_time_utc"] else row["log_date"][:10])
        nba = _find_nba_result(team_conn, home_id, away_id, game_date)
        if nba is None:
            continue  # nba.com still has not supplied it
        h, a, game_id = nba
        if h is None or a is None or h <= 0 or a <= 0 or h == a:
            continue  # a broken nba.com box score confirms nothing; grade() reports those
        winner = row["home_team"] if h > a else row["away_team"]
        if winner == row["actual_winner"] and (h + a) == row["actual_total"]:
            odds_conn.execute("UPDATE predictions_log SET result_confirmed_at = ? WHERE id = ?",
                              (stamp, row["id"]))
            continue
        text = (f"nba.com {game_id}: {row['away_team']} {a}, {row['home_team']} {h}; graded from "
                f"ESPN as {row['actual_winner']} with total {row['actual_total']:g}")
        odds_conn.execute("UPDATE predictions_log SET result_conflict = ? WHERE id = ?",
                          (text[:500], row["id"]))
        conflicts.append(text)
    return conflicts


def _conflict_message(conflicts: list) -> str:
    return (f"{len(conflicts)} ESPN grade(s) DISAGREE with nba.com's box score. The ledger "
            f"cannot regrade them; each row now carries result_conflict and the track record "
            f"says so. A person must look: " + "; ".join(conflicts[:10]))


def _played_on_date(team_conn, name_to_id: dict, pick) -> "bool | None":
    """Is this game's box score on the logged tip's US Eastern date?

    The grader accepts +/-1 day so a game moved a day still gets its result;
    closing line value must not, because the market we logged against was for
    the original date. None when the teams are unknown to the archive.
    """
    home_id = name_to_id.get((pick["home_team"] or "").strip().lower())
    away_id = name_to_id.get((pick["away_team"] or "").strip().lower())
    if not home_id or not away_id:
        return None
    day = _et_date(pick["game_start_time_utc"])
    row = team_conn.execute(
        "SELECT 1 FROM team_game_advanced WHERE team_id = ? AND opp_team_id = ? "
        "AND game_date = ? LIMIT 1",
        (home_id, away_id, day),
    ).fetchone()
    if row is not None:
        return True
    # A pick graded from ESPN's final score (nba.com's box score not landed)
    # was played on its date just the same. Without this, every such pick
    # would be settled for good as 'tip_moved' and never get its CLV.
    try:
        row = team_conn.execute(
            "SELECT 1 FROM espn_box_scores WHERE game_date = ? AND "
            "((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?)) "
            "LIMIT 1",
            (day, home_id, away_id, away_id, home_id),
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    return row is not None


def price_clv(odds_db: str = None, team_db: str = None, now=None) -> dict:
    """Attach closing line value to graded predictions, from our own snapshots.

    WHAT THIS ASKS, AND WHY IT IS NOT ROI. Did we take a better price than the
    market's last one before tip? Return on investment needs hundreds of
    settled bets to say anything; closing line value says something after a few
    dozen, and it does not depend on who won. It is evidence, not profit.

    The rules -- what "the close" is, per book and for the consensus, the
    30-minute limit, the fallbacks, when a pick settles and why it settles
    once -- live in src/Utils/nba_clv.py (method bb-nba-clv-v1), with the four
    numbers it stores. Until 2026-09-28 this function priced against the same
    book's latest pre-tip row however old it was (the distance was recorded,
    not enforced), had no no-vig or consensus figure, and re-priced any row
    whose clv was still NULL every morning. The table was empty throughout, so
    nothing it wrote needs revisiting.

    Only the CLV columns are written; the pick's own columns are frozen by
    the predictions_log triggers and are never in the UPDATE.
    """
    from src.Utils import nba_clv

    odds_db = odds_db or ODDS_DB
    team_db = team_db or TEAM_DB
    if not os.path.exists(odds_db):
        return {}
    conn = sqlite3.connect(odds_db)
    conn.row_factory = sqlite3.Row
    team_conn = (sqlite3.connect(f"file:{team_db}?mode=ro", uri=True)
                 if os.path.exists(team_db) else None)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                            "AND name='predictions_log'").fetchone():
            return {}
        if team_conn is None:
            logger.warning("TeamData not found: no pick can be checked as played on its "
                           "date, so none is priced today.")
            return nba_clv.price_all(conn, lambda _p: None, now=now)
        name_to_id = _team_name_to_id(team_conn)
        return nba_clv.price_all(
            conn, lambda p: _played_on_date(team_conn, name_to_id, p), now=now)
    finally:
        conn.close()
        if team_conn is not None:
            team_conn.close()


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

        ensure_grade_columns(odds_conn)
        name_to_id = _team_name_to_id(team_conn)
        # Before grading anything new: check yesterday's ESPN grades against
        # any nba.com box score that has landed since.
        conflicts = confirm_espn_grades(odds_conn, team_conn, name_to_id)
        odds_conn.commit()

        if not ungraded:
            logger.info("No ungraded predictions.")
            if conflicts:
                raise RuntimeError(_conflict_message(conflicts))
            return 0

        unmatched_names, no_box_score, implausible = [], 0, []
        from_espn = []
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
            # nba.com's box score first, always. ESPN's final score only when
            # nba.com's has not landed (the fallback ran because it failed).
            source = "nba.com"
            result = _find_nba_result(team_conn, home_id, away_id, game_date)
            if result is None:
                result = _find_espn_result(team_conn, home_id, away_id, game_date)
                source = "espn"
            if result is None:
                no_box_score += 1
                continue  # box score not ingested yet

            home_pts, away_pts, source_ref = result
            # A tie, a zero or a missing score is a broken box score, not a
            # result. `home_pts > away_pts` used to turn 0-0 (what the parser
            # stores for a skeleton box score) into an AWAY win, and the
            # no-regrade trigger makes that permanent. Refuse it and say so.
            if (home_pts is None or away_pts is None or home_pts <= 0 or away_pts <= 0
                    or home_pts == away_pts):
                implausible.append(f"{row['away_team']} @ {row['home_team']} "
                                   f"{game_date} ({source}): {away_pts}-{home_pts}")
                continue
            winner = row["home_team"] if home_pts > away_pts else row["away_team"]
            odds_conn.execute(
                "UPDATE predictions_log SET actual_winner = ?, actual_total = ?, "
                "result_source = ?, result_source_ref = ? WHERE id = ?",
                (winner, home_pts + away_pts, source, source_ref, row["id"]),
            )
            graded += 1
            if source == "espn":
                from_espn.append(f"{row['away_team']} @ {row['home_team']} {game_date}")

        odds_conn.commit()
        logger.info("Graded %d of %d ungraded predictions (%d from ESPN's final score, %d still "
                    "waiting on a box score, %d with a team name the archive does not know, %d "
                    "refused because the archived score is not a result).",
                    graded, len(ungraded), len(from_espn), no_box_score, len(unmatched_names),
                    len(implausible))
        if from_espn:
            logger.warning("Graded from ESPN (nba.com's box score has not landed; it is checked "
                           "against these when it does): %s", ", ".join(from_espn[:10]))
        if unmatched_names:
            logger.warning("Ungradeable team names (fine if these are WNBA; a real NBA "
                           "team here will never be graded): %s",
                           ", ".join(sorted(set(unmatched_names))[:10]))
        errors = []
        if implausible:
            # Raised after the commit, so every other game still grades; the
            # daily job turns red, because this needs a person to fix the box
            # score before the pick can be graded at all.
            errors.append(
                f"{len(implausible)} prediction(s) NOT graded: the archive's score is not "
                f"a result (tie, zero or missing): " + "; ".join(implausible[:10]))
        if conflicts:
            errors.append(_conflict_message(conflicts))
        if errors:
            raise RuntimeError(" | ".join(errors))
        return graded
    finally:
        odds_conn.close()
        team_conn.close()


if __name__ == "__main__":
    grade()
    sys.exit(0)
