"""
rehearse_ledger.py
==================
Proves the NBA prediction ledger works, today, without waiting for a game.

WHY THIS EXISTS. `predictions_log` has never logged a row in production. The
product's central claim -- "every pick is written down before tip-off and
graded in public, losses included" -- rests entirely on a path that has never
run on a real game day. The plan for that was "watch opening night and fix it
on the spot", on 20 October, which is the worst possible night to find a bug.

So this rehearses the path instead. It uses the REAL writer
(`main_api.log_predictions`), the REAL grader (`grade_predictions.grade`), the
REAL schema with its constraints and triggers, and REAL finished games with
REAL final scores out of the archive. The only thing it substitutes is the
slate, because the odds feed has no NBA games in the offseason.

WHAT IT CANNOT PROVE. It does not run `PredictionRunner`, so it says nothing
about whether the model loads, whether the odds provider returns NBA rather
than falling back to another league, or whether the feed supplies a tip-off
time on the night. Those need a live slate. What it does prove is that
everything downstream of "here is a list of predictions" works end to end:
the constraints hold, the rows land, grading matches the box score, and the
record computes.

IT NEVER TOUCHES PRODUCTION. The ledger it writes to is a scratch file in a
temp directory, created fresh each run. The archive is opened read-only. If
anything here could write to `Data/OddsData.sqlite` the whole point would be
lost, so it refuses to run if the scratch path resolves anywhere near it.

WHY THE PICKS ARE FIXED, NOT MODELLED. Every rehearsal pick is "home team",
chosen because it is deterministic and obviously not a forecast. The rehearsal
reports how many rows graded, never a win rate. A percentage computed from
always-picking-home on a handful of games is not a model result, and putting
one in this output is how a meaningless number ends up quoted as a real one.

Usage:
    venv/Scripts/python.exe rehearse_ledger.py
    venv/Scripts/python.exe rehearse_ledger.py --games 12 --season 2025-26
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rehearse_ledger")

TEAM_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
REAL_ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(label)
    logger.info("  %s  %s%s", "PASS" if ok else "FAIL", label, f" - {detail}" if detail else "")


def real_finished_games(team_db: str, season: str, limit: int):
    """Real games, with their real final scores, from our own archive.

    Home side only (`pts` is the row team's score), and only games whose
    opponent row also exists, so the grader can find them the way it will on a
    real morning.

    One game per matchup. The end of a season is a playoff series -- the last
    eight games of 2025-26 are five Knicks-Spurs Finals games -- and a slate of
    the same two teams five times is both unlike a real night and a trap: any
    lookup keyed on the pairing collapses them. A real slate is many different
    matchups, so the fixture is too.
    """
    conn = sqlite3.connect(f"file:{team_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT t.game_id, t.game_date, t.team_id AS home_id, t.opp_team_id AS away_id,
                   t.pts AS home_pts, t.opp_pts AS away_pts,
                   h.full_name AS home_name, a.full_name AS away_name
            FROM team_game_advanced t
            JOIN team_metadata h ON h.team_id = t.team_id
            JOIN team_metadata a ON a.team_id = t.opp_team_id
            WHERE t.season = ? AND t.pts IS NOT NULL AND t.opp_pts IS NOT NULL
              AND t.pts > t.opp_pts     -- home wins, so the expected grade is knowable
            ORDER BY t.game_date DESC
            """,
            (season,),
        ).fetchall()
        seen, out = set(), []
        for r in rows:
            pair = (r["home_name"], r["away_name"])
            if pair in seen:
                continue
            seen.add(pair)
            out.append(dict(r))
            if len(out) >= limit:
                break
        return out
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Rehearse the NBA prediction ledger end to end.")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--season", default="2025-26")
    args = ap.parse_args()

    scratch_dir = tempfile.mkdtemp(prefix="ledger_rehearsal_")
    scratch_db = os.path.join(scratch_dir, "rehearsal_odds.sqlite")
    if os.path.abspath(scratch_db) == os.path.abspath(REAL_ODDS_DB):
        raise SystemExit("refusing to rehearse against the real ledger")

    logger.info("=== ledger rehearsal ===")
    logger.info("scratch ledger: %s", scratch_db)
    logger.info("archive (read-only): %s", TEAM_DB)

    games = real_finished_games(TEAM_DB, args.season, args.games)
    if not games:
        logger.error("No finished games found for season %s. Nothing to rehearse against.",
                     args.season)
        return 1
    logger.info("rehearsing against %d real finished game(s) from %s", len(games), args.season)

    import main_api
    import grade_predictions

    def scratch_conn():
        c = sqlite3.connect(scratch_db)
        c.row_factory = sqlite3.Row
        return c

    main_api._odds_snapshot_conn = scratch_conn

    # --- build a slate shaped exactly like PredictionRunner's output ---------
    # Tip-off is the real game date at 19:00Z; the prediction is written two
    # hours earlier, which is the relationship the CHECK constraint enforces.
    slate = []
    for g in games:
        tip = datetime.fromisoformat(f"{g['game_date'][:10]}T19:00:00+00:00")
        slate.append({
            "home_team": g["home_name"], "away_team": g["away_name"],
            "game_start_time_utc": tip.isoformat(),
            "predicted_winner": g["home_name"],      # fixed, not modelled. See the module docstring.
            "winner_confidence": 55.0,
            "home_odds": -150, "away_odds": 130,
            "under_over_line": 224.5,
            "under_over_prediction": "OVER", "under_over_confidence": 52.0,
            "model": "rehearsal (not a model output)",
            "expected_value": {"home_team": 0.01, "away_team": -0.02},
            "_logged_at": (tip - timedelta(hours=2)).isoformat(),
        })

    # log_predictions stamps logged_at with "now", which for a game played
    # months ago would be after tip-off and correctly refused. So the rehearsal
    # moves the clock rather than weakening the rule: this is the one thing it
    # has to fake, and faking the clock is safer than faking the constraint.
    real_datetime = main_api.datetime
    written = 0
    for p in slate:
        stamp = datetime.fromisoformat(p["_logged_at"])

        # Subclass rather than replace: main_api also calls fromisoformat and
        # astimezone on this name, and swapping the whole class out breaks
        # them. Only the clock moves.
        class _FrozenClock(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return stamp if tz else stamp.replace(tzinfo=None)

            @classmethod
            def utcnow(cls):
                return stamp.replace(tzinfo=None)

        main_api.datetime = _FrozenClock
        try:
            main_api.log_predictions({"predictions": [p]}, "rehearsal", "NBA")
            written += 1
        finally:
            main_api.datetime = real_datetime

    conn = scratch_conn()
    logged = conn.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
    logger.info("")
    logger.info("--- the writer ---")
    check("every prediction was logged before its tip-off", logged == len(slate),
          f"{logged}/{len(slate)} rows")
    late = conn.execute(
        "SELECT COUNT(*) FROM predictions_log WHERE logged_at >= game_start_time_utc"
    ).fetchone()[0]
    check("no row claims to predate a game it does not", late == 0, f"{late} violations")

    # --- the constraints, exercised rather than trusted ---------------------
    logger.info("")
    logger.info("--- the guarantees ---")
    row_id = conn.execute("SELECT id FROM predictions_log LIMIT 1").fetchone()[0]
    try:
        conn.execute("UPDATE predictions_log SET predicted_winner='CHANGED' WHERE id=?", (row_id,))
        check("a logged pick cannot be rewritten", False, "the update was accepted")
    except sqlite3.IntegrityError:
        check("a logged pick cannot be rewritten", True)
    try:
        conn.execute("DELETE FROM predictions_log WHERE id=?", (row_id,))
        check("a logged pick cannot be deleted", False, "the delete was accepted")
    except sqlite3.IntegrityError:
        check("a logged pick cannot be deleted", True)
    conn.commit()
    conn.close()

    # --- the grader, on real box scores -------------------------------------
    logger.info("")
    logger.info("--- the grader, against real final scores ---")
    graded = grade_predictions.grade(odds_db=scratch_db, team_db=TEAM_DB)
    check("the grader filled in results", graded > 0, f"{graded} of {logged} graded")

    conn = scratch_conn()
    # Keyed by game, not by matchup: two teams can meet more than once.
    by_game = {(g["home_name"], g["away_name"], g["game_date"][:10]): g for g in games}
    correct_totals = mismatches = 0
    for r in conn.execute("SELECT home_team, away_team, actual_winner, actual_total, "
                          "game_start_time_utc FROM predictions_log "
                          "WHERE actual_winner IS NOT NULL"):
        g = by_game.get((r["home_team"], r["away_team"], r["game_start_time_utc"][:10]))
        if not g:
            continue
        if r["actual_total"] == g["home_pts"] + g["away_pts"] and r["actual_winner"] == g["home_name"]:
            correct_totals += 1
        else:
            mismatches += 1
    check("every graded row matches the archive's box score", mismatches == 0,
          f"{correct_totals} verified, {mismatches} wrong")

    if graded:
        first = conn.execute("SELECT id FROM predictions_log WHERE actual_winner IS NOT NULL "
                             "LIMIT 1").fetchone()[0]
        try:
            conn.execute("UPDATE predictions_log SET actual_winner='SOMEONE ELSE' WHERE id=?",
                         (first,))
            check("a graded pick cannot be regraded", False, "the regrade was accepted")
        except sqlite3.IntegrityError:
            check("a graded pick cannot be regraded", True)
    conn.commit()

    # --- what the public endpoint would serve -------------------------------
    logger.info("")
    logger.info("--- the public payload ---")
    out = main_api.get_prediction_log(days=3650)
    leaked = [k for k in out["summary"] if k.startswith("ou_")]
    leaked += [k for k in (out["predictions"][0] if out["predictions"] else {})
               if k in ("ou_prediction", "ou_confidence")]
    check("the withdrawn totals pick is not in the payload", not leaked, str(leaked))
    check("the endpoint reports a graded count", out["summary"]["graded"] == graded,
          f"graded={out['summary']['graded']}")
    conn.close()

    logger.info("")
    logger.info("=" * 60)
    if FAILED:
        logger.error("REHEARSAL FAILED: %d check(s)", len(FAILED))
        for f in FAILED:
            logger.error("  - %s", f)
        return 1
    logger.info("REHEARSAL PASSED: %d checks, %d real games, 0 production writes",
                len(PASSED), len(games))
    logger.info("Not proven here, and still only provable on a live slate: that "
                "PredictionRunner loads the model, that the odds provider returns NBA, "
                "and that the feed carries a tip-off time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
