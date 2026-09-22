"""
preflight_opening_night.py
==========================
Everything that has to be true for NBA opening night to work, checked in one
command, on any day.

WHY THIS EXISTS. The prediction path depends on about a dozen things being in
the right state, and most of them fail QUIETLY. A schedule file a year out of
date returns a plausible number of rest days rather than an error. A team-stats
snapshot written daily looks fresh whether or not it holds this season. A
season constant nobody bumped serves last season's page with no warning. Each
one is individually obvious and collectively impossible to hold in your head
at 9am on 20 October.

`rehearse_ledger.py` already proves the ledger works: constraints, grading,
the public payload. It says nothing about CONFIGURATION, which is what this
covers. Run both.

READ THE "NOT YET" RESULTS PROPERLY. Several checks cannot pass before the
season starts and say so rather than failing: there is no NBA slate in
September, and no 2026-27 game has been played, so anything downstream of a
played game is pending rather than broken. The script separates "wrong" from
"not yet" so that a clean offseason run is actually clean, and the same
command becomes meaningful on the night without being edited.

Usage:
    venv/Scripts/python.exe preflight_opening_night.py
    venv/Scripts/python.exe preflight_opening_night.py --as-of 2026-10-21
    venv/Scripts/python.exe preflight_opening_night.py --skip-network
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

TEAM_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
MODEL_DIR = os.path.join(REPO_ROOT, "Models", "candidate_2026-08")

#: Opening night. Update when the schedule is published.
OPENING_NIGHT = date(2026, 10, 20)

OK, BAD, PENDING = "PASS", "FAIL", "not yet"
results: list[tuple[str, str, str]] = []


def check(label: str, state: str, detail: str = "") -> None:
    results.append((state, label, detail))
    print(f"  {state:>7}  {label}" + (f" — {detail}" if detail else ""))


def season_for(day: date) -> str:
    """The NBA season label a date belongs to. October starts a new one."""
    start = day.year if day.month >= 10 else day.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def section(name: str) -> None:
    print(f"\n=== {name} ===")


# --------------------------------------------------------------------------
def check_season_labels(today: date) -> None:
    section("SEASON LABELS")
    expected = season_for(today)
    import main_api
    backend = main_api.CURRENT_SEASON
    started = today >= OPENING_NIGHT
    if backend == expected:
        check("backend CURRENT_SEASON matches today's season", OK, backend)
    elif not started:
        check("backend CURRENT_SEASON matches today's season", PENDING,
              f"is {backend}; becomes {expected} on {OPENING_NIGHT}. Bump it that morning.")
    else:
        check("backend CURRENT_SEASON matches today's season", BAD,
              f"is {backend}, should be {expected}. Every endpoint defaulting to "
              f"CURRENT_SEASON is serving last season.")

    # The frontend keeps its own copies and this script cannot import TypeScript,
    # so name them rather than pretend to have checked them.
    check("frontend season constants are a MANUAL step", PENDING,
          "CURRENT_SEASON in src/lib/nba-api.ts and CURRENT_END_YEAR in "
          "src/lib/archive-seasons.ts must be bumped by hand the same morning")


def check_team_stats(today: date) -> None:
    section("TEAM STATS THE MODEL PREDICTS FROM")
    if not os.path.exists(TEAM_DB):
        return check("team stats database exists", BAD, TEAM_DB)
    conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' "
            "ORDER BY name DESC LIMIT 1").fetchone()
        if not row:
            return check("a team-stats snapshot exists", BAD, "no dated table")
        table = row[0]
        try:
            age = (today - datetime.strptime(table, "%Y-%m-%d").date()).days
        except ValueError:
            return check("the newest snapshot is date-named", BAD, table)
        check("a team-stats snapshot exists", OK, f"{table}, {age} day(s) old")

        import main_api
        if age <= main_api.TEAM_STATS_MAX_AGE_DAYS:
            check("the snapshot is fresh", OK, f"{age} <= {main_api.TEAM_STATS_MAX_AGE_DAYS}")
        elif today < OPENING_NIGHT:
            # refresh_team_stats writes nothing when the new season has no games
            # yet, so between 1 October and opening night the newest table stops
            # advancing and the age warning fires while nothing is actually wrong.
            check("the snapshot is fresh", PENDING,
                  f"{age} days old, expected between 1 October and opening night: "
                  "the refresh writes nothing until a game is played")
        else:
            check("the snapshot is fresh", BAD,
                  f"{age} days old — run refresh_team_stats.py")

        gp = conn.execute(f'SELECT MAX(GP) FROM "{table}"').fetchone()[0]
        if today < OPENING_NIGHT:
            check("the snapshot holds this season", PENDING,
                  f"max GP {gp} — last season's finals, which is the only sensible "
                  "input until a 2026-27 game is played")
        elif gp and gp >= 82:
            check("the snapshot holds this season", BAD,
                  f"max GP is {gp}: a full season, so this is LAST season's numbers")
        else:
            check("the snapshot holds this season", OK, f"max GP {gp}")
    finally:
        conn.close()


def check_days_rest(today: date) -> None:
    section("DAYS REST")
    # Before October, season_for(today) is the season that just ENDED. What
    # matters for opening night is the one about to start, so ask about that.
    season = season_for(max(today, OPENING_NIGHT))
    conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM box_scores WHERE game_date IS NOT NULL"
                             ).fetchone()[0]
        this_season = conn.execute(
            "SELECT COUNT(*) FROM box_scores WHERE season = ?", (season,)).fetchone()[0]
    finally:
        conn.close()
    check("the archive can supply game dates", OK if total else BAD, f"{total:,} games")

    if today < OPENING_NIGHT:
        check(f"the archive has {season} games", PENDING,
              f"{this_season} so far, and the season has not started. Opening night "
              "correctly reads as fully rested for everyone; real rest appears from "
              "each team's second game.")
    else:
        check(f"the archive has {season} games", OK if this_season else BAD,
              f"{this_season} game(s) — if this stays 0 after games are played, the "
              "backfill is not running and rest will be wrong all season")

    csvs = sorted(glob.glob(os.path.join(REPO_ROOT, "Data", "nba-*-UTC.csv")), reverse=True)
    newest = os.path.basename(csvs[0]) if csvs else None
    check("the schedule CSV is only a fallback now", OK,
          f"newest is {newest}; days-rest reads box_scores first, so a stale CSV no "
          "longer silently pins every team at maximum rest")


def check_ledger() -> None:
    section("THE PREDICTION LEDGER")
    if not os.path.exists(ODDS_DB):
        return check("ledger database exists", BAD, ODDS_DB)
    conn = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions_log'"
        ).fetchone()
        if not sql:
            return check("predictions_log exists", BAD, "table missing")
        ddl = sql[0] or ""
        check("a pick cannot be logged after tip-off",
              OK if "CHECK (logged_at < game_start_time_utc)" in ddl else BAD,
              "CHECK constraint on the table")
        check("tip-off time is required",
              OK if "game_start_time_utc TEXT NOT NULL" in ddl else BAD, "NOT NULL")
        triggers = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name='predictions_log'")}
        check("picks are frozen, undeletable and ungradeable twice",
              OK if len(triggers) >= 3 else BAD,
              f"{len(triggers)} trigger(s): {', '.join(sorted(triggers)) or 'none'}")

        n = conn.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
        check("rows in the ledger", OK, f"{n} — empty is correct before opening night")

        snap_prov = conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('odds_snapshots') "
            "WHERE name='provenance'").fetchone()[0]
        check("odds snapshots record how we came to know the price",
              OK if snap_prov else BAD,
              "provenance column present" if snap_prov else "missing — a repaired "
              "price would be indistinguishable from a watched one")
    finally:
        conn.close()
    check("the ledger path is separately rehearsed", OK,
          "run rehearse_ledger.py; this script checks configuration, not behaviour")


def check_model() -> None:
    section("THE SERVING MODEL")
    manifest = os.path.join(MODEL_DIR, "feature_manifest.json")
    if not os.path.exists(manifest):
        return check("the sealed candidate model is present", BAD, MODEL_DIR)
    with open(manifest, encoding="utf-8") as fh:
        m = json.load(fh)
    cols = m.get("feature_columns") or []
    check("the sealed candidate model is present", OK,
          f"{os.path.basename(MODEL_DIR)}, {len(cols)} features")
    check("the feature count matches the manifest",
          OK if len(cols) == m.get("n_features") else BAD,
          f"{len(cols)} vs n_features {m.get('n_features')}")
    rest = [c for c in cols if "rest" in c.lower()]
    check("the rest features the model expects are named",
          OK if len(rest) >= 2 else BAD, ", ".join(rest))
    # Importing the module proves nothing. The first version of this check was
    # `candidate_live imports` and it passed happily on 2026-09-22 while
    # get_candidate() was returning None and every prediction was falling back
    # to the old model. A check that cannot fail is worse than no check.
    try:
        from src.Predict import candidate_live
        cand = candidate_live.get_candidate()
    except Exception as e:
        return check("the candidate model actually loads", BAD, str(e)[:160])
    if cand is None:
        return check(
            "the candidate model actually loads", BAD,
            f"get_candidate() returned None ({candidate_live._instance_error}). Every "
            "prediction silently falls back to the OLD model while the site advertises "
            "the candidate's 67.2%.")
    check("the candidate model actually loads", OK,
          f"{len(cand.tg):,} team-game rows through {cand._tg_max_date}")

    # And that it produces a number, which is the only proof the feature
    # pipeline still lines up with the sealed artifact.
    try:
        import pandas as pd
        conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' "
            "ORDER BY name DESC LIMIT 1").fetchone()[0]
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn, index_col="index")
        conn.close()
        home, away = "Boston Celtics", "Los Angeles Lakers"
        row = pd.concat([df[df.TEAM_NAME == home].iloc[0],
                         df[df.TEAM_NAME == away].iloc[0].rename(lambda x: x + ".1")])
        row["Days-Rest-Home"], row["Days-Rest-Away"] = 2.0, 1.0
        frame = pd.DataFrame([row])
        frame = frame.drop(columns=[c for c in frame.columns
                                    if not pd.api.types.is_numeric_dtype(frame[c])],
                           errors="ignore")
        p = float(cand.predict(frame, [(home, away)], ["2026-04-01"])[0])
    except Exception as e:
        return check("the candidate model produces a probability", BAD, str(e)[:160])
    check("the candidate model produces a probability",
          OK if 0.0 < p < 1.0 else BAD, f"{p:.4f} on a synthetic matchup")

    # Every team the archive has ever named must map to a modern franchise, or
    # the canonical map raises and takes the whole model down with it. This is
    # what actually broke: the 1996-2001 backfill introduced two names nobody
    # had taught the normalizer.
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src", "Process-Data"))
        from retrain_features import normalize_team
        import json as _json
        conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
        names = set()
        for (tj,) in conn.execute("SELECT traditional_json FROM box_scores"):
            t = _json.loads(tj)["boxScoreTraditional"]
            for side in ("homeTeam", "awayTeam"):
                s = t[side]
                names.add(f"{s.get('teamCity','').strip()} "
                          f"{s.get('teamName','').strip()}".strip())
        conn.close()
        unknown = sorted(n for n in names if not _known(normalize_team, n))
    except Exception as e:
        return check("every team name in the archive is known", BAD, str(e)[:160])
    check("every team name in the archive is known",
          OK if not unknown else BAD,
          f"{len(names)} distinct names" if not unknown
          else f"unmapped: {', '.join(unknown)} — add them to TEAM_NAME_MAP")


def _known(normalize, name: str) -> bool:
    try:
        normalize(name)
        return True
    except KeyError:
        return False


def check_feed(today: date, skip_network: bool) -> None:
    section("THE ODDS FEED")
    if skip_network:
        return check("odds feed", PENDING, "--skip-network")
    try:
        from src.Utils.odds_api_client import fetch_nba_events
        events, quota = fetch_nba_events()
    except Exception as e:
        return check("The Odds API answers", BAD, str(e)[:120])
    check("The Odds API answers", OK,
          f"{len(events)} NBA event(s) on the board, 0 credits")
    remaining = quota.get("remaining")
    try:
        left = int(remaining)
    except (TypeError, ValueError):
        left = None
    if left is None:
        check("quota is readable", BAD, str(remaining))
    elif left < 60:
        check("quota is above the floor of 60", BAD,
              f"{left} left — recorders will refuse routine captures")
    else:
        check("quota is above the floor of 60", OK, f"{left} left")
        if today >= OPENING_NIGHT and left < 500:
            check("quota will last the month", BAD,
                  f"{left} left; a full NBA month measures ~1,300. See "
                  "forecast_odds_credits.py")

    # The provider that actually feeds predictions falls back to WNBA out of
    # season, and the writer refuses to log anything that is not NBA.
    try:
        from src.DataProviders.SbrOddsProvider import SbrOddsProvider
        provider = SbrOddsProvider(sportsbook="fanduel", sport="NBA")
        resolved = provider.get_resolved_sport() or "unknown"
    except Exception as e:
        return check("the prediction feed resolves to NBA", BAD, str(e)[:120])
    if resolved == "NBA":
        check("the prediction feed resolves to NBA", OK, resolved)
    elif today < OPENING_NIGHT:
        check("the prediction feed resolves to NBA", PENDING,
              f"resolves to '{resolved}' in the offseason; nothing is logged, which "
              "is correct. It MUST read NBA on the night.")
    else:
        check("the prediction feed resolves to NBA", BAD,
              f"resolves to '{resolved}' — no NBA pick will be logged tonight")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check everything opening night depends on.")
    ap.add_argument("--as-of", help="Pretend it is this date (YYYY-MM-DD).")
    ap.add_argument("--skip-network", action="store_true")
    args = ap.parse_args()

    today = (datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of
             else datetime.now(timezone.utc).date())
    days = (OPENING_NIGHT - today).days

    print("=" * 72)
    print(f"opening-night preflight — {today}, {days} day(s) to {OPENING_NIGHT}")
    print("=" * 72)

    check_season_labels(today)
    check_team_stats(today)
    check_days_rest(today)
    check_ledger()
    check_model()
    check_feed(today, args.skip_network)

    bad = [r for r in results if r[0] == BAD]
    pending = [r for r in results if r[0] == PENDING]
    print("\n" + "=" * 72)
    print(f"{len(results) - len(bad) - len(pending)} pass, {len(pending)} not yet, "
          f"{len(bad)} wrong")
    for _, label, detail in pending:
        print(f"  not yet  {label}")
    for _, label, detail in bad:
        print(f"  WRONG    {label} — {detail}")
    if bad:
        print("\nSomething opening night depends on is wrong. Fix before 20 October.")
        return 1
    print("\nNothing is wrong. The 'not yet' items are the ones to re-run on the night.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
