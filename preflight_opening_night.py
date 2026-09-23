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


def new_season_started(today: date) -> bool:
    """Has any game of the upcoming/current season actually been played?

    The team-stats checks used to key off the calendar (`today <
    OPENING_NIGHT`). On opening night itself that is false, but no 2026-27
    game has finished yet, so the snapshot is legitimately last season's and
    legitimately weeks old -- and both checks went WRONG with advice ("run
    refresh_team_stats.py") that cannot help, since the refresh writes
    nothing until a game is played. Two red herrings on the one morning the
    output matters is how a real finding gets scrolled past. Ask the
    archive instead of the calendar.
    """
    season = season_for(max(today, OPENING_NIGHT))
    try:
        conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
        try:
            n = conn.execute("SELECT COUNT(*) FROM box_scores WHERE season = ?",
                             (season,)).fetchone()[0]
        finally:
            conn.close()
    except Exception:
        return today > OPENING_NIGHT
    return n > 0


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
        elif not new_season_started(today):
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
        if not new_season_started(today):
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

    # <=, not <: on the morning of opening night no game has been played, so
    # zero is correct. From the 21st the backfill must have landed some.
    if today <= OPENING_NIGHT:
        check(f"the archive has {season} games", PENDING,
              f"{this_season} so far, and the season has not started. Opening night "
              "correctly reads as fully rested for everyone; real rest appears from "
              "each team's second game.")
    else:
        check(f"the archive has {season} games", OK if this_season else BAD,
              f"{this_season} game(s) — if this stays 0 after games are played, the "
              "backfill is not running and rest will be wrong all season")

    # The names the runner will actually be handed come from the team-stats
    # snapshot, and the archive spells one of them differently: 'LA Clippers'
    # there, 'Los Angeles Clippers' here. Every one has to land on real rest,
    # or that team quietly drops to the stale-CSV path on its own.
    try:
        import main_api
        import pandas as pd
        conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
        newest = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' "
            "ORDER BY name DESC LIMIT 1").fetchone()[0]
        names = [r[0] for r in conn.execute(f'SELECT TEAM_NAME FROM "{newest}"')]
        conn.close()
        keyed = main_api.PredictionRunner._last_game_dates(
            type("_", (), {"project_root": REPO_ROOT})()) or {}
        unresolved = [n for n in names if main_api._canonical_team(n) not in keyed]
    except Exception as e:
        unresolved = None
        check("every team the feed can name has rest history", BAD, str(e)[:140])
    if unresolved is not None:
        check("every team the feed can name has rest history",
              OK if not unresolved else BAD,
              f"{len(names)} names, all resolved" if not unresolved
              else f"no rest history for: {', '.join(unresolved)}")

    csvs = sorted(glob.glob(os.path.join(REPO_ROOT, "Data", "nba-*-UTC.csv")), reverse=True)
    newest = os.path.basename(csvs[0]) if csvs else None
    check("the schedule CSV is only a fallback now", OK,
          f"newest is {newest}; days-rest reads box_scores first, so a stale CSV no "
          "longer silently pins every team at maximum rest")


def check_job_coverage(today: date) -> None:
    """Did the scheduled jobs actually FIRE as often as they are set to?

    Everything else in this file checks that the machinery is correct. This
    checks that it ran, which is a different question and was the one nobody
    was asking: on 2026-09-22 the hourly job had fired 38 times out of ~76 due
    and the 15-minute job 156 out of ~306. Both near 50%, in long contiguous
    blocks -- the machine was asleep.

    That is not a cosmetic gap. Replaying a real NBA schedule at a 50% miss
    rate puts 25.1% of tip-offs on a closing line more than 20 minutes early,
    worst case 157 minutes, against 0.0% when every run fires. A quarter of the
    season's closing lines, and no API tier buys it back.

    Cause was four booleans on the task definitions (WakeToRun,
    StartWhenAvailable and both battery settings), corrected 2026-09-22. This
    check exists so the correction is visible and a regression is not.

    Window is the last 24 hours, so it reflects the machine's CURRENT
    behaviour rather than averaging in a period that has already been fixed.
    """
    section("DID THE JOBS ACTUALLY RUN")
    import re
    cutoff = datetime.now() - timedelta(hours=24)
    jobs = (("frequent", 15), ("hourly", 60))
    for name, every_min in jobs:
        path = os.path.join(REPO_ROOT, "logs", f"scheduled_{name}.log")
        if not os.path.exists(path):
            check(f"{name} job has a log", BAD, path)
            continue
        stamps = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - INFO - === \w+ run starting",
                             line)
                if m:
                    t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    if t >= cutoff:
                        stamps.append(t)
        expected = max(1, int(24 * 60 / every_min))
        pct = 100.0 * len(stamps) / expected
        detail = f"{len(stamps)}/{expected} runs in 24h ({pct:.0f}%)"
        if pct >= 90:
            check(f"the {name} job is firing on schedule", OK, detail)
        elif pct >= 75:
            check(f"the {name} job is firing on schedule", PENDING,
                  detail + " — some runs missed; watch it")
        else:
            check(f"the {name} job is firing on schedule", BAD,
                  detail + " — the machine is sleeping through runs. Check "
                  "WakeToRun / StartWhenAvailable / the battery settings on the "
                  "task; see the NFL runbook.")


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

    # "Why this pick" is logged with every pick. If explaining breaks, picks
    # still go out without one (main_api falls back to predict()), so this is
    # the only place that failure would be noticed.
    try:
        p2, reasons = cand.predict_explained(frame, [(home, away)], ["2026-04-01"])
        groups = (reasons[0] or {}).get("groups") or []
        same = abs(float(p2[0]) - p) < 1e-12
    except Exception as e:
        return check("pick explanations work and match the pick", BAD, str(e)[:160])
    check("pick explanations work and match the pick",
          OK if same and groups else BAD,
          f"{len(groups)} factors, top: {groups[0]['label']}" if same and groups
          else "explained probability differs from predict()" if not same else "no factors")

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
    # Every game must carry a tip-off time, or log_predictions refuses it and
    # it can never be logged later. This check would have caught the defect
    # found on 2026-09-22 months earlier: the provider read a key the scraper
    # never writes, so 100% of games arrived with no start time. It passed
    # every other check here, because resolving to the right sport and
    # returning games are both things a broken feed can still do.
    try:
        slate = provider.get_odds() or {}
        with_tip = sum(1 for g in slate.values() if g.get("game_start_time_utc"))
        if not slate:
            check("every game in the feed carries a tip-off time", PENDING,
                  "no games on the board to inspect")
        else:
            check("every game in the feed carries a tip-off time",
                  OK if with_tip == len(slate) else BAD,
                  f"{with_tip}/{len(slate)} games"
                  + ("" if with_tip == len(slate) else
                     " — games without one are refused by the ledger and lost for good"))
    except Exception as e:
        check("every game in the feed carries a tip-off time", BAD, str(e)[:140])

    if resolved == "NBA":
        check("the prediction feed resolves to NBA", OK, resolved)
    elif today < OPENING_NIGHT:
        check("the prediction feed resolves to NBA", PENDING,
              f"resolves to '{resolved}' in the offseason; nothing is logged, which "
              "is correct. It MUST read NBA on the night.")
    else:
        check("the prediction feed resolves to NBA", BAD,
              f"resolves to '{resolved}' — no NBA pick will be logged tonight")


def check_operations(today: date) -> None:
    section("OPERATIONS")
    # Two things read OPENING_NIGHT besides this script: the preseason guard in
    # main_api.log_predictions (keeps exhibition picks off the public record)
    # and the in-season test that turns an empty odds board into a failure.
    # Left stale, both quietly stop working next fall.
    stale = today > OPENING_NIGHT + timedelta(days=270)
    check("OPENING_NIGHT is this season's", BAD if stale else OK,
          f"{OPENING_NIGHT} is last season's: bump it (and the three season constants)"
          if stale else f"{OPENING_NIGHT}")

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
    except Exception:
        pass
    configured = bool((os.environ.get("LEDGER_SYNC_URL") or "").strip()
                      and (os.environ.get("LEDGER_SYNC_SECRET") or "").strip())
    check("the ledger is mirrored to the public server", OK if configured else PENDING,
          "LEDGER_SYNC_URL and LEDGER_SYNC_SECRET set" if configured
          else "no public server yet; push_ledger.py prints SKIPPED (DEPLOY.md 3a)")

    # The data is backed up only by backup_to_drive.py (OneDrive is full and
    # syncs nothing). A missing or old backup is a reminder, not a failure.
    root = os.environ.get("BACKUP_ROOT") or ("D:" + chr(92) + "BettingBuddy-Backup")
    if not os.path.isdir(root):
        check("a recent data backup exists", PENDING,
              f"{root} not found (drive unplugged?); run backup_to_drive.py")
        return
    dated = sorted(d for d in os.listdir(root)
                   if len(d) >= 10 and d[:4].isdigit() and os.path.exists(os.path.join(root, d, "README.txt")))
    if not dated:
        check("a recent data backup exists", PENDING, f"no completed backup in {root}")
        return
    last = datetime.strptime(dated[-1][:10], "%Y-%m-%d").date()
    age = (today - last).days
    check("a recent data backup exists", OK if age <= 7 else PENDING,
          f"last {last} ({age} day(s) ago)" + ("" if age <= 7 else "; run backup_to_drive.py"))


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
    check_job_coverage(today)
    check_ledger()
    check_model()
    check_feed(today, args.skip_network)
    check_operations(today)

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
