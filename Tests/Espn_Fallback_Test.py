"""
Espn_Fallback_Test.py
=====================
Pins the two ways the served model's inputs are kept current while
stats.nba.com refuses this PC (2026-09-28):

  - team_stats_from_archive.py rebuilds the team-stats snapshot from our own
    box scores, in nba.com's exact table shape;
  - src/Utils/espn_boxscore.py ingests NEW games from ESPN into their own
    tables (source = 'espn'), read only for games box_scores does not hold;
  - daily_update.py uses both, and only when the nba.com step failed.

No network anywhere: ESPN answers come from the trimmed fixtures in
Tests/fixtures/espn (real payloads fetched 2026-09-28), HTTP is a stub, and
every database is a temp file. The nba.com values the ESPN fixtures are
checked against are our own stored nba.com box scores for the same games.
Two tests read the real archive, read-only, and skip when it is absent.
"""

import gzip
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import daily_update  # noqa: E402
import job_health  # noqa: E402
import team_stats_from_archive as ts  # noqa: E402
from src.Utils import espn_boxscore as eb  # noqa: E402

FIX = os.path.join(ROOT, "Tests", "fixtures", "espn")
REAL_DB = os.path.join(ROOT, "Data", "TeamData.sqlite")

TEAMS = [(1610612737, 'ATL', 'Atlanta Hawks'), (1610612738, 'BOS', 'Boston Celtics'),
         (1610612739, 'CLE', 'Cleveland Cavaliers'), (1610612740, 'NOP', 'New Orleans Pelicans'),
         (1610612741, 'CHI', 'Chicago Bulls'), (1610612742, 'DAL', 'Dallas Mavericks'),
         (1610612743, 'DEN', 'Denver Nuggets'), (1610612744, 'GSW', 'Golden State Warriors'),
         (1610612745, 'HOU', 'Houston Rockets'), (1610612746, 'LAC', 'Los Angeles Clippers'),
         (1610612747, 'LAL', 'Los Angeles Lakers'), (1610612748, 'MIA', 'Miami Heat'),
         (1610612749, 'MIL', 'Milwaukee Bucks'), (1610612750, 'MIN', 'Minnesota Timberwolves'),
         (1610612751, 'BKN', 'Brooklyn Nets'), (1610612752, 'NYK', 'New York Knicks'),
         (1610612753, 'ORL', 'Orlando Magic'), (1610612754, 'IND', 'Indiana Pacers'),
         (1610612755, 'PHI', 'Philadelphia 76ers'), (1610612756, 'PHX', 'Phoenix Suns'),
         (1610612757, 'POR', 'Portland Trail Blazers'), (1610612758, 'SAC', 'Sacramento Kings'),
         (1610612759, 'SAS', 'San Antonio Spurs'), (1610612760, 'OKC', 'Oklahoma City Thunder'),
         (1610612761, 'TOR', 'Toronto Raptors'), (1610612762, 'UTA', 'Utah Jazz'),
         (1610612763, 'MEM', 'Memphis Grizzlies'), (1610612764, 'WAS', 'Washington Wizards'),
         (1610612765, 'DET', 'Detroit Pistons'), (1610612766, 'CHA', 'Charlotte Hornets')]
IND, PHI, NYK, DET, TOR, SAS, CHA, MIA = (1610612754, 1610612755, 1610612752, 1610612765,
                                          1610612761, 1610612759, 1610612766, 1610612748)

#: nba.com's own box score for the fixture games (traditional totals; TOV is
#: the advanced box score's recovered team total), from our archive.
NBA_COM = {
    ("0022500502", DET): dict(PTS=121, FGM=47, FGA=86, FG3M=16, FG3A=31, FTM=11, FTA=16, OREB=14, DREB=30,
                              REB=44, AST=25, STL=12, BLK=11, PF=15, TOV=16, MIN=240),
    ("0022500502", NYK): dict(PTS=90, FGM=32, FGA=76, FG3M=13, FG3A=30, FTM=13, FTA=14, OREB=5, DREB=25,
                              REB=30, AST=15, STL=7, BLK=3, PF=17, TOV=20, MIN=240),
    ("0022501175", IND): dict(PTS=94, FGM=33, FGA=88, FG3M=14, FG3A=50, FTM=14, FTA=16, OREB=10, DREB=42,
                              REB=52, AST=25, STL=6, BLK=8, PF=15, TOV=21, MIN=240),
    ("0022501175", PHI): dict(PTS=105, FGM=42, FGA=104, FG3M=5, FG3A=29, FTM=16, FTA=19, OREB=16, DREB=42,
                              REB=58, AST=17, STL=13, BLK=2, PF=14, TOV=8, MIN=240),
}


def fixture(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as f:
        return json.load(f)


class _Resp:
    def __init__(self, body, code=200):
        self.status_code, self._body = code, body

    def json(self):
        return self._body


class FakeHttp:
    """ESPN, played by the fixtures. Records every request."""

    def __init__(self, scoreboards=None, summaries=None):
        self.scoreboards = scoreboards or {}
        self.summaries = summaries or {}
        self.calls = []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url == eb.SCOREBOARD_URL:
            body = self.scoreboards.get(params["dates"], {"events": []})
        elif url == eb.SUMMARY_URL:
            body = self.summaries.get(params["event"])
            if body is None:
                return _Resp({}, 404)
        else:
            raise AssertionError(f"unexpected URL {url}")
        return _Resp(body)


def _box_json(game_id, home, away, hs, as_, clock="240:00"):
    """Minimal nba.com traditional/advanced box score JSON, as box_scores stores it."""
    def team(tid, s):
        name = dict((t[0], t[2]) for t in TEAMS)[tid]
        city, _, nick = name.rpartition(" ")
        return {"teamId": tid, "teamCity": city, "teamName": nick,
                "statistics": {"minutes": clock, "fieldGoalsMade": s["FGM"], "fieldGoalsAttempted": s["FGA"],
                               "threePointersMade": s["FG3M"], "threePointersAttempted": s["FG3A"],
                               "freeThrowsMade": s["FTM"], "freeThrowsAttempted": s["FTA"],
                               "reboundsOffensive": s["OREB"], "reboundsDefensive": s["DREB"],
                               "reboundsTotal": s["REB"], "assists": s["AST"], "steals": s["STL"],
                               "blocks": s["BLK"], "turnovers": s["TOV"], "foulsPersonal": s["PF"],
                               "points": s["PTS"]}}

    def adv(tid, s):
        # estimatedTeamTurnoverPercentage * possessions / 100 = TOV
        return {"teamId": tid, "statistics": {"minutes": clock, "possessions": 100.0,
                                              "estimatedTeamTurnoverPercentage": float(s["TOV"])}}
    trad = {"boxScoreTraditional": {"gameId": game_id, "homeTeam": team(home, hs), "awayTeam": team(away, as_)}}
    advj = {"boxScoreAdvanced": {"gameId": game_id, "homeTeam": adv(home, hs), "awayTeam": adv(away, as_)}}
    return json.dumps(trad), json.dumps(advj)


def _line(pts, **kw):
    s = dict(FGM=40, FGA=85, FG3M=12, FG3A=34, FTM=15, FTA=20, OREB=10, DREB=33, REB=43, AST=25,
             STL=8, BLK=5, TOV=13, PF=19, PTS=pts)
    s.update(kw)
    return s


def make_db(path):
    """A throwaway TeamData: the tables the fallback reads, nothing else."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE team_metadata (team_id INTEGER PRIMARY KEY, full_name TEXT, abbreviation TEXT);
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, last_team_id INTEGER);
        CREATE TABLE box_scores (game_id TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, home_team_id INTEGER,
            away_team_id INTEGER, season TEXT, season_type TEXT, game_date TEXT,
            traditional_json TEXT, advanced_json TEXT, pbp_json TEXT);
        CREATE TABLE game_results (game_id TEXT NOT NULL, team_id INTEGER NOT NULL, season TEXT,
            season_type TEXT, game_date TEXT, team_abbr TEXT, team_name TEXT, matchup TEXT, wl TEXT,
            pts INTEGER, PRIMARY KEY (game_id, team_id));
        CREATE TABLE nba_response_mirror (cache_key TEXT PRIMARY KEY, endpoint TEXT NOT NULL,
            params TEXT NOT NULL, season TEXT, season_type TEXT, fetched_at TEXT NOT NULL,
            final INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL, bytes INTEGER, payload BLOB NOT NULL);
    """)
    con.executemany("INSERT INTO team_metadata VALUES (?, ?, ?)", [(t, n, a) for t, a, n in TEAMS])
    con.executemany("INSERT INTO players VALUES (?, ?, ?)",
                    [(1630169, "Tyrese Haliburton", IND), (1630178, "Tyrese Maxey", PHI)])
    con.executemany("INSERT INTO game_results VALUES (?,?,?,?,?,?,?,?,?,?)", [
        ("0022501175", IND, "2025-26", "Regular Season", "2026-04-10", "IND", "Indiana Pacers", "IND vs. PHI", "L", 94),
        ("0022501175", PHI, "2025-26", "Regular Season", "2026-04-10", "PHI", "Philadelphia 76ers", "PHI @ IND", "W", 105),
        ("0052500111", CHA, "2025-26", "PlayIn", "2026-04-14", "CHA", "Charlotte Hornets", "CHA vs. MIA", "W", 127),
        ("0052500111", MIA, "2025-26", "PlayIn", "2026-04-14", "MIA", "Miami Heat", "MIA @ CHA", "L", 126),
    ])
    sched = {"leagueSchedule": {"seasonYear": "2025-26", "gameDates": [{"gameDate": "12/16/2025 00:00:00", "games": [
        {"gameId": "0062500001", "gameDateEst": "2025-12-16T00:00:00Z",
         "homeTeam": {"teamId": NYK}, "awayTeam": {"teamId": SAS}}]}]}}
    con.execute("INSERT INTO nba_response_mirror VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("scheduleleaguev2_league_id=00_season=2025-26", "scheduleleaguev2", "{}", "2025-26", None,
                 "2026-09-28T00:00:00+00:00", 0, "disk-cache", 0, gzip.compress(json.dumps(sched).encode())))
    con.commit()
    return con


def add_box(con, game_id, gdate, home, away, hs, as_, season="2025-26", stype="Regular Season", clock="240:00"):
    tj, aj = _box_json(game_id, home, away, hs, as_, clock)
    con.execute("INSERT INTO box_scores VALUES (?,?,?,?,?,?,?,?,?,NULL)",
                (game_id, "2026-01-01T00:00:00", home, away, season, stype, gdate, tj, aj))
    con.commit()


def add_reference_snapshot(con, name="2026-04-01"):
    bm = ts._harness()
    cols = ([("index", "INTEGER"), ("TEAM_ID", "INTEGER"), ("TEAM_NAME", "TEXT")]
            + [(c, "INTEGER" if c in ("GP", "W", "L") else "REAL") for c in bm.BASE]
            + [(c + "_RANK", "INTEGER") for c in bm.BASE] + [("Date", "TEXT")])
    con.execute('CREATE TABLE "%s" (%s)' % (name, ", ".join('"%s" %s' % (c, t) for c, t in cols)))
    con.execute(f'INSERT INTO "{name}" ("index", TEAM_ID, TEAM_NAME) VALUES (0, ?, ?)', (IND, "Indiana Pacers"))
    con.commit()
    return cols


class TempDbCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "TeamData.sqlite")
        self.con = make_db(self.db)

    def tearDown(self):
        self.con.close()
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. The ESPN parser against nba.com's own numbers
# ---------------------------------------------------------------------------
class ParserTest(unittest.TestCase):

    def check_against_nba(self, event_id, game_id, home, away):
        p = eb.parse_summary(fixture(f"summary_{event_id}.json"))
        for side, tid in (("home", home), ("away", away)):
            want = NBA_COM[(game_id, tid)]
            got = {k: p["teams"][side][k] for k in want}
            self.assertEqual(got, want, f"{event_id} {side}")
        return p

    def test_regular_season_game_matches_nba_com(self):
        p = self.check_against_nba("401810357", "0022500502", DET, NYK)
        self.assertEqual(p["game_date"], "2026-01-05")       # 00:00Z on the 6th is the 5th in New York
        self.assertEqual(p["periods"], 4)
        self.assertEqual(p["espn_season_type"], 2)

    def test_negative_team_turnovers_read_as_zero(self):
        # ESPN sent teamTurnovers = -3 for IND: its own total (18) is short;
        # players' 21 + max(-3, 0) is nba.com's recovered 21.
        p = self.check_against_nba("401811030", "0022501175", IND, PHI)
        self.assertEqual(p["teams"]["home"]["TOV_TEAM_RAW"], -3)
        self.assertEqual(p["teams"]["home"]["TOV"], 21)

    def test_overtime_minutes(self):
        p = eb.parse_summary(fixture("summary_401866755.json"))
        self.assertEqual(p["periods"], 5)
        self.assertEqual(p["teams"]["home"]["MIN"], 265)
        self.assertEqual(p["espn_season_type"], 5)

    def test_players_parse_and_add_up(self):
        p = eb.parse_summary(fixture("summary_401810357.json"))
        played = [x for x in p["players"] if not x["did_not_play"]]
        for side in ("home", "away"):
            pts = sum(x["PTS"] for x in played if x["side"] == side)
            self.assertEqual(pts, p["teams"][side]["PTS"])
        self.assertTrue(all(isinstance(x["MIN"], int) for x in played))

    def test_unfinished_game_is_refused(self):
        body = fixture("summary_401810357.json")
        body["header"]["competitions"][0]["status"]["type"].update(name="STATUS_IN_PROGRESS", completed=False)
        with self.assertRaises(eb.EspnParseError):
            eb.parse_summary(body)

    def test_changed_shape_is_refused_not_guessed(self):
        body = fixture("summary_401810357.json")
        for t in body["boxscore"]["teams"]:
            t["statistics"] = [s for s in t["statistics"] if s["name"] != "assists"]
        with self.assertRaises(eb.EspnParseError):
            eb.parse_summary(body)

    def test_et_date(self):
        self.assertEqual(eb.et_date("2026-01-06T00:00Z"), "2026-01-05")
        self.assertEqual(eb.et_date("2026-04-10T23:30Z"), "2026-04-10")

    def test_season_from_id(self):
        self.assertEqual(eb.season_from_nba_id("0022500502"), ("2025-26", "Regular Season"))
        self.assertEqual(eb.season_from_nba_id("0052500111"), ("2025-26", "PlayIn"))
        self.assertEqual(eb.season_from_nba_id("0029600332"), ("1996-97", "Regular Season"))
        self.assertEqual(eb.season_from_nba_id("0062500001")[1], None)


# ---------------------------------------------------------------------------
# 2. Storing, provenance, and which games count
# ---------------------------------------------------------------------------
class StoreTest(TempDbCase):

    def test_store_maps_id_and_records_provenance(self):
        row = eb.store_game(self.con, fixture("summary_401811030.json"))
        self.assertEqual((row["game_id"], row["id_source"], row["usable"]), ("0022501175", "game_results", 1))
        self.assertEqual((row["season"], row["season_type"]), ("2025-26", "Regular Season"))
        for table in ("espn_box_scores", "espn_team_box", "espn_player_box"):
            sources = {r[0] for r in self.con.execute(f"SELECT source FROM {table}")}
            self.assertEqual(sources, {"espn"}, table)
        with self.assertRaises(sqlite3.IntegrityError):
            self.con.execute("UPDATE espn_team_box SET source = 'nba.com'")
        payload = self.con.execute("SELECT payload FROM espn_box_scores").fetchone()[0]
        self.assertIn("boxscore", json.loads(gzip.decompress(payload)))
        pid = self.con.execute("SELECT player_id FROM espn_player_box WHERE player_name='Tyrese Maxey'").fetchone()
        self.assertEqual(pid[0], 1630178)

    def test_cup_final_is_held_not_counted(self):
        # The summary header carries no note; the schedule maps it to 006...
        row = eb.store_game(self.con, fixture("summary_401809839.json"))
        self.assertEqual(row["game_id"], "0062500001")
        self.assertEqual(row["usable"], 0)
        # ...and the scoreboard's note alone is enough too.
        self.con.execute("DELETE FROM nba_response_mirror")
        notes = [n["headline"] for n in fixture("scoreboard_20251216.json")["events"][0]["competitions"][0]["notes"]]
        row = eb.store_game(self.con, fixture("summary_401809839.json"), scoreboard_notes=notes)
        self.assertEqual(row["usable"], 0)
        self.assertIn("NBA Cup Championship", row["exclude_reason"])
        self.assertTrue(eb.espn_team_game_rows(self.con).empty)

    def test_unmapped_regular_season_game_is_not_counted(self):
        self.con.execute("DELETE FROM game_results")
        row = eb.store_game(self.con, fixture("summary_401811030.json"))
        self.assertIsNone(row["game_id"])
        self.assertEqual(row["usable"], 0)

    def test_unmapped_play_in_counts_under_an_espn_key(self):
        self.con.execute("DELETE FROM game_results")
        row = eb.store_game(self.con, fixture("summary_401866755.json"))
        self.assertEqual((row["usable"], row["season_type"]), (1, "PlayIn"))
        self.assertEqual(row["model_game_id"], "espn:401866755")

    def test_rows_have_the_harness_shape_and_opponent_stats(self):
        eb.store_game(self.con, fixture("summary_401811030.json"))
        rows = eb.espn_team_game_rows(self.con)
        self.assertEqual(len(rows), 2)
        ind = rows[rows.team_id == IND].iloc[0]
        self.assertEqual((ind.BLKA, ind.PFD, ind.OPP_PTS, ind.W, ind.PLUS_MINUS), (2, 14, 105, 0, -11))
        self.assertEqual(ind.game_id, "0022501175")

    def test_nba_com_box_score_shadows_espn(self):
        eb.store_game(self.con, fixture("summary_401811030.json"))
        self.assertEqual(eb.stand_in_count(self.con), 1)
        add_box(self.con, "0022501175", "2026-04-10", IND, PHI, _line(94), _line(105))
        self.assertTrue(eb.espn_team_game_rows(self.con).empty)
        self.assertEqual(eb.stand_in_count(self.con), 0)
        self.assertEqual(eb.recent_sources(self.con, "2026-04-01")["espn_shadowed"], 1)

    def test_postponed_game_held_under_another_date_is_still_shadowed(self):
        # box_scores can hold a postponed game under its original date.
        eb.store_game(self.con, fixture("summary_401811030.json"))
        add_box(self.con, "0022501175", "2026-01-25", IND, PHI, _line(94), _line(105))
        self.assertTrue(eb.espn_team_game_rows(self.con).empty)

    def test_merge_returns_the_same_frame_when_nothing_stands_in(self):
        tg = pd.DataFrame({"game_id": ["x"], "game_date": ["2026-01-01"], "team_id": [IND], "team_name": ["Indiana Pacers"]})
        self.assertIs(eb.merge_team_games(self.con, tg), tg)          # no ESPN tables
        eb.ensure_tables(self.con)
        self.assertIs(eb.merge_team_games(self.con, tg), tg)          # empty ESPN tables

    def test_merge_adds_espn_rows_with_nba_names(self):
        eb.store_game(self.con, fixture("summary_401811030.json"))
        base = pd.DataFrame([dict(game_id="0022501100", season="2025-26", season_type="Regular Season",
                                  game_date="2026-04-01", team_id=IND, team_name="Indiana Pacers", is_home=True)])
        out = eb.merge_team_games(self.con, base)
        self.assertEqual(sorted(out.source), ["espn", "espn", "nba.com"])
        self.assertEqual(out[out.team_id == IND].team_name.tolist(), ["Indiana Pacers"] * 2)
        self.assertEqual(out[out.team_id == PHI].team_name.tolist(), ["Philadelphia 76ers"])

    def test_days_rest_rows_work_with_row_factory(self):
        eb.store_game(self.con, fixture("summary_401811030.json"))
        self.con.row_factory = sqlite3.Row
        rows = eb.team_date_rows(self.con)
        self.assertEqual(sorted(r["team"] for r in rows), ["Indiana Pacers", "Philadelphia 76ers"])
        self.assertEqual({r["d"] for r in rows}, {"2026-04-10"})


# ---------------------------------------------------------------------------
# 3. Fetching: polite, ESPN only, skips what we hold
# ---------------------------------------------------------------------------
class IngestTest(TempDbCase):

    def http(self):
        return FakeHttp(scoreboards={"20260410": fixture("scoreboard_20260410.json")},
                        summaries={"401811030": fixture("summary_401811030.json")})

    def test_ingest_fetches_only_missing_games(self):
        add_box(self.con, "0022501176", "2026-04-10", NYK, TOR, _line(110), _line(100))  # TOR @ NY held
        http = self.http()
        res = eb.ingest_dates(self.con, [date(2026, 4, 10)], eb.EspnClient(min_gap=0, http_get=http))
        self.assertEqual((res["final_events"], res["in_box_scores"], len(res["stored"])), (2, 1, 1))
        self.assertEqual(len(http.calls), 2)                  # one scoreboard, one summary
        # A second run asks for nothing but the scoreboard.
        http2 = self.http()
        res2 = eb.ingest_dates(self.con, [date(2026, 4, 10)], eb.EspnClient(min_gap=0, http_get=http2))
        self.assertEqual((res2["already_from_espn"], len(http2.calls)), (1, 1))

    def test_client_refuses_any_host_but_espn(self):
        c = eb.EspnClient(min_gap=0, http_get=self.http())
        for url in ("https://stats.nba.com/stats/leaguegamelog", "https://cdn.nba.com/static/json/x.json"):
            with self.assertRaises(eb.EspnUnavailable):
                c.get_json(url, {})

    def test_request_ceiling(self):
        c = eb.EspnClient(max_requests=1, min_gap=0, http_get=self.http())
        c.scoreboard(date(2026, 4, 10))
        with self.assertRaises(eb.EspnUnavailable):
            c.summary("401811030")

    def test_unreadable_espn_is_an_error_not_an_empty_day(self):
        c = eb.EspnClient(min_gap=0, http_get=lambda *a, **k: _Resp({}, 503))
        with self.assertRaises(eb.EspnUnavailable):
            eb.ingest_dates(self.con, [date(2026, 4, 10)], c)


# ---------------------------------------------------------------------------
# 4. The snapshot builder
# ---------------------------------------------------------------------------
class BuilderArithmeticTest(unittest.TestCase):

    def frame(self):
        rows = []
        # A's margins in its first four: +11, -21, -1, -10 -> -21 / 4 = -5.25
        games = [("g1", "2025-10-22", "A", "B", 111, 100), ("g2", "2025-10-24", "B", "A", 121, 100),
                 ("g3", "2025-10-26", "A", "C", 90, 91), ("g4", "2025-10-28", "C", "A", 100, 90),
                 ("g5", "2025-10-30", "A", "B", 80, 80 + 1), ("p1", "2026-04-20", "A", "B", 150, 50)]
        ids = {"A": 1, "B": 2, "C": 3}
        for gid, d, h, a, hp, ap in games:
            for tid, opp, pts, opp_pts, home in ((h, a, hp, ap, True), (a, h, ap, hp, False)):
                rows.append(dict(game_id=gid, season="2025-26",
                                 season_type="Playoffs" if gid.startswith("p") else "Regular Season",
                                 game_date=d, team_id=ids[tid], team_name=tid, is_home=home, MIN=240.0,
                                 FGM=40, FGA=80, FG3M=10, FG3A=30, FTM=10, FTA=20, OREB=10, DREB=30, REB=40,
                                 AST=20, TOV=12, STL=7, BLK=5, BLKA=5, PF=20, PFD=20, PTS=pts, OPP_PTS=opp_pts))
        tg = pd.DataFrame(rows)
        tg["W"] = (tg.PTS > tg.OPP_PTS).astype(int)
        tg["L"] = 1 - tg.W
        tg["PLUS_MINUS"] = tg.PTS - tg.OPP_PTS
        return tg

    def test_cutoff_is_strictly_before_and_regular_season_only(self):
        tg = self.frame()
        snap = ts.build_snapshot(tg, "2025-26", "2025-10-28")        # g1-g3 only
        self.assertEqual(int(snap.loc[1, "GP"]), 3)
        full = ts.build_snapshot(tg, "2025-26", "2026-06-01")        # playoffs never count
        self.assertEqual(int(full.loc[1, "GP"]), 5)
        self.assertIsNone(ts.build_snapshot(tg, "2025-26", "2025-10-22"))

    def test_rounding_is_half_away_from_zero(self):
        tg = self.frame()
        snap = ts.build_snapshot(tg, "2025-26", "2025-10-30")
        self.assertEqual(snap.loc[1, "PLUS_MINUS"], -5.3)              # nba.com's rounding; half-up says -5.2
        self.assertEqual(float(ts.round_half_away(-5.25, 1)), -5.3)
        self.assertEqual(float(ts.round_half_away(5.25, 1)), 5.3)

    def test_ranks_share_the_best_and_run_the_right_way(self):
        snap = ts.build_snapshot(self.frame(), "2025-26", "2026-06-01")
        self.assertEqual(snap.loc[:, "FGM_RANK"].tolist(), [1, 1, 1])   # all tied
        worst_l = snap.L.idxmax()
        self.assertEqual(int(snap.loc[worst_l, "L_RANK"]), 3)          # most losses ranks last
        self.assertEqual(snap.dtypes["GP"].kind, "i")

    def test_clock_rule(self):
        self.assertEqual(ts._clock_minutes("240:00"), 240.0)
        self.assertLess(ts._clock_minutes("239:60"), 240.0)
        self.assertGreater(ts._clock_minutes("239:60"), 239.99)
        self.assertAlmostEqual(ts._clock_minutes("265:30"), 265.5)


class RefreshTest(TempDbCase):

    def setUp(self):
        super().setUp()
        self.contract = add_reference_snapshot(self.con)
        add_box(self.con, "0022501150", "2026-04-08", IND, PHI, _line(120), _line(100))
        add_box(self.con, "0022501160", "2026-04-09", DET, NYK, _line(99), _line(101), clock="239:60")

    def test_refresh_writes_nba_shape_and_provenance(self):
        eb.store_game(self.con, fixture("summary_401811030.json"))     # IND-PHI 04-10 from ESPN
        name = ts.refresh(as_of=date(2026, 4, 11), season="2025-26", db_path=self.db)
        self.assertEqual(name, "2026-04-11")
        cols = [(r[1], r[2]) for r in self.con.execute('PRAGMA table_info("2026-04-11")')]
        self.assertEqual(cols, self.contract)
        df = pd.read_sql_query('SELECT * FROM "2026-04-11"', self.con)
        ind = df[df.TEAM_ID == IND].iloc[0]
        self.assertEqual((ind.GP, ind.W, ind.L, ind.Date, ind.TEAM_NAME), (2, 1, 1, "2026-04-11", "Indiana Pacers"))
        self.assertEqual(ind.PTS, 107.0)                               # (120 + 94) / 2
        self.assertEqual(ind.TOV, 17.0)                                # (13 + 21) / 2
        src = ts.snapshot_source(self.con, "2026-04-11")
        self.assertEqual((src["source"], src["n_games"], src["n_games_espn"], src["games_through"]),
                         ("archive rebuild", 3, 1, "2026-04-10"))
        self.assertEqual(df["index"].tolist(), list(range(len(df))))
        self.assertEqual(df.TEAM_NAME.tolist(), sorted(df.TEAM_NAME))

    def test_never_overwrites_an_nba_com_table(self):
        add_reference_snapshot(self.con, "2026-04-11")
        ts.refresh(as_of=date(2026, 4, 11), season="2025-26", db_path=self.db)
        n = self.con.execute('SELECT COUNT(*) FROM "2026-04-11"').fetchone()[0]
        self.assertEqual(n, 1)                                         # untouched
        self.assertIn("nba.com", ts.snapshot_source(self.con, "2026-04-11")["source"])

    def test_replaces_its_own_earlier_rebuild(self):
        ts.refresh(as_of=date(2026, 4, 11), season="2025-26", db_path=self.db)
        eb.store_game(self.con, fixture("summary_401811030.json"))
        ts.refresh(as_of=date(2026, 4, 11), season="2025-26", db_path=self.db)
        self.assertEqual(ts.snapshot_source(self.con, "2026-04-11")["n_games_espn"], 1)

    def test_nothing_played_writes_nothing(self):
        self.assertIsNone(ts.refresh(as_of=date(2026, 4, 8), season="2025-26", db_path=self.db))
        self.assertIsNone(ts.refresh(as_of=date(2026, 10, 21), season="2026-27", db_path=self.db))

    def test_clock_rank_rule_reaches_the_table(self):
        name = ts.refresh(as_of=date(2026, 4, 10), season="2025-26", db_path=self.db)
        df = pd.read_sql_query(f'SELECT TEAM_ID, MIN, MIN_RANK FROM "{name}"', self.con).set_index("TEAM_ID")
        self.assertEqual(df.loc[DET, "MIN"], 48.0)
        self.assertEqual(int(df.loc[IND, "MIN_RANK"]), 1)
        self.assertEqual(int(df.loc[DET, "MIN_RANK"]), 3)             # '239:60' sorts below 240


@unittest.skipUnless(os.path.exists(REAL_DB), "needs the real archive")
class RealArchiveTest(unittest.TestCase):
    """Read-only against Data/TeamData.sqlite: the rebuild reproduces stored
    nba.com snapshots (measured over all 4,015 non-empty tables 2026-09-28:
    99.966% of cells; these pin a few, and the known residue)."""

    def cmp(self, table):
        con = sqlite3.connect("file:" + REAL_DB.replace("\\", "/") + "?mode=ro", uri=True)
        try:
            if not con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                self.skipTest(f"no stored snapshot {table}")
        finally:
            con.close()
        return ts.compare(table)

    def test_exact_tables(self):
        for table in ("2024-04-29", "2015-01-15", "2019-02-01", "2008-12-01"):
            with self.subTest(table=table):
                r = self.cmp(table)
                self.assertTrue(r["team_sets_equal"])
                self.assertEqual(r["exact"], r["cells"], r["mismatches"][:5])

    def test_2025_26_final_table_residue_is_known(self):
        # nba.com's dashboard total differs from the sum of nba.com's own box
        # scores by one rebound (PHI DREB 2612 vs <=2611) and, through a tie,
        # one rank. Nothing else in 1,560 cells.
        r = self.cmp("2026-09-27")
        got = {(m["team_id"], m["column"]) for m in r["mismatches"]}
        self.assertEqual(got, {(PHI, "DREB"), (1610612758, "BLKA_RANK")})


# ---------------------------------------------------------------------------
# 5. The daily job: fallback only on failure, refusal path, health check
# ---------------------------------------------------------------------------
class _FixedDate(date):
    fixed = date(2027, 1, 15)

    @classmethod
    def today(cls):
        return cls.fixed


class DailyJobTest(unittest.TestCase):

    def run_main(self, backfill_ok=True, stats_ok=True, espn=("covered", 3), rebuilt="2027-01-15"):
        _FixedDate.fixed = date(2027, 1, 15)
        with mock.patch.object(daily_update, "date", _FixedDate), \
             mock.patch.object(daily_update, "run_backfill", return_value=backfill_ok), \
             mock.patch.object(daily_update, "espn_box_score_fallback", return_value=espn) as fb, \
             mock.patch.object(daily_update, "refresh_play_by_play", return_value=True), \
             mock.patch.object(daily_update, "refresh_team_stats_snapshot", return_value=stats_ok), \
             mock.patch.object(daily_update, "rebuild_team_stats_from_archive", return_value=rebuilt) as rb, \
             mock.patch.object(daily_update, "grade_logged_predictions", return_value=True), \
             mock.patch.object(daily_update, "log_todays_predictions", return_value="logged"), \
             mock.patch.object(daily_update, "commit_logged_picks", return_value="nothing"), \
             mock.patch.object(daily_update, "publish_commitments", return_value="skipped"), \
             mock.patch.object(daily_update, "snapshot_odds_board", return_value="ok"), \
             mock.patch.object(daily_update, "refresh_periodic_ingests", return_value=True), \
             mock.patch.object(daily_update, "publish_ledger", return_value="skipped"), \
             mock.patch.object(daily_update, "run_integrity_audit", return_value=0), \
             mock.patch.object(daily_update, "run_preflight", return_value=0), \
             mock.patch.object(daily_update.logger, "info") as info, \
             mock.patch.object(daily_update.logger, "error") as err:
            code = daily_update.main()
        return code, fb, rb, info, err

    def test_nba_com_working_runs_no_fallback(self):
        code, fb, rb, info, _ = self.run_main()
        self.assertEqual(code, 0)
        fb.assert_not_called()
        rb.assert_not_called()
        final = info.call_args_list[-1][0]
        self.assertNotIn("team stats", final[0] % final[1:])

    def test_failed_backfill_calls_espn_and_stays_red(self):
        code, fb, _, _, err = self.run_main(backfill_ok=False)
        fb.assert_called_once()
        self.assertEqual(code, 1)
        self.assertIn("ESPN fallback covered: 3 game(s)", err.call_args[0][1])

    def test_refused_dashboard_uses_the_archive_rebuild(self):
        code, _, rb, info, _ = self.run_main(stats_ok=False)
        rb.assert_called_once()
        self.assertEqual(code, 0)
        final = info.call_args_list[-1][0]
        self.assertIn("team stats: archive rebuild", final[0] % final[1:])

    def test_rebuild_on_an_incomplete_archive_is_a_failure(self):
        code, _, _, _, err = self.run_main(backfill_ok=False, stats_ok=False, espn=("unavailable", 0))
        self.assertEqual(code, 1)
        self.assertIn("team-stats refresh", err.call_args[0][1])

    def test_rebuild_after_espn_covered_is_fine_for_team_stats(self):
        code, _, _, _, err = self.run_main(backfill_ok=False, stats_ok=False, espn=("covered", 5))
        self.assertEqual(code, 1)                                     # the nba.com backfill still failed
        self.assertNotIn("team-stats refresh", err.call_args[0][1])


class FallbackStepTest(TempDbCase):

    def test_fallback_window_and_result(self):
        add_box(self.con, "0022501150", "2026-04-08", IND, PHI, _line(120), _line(100))
        add_box(self.con, "0022501176", "2026-04-10", NYK, TOR, _line(110), _line(100))  # TOR @ NY held
        http = FakeHttp(scoreboards={"20260410": fixture("scoreboard_20260410.json")},
                        summaries={"401811030": fixture("summary_401811030.json")})
        with mock.patch.object(daily_update, "_nba_games_expected", return_value=True):
            status, n = daily_update.espn_box_score_fallback(
                "2025-26", today=date(2026, 4, 11), client=eb.EspnClient(min_gap=0, http_get=http),
                db_path=self.db)
        self.assertEqual((status, n), ("covered", 1))
        days = [c[1]["dates"] for c in http.calls if c[0] == eb.SCOREBOARD_URL]
        self.assertEqual(days[0], "20260401")                          # ten days back
        self.assertEqual(days[-1], "20260410")                         # through yesterday

    def test_offseason_skips(self):
        with mock.patch.object(daily_update, "_nba_games_expected", return_value=False):
            self.assertEqual(daily_update.espn_box_score_fallback("2025-26", db_path=self.db), ("skipped", 0))

    def test_espn_down_is_unavailable_not_covered(self):
        with mock.patch.object(daily_update, "_nba_games_expected", return_value=True):
            status, _ = daily_update.espn_box_score_fallback(
                "2025-26", today=date(2026, 4, 11),
                client=eb.EspnClient(min_gap=0, http_get=lambda *a, **k: _Resp({}, 503)), db_path=self.db)
        self.assertEqual(status, "unavailable")


class RefusalPathTest(TempDbCase):
    """refresh_team_stats refused by nba.com (switch off, or the breaker
    open) -> the archive rebuild writes the table. No network: requests is
    booby-trapped, so a request that slipped past the guard would fail the test."""

    def setUp(self):
        super().setUp()
        add_reference_snapshot(self.con)
        add_box(self.con, "0022501150", "2026-04-08", IND, PHI, _line(120), _line(100))
        self.trap = mock.patch("requests.Session.request", side_effect=AssertionError("network!"))
        self.trap2 = mock.patch("requests.get", side_effect=AssertionError("network!"))
        self.trap.start(); self.trap2.start()

    def tearDown(self):
        self.trap.stop(); self.trap2.stop()
        super().tearDown()

    def _refresh_nba(self):
        import refresh_team_stats
        with mock.patch.object(refresh_team_stats, "DB_PATH", self.db):
            return refresh_team_stats.refresh(as_of=date(2026, 4, 10), season="2025-26")

    def test_switch_off(self):
        from src.Utils.nba_stats_client import LiveFetchDisabled
        with mock.patch.dict(os.environ, {"NBA_STATS_LIVE": "off", "NBA_MIRROR_DB": self.db}):
            with self.assertRaises(LiveFetchDisabled):
                self._refresh_nba()
        self.assertEqual(ts.refresh(as_of=date(2026, 4, 10), season="2025-26", db_path=self.db), "2026-04-10")

    def test_breaker_open(self):
        from datetime import datetime, timedelta, timezone
        from src.Utils import nba_outbound_guard as guard
        from src.Utils.nba_stats_client import OutboundRefused
        state = os.path.join(self.dir, "outbound.sqlite")
        with mock.patch.dict(os.environ, {"NBA_OUTBOUND_DB": state, "NBA_STATS_LIVE": "on",
                                          "NBA_MIRROR_DB": self.db}):
            c = guard._connect()
            c.execute("UPDATE breaker SET failure_streak = 10, open_until = ? WHERE id = 1",
                      ((datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),))
            c.close()
            with mock.patch("src.Utils.nba_stats_client.time.sleep"):
                with self.assertRaises(OutboundRefused):
                    self._refresh_nba()
        self.assertEqual(ts.refresh(as_of=date(2026, 4, 10), season="2025-26", db_path=self.db), "2026-04-10")


class HealthCheckTest(TempDbCase):
    NOW = __import__("datetime").datetime(2026, 4, 12, 15, 0, tzinfo=__import__("datetime").timezone.utc)

    def test_ok_without_stand_ins(self):
        add_box(self.con, "0022501150", "2026-04-08", IND, PHI, _line(120), _line(100))
        r = job_health.check_box_score_sources(self.NOW, db=self.db)
        self.assertEqual(r["status"], job_health.OK)
        self.assertIn("all 1 game(s) are nba.com", r["summary"])

    def test_warns_and_names_espn_games(self):
        eb.store_game(self.con, fixture("summary_401811030.json"))
        add_reference_snapshot(self.con)
        add_box(self.con, "0022501150", "2026-04-08", IND, PHI, _line(120), _line(100))
        ts.refresh(as_of=date(2026, 4, 11), season="2025-26", db_path=self.db)
        r = job_health.check_box_score_sources(self.NOW, db=self.db)
        self.assertEqual(r["status"], job_health.WARN)
        self.assertIn("0022501175", r["summary"])
        self.assertIn("rebuilt from the archive", r["summary"])
        self.assertEqual(r["evidence"]["snapshot_espn_games"], 1)


if __name__ == "__main__":
    unittest.main()
