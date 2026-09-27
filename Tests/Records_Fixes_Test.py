"""Records pages: box-score line scores, standings, the market's W-L, splits months.

Nav audit 2026-09-24, fixed 2026-09-27:
- bug 2: since nba.com's v2 summary feed died (2025-04-10) its game_line_scores
  rows are NULL shells, and the endpoint turned NULL into 0, so ~1,200
  2025-26 box scores showed a 0-0-0-0 line score, a "0 h 00 min" game and
  "Inactive: none listed". NULL is unknown: the quarters come from the
  play-by-play when they add up to the box score, and the rest is hidden.
- bug 8: standings tagged PLAYOFFS / PLAY-IN by rank in every season, and
  took the conference from today's alignment (2003-04 New Orleans in the West).
- bug 9: Against the Market counted straight-up W-L only over games with a
  line (2022-23 76ers 54-27; they went 54-28).
- bug 20: a player's month splits sorted alphabetically.
Synthetic databases, plus checks against the real archive when present.
"""
import json
import os
import sqlite3
import unittest
from unittest import mock

import main_api
from src.Utils import Market as market

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_DB = os.path.join(REPO, "Data", "TeamData.sqlite")
ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _game_db(line_null=True, pbp_final=(150, 129), box_final=(150, 129)):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE box_scores (game_id TEXT, home_team_id INTEGER, away_team_id INTEGER, traditional_json TEXT)")
    c.execute("CREATE TABLE game_line_scores (game_id TEXT, team_id INTEGER, team_abbr TEXT, q1 INTEGER, q2 INTEGER, "
              "q3 INTEGER, q4 INTEGER, " + ", ".join(f"ot{i} INTEGER" for i in range(1, 11)) + ", pts INTEGER)")
    c.execute("CREATE TABLE pbp_events (game_id TEXT, action_id INTEGER, period INTEGER, score_home INTEGER, score_away INTEGER)")
    c.execute("CREATE TABLE game_info (game_id TEXT, attendance INTEGER, game_time TEXT, natl_tv TEXT, "
              "home_team_id INTEGER, visitor_team_id INTEGER, ingested_at TEXT)")
    c.execute("CREATE TABLE game_inactives (game_id TEXT, team_id INTEGER, team_abbr TEXT, player_id INTEGER, "
              "first_name TEXT, last_name TEXT, jersey_num TEXT)")
    trad = {"boxScoreTraditional": {"homeTeam": {"statistics": {"points": box_final[0]}},
                                    "awayTeam": {"statistics": {"points": box_final[1]}}}}
    c.execute("INSERT INTO box_scores VALUES ('G', 1, 2, ?)", (json.dumps(trad),))
    for tid in (1, 2):
        if line_null:
            c.execute("INSERT INTO game_line_scores (game_id, team_id) VALUES ('G', ?)", (tid,))
    # MIA 40/36/37/37 v WAS 29/33/35/32; a stale lower score mid-Q4 must not matter
    h, a, n = 0, 0, 0
    for period, (dh, da) in enumerate([(40, 29), (36, 33), (37, 35), (37, 32)], start=1):
        for step in (0.5, 1.0):
            n += 1
            c.execute("INSERT INTO pbp_events VALUES ('G', ?, ?, ?, ?)",
                      (n, period, h + round(dh * step), a + round(da * step)))
        h, a = h + dh, a + da
    c.execute("INSERT INTO pbp_events VALUES ('G', 99, 4, 60, 55)")
    if pbp_final != (150, 129):
        c.execute("UPDATE pbp_events SET score_home = ? WHERE action_id = 8", (pbp_final[0],))
    c.execute("INSERT INTO game_info VALUES ('G', NULL, '0:00', NULL, 1, 2, 'x')")
    return c


class LineScoreTest(unittest.TestCase):
    def setUp(self):
        main_api._market_cache.clear()
        self.addCleanup(main_api._market_cache.clear)

    def _call(self, conn, fn, gid="G"):
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(conn)):
            return fn(gid)

    def test_null_official_line_falls_back_to_the_play_by_play(self):
        r = self._call(_game_db(), main_api.get_game_line_score)
        self.assertTrue(r["available"])
        self.assertEqual(r["source"], "play-by-play period-end scores")
        self.assertEqual([p["home"] for p in r["periods"]], [40, 36, 37, 37])
        self.assertEqual([p["away"] for p in r["periods"]], [29, 33, 35, 32])
        self.assertEqual(r["final"], {"home": 150, "away": 129})

    def test_play_by_play_that_disagrees_with_the_box_score_is_not_shown(self):
        r = self._call(_game_db(box_final=(151, 129)), main_api.get_game_line_score)
        self.assertFalse(r["available"])

    def test_empty_summary_is_unknown_not_zero(self):
        r = self._call(_game_db(), main_api.get_game_info)
        self.assertTrue(r["available"])
        self.assertIsNone(r["game_time"])
        self.assertFalse(r["inactives_known"])


class SeasonConferenceTest(unittest.TestCase):
    def test_a_team_is_filed_where_its_schedule_puts_it(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE game_results (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT)")
        east, west = [1, 2, 3, 4], [5, 6, 7, 8]
        stored = {t: "East" for t in east} | {t: "West" for t in west}
        stored[4] = "West"            # today's alignment files team 4 in the West
        gid = 0

        def play(x, y, n):
            nonlocal gid
            for _ in range(n):
                gid += 1
                c.executemany("INSERT INTO game_results VALUES (?, ?, 'S', 'Regular Season')", [(gid, x), (gid, y)])
        for group in (east, west):
            for i, x in enumerate(group):
                for y in group[i + 1:]:
                    play(x, y, 4)
        for x in east:
            for y in west:
                play(x, y, 2)
        main_api._season_conference_cache.pop("S", None)
        got = main_api._season_conferences(c, "S", stored)
        main_api._season_conference_cache.pop("S", None)
        self.assertEqual(got[4], "East")
        self.assertEqual({t for t, conf in got.items() if conf == "East"}, set(east))


@unittest.skipUnless(os.path.exists(TEAM_DB), "archive database not present")
class StandingsArchiveTest(unittest.TestCase):
    def test_2003_04_hornets_were_in_the_east(self):
        rows = main_api.get_stats_standings(season="2003-04")
        nop = next(r for r in rows if r["abbreviation"] == "NOP")
        self.assertEqual(nop["conference"], "East")
        self.assertIsNone(nop["division"])
        self.assertEqual((nop["wins"], nop["losses"]), (41, 41))
        self.assertEqual(sum(r["conference"] == "East" for r in rows), 15)

    def test_postseason_tags_come_from_the_games_played(self):
        before = main_api.get_stats_standings(season="2015-16")
        self.assertTrue(all(r["postseason_known"] for r in before))
        self.assertEqual(sum(r["postseason"] == "playoffs" for r in before), 16)
        self.assertFalse(any(r["postseason"] and r["postseason"].startswith("play_in") for r in before))
        bubble = main_api.get_stats_standings(season="2019-20")
        self.assertEqual(sorted(r["abbreviation"] for r in bubble if r["postseason"] == "play_in"), ["MEM"])
        self.assertEqual([r["abbreviation"] for r in bubble if r["postseason"] == "play_in_to_playoffs"], ["POR"])
        modern = main_api.get_stats_standings(season="2024-25")
        self.assertEqual(sum(r["postseason"] == "play_in_to_playoffs" for r in modern), 4)
        self.assertEqual(sum(r["postseason"] == "play_in" for r in modern), 4)


@unittest.skipUnless(os.path.exists(TEAM_DB) and os.path.exists(ODDS_DB), "archive databases not present")
class MarketRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        market._lines = None
        cls.team = sqlite3.connect(TEAM_DB)
        cls.odds = sqlite3.connect(ODDS_DB)

    @classmethod
    def tearDownClass(cls):
        cls.team.close()
        cls.odds.close()
        market._lines = None

    def test_straight_up_is_the_full_record(self):
        teams = {t["abbr"]: t for t in market.season_market(self.team, self.odds, "2022-23")["teams"]}
        self.assertEqual(teams["PHI"]["su"], [54, 28])
        self.assertEqual(teams["BOS"]["su"], [57, 25])
        self.assertEqual(teams["PHI"]["su_lined"], [54, 27])   # the graded games keep their own n
        self.assertEqual(teams["PHI"]["graded"], 81)

    def test_2023_24_is_graded(self):
        data = market.season_market(self.team, self.odds, "2023-24")
        self.assertTrue(data["available"])
        self.assertGreater(data["league"]["graded"], 1100)
        covers, fails, _ = data["league"]["home_ats"]
        self.assertTrue(0.44 < covers / (covers + fails) < 0.56)


class SplitMonthsTest(unittest.TestCase):
    def test_months_run_in_season_order(self):
        months = ["April", "December", "February", "January", "March", "November", "October", "July"]
        self.assertEqual(sorted(months, key=main_api._season_month_rank),
                         ["October", "November", "December", "January", "February", "March", "April", "July"])


if __name__ == "__main__":
    unittest.main()
