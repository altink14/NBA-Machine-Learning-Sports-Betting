"""Draft Board, Prospect Pipeline and Rookie Class fixes (2026-09-28).

- Pipeline: nba.com spells schools formally ("California-Los Angeles"); the
  page shows the common name, grouped after the rename. A blank organization
  was counted as one more "program"; it is now left out and counted as
  no_school.
- Draft class: "traded to - N of this class" counted only players whose bio
  was on file (2015 said 1). The endpoint reports known / unknown / moved.
- Rookies: each rookie's Pick Value band and that band's typical rookie year
  (a pick who never played counts as zero, the shown class left out), and the
  next class's opening night read from the schedule cache only.
Temp databases only, except where stated.
"""
import sqlite3
import unittest
from unittest import mock

import main_api
from src.Utils.school_names import common_school_name


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _draft_db():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE draft_history (person_id INTEGER, player_name TEXT, season INTEGER, "
              "round_number INTEGER, round_pick INTEGER, overall_pick INTEGER, team_id INTEGER, "
              "team_city TEXT, team_name TEXT, team_abbreviation TEXT, organization TEXT, "
              "organization_type TEXT, fetched_at TEXT)")
    c.execute("CREATE TABLE players (player_id INTEGER, full_name TEXT)")
    c.execute("CREATE TABLE player_bio (player_id INTEGER, position TEXT, height TEXT, weight TEXT, "
              "country TEXT, team_abbr TEXT)")
    rows = [
        (1, "A", 2015, 1, 1, 1, "California-Los Angeles", "College/University"),
        (2, "B", 2015, 1, 2, 2, "California-Los Angeles", "College/University"),
        (3, "C", 2015, 1, 3, 20, "Duke", "College/University"),
        (4, "D", 2015, 2, 1, 31, "", ""),
        (5, "E", 2015, 2, 2, 32, "", ""),
        (6, "F", 2015, 2, 3, 33, "Wisconsin-Milwaukee", "College/University"),
        (7, "G", 2014, 2, 3, 40, "Milwaukee", "College/University"),
    ]
    for pid, name, season, rnd, rp, op, org, typ in rows:
        c.execute("INSERT INTO draft_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (pid, name, season, rnd, rp, op, 1, "Minnesota", "Timberwolves", "MIN", org, typ, None))
    # Bios on file for two players only: one moved, one stayed.
    c.execute("INSERT INTO player_bio VALUES (1, 'C', '7-0', '248', 'USA', 'NYK')")
    c.execute("INSERT INTO player_bio VALUES (2, 'G', '6-4', '195', 'USA', 'MIN')")
    return c


class SchoolNamesTest(unittest.TestCase):

    def test_common_names_and_passthrough(self):
        self.assertEqual(common_school_name("California-Los Angeles"), "UCLA")
        self.assertEqual(common_school_name("Miami (FL)"), "Miami")
        self.assertEqual(common_school_name("Miami (OH)"), "Miami (OH)")
        self.assertEqual(common_school_name("Kentucky"), "Kentucky")

    def test_blank_is_none_not_a_school(self):
        self.assertIsNone(common_school_name(""))
        self.assertIsNone(common_school_name("   "))
        self.assertIsNone(common_school_name(None))


class DraftPagesTest(unittest.TestCase):

    def run_with(self, fn, *a, **kw):
        c = _draft_db()
        self.addCleanup(c.close)
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)), \
             mock.patch.object(main_api, "_ensure_draft_history", lambda conn: None):
            return fn(*a, **kw)

    def test_pipeline_uses_common_names_and_excludes_blank(self):
        d = self.run_with(main_api.get_draft_pipeline, since=2014, limit=40)
        orgs = {p["org"]: p for p in d["programs"]}
        self.assertIn("UCLA", orgs)
        self.assertEqual(orgs["UCLA"]["picks"], 2)
        self.assertEqual(orgs["UCLA"]["official_names"], ["California-Los Angeles"])
        # The two Milwaukee spellings are one school.
        self.assertEqual(orgs["Milwaukee"]["picks"], 2)
        self.assertNotIn("", orgs)
        self.assertNotIn(None, orgs)
        t = d["totals"]
        self.assertEqual((t["picks"], t["no_school"], t["picks_with_school"], t["orgs"]), (7, 2, 5, 3))
        self.assertNotIn("", [b["org_type"] for b in d["by_type"]])

    def test_unnumbered_early_picks_are_not_lottery_picks(self):
        # nba.com stores territorial and most 1949-56 picks as overall pick 0.
        # 0 <= 14 used to make every one a "lottery" pick and "#0" a best pick.
        def run():
            c = main_api.get_db_conn()
            c.execute("INSERT INTO draft_history VALUES (8, 'Territorial', 1960, 0, 0, 0, 1, 'X', 'Y', 'MIN', "
                      "'Duke', 'College/University', NULL)")
            return main_api.get_draft_pipeline(since=1950, limit=40)
        d = self.run_with(run)
        duke = next(p for p in d["programs"] if p["org"] == "Duke")
        self.assertEqual((duke["picks"], duke["numbered_picks"], duke["lottery"], duke["best_pick"]), (2, 1, 0, 20))
        self.assertEqual(duke["lottery_rate"], 0.0)
        self.assertEqual(d["totals"]["unnumbered_picks"], 1)

    def test_draft_class_reports_known_and_unknown_current_team(self):
        d = self.run_with(main_api.get_draft_class, 2015)
        self.assertEqual((d["current_team_known"], d["current_team_unknown"], d["moved_count"]), (2, 4, 1))
        by_id = {p["person_id"]: p for p in d["picks"]}
        self.assertEqual(by_id[1]["moved_to"], "NYK")
        self.assertEqual(by_id[1]["organization"], "UCLA")
        self.assertEqual(by_id[1]["organization_official"], "California-Los Angeles")
        self.assertIsNone(by_id[4]["organization"])


def _rookie_db():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE draft_history (person_id INTEGER, player_name TEXT, season INTEGER, "
              "round_number INTEGER, overall_pick INTEGER, team_abbreviation TEXT, organization TEXT)")
    c.execute("CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, season_type TEXT, "
              "team_id INTEGER, gp INTEGER, gs INTEGER, min REAL, fgm INTEGER, fga INTEGER, fg_pct REAL, "
              "fg3m INTEGER, fg3a INTEGER, fg3_pct REAL, ftm INTEGER, fta INTEGER, ft_pct REAL, "
              "reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pts INTEGER)")
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT)")
    c.execute("CREATE TABLE player_game_log (player_id INTEGER, team_id INTEGER, game_id TEXT, game_date TEXT)")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT)")
    rs = "Regular Season"

    def line(pid, season, gp, mins, pts):
        c.execute("INSERT INTO player_season_totals VALUES (?,?,?,1,?,0,?,0,0,NULL,0,0,NULL,0,0,NULL,0,0,0,0,0,?)",
                  (pid, season, rs, gp, mins, pts))

    # Two earlier pick-2 rookies (30 and 20 mpg) and one pick-50 who never played.
    c.execute("INSERT INTO draft_history VALUES (1, 'Old Two', 2023, 1, 2, 'BOS', 'Duke')")
    c.execute("INSERT INTO draft_history VALUES (2, 'Older Two', 2022, 1, 2, 'BOS', 'Duke')")
    c.execute("INSERT INTO draft_history VALUES (3, 'Never Played', 2023, 2, 50, 'BOS', '')")
    line(1, "2023-24", 10, 300, 150)
    line(2, "2022-23", 10, 200, 100)
    # The class being shown, whose own line must not move its baseline.
    c.execute("INSERT INTO draft_history VALUES (9, 'This Year', 2025, 1, 2, 'BOS', 'California-Los Angeles')")
    line(9, "2025-26", 10, 400, 400)
    return c


class RookieBaselineTest(unittest.TestCase):

    def fetch(self, cache=None):
        c = _rookie_db()
        self.addCleanup(c.close)
        c.execute("INSERT INTO draft_history VALUES (20, 'Next Class', 2026, 1, 1, 'WAS', 'Duke')")
        from src.Utils import nba_stats_client as nsc
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)), \
             mock.patch.object(main_api, "_ensure_draft_history", lambda conn: None), \
             mock.patch.object(main_api, "_schedule_season", lambda: "2026-27"), \
             mock.patch.object(nsc, "_read_cache", lambda path, ttl: cache), \
             mock.patch.object(nsc.NBAStatsClient, "_fetch", side_effect=AssertionError("no network")):
            return main_api.get_rookies("2025-26")

    def test_band_medians_count_a_dnp_as_zero_and_skip_own_class(self):
        d = self.fetch()
        bands = {b["label"]: b for b in d["slot_baselines"]["bands"]}
        # Picks 1-3: the two earlier pick-2 rookies only (30 and 20 mpg);
        # the shown class's 40 mpg is left out.
        self.assertEqual(bands["1-3"]["n_players"], 2)
        self.assertEqual(bands["1-3"]["median_mpg"], 30.0)
        self.assertEqual(bands["46-60"]["median_mpg"], 0.0)
        self.assertEqual(bands["46-60"]["share_played"], 0.0)
        r = next(x for x in d["rookies"] if x["player_id"] == 9)
        self.assertEqual(r["slot_band"], "1-3")
        self.assertEqual(r["organization"], "UCLA")

    def test_next_class_opening_night_from_cached_schedule(self):
        cache = {"leagueSchedule": {"gameDates": [{"games": [
            {"gameId": "0012600001", "gameDateEst": "2026-10-03T00:00:00Z"},
            {"gameId": "0022600002", "gameDateEst": "2026-10-21T00:00:00Z"},
            {"gameId": "0022600001", "gameDateEst": "2026-10-20T00:00:00Z"},
        ]}]}}
        nxt = self.fetch(cache)["next_class"]
        self.assertEqual((nxt["draft_year"], nxt["season"], nxt["picks"], nxt["opening_night"]),
                         (2026, "2026-27", 1, "2026-10-20"))

    def test_no_cached_schedule_means_unknown_date_not_a_guess(self):
        nxt = self.fetch(None)["next_class"]
        self.assertIsNone(nxt["opening_night"])


if __name__ == "__main__":
    unittest.main()
