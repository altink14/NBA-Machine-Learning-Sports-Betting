"""A rookie traded mid-season is one row, not one per team (found 2026-09-23).

player_season_totals holds a row per team, and /api/stats/rookies joined them
directly, so Walter Clayton Jr. (UTA then MEM, 2025-26) was listed twice at
pick 18 and React warned about the duplicate key. Temp database only.
"""
import sqlite3
import unittest
from unittest import mock

import main_api


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _db():
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
    c.executemany("INSERT INTO team_metadata VALUES (?, ?)", [(1, "UTA"), (2, "MEM"), (3, "BOS")])
    c.execute("INSERT INTO draft_history VALUES (10, 'Traded Guy', 2025, 1, 18, 'WAS', 'Florida')")
    c.execute("INSERT INTO draft_history VALUES (11, 'Stayed Put', 2025, 1, 19, 'BOS', 'Duke')")
    rs = "Regular Season"
    # MEM is team 2 and came second in time; UTA is team 1 and came first.
    c.execute("INSERT INTO player_season_totals VALUES (10,'2025-26',?,2,24,6,599,75,205,.366,31,101,.307,51,59,.864,51,136,20,7,59,232)", (rs,))
    c.execute("INSERT INTO player_season_totals VALUES (10,'2025-26',?,1,45,0,811,102,256,.398,40,130,.308,29,31,.935,88,143,24,12,57,306)", (rs,))
    c.execute("INSERT INTO player_season_totals VALUES (11,'2025-26',?,3,80,80,2400,400,800,.5,100,250,.4,150,200,.75,300,200,50,30,100,1050)", (rs,))
    c.executemany("INSERT INTO box_scores VALUES (?, '2025-26', ?)", [("g1", rs), ("g2", rs)])
    c.execute("INSERT INTO player_game_log VALUES (10, 1, 'g1', '2025-10-22')")
    c.execute("INSERT INTO player_game_log VALUES (10, 2, 'g2', '2026-02-10')")
    return c


class RookiesTradedTest(unittest.TestCase):

    def fetch(self):
        c = _db()
        self.addCleanup(c.close)
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)), \
             mock.patch.object(main_api, "_ensure_draft_history", lambda conn: None):
            rows = main_api.get_rookies("2025-26")["rookies"]
        return {r["player_id"]: r for r in rows}, rows

    def test_one_row_per_rookie_with_summed_totals(self):
        by_id, rows = self.fetch()
        self.assertEqual(len(rows), 2)
        t = by_id[10]
        self.assertEqual((t["gp"], t["gs"], t["pts"], t["ast"]), (69, 6, 538, 279))

    def test_percentages_come_from_makes_and_attempts(self):
        t = self.fetch()[0][10]
        self.assertAlmostEqual(t["fg_pct"], 177 / 461)
        self.assertAlmostEqual(t["ft_pct"], 80 / 90)

    def test_teams_listed_in_the_order_he_played_for_them(self):
        by_id = self.fetch()[0]
        self.assertEqual(by_id[10]["team_abbr"], "UTA/MEM")
        self.assertEqual(by_id[11]["team_abbr"], "BOS")
        self.assertNotIn("n_teams", by_id[10])


if __name__ == "__main__":
    unittest.main()
