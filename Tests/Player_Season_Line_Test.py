"""A traded player's season is one line; unknown advanced values are None, never 0.0.

Found 2026-09-23: the player page took an arbitrary stint ("LIMIT 1", no
order), and player_season_advanced held 0.0 placeholders for a traded
player's earlier stints (nba.com reports those figures once per season,
under his latest team). Temp database only.
"""
import sqlite3
import unittest
from unittest import mock

import main_api

COLS = ["gp", "gs", "min", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct",
        "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]
ADV = ["ts_pct", "usg_pct", "off_rating", "def_rating", "net_rating", "ast_pct", "reb_pct", "efg_pct", "tov_pct", "pace"]


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
    c.execute("CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, season_type TEXT, team_id INTEGER, "
              + ", ".join(f"{k} REAL" for k in COLS) + ")")
    c.execute("CREATE TABLE player_season_advanced (id INTEGER PRIMARY KEY, player_id INTEGER, season TEXT, "
              "season_type TEXT, team_id INTEGER, " + ", ".join(f"{k} REAL" for k in ADV) + ")")
    c.execute("CREATE TABLE player_season_stats (player_id INTEGER, season TEXT, season_type TEXT, team_id INTEGER, "
              "gp INTEGER, usg_pct REAL, off_rating REAL, def_rating REAL, net_rating REAL, ast_pct REAL, reb_pct REAL, pace REAL)")
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT)")
    c.execute("CREATE TABLE player_game_log (player_id INTEGER, team_id INTEGER, game_id TEXT, game_date TEXT)")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT)")
    c.executemany("INSERT INTO team_metadata VALUES (?, ?)", [(1, "WAS"), (2, "DAL"), (3, "BOS")])
    rs = "Regular Season"

    def stint(pid, team, gp, pts, fgm, fga):
        row = {k: 0 for k in COLS}
        row.update(gp=gp, pts=pts, fgm=fgm, fga=fga, min=gp * 20)
        c.execute("INSERT INTO player_season_totals VALUES (?, 'S', ?, ?, " + ", ".join("?" for _ in COLS) + ")",
                  [pid, rs, team] + [row[k] for k in COLS])
    # traded: WAS then DAL. The old backfill left WAS at 0.0 and gave DAL the season's figures.
    stint(9, 1, 45, 490, 200, 290)
    stint(9, 2, 29, 324, 130, 170)
    c.execute("INSERT INTO player_season_advanced (player_id, season, season_type, team_id, usg_pct, off_rating, pace) "
              "VALUES (9, 'S', ?, 1, 0, 0, 0)", (rs,))
    c.execute("INSERT INTO player_season_advanced (player_id, season, season_type, team_id, usg_pct, off_rating, pace) "
              "VALUES (9, 'S', ?, 2, 0.145, 113.9, 103.3)", (rs,))
    c.execute("INSERT INTO player_season_stats VALUES (9, 'S', ?, 2, 74, 0.145, 113.9, 116.0, -2.2, 0.08, 0.15, 103.3)", (rs,))
    c.executemany("INSERT INTO box_scores VALUES (?, 'S', ?)", [("g1", rs), ("g2", rs)])
    c.execute("INSERT INTO player_game_log VALUES (9, 1, 'g1', '2023-10-25')")
    c.execute("INSERT INTO player_game_log VALUES (9, 2, 'g2', '2024-02-10')")
    # one team, but the row was never filled (pace 0)
    stint(4, 3, 60, 600, 250, 500)
    c.execute("INSERT INTO player_season_advanced (player_id, season, season_type, team_id, ts_pct, usg_pct, off_rating, pace) "
              "VALUES (4, 'S', ?, 3, 0.55, 0, 0, 0)", (rs,))
    return c


class PlayerSeasonLineTest(unittest.TestCase):

    def setUp(self):
        self.c = _db()
        self.addCleanup(self.c.close)

    def test_traded_season_is_summed_with_whole_season_rates(self):
        totals, adv, n = main_api._player_season_line(self.c, 9, "S")
        self.assertEqual(n, 2)
        self.assertEqual((totals["gp"], totals["pts"], totals["team_abbr"]), (74, 814, "WAS/DAL"))
        self.assertAlmostEqual(totals["fg_pct"], 330 / 460)
        self.assertEqual((adv["usg_pct"], adv["off_rating"], adv["pace"]), (0.145, 113.9, 103.3))
        self.assertAlmostEqual(adv["ts_pct"], 814 / (2 * 460))

    def test_unfilled_placeholder_is_none_not_zero(self):
        _, adv, n = main_api._player_season_line(self.c, 4, "S")
        self.assertEqual(n, 1)
        self.assertIsNone(adv["usg_pct"])
        self.assertIsNone(adv["pace"])
        self.assertEqual(adv["ts_pct"], 0.55)

    def test_career_has_a_tot_row_then_stints_without_invented_rates(self):
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(self.c)):
            rows = [r for r in main_api.get_player_career(9) if r["season"] == "S"]
        self.assertEqual([r["team_abbr"] for r in rows], ["TOT", "WAS", "DAL"])
        self.assertTrue(rows[0]["is_total"])
        self.assertEqual((rows[0]["gp"], rows[0]["usg_pct"]), (74, 0.145))
        for stint in rows[1:]:
            self.assertIsNone(stint["usg_pct"])
            self.assertIsNone(stint["off_rating"])
            self.assertIsNotNone(stint["ts_pct"])


if __name__ == "__main__":
    unittest.main()
