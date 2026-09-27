"""Streak board: regular season and playoffs are separate sequences.

Found 2026-09-24: the board chained regular-season and playoff games into one
sequence, so a sub-10 playoff night ended LeBron's 10-point streak at 868
games (the league's record is 1,297 regular-season games) and playoff games
padded Curry's made-three streak from 157 to 196. Temp database only.
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
    c.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT)")
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT, full_name TEXT)")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT, game_date TEXT)")
    c.execute("CREATE TABLE game_results (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT, "
              "game_date TEXT, wl TEXT)")
    c.execute("CREATE TABLE player_game_log (game_id TEXT, player_id INTEGER, team_id INTEGER, game_date TEXT, "
              "pts INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, fg3m INTEGER)")
    c.executemany("INSERT INTO team_metadata VALUES (?, ?, ?)", [(1, "AAA", "Alpha"), (2, "BBB", "Beta")])
    c.execute("INSERT INTO players VALUES (7, 'Scorer')")
    return c


def _game(c, n, date, season_type, scorer_pts, alpha_wl):
    prefix = main_api.GAME_ID_PREFIX_BY_SEASON_TYPE[season_type]
    gid = f"{prefix}0000{n:03d}"
    c.execute("INSERT INTO box_scores VALUES (?, '2014-15', ?, ?)", (gid, season_type, date))
    c.execute("INSERT INTO game_results VALUES (?, 1, '2014-15', ?, ?, ?)", (gid, season_type, date, alpha_wl))
    c.execute("INSERT INTO game_results VALUES (?, 2, '2014-15', ?, ?, ?)",
              (gid, season_type, date, "L" if alpha_wl == "W" else "W"))
    c.execute("INSERT INTO player_game_log VALUES (?, 7, 1, ?, ?, 0, 0, 0, 0, 1)", (gid, date, scorer_pts))


class StreakBoardTest(unittest.TestCase):

    def setUp(self):
        main_api._streak_cache.clear()
        self.c = _db()
        # 14 regular-season games, all 10+ points and all wins, with a
        # 6-point playoff loss in the middle of them (as a season boundary is).
        for i in range(7):
            _game(self.c, i, f"2015-03-{10 + i:02d}", "Regular Season", 20, "W")
        _game(self.c, 50, "2015-05-01", "Playoffs", 6, "L")
        _game(self.c, 51, "2015-05-03", "Playoffs", 12, "W")
        for i in range(7, 14):
            _game(self.c, i, f"2015-11-{i:02d}", "Regular Season", 20, "W")

    def board(self, **kw):
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(self.c)):
            return main_api.get_streak_board(limit=50, mode="longest", **kw)

    def longest(self, out, kind):
        return max((s["length"] for s in out["streaks"] if s["kind"] == kind), default=None)

    def test_regular_season_streak_runs_through_the_playoffs(self):
        out = self.board()
        self.assertEqual(out["season_type"], "Regular Season")
        self.assertEqual(self.longest(out, "pts10"), 14)    # not 7 + 7 broken by the playoff night
        self.assertEqual(self.longest(out, "team_win"), 14)  # the Warriors' 28 began this way
        self.assertEqual(self.longest(out, "three"), 14)     # playoff games do not pad it

    def test_playoffs_are_their_own_board(self):
        out = self.board(season_type="Playoffs")
        self.assertEqual(out["season_type"], "Playoffs")
        # Two playoff games: below every floor, so nothing is ranked.
        self.assertIsNone(self.longest(out, "pts10"))
        self.assertIsNone(self.longest(out, "three"))

    def test_the_two_boards_are_cached_separately(self):
        rs = self.board()
        po = self.board(season_type="Playoffs")
        self.assertNotEqual(rs["archive"]["last_game"], po["archive"]["last_game"])
        self.assertEqual(self.board()["season_type"], "Regular Season")

    def test_unknown_season_type_is_refused(self):
        with self.assertRaises(main_api.HTTPException):
            self.board(season_type="PlayIn")


if __name__ == "__main__":
    unittest.main()
