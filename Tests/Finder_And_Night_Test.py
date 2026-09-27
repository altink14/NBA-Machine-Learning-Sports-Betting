"""Game Finder totals and venue; the daily board's game labels.

Found 2026-09-24: the Game Finder page printed "100 games found" because the
endpoint returned only the rows it sent (the archive holds 2,761 40-point
games), and every row read "vs" because no venue was sent (Luka Doncic's 73
was at Atlanta). The daily leaders board could not say which game a night's
one game was (the 2026 Finals Game 5). Temp database only.
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
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT)")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT, game_date TEXT, "
              "home_team_id INTEGER, away_team_id INTEGER)")
    c.execute("CREATE TABLE game_results (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT, "
              "game_date TEXT, wl TEXT)")
    c.execute("CREATE TABLE team_game_advanced (game_id TEXT, team_id INTEGER, opp_team_id INTEGER, "
              "pts INTEGER, opp_pts INTEGER)")
    c.execute("CREATE TABLE player_game_log (game_id TEXT, player_id INTEGER, team_id INTEGER, game_date TEXT, "
              "min REAL, pts INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, "
              "fg3m INTEGER, fgm INTEGER, fga INTEGER)")
    c.executemany("INSERT INTO team_metadata VALUES (?, ?)", [(1, "DAL"), (2, "ATL"), (3, "CHI"), (4, "UTA")])
    c.executemany("INSERT INTO players VALUES (?, ?)", [(77, "Luka"), (23, "Jordan"), (9, "Bench")])
    return c


def _game(c, gid, date, season, season_type, home, away, home_won, lines, advanced=True):
    c.execute("INSERT INTO box_scores VALUES (?, ?, ?, ?, ?, ?)", (gid, season, season_type, date, home, away))
    if season_type != "PlayIn":   # the league game log carries no play-in games
        c.execute("INSERT INTO game_results VALUES (?, ?, ?, ?, ?, ?)",
                  (gid, home, season, season_type, date, "W" if home_won else "L"))
        c.execute("INSERT INTO game_results VALUES (?, ?, ?, ?, ?, ?)",
                  (gid, away, season, season_type, date, "L" if home_won else "W"))
    if advanced:
        c.execute("INSERT INTO team_game_advanced VALUES (?, ?, ?, ?, ?)",
                  (gid, home, away, 100 if home_won else 90, 90 if home_won else 100))
        c.execute("INSERT INTO team_game_advanced VALUES (?, ?, ?, ?, ?)",
                  (gid, away, home, 90 if home_won else 100, 100 if home_won else 90))
    for pid, team, pts in lines:
        c.execute("INSERT INTO player_game_log VALUES (?, ?, ?, ?, 36, ?, 5, 5, 1, 0, 2, 3, 20, 40)",
                  (gid, pid, team, date, pts))


class GameFinderTest(unittest.TestCase):

    def setUp(self):
        self.c = _db()
        # Luka's 73: DAL at ATL (ATL is home), Dallas won.
        _game(self.c, "0022300667", "2024-01-26", "2023-24", "Regular Season", 2, 1, False, [(77, 1, 73)])
        # Two more 40-point games, one of them in a game with no advanced box.
        _game(self.c, "0022300700", "2024-02-01", "2023-24", "Regular Season", 1, 3, True, [(77, 1, 45)])
        _game(self.c, "0022300701", "2024-02-03", "2023-24", "Regular Season", 1, 3, True, [(77, 1, 41)],
              advanced=False)

    def find(self, **kw):
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(self.c)):
            return main_api.finder_player_games(**kw)

    def test_total_is_every_match_not_the_rows_returned(self):
        out = self.find(min_pts=40, limit=2)
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["total"], 3)
        self.assertTrue(out["truncated"])

    def test_venue_and_opponent_come_from_the_box_score(self):
        top = self.find(min_pts=70)["results"][0]
        self.assertEqual((top["pts"], top["is_home"], top["opp_abbr"], top["won"]), (73, False, "ATL", True))

    def test_a_game_without_an_advanced_box_is_still_found(self):
        rows = self.find(min_pts=40, sort="game_date")["results"]
        self.assertEqual(rows[0]["game_id"], "0022300701")
        self.assertEqual((rows[0]["is_home"], rows[0]["opp_abbr"], rows[0]["won"]), (True, "CHI", True))


class NightGamesTest(unittest.TestCase):

    def test_old_sequential_playoff_ids_still_get_round_and_game(self):
        c = _db()
        # 1996-97 playoff ids are plain sequence numbers; CHI met ATL, then UTA.
        n = 1
        for opp, games in ((2, 5), (4, 6)):
            for g in range(games):
                gid = f"00496000{n:02d}"
                _game(c, gid, f"1997-0{5 if opp == 2 else 6}-{10 + g:02d}", "1996-97", "Playoffs",
                      3, opp, True, [(23, 3, 30), (9, opp, 10)])
                n += 1
        out = main_api._night_games(c, "1997-06-15", ["0049600011"])
        self.assertEqual(out[0]["stage"], "Conference semifinals, Game 6")
        self.assertEqual((out[0]["away"], out[0]["home"], out[0]["home_pts"], out[0]["away_pts"]),
                         ("UTA", "CHI", 30, 10))

    def test_regular_season_and_play_in(self):
        c = _db()
        _game(c, "0022500001", "2025-10-21", "2025-26", "Regular Season", 1, 2, True, [(77, 1, 30)])
        _game(c, "0052500101", "2026-04-15", "2025-26", "PlayIn", 1, 2, True, [(77, 1, 30)])
        stages = [g["stage"] for g in main_api._night_games(c, "x", ["0022500001", "0052500101"])]
        self.assertEqual(stages, ["Regular season", "Play-in"])


class BlowoutReferenceTest(unittest.TestCase):
    # The Best games page said "a 40-point blowout scores under 1"; the median
    # 40-point game actually scored 1.84. The scale is now measured.

    def test_median_of_games_decided_by_the_margin(self):
        games = [{"score": s, "margin": m} for s, m in ((5.0, 3), (2.0, 30), (1.0, 35), (3.0, 41), (9.0, 29))]
        self.assertEqual(main_api._blowout_reference(games), {"margin": 30, "games": 3, "median_score": 2.0})

    def test_no_blowouts_is_none_not_zero(self):
        self.assertIsNone(main_api._blowout_reference([{"score": 5.0, "margin": 2}]))


if __name__ == "__main__":
    unittest.main()
