"""League leaders: per-game boards, one row per player, scaled minimums.

Found 2026-09-23. The boards were the top N by season TOTAL re-ranked per
game on the page, so a per-game leader outside the top N vanished; a traded
player was two partial players; and the attempt minimums were never scaled
to a short season. Temp database only.
"""
import math
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


COLS = ["gp", "gs", "min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]


def _db(team_games=82):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, season_type TEXT, team_id INTEGER, "
              + ", ".join(f"{k} REAL" for k in COLS) + ", fg_pct REAL, fg3_pct REAL, ft_pct REAL)")
    c.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT)")
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT)")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT, home_team_id INTEGER, away_team_id INTEGER)")
    c.execute("CREATE TABLE player_game_log (player_id INTEGER, team_id INTEGER, game_id TEXT, game_date TEXT)")
    c.executemany("INSERT INTO team_metadata VALUES (?, ?)", [(1, "AAA"), (2, "BBB"), (3, "CCC")])
    for g in range(team_games):   # team 1 plays every game; that sets the season's length
        c.execute("INSERT INTO box_scores VALUES (?, 'S', 'Regular Season', 1, ?)", (f"g{g}", 2 + g % 2))
    return c


def _player(c, pid, name, team, **stats):
    c.execute("INSERT OR IGNORE INTO players VALUES (?, ?)", (pid, name))
    row = {k: 0 for k in COLS}
    row.update(stats)
    c.execute("INSERT INTO player_season_totals (player_id, season, season_type, team_id, " + ", ".join(COLS)
              + ") VALUES (?, 'S', 'Regular Season', ?, " + ", ".join("?" for _ in COLS) + ")",
              [pid, team] + [row[k] for k in COLS])


class LeagueLeadersTest(unittest.TestCase):

    def board(self, c, cats, **kw):
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)):
            return main_api.get_stats_leader_board(cats, "S", **kw)

    def test_per_game_leader_outside_the_top_ten_by_total_is_found(self):
        c = _db()
        for i in range(12):   # twelve durable scorers with big totals
            _player(c, 100 + i, f"Durable {i}", 1, gp=80, pts=2000 - i)
        _player(c, 7, "Missed Some", 2, gp=60, pts=1800)   # 30.0 a game, 12th by total
        rows = self.board(c, "pts", limit=10)["boards"]["pts"]
        self.assertEqual(rows[0]["full_name"], "Missed Some")

    def test_seventy_percent_of_team_games_to_qualify(self):
        c = _db()
        _player(c, 1, "Qualified", 1, gp=58, pts=58 * 25)
        _player(c, 2, "Too Few", 1, gp=57, pts=57 * 40)
        out = self.board(c, "pts")
        self.assertEqual(out["rules"]["min_games"], 58)
        self.assertEqual([r["full_name"] for r in out["boards"]["pts"]], ["Qualified"])

    def test_traded_player_is_one_row_with_both_teams(self):
        c = _db()
        _player(c, 5, "Traded", 2, gp=30, pts=600, fgm=240, fga=400)
        _player(c, 5, "Traded", 3, gp=40, pts=900, fgm=350, fga=600)
        c.execute("INSERT INTO player_game_log VALUES (5, 3, 'g0', '2025-10-22')")
        c.execute("INSERT INTO player_game_log VALUES (5, 2, 'g1', '2026-02-10')")
        out = self.board(c, "pts,fg_pct")
        pts = out["boards"]["pts"]
        self.assertEqual(len(pts), 1)
        self.assertEqual((pts[0]["gp"], pts[0]["pts"], pts[0]["team_abbr"]), (70, 1500, "CCC/BBB"))
        self.assertAlmostEqual(out["boards"]["fg_pct"][0]["fg_pct"], 590 / 1000)

    def test_attempt_minimums_scale_with_a_short_season(self):
        c = _db(team_games=50)
        rules = self.board(c, "fg_pct")["rules"]
        self.assertEqual(rules["team_games"], 50)
        self.assertEqual(rules["min_games"], 35)
        self.assertEqual(rules["min_attempts"]["fg_pct"], math.ceil(300 * 50 / 82))

    def test_single_board_default_is_still_by_total(self):
        c = _db()
        _player(c, 1, "Volume", 1, gp=82, pts=2000)
        _player(c, 2, "Rate", 1, gp=60, pts=1900)
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)):
            self.assertEqual(main_api.get_stats_leaders("pts", "S")[0]["full_name"], "Volume")
            self.assertEqual(main_api.get_stats_leaders("pts", "S", rank="per_game")[0]["full_name"], "Rate")

    def test_unknown_category_is_refused(self):
        c = _db()
        with self.assertRaises(main_api.HTTPException):
            self.board(c, "pts,plus_minus")


if __name__ == "__main__":
    unittest.main()
