"""The career table is built from every season or not at all.

Found 2026-09-27 (nav audit bug 7): ingest_career_totals.py summed one
stats.nba.com request per season, skipped a season whose request failed and
wrote anyway when 80% had answered. The 2011-12 lockout season went missing
from every career (Kobe 1,288 games instead of 1,346). It now sums the
archive's own player_season_totals and refuses a window with any missing or
half-built season. Temp database only.
"""
import sqlite3
import unittest

import ingest_career_totals as ict


def _db(seasons_with_players=("1996-97", "1997-98", "1998-99")):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT)")
    c.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT)")
    c.execute("CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, season_type TEXT, "
              "team_id INTEGER, gp INTEGER, min REAL, pts INTEGER, reb INTEGER, ast INTEGER)")
    c.executemany("INSERT INTO players VALUES (?, ?)", [(1, "Traded Guy"), (2, "One Season"), (3, "Filler")])
    for season in ("1996-97", "1997-98", "1998-99"):
        for g in range(10):
            c.execute("INSERT INTO box_scores VALUES (?, ?, 'Regular Season')", (f"{season}-{g}", season))
    for season in seasons_with_players:
        # 10 games x 2 teams = 20 team-games; ~10.5 appearances per team-game
        c.execute("INSERT INTO player_season_totals VALUES (3, ?, 'Regular Season', 9, 200, 4000, 900, 400, 300)",
                  (season,))
        c.execute("INSERT INTO player_season_totals VALUES (1, ?, 'Regular Season', 7, 6, 180, 60, 20, 10)", (season,))
        c.execute("INSERT INTO player_season_totals VALUES (1, ?, 'Regular Season', 8, 4, 120, 40, 10, 5)", (season,))
    if "1997-98" in seasons_with_players:
        c.execute("INSERT INTO player_season_totals VALUES (2, '1997-98', 'Regular Season', 7, 1, 3, 0, 0, 0)")
    return c


class CareerSpanTest(unittest.TestCase):
    def test_a_missing_season_refuses_the_whole_build(self):
        c = _db(seasons_with_players=("1996-97", "1998-99"))   # the middle season never built
        self.assertTrue(any("1997-98" in p for p in ict.window_problems(c, 1996, 1998)))
        self.assertEqual(ict.build(c, 1996, 1998), 1)
        self.assertIsNone(c.execute("SELECT name FROM sqlite_master WHERE name = 'player_career_span'").fetchone())

    def test_a_half_built_season_refuses_too(self):
        c = _db()
        c.execute("UPDATE player_season_totals SET gp = gp / 4 WHERE season = '1998-99'")
        self.assertTrue(any("1998-99" in p for p in ict.window_problems(c, 1996, 1998)))

    def test_complete_window_sums_stints_and_keeps_one_season_players(self):
        c = _db()
        self.assertEqual(ict.window_problems(c, 1996, 1998), [])
        self.assertEqual(ict.build(c, 1996), 0)   # window end = latest archived season
        rows = {r[0]: r for r in c.execute("SELECT player_id, seasons, gp, pts, first_season_in_window, "
                                            "last_season_in_window, window_last FROM player_career_span")}
        self.assertEqual(rows[1][1:], (3, 30, 300, 1996, 1998, 1998))   # two stints a season, summed
        self.assertEqual(rows[2][1:4], (1, 1, 0))                       # one season is still a career
        # a rebuild replaces the table rather than leaving stale rows
        c.execute("DELETE FROM player_season_totals WHERE player_id = 2")
        self.assertEqual(ict.build(c, 1996), 0)
        self.assertIsNone(c.execute("SELECT 1 FROM player_career_span WHERE player_id = 2").fetchone())


if __name__ == "__main__":
    unittest.main()
