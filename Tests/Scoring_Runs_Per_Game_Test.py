"""Scoring runs per game: the denominator is the run table's seasons.

Found 2026-09-24 by the chat's game tools. With no season given,
/api/stats/runs divided each team's play-by-play runs (2019-20 on) by every
game it has played since 1996-97 (NYK: 2,386), so every per-game rate was
several times too low. Temp database only.
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
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT)")
    c.execute("CREATE TABLE team_game_advanced (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT)")
    c.execute("CREATE TABLE scoring_runs (game_id TEXT, season TEXT, season_type TEXT, game_date TEXT, "
              "team_tricode TEXT, opp_tricode TEXT, points INTEGER, start_period INTEGER)")
    c.executemany("INSERT INTO team_metadata VALUES (?, ?)", [(1, "AAA"), (2, "BBB")])
    # Ten games a season in 2010-11 (no play-by-play) and in 2020-21 (covered).
    for season in ("2010-11", "2020-21"):
        for g in range(10):
            gid = f"{season}-{g}"
            c.execute("INSERT INTO team_game_advanced VALUES (?, 1, ?, 'Regular Season')", (gid, season))
            c.execute("INSERT INTO team_game_advanced VALUES (?, 2, ?, 'Regular Season')", (gid, season))
    # Five runs, all in the covered season, all by AAA.
    for g in range(5):
        c.execute("INSERT INTO scoring_runs VALUES (?, '2020-21', 'Regular Season', '2021-01-01', 'AAA', 'BBB', 12, 3)",
                  (f"2020-21-{g}",))
    c.commit()
    return c


class TestRunsPerGame(unittest.TestCase):

    def _teams(self, **kw):
        conn = _db()
        with mock.patch.object(main_api, "get_db_conn", return_value=_KeepOpen(conn)):
            out = main_api.get_scoring_runs(**kw)
        return {t["team"]: t for t in out["teams"]}

    def test_no_season_counts_only_covered_seasons(self):
        teams = self._teams()
        self.assertEqual(teams["AAA"]["games"], 10)          # not 20
        self.assertEqual(teams["AAA"]["delivered_per_game"], 0.5)  # 5 runs / 10 games, not / 20
        self.assertEqual(teams["BBB"]["allowed_per_game"], 0.5)

    def test_one_season_unchanged(self):
        teams = self._teams(season="2020-21")
        self.assertEqual(teams["AAA"]["games"], 10)
        self.assertEqual(teams["AAA"]["delivered_per_game"], 0.5)


if __name__ == "__main__":
    unittest.main()
