"""grade() finds play-in games, and refuses a score that is not a result.

Scratch databases only: a minimal predictions_log and a minimal archive in a
temp directory. Nothing here opens Data/.
"""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import grade_predictions  # noqa: E402

LAL, NOP = 1610612747, 1610612740


class GradeResultTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_grade_test_")
        self.odds = os.path.join(self.tmp, "OddsData.sqlite")
        self.team = os.path.join(self.tmp, "TeamData.sqlite")
        c = sqlite3.connect(self.team)
        c.executescript("""
            CREATE TABLE team_metadata (team_id INTEGER PRIMARY KEY, full_name TEXT);
            CREATE TABLE team_game_advanced (game_id TEXT, team_id INTEGER, opp_team_id INTEGER,
                season TEXT, season_type TEXT, game_date TEXT, pts INTEGER, opp_pts INTEGER);
        """)
        c.executemany("INSERT INTO team_metadata VALUES (?,?)",
                      [(LAL, "Los Angeles Lakers"), (NOP, "New Orleans Pelicans")])
        c.commit()
        c.close()
        c = sqlite3.connect(self.odds)
        c.execute("""CREATE TABLE predictions_log (id INTEGER PRIMARY KEY, log_date TEXT,
                     home_team TEXT, away_team TEXT, game_start_time_utc TEXT,
                     actual_winner TEXT, actual_total INTEGER)""")
        # 2024-04-16 play-in, LAL @ NOP, 19:00 ET tip.
        c.execute("INSERT INTO predictions_log (log_date, home_team, away_team, game_start_time_utc) "
                  "VALUES ('2024-04-16', 'New Orleans Pelicans', 'Los Angeles Lakers', "
                  "'2024-04-16T23:00:00Z')")
        c.commit()
        c.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _box(self, nop_pts, lal_pts, season_type="PlayIn"):
        c = sqlite3.connect(self.team)
        c.executemany("INSERT INTO team_game_advanced VALUES (?,?,?,?,?,?,?,?)", [
            ("0052300121", NOP, LAL, "2023-24", season_type, "2024-04-16", nop_pts, lal_pts),
            ("0052300121", LAL, NOP, "2023-24", season_type, "2024-04-16", lal_pts, nop_pts),
        ])
        c.commit()
        c.close()

    def _graded(self):
        c = sqlite3.connect(self.odds)
        try:
            return c.execute("SELECT actual_winner, actual_total FROM predictions_log").fetchone()
        finally:
            c.close()

    def test_a_play_in_game_is_found_and_graded(self):
        self._box(106, 110)
        self.assertEqual(grade_predictions.grade(self.odds, self.team), 1)
        self.assertEqual(self._graded(), ("Los Angeles Lakers", 216))

    def test_no_box_score_leaves_it_ungraded_without_error(self):
        self.assertEqual(grade_predictions.grade(self.odds, self.team), 0)
        self.assertEqual(self._graded(), (None, None))

    def test_a_zero_zero_score_is_refused_loudly(self):
        self._box(0, 0)
        with self.assertRaises(RuntimeError):
            grade_predictions.grade(self.odds, self.team)
        self.assertEqual(self._graded(), (None, None), "0-0 must not become an away win")

    def test_a_tie_is_refused_loudly(self):
        self._box(100, 100)
        with self.assertRaises(RuntimeError):
            grade_predictions.grade(self.odds, self.team)
        self.assertEqual(self._graded(), (None, None))


if __name__ == "__main__":
    unittest.main()
