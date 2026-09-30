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


class EspnGradeTest(GradeResultTest):
    """Grading from ESPN's final score while nba.com's box score is missing
    (2026-09-29), and the check against nba.com's once it lands."""

    def setUp(self):
        super().setUp()
        from src.Utils import espn_boxscore
        c = sqlite3.connect(self.team)
        espn_boxscore.ensure_tables(c)
        c.commit()
        c.close()
        # The real ledger's guards, so the confirm/conflict writes are judged
        # exactly as they would be on the home PC.
        c = sqlite3.connect(self.odds)
        c.executescript("""
            CREATE TRIGGER predictions_log_no_regrade BEFORE UPDATE ON predictions_log
            WHEN OLD.actual_winner IS NOT NULL AND NEW.actual_winner IS NOT OLD.actual_winner
            BEGIN SELECT RAISE(ABORT, 'a graded prediction cannot be regraded'); END;
            CREATE TRIGGER predictions_log_no_delete BEFORE DELETE ON predictions_log
            BEGIN SELECT RAISE(ABORT, 'the prediction log is append-only'); END;
        """)
        c.commit()
        c.close()

    def _espn(self, home_id, home_pts, away_id, away_pts, day="2024-04-16", event="401654321"):
        c = sqlite3.connect(self.team)
        c.execute(
            "INSERT INTO espn_box_scores (espn_event_id, game_id, model_game_id, season, season_type, "
            "game_date, home_team_id, away_team_id, home_pts, away_pts, periods, usable, source_url, "
            "fetched_at, parser_version, payload) VALUES (?, NULL, ?, '2023-24', 'PlayIn', ?, ?, ?, ?, ?, "
            "4, 1, 'https://site.api.espn.com/x', '2024-04-17T12:00:00+00:00', 1, x'00')",
            (event, f"espn:{event}", day, home_id, away_id, home_pts, away_pts))
        c.commit()
        c.close()

    def _row(self):
        c = sqlite3.connect(self.odds)
        c.row_factory = sqlite3.Row
        try:
            return dict(c.execute("SELECT * FROM predictions_log").fetchone())
        finally:
            c.close()

    def test_espn_grades_when_nba_com_has_no_box_score(self):
        self._espn(NOP, 106, LAL, 110)
        self.assertEqual(grade_predictions.grade(self.odds, self.team), 1)
        r = self._row()
        self.assertEqual((r["actual_winner"], r["actual_total"]), ("Los Angeles Lakers", 216))
        self.assertEqual((r["result_source"], r["result_source_ref"]), ("espn", "401654321"))
        self.assertIsNone(r["result_confirmed_at"])

    def test_neutral_site_filed_the_other_way_round_reads_points_by_team(self):
        self._espn(LAL, 110, NOP, 106)   # ESPN put the Lakers at home
        grade_predictions.grade(self.odds, self.team)
        self.assertEqual(self._row()["actual_winner"], "Los Angeles Lakers")

    def test_nba_com_wins_when_both_are_there(self):
        self._box(106, 110)
        self._espn(NOP, 999, LAL, 1)     # would say NOP; must not be read
        grade_predictions.grade(self.odds, self.team)
        r = self._row()
        self.assertEqual(r["actual_winner"], "Los Angeles Lakers")
        self.assertEqual((r["result_source"], r["result_source_ref"]), ("nba.com", "0052300121"))

    def test_an_espn_zero_zero_is_refused(self):
        self._espn(NOP, 0, LAL, 0)
        with self.assertRaises(RuntimeError):
            grade_predictions.grade(self.odds, self.team)
        self.assertIsNone(self._row()["actual_winner"])

    def test_nba_com_agreeing_later_confirms_the_espn_grade(self):
        self._espn(NOP, 106, LAL, 110)
        grade_predictions.grade(self.odds, self.team)
        self._box(106, 110)
        grade_predictions.grade(self.odds, self.team)
        r = self._row()
        self.assertIsNotNone(r["result_confirmed_at"])
        self.assertIsNone(r["result_conflict"])

    def test_nba_com_disagreeing_later_is_loud_once_and_kept_on_the_row(self):
        self._espn(NOP, 106, LAL, 110)
        grade_predictions.grade(self.odds, self.team)
        self._box(111, 110)              # nba.com says NOP won
        with self.assertRaisesRegex(RuntimeError, "DISAGREE"):
            grade_predictions.grade(self.odds, self.team)
        r = self._row()
        self.assertEqual(r["actual_winner"], "Los Angeles Lakers", "a grade is never changed")
        self.assertIn("0052300121", r["result_conflict"])
        self.assertIsNone(r["result_confirmed_at"])
        self.assertEqual(grade_predictions.grade(self.odds, self.team), 0)   # not re-raised daily

    def test_an_espn_graded_game_counts_as_played_on_its_date_for_clv(self):
        self._espn(NOP, 106, LAL, 110)
        c = sqlite3.connect(self.team)
        try:
            names = grade_predictions._team_name_to_id(c)
            pick = {"home_team": "New Orleans Pelicans", "away_team": "Los Angeles Lakers",
                    "game_start_time_utc": "2024-04-16T23:00:00Z"}
            self.assertTrue(grade_predictions._played_on_date(c, names, pick))
            pick["game_start_time_utc"] = "2024-04-18T23:00:00Z"
            self.assertFalse(grade_predictions._played_on_date(c, names, pick))
        finally:
            c.close()


if __name__ == "__main__":
    unittest.main()
