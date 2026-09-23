"""Preseason picks never reach the public record; failed runs never report green.

Temp OddsData only; the prediction runner is mocked.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import daily_update
import main_api

ON = date(2026, 10, 20)


def pick(tip: datetime, home="Boston Celtics", away="New York Knicks"):
    return {"home_team": home, "away_team": away, "home_odds": -150, "away_odds": 130,
            "predicted_winner": home, "winner_confidence": 61.0, "model": "xgboost_cand_2026-08",
            "expected_value": {"home_team": 1.0, "away_team": -2.0},
            "game_start_time_utc": tip.replace(microsecond=0).isoformat()}


class PreseasonWindowTest(unittest.TestCase):

    def setUp(self):
        self.p = mock.patch("preflight_opening_night.OPENING_NIGHT", ON)
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def test_the_window_is_the_weeks_before_opening_night(self):
        f = main_api._is_nba_preseason
        self.assertTrue(f("2026-10-05T23:30:00+00:00"))      # preseason
        self.assertTrue(f("2026-10-19T23:30:00+00:00"))      # the night before
        self.assertFalse(f("2026-10-20T23:30:00+00:00"))     # opening night itself
        self.assertFalse(f("2026-04-10T23:30:00+00:00"))     # last season (rehearsal games)
        self.assertFalse(f(None))

    def test_eastern_date_decides_not_utc(self):
        # 00:30 UTC on the 21st is 20:30 ET on the 20th: opening night, not preseason.
        self.assertFalse(main_api._is_nba_preseason("2026-10-21T00:30:00+00:00"))
        # 02:00 UTC on the 20th is 22:00 ET on the 19th: still preseason.
        self.assertTrue(main_api._is_nba_preseason("2026-10-20T02:00:00+00:00"))


class LogPreseasonTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="preseason_test_")
        self.patches = [mock.patch.object(main_api, "ODDS_DB_PATH", os.path.join(self.dir, "OddsData.sqlite"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_preseason_games_are_counted_and_not_written(self):
        tip = datetime.now(timezone.utc) + timedelta(days=1)
        opening = (tip + timedelta(days=10)).date()   # tomorrow is in the preseason window
        with mock.patch("preflight_opening_night.OPENING_NIGHT", opening):
            counts = main_api.log_predictions({"predictions": [pick(tip)]}, "fanduel", "NBA")
        self.assertEqual(counts["preseason"], 1)
        self.assertEqual(counts["written"], 0)
        c = sqlite3.connect(main_api.ODDS_DB_PATH)
        try:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0], 0)
        finally:
            c.close()

    def test_regular_season_games_still_log(self):
        tip = datetime.now(timezone.utc) + timedelta(days=1)
        opening = (tip - timedelta(days=30)).date()   # the season is under way
        with mock.patch("preflight_opening_night.OPENING_NIGHT", opening):
            counts = main_api.log_predictions({"predictions": [pick(tip)]}, "fanduel", "NBA")
        self.assertEqual((counts["written"], counts["preseason"]), (1, 0))

    def test_other_sports_are_not_affected(self):
        tip = datetime.now(timezone.utc) + timedelta(days=1)
        opening = (tip + timedelta(days=10)).date()
        with mock.patch("preflight_opening_night.OPENING_NIGHT", opening):
            counts = main_api.log_predictions({"predictions": [pick(tip, "Las Vegas Aces", "New York Liberty")]},
                                              "fanduel", "WNBA")
        self.assertEqual(counts["preseason"], 0)


class DailyStatusTest(unittest.TestCase):
    """daily_update turns a failed or market-only run into a failure."""

    def run_with(self, result, resolved="NBA"):
        runner = mock.Mock()
        runner.resolved_sport = resolved
        runner.run_predictions.return_value = result
        with mock.patch.object(main_api, "PredictionRunner", return_value=runner), \
                mock.patch.object(main_api, "log_predictions", return_value={"written": 0}) as lp:
            return daily_update.log_todays_predictions(), lp

    def test_failed_status_is_a_failure_even_in_the_offseason(self):
        with mock.patch.object(daily_update, "_nba_games_expected", return_value=False):
            status, lp = self.run_with({"status": "failed", "error": "team stats missing", "predictions": []})
        self.assertEqual(status, "failed")
        lp.assert_not_called()

    def test_market_only_is_a_failure_not_logged(self):
        status, lp = self.run_with({"status": "market_only", "predictions": [{"model": "implied_probability_sim"}]})
        self.assertEqual(status, "failed")
        lp.assert_not_called()

    def test_no_games_in_the_offseason_is_still_offseason(self):
        with mock.patch.object(daily_update, "_nba_games_expected", return_value=False):
            status, _ = self.run_with({"status": "no_games", "predictions": []}, resolved="WNBA")
        self.assertEqual(status, "offseason")


if __name__ == "__main__":
    unittest.main()
