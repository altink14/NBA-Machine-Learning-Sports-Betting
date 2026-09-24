"""The public record seals a pick until its game starts (owner's call, 2026-09-24).

The model's picks are Pro. /api/prediction-log is public and used to serve
every pending pick with its confidence and edge before the game.
"""
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import main_api


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _row(conn, key, logged, tip, winner="Boston Celtics", actual=None):
    conn.execute(
        "INSERT INTO predictions_log (logged_at, log_date, sport, sportsbook, game_key, home_team, away_team, "
        "game_start_time_utc, home_ml, away_ml, predicted_winner, winner_confidence, ev_home, ev_away, model, "
        "actual_winner, why_json) VALUES (?, ?, 'NBA', 'fanduel', ?, 'Boston Celtics', 'New York Knicks', ?, "
        "-150, 130, ?, 64.2, 0.031, -0.05, 'candidate', ?, '[1]')",
        (main_api._utc_iso(logged), logged.strftime("%Y-%m-%d"), key, main_api._utc_iso(tip), winner, actual))


class PredictionLogSealTest(unittest.TestCase):

    def setUp(self):
        self.c = sqlite3.connect(":memory:")
        self.c.row_factory = sqlite3.Row
        main_api._ensure_prediction_log_schema(self.c)
        now = datetime.now(timezone.utc)
        _row(self.c, "future", now - timedelta(hours=2), now + timedelta(hours=3))
        _row(self.c, "past", now - timedelta(days=1, hours=3), now - timedelta(days=1), actual="Boston Celtics")
        self.addCleanup(self.c.close)

    def fetch(self):
        with mock.patch.object(main_api, "_odds_snapshot_conn", lambda: _KeepOpen(self.c)):
            out = main_api.get_prediction_log(days=30)
        return {p["game_key"]: p for p in out["predictions"]}, out

    def test_a_pending_pick_is_sealed_but_its_existence_and_time_are_public(self):
        rows, _ = self.fetch()
        f = rows["future"]
        self.assertTrue(f["sealed_until_tipoff"])
        for field in ("predicted_winner", "winner_confidence", "ev_home", "ev_away", "why_json"):
            self.assertIsNone(f[field], field)
        self.assertTrue(f["logged_at"])
        self.assertEqual(f["home_team"], "Boston Celtics")

    def test_a_started_game_shows_the_pick(self):
        rows, out = self.fetch()
        p = rows["past"]
        self.assertFalse(p["sealed_until_tipoff"])
        self.assertEqual((p["predicted_winner"], p["winner_confidence"]), ("Boston Celtics", 64.2))
        self.assertEqual(out["summary"]["graded"], 1)


if __name__ == "__main__":
    unittest.main()
