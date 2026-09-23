"""/predictions must never report a failure as an empty slate.

Before 2026-09-23 every failure inside PredictionRunner.run_predictions came
back as HTTP 200 with `predictions: []` and an "error" string no caller read,
so the picks board said "no games on the board" when the truth was "we could
not compute the picks". These tests pin the contract:

  status ok / market_only / no_games / no_odds  -> HTTP 200
  status failed                                 -> HTTP 503, detail = why
  unknown sportsbook or sport                   -> HTTP 400

Nothing here touches the production databases: snapshot_odds, get_db_conn,
the ledger connection and the injury feed are all patched out.
"""
import os
import sqlite3
import tempfile
import types
import unittest
from datetime import date
from unittest import mock

import pandas as pd

import main_api


def _odds(games):
    """{'Home:Away': {...}} in SbrOddsProvider.get_odds() shape."""
    out = {}
    for home, away, hml, aml in games:
        out[f"{home}:{away}"] = {
            "under_over_odds": 221.5,
            home: {"money_line_odds": hml},
            away: {"money_line_odds": aml},
            "game_start_time_utc": "2026-10-21T23:30:00+00:00",
        }
    return out


def _runner(odds, scraped_games=None, resolved="NBA", team_names=()):
    """A PredictionRunner without its constructor (which loads models and DBs)."""
    r = object.__new__(main_api.PredictionRunner)
    r.sportsbook = "fanduel"
    r.sport = "NBA"
    r.kelly_criterion = True
    r.model_name = "xgboost"
    r.resolved_sport = resolved
    r.team_stats_table = "2026-09-22"
    r.team_stats_df = pd.DataFrame({"TEAM_NAME": list(team_names)})
    games = scraped_games if scraped_games is not None else [{} for _ in odds]
    r.odds_provider = types.SimpleNamespace(get_odds=lambda: odds, games=games)
    return r


class _Isolated(unittest.TestCase):
    """Patch out every write and every network call run_predictions can make."""

    def setUp(self):
        patches = [
            mock.patch.object(main_api, "snapshot_odds", lambda *a, **k: None),
            mock.patch.object(main_api, "get_db_conn", lambda: None),
            mock.patch.object(main_api.PredictionRunner, "_attach_availability",
                              lambda self, result: result),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def season(self, under_way):
        p = mock.patch.object(main_api, "_nba_regular_season_under_way", lambda *a, **k: under_way)
        p.start()
        self.addCleanup(p.stop)


class TestRunPredictionsStatus(_Isolated):

    def test_empty_feed_offseason_is_no_odds_and_says_it_cannot_tell(self):
        self.season(False)
        res = _runner({}, scraped_games=[]).run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_NO_ODDS)
        self.assertEqual(res["predictions"], [])
        self.assertNotIn("error", res)
        self.assertIn("failed scrape", res["message"])

    def test_empty_feed_during_the_season_is_a_failure(self):
        self.season(True)
        res = _runner({}, scraped_games=[]).run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_FAILED)
        self.assertIn("regular season", res["error"])

    def test_games_on_board_but_book_has_no_price(self):
        self.season(False)
        res = _runner({}, scraped_games=[{}, {}, {}]).run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_NO_ODDS)
        self.assertIn("3 NBA game(s)", res["message"])
        self.assertIn("fanduel", res["message"])

    def test_wnba_fallback_in_the_offseason_is_no_games(self):
        self.season(False)
        odds = _odds([("Las Vegas Aces", "New York Liberty", -150, 130)])
        res = _runner(odds, resolved="WNBA").run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_NO_GAMES)
        self.assertEqual(res["predictions"], [])
        self.assertEqual(res["resolved_sport"], "WNBA")

    def test_wnba_fallback_during_the_season_is_a_failure(self):
        self.season(True)
        odds = _odds([("Las Vegas Aces", "New York Liberty", -150, 130)])
        res = _runner(odds, resolved="WNBA").run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_FAILED)

    def test_nba_odds_with_unrecognised_names_is_a_failure(self):
        self.season(False)
        odds = _odds([("Boston Celts", "New York Knickerbockers", -150, 130)])
        res = _runner(odds, resolved="NBA").run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_FAILED)
        self.assertIn("team names", res["error"])

    def test_no_team_stats_is_market_only_with_no_invented_edge(self):
        self.season(False)
        odds = _odds([("Boston Celtics", "New York Knicks", -200, 170)])
        res = _runner(odds, team_names=["Somebody Else"]).run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_MARKET_ONLY)
        (p,) = res["predictions"]
        self.assertEqual(p["model"], main_api.SIMULATED_MODEL_TAG)
        fair_home = main_api.devig.fair_probs([
            main_api.parlay.american_to_true_decimal(-200.0),
            main_api.parlay.american_to_true_decimal(170.0)])[0]
        # The market's own probability, not the market's plus two points.
        self.assertAlmostEqual(p["winner_confidence"], round(fair_home * 100, 2), places=2)
        self.assertIsNone(p["under_over_prediction"])
        self.assertIsNone(p["under_over_confidence"])
        # At the fair price there is no positive edge on either side.
        self.assertLessEqual(p["expected_value"]["home_team"], 0.0)
        self.assertLessEqual(p["expected_value"]["away_team"], 0.0)

    def test_stats_for_home_side_only_is_a_failure(self):
        self.season(False)
        odds = _odds([("Boston Celtics", "New York Knicks", -200, 170)])
        # has_stats looks at the home side; preparation needs both.
        res = _runner(odds, team_names=["Boston Celtics"]).run_predictions()
        self.assertEqual(res["status"], main_api.PRED_STATUS_FAILED)
        self.assertIn("Could not prepare valid data", res["error"])


class TestRegularSeasonWindow(unittest.TestCase):

    def test_window_edges(self):
        from preflight_opening_night import OPENING_NIGHT
        f = main_api._nba_regular_season_under_way
        self.assertFalse(f(date(2026, 9, 23)))
        self.assertFalse(f(OPENING_NIGHT.replace(day=OPENING_NIGHT.day - 1)))
        self.assertTrue(f(OPENING_NIGHT))
        end = OPENING_NIGHT.replace(year=OPENING_NIGHT.year + 1, month=4, day=10)
        self.assertTrue(f(end))
        self.assertFalse(f(end.replace(day=11)))
        self.assertFalse(f(end.replace(month=6, day=15)))


class _FakeRunner:
    result = None

    def __init__(self, sportsbook, kelly_criterion, sport="NBA"):
        self.resolved_sport = sport

    def run_predictions(self):
        return dict(_FakeRunner.result)


class TestPredictionsEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(main_api.app)

    def setUp(self):
        main_api.predictions_cache.clear()
        for p in (mock.patch.object(main_api, "PredictionRunner", _FakeRunner),
                  mock.patch.object(main_api, "LOG_PREDICTIONS_ON_REQUEST", False),
                  mock.patch.object(main_api, "PREDICTIONS_SOURCE", "live")):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(main_api.predictions_cache.clear)

    def test_failure_is_503_with_the_reason_and_is_not_cached(self):
        _FakeRunner.result = main_api._prediction_result(
            main_api.PRED_STATUS_FAILED, "fanduel", error="the scrape fell over")
        r = self.client.get("/predictions")
        self.assertEqual(r.status_code, 503)
        self.assertIn("the scrape fell over", r.json()["detail"])
        self.assertEqual(main_api.predictions_cache, {})

    def test_no_games_is_200_with_status(self):
        _FakeRunner.result = main_api._prediction_result(
            main_api.PRED_STATUS_NO_GAMES, "fanduel", message="offseason")
        r = self.client.get("/predictions")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "no_games")
        self.assertEqual(r.json()["predictions"], [])

    def test_unknown_sportsbook_is_400(self):
        r = self.client.get("/predictions", params={"sportsbook": "fandual"})
        self.assertEqual(r.status_code, 400)

    def test_other_sport_is_400(self):
        r = self.client.get("/predictions", params={"sport": "WNBA"})
        self.assertEqual(r.status_code, 400)


class TestLedgerModeStatus(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)

        def conn():
            c = sqlite3.connect(self.path)
            c.row_factory = sqlite3.Row
            return c
        p = mock.patch.object(main_api, "_odds_snapshot_conn", conn)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

    def test_empty_ledger_says_not_logged(self):
        out = main_api._predictions_from_ledger("fanduel", True, "NBA")
        self.assertEqual(out["status"], main_api.PRED_STATUS_NOT_LOGGED)
        self.assertEqual(out["predictions"], [])
        self.assertIn("note", out)

    def test_logged_rows_say_ok(self):
        now = main_api.datetime.now(main_api.timezone.utc)
        log_date = (main_api.to_nba_date(now) or now.date()).isoformat()
        c = sqlite3.connect(self.path)
        main_api._ensure_prediction_log_schema(c)
        c.execute(
            "INSERT INTO predictions_log (logged_at, log_date, sport, sportsbook, game_key, home_team, "
            "away_team, game_start_time_utc, home_ml, away_ml, predicted_winner, winner_confidence, model) "
            "VALUES ('2000-01-01T00:00:00+00:00', ?, 'NBA', 'fanduel', 'A:B', 'A', 'B', "
            "'2099-01-01T00:00:00+00:00', -150, 130, 'A', 60.0, 'candidate')", (log_date,))
        c.commit()
        c.close()
        out = main_api._predictions_from_ledger("fanduel", True, "NBA")
        self.assertEqual(out["status"], main_api.PRED_STATUS_OK)
        self.assertEqual(len(out["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
