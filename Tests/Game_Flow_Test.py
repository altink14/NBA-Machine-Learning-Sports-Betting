"""
Unit tests for the game-flow transform (src/Utils/game_flow.py) plus a
TestClient check of /api/games/{game_id}/game-flow against a real game whose
playbyplayv3 response is already on disk in Data/nba_cache (no network).
"""

import unittest

from src.Utils.game_flow import (
    parse_clock,
    elapsed_seconds,
    build_game_flow,
    _detect_runs,
)


def act(period, clock, score_home="", score_away=""):
    """Minimal synthetic playbyplayv3 action (field names match the real feed)."""
    return {
        "period": period,
        "clock": clock,
        "scoreHome": score_home,
        "scoreAway": score_away,
    }


HOME = {"abbr": "BOS", "name": "Boston Celtics"}
AWAY = {"abbr": "NYK", "name": "New York Knicks"}


class TestParseClock(unittest.TestCase):

    def test_full_period(self):
        self.assertEqual(parse_clock("PT12M00.00S"), 720.0)

    def test_fractional_seconds(self):
        self.assertEqual(parse_clock("PT00M34.50S"), 34.5)

    def test_unpadded(self):
        self.assertEqual(parse_clock("PT1M5S"), 65.0)

    def test_invalid(self):
        self.assertIsNone(parse_clock(None))
        self.assertIsNone(parse_clock(""))
        self.assertIsNone(parse_clock("garbage"))
        self.assertIsNone(parse_clock("PT"))


class TestElapsedSeconds(unittest.TestCase):

    def test_game_start(self):
        self.assertEqual(elapsed_seconds(1, "PT12M00.00S"), 0)

    def test_mid_first_period(self):
        # 12:00 - 11:34 = 26 s elapsed
        self.assertEqual(elapsed_seconds(1, "PT11M34.00S"), 26)

    def test_period_boundaries(self):
        self.assertEqual(elapsed_seconds(1, "PT00M00.00S"), 720)
        self.assertEqual(elapsed_seconds(2, "PT12M00.00S"), 720)
        self.assertEqual(elapsed_seconds(4, "PT00M00.00S"), 2880)

    def test_overtime_is_five_minutes(self):
        self.assertEqual(elapsed_seconds(5, "PT05M00.00S"), 2880)
        self.assertEqual(elapsed_seconds(5, "PT03M30.00S"), 2970)
        self.assertEqual(elapsed_seconds(6, "PT05M00.00S"), 3180)

    def test_invalid_clock(self):
        self.assertIsNone(elapsed_seconds(1, "bad"))


class TestBuildGameFlow(unittest.TestCase):

    def setUp(self):
        self.actions = [
            act(1, "PT12M00.00S", "0", "0"),   # period marker: no new point
            act(1, "PT11M40.00S"),              # non-scoring event: skipped
            act(1, "PT11M00.00S", "2", "0"),   # t=60   home +2
            act(1, "PT10M00.00S", "2", "3"),   # t=120  away +3
            act(1, "PT09M00.00S", "4", "3"),   # t=180  home +2
            act(1, "PT08M30.00S", "4", "4"),   # t=210  away +1 (tie)
            act(1, "PT08M00.00S", "4", "6"),   # t=240  away +2
            act(1, "PT02M00.00S", "6", "6"),   # t=600  home +2 (tie)
            act(1, "PT01M00.00S", "8", "6"),   # t=660  home +2
            act(2, "PT11M00.00S", "10", "6"),  # t=780  home +2
            act(2, "PT10M00.00S", "12", "6"),  # t=840  home +2
            act(2, "PT09M00.00S", "14", "6"),  # t=900  home +2
            act(2, "PT08M00.00S", "16", "6"),  # t=960  home +2 -> 12-0 run
            act(2, "PT05M00.00S", "16", "8"),  # t=1140 away +2
        ]
        self.flow = build_game_flow("TEST123456", self.actions, HOME, AWAY)

    def test_identity_and_final(self):
        self.assertEqual(self.flow["game_id"], "TEST123456")
        self.assertEqual(self.flow["home"], HOME)
        self.assertEqual(self.flow["away"], AWAY)
        self.assertEqual(self.flow["final"], {"home": 16, "away": 8})

    def test_series_is_compact(self):
        # 0-0 start + 12 scoring events + terminal end-of-game point
        series = self.flow["series"]
        self.assertEqual(len(series), 14)
        self.assertEqual(
            series[0],
            {"t": 0, "period": 1, "margin": 0, "home_score": 0, "away_score": 0},
        )
        # terminal point sits at the final horn of period 2
        self.assertEqual(
            series[-1],
            {"t": 1440, "period": 2, "margin": 8, "home_score": 16, "away_score": 8},
        )

    def test_series_points(self):
        series = self.flow["series"]
        self.assertEqual(
            series[1],
            {"t": 60, "period": 1, "margin": 2, "home_score": 2, "away_score": 0},
        )
        self.assertEqual(
            series[2],
            {"t": 120, "period": 1, "margin": -1, "home_score": 2, "away_score": 3},
        )
        # margins always equal home - away, t never decreases
        for prev, cur in zip(series, series[1:]):
            self.assertEqual(cur["margin"], cur["home_score"] - cur["away_score"])
            self.assertGreaterEqual(cur["t"], prev["t"])

    def test_run_detection(self):
        runs = self.flow["runs"]
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run["team"], "home")
        self.assertEqual(run["points"], 12)
        self.assertEqual(run["opp_points"], 0)
        self.assertEqual(run["start_t"], 600)
        self.assertEqual(run["end_t"], 960)
        self.assertEqual(run["label"], "12-0 run")

    def test_lead_changes_and_ties(self):
        # margins: +2, -1, +1, 0, -2, 0, +2, ... => 4 sign flips, 2 ties
        self.assertEqual(self.flow["lead_changes"], 4)
        self.assertEqual(self.flow["ties"], 2)


class TestRunEdgeCases(unittest.TestCase):

    def test_run_with_two_opp_points(self):
        events = [
            {"t": 0, "team": "away", "pts": 3},
            {"t": 10, "team": "home", "pts": 2},
            {"t": 20, "team": "away", "pts": 3},
            {"t": 30, "team": "away", "pts": 3},
        ]
        runs = _detect_runs(events)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["team"], "away")
        self.assertEqual(runs[0]["label"], "9-2 run")

    def test_no_run_when_opponent_scores_three(self):
        events = [
            {"t": 0, "team": "home", "pts": 2},
            {"t": 10, "team": "home", "pts": 3},
            {"t": 20, "team": "away", "pts": 3},   # breaks the window
            {"t": 30, "team": "home", "pts": 2},
            {"t": 40, "team": "home", "pts": 2},
        ]
        self.assertEqual(_detect_runs(events), [])

    def test_at_most_six_runs_ranked_by_differential(self):
        # 8 disjoint home runs of increasing size; only the 6 biggest survive.
        events = []
        t = 0
        for size in range(8, 24, 2):  # 8, 10, ..., 22 point bursts
            for _ in range(size // 2):
                events.append({"t": t, "team": "home", "pts": 2})
                t += 10
            # opponent 3-pointer terminates each window
            events.append({"t": t, "team": "away", "pts": 3})
            t += 10
        runs = _detect_runs(events)
        self.assertEqual(len(runs), 6)
        self.assertEqual(sorted(r["points"] for r in runs), [12, 14, 16, 18, 20, 22])
        # chronological output
        self.assertEqual([r["start_t"] for r in runs], sorted(r["start_t"] for r in runs))


class TestGameFlowEndpoint(unittest.TestCase):
    """Endpoint test against game 0022400001, whose playbyplayv3 payload is
    permanently cached in Data/nba_cache - no network involved."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main_api
        cls.client = TestClient(main_api.app)

    def test_real_cached_game(self):
        resp = self.client.get("/api/games/0022400001/game-flow")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["game_id"], "0022400001")
        self.assertEqual(data["home"]["abbr"], "BOS")
        self.assertEqual(data["away"]["abbr"], "ATL")
        self.assertEqual(data["final"], {"home": 116, "away": 117})

        series = data["series"]
        self.assertEqual(
            series[0],
            {"t": 0, "period": 1, "margin": 0, "home_score": 0, "away_score": 0},
        )
        self.assertEqual(series[-1]["home_score"], 116)
        self.assertEqual(series[-1]["away_score"], 117)
        for prev, cur in zip(series, series[1:]):
            self.assertEqual(cur["margin"], cur["home_score"] - cur["away_score"])
            self.assertGreaterEqual(cur["t"], prev["t"])

        self.assertIsInstance(data["lead_changes"], int)
        self.assertIsInstance(data["ties"], int)
        self.assertLessEqual(len(data["runs"]), 6)
        for run in data["runs"]:
            self.assertIn(run["team"], ("home", "away"))
            self.assertGreaterEqual(run["points"], 8)
            self.assertLessEqual(run["opp_points"], 2)
            self.assertEqual(run["label"], f"{run['points']}-{run['opp_points']} run")
            self.assertLessEqual(run["start_t"], run["end_t"])

    def test_invalid_game_id_404(self):
        resp = self.client.get("/api/games/notagame/game-flow")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
