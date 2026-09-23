"""/api/player-shot-chart season_type contract (fixed 2026-09-23; the
frontend toggle is built against it):

  season_type: "Regular Season" (default) | "Playoffs"; anything else -> 400
  passed to ShotChartDetail (which also returns the league averages)
  part of every cache key, echoed in the response as `season_type`

No network: shotchartdetail is replaced by a fake that records its kwargs.
"""
import types
import unittest
from unittest import mock

import main_api


def _payload(n_shots=1, with_shot_set=True):
    shot_headers = ["GAME_ID", "GAME_DATE", "EVENT_TYPE", "ACTION_TYPE", "SHOT_TYPE",
                    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE",
                    "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "PERIOD", "GAME_EVENT_ID"]
    rows = [["0042500401", "20260604", "Made Shot", "Jump Shot", "2PT Field Goal",
             "Mid-Range", "Center(C)", "16-24 ft.", 18, 0, 180, 1, 1, 10 + i]
            for i in range(n_shots)]
    sets = []
    if with_shot_set:
        sets.append({"name": "Shot_Chart_Detail", "headers": shot_headers, "rowSet": rows})
    sets.append({"name": "LeagueAverages",
                 "headers": ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "FGA", "FGM", "FG_PCT"],
                 "rowSet": [["Mid-Range", "Center(C)", "16-24 ft.", 100, 42, 0.42]]})
    return {"resultSets": sets}


class _FakeShotChart:
    calls = []
    payload = None

    def __init__(self, **kwargs):
        _FakeShotChart.calls.append(kwargs)

    def get_dict(self):
        return _FakeShotChart.payload


class TestPlayerShotChartSeasonType(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(main_api.app)

    def setUp(self):
        _FakeShotChart.calls = []
        _FakeShotChart.payload = _payload()
        fake_module = types.SimpleNamespace(ShotChartDetail=_FakeShotChart)
        p = mock.patch.object(main_api, "shotchartdetail", fake_module)
        p.start()
        self.addCleanup(p.stop)
        main_api.player_shot_chart_cache.clear()
        self.addCleanup(main_api.player_shot_chart_cache.clear)

    def get(self, **params):
        return self.client.get("/api/player-shot-chart",
                               params={"player_id": 1628983, "season": "2025-26", **params})

    def test_default_is_regular_season_and_echoed(self):
        r = self.get()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["season_type"], "Regular Season")
        self.assertEqual(_FakeShotChart.calls[-1]["season_type_all_star"], "Regular Season")

    def test_playoffs_is_passed_through_and_echoed(self):
        r = self.get(season_type="Playoffs")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["season_type"], "Playoffs")
        self.assertEqual(_FakeShotChart.calls[-1]["season_type_all_star"], "Playoffs")
        self.assertEqual(len(body["shots"]), 1)
        self.assertEqual(body["league_averages"], body["averages"])

    def test_anything_else_is_400_and_never_reaches_upstream(self):
        for bad in ("playoffs", "Pre Season", "All Star", "PlayIn", ""):
            r = self.get(season_type=bad)
            self.assertEqual(r.status_code, 400, bad)
        self.assertEqual(_FakeShotChart.calls, [])

    def test_season_types_are_cached_separately(self):
        self.get(season_type="Regular Season")
        _FakeShotChart.payload = _payload(n_shots=3)
        r = self.get(season_type="Playoffs")
        self.assertEqual(len(r.json()["shots"]), 3)
        self.assertEqual(len(_FakeShotChart.calls), 2)
        # Both now served from cache, each its own.
        self.assertEqual(len(self.get(season_type="Regular Season").json()["shots"]), 1)
        self.assertEqual(len(self.get(season_type="Playoffs").json()["shots"]), 3)
        self.assertEqual(len(_FakeShotChart.calls), 2)

    def test_missing_shot_result_set_is_an_error_not_an_empty_chart(self):
        _FakeShotChart.payload = _payload(with_shot_set=False)
        r = self.get(season_type="Playoffs")
        self.assertEqual(r.status_code, 502)
        self.assertEqual(main_api.player_shot_chart_cache, {})


if __name__ == "__main__":
    unittest.main()
