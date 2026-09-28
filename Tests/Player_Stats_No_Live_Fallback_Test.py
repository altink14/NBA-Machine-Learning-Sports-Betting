"""/api/player-stats never fetches nba.com live (2026-09-28).

A season with no archived rows used to fall through to a live nba.com call
that answered in a different shape (team_abbreviation, an invented
power_index). It now returns an empty list, so the shape every page reads is
the archive's, including on opening night before the first ingest.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import main_api  # noqa: E402


class PlayerStatsNoLiveFallbackTest(unittest.TestCase):
    def setUp(self):
        main_api.player_stats_cache.clear()
        self.client = TestClient(main_api.app)

    def test_unarchived_season_is_empty_and_calls_nothing(self):
        boom = mock.MagicMock(side_effect=AssertionError("nba.com must not be called"))
        with mock.patch.object(main_api, "leaguedashplayerstats", mock.MagicMock(LeagueDashPlayerStats=boom)):
            res = self.client.get("/api/player-stats", params={"season": "2099-00"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), [])
        boom.assert_not_called()

    def test_archived_season_keeps_the_archive_shape(self):
        res = self.client.get("/api/player-stats", params={"season": "2025-26"})
        self.assertEqual(res.status_code, 200)
        rows = res.json()
        if not rows:
            self.skipTest("local archive has no 2025-26 player_season_stats")
        self.assertIn("player_id", rows[0])
        self.assertNotIn("power_index", rows[0])
        self.assertNotIn("team_abbreviation", rows[0])

    def test_default_season_is_the_current_one(self):
        with mock.patch.object(main_api, "CURRENT_SEASON", main_api.CURRENT_SEASON):
            import inspect
            default = inspect.signature(main_api.get_player_stats).parameters["season"].default
        self.assertEqual(default, main_api.CURRENT_SEASON)


if __name__ == "__main__":
    unittest.main()
