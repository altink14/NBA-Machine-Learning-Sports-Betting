"""Hustle stats nba.com did not count come back null, not zero (nav audit bug 16).

nba.com's leaguehustlestatsplayer sends BOX_OUTS = 0 for every player in
2015-16 (147/147) and 2016-17 (485/485): box-outs were not tracked until
2017-18. The Hustle Builder ranked players on that column of fake zeros.

No network: the stats client is mocked.
"""

import unittest
from unittest import mock

from src.Utils import Hustle


def row(pid, g=60, box=0, charges=0, defl=50):
    return {
        "PLAYER_ID": pid, "PLAYER_NAME": f"P{pid}", "TEAM_ABBREVIATION": "BOS",
        "G": g, "MIN": 1500.0, "DEFLECTIONS": defl, "SCREEN_ASSISTS": 10,
        "SCREEN_AST_PTS": 22, "LOOSE_BALLS_RECOVERED": 12, "CHARGES_DRAWN": charges,
        "CONTESTED_SHOTS": 300, "BOX_OUTS": box,
    }


class ShapePlayersTest(unittest.TestCase):

    def test_all_zero_stat_is_untracked_and_null(self):
        players, untracked = Hustle.shape_players([row(1), row(2), row(3)])
        self.assertEqual(untracked, ["charges_drawn", "box_outs"])
        self.assertTrue(all(p["box_outs"] is None for p in players))
        self.assertEqual([p["deflections"] for p in players], [50, 50, 50])

    def test_a_tracked_stat_keeps_its_real_zeros(self):
        players, untracked = Hustle.shape_players([row(1, box=0, charges=3), row(2, box=140, charges=0)])
        self.assertEqual(untracked, [])
        self.assertEqual([p["box_outs"] for p in players], [0, 140])
        self.assertEqual([p["charges_drawn"] for p in players], [3, 0])

    def test_missing_column_counts_as_untracked(self):
        r = row(1, box=5)
        del r["BOX_OUTS"]
        players, untracked = Hustle.shape_players([r])
        self.assertIn("box_outs", untracked)
        self.assertIsNone(players[0]["box_outs"])

    def test_empty_season(self):
        self.assertEqual(Hustle.shape_players([]), ([], []))
        self.assertEqual(Hustle.season_coverage([]), {"players": 0, "max_games_played": None})

    def test_coverage_exposes_a_partial_season(self):
        # 2015-16's regular season: nobody past 2 games.
        cov = Hustle.season_coverage([row(1, g=1), row(2, g=2)])
        self.assertEqual(cov, {"players": 2, "max_games_played": 2})


class HustleEndpointTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import main_api
        from fastapi.testclient import TestClient
        cls.main_api = main_api
        cls.client = TestClient(main_api.app)

    def _get(self, season, rows):
        fake = mock.Mock()
        fake.league_hustle_stats.return_value = rows
        if getattr(self.main_api, "limiter", None) is not None:
            self.main_api.limiter.reset()
        with mock.patch("src.Utils.nba_stats_client.get_client", return_value=fake):
            r = self.client.get(f"/api/stats/hustle?season={season}")
        return r, fake

    def test_untracked_box_outs_are_null_in_the_response(self):
        r, _ = self._get("2016-17", [row(1, charges=2), row(2, charges=0)])
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["untracked"], ["box_outs"])
        self.assertEqual([p["box_outs"] for p in d["players"]], [None, None])
        self.assertEqual(d["coverage"]["players"], 2)

    def test_a_finished_season_is_cached_for_good(self):
        _, fake = self._get("2016-17", [row(1, box=3)])
        self.assertIsNone(fake.league_hustle_stats.call_args.kwargs["ttl"])
        _, fake = self._get(self.main_api.CURRENT_SEASON, [row(1, box=3)])
        self.assertEqual(fake.league_hustle_stats.call_args.kwargs["ttl"], 3600)


if __name__ == "__main__":
    unittest.main()
