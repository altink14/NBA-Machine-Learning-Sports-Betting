import unittest
from unittest import mock

from src.Utils import availability
from src.Utils import espn_injuries


def rating(player_id, name, impact, mpg, team="AAA"):
    return {
        "player_id": player_id,
        "name": name,
        "team_abbr": team,
        "season": "2025-26",
        "gp": 60,
        "min_per_g": mpg,
        "off_impact": impact,
        "def_impact": 0.0,
        "total_impact": impact,
        "impact_rank": None,
    }


SYNTH_RATINGS = [
    rating(1, "Star Guy", 6.0, 36.0),
    rating(2, "Solid Starter", 2.0, 30.0),
    rating(3, "Bad Backup", -2.0, 12.0),
    rating(4, "Iron Man", 20.0, 48.0),  # absurd on purpose - tests caps
]


class TestAdjustedRatingDelta(unittest.TestCase):

    def test_empty_out_list_is_zero(self):
        result = availability.adjusted_rating_delta("AAA", [], "2025-26", ratings=SYNTH_RATINGS)
        self.assertEqual(result["delta_per_100"], 0.0)
        self.assertEqual(result["players"], [])

    def test_star_out_costs_impact_times_minute_share(self):
        result = availability.adjusted_rating_delta("AAA", [1], "2025-26", ratings=SYNTH_RATINGS)
        # 6.0 impact * (36/48) = 4.5 lost
        self.assertAlmostEqual(result["delta_per_100"], -4.5, places=6)
        p = result["players"][0]
        self.assertEqual(p["name"], "Star Guy")
        self.assertTrue(p["rated"])
        self.assertAlmostEqual(p["min_share"], 0.75, places=6)
        self.assertAlmostEqual(p["contribution"], 4.5, places=6)

    def test_multiple_players_sum(self):
        result = availability.adjusted_rating_delta("AAA", [1, 2], "2025-26", ratings=SYNTH_RATINGS)
        # 6.0*0.75 + 2.0*(30/48) = 4.5 + 1.25
        self.assertAlmostEqual(result["delta_per_100"], -5.75, places=6)

    def test_negative_impact_player_out_helps(self):
        result = availability.adjusted_rating_delta("AAA", [3], "2025-26", ratings=SYNTH_RATINGS)
        # -2.0 * (12/48) = -0.5 lost -> +0.5 delta
        self.assertAlmostEqual(result["delta_per_100"], 0.5, places=6)

    def test_minutes_and_contribution_caps(self):
        result = availability.adjusted_rating_delta("AAA", [4], "2025-26", ratings=SYNTH_RATINGS)
        p = result["players"][0]
        # Minutes capped at 40 -> share 40/48; 20 * 0.8333 = 16.67, then
        # clamped to the per-player cap of 8.
        self.assertAlmostEqual(p["min_share"], round(40.0 / 48.0, 3), places=6)
        self.assertAlmostEqual(p["contribution"], availability.PLAYER_CONTRIBUTION_CAP, places=6)
        self.assertAlmostEqual(result["delta_per_100"], -8.0, places=6)

    def test_team_delta_cap(self):
        stars = [rating(i, f"Star {i}", 8.0, 48.0) for i in range(1, 5)]
        result = availability.adjusted_rating_delta(
            "AAA", [1, 2, 3, 4], "2025-26", ratings=stars
        )
        self.assertAlmostEqual(result["delta_per_100"], -availability.TEAM_DELTA_CAP, places=6)

    def test_unknown_player_contributes_zero(self):
        result = availability.adjusted_rating_delta("AAA", [999], "2025-26", ratings=SYNTH_RATINGS)
        self.assertEqual(result["delta_per_100"], 0.0)
        p = result["players"][0]
        self.assertFalse(p["rated"])
        self.assertIsNone(p["impact"])
        self.assertEqual(p["contribution"], 0.0)

    def test_accepts_dict_ratings(self):
        index = {r["player_id"]: r for r in SYNTH_RATINGS}
        result = availability.adjusted_rating_delta("AAA", [1], "2025-26", ratings=index)
        self.assertAlmostEqual(result["delta_per_100"], -4.5, places=6)


class TestMatchupAvailability(unittest.TestCase):

    def _absences(self):
        return {
            "by_team": {
                "AAA": [
                    {"player_id": 1, "name": "Star Guy", "status": "Out", "detail": "Knee"},
                    {"player_id": None, "name": "Unknown Rookie", "status": "Out", "detail": ""},
                ],
            },
            "counted_statuses": ["Out", "Doubtful"],
            "total_counted": 2,
            "unmatched_names": ["Unknown Rookie"],
            "match_rate": 0.5,
            "fetched_at": "2026-08-10T00:00:00+00:00",
            "source": "espn",
        }

    def test_shape_and_deltas(self):
        with mock.patch.object(
            availability.player_impact, "get_impact_ratings", return_value=SYNTH_RATINGS
        ):
            result = availability.matchup_availability("AAA", "BBB", "2025-26", absences=self._absences())
        self.assertEqual(result["home"]["team"], "AAA")
        self.assertEqual(result["away"]["team"], "BBB")
        self.assertAlmostEqual(result["home"]["delta_per_100"], -4.5, places=6)
        self.assertEqual(result["away"]["players_out"], [])
        self.assertEqual(result["away"]["delta_per_100"], 0.0)
        # The unmatched rookie is still listed, with zero contribution.
        names = [p["name"] for p in result["home"]["players_out"]]
        self.assertIn("Unknown Rookie", names)
        rookie = next(p for p in result["home"]["players_out"] if p["name"] == "Unknown Rookie")
        self.assertEqual(rookie["contribution"], 0.0)
        self.assertEqual(result["statuses_counted"], ["Out", "Doubtful"])
        self.assertIn("note", result)


class TestNameNormalization(unittest.TestCase):

    def test_suffixes_dropped(self):
        self.assertEqual(espn_injuries.normalize_name("Jaren Jackson Jr."), "jaren jackson")
        self.assertEqual(espn_injuries.normalize_name("Jimmy Butler III"), "jimmy butler")
        self.assertEqual(espn_injuries.normalize_name("Wendell Moore Jr"), "wendell moore")

    def test_punctuation_and_accents(self):
        self.assertEqual(espn_injuries.normalize_name("Nikola Jokić"), "nikola jokic")
        self.assertEqual(espn_injuries.normalize_name("Shaquille O'Neal"), "shaquille oneal")
        self.assertEqual(espn_injuries.normalize_name("P.J. Washington"), "pj washington")
        # Hyphens are word breaks, not deletions.
        self.assertEqual(espn_injuries.normalize_name("Karl-Anthony Towns"), "karl anthony towns")

    def test_empty(self):
        self.assertEqual(espn_injuries.normalize_name(""), "")
        self.assertEqual(espn_injuries.normalize_name(None), "")


class TestEspnParsing(unittest.TestCase):

    PAYLOAD = {
        "injuries": [
            {
                "displayName": "Golden State Warriors",
                "injuries": [
                    {
                        "athlete": {"displayName": "Jimmy Butler III", "position": {"abbreviation": "SF"}},
                        "status": "Out",
                        "date": "2026-08-01T00:00Z",
                        "shortComment": "Out indefinitely.",
                        "details": {"type": "Knee", "location": "Leg", "detail": "Surgery"},
                    },
                    {
                        "athlete": {"displayName": "Stephen Curry"},
                        "status": "Questionable",
                        "details": {"type": "Ankle"},
                    },
                ],
            },
            {
                "displayName": "Minnesota Timberwolves",
                "injuries": [
                    {
                        "athlete": {"displayName": "Donte DiVincenzo"},
                        "status": "Day-To-Day",
                        "details": {},
                    }
                ],
            },
        ]
    }

    def test_parse_extracts_all_entries(self):
        entries = espn_injuries.parse_injuries(self.PAYLOAD)
        self.assertEqual(len(entries), 3)
        butler = entries[0]
        self.assertEqual(butler["team_name"], "Golden State Warriors")
        self.assertEqual(butler["player_name"], "Jimmy Butler III")
        self.assertEqual(butler["status"], "Out")
        self.assertEqual(butler["detail"], "Knee (Leg) - Surgery")

    def test_parse_tolerates_garbage(self):
        self.assertEqual(espn_injuries.parse_injuries(None), [])
        self.assertEqual(espn_injuries.parse_injuries({}), [])
        self.assertEqual(espn_injuries.parse_injuries({"injuries": "nope"}), [])
        self.assertEqual(espn_injuries.parse_injuries({"injuries": [None, 42]}), [])

    def test_build_absences_counts_only_out_and_doubtful(self):
        """Questionable and Day-To-Day must NEVER count (game-time
        decisions poison the adjustment - documented policy)."""
        entries = espn_injuries.parse_injuries(self.PAYLOAD)
        result = espn_injuries.build_absences(entries)
        self.assertEqual(result["total_counted"], 1)
        gsw = result["by_team"].get("GSW", [])
        self.assertEqual(len(gsw), 1)
        self.assertEqual(gsw[0]["name"], "Jimmy Butler III")
        self.assertIsNotNone(gsw[0]["player_id"])  # matched via players table
        self.assertNotIn("MIN", result["by_team"])
        self.assertEqual(result["match_rate"], 1.0)

    def test_build_absences_reports_unmatched(self):
        entries = [{
            "team_name": "Golden State Warriors",
            "player_name": "Totally Fictional Player",
            "status": "Out",
            "detail": "",
        }]
        result = espn_injuries.build_absences(entries)
        self.assertEqual(result["unmatched_names"], ["Totally Fictional Player"])
        self.assertEqual(result["match_rate"], 0.0)
        self.assertIsNone(result["by_team"]["GSW"][0]["player_id"])


class TestFeedNeverRaises(unittest.TestCase):

    def test_dead_feed_returns_empty_structure(self):
        with mock.patch.object(espn_injuries.requests, "get", side_effect=Exception("boom")):
            result = espn_injuries.get_absences(force_refresh=True)
        self.assertEqual(result["by_team"], {})
        self.assertEqual(result["source"], "unavailable")
        self.assertEqual(result["total_counted"], 0)

    def test_http_error_returns_empty_structure(self):
        fake = mock.Mock(status_code=503)
        with mock.patch.object(espn_injuries.requests, "get", return_value=fake):
            result = espn_injuries.get_absences(force_refresh=True)
        self.assertEqual(result["by_team"], {})
        self.assertEqual(result["source"], "unavailable")


class TestApiEndpoints(unittest.TestCase):
    """TestClient checks - additive endpoints and the prediction attach."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main_api
        cls.main_api = main_api
        cls.client = TestClient(main_api.app)

    def test_impact_ratings_endpoint_shape(self):
        r = self.client.get("/api/impact-ratings", params={"season": "2025-26", "min_gp": 20})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        for key in ("season", "min_gp", "count", "methodology", "players"):
            self.assertIn(key, data)
        self.assertGreater(data["count"], 100)
        first = data["players"][0]
        for key in ("player_id", "name", "team_abbr", "season", "gp",
                    "min_per_g", "off_impact", "def_impact", "total_impact", "impact_rank"):
            self.assertIn(key, first)
        self.assertEqual(first["impact_rank"], 1)

    def test_impact_ratings_unknown_season_404(self):
        r = self.client.get("/api/impact-ratings", params={"season": "1990-91"})
        self.assertEqual(r.status_code, 404)

    def test_player_impact_endpoint(self):
        r = self.client.get("/api/players/1628983/impact")  # SGA
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["player_id"], 1628983)
        self.assertGreaterEqual(len(data["seasons"]), 3)
        self.assertIn("methodology", data)

    def test_player_impact_unknown_player_404(self):
        r = self.client.get("/api/players/999999999/impact")
        self.assertEqual(r.status_code, 404)

    def test_matchup_availability_endpoint(self):
        synthetic = {
            "by_team": {},
            "counted_statuses": ["Out", "Doubtful"],
            "total_counted": 0,
            "unmatched_names": [],
            "match_rate": 1.0,
            "fetched_at": "2026-08-10T00:00:00+00:00",
            "source": "espn",
        }
        with mock.patch.object(espn_injuries, "get_absences", return_value=synthetic):
            r = self.client.get("/api/matchups/availability", params={"home": "OKC", "away": "DEN"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["home"]["team"], "OKC")
        self.assertEqual(data["away"]["team"], "DEN")
        self.assertEqual(data["home"]["players_out"], [])
        self.assertEqual(data["home"]["delta_per_100"], 0.0)
        self.assertIn("note", data)

    def test_matchup_availability_unknown_team_404(self):
        r = self.client.get("/api/matchups/availability", params={"home": "NOPE", "away": "DEN"})
        self.assertEqual(r.status_code, 404)

    def test_attach_availability_adds_field_without_touching_predictions(self):
        import types
        synthetic = {
            "by_team": {
                "GSW": [{"player_id": 1, "name": "Star Guy", "status": "Out", "detail": ""}],
            },
            "counted_statuses": ["Out", "Doubtful"],
            "total_counted": 1,
            "unmatched_names": [],
            "match_rate": 1.0,
            "fetched_at": "2026-08-10T00:00:00+00:00",
            "source": "espn",
        }
        fake_runner = types.SimpleNamespace(resolved_sport="NBA", sport="NBA")
        result = {
            "sportsbook": "fanduel",
            "predictions": [{
                "home_team": "Golden State Warriors",
                "away_team": "Denver Nuggets",
                "winner_confidence": 61.2,
                "predicted_winner": "Golden State Warriors",
            }],
        }
        before = dict(result["predictions"][0])
        with mock.patch.object(espn_injuries, "get_absences", return_value=synthetic), \
             mock.patch.object(
                 self.main_api.availability_adjust.player_impact,
                 "get_impact_ratings",
                 return_value=[rating(1, "Star Guy", 6.0, 36.0, team="GSW")],
             ):
            out = self.main_api.PredictionRunner._attach_availability(fake_runner, result)
        pred = out["predictions"][0]
        # Everything that was there before is unchanged.
        for key, value in before.items():
            self.assertEqual(pred[key], value)
        # The additive field is present and correct.
        self.assertIn("availability", pred)
        self.assertAlmostEqual(pred["availability"]["home_delta"], -4.5, places=6)
        self.assertEqual(pred["availability"]["away_delta"], 0.0)
        self.assertEqual(pred["availability"]["players_out"], ["Star Guy"])
        self.assertEqual(pred["availability"]["note"], "impact-adjusted")

    def test_attach_availability_skips_non_nba(self):
        import types
        fake_runner = types.SimpleNamespace(resolved_sport="WNBA", sport="NBA")
        result = {"sportsbook": "fanduel", "predictions": [{"home_team": "Las Vegas Aces", "away_team": "New York Liberty"}]}
        out = self.main_api.PredictionRunner._attach_availability(fake_runner, result)
        self.assertNotIn("availability", out["predictions"][0])

    def test_attach_availability_never_raises(self):
        import types
        fake_runner = types.SimpleNamespace(resolved_sport="NBA", sport="NBA")
        result = {"sportsbook": "fanduel", "predictions": [{"home_team": "Golden State Warriors", "away_team": "Denver Nuggets"}]}
        with mock.patch.object(espn_injuries, "get_absences", side_effect=Exception("feed exploded")):
            out = self.main_api.PredictionRunner._attach_availability(fake_runner, result)
        # Predictions unchanged, no crash.
        self.assertNotIn("availability", out["predictions"][0])

    def test_error_results_pass_through_unchanged(self):
        import types
        fake_runner = types.SimpleNamespace(resolved_sport="NBA", sport="NBA")
        result = {"error": "No odds data found.", "predictions": []}
        out = self.main_api.PredictionRunner._attach_availability(fake_runner, result)
        self.assertEqual(out, {"error": "No odds data found.", "predictions": []})


if __name__ == "__main__":
    unittest.main()
