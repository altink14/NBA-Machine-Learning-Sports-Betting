"""Play-in games (season_type 'PlayIn', game ids 005...) are neither regular
season nor playoffs, and since they entered box_scores several endpoints
treated them as one or the other. Also: odds snapshots must not be filed under
a guessed sport.

The endpoint tests read the real archive READ-ONLY (plain SELECTs through the
normal handlers) and pin facts that are settled history: the 2025-26 play-in
tournament, played 14-17 April 2026. The player-page and schedule tests use a
temp database / a mocked feed. Nothing writes to production.
"""
import sqlite3
import types
import unittest
from unittest import mock

import pandas as pd

import main_api


class _Client:
    client = None

    @classmethod
    def get(cls, path, **params):
        if cls.client is None:
            from fastapi.testclient import TestClient
            cls.client = TestClient(main_api.app)
        return cls.client.get(path, params=params)


class TestHistoricalMatchupSeries(unittest.TestCase):
    """PHX and GSW met in the 2026 play-in (0052500211, 17 April 2026)."""

    def test_play_in_is_not_in_the_season_series(self):
        r = _Client.get("/api/historical/matchup", team1="PHX", team2="GSW", season=2026)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["series_scope"], "Regular Season")
        self.assertTrue(all(m["season_type"] == "Regular Season" for m in body["matchups"]))
        self.assertEqual(body["total_games"], len(body["matchups"]))
        post = {p["game_id"]: p for p in body["postseason"]}
        self.assertIn("0052500211", post)
        self.assertEqual(post["0052500211"]["season_type"], "PlayIn")

    def test_unknown_team_is_404_not_a_zero_series(self):
        r = _Client.get("/api/historical/matchup", team1="PHX", team2="Nowhere Nobodies", season=2026)
        self.assertEqual(r.status_code, 404)


class TestTeamGamesSeasonType(unittest.TestCase):

    def test_every_row_says_its_season_type(self):
        r = _Client.get("/api/teams/PHX/games", season="2025-26")
        self.assertEqual(r.status_code, 200)
        rows = r.json()
        types_seen = {g["game_id"][:3]: g["season_type"] for g in rows}
        self.assertEqual(types_seen.get("005"), "PlayIn")
        self.assertEqual(types_seen.get("002"), "Regular Season")

    def test_regular_season_filter_excludes_play_in(self):
        r = _Client.get("/api/teams/PHX/games", season="2025-26", season_type="Regular Season")
        self.assertEqual(r.status_code, 200)
        rows = r.json()
        self.assertTrue(rows)
        self.assertTrue(all(g["game_id"].startswith("002") for g in rows))
        self.assertLessEqual(len(rows), 82)

    def test_bad_season_type_is_400(self):
        self.assertEqual(_Client.get("/api/teams/PHX/games", season_type="Play-In").status_code, 400)


class TestDailyLeadersSeasonType(unittest.TestCase):
    """17 April 2026 had two play-in games and nothing else."""

    def test_filter_is_honoured(self):
        r = _Client.get("/api/stats/daily-leaders", date="2026-04-17", season_type="PlayIn")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["games"], 2)
        self.assertEqual(r.json()["season_type"], "PlayIn")
        r = _Client.get("/api/stats/daily-leaders", date="2026-04-17", season_type="Regular Season")
        self.assertEqual(r.json()["games"], 0)
        self.assertEqual(r.json()["leaders"], [])

    def test_default_still_counts_every_game_type(self):
        r = _Client.get("/api/stats/daily-leaders", date="2026-04-17")
        self.assertEqual(r.json()["games"], 2)
        self.assertIsNone(r.json()["season_type"])

    def test_bad_season_type_is_400(self):
        self.assertEqual(_Client.get("/api/stats/daily-leaders", season_type="Finals").status_code, 400)


class TestPlayInLabels(unittest.TestCase):

    def test_prefix_table_uses_the_archive_spelling(self):
        self.assertEqual(main_api.SEASON_TYPE_BY_PREFIX["005"], "PlayIn")

    def test_schedule_filter_matches_play_in_under_either_spelling(self):
        feed = {"gameDates": [{"gameDate": "04/15/2027", "games": [
            {"gameId": "0052600101", "gameDateTimeEst": "2027-04-15T19:30:00Z",
             "homeTeam": {"teamTricode": "PHI"}, "awayTeam": {"teamTricode": "ORL"}},
            {"gameId": "0022600999", "gameDateTimeEst": "2027-04-12T19:30:00Z",
             "homeTeam": {"teamTricode": "BOS"}, "awayTeam": {"teamTricode": "NYK"}},
        ]}]}
        fake = types.SimpleNamespace(schedule_league_v2=lambda season: feed)
        with mock.patch("src.Utils.nba_stats_client.get_client", lambda *a, **k: fake):
            for spelling in ("PlayIn", "Play-In"):
                r = _Client.get("/api/schedule", season="2026-27", season_type=spelling)
                self.assertEqual(r.status_code, 200, r.text)
                ids = [g["game_id"] for d in r.json().get("dates", []) for g in d.get("games", [])]
                self.assertEqual(ids, ["0052600101"], spelling)

    def test_heat_calendar_labels_play_in_as_the_archive_does(self):
        # Devin Booker played in the 2026 play-in (PHX).
        r = _Client.get("/api/players/1626164/heat-calendar", season="2025-26")
        self.assertEqual(r.status_code, 200)
        labels = {e["game_id"]: e["season_type"] for e in r.json()["entries"]}
        # 0052500121 (vs POR) and 0052500211 (vs GSW).
        self.assertEqual(labels.get("0052500121"), "PlayIn")
        self.assertEqual(labels.get("0052500211"), "PlayIn")


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


class TestPlayerSeasonFallback(unittest.TestCase):
    """No season-totals row: the page falls back to summing the game log,
    which must not fold play-in or playoff games into season averages."""

    def test_fallback_sums_regular_season_only(self):
        c = sqlite3.connect(":memory:")
        self.addCleanup(c.close)
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE players (player_id INTEGER, full_name TEXT, first_name TEXT,
                                  last_name TEXT, is_active INTEGER);
            CREATE TABLE player_bio (player_id INTEGER, team_abbr TEXT, position TEXT,
                                     height TEXT, weight TEXT, fetched_at TEXT,
                                     years_experience INTEGER);
            CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, season_type TEXT,
                                               team_id INTEGER);
            CREATE TABLE player_season_advanced (player_id INTEGER, season TEXT, season_type TEXT);
            CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT);
            CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT);
            CREATE TABLE player_game_log (player_id INTEGER, game_id TEXT, pts INTEGER,
                                          ast INTEGER, reb INTEGER, min REAL);
            INSERT INTO players VALUES (5, 'Test Guard', 'Test', 'Guard', 1);
            INSERT INTO player_bio VALUES (5, 'PHX', NULL, NULL, NULL, '2026-01-01', 3);
            INSERT INTO box_scores VALUES ('0022500001', '2025-26', 'Regular Season');
            INSERT INTO box_scores VALUES ('0022500002', '2025-26', 'Regular Season');
            INSERT INTO box_scores VALUES ('0052500211', '2025-26', 'PlayIn');
            INSERT INTO box_scores VALUES ('0042500101', '2025-26', 'Playoffs');
            INSERT INTO player_game_log VALUES (5, '0022500001', 10, 2, 3, 30);
            INSERT INTO player_game_log VALUES (5, '0022500002', 20, 4, 5, 32);
            INSERT INTO player_game_log VALUES (5, '0052500211', 50, 10, 10, 44);
            INSERT INTO player_game_log VALUES (5, '0042500101', 40, 10, 10, 44);
        """)
        req = types.SimpleNamespace()
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)):
            # Past the rate limiter, which insists on a real Request.
            out = main_api.get_player_by_id.__wrapped__(req, 5, season="2025-26")
        self.assertEqual(out["totals"]["games"], 2)
        self.assertEqual(out["totals"]["pts_per_game"], 15.0)
        # Unknown bio fields are unknown, not invented.
        self.assertEqual(out["bio"]["position"], "N/A")
        self.assertIsNone(out["bio"]["instagram"])


class TestSnapshotNeverGuessesTheSport(unittest.TestCase):

    def _runner_with_unknown_sport(self):
        class Provider:
            def __init__(self, sportsbook, sport):
                pass

            def get_resolved_sport(self):
                raise RuntimeError("feed changed shape")

            def get_odds(self):
                return {"Las Vegas Aces:New York Liberty": {
                    "under_over_odds": 160.5,
                    "Las Vegas Aces": {"money_line_odds": -150},
                    "New York Liberty": {"money_line_odds": 130},
                    "game_start_time_utc": "2026-09-24T23:30:00+00:00"}}
            games = [{}]

        R = main_api.PredictionRunner
        with mock.patch.object(main_api, "SbrOddsProvider", Provider), \
             mock.patch.object(R, "_load_team_stats", lambda self: pd.DataFrame({"TEAM_NAME": []})), \
             mock.patch.object(R, "_last_game_dates", lambda self: {"x": []}), \
             mock.patch.object(R, "_load_schedule", lambda self: None), \
             mock.patch.object(R, "_load_xgboost_models", lambda self: (None, None)):
            return R(sportsbook="fanduel", kelly_criterion=True, sport="NBA")

    def test_unknown_sport_is_none_and_no_snapshot_is_saved(self):
        runner = self._runner_with_unknown_sport()
        self.assertIsNone(runner.resolved_sport)
        with mock.patch.object(main_api, "snapshot_odds") as snap, \
             mock.patch.object(main_api, "_nba_regular_season_under_way", lambda *a, **k: False):
            res = runner.run_predictions()
        snap.assert_not_called()
        # WNBA team names never become NBA picks.
        self.assertEqual(res["predictions"], [])


if __name__ == "__main__":
    unittest.main()
