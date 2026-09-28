"""NBA_STATS_LIVE=off: the public server never calls stats.nba.com (2026-09-28).

stats.nba.com refuses cloud IPs, so on the server a request-time call hung for
the library's timeout and then failed, once per visitor. With the switch off
the client serves its disk cache or the published mirror, and otherwise raises
LiveFetchDisabled at once; routes answer from our own tables where we hold the
data (play-by-play, shot charts, bios) and a fast, explicit 503 'live-only'
where we do not.
"""

import gzip
import json
import os
import socket
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import main_api  # noqa: E402
from src.Utils import nba_mirror  # noqa: E402
from src.Utils import nba_stats_client as nsc  # noqa: E402
from src.Utils import pbp_archive  # noqa: E402

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "TeamData.sqlite")
PBP_GAME = "0022500938"          # Adebayo's 83, held in pbp_events
NO_PBP_GAME = "0029600001"       # 1996-97: before our play-by-play floor


_real_getaddrinfo = socket.getaddrinfo


def _no_network(host, *a, **k):
    # Loopback stays open: asyncio and the test client talk to themselves.
    if host in ("localhost", "127.0.0.1", "::1", None):
        return _real_getaddrinfo(host, *a, **k)
    raise AssertionError(f"a network lookup of {host} was attempted with NBA_STATS_LIVE=off")


class _OffMixin:
    """Switch off, an empty disk cache, an empty mirror, and no sockets."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mirror_db = os.path.join(self.tmp.name, "mirror.sqlite")
        self.patches = [
            mock.patch.dict(os.environ, {"NBA_STATS_LIVE": "off", "NBA_MIRROR_DB": self.mirror_db}),
            mock.patch.object(nsc, "_CACHE_ROOT", Path(self.tmp.name)),
            mock.patch.object(socket, "getaddrinfo", _no_network),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()


class SwitchTest(_OffMixin, unittest.TestCase):
    def test_default_is_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NBA_STATS_LIVE", None)
            self.assertTrue(nsc.live_fetch_enabled())
        for v in ("off", "OFF", "0", "false", "no"):
            with mock.patch.dict(os.environ, {"NBA_STATS_LIVE": v}):
                self.assertFalse(nsc.live_fetch_enabled())

    def test_fetch_raises_at_once_without_touching_the_endpoint(self):
        endpoint = mock.MagicMock(side_effect=AssertionError("endpoint must not be built"))
        client = nsc.NBAStatsClient()
        t = time.time()
        with self.assertRaises(nsc.LiveFetchDisabled) as ctx:
            client._fetch("leaguehustlestatsplayer", endpoint, {"season": "2031-32"})
        self.assertLess(time.time() - t, 0.5)
        self.assertEqual(ctx.exception.endpoint, "leaguehustlestatsplayer")
        endpoint.assert_not_called()

    def test_outbound_slot_refuses_before_the_lock(self):
        client = nsc.NBAStatsClient()
        client._lock.acquire()   # a caller holding the lock must not make us wait
        try:
            with self.assertRaises(nsc.LiveFetchDisabled):
                with client.outbound_slot("playercareerstats"):
                    self.fail("the body must not run")
        finally:
            client._lock.release()

    def test_nba_api_itself_is_guarded(self):
        from nba_api.stats.endpoints import playerawards
        with self.assertRaises(nsc.LiveFetchDisabled):
            playerawards.PlayerAwards(player_id=2544)

    def test_cached_responses_still_serve(self):
        client = nsc.NBAStatsClient()
        endpoint = mock.MagicMock(side_effect=AssertionError("must be served from disk"))
        nsc._write_cache(nsc._cache_path("playbyplayv3", {"game_id": "X1"}), {"game": {"actions": [1]}})
        self.assertEqual(client._fetch("playbyplayv3", endpoint, {"game_id": "X1"}, ttl=None),
                         {"game": {"actions": [1]}})
        nsc._write_cache(nsc._cache_path("leaguedashlineups", {"season": "Y"}), {"fresh": True})
        self.assertEqual(client._fetch("leaguedashlineups", endpoint, {"season": "Y"}, ttl=3600),
                         {"fresh": True})

    def test_expired_copy_is_not_passed_off_as_current(self):
        path = nsc._cache_path("leaguedashlineups", {"season": "Z"})
        nsc._write_cache(path, {"old": True})
        old = time.time() - 7200
        os.utime(nsc.gz_path(path), (old, old))
        with self.assertRaises(nsc.LiveFetchDisabled):
            nsc.NBAStatsClient()._fetch("leaguedashlineups", mock.MagicMock(), {"season": "Z"}, ttl=3600)

    def test_long_cache_names_round_trip(self):
        # Shot-quality keys run past Windows' 260-character limit under the repo folder.
        params = {"season": "2025-26", "close_def_dist_range_nullable": "6+ Feet - Wide Open",
                  "general_range_nullable": "Less Than 10 ft", "per_mode_simple": "Totals",
                  "season_type_all_star": "Regular Season", "padding": "p" * 20}
        path = nsc._cache_path("leaguedashplayerptshot", params)
        if len(os.path.abspath(str(path))) < 262:
            self.skipTest("temp folder too short to cross the 260-character limit")
        nsc._write_cache(path, {"ok": 1})
        self.assertEqual(nsc._read_cache(path, None), {"ok": 1})
        os.unlink(nsc._fs(nsc.gz_path(path)))   # rmtree cannot see past 260 characters


class MirrorTest(_OffMixin, unittest.TestCase):
    def _mirror(self, endpoint, params, data, final=True):
        conn = sqlite3.connect(self.mirror_db)
        nba_mirror.ensure_table(conn)
        from datetime import datetime, timezone
        nba_mirror._upsert(conn, endpoint, params, "2031-32", "Regular Season",
                           datetime.now(timezone.utc), final, "live", data)
        conn.commit()
        conn.close()

    def test_mirror_serves_when_the_disk_has_nothing(self):
        (ep, cls, params, ttl), = [t for t in nba_mirror.season_targets("2031-32", "Regular Season")
                                   if t[0] == "leaguehustlestatsplayer"]
        self._mirror(ep, params, {"resultSets": [{"name": "HustleStatsPlayer", "headers": ["PLAYER_ID"],
                                                  "rowSet": [[1], [2]]}]})
        rows = nsc.NBAStatsClient().league_hustle_stats(season="2031-32")
        self.assertEqual([r["PLAYER_ID"] for r in rows], [1, 2])

    def test_zone_averages_are_trimmed_to_the_averages(self):
        data = {"resultSets": [{"name": "Shot_Chart_Detail", "headers": ["X"], "rowSet": [[1]] * 1000},
                               {"name": "LeagueAverages", "headers": ["FGA"], "rowSet": [[5]]}]}
        self.assertEqual(nba_mirror.trim("shotchartleagueavg", data),
                         {"resultSets": [{"name": "LeagueAverages", "headers": ["FGA"], "rowSet": [[5]]}]})
        self.assertIs(nba_mirror.trim("leaguedashlineups", data), data)

    def test_targets_are_the_keys_routes_ask_for(self):
        # The recorder calls the client's own methods, so a route asking for
        # the same season asks for exactly these keys.
        keys = {nsc.cache_key(e, p) for e, _, p, _ in nba_mirror.season_targets("2025-26", "Regular Season")}
        self.assertEqual(len(keys), 22 + 1 + 8 + 1 + 4 + 1)
        rec = nba_mirror._Recorder()
        rec.league_dash_lineups(season="2025-26", season_type="Regular Season",
                                group_quantity=5, measure_type="Advanced")
        self.assertIn(nsc.cache_key(rec.calls[0][0], rec.calls[0][2]), keys)
        self.assertEqual(nba_mirror.PLAY_TYPES, main_api._PLAY_TYPES)

    def test_partial_season_is_not_final(self):
        from datetime import date, datetime, timezone
        fetched = datetime(2026, 1, 5, tzinfo=timezone.utc)
        self.assertFalse(nba_mirror.is_final("2026-04-12", fetched, date(2026, 9, 28)))
        self.assertTrue(nba_mirror.is_final("2026-04-12", datetime(2026, 5, 1, tzinfo=timezone.utc),
                                            date(2026, 9, 28)))
        self.assertFalse(nba_mirror.is_final("2026-09-25", datetime(2026, 9, 27, tzinfo=timezone.utc),
                                             date(2026, 9, 28)))

    def test_refresh_skips_final_rows_and_survives_failures(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE box_scores (season TEXT, season_type TEXT, game_date TEXT)")
        conn.execute("INSERT INTO box_scores VALUES ('2031-32', 'Regular Season', '2020-04-10')")
        client = mock.MagicMock()
        client._fetch.side_effect = lambda ep, cls, params, ttl=None: (
            (_ for _ in ()).throw(RuntimeError("down")) if ep == "synergyplaytypes" else {"resultSets": []})
        counts = nba_mirror.refresh_live(conn, "2031-32", None, client=client)
        self.assertEqual(counts["failed"], 22)
        self.assertEqual(counts["copied"], 1 + 8 + 1 + 4 + 1)
        client._fetch.reset_mock()
        again = nba_mirror.refresh_live(conn, "2031-32", None, client=client)
        self.assertEqual(again["already_final"], 15)
        self.assertEqual(client._fetch.call_count, 22)   # only the failures are retried

    def test_registry_job_is_off_until_switched_on(self):
        import refresh_registry
        with mock.patch.dict(os.environ, {"NBA_MIRROR_REFRESH": ""}):
            self.assertIn("skipped", refresh_registry._server_mirror())
        self.assertIn("server_mirror", [j.name for j in refresh_registry.JOBS])


class ZoneTest(unittest.TestCase):
    def test_zones_and_distance(self):
        z = pbp_archive.classify_zone
        self.assertEqual(z(0, 10, False), ("Restricted Area", "Center(C)", "Less Than 8 ft."))
        self.assertEqual(z(-230, 20, True), ("Left Corner 3", "Left Side(L)", "24+ ft."))
        self.assertEqual(z(230, 20, True), ("Right Corner 3", "Right Side(R)", "24+ ft."))
        self.assertEqual(z(0, 260, True), ("Above the Break 3", "Center(C)", "24+ ft."))
        self.assertEqual(z(0, 100, False), ("In The Paint (Non-RA)", "Center(C)", "8-16 ft."))
        self.assertEqual(z(0, 450, True), ("Backcourt", "Back Court(BC)", "Back Court Shot"))
        self.assertEqual(pbp_archive.shot_distance(28, 47), 5)


@unittest.skipUnless(os.path.exists(DB), "needs the local archive")
class RoutesOffTest(_OffMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        for cache in (main_api.shot_chart_cache, main_api.player_shot_chart_cache,
                      main_api.game_flow_cache, main_api._playtypes_cache,
                      main_api._rebounding_cache, main_api.league_shot_averages_cache):
            cache.clear()
        self.client = TestClient(main_api.app)
        self.db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    def tearDown(self):
        self.db.close()
        super().tearDown()

    def _live_only(self, res):
        self.assertEqual(res.status_code, 503, res.text)
        body = res.json()
        self.assertFalse(body["available"])
        self.assertEqual(body["reason"], "live-only")

    def test_live_only_routes_answer_fast_and_say_why(self):
        for url in ("/api/players/1628389/matchups?season=2025-26",
                    "/api/playtypes/teams?season=2031-32",
                    "/api/stats/hustle?season=2031-32",
                    "/api/lineups?season=2031-32",
                    "/api/schedule?season=2031-32"):
            t = time.time()
            self._live_only(self.client.get(url))
            self.assertLess(time.time() - t, 2.0, url)

    def test_play_by_play_comes_from_the_archive(self):
        n = self.db.execute("SELECT COUNT(*) FROM pbp_events WHERE game_id = ?", (PBP_GAME,)).fetchone()[0]
        res = self.client.get(f"/api/games/{PBP_GAME}/play-by-play")
        self.assertEqual(res.status_code, 200)
        actions = res.json()
        self.assertEqual(len(actions), n)
        last = self.db.execute(
            "SELECT score_home FROM pbp_events WHERE game_id = ? AND score_home IS NOT NULL "
            "ORDER BY action_id DESC LIMIT 1", (PBP_GAME,)).fetchone()[0]
        self.assertEqual(max(int(a["scoreHome"]) for a in actions if a["scoreHome"]), last)
        flow = self.client.get(f"/api/games/{PBP_GAME}/game-flow")
        self.assertEqual(flow.status_code, 200)

    def test_game_shot_chart_comes_from_the_archive(self):
        fga, fgm = self.db.execute(
            "SELECT COUNT(*), SUM(shot_result = 'Made') FROM pbp_events "
            "WHERE game_id = ? AND is_field_goal = 1", (PBP_GAME,)).fetchone()
        res = self.client.get(f"/api/games/{PBP_GAME}/shot-chart")
        self.assertEqual(res.status_code, 200)
        shots = res.json()["shots"]
        self.assertEqual(len(shots), fga)
        self.assertEqual(sum(s["result"] == "made" for s in shots), fgm)

    def test_pre_archive_game_is_live_only(self):
        self._live_only(self.client.get(f"/api/games/{NO_PBP_GAME}/play-by-play"))
        self._live_only(self.client.get(f"/api/games/{NO_PBP_GAME}/shot-chart"))

    def test_player_shot_chart_from_the_archive(self):
        res = self.client.get("/api/player-shot-chart",
                              params={"player_id": 1628389, "season": "2024-25"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        fga = self.db.execute(
            "SELECT COUNT(*) FROM pbp_events e JOIN box_scores b ON b.game_id = e.game_id "
            "WHERE e.person_id = 1628389 AND e.is_field_goal = 1 AND b.season = '2024-25' "
            "AND b.season_type = 'Regular Season'").fetchone()[0]
        self.assertEqual(len(body["shots"]), fga)
        self.assertTrue(body["league_averages_unavailable"])   # empty mirror in this test
        self.assertEqual(body["league_averages"], [])
        self.assertIn("games_with_play_by_play", body["coverage"])
        self._live_only(self.client.get("/api/player-shot-chart",
                                        params={"player_id": 2544, "season": "2015-16"}))

    def test_bio_from_the_directory_without_a_fetch(self):
        row = self.db.execute(
            "SELECT p.player_id, p.position, p.college FROM players p "
            "LEFT JOIN player_bio b ON b.player_id = p.player_id "
            "WHERE b.player_id IS NULL AND p.position IS NOT NULL AND p.position != '' "
            "AND p.college IS NOT NULL AND p.college != '' LIMIT 1").fetchone()
        if not row:
            self.skipTest("every player has a bio row")
        with mock.patch.object(main_api, "commonplayerinfo",
                               mock.MagicMock(CommonPlayerInfo=mock.MagicMock(side_effect=AssertionError))):
            res = self.client.get(f"/api/players/{row[0]}")
        self.assertEqual(res.status_code, 200)
        bio = res.json()["bio"]
        self.assertEqual(bio["position"], row[1])
        self.assertEqual(bio["college"], row[2])
        self.assertEqual(bio["born"], "N/A")   # not in any table we hold: unknown, not guessed

    def test_heat_calendar_for_a_pre_archive_player_has_totals(self):
        # Found in the rehearsal: the player page crashed on this answer.
        pid = self.db.execute(
            "SELECT p.player_id FROM players p WHERE NOT EXISTS "
            "(SELECT 1 FROM player_game_log g WHERE g.player_id = p.player_id) LIMIT 1").fetchone()
        if not pid:
            self.skipTest("every player has game logs")
        body = self.client.get(f"/api/players/{pid[0]}/heat-calendar").json()
        self.assertEqual(body["games"], 0)
        self.assertEqual(body["totals"], {"pts": None, "mean_game_score": None})

    def test_refused_pass_fetch_is_not_recorded_as_no_passes(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE player_season_totals (player_id INT, team_id INT, season TEXT, season_type TEXT);"
            "CREATE TABLE players (player_id INT, full_name TEXT);"
            "INSERT INTO player_season_totals VALUES (1, 99, '2031-32', 'Regular Season');"
            "INSERT INTO players VALUES (1, 'A Player');")
        with self.assertRaises(nsc.LiveFetchDisabled):
            main_api._ensure_team_passing(conn, 99, "2031-32", "Regular Season")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM team_passing_fetch_log").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
