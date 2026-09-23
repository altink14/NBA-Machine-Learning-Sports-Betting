"""Answers that used to look complete while hiding a failure now say so.

Reads the real TeamData read-only; every path that could reach stats.nba.com
is mocked to fail.
"""

import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest import mock

from fastapi.testclient import TestClient

import main_api
import preflight_opening_night as pf


class StandingsTest(unittest.TestCase):

    def setUp(self):
        self.c = TestClient(main_api.app)

    def status(self, q):
        return self.c.get("/api/stats/standings?" + q).status_code

    def test_an_archived_season_returns_its_teams(self):
        r = self.c.get("/api/stats/standings?season=2024-25")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 30)

    def test_a_season_outside_the_archive_is_404_not_an_empty_league(self):
        self.assertEqual(self.status("season=1990-91"), 404)

    def test_malformed_seasons_are_400(self):
        for bad in ("2026", "2025-27", "25-26", "abcd-ef"):
            self.assertEqual(self.status(f"season={bad}"), 400, bad)
        self.assertEqual(self.c.get("/api/seasons/2026").status_code, 400)

    def test_play_in_has_no_standings_table(self):
        self.assertEqual(self.status("season=2024-25&season_type=PlayIn"), 400)

    def test_the_new_season_before_its_first_game_is_a_genuine_empty(self):
        future = f"{int(main_api.CURRENT_SEASON[:4]) + 1}-{(int(main_api.CURRENT_SEASON[:4]) + 2) % 100:02d}"
        r = self.c.get(f"/api/stats/standings?season={future}")
        self.assertEqual((r.status_code, r.json()), (200, []))


class LeagueAveragesTest(unittest.TestCase):

    def test_a_failed_fetch_is_none_not_an_empty_league(self):
        main_api.league_shot_averages_cache.pop("2003-04_Regular Season", None)
        with mock.patch("src.Utils.nba_stats_client.get_client", side_effect=RuntimeError("blocked")):
            self.assertIsNone(main_api._get_league_shot_averages("2003-04"))
        self.assertNotIn("2003-04_Regular Season", main_api.league_shot_averages_cache,
                         "a failure must not be cached")


class MilestonesTest(unittest.TestCase):

    def test_players_that_could_not_be_checked_are_reported(self):
        with mock.patch.object(main_api, "_career_official_freshness", side_effect=RuntimeError("down")):
            d = TestClient(main_api.app).get("/api/stats/milestones").json()
        self.assertGreater(len(d["unavailable"]), 0)
        self.assertEqual(d["players_checked"], 0)
        self.assertEqual(d["count"], 0)


class DnaTest(unittest.TestCase):

    def test_a_failed_tracking_fetch_is_flagged_not_just_null(self):
        boom = mock.Mock(side_effect=RuntimeError("tracking down"))
        with mock.patch.object(main_api, "get_shot_quality", boom), \
                mock.patch.object(main_api, "_rebounding_for", boom), \
                mock.patch.object(main_api, "_clutch_for", boom):
            r = TestClient(main_api.app).get("/api/2k/dna/2544?season=2023-24")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["unavailable"], {"shot_quality": True, "rebounding": True, "clutch": True})
        self.assertIsNone(d["shot_quality"])


class PreflightOperationsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="preflight_ops_")
        pf.results.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        pf.results.clear()

    def run_ops(self, today, backup_dirs=(), env=None):
        for d in backup_dirs:
            os.makedirs(os.path.join(self.tmp, d))
            open(os.path.join(self.tmp, d, "README.txt"), "w").close()
        e = {"BACKUP_ROOT": self.tmp, "LEDGER_SYNC_URL": "", "LEDGER_SYNC_SECRET": ""}
        e.update(env or {})
        with mock.patch.dict(os.environ, e), mock.patch("dotenv.load_dotenv"):
            pf.check_operations(today)
        return {label: state for state, label, _ in pf.results}

    def test_recent_backup_passes_old_one_reminds(self):
        r = self.run_ops(date(2026, 9, 25), ["2026-09-23"])
        self.assertEqual(r["a recent data backup exists"], pf.OK)
        pf.results.clear()
        shutil.rmtree(self.tmp); os.makedirs(self.tmp)
        r = self.run_ops(date(2026, 11, 20), ["2026-09-23"])
        self.assertEqual(r["a recent data backup exists"], pf.PENDING)

    def test_an_incomplete_backup_does_not_count(self):
        os.makedirs(os.path.join(self.tmp, "2026-09-24"))   # no README: the run failed
        r = self.run_ops(date(2026, 9, 25))
        self.assertEqual(r["a recent data backup exists"], pf.PENDING)

    def test_a_stale_opening_night_is_wrong(self):
        r = self.run_ops(pf.OPENING_NIGHT.replace(year=pf.OPENING_NIGHT.year + 1))
        self.assertEqual(r["OPENING_NIGHT is this season's"], pf.BAD)

    def test_mirror_configured_passes(self):
        r = self.run_ops(date(2026, 9, 25), env={"LEDGER_SYNC_URL": "https://x", "LEDGER_SYNC_SECRET": "s"})
        self.assertEqual(r["the ledger is mirrored to the public server"], pf.OK)


if __name__ == "__main__":
    unittest.main()
