"""Crews from boxscoresummaryv3 land on the same officials v2 would have named.

No network: the client's _fetch is mocked, and the backfill runs against a
throwaway database.
"""

import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

from src.Utils.nba_stats_client import NBAStatsClient

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "backfill_officials", os.path.join(REPO, "src", "Process-Data", "backfill_officials.py"))
backfill_officials = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill_officials)


def v3_payload(game_id, officials):
    return {"boxScoreSummary": {"gameId": game_id, "officials": officials}}


ZARBA = {"personId": 2534, "firstName": "Zach", "familyName": "Zarba", "jerseyNum": "15  ",
         "name": "Zach Zarba", "nameI": "Z. Zarba", "assignment": "OFFICIAL1"}
DALEN = {"personId": 200833, "firstName": "Eric", "familyName": "Dalen", "jerseyNum": "37  ",
         "name": "Eric Dalen", "nameI": "E. Dalen", "assignment": "OFFICIAL2"}


class OfficialsV3ClientTest(unittest.TestCase):

    def test_maps_to_the_v2_row_shape_and_strips_jersey_padding(self):
        c = NBAStatsClient()
        with mock.patch.object(c, "_fetch", return_value=v3_payload("0022500001", [ZARBA, DALEN])):
            crew = c.officials_v3("0022500001")
        self.assertEqual(crew, [
            {"OFFICIAL_ID": 2534, "FIRST_NAME": "Zach", "LAST_NAME": "Zarba", "JERSEY_NUM": "15"},
            {"OFFICIAL_ID": 200833, "FIRST_NAME": "Eric", "LAST_NAME": "Dalen", "JERSEY_NUM": "37"},
        ])

    def test_an_answer_about_another_game_is_refused(self):
        c = NBAStatsClient()
        with mock.patch.object(c, "_fetch", return_value=v3_payload("0022500999", [ZARBA])):
            with self.assertRaises(ValueError):
                c.officials_v3("0022500001")

    def test_no_crew_is_an_empty_list_not_an_error(self):
        c = NBAStatsClient()
        with mock.patch.object(c, "_fetch", return_value=v3_payload("0022500001", [])):
            self.assertEqual(c.officials_v3("0022500001"), [])

    def test_fresh_bypasses_the_cache(self):
        c = NBAStatsClient()
        with mock.patch.object(c, "_fetch", return_value=v3_payload("0022500001", [])) as f:
            c.officials_v3("0022500001", fresh=True)
        self.assertEqual(f.call_args.kwargs["ttl"], 0)


class BackfillFallbackTest(unittest.TestCase):
    """The order the backfill asks in, and what it records."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="officials_v3_test_")
        self.db = os.path.join(self.dir, "TeamData.sqlite")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, game_date TEXT)")
        conn.executemany("INSERT INTO box_scores VALUES (?,?,?)", [
            ("G_NEW", "2025-26", "2026-01-02"),     # never asked
            ("G_EMPTY", "2025-26", "2026-01-01"),   # asked before, v2 said no crew
        ])
        backfill_officials.ensure_schema(conn)
        conn.execute("INSERT INTO officials_fetch (game_id, fetched_at, n_officials) "
                     "VALUES ('G_EMPTY', '2026-08-30', 0)")
        conn.commit()
        conn.close()
        self.client = mock.Mock()
        self.client.boxscore_summary.return_value = {"Officials": []}
        self.client.officials_v3.return_value = [
            {"OFFICIAL_ID": 2534, "FIRST_NAME": "Zach", "LAST_NAME": "Zarba", "JERSEY_NUM": "15"}]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_backfill(self, *extra):
        argv = ["backfill_officials.py", "--db", self.db, "--seasons", "2025-26", *extra]
        with mock.patch.object(backfill_officials, "get_client", return_value=self.client), \
                mock.patch.object(sys, "argv", argv):
            self.assertEqual(backfill_officials.main(), 0)
        conn = sqlite3.connect(self.db)
        try:
            return {r[0]: r[1:] for r in conn.execute(
                "SELECT game_id, n_officials, source FROM officials_fetch")}
        finally:
            conn.close()

    def test_empty_v2_answer_falls_through_to_v3(self):
        fetch = self.run_backfill()
        self.client.boxscore_summary.assert_called_once_with("G_NEW")
        self.client.officials_v3.assert_called_once_with("G_NEW", fresh=False)
        self.assertEqual(fetch["G_NEW"], (1, "v3"))

    def test_v2_crew_is_used_and_v3_not_asked(self):
        self.client.boxscore_summary.return_value = {"Officials": [
            {"OFFICIAL_ID": 2534, "FIRST_NAME": "Zach", "LAST_NAME": "Zarba", "JERSEY_NUM": "15"}]}
        fetch = self.run_backfill()
        self.client.officials_v3.assert_not_called()
        self.assertEqual(fetch["G_NEW"], (1, "v2"))

    def test_retry_goes_straight_to_v3_fresh(self):
        fetch = self.run_backfill("--retry-empty")
        # G_EMPTY: v2 already said no, so only v3 is asked, bypassing the cache.
        self.client.officials_v3.assert_any_call("G_EMPTY", fresh=True)
        self.assertNotIn(mock.call("G_EMPTY"), self.client.boxscore_summary.call_args_list)
        self.assertEqual(fetch["G_EMPTY"], (1, "v3"))

    def test_a_crew_lands_on_the_official_v2_already_knew(self):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO officials VALUES (2534, 'Zach', 'Zarba', '15')")
        conn.commit()
        conn.close()
        self.run_backfill("--retry-empty")
        conn = sqlite3.connect(self.db)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM officials").fetchone()[0], 1)
            self.assertEqual(sorted(r[0] for r in conn.execute(
                "SELECT game_id FROM game_officials WHERE official_id = 2534")),
                ["G_EMPTY", "G_NEW"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
