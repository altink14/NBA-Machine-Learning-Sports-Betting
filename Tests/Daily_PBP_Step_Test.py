"""The daily play-by-play step: in season only, reads backfill_pbp's summary."""
import subprocess
import unittest
from unittest import mock

import daily_update


def _done(code, out=""):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout="", stderr=out)


class DailyPBPStepTest(unittest.TestCase):

    def run_step(self, results, in_season=True):
        with mock.patch.object(daily_update, "_nba_games_expected", lambda *a, **k: in_season), \
             mock.patch.object(daily_update.subprocess, "run", side_effect=results) as run:
            ok = daily_update.refresh_play_by_play("2026-27")
        return ok, run

    def test_offseason_is_skipped_and_ok(self):
        ok, run = self.run_step([], in_season=False)
        self.assertTrue(ok)
        run.assert_not_called()

    def test_new_games_rebuild_the_runs(self):
        ok, run = self.run_step([
            _done(0, "INFO - 100 of 110 games already have play-by-play; 10 to fetch.\nINFO - Done in 0.4 min. 4,300 events written. 0 failure(s).\n"),
            _done(0, "rebuilt"),
        ])
        self.assertTrue(ok)
        self.assertEqual(run.call_count, 2)
        self.assertIn("ingest_scoring_runs.py", run.call_args_list[1].args[0][1])

    def test_nothing_new_skips_the_rebuild(self):
        ok, run = self.run_step([_done(0, "INFO - 110 of 110 games already have play-by-play; 0 to fetch.\nINFO - Nothing to do.\n")])
        self.assertTrue(ok)
        self.assertEqual(run.call_count, 1)

    def test_a_failed_game_fails_the_step(self):
        ok, _ = self.run_step([
            _done(0, "INFO - 9 to fetch.\nINFO - Done in 0.4 min. 3,000 events written. 2 failure(s).\n"),
            _done(0, "rebuilt"),
        ])
        self.assertFalse(ok)

    def test_a_crash_is_a_failure_not_an_exception(self):
        ok, _ = self.run_step([OSError("boom")])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
