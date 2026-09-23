"""
Scheduled_Jobs_Test.py
======================
Pins what the scheduled jobs in src/Sports/run_scheduled.py actually run.

A recorder that exists but is not in a job records nothing, and nothing about
that looks broken. These tests fail if the NBA injury recorder or the NBA odds
repair falls out of its job, points at a file that is not there, or loses the
flags that keep an unattended run inside its budget.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Sports import run_scheduled as S  # noqa: E402


def _steps(mode):
    return {name: argv for name, argv in S.JOBS[mode]}


class TestJobs(unittest.TestCase):

    def test_every_step_points_at_a_real_script(self):
        for mode, steps in S.JOBS.items():
            for name, argv in steps:
                self.assertTrue(os.path.exists(os.path.join(S.REPO_ROOT, argv[0])),
                                f"{mode}/{name}: {argv[0]} does not exist")

    def test_nba_injury_recorder_runs_hourly(self):
        self.assertEqual(_steps("hourly").get("injury recorder (NBA)"),
                         ["src/Sports/nba/poll_injuries.py"])

    def test_nba_odds_repair_runs_daily_unattended_after_the_nfl_one(self):
        names = [n for n, _ in S.JOBS["daily"]]
        argv = _steps("daily").get("repair missed odds (NBA)")
        self.assertIsNotNone(argv)
        self.assertEqual(argv[:3], ["src/Sports/repair_odds.py", "--sport", "nba"])
        self.assertIn("--unattended", argv, "without it the 3-day / 90-credit guards are off")
        self.assertGreater(names.index("repair missed odds (NBA)"),
                           names.index("repair missed odds (NFL)"))


if __name__ == "__main__":
    unittest.main()
