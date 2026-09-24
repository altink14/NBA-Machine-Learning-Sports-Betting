"""The daily job's integrity-audit step reports failures and never raises."""
import subprocess
import unittest
from unittest import mock

import daily_update


def _done(code, out):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr="")


class DailyAuditStepTest(unittest.TestCase):

    def test_clean_audit_is_zero(self):
        with mock.patch.object(daily_update.subprocess, "run", return_value=_done(0, "   PASS  a\n\nall checks pass\n")):
            self.assertEqual(daily_update.run_integrity_audit("2025-26"), 0)

    def test_failures_are_counted_and_logged(self):
        out = ("   PASS  a\n   FAIL  team points: 2\n           {'game_id': 'x'}\n"
               "  ERROR  quarters: no such column\n\n2 of 16 checks failed\n")
        with mock.patch.object(daily_update.subprocess, "run", return_value=_done(1, out)), \
             self.assertLogs(daily_update.logger, level="ERROR") as logs:
            self.assertEqual(daily_update.run_integrity_audit("2025-26"), 2)
        self.assertTrue(any("team points" in m for m in logs.output))

    def test_a_crash_is_minus_one_not_an_exception(self):
        with mock.patch.object(daily_update.subprocess, "run", side_effect=OSError("boom")):
            self.assertEqual(daily_update.run_integrity_audit("2025-26"), -1)


if __name__ == "__main__":
    unittest.main()
