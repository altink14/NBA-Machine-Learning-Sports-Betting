"""Daily jobs run daily, and a failed preflight check is always named somewhere.

Temp databases and folders only.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import refresh_registry as rr
import preflight_opening_night as pf


class DueTest(unittest.TestCase):

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(rr.SCHEMA)
        self.job = rr.Job("player_directory", 1, lambda: "", "test")
        self.now = datetime(2026, 9, 23, 13, 4, 34, tzinfo=timezone.utc)

    def last_ran(self, when, status="ok"):
        self.conn.execute("INSERT OR REPLACE INTO ingest_runs (name, last_run_at, last_status) VALUES (?,?,?)",
                          ("player_directory", when.isoformat(), status))

    def test_yesterdays_slightly_later_finish_is_still_due(self):
        # The real case: finished 09:11:54 local yesterday, checked 09:04:34 today.
        self.last_ran(datetime(2026, 9, 22, 13, 11, 54, tzinfo=timezone.utc))
        self.assertTrue(rr.due(self.job, self.now, self.conn))

    def test_not_twice_in_one_morning(self):
        self.last_ran(self.now - timedelta(minutes=30))
        self.assertFalse(rr.due(self.job, self.now, self.conn))

    def test_weekly_job_runs_on_the_seventh_day(self):
        weekly = rr.Job("hall_of_fame", 7, lambda: "", "test")
        self.conn.execute("INSERT INTO ingest_runs (name, last_run_at, last_status) VALUES (?,?,?)",
                          ("hall_of_fame", (self.now - timedelta(days=7) + timedelta(minutes=10)).isoformat(), "ok"))
        self.assertTrue(rr.due(weekly, self.now, self.conn))

    def test_a_failure_retries_the_next_morning(self):
        self.last_ran(self.now - timedelta(hours=1), status="failed")
        self.assertTrue(rr.due(self.job, self.now, self.conn))


class PreflightReportTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="preflight_report_")
        pf.results.clear()

    def tearDown(self):
        pf.results.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_result_is_written_and_failures_are_kept(self):
        pf.results.extend([(pf.OK, "a thing works", ""),
                           (pf.BAD, "frequent job has a log", "no line in 24 h — check the task")])
        bad = [r for r in pf.results if r[0] == pf.BAD]
        with mock.patch.object(pf, "REPO_ROOT", self.tmp):
            pf._write_report(date(2026, 9, 23), bad, [])
            pf._write_report(date(2026, 9, 24), bad, [])
        latest = open(os.path.join(self.tmp, "logs", "preflight_latest.txt"), encoding="utf-8").read()
        self.assertIn("1 pass, 0 not yet, 1 wrong", latest)
        self.assertIn("frequent job has a log -- no line in 24 h — check the task", latest)
        history = open(os.path.join(self.tmp, "logs", "preflight_history.log"), encoding="utf-8").read()
        self.assertEqual(history.count("WRONG frequent job has a log"), 2, "history accumulates")


class EncodingTest(unittest.TestCase):

    def test_children_of_the_scheduled_jobs_write_utf8(self):
        import daily_update  # noqa: F401  (sets the variable on import)
        self.assertEqual(os.environ.get("PYTHONIOENCODING"), "utf-8")
        out = subprocess.run([sys.executable, "-c", "print('  WRONG    x — y → z')"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        self.assertIn("— y → z", out, "a dash must survive the round trip")

    def test_the_daily_log_can_write_what_the_children_print(self):
        import logging
        path = os.path.join(tempfile.mkdtemp(prefix="log_enc_"), "daily_update.log")
        h = logging.FileHandler(path, encoding="utf-8", errors="backslashreplace")
        rec = logging.LogRecord("t", logging.ERROR, __file__, 1, "  WRONG � → x", None, None)
        h.emit(rec)
        h.close()
        self.assertIn("WRONG", open(path, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
