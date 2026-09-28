"""
Job_Health_Test.py
==================
Pins job_health.py, the read-only checker for every scheduled job.

What it must never do is the thing it exists to catch: report "ok" for a job
that did not run, ran and failed, or ran and left nothing behind. So each
test builds the evidence for one of those shapes -- a fixture log, a fake
Task Scheduler answer, a throwaway database -- and checks the verdict.

No network, no PowerShell, no real logs or databases: Task Scheduler is a
dict, every file lives in a temp folder, and the notifier's HTTP call is a
stub that records what it was asked to send.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import job_health as jh  # noqa: E402

NOW = datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc)      # in season, mid-week
OFFSEASON = datetime(2026, 8, 10, 18, 0, tzinfo=timezone.utc)


def stamp(dt):
    """A log timestamp exactly as logging writes it: local wall clock."""
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S") + ",123"


def sched_log(starts, mode="frequent", fail_at=None, unfinished_last=False):
    lines = []
    for i, t in enumerate(starts):
        lines.append(f"{stamp(t)} - INFO - === {mode} run starting ({t.isoformat()}) ===")
        if unfinished_last and i == len(starts) - 1:
            break
        if fail_at is not None and i == fail_at:
            lines.append(f"{stamp(t)} - ERROR - odds recorder (NBA): exit 1 | Traceback: boom")
            lines.append(f"{stamp(t)} - ERROR - === {mode} run finished with 1 failure(s): odds recorder (NBA) ===")
        else:
            lines.append(f"{stamp(t)} - INFO - odds recorder (NBA): ok | skipping: nearest tip is 5h away")
            lines.append(f"{stamp(t)} - INFO - publish ledger: ok | SKIPPED: LEDGER_SYNC_URL / LEDGER_SYNC_SECRET not set")
            lines.append(f"{stamp(t)} - INFO - === {mode} run finished clean ===")
    return lines


def task(last_run, result=0, state="Ready", repeat="PT15M", missed=0, daily=False, next_run=None):
    trig = ({"kind": "MSFT_TaskDailyTrigger", "repeat": None, "days_interval": 1, "enabled": True} if daily
            else {"kind": "MSFT_TaskTimeTrigger", "repeat": repeat, "days_interval": None, "enabled": True})
    return {"name": "x", "state": state, "last_run": last_run.strftime("%Y-%m-%dT%H:%M:%SZ") if last_run else None,
            "last_result": result, "next_run": (next_run or NOW + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "missed_runs": missed, "logon_type": "Interactive", "triggers": [trig], "action": "python x.py"}


FREQ = jh.TASKS[0]
DAILY = next(s for s in jh.TASKS if s.job_id == "daily_update")


class SchedulerTest(unittest.TestCase):

    def sched(self, info, name=FREQ.task_name):
        return {"available": True, "error": None, "tasks": {name: info}}

    def test_healthy_task_is_ok(self):
        c = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(minutes=5))), NOW)
        self.assertEqual(c["status"], jh.OK, c["summary"])

    def test_missing_task_fails(self):
        c = jh.check_scheduler(FREQ, {"available": True, "tasks": {}}, NOW)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("does not exist", c["summary"])

    def test_disabled_task_fails(self):
        c = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(minutes=5), state="Disabled")), NOW)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("DISABLED", c["summary"])

    def test_nonzero_last_result_fails_and_is_explained(self):
        c = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(minutes=5), result=1)), NOW)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("exit 1", c["summary"])

    def test_running_now_is_not_a_failure(self):
        c = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(minutes=2), result=267009)), NOW)
        self.assertEqual(c["status"], jh.OK)

    def test_overdue_warns_then_fails(self):
        warn = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(minutes=50))), NOW)
        dead = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(hours=5))), NOW)
        self.assertEqual(warn["status"], jh.WARN)
        self.assertEqual(dead["status"], jh.FAIL)
        self.assertIn("stopped running", dead["summary"])

    def test_changed_cadence_fails(self):
        c = jh.check_scheduler(FREQ, self.sched(task(NOW - timedelta(minutes=5), repeat="PT1H")), NOW)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("PT1H", c["summary"])

    def test_daily_missed_run_is_said_even_when_recent(self):
        info = task(NOW - timedelta(hours=20), daily=True, missed=1)
        c = jh.check_scheduler(DAILY, self.sched(info, DAILY.task_name), NOW)
        self.assertEqual(c["status"], jh.WARN)
        self.assertIn("1 missed run", c["summary"])

    def test_scheduler_unavailable_is_unknown_not_ok(self):
        c = jh.check_scheduler(FREQ, {"available": False, "tasks": {}, "error": "no"}, NOW)
        self.assertEqual(c["status"], jh.UNKNOWN)

    def test_reader_parses_powershell_json_and_single_objects(self):
        one = {"name": "BettingBuddy Odds Recorder", "state": "Ready", "last_result": 0}
        fake = mock.Mock(return_value=mock.Mock(returncode=0, stdout=json.dumps(one), stderr=""))
        got = jh.read_task_scheduler(runner=fake)
        self.assertTrue(got["available"])
        self.assertIn("BettingBuddy Odds Recorder", got["tasks"])
        # Read-only: the only command it may run is the Get-ScheduledTask query.
        cmd = " ".join(fake.call_args[0][0])
        self.assertIn("Get-ScheduledTask", cmd)
        for verb in ("Register-", "Set-ScheduledTask", "Disable-", "Enable-", "Start-ScheduledTask",
                     "Unregister-", "schtasks /create", "schtasks /change"):
            self.assertNotIn(verb, cmd)

    def test_reader_failure_is_reported(self):
        fake = mock.Mock(return_value=mock.Mock(returncode=1, stdout="", stderr="access denied"))
        got = jh.read_task_scheduler(runner=fake)
        self.assertFalse(got["available"])
        self.assertIn("access denied", got["error"])


class LogTest(unittest.TestCase):

    def runs(self, lines, mode="frequent"):
        return jh.parse_scheduled_log(lines, mode)

    def every_15(self, hours=24, gap=None):
        starts, t = [], NOW - timedelta(hours=hours) + timedelta(minutes=1)
        while t <= NOW:
            if not (gap and gap[0] <= t < gap[1]):
                starts.append(t)
            t += timedelta(minutes=15)
        return starts

    def test_clean_full_coverage_is_ok(self):
        starts = self.every_15()
        c = jh.check_log(FREQ, self.runs(sched_log(starts)), NOW, starts[-1])
        self.assertEqual(c["status"], jh.OK, c["summary"])
        self.assertEqual(c["evidence"]["runs_last_24h"], 96)

    def test_last_run_with_failures_names_the_step(self):
        starts = self.every_15()
        runs = self.runs(sched_log(starts, fail_at=len(starts) - 1))
        c = jh.check_log(FREQ, runs, NOW, starts[-1])
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("odds recorder (NBA)", c["summary"])
        self.assertTrue(any("boom" in e for e in c["evidence"]["errors"]))

    def test_sleeping_pc_fails_on_coverage_and_names_the_gap(self):
        gap = (NOW - timedelta(hours=20), NOW - timedelta(hours=5))
        starts = self.every_15(gap=gap)
        c = jh.check_log(FREQ, self.runs(sched_log(starts)), NOW, starts[-1])
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("Longest silence", c["summary"])
        self.assertGreater(c["evidence"]["longest_gap"]["hours"], 14.9)

    def test_started_by_windows_but_never_logged_fails(self):
        # The venv broke: Task Scheduler keeps starting it, nothing is logged.
        starts = self.every_15(hours=3)[:4]
        c = jh.check_log(FREQ, self.runs(sched_log(starts)), NOW, NOW - timedelta(minutes=14))
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("did not get as far as logging", c["summary"])

    def test_unfinished_run_past_timeout_is_a_crash(self):
        starts = self.every_15()
        starts = starts[:-4]
        runs = self.runs(sched_log(starts, unfinished_last=True))
        c = jh.check_log(FREQ, runs, NOW, starts[-1])
        self.assertIn("never finished", c["summary"])

    def test_missing_log_after_a_run_fails(self):
        c = jh.check_log(FREQ, None, NOW, NOW - timedelta(minutes=5))
        self.assertEqual(c["status"], jh.FAIL)

    def test_daily_update_ok_but_preflight_wrong_warns(self):
        t = NOW - timedelta(hours=3)
        lines = [f"{stamp(t)} - INFO - === Daily update starting for season 2026-27 ===",
                 f"{stamp(t)} - ERROR - Preflight found problems — 25 pass, 4 not yet, 2 wrong",
                 f"{stamp(t)} - INFO - === Daily update finished OK (predictions: logged, "
                 "preflight: 2 WRONG, audit: clean) ==="]
        runs = jh.parse_daily_update_log(lines)
        self.assertEqual(runs[-1]["verdicts"]["preflight"], "2 WRONG")
        c = jh.check_log(DAILY, runs, NOW, t)
        self.assertEqual(c["status"], jh.WARN)
        self.assertIn("2 WRONG", c["summary"])

    def test_daily_update_with_errors_fails(self):
        t = NOW - timedelta(hours=3)
        lines = [f"{stamp(t)} - INFO - === Daily update starting for season 2026-27 ===",
                 f"{stamp(t)} - ERROR - Prediction run failed: no odds",
                 f"{stamp(t)} - ERROR - === Daily update finished WITH ERRORS: prediction logging, backfill ==="]
        c = jh.check_log(DAILY, jh.parse_daily_update_log(lines), NOW, t)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("prediction logging", c["summary"])
        self.assertIn("Prediction run failed: no odds", c["evidence"]["errors"])

    def test_ledger_publish_skipped_is_off_not_ok(self):
        runs = self.runs(sched_log([NOW - timedelta(minutes=5)], mode="hourly"), "hourly")
        self.assertEqual(jh.check_ledger_publish(runs)["status"], jh.OFF)


class DataTraceTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def db(self, name, sql):
        path = os.path.join(self.tmp.name, name)
        conn = sqlite3.connect(path)
        conn.executescript(sql)
        conn.commit()
        conn.close()
        return path

    def test_injury_poll_failed_or_stale(self):
        schema = ("CREATE TABLE nba_injury_polls (id INTEGER PRIMARY KEY, started_at TEXT, status TEXT, "
                  "error TEXT);")
        ok = (NOW - timedelta(minutes=20)).isoformat()
        p = self.db("a.sqlite", schema + f"INSERT INTO nba_injury_polls VALUES (1,'{ok}','ok',NULL);")
        self.assertEqual(jh.check_nba_injury_polls(NOW, p)["status"], jh.OK)
        p = self.db("b.sqlite", schema + f"INSERT INTO nba_injury_polls VALUES (1,'{ok}','failed','HTTP 503');")
        c = jh.check_nba_injury_polls(NOW, p)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertEqual(c["evidence"]["error"], "HTTP 503")
        old = (NOW - timedelta(hours=6)).isoformat()
        p = self.db("c.sqlite", schema + f"INSERT INTO nba_injury_polls VALUES (1,'{old}','ok',NULL);")
        self.assertEqual(jh.check_nba_injury_polls(NOW, p)["status"], jh.FAIL)

    def test_nfl_pick_ungraded_long_after_kickoff_fails(self):
        schema = "CREATE TABLE ledger (id INTEGER PRIMARY KEY, result TEXT, event_start_utc TEXT, graded_at TEXT);"
        stale = (NOW - timedelta(days=3)).isoformat()
        fresh = (NOW - timedelta(hours=6)).isoformat()
        p = self.db("l.sqlite", schema + f"INSERT INTO ledger VALUES (1,'pending','{fresh}',NULL);")
        self.assertEqual(jh.check_nfl_grading(NOW, p)["status"], jh.OK)
        p = self.db("m.sqlite", schema + f"INSERT INTO ledger VALUES (1,'pending','{stale}',NULL);")
        c = jh.check_nfl_grading(NOW, p)
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("grader is running but not grading", c["summary"])

    def test_predictions_log_in_season_vs_offseason(self):
        schema = "CREATE TABLE predictions_log (id INTEGER PRIMARY KEY, sport TEXT, logged_at TEXT);"
        p = self.db("p.sqlite", schema)
        self.assertEqual(jh.check_nba_predictions_log(OFFSEASON, p)["status"], jh.OFF)
        self.assertEqual(jh.check_nba_predictions_log(NOW, p)["status"], jh.FAIL)   # empty in season
        t = (NOW - timedelta(hours=10)).isoformat()
        p = self.db("q.sqlite", schema + f"INSERT INTO predictions_log VALUES (1,'NBA','{t}');")
        self.assertEqual(jh.check_nba_predictions_log(NOW, p)["status"], jh.OK)

    def test_team_stats_snapshot_age(self):
        today = NOW.astimezone().date()
        p = self.db("t.sqlite", f'CREATE TABLE "{today - timedelta(days=5)}" (x);')
        self.assertEqual(jh.check_team_stats_snapshot(NOW, p)["status"], jh.FAIL)
        p = self.db("u.sqlite", f'CREATE TABLE "{today}" (x);')
        self.assertEqual(jh.check_team_stats_snapshot(NOW, p)["status"], jh.OK)

    def test_team_stats_quiet_before_opening_night_is_not_a_failure(self):
        oct5 = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
        p = self.db("v.sqlite", 'CREATE TABLE "2026-09-30" (x);')
        with mock.patch.object(jh, "_opening_night", return_value=datetime(2026, 10, 20).date()):
            self.assertEqual(jh.check_team_stats_snapshot(oct5, p)["status"], jh.OFF)

    def test_game_without_play_by_play_is_named(self):
        sql = ("CREATE TABLE box_scores (game_id TEXT, season TEXT, game_date TEXT);"
               "CREATE TABLE pbp_events (game_id TEXT, action_id INTEGER);"
               "INSERT INTO box_scores VALUES ('0022600001','2026-27','2026-11-01'),"
               "('0022600002','2026-27','2026-11-02'),('0022600003','2026-27','2026-11-10');"
               "INSERT INTO pbp_events VALUES ('0022600001', 1);")
        c = jh.check_archive_freshness(NOW, self.db("g.sqlite", sql))
        self.assertEqual(c["status"], jh.FAIL)
        self.assertEqual(c["evidence"]["missing_pbp"], 1)    # today's game is not due yet
        self.assertIn("0022600002", c["summary"])

    def test_odds_heartbeat_missing_table_is_unknown_and_failed_poll_fails(self):
        p = self.db("o.sqlite", "CREATE TABLE odds_snapshots (captured_at TEXT);")
        self.assertEqual(jh.check_odds_heartbeat(NOW, p)["status"], jh.UNKNOWN)
        t = (NOW - timedelta(hours=2)).isoformat()
        p = self.db("o2.sqlite", "CREATE TABLE odds_polls (id INTEGER PRIMARY KEY, polled_at TEXT, sport TEXT, "
                    f"status TEXT, error TEXT); INSERT INTO odds_polls VALUES (1,'{t}','NBA','failed','401');")
        self.assertEqual(jh.check_odds_heartbeat(NOW, p)["status"], jh.FAIL)

    def test_failed_periodic_ingest_fails(self):
        sql = ("CREATE TABLE ingest_runs (name TEXT PRIMARY KEY, last_run_at TEXT, last_status TEXT, "
               "last_detail TEXT, consecutive_failures INTEGER);"
               f"INSERT INTO ingest_runs VALUES ('officials','{NOW.isoformat()}','failed','timeout',3);")
        c = jh.check_periodic_ingests(NOW, self.db("i.sqlite", sql))
        self.assertEqual(c["status"], jh.FAIL)
        self.assertIn("officials FAILED 3", c["summary"])

    def test_preflight_report_wrong_items_are_listed(self):
        path = os.path.join(self.tmp.name, "preflight_latest.txt")
        t = (NOW - timedelta(hours=2)).astimezone().replace(tzinfo=None).isoformat(timespec="seconds")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"preflight {t} (as of 2026-11-10): 25 pass, 4 not yet, 1 wrong\n"
                     "   PASS  a thing -- fine\n"
                     "   FAIL  the hourly job is firing on schedule -- 9/24 runs in 24h (38%) — asleep\n")
        c = jh.check_preflight(NOW, path)
        self.assertEqual(c["status"], jh.WARN)
        self.assertIn("the hourly job is firing on schedule (9/24 runs in 24h (38%))", c["summary"])

    def test_backup_age_and_unplugged_drive(self):
        root = os.path.join(self.tmp.name, "bk")
        self.assertEqual(jh.check_backup(NOW, root)["status"], jh.UNKNOWN)
        old = os.path.join(root, (NOW.date() - timedelta(days=20)).isoformat())
        os.makedirs(old)
        open(os.path.join(old, "README.txt"), "w").close()
        self.assertEqual(jh.check_backup(NOW, root)["status"], jh.FAIL)
        # A half-finished copy (no README) does not count as a backup.
        os.makedirs(os.path.join(root, NOW.date().isoformat()))
        self.assertEqual(jh.check_backup(NOW, root)["status"], jh.FAIL)


class RollupTest(unittest.TestCase):

    def test_worst_wins_and_unknown_is_never_ok(self):
        self.assertEqual(jh.worst([jh.OK, jh.OFF]), jh.OK)
        self.assertEqual(jh.worst([jh.OK, jh.UNKNOWN]), jh.UNKNOWN)
        self.assertEqual(jh.worst([jh.UNKNOWN, jh.WARN, jh.OK]), jh.WARN)
        self.assertEqual(jh.worst([jh.WARN, jh.FAIL]), jh.FAIL)

    def test_collect_survives_a_broken_check_and_flags_unmonitored_tasks(self):
        sched = {"available": True, "error": None,
                 "tasks": {"BettingBuddy Something New": task(NOW - timedelta(minutes=5))}}
        with mock.patch.object(jh, "check_backup", side_effect=RuntimeError("drive exploded")):
            report = jh.collect(now=NOW, sched=sched)
        by_id = {j["id"]: j for j in report["jobs"]}
        self.assertEqual(by_id["backup"]["status"], jh.UNKNOWN)
        self.assertIn("drive exploded", by_id["backup"]["summary"])
        self.assertEqual(by_id["unmonitored:BettingBuddy Something New"]["status"], jh.WARN)
        # The known tasks are missing from this fake scheduler: every required
        # one fails; the checker's own optional task is merely "off".
        for spec in jh.TASKS:
            self.assertEqual(by_id[spec.job_id]["status"], jh.OFF if spec.optional else jh.FAIL)
        self.assertIn("job_health", by_id)
        self.assertEqual(report["overall"], jh.FAIL)
        text = jh.summarize(report)
        self.assertTrue(text.startswith("Job health"))
        self.assertIn("does not exist in Task Scheduler", text)


def _report(status):
    check = {"id": "log", "name": "Log", "status": status, "summary": "Only 9 of 24 runs happened."}
    return {"overall": status, "generated_local": "2026-11-10 13:00", "host": "tini",
            "jobs": [{"id": "hourly", "name": "Injuries", "status": status, "summary": "", "checks": [check]}]}


class NotifyTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = os.path.join(self.tmp.name, "notified.json")
        self.sent = []

    def post(self, url, data, headers):
        self.sent.append((url, data, headers))
        return 200

    def test_does_nothing_unless_configured(self):
        r = jh.notify(_report(jh.FAIL), env={}, post=self.post, state_path=self.state, now=NOW)
        self.assertEqual(self.sent, [])
        self.assertEqual(r["configured"], [])
        self.assertFalse(os.path.exists(self.state))

    def test_ntfy_sends_once_then_repeats_only_after_12h_or_a_change(self):
        env = {"JOB_HEALTH_NTFY_TOPIC": "bb-test-topic"}
        jh.notify(_report(jh.FAIL), env=env, post=self.post, state_path=self.state, now=NOW)
        self.assertEqual(len(self.sent), 1)
        url, body, headers = self.sent[0]
        self.assertEqual(url, "https://ntfy.sh/bb-test-topic")
        self.assertIn(b"Only 9 of 24", body)
        self.assertEqual(headers["Priority"], "high")
        jh.notify(_report(jh.FAIL), env=env, post=self.post, state_path=self.state, now=NOW + timedelta(hours=1))
        self.assertEqual(len(self.sent), 1)                                    # same problem, an hour later
        jh.notify(_report(jh.WARN), env=env, post=self.post, state_path=self.state, now=NOW + timedelta(hours=2))
        self.assertEqual(len(self.sent), 2)                                    # changed
        jh.notify(_report(jh.WARN), env=env, post=self.post, state_path=self.state, now=NOW + timedelta(hours=15))
        self.assertEqual(len(self.sent), 3)                                    # still there after 12 h
        ok = {"overall": jh.OK, "generated_local": "x", "host": "tini", "jobs": []}
        jh.notify(ok, env=env, post=self.post, state_path=self.state, now=NOW + timedelta(hours=16))
        self.assertIn(b"healthy again", self.sent[-1][1])                      # one all-clear
        n = len(self.sent)
        jh.notify(ok, env=env, post=self.post, state_path=self.state, now=NOW + timedelta(hours=17))
        self.assertEqual(len(self.sent), n)                                    # and only one

    def test_email_needs_both_address_and_resend_key(self):
        jh.notify(_report(jh.FAIL), env={"JOB_HEALTH_EMAIL_TO": "a@b.c"}, post=self.post,
                  state_path=self.state, now=NOW)
        self.assertEqual(self.sent, [])
        jh.notify(_report(jh.FAIL), env={"JOB_HEALTH_EMAIL_TO": "a@b.c", "RESEND_API_KEY": "re_test"},
                  post=self.post, state_path=self.state, now=NOW)
        self.assertEqual(self.sent[0][0], "https://api.resend.com/emails")
        self.assertEqual(json.loads(self.sent[0][1])["to"], ["a@b.c"])

    def test_ping_url_reports_fail_suffix(self):
        env = {"JOB_HEALTH_PING_URL": "https://hc-ping.com/uuid"}
        jh.notify(_report(jh.FAIL), env=env, post=self.post, state_path=self.state, now=NOW)
        self.assertEqual(self.sent[0][0], "https://hc-ping.com/uuid/fail")
        ok = {"overall": jh.OK, "generated_local": "x", "host": "tini", "jobs": []}
        jh.notify(ok, env=env, post=self.post, state_path=self.state, now=NOW)
        self.assertEqual(self.sent[-1][0], "https://hc-ping.com/uuid")

    def test_a_channel_error_is_reported_not_raised(self):
        def boom(*a):
            raise OSError("offline")
        r = jh.notify(_report(jh.FAIL), env={"JOB_HEALTH_NTFY_TOPIC": "t"}, post=boom,
                      state_path=self.state, now=NOW)
        self.assertEqual(r["sent"], [])
        self.assertIn("offline", r["errors"][0])
        self.assertFalse(os.path.exists(self.state))    # unsent, so it retries next run


class EndpointTest(unittest.TestCase):
    """GET /api/admin/health: refuses without the internal key, and never
    runs without INTERNAL_API_KEY configured."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main_api
        cls.main_api = main_api
        cls.client = TestClient(main_api.app)

    def test_unconfigured_is_503(self):
        with mock.patch.object(self.main_api, "INTERNAL_API_KEY", ""):
            self.assertEqual(self.client.get("/api/admin/health").status_code, 503)

    def test_wrong_or_missing_key_is_401(self):
        with mock.patch.object(self.main_api, "INTERNAL_API_KEY", "k-test"):
            self.assertEqual(self.client.get("/api/admin/health").status_code, 401)
            self.assertEqual(self.client.get("/api/admin/health", headers={"X-Internal-Key": "nope"}).status_code, 401)

    def test_right_key_returns_the_report(self):
        fake = {"overall": "ok", "counts": {}, "jobs": [], "generated_local": "x", "host": "h",
                "scheduler_available": True, "scheduler_error": None, "notes": []}
        with mock.patch.object(self.main_api, "INTERNAL_API_KEY", "k-test"), \
                mock.patch.object(jh, "collect", return_value=dict(fake)), \
                mock.patch.object(jh, "summarize", return_value="Job health ..."):
            r = self.client.get("/api/admin/health?refresh=true", headers={"X-Internal-Key": "k-test"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["overall"], "ok")
        self.assertEqual(r.json()["summary_text"], "Job health ...")


if __name__ == "__main__":
    unittest.main()
