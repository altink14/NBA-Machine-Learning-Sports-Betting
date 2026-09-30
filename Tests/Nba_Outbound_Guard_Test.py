"""The stats.nba.com ceiling (2026-09-28): daily and hourly budgets and a
circuit breaker, enforced at nba_api's HTTP send so no caller can skip it.

Written after a 12,000-request backfill got the home PC's address blocked by
nba.com's edge. Every test runs against a temporary state file and a stubbed
network: nothing here reaches stats.nba.com.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Utils import nba_stats_client as client  # noqa: E402
from src.Utils import nba_outbound_guard as guard  # noqa: E402

T0 = datetime(2026, 10, 21, 13, 0, tzinfo=timezone.utc)


class GuardTest(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        os.remove(self.db)
        self.env = mock.patch.dict(os.environ, {
            "NBA_OUTBOUND_DB": self.db, "NBA_DAILY_REQUEST_BUDGET": "5",
            "NBA_HOURLY_REQUEST_BUDGET": "3", "NBA_BREAKER_FAILURES": "2", "NBA_BREAKER_HOURS": "72",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        try:
            os.remove(self.db)
        except OSError:
            pass

    def test_hourly_then_daily_budget(self):
        for _ in range(3):
            guard.before_request("x", now=T0)
        with self.assertRaises(client.OutboundRefused):
            guard.before_request("x", now=T0)
        later = T0 + timedelta(hours=1)
        guard.before_request("x", now=later)
        guard.before_request("x", now=later)
        with self.assertRaisesRegex(client.OutboundRefused, "today's"):
            guard.before_request("x", now=later)
        self.assertEqual(guard.status(now=later)["used_today"], 5)

    def test_refusal_is_a_live_fetch_disabled(self):
        self.assertTrue(issubclass(client.OutboundRefused, client.LiveFetchDisabled))

    def test_breaker_opens_after_failure_streak_and_blocks(self):
        guard.record_result(False, "timeout", now=T0)
        guard.record_result(False, "timeout", now=T0)
        st = guard.status(now=T0)
        self.assertEqual(st["breaker"], "open")
        with self.assertRaisesRegex(client.OutboundRefused, "paused until"):
            guard.before_request("x", now=T0 + timedelta(hours=71))

    def test_success_resets_the_streak(self):
        guard.record_result(False, "timeout", now=T0)
        guard.record_result(True, now=T0)
        guard.record_result(False, "timeout", now=T0)
        self.assertEqual(guard.status(now=T0)["breaker"], "closed")

    def test_one_probe_after_cooldown_and_doubling_on_failure(self):
        guard.record_result(False, "403", now=T0)
        guard.record_result(False, "403", now=T0)
        after = T0 + timedelta(hours=73)
        guard.before_request("probe", now=after)          # the probe is let through
        with self.assertRaisesRegex(client.OutboundRefused, "probe"):
            guard.before_request("second", now=after)     # nothing else meanwhile
        guard.record_result(False, "403", now=after)      # probe failed
        st = guard.status(now=after)
        self.assertEqual(st["breaker"], "open")
        self.assertEqual(st["cooldown_hours"], 144)
        again = after + timedelta(hours=145)
        guard.before_request("probe2", now=again)
        guard.record_result(True, now=again)              # recovered
        self.assertEqual(guard.status(now=again)["breaker"], "closed")

    def test_cooldown_is_capped_at_a_week(self):
        with mock.patch.dict(os.environ, {"NBA_BREAKER_HOURS": "100"}):
            guard.record_result(False, "x", now=T0)
            guard.record_result(False, "x", now=T0)
            t = T0 + timedelta(hours=101)
            guard.before_request("p", now=t)
            guard.record_result(False, "x", now=t)
            self.assertEqual(guard.status(now=t)["cooldown_hours"], guard.MAX_COOLDOWN_HOURS)

    def test_http_send_is_guarded_and_non_200_counts_as_failure(self):
        from nba_api.stats.library.http import NBAStatsHTTP

        class Resp:
            _status_code = 403

        sent = []
        original = getattr(NBAStatsHTTP.send_api_request, "__wrapped_original__", None)
        with mock.patch.object(client, "live_fetch_enabled", return_value=True):
            # Rebuild the guard around a fake network call.
            real = NBAStatsHTTP.send_api_request
            try:
                def fake(self, endpoint, *a, **k):
                    sent.append(endpoint)
                    return Resp()
                NBAStatsHTTP.send_api_request = fake
                client.install_live_guard()
                http = NBAStatsHTTP()
                http.send_api_request("leaguegamelog", {})
                http.send_api_request("leaguegamelog", {})
                self.assertEqual(guard.status()["breaker"], "open")
                with self.assertRaises(client.OutboundRefused):
                    http.send_api_request("leaguegamelog", {})
                self.assertEqual(len(sent), 2)            # the refused one never went out
            finally:
                NBAStatsHTTP.send_api_request = real
        del original

    def test_fetch_does_not_retry_a_refusal(self):
        c = client.NBAStatsClient(rate_delay=0)

        class Endpoint:
            def __init__(self, **kw):
                raise client.OutboundRefused("budget spent")

        with mock.patch.object(client, "_read_cache", return_value=None), \
                mock.patch.object(client.time, "sleep") as slept:
            with self.assertRaises(client.OutboundRefused):
                c._fetch("x", Endpoint, {"a": 1})
        slept.assert_not_called()

    def test_a_refusal_falls_back_to_the_mirror(self):
        """Breaker open on the home PC: the mirrored copy answers, as it does
        with live fetching off (the 2026-27 schedule 503'd without this)."""
        c = client.NBAStatsClient(rate_delay=0)

        class Endpoint:
            def __init__(self, **kw):
                raise client.OutboundRefused("paused")

        with mock.patch.object(client, "_read_cache", return_value=None), \
                mock.patch.object(client, "live_fetch_enabled", return_value=True), \
                mock.patch.object(client, "read_mirror", return_value={"leagueSchedule": {"x": 1}}) as mirror:
            self.assertEqual(c._fetch("scheduleleaguev2", Endpoint, {"season": "2026-27"}),
                             {"leagueSchedule": {"x": 1}})
        mirror.assert_called_once()
        with mock.patch.object(client, "_read_cache", return_value=None), \
                mock.patch.object(client, "live_fetch_enabled", return_value=True), \
                mock.patch.object(client, "read_mirror", return_value=None):
            with self.assertRaises(client.OutboundRefused):     # no copy: still refused, still no retry
                c._fetch("scheduleleaguev2", Endpoint, {"season": "2026-27"})


if __name__ == "__main__":
    unittest.main()
