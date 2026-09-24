"""Our own server-side renderer is not rate-limited; everyone else still is.

In production the Next.js server fetches from this API on behalf of every
visitor, from a few shared IPs, so per-IP limits would be shared by all of
them (a crawler would get 429s). A request carrying INTERNAL_API_KEY is not
counted. Every outbound nba.com request is paced by one lock regardless.
"""
import unittest
from unittest import mock

import main_api


@unittest.skipUnless(main_api.SLOWAPI_AVAILABLE, "slowapi not installed")
class InternalRateLimitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi import Request

        @main_api.app.get("/__internal_probe")
        @main_api.limiter.limit("2/minute")
        def _probe(request: Request):
            return {"ok": True}

    def setUp(self):
        from fastapi.testclient import TestClient
        main_api.limiter.reset()
        self.addCleanup(main_api.limiter.reset)
        self.client = TestClient(main_api.app)

    def codes(self, headers=None, n=3):
        return [self.client.get("/__internal_probe", headers=headers or {}).status_code for _ in range(n)]

    def test_visitors_are_still_limited(self):
        with mock.patch.object(main_api, "INTERNAL_API_KEY", "s3cret"):
            self.assertEqual(self.codes(), [200, 200, 429])

    def test_our_renderer_with_the_key_is_not_counted(self):
        with mock.patch.object(main_api, "INTERNAL_API_KEY", "s3cret"):
            self.assertEqual(self.codes({"X-Internal-Key": "s3cret"}, n=5), [200] * 5)

    def test_a_wrong_key_is_just_a_visitor(self):
        with mock.patch.object(main_api, "INTERNAL_API_KEY", "s3cret"):
            self.assertEqual(self.codes({"X-Internal-Key": "guess"}), [200, 200, 429])

    def test_no_key_configured_means_no_exemption(self):
        with mock.patch.object(main_api, "INTERNAL_API_KEY", ""):
            self.assertEqual(self.codes({"X-Internal-Key": ""}), [200, 200, 429])


class OutboundSlotTest(unittest.TestCase):

    def test_direct_calls_share_the_clients_gap(self):
        import time
        from src.Utils.nba_stats_client import NBAStatsClient
        c = NBAStatsClient(rate_delay=0.2)
        t0 = time.time()
        for _ in range(3):
            with c.outbound_slot("probe"):
                pass
        self.assertGreaterEqual(time.time() - t0, 0.38)   # two enforced gaps


if __name__ == "__main__":
    unittest.main()
