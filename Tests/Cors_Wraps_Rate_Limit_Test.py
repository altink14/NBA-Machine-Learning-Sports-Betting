"""A rate-limited answer still carries CORS headers (found 2026-09-23).

CORSMiddleware was added before SlowAPIMiddleware, which made the limiter the
outer layer: its 429 skipped CORS, and the browser showed "blocked by CORS
policy" instead of a rate limit. CORS must be the outermost middleware.
"""
import unittest

from starlette.middleware.cors import CORSMiddleware

import main_api


class CorsWrapsRateLimitTest(unittest.TestCase):

    def test_cors_is_the_outermost_middleware(self):
        # user_middleware lists the outermost layer first.
        self.assertIs(main_api.app.user_middleware[0].cls, CORSMiddleware)

    @unittest.skipUnless(main_api.SLOWAPI_AVAILABLE, "slowapi not installed")
    def test_a_429_carries_the_allow_origin_header(self):
        from fastapi import Request
        from fastapi.testclient import TestClient

        # A throwaway route with a tiny limit, so the test never loads a real one.
        @main_api.app.get("/__cors_probe")
        @main_api.limiter.limit("2/minute")
        def _probe(request: Request):
            return {"ok": True}

        origin = main_api.CORS_ORIGINS[0]
        main_api.limiter.reset()
        self.addCleanup(main_api.limiter.reset)
        client = TestClient(main_api.app)
        codes = [client.get("/__cors_probe", headers={"Origin": origin}) for _ in range(3)]
        self.assertEqual([r.status_code for r in codes], [200, 200, 429])
        self.assertEqual(codes[-1].headers.get("access-control-allow-origin"), origin)

if __name__ == "__main__":
    unittest.main()
