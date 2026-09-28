"""The Odds API key never leaves the NBA odds client in an error (2026-09-28).

requests quotes the full URL, key included, in its exception messages. That
text reached the logs, odds_polls.error (which ledger_sync carries to the
public server) and the job-health report. Transport errors, non-2xx replies
and stored poll errors are now redacted, and the health report scrubs any
key=... it quotes.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Utils import odds_api_client as oac  # noqa: E402

KEY = "0123456789abcdef0123456789abcdef"


class KeyRedactionTest(unittest.TestCase):
    def test_redact(self):
        text = f"Max retries exceeded with url: /v4/sports/basketball_nba/odds?apiKey={KEY}&regions=us"
        self.assertNotIn(KEY, oac.redact(text))
        self.assertIn("apiKey=REDACTED&regions=us", oac.redact(text))

    def test_transport_error_carries_no_key(self):
        err = requests.ConnectionError(f"HTTPSConnectionPool: url /odds?apiKey={KEY}&x=1")
        with mock.patch.object(oac.requests, "get", side_effect=err):
            with self.assertRaises(oac.OddsApiError) as ctx:
                oac.fetch_nba_odds(api_key=KEY)
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_events_http_error_carries_no_key(self):
        resp = mock.MagicMock(status_code=503, ok=False, text=f"bad gateway for apiKey={KEY}", headers={})
        with mock.patch.object(oac.requests, "get", return_value=resp):
            with self.assertRaises(oac.OddsApiError) as ctx:
                oac.fetch_nba_events(api_key=KEY)
        self.assertNotIn(KEY, str(ctx.exception))

    def test_stored_poll_error_is_redacted(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            oac.ensure_heartbeat_schema(conn)
            oac.record_poll(conn, polled_at="2026-09-28T09:03:00+00:00", sport="NBA", source=oac.SOURCE_ODDS_API,
                            status="failed", covers_board=True, markets="ml,spread,total",
                            error=f"url /odds?apiKey={KEY}")
            stored = conn.execute("SELECT error FROM odds_polls").fetchone()[0]
            conn.close()
            self.assertNotIn(KEY, stored)
            self.assertIn("apiKey=REDACTED", stored)
        finally:
            os.remove(path)

    def test_health_report_scrubs_keys(self):
        import job_health
        report = job_health._scrub({"a": [f"x?apiKey={KEY}&y", {"b": f"token={KEY}"}], "n": 3})
        self.assertNotIn(KEY, str(report))
        self.assertEqual(report["n"], 3)


if __name__ == "__main__":
    unittest.main()
