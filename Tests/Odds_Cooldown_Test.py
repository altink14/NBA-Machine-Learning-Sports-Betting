"""The recorder's cooldown counts from the last PAID poll (2026-09-28).

The snapshot writer stores a row only when a price changes, so the old
cooldown (MAX(captured_at) of odds_snapshots) never held on a quiet board.
_minutes_since_last_capture() now also reads odds_polls.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import snapshot_odds_api as rec  # noqa: E402


def iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


class CooldownTest(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE odds_snapshots (captured_at TEXT)")
        conn.execute("INSERT INTO odds_snapshots VALUES (?)", (iso(300),))  # last CHANGE 5h ago
        conn.commit()
        conn.close()
        self.patch = mock.patch.object(rec, "DB_PATH", self.db)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.remove(self.db)

    def add_polls(self, *polls):
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE IF NOT EXISTS odds_polls (polled_at TEXT, sport TEXT, source TEXT, status TEXT, events INTEGER)")
        conn.executemany("INSERT INTO odds_polls VALUES (?, ?, ?, ?, ?)", polls)
        conn.commit()
        conn.close()

    def test_without_poll_log_falls_back_to_snapshots(self):
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 300, delta=1)

    def test_recent_ok_poll_holds_the_cooldown(self):
        self.add_polls((iso(10), "NBA", rec.SOURCE_ODDS_API, "ok", 12))
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 10, delta=1)

    def test_empty_poll_spent_credits_too(self):
        self.add_polls((iso(20), "NBA", rec.SOURCE_ODDS_API, "empty", 0))
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 20, delta=1)

    def test_failed_fetch_with_no_answer_does_not_count(self):
        self.add_polls((iso(5), "NBA", rec.SOURCE_ODDS_API, "failed", None))
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 300, delta=1)

    def test_failed_archive_write_after_an_answer_counts(self):
        self.add_polls((iso(7), "NBA", rec.SOURCE_ODDS_API, "failed", 9))
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 7, delta=1)

    def test_sbr_and_other_sports_are_ignored(self):
        self.add_polls((iso(3), "NBA", "sbr", "ok", 5), (iso(4), "NFL", rec.SOURCE_ODDS_API, "ok", 5))
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 300, delta=1)

    def test_newer_price_change_wins_over_older_poll(self):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO odds_snapshots VALUES (?)", (iso(2),))
        conn.commit()
        conn.close()
        self.add_polls((iso(40), "NBA", rec.SOURCE_ODDS_API, "ok", 12))
        self.assertAlmostEqual(rec._minutes_since_last_capture(), 2, delta=1)

    def test_should_capture_respects_cooldown_on_a_quiet_board(self):
        self.add_polls((iso(10), "NBA", rec.SOURCE_ODDS_API, "ok", 12))
        with mock.patch.object(rec, "_minutes_to_nearest_tip", return_value=20.0):
            ok, why = rec.should_capture([object()])
        # In the closing window the ladder's cooldown is short; outside it,
        # a 10-minute-old poll must hold. Check against the ladder itself.
        window_cooldown = next(c for w, c in rec.CAPTURE_LADDER if 20.0 <= w)
        self.assertEqual(ok, 10 >= window_cooldown, why)


if __name__ == "__main__":
    unittest.main()
