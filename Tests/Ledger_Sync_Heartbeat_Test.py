"""The odds heartbeat reaches the public server (owner's decision 2026-09-28).

ledger_sync now carries odds_polls (one row per poll, integer id) and
odds_seen (keyed by source, sport, game, book and market, no id) under the
same rules as everything else it mirrors: nothing the server holds may be
lost, a row may not change which game it is about, and the server's
fingerprint must equal the home copy's. odds_seen's last_seen_at moves on
every poll, so its rows update in place; that is allowed because it is an
observation table with no guard triggers, exactly like odds_snapshots.
"""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tests.Ledger_Sync_Test import make_db, add_pick  # noqa: E402
from src.Utils import ledger_sync  # noqa: E402
from src.Utils.odds_api_client import ensure_heartbeat_schema  # noqa: E402


def add_poll(conn, polled_at, status="ok"):
    cur = conn.execute(
        "INSERT INTO odds_polls (polled_at, sport, source, status, covers_board, books, markets, events, book_rows) "
        "VALUES (?, 'NBA', 'odds_api', ?, 1, NULL, 'ml,spread,total', 3, 9)", (polled_at, status))
    conn.commit()
    return cur.lastrowid


def see(conn, poll_id, polled_at, book="draftkings", game="Boston Celtics:New York Knicks", market="ml"):
    conn.execute(
        "INSERT INTO odds_seen (source, sport, game_key, sportsbook, market, first_seen_at, last_seen_at, last_poll_id) "
        "VALUES ('odds_api', 'NBA', ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (source, sport, game_key, sportsbook, market) "
        "DO UPDATE SET last_seen_at = excluded.last_seen_at, last_poll_id = excluded.last_poll_id",
        (game, book, market, polled_at, polled_at, poll_id))
    conn.commit()


class HeartbeatSyncTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="heartbeat_sync_test_")
        self.home = make_db(os.path.join(self.dir, "home.sqlite"))
        ensure_heartbeat_schema(self.home)
        self.server = make_db(os.path.join(self.dir, "server.sqlite"), with_snapshots=False)
        add_pick(self.home)

    def tearDown(self):
        self.home.close()
        self.server.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def upload(self):
        dest = os.path.join(self.dir, f"upload_{len(os.listdir(self.dir))}.sqlite")
        self.home.execute("VACUUM INTO ?", (dest,))
        return dest

    def home_fp(self, path):
        c = sqlite3.connect(path)
        try:
            return ledger_sync.fingerprint(c)
        finally:
            c.close()

    def test_first_sync_carries_both_tables_and_fingerprints_match(self):
        pid = add_poll(self.home, "2026-10-19T12:00:00+00:00")
        see(self.home, pid, "2026-10-19T12:00:00+00:00")
        see(self.home, pid, "2026-10-19T12:00:00+00:00", book="fanduel")
        up = self.upload()
        out = ledger_sync.merge(self.server, up)
        self.assertEqual(out["tables"]["odds_polls"]["inserted"], 1)
        self.assertEqual(out["tables"]["odds_seen"]["inserted"], 2)
        self.assertEqual(out["fingerprint"], self.home_fp(up))
        self.assertIsNone(out["fingerprint"]["odds_seen"]["max_id"])

    def test_a_later_sighting_updates_last_seen_in_place(self):
        pid = add_poll(self.home, "2026-10-19T12:00:00+00:00")
        see(self.home, pid, "2026-10-19T12:00:00+00:00")
        ledger_sync.merge(self.server, self.upload())
        pid2 = add_poll(self.home, "2026-10-19T18:00:00+00:00")
        see(self.home, pid2, "2026-10-19T18:00:00+00:00")
        up = self.upload()
        out = ledger_sync.merge(self.server, up)
        self.assertEqual(out["tables"]["odds_polls"]["inserted"], 1)
        self.assertEqual(out["tables"]["odds_seen"]["inserted"], 0)
        self.assertEqual(out["tables"]["odds_seen"]["updated"], 1)
        row = self.server.execute("SELECT first_seen_at, last_seen_at, last_poll_id FROM odds_seen").fetchone()
        self.assertEqual(row, ("2026-10-19T12:00:00+00:00", "2026-10-19T18:00:00+00:00", pid2))
        self.assertEqual(out["fingerprint"], self.home_fp(up))

    def test_losing_a_sighting_is_refused_and_writes_nothing(self):
        pid = add_poll(self.home, "2026-10-19T12:00:00+00:00")
        see(self.home, pid, "2026-10-19T12:00:00+00:00")
        see(self.home, pid, "2026-10-19T12:00:00+00:00", book="fanduel")
        ledger_sync.merge(self.server, self.upload())
        self.home.execute("DELETE FROM odds_seen WHERE sportsbook = 'fanduel'")
        self.home.commit()
        before = ledger_sync.fingerprint(self.server)
        with self.assertRaises(ledger_sync.SyncRefused):
            ledger_sync.merge(self.server, self.upload())
        self.assertEqual(ledger_sync.fingerprint(self.server), before)

    def test_losing_a_poll_is_refused(self):
        add_poll(self.home, "2026-10-19T12:00:00+00:00")
        ledger_sync.merge(self.server, self.upload())
        self.home.execute("DELETE FROM odds_polls")
        self.home.commit()
        with self.assertRaises(ledger_sync.SyncRefused):
            ledger_sync.merge(self.server, self.upload())

    def test_a_poll_id_that_changes_its_time_is_refused(self):
        add_poll(self.home, "2026-10-19T12:00:00+00:00")
        ledger_sync.merge(self.server, self.upload())
        self.home.execute("UPDATE odds_polls SET polled_at = '2026-10-19T13:00:00+00:00'")
        self.home.commit()
        with self.assertRaises(ledger_sync.SyncRefused):
            ledger_sync.merge(self.server, self.upload())

    def test_a_home_copy_without_the_heartbeat_still_syncs(self):
        # A home PC that has not run a heartbeat poll yet has no such tables.
        home = make_db(os.path.join(self.dir, "old_home.sqlite"))
        add_pick(home)
        dest = os.path.join(self.dir, "old_upload.sqlite")
        home.execute("VACUUM INTO ?", (dest,))
        home.close()
        out = ledger_sync.merge(self.server, dest)
        self.assertNotIn("odds_seen", out["tables"])
        self.assertEqual(out["tables"]["predictions_log"]["inserted"], 1)


if __name__ == "__main__":
    unittest.main()
