"""The pick commitments: a hash per day, chained, revealed only after tip-off.

Every test builds throwaway databases in a temp directory; nothing touches
Data/OddsData.sqlite. See src/Utils/ledger_commit.py for the method.
"""

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import commit_ledger
import main_api
import publish_commitments
from src.Sports import run_scheduled
from src.Utils import ledger_commit as lc
from src.Utils import ledger_sync
from Tests.Ledger_Sync_Test import make_db

NONCE_A = "a" * 64


def pick(conn, log_date, home, away, tip, logged="2026-10-21T13:00:00+00:00",
         winner=None, conf=64.25, home_ml=-150, away_ml=130, book="fanduel"):
    cur = conn.execute(
        "INSERT INTO predictions_log (logged_at, log_date, sport, sportsbook, game_key, home_team, "
        "away_team, game_start_time_utc, home_ml, away_ml, ou_line, predicted_winner, "
        "winner_confidence, ou_prediction, ou_confidence, ev_home, ev_away, model) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (logged, log_date, "NBA", book, f"{home}:{away}", home, away, tip, home_ml, away_ml,
         221.5, winner or home, conf, "OVER", 58.0, 0.04, -0.07, "candidate_2026-08"))
    conn.commit()
    return cur.lastrowid


def day1(conn):
    pick(conn, "2026-10-21", "Boston Celtics", "New York Knicks", "2026-10-21T23:30:00+00:00")
    pick(conn, "2026-10-21", "Los Angeles Lakers", "Golden State Warriors",
         "2026-10-22T02:00:00+00:00", winner="Golden State Warriors", conf=55.1, home_ml=110, away_ml=-130)


def nonces(*values):
    it = iter(values)
    return lambda: next(it)


class CanonicalFormTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_commit_test_")
        self.conn = make_db(os.path.join(self.dir, "home.sqlite"))

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_canonical_row_and_preimage_are_exactly_as_documented(self):
        day1(self.conn)
        self.conn.row_factory = sqlite3.Row
        row = self.conn.execute("SELECT * FROM predictions_log WHERE id = 1").fetchone()
        self.conn.row_factory = None
        canon = lc.canonical_row(row)
        self.assertEqual(canon, [1, "2026-10-21", "NBA", "fanduel", "Boston Celtics:New York Knicks",
                                 "2026-10-21T23:30:00+00:00", "2026-10-21T13:00:00+00:00",
                                 "candidate_2026-08", "Boston Celtics", "64.250000",
                                 "-150.000000", "130.000000"])
        self.assertEqual(
            lc.preimage("2026-10-21", 1, NONCE_A, [canon]),
            '["bb-ledger-commit-v1","2026-10-21",1,"' + NONCE_A + '",[[1,"2026-10-21","NBA",'
            '"fanduel","Boston Celtics:New York Knicks","2026-10-21T23:30:00+00:00",'
            '"2026-10-21T13:00:00+00:00","candidate_2026-08","Boston Celtics","64.250000",'
            '"-150.000000","130.000000"]]]')

    def test_the_hash_of_a_fixed_day_never_changes(self):
        # Pinned. If this fails the canonical form changed, and every
        # commitment ever made under v1 would stop verifying: give the new
        # form a new METHOD_VERSION instead.
        day1(self.conn)
        out = lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00", nonce_source=nonces(NONCE_A))
        self.assertEqual(len(out), 1)
        c = self.conn.execute("SELECT picks_sha256, chain_sha256, prev_chain_sha256 FROM ledger_commitments").fetchone()
        self.assertEqual(c[2], lc.GENESIS)
        self.assertEqual(c[0], PINNED_PICKS)
        self.assertEqual(c[1], PINNED_CHAIN)

    def test_numbers_do_not_depend_on_how_they_were_stored(self):
        self.assertEqual(lc.canonical_value("home_ml", -150), "-150.000000")
        self.assertEqual(lc.canonical_value("home_ml", -150.0), "-150.000000")
        self.assertEqual(lc.canonical_value("winner_confidence", 67.23), "67.230000")
        self.assertIsNone(lc.canonical_value("away_ml", None))

    def test_the_withdrawn_over_under_pick_and_grading_are_not_in_the_hash(self):
        self.assertNotIn("ou_prediction", lc.CANONICAL_FIELDS)
        self.assertNotIn("ou_confidence", lc.CANONICAL_FIELDS)
        self.assertNotIn("actual_winner", lc.CANONICAL_FIELDS)
        day1(self.conn)
        lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00", nonce_source=nonces(NONCE_A))
        # Grading afterwards must leave the commitment verifying.
        self.conn.execute("UPDATE predictions_log SET actual_winner = home_team, actual_total = 220")
        self.conn.commit()
        self.assertTrue(lc.verify(self.conn)["intact"])

    def test_the_simulated_tag_matches_the_api(self):
        self.assertEqual(lc.SIMULATED_MODEL_TAG, main_api.SIMULATED_MODEL_TAG)


class CommitAndChainTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_commit_test_")
        self.path = os.path.join(self.dir, "home.sqlite")
        self.conn = make_db(self.path)

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def three_days(self):
        day1(self.conn)
        lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00")
        pick(self.conn, "2026-10-22", "Denver Nuggets", "Utah Jazz", "2026-10-23T01:00:00+00:00",
             logged="2026-10-22T13:00:00+00:00")
        lc.commit_pending(self.conn, now="2026-10-22T13:05:00+00:00")
        pick(self.conn, "2026-10-23", "Miami Heat", "Orlando Magic", "2026-10-23T23:00:00+00:00",
             logged="2026-10-23T13:00:00+00:00")
        lc.commit_pending(self.conn, now="2026-10-23T13:05:00+00:00")

    def test_no_picks_means_no_commitment(self):
        self.assertEqual(lc.commit_pending(self.conn), [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM ledger_commitments").fetchone()[0], 0)
        day1(self.conn)
        self.assertEqual(len(lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00")), 1)
        # Run again with nothing new: nothing written.
        self.assertEqual(lc.commit_pending(self.conn, now="2026-10-21T14:05:00+00:00"), [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM ledger_commitments").fetchone()[0], 1)

    def test_a_pick_logged_later_that_day_gets_its_own_commitment(self):
        day1(self.conn)
        lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00")
        pick(self.conn, "2026-10-21", "Phoenix Suns", "Sacramento Kings", "2026-10-22T02:30:00+00:00",
             logged="2026-10-21T15:00:00+00:00")
        out = lc.commit_pending(self.conn, now="2026-10-21T15:05:00+00:00")
        self.assertEqual([(o["log_date"], o["seq"], o["n_picks"]) for o in out], [("2026-10-21", 2, 1)])
        self.assertTrue(lc.verify(self.conn)["intact"])

    def test_the_chain_links_every_day_to_the_one_before(self):
        self.three_days()
        rows = self.conn.execute(
            "SELECT prev_chain_sha256, chain_sha256 FROM ledger_commitments ORDER BY id").fetchall()
        self.assertEqual(rows[0][0], lc.GENESIS)
        self.assertEqual(rows[1][0], rows[0][1])
        self.assertEqual(rows[2][0], rows[1][1])
        v = lc.verify(self.conn)
        self.assertTrue(v["intact"])
        self.assertEqual(v["head"], rows[2][1])

    def test_changing_a_past_pick_breaks_the_proof(self):
        self.three_days()
        # The triggers stop this; someone with the file can drop them.
        self.conn.execute("DROP TRIGGER predictions_log_immutable")
        self.conn.execute("UPDATE predictions_log SET predicted_winner = 'New York Knicks' WHERE id = 1")
        self.conn.commit()
        v = lc.verify(self.conn)
        self.assertFalse(v["intact"])
        self.assertEqual([e["rows"] for e in v["entries"]], ["changed", "match", "match"])

    def test_changing_the_odds_or_confidence_breaks_it_too(self):
        self.three_days()
        self.conn.execute("DROP TRIGGER predictions_log_immutable")
        self.conn.execute("UPDATE predictions_log SET home_ml = -155 WHERE id = 3")  # no trigger guards odds
        self.conn.commit()
        self.assertEqual([e["rows"] for e in lc.verify(self.conn)["entries"]], ["match", "changed", "match"])

    def test_rewriting_a_commitment_to_match_breaks_every_later_link(self):
        self.three_days()
        self.conn.execute("DROP TRIGGER predictions_log_immutable")
        self.conn.execute("DROP TRIGGER ledger_commitments_no_update")
        self.conn.execute("UPDATE predictions_log SET predicted_winner = 'New York Knicks' WHERE id = 1")
        # A forger recomputes day 1's hashes so that day checks out on its own...
        c = self.conn.execute("SELECT * FROM ledger_commitments WHERE id = 1").fetchone()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM ledger_commitments").description]
        c = dict(zip(cols, c))
        self.conn.row_factory = sqlite3.Row
        rows = [lc.canonical_row(r) for r in self.conn.execute("SELECT * FROM predictions_log WHERE id IN (1, 2) ORDER BY id")]
        self.conn.row_factory = None
        p = lc.picks_hash(c["log_date"], c["seq"], c["nonce"], rows)
        ch = lc.chain_hash(lc.GENESIS, c["log_date"], c["seq"], c["n_picks"], c["committed_at"], p)
        self.conn.execute("UPDATE ledger_commitments SET picks_sha256 = ?, chain_sha256 = ? WHERE id = 1", (p, ch))
        self.conn.commit()
        v = lc.verify(self.conn)
        # ...and day 2's link, which covers the old day-1 hash, no longer follows.
        self.assertEqual([e["chain"] for e in v["entries"]], ["ok", "broken", "ok"])
        self.assertFalse(v["intact"])

    def test_the_table_refuses_edits_and_deletes(self):
        day1(self.conn)
        lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00")
        with self.assertRaises(sqlite3.DatabaseError):
            self.conn.execute("UPDATE ledger_commitments SET n_picks = 1")
        with self.assertRaises(sqlite3.DatabaseError):
            self.conn.execute("DELETE FROM ledger_commitments")

    def test_committed_before_first_tipoff_is_recorded_honestly(self):
        day1(self.conn)
        out = lc.commit_pending(self.conn, now="2026-10-22T00:00:00+00:00")  # after the 23:30 tip
        self.assertFalse(out[0]["before_first_tipoff"])
        view = lc.public_view(self.conn, now="2026-10-22T00:00:00+00:00")
        self.assertFalse(view["commitments"][0]["committed_before_first_tipoff"])

    def test_the_cli_refuses_on_the_public_server(self):
        day1(self.conn)
        with mock.patch.dict(os.environ, {"PREDICTIONS_SOURCE": "ledger"}), mock.patch("builtins.print"):
            self.assertEqual(commit_ledger.run(self.path), 1)
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'ledger_commitments'").fetchone()[0], 0)
        with mock.patch.dict(os.environ, {"PREDICTIONS_SOURCE": "live"}), mock.patch("builtins.print"):
            self.assertEqual(commit_ledger.run(self.path), 0)
            self.assertEqual(commit_ledger.run(self.path, verify_only=True), 0)


class SealTest(unittest.TestCase):
    """Nothing about a pick is revealed before every game it covers has started."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_commit_seal_")
        self.path = os.path.join(self.dir, "OddsData.sqlite")
        self.conn = make_db(self.path)
        day1(self.conn)  # tips 23:30Z and 02:00Z next day
        lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00", nonce_source=nonces(NONCE_A))

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_before_any_tip_only_the_hashes_show(self):
        c = lc.public_view(self.conn, now="2026-10-21T20:00:00+00:00")["commitments"][0]
        self.assertFalse(c["revealed"])
        self.assertIsNone(c["nonce"])
        self.assertIsNone(c["picks"])
        self.assertEqual(c["picks_sha256"], PINNED_PICKS)

    def test_one_game_started_is_not_enough(self):
        c = lc.public_view(self.conn, now="2026-10-22T00:00:00+00:00")["commitments"][0]
        self.assertFalse(c["revealed"])
        self.assertIsNone(c["nonce"])
        self.assertIsNone(c["picks"])

    def test_after_the_last_tip_anyone_can_recompute_it(self):
        c = lc.public_view(self.conn, now="2026-10-22T02:00:00+00:00")["commitments"][0]
        self.assertTrue(c["revealed"])
        self.assertEqual(c["nonce"], NONCE_A)
        # What the browser does: rebuild the preimage from the served values.
        text = json.dumps(["bb-ledger-commit-v1", c["log_date"], c["seq"], c["nonce"], c["picks"]],
                          ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), c["picks_sha256"])
        link = json.dumps(["bb-ledger-commit-v1", "chain", c["prev_chain_sha256"], c["log_date"], c["seq"],
                           c["n_picks"], c["committed_at"], c["picks_sha256"]], separators=(",", ":"))
        self.assertEqual(hashlib.sha256(link.encode("utf-8")).hexdigest(), c["chain_sha256"])

    def test_the_endpoint_serves_no_sealed_pick(self):
        client = TestClient(main_api.app)
        with mock.patch.object(main_api, "ODDS_DB_PATH", self.path), \
                mock.patch.object(main_api, "_utc_iso", return_value="2026-10-22T00:00:00+00:00"), \
                mock.patch.object(main_api, "API_KEY", ""):
            r = client.get("/api/ledger/commitments?days=1000")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(len(body["commitments"]), 1)
        self.assertFalse(body["commitments"][0]["revealed"])
        self.assertNotIn(NONCE_A, r.text)
        self.assertNotIn("Golden State Warriors", r.text)  # a pick, and the game it names
        self.assertNotIn("64.25", r.text)
        self.assertTrue(body["chain"]["intact"])
        self.assertEqual(body["method"]["version"], lc.METHOD_VERSION)

    def test_the_endpoint_reveals_after_the_last_tip(self):
        client = TestClient(main_api.app)
        with mock.patch.object(main_api, "ODDS_DB_PATH", self.path), \
                mock.patch.object(main_api, "_utc_iso", return_value="2026-10-22T03:00:00+00:00"), \
                mock.patch.object(main_api, "API_KEY", ""):
            c = client.get("/api/ledger/commitments?days=1000").json()["commitments"][0]
        self.assertTrue(c["revealed"])
        self.assertEqual(len(c["picks"]), 2)
        self.assertEqual(c["server_check"], {"chain": "ok", "rows": "match"})

    def test_the_endpoint_is_empty_and_honest_before_any_commitment(self):
        empty = os.path.join(self.dir, "empty.sqlite")
        make_db(empty).close()
        client = TestClient(main_api.app)
        with mock.patch.object(main_api, "ODDS_DB_PATH", empty), mock.patch.object(main_api, "API_KEY", ""):
            body = client.get("/api/ledger/commitments").json()
        self.assertEqual(body["commitments"], [])
        self.assertEqual(body["chain"]["length"], 0)
        self.assertIsNone(body["public_copy_url"])


class SyncCarriesCommitmentsTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_commit_sync_")
        self.home = make_db(os.path.join(self.dir, "home.sqlite"))
        self.server = make_db(os.path.join(self.dir, "server.sqlite"), with_snapshots=False)
        lc.ensure_schema(self.server)  # what the sync endpoint does first
        day1(self.home)
        lc.commit_pending(self.home, now="2026-10-21T13:05:00+00:00")

    def tearDown(self):
        self.home.close()
        self.server.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def upload(self, conn=None):
        dest = os.path.join(self.dir, f"upload_{len(os.listdir(self.dir))}.sqlite")
        (conn or self.home).execute("VACUUM INTO ?", (dest,))
        return dest

    def test_the_table_reaches_the_server_and_fingerprints_match(self):
        up = self.upload()
        out = ledger_sync.merge(self.server, up)
        self.assertEqual(out["tables"]["ledger_commitments"]["inserted"], 1)
        c = sqlite3.connect(up)
        self.assertEqual(out["fingerprint"], ledger_sync.fingerprint(c))
        c.close()
        self.assertTrue(lc.verify(self.server)["intact"])
        # Idempotent, and a later day arrives as one more row.
        pick(self.home, "2026-10-22", "Denver Nuggets", "Utah Jazz", "2026-10-23T01:00:00+00:00",
             logged="2026-10-22T13:00:00+00:00")
        lc.commit_pending(self.home, now="2026-10-22T13:05:00+00:00")
        out = ledger_sync.merge(self.server, self.upload())
        self.assertEqual(out["tables"]["ledger_commitments"], {"inserted": 1, "updated": 0,
                                                               "columns_added": [], "rows": 2})

    def test_a_changed_commitment_is_refused_and_nothing_written(self):
        ledger_sync.merge(self.server, self.upload())
        before = ledger_sync.fingerprint(self.server)
        dest = self.upload()
        c = sqlite3.connect(dest)
        c.execute("DROP TRIGGER ledger_commitments_no_update")
        c.execute("UPDATE ledger_commitments SET committed_at = '2026-10-21T09:00:00+00:00'")
        c.commit()
        c.close()
        with self.assertRaises(ledger_sync.SyncRefused):
            ledger_sync.merge(self.server, dest)
        self.assertEqual(ledger_sync.fingerprint(self.server), before)

    def test_a_deleted_commitment_is_refused(self):
        ledger_sync.merge(self.server, self.upload())
        dest = self.upload()
        c = sqlite3.connect(dest)
        c.execute("DROP TRIGGER ledger_commitments_no_delete")
        c.execute("DELETE FROM ledger_commitments")
        c.commit()
        c.close()
        with self.assertRaises(ledger_sync.SyncRefused):
            ledger_sync.merge(self.server, dest)

    def test_a_first_upload_with_a_broken_chain_is_refused(self):
        dest = self.upload()
        c = sqlite3.connect(dest)
        c.execute("DROP TRIGGER ledger_commitments_no_update")
        c.execute("UPDATE ledger_commitments SET picks_sha256 = ?", ("0" * 64,))
        c.commit()
        c.close()
        with self.assertRaises(ledger_sync.SyncRefused) as ctx:
            ledger_sync.merge(self.server, dest)
        self.assertIn("do not verify", str(ctx.exception))
        self.assertEqual(self.server.execute("SELECT COUNT(*) FROM ledger_commitments").fetchone()[0], 0)
        self.assertEqual(self.server.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0], 0)

    def test_a_server_without_the_guarded_table_is_refused(self):
        bare = make_db(os.path.join(self.dir, "bare.sqlite"), with_snapshots=False)
        try:
            with self.assertRaises(ledger_sync.SyncRefused):
                ledger_sync.merge(bare, self.upload())
        finally:
            bare.close()

    def test_a_home_copy_without_commitments_still_syncs(self):
        home = make_db(os.path.join(self.dir, "old_home.sqlite"))
        day1(home)
        out = ledger_sync.merge(self.server, self.upload(home))
        home.close()
        self.assertNotIn("ledger_commitments", out["tables"])

    def test_the_sync_endpoint_creates_the_guarded_table_first(self):
        server_path = os.path.join(self.dir, "OddsData.sqlite")
        make_db(server_path, with_snapshots=False).close()  # an older server: no commitments table
        dest = self.upload()
        with open(dest, "rb") as fh:
            body = gzip.compress(fh.read())
        with mock.patch.object(main_api, "ODDS_DB_PATH", server_path), \
                mock.patch.object(main_api, "LEDGER_SYNC_SECRET", "s3cret-for-tests"), \
                mock.patch.object(main_api, "API_KEY", ""):
            r = TestClient(main_api.app).post("/api/admin/ledger/sync", content=body,
                                              headers={"X-Ledger-Sync-Secret": "s3cret-for-tests"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["tables"]["ledger_commitments"]["inserted"], 1)
        c = sqlite3.connect(server_path)
        try:
            triggers = {x[0] for x in c.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='ledger_commitments'")}
        finally:
            c.close()
        self.assertEqual(triggers, set(lc.GUARD_TRIGGERS))


class FakeGit:
    """Stands in for git: HEAD holds the last committed file; push is recorded."""

    def __init__(self, head=""):
        self.head = head
        self.pushed = 0
        self.calls = []

    def __call__(self, repo, *args):
        self.calls.append(args[0])
        ok = subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[0] == "show":
            return subprocess.CompletedProcess(args, 0 if self.head else 128, stdout=self.head, stderr="")
        if args[0] == "commit":
            with open(os.path.join(repo, *publish_commitments.COMMITMENTS_FILE.split("/")), encoding="utf-8") as fh:
                self.head = fh.read()
        if args[0] == "push":
            self.pushed += 1
        return ok


class PublishTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_commit_publish_")
        self.conn = make_db(os.path.join(self.dir, "home.sqlite"))
        day1(self.conn)
        lc.commit_pending(self.conn, now="2026-10-21T13:05:00+00:00", nonce_source=nonces(NONCE_A))
        self.repo = os.path.join(self.dir, "public")
        os.makedirs(os.path.join(self.repo, ".git"))

    def tearDown(self):
        self.conn.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_published_lines_carry_no_nonce_and_no_pick(self):
        lines = publish_commitments.public_lines(self.conn)
        self.assertEqual(len(lines), 1)
        self.assertNotIn(NONCE_A, lines[0])
        self.assertNotIn("Boston Celtics", lines[0])
        self.assertEqual(json.loads(lines[0])["chain_sha256"], PINNED_CHAIN)

    def test_without_the_flag_it_only_prints(self):
        git = FakeGit()
        with mock.patch("builtins.print") as out:
            self.assertEqual(publish_commitments.publish(self.conn, self.repo, False, git=git), 0)
        self.assertEqual(git.calls, [])
        self.assertTrue(out.call_args_list[0][0][0].startswith("DRY RUN"))

    def test_without_a_target_it_skips(self):
        git = FakeGit()
        with mock.patch("builtins.print") as out:
            self.assertEqual(publish_commitments.publish(self.conn, None, True, git=git), 0)
        self.assertEqual(git.calls, [])
        self.assertTrue(out.call_args[0][0].startswith("SKIPPED"))

    def test_publish_appends_and_is_idempotent(self):
        git = FakeGit()
        with mock.patch("builtins.print"):
            self.assertEqual(publish_commitments.publish(self.conn, self.repo, True, git=git), 0)
            self.assertEqual(git.head.count("\n"), 1)
            pick(self.conn, "2026-10-22", "Denver Nuggets", "Utah Jazz", "2026-10-23T01:00:00+00:00",
                 logged="2026-10-22T13:00:00+00:00")
            lc.commit_pending(self.conn, now="2026-10-22T13:05:00+00:00")
            self.assertEqual(publish_commitments.publish(self.conn, self.repo, True, git=git), 0)
            self.assertEqual(git.head.count("\n"), 2)
            self.assertEqual(publish_commitments.publish(self.conn, self.repo, True, git=git), 0)
        self.assertEqual(git.head.count("\n"), 2)
        self.assertEqual(git.head.splitlines(), publish_commitments.public_lines(self.conn))

    def test_a_published_file_that_disagrees_is_refused(self):
        line = json.loads(publish_commitments.public_lines(self.conn)[0])
        line["n_picks"] = 3
        git = FakeGit(head=json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(publish_commitments.PublishRefused):
            publish_commitments.publish(self.conn, self.repo, True, git=git)
        self.assertNotIn("commit", git.calls)

    def test_a_broken_chain_is_never_published(self):
        self.conn.execute("DROP TRIGGER predictions_log_immutable")
        self.conn.execute("UPDATE predictions_log SET winner_confidence = 70 WHERE id = 1")
        self.conn.commit()
        with self.assertRaises(publish_commitments.PublishRefused):
            publish_commitments.public_lines(self.conn)


class ScheduleTest(unittest.TestCase):

    def test_hourly_commits_before_it_publishes_and_pushes(self):
        names = [n for n, _ in run_scheduled.JOBS["hourly"]]
        self.assertLess(names.index("commit ledger (NBA)"), names.index("publish commitments"))
        self.assertLess(names.index("publish commitments"), names.index("publish ledger"))
        self.assertIn("--publish", dict(run_scheduled.JOBS["hourly"])["publish commitments"])


PINNED_PICKS = None
PINNED_CHAIN = None


def _pin():
    """The pinned hashes, computed by hand from the documented form (not by lc)."""
    rows = [
        [1, "2026-10-21", "NBA", "fanduel", "Boston Celtics:New York Knicks", "2026-10-21T23:30:00+00:00",
         "2026-10-21T13:00:00+00:00", "candidate_2026-08", "Boston Celtics", "64.250000",
         "-150.000000", "130.000000"],
        [2, "2026-10-21", "NBA", "fanduel", "Los Angeles Lakers:Golden State Warriors",
         "2026-10-22T02:00:00+00:00", "2026-10-21T13:00:00+00:00", "candidate_2026-08",
         "Golden State Warriors", "55.100000", "110.000000", "-130.000000"],
    ]
    pre = json.dumps(["bb-ledger-commit-v1", "2026-10-21", 1, NONCE_A, rows], separators=(",", ":"))
    picks = hashlib.sha256(pre.encode()).hexdigest()
    link = json.dumps(["bb-ledger-commit-v1", "chain", "0" * 64, "2026-10-21", 1, 2,
                       "2026-10-21T13:05:00+00:00", picks], separators=(",", ":"))
    return picks, hashlib.sha256(link.encode()).hexdigest()


PINNED_PICKS, PINNED_CHAIN = _pin()
# And as literals, so a change to json.dumps itself would show up here too.
assert PINNED_PICKS == "a48200ad66af316a1c7ae2954982ddc954ef3ef06cbc21a8a03eb61b2dcf81a2", PINNED_PICKS
assert PINNED_CHAIN == "62c50e566eeada2e5b7a370651f010c5679dd2f48b5991bc941ad38093deace8", PINNED_CHAIN


if __name__ == "__main__":
    unittest.main()
