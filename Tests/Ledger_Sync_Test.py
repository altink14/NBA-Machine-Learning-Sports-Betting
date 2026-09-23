"""The public copy of the ledger can only grow, and only by what the home PC wrote.

Every test here builds a throwaway "home" database and a throwaway "server"
database in a temp directory. Nothing touches Data/OddsData.sqlite.
"""

import gzip
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient

import main_api
from src.Sports.ledger import ensure_ledger
from src.Utils import ledger_sync

ODDS_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, captured_at TEXT NOT NULL, sport TEXT NOT NULL,
    sportsbook TEXT NOT NULL, game_key TEXT NOT NULL, home_team TEXT NOT NULL,
    away_team TEXT NOT NULL, home_ml REAL, away_ml REAL, ou_line REAL,
    game_start_time_utc TEXT, provenance TEXT NOT NULL DEFAULT 'observed')
"""

NOW = datetime.now(timezone.utc).replace(microsecond=0)
LOGGED = NOW.isoformat()
TIP = (NOW + timedelta(days=1)).isoformat()
TODAY = (main_api.to_nba_date(NOW) or NOW.date()).isoformat()


def make_db(path, with_snapshots=True):
    conn = sqlite3.connect(path)
    conn.executescript(main_api._PREDICTION_LOG_SCHEMA)
    ensure_ledger(conn)
    if with_snapshots:
        conn.executescript(ODDS_SNAPSHOTS_SQL)
    conn.commit()
    return conn


def add_pick(conn, home="Boston Celtics", away="New York Knicks", winner=None, conf=64.0,
             log_date=TODAY, sportsbook="fanduel"):
    conn.execute(
        "INSERT INTO predictions_log (logged_at, log_date, sport, sportsbook, game_key, home_team, "
        "away_team, game_start_time_utc, home_ml, away_ml, ou_line, predicted_winner, "
        "winner_confidence, ev_home, ev_away, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (LOGGED, log_date, "NBA", sportsbook, f"{home}:{away}", home, away, TIP, -150, 130, 221.5,
         winner or home, conf, 0.04, -0.07, "candidate_2026-08"))
    conn.commit()


def add_nfl_row(conn, game_id="2026_05_KC_BUF"):
    conn.execute(
        "INSERT INTO ledger (sport, league, game_id, market_type, side, price_taken, model_version, "
        "model_prob, created_at, event_start_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("football", "NFL", game_id, "moneyline", "home", -120, "nfl_v2", 0.55, LOGGED, TIP))
    conn.commit()


class LedgerSyncTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_sync_test_")
        self.home_path = os.path.join(self.dir, "home.sqlite")
        self.server_path = os.path.join(self.dir, "server.sqlite")
        self.home = make_db(self.home_path)
        self.server = make_db(self.server_path, with_snapshots=False)

    def tearDown(self):
        self.home.close()
        self.server.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def upload(self, conn=None):
        """What push_ledger sends: a VACUUM INTO copy of the home database."""
        conn = conn or self.home
        dest = os.path.join(self.dir, f"upload_{len(os.listdir(self.dir))}.sqlite")
        conn.execute("VACUUM INTO ?", (dest,))
        return dest

    def local_fp(self, path):
        c = sqlite3.connect(path)
        try:
            return ledger_sync.fingerprint(c)
        finally:
            c.close()

    def server_fp(self):
        return ledger_sync.fingerprint(self.server)

    # --- what should happen -------------------------------------------------

    def test_first_sync_copies_everything_and_fingerprints_match(self):
        add_pick(self.home)
        add_pick(self.home, "Denver Nuggets", "Utah Jazz")
        add_nfl_row(self.home)
        self.home.execute("INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                          "home_team, away_team, home_ml, away_ml) VALUES (?,?,?,?,?,?,?,?)",
                          (LOGGED, "NBA", "fanduel", "a:b", "a", "b", -110, -110))
        self.home.commit()
        up = self.upload()
        result = ledger_sync.merge(self.server, up)
        self.assertEqual(result["tables"]["predictions_log"]["inserted"], 2)
        self.assertEqual(result["tables"]["ledger"]["inserted"], 1)
        self.assertEqual(result["tables"]["odds_snapshots"]["inserted"], 1)
        self.assertEqual(result["fingerprint"], self.local_fp(up))

    def test_grading_flows_through_as_an_update(self):
        add_pick(self.home)
        ledger_sync.merge(self.server, self.upload())
        # The grader adds its CLV columns with ALTER TABLE, so the home copy
        # has them and a server booted from an older snapshot may not.
        for col in ("closing_home_ml REAL", "clv REAL"):
            self.home.execute(f"ALTER TABLE predictions_log ADD COLUMN {col}")
        self.home.execute("UPDATE predictions_log SET actual_winner='Boston Celtics', "
                          "actual_total=219, closing_home_ml=-165, clv=0.031")
        self.home.commit()
        up = self.upload()
        result = ledger_sync.merge(self.server, up)
        self.assertEqual(result["tables"]["predictions_log"],
                         {"inserted": 0, "updated": 1,
                          "columns_added": ["closing_home_ml", "clv"], "rows": 1})
        self.assertEqual(result["fingerprint"], self.local_fp(up))

    def test_same_upload_twice_changes_nothing(self):
        add_pick(self.home)
        up = self.upload()
        ledger_sync.merge(self.server, up)
        again = ledger_sync.merge(self.server, up)
        self.assertEqual(again["tables"]["predictions_log"]["inserted"], 0)
        self.assertEqual(again["tables"]["predictions_log"]["updated"], 0)

    def test_nfl_ledger_grading_is_mirrored(self):
        add_nfl_row(self.home)
        ledger_sync.merge(self.server, self.upload())
        self.home.execute("UPDATE ledger SET result='loss', graded_at=?, closing_price=-130",
                          (LOGGED,))
        self.home.commit()
        up = self.upload()
        ledger_sync.merge(self.server, up)
        self.assertEqual(self.server.execute("SELECT result FROM ledger").fetchone()[0], "loss")
        self.assertEqual(self.server_fp(), self.local_fp(up))

    def test_a_column_added_at_home_is_added_on_the_server(self):
        add_pick(self.home)
        self.home.execute("ALTER TABLE predictions_log ADD COLUMN closing_book TEXT")
        self.home.commit()
        result = ledger_sync.merge(self.server, self.upload())
        self.assertEqual(result["tables"]["predictions_log"]["columns_added"], ["closing_book"])

    # --- what must be refused, with nothing written -------------------------

    def assertRefusedUnchanged(self, upload_path, fragment):
        before = self.server_fp()
        with self.assertRaises(ledger_sync.SyncRefused) as ctx:
            ledger_sync.merge(self.server, upload_path)
        self.assertIn(fragment, str(ctx.exception))
        self.assertEqual(self.server_fp(), before, "a refused sync must write nothing")

    def tampered(self, sql, params=()):
        """An upload in which the home copy's own triggers were bypassed."""
        path = self.upload()
        c = sqlite3.connect(path)
        for trig in ("predictions_log_immutable", "predictions_log_no_regrade",
                     "predictions_log_no_delete", "ledger_prediction_is_immutable",
                     "ledger_no_regrade", "ledger_no_delete"):
            c.execute(f"DROP TRIGGER IF EXISTS {trig}")
        c.execute(sql, params)
        c.commit()
        c.close()
        return path

    def test_changed_pick_is_refused(self):
        add_pick(self.home)
        ledger_sync.merge(self.server, self.upload())
        up = self.tampered("UPDATE predictions_log SET predicted_winner='New York Knicks'")
        self.assertRefusedUnchanged(up, "immutable")

    def test_changed_confidence_is_refused(self):
        add_pick(self.home)
        ledger_sync.merge(self.server, self.upload())
        up = self.tampered("UPDATE predictions_log SET winner_confidence=71.0")
        self.assertRefusedUnchanged(up, "immutable")

    def test_regrade_is_refused(self):
        add_pick(self.home)
        self.home.execute("UPDATE predictions_log SET actual_winner='Boston Celtics'")
        self.home.commit()
        ledger_sync.merge(self.server, self.upload())
        up = self.tampered("UPDATE predictions_log SET actual_winner='New York Knicks'")
        self.assertRefusedUnchanged(up, "regraded")

    def test_nfl_regrade_is_refused(self):
        add_nfl_row(self.home)
        self.home.execute("UPDATE ledger SET result='win'")
        self.home.commit()
        ledger_sync.merge(self.server, self.upload())
        up = self.tampered("UPDATE ledger SET result='loss'")
        self.assertRefusedUnchanged(up, "regraded")

    def test_a_deleted_row_is_refused(self):
        add_pick(self.home)
        add_pick(self.home, "Denver Nuggets", "Utah Jazz")
        ledger_sync.merge(self.server, self.upload())
        up = self.tampered("DELETE FROM predictions_log WHERE home_team='Denver Nuggets'")
        self.assertRefusedUnchanged(up, "deletion")

    def test_a_late_pick_is_refused(self):
        add_pick(self.home)
        ledger_sync.merge(self.server, self.upload())
        # A home copy whose CHECK was stripped, carrying a pick made after tip-off.
        path = os.path.join(self.dir, "late.sqlite")
        c = sqlite3.connect(path)
        sql = self.home.execute("SELECT sql FROM sqlite_master WHERE name='predictions_log'").fetchone()[0]
        c.execute(sql.replace("CHECK (logged_at < game_start_time_utc),", ""))
        c.execute("ATTACH DATABASE ? AS h", (self.home_path,))
        c.execute("INSERT INTO predictions_log SELECT * FROM h.predictions_log")
        c.commit()
        c.execute("DETACH DATABASE h")
        c.execute("INSERT INTO predictions_log (logged_at, log_date, sport, sportsbook, game_key, "
                  "home_team, away_team, game_start_time_utc, predicted_winner) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (TIP, TODAY, "NBA", "fanduel", "Miami Heat:Orlando Magic", "Miami Heat",
                   "Orlando Magic", LOGGED, "Miami Heat"))
        c.commit()
        c.close()
        self.assertRefusedUnchanged(path, "CHECK")

    def test_independently_written_copies_are_refused(self):
        # The server wrote its own pick for one game; home's id 1 is another game.
        add_pick(self.server, "Phoenix Suns", "Dallas Mavericks")
        add_pick(self.home)
        self.assertRefusedUnchanged(self.upload(), "different game")

    def test_an_unguarded_server_is_refused(self):
        add_pick(self.home)
        self.server.execute("DROP TRIGGER predictions_log_immutable")
        self.server.commit()
        self.assertRefusedUnchanged(self.upload(), "guard trigger")

    def test_a_file_that_is_not_sqlite_is_refused(self):
        path = os.path.join(self.dir, "junk.sqlite")
        with open(path, "wb") as fh:
            fh.write(b"this is not a database" * 100)
        self.assertRefusedUnchanged(path, "SQLite")


class LedgerSyncEndpointTest(unittest.TestCase):
    """The HTTP route and the ledger-served /predictions, against temp files."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ledger_sync_api_")
        self.server_path = os.path.join(self.dir, "OddsData.sqlite")
        make_db(self.server_path, with_snapshots=False).close()
        self.home_path = os.path.join(self.dir, "home.sqlite")
        self.home = make_db(self.home_path)
        add_pick(self.home)
        self.patches = [mock.patch.object(main_api, "ODDS_DB_PATH", self.server_path),
                        mock.patch.object(main_api, "LEDGER_SYNC_SECRET", "s3cret-for-tests"),
                        mock.patch.object(main_api, "API_KEY", "")]
        for p in self.patches:
            p.start()
        self.client = TestClient(main_api.app)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.home.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def payload(self):
        dest = os.path.join(self.dir, f"up_{len(os.listdir(self.dir))}.sqlite")
        self.home.execute("VACUUM INTO ?", (dest,))
        with open(dest, "rb") as fh:
            return gzip.compress(fh.read())

    def post(self, body, secret="s3cret-for-tests"):
        return self.client.post("/api/admin/ledger/sync", content=body,
                                headers={"X-Ledger-Sync-Secret": secret})

    def test_unconfigured_server_refuses(self):
        with mock.patch.object(main_api, "LEDGER_SYNC_SECRET", ""):
            self.assertEqual(self.post(self.payload()).status_code, 503)

    def test_wrong_secret_refuses(self):
        self.assertEqual(self.post(self.payload(), secret="nope").status_code, 401)

    def test_push_then_serve_the_logged_pick(self):
        r = self.post(self.payload())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["tables"]["predictions_log"]["inserted"], 1)
        with mock.patch.object(main_api, "PREDICTIONS_SOURCE", "ledger"):
            p = self.client.get("/predictions?sportsbook=fanduel").json()
        self.assertEqual(p["source"], "ledger")
        self.assertEqual(len(p["predictions"]), 1)
        pick = p["predictions"][0]
        self.assertEqual(pick["predicted_winner"], "Boston Celtics")
        self.assertEqual(pick["winner_confidence"], 64.0)
        self.assertEqual(pick["logged_at"], LOGGED)
        self.assertIsNone(pick["under_over_prediction"], "the withdrawn O/U pick must not return")

    def test_tampered_push_is_409_and_writes_nothing(self):
        self.assertEqual(self.post(self.payload()).status_code, 200)
        dest = os.path.join(self.dir, "tampered.sqlite")
        self.home.execute("VACUUM INTO ?", (dest,))
        c = sqlite3.connect(dest)
        c.execute("DROP TRIGGER predictions_log_immutable")
        c.execute("UPDATE predictions_log SET predicted_winner='New York Knicks'")
        c.commit()
        c.close()
        with open(dest, "rb") as fh:
            r = self.post(gzip.compress(fh.read()))
        self.assertEqual(r.status_code, 409, r.text)
        c = sqlite3.connect(self.server_path)
        self.assertEqual(c.execute("SELECT predicted_winner FROM predictions_log").fetchone()[0],
                         "Boston Celtics")
        c.close()

    def test_ledger_mode_with_nothing_logged_says_so(self):
        with mock.patch.object(main_api, "PREDICTIONS_SOURCE", "ledger"):
            p = self.client.get("/predictions?sportsbook=fanduel").json()
        self.assertEqual(p["predictions"], [])
        self.assertIn("No picks have been logged", p["note"])


if __name__ == "__main__":
    unittest.main()
