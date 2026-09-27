"""
Odds_Board_Test.py
==================
Guards src/Utils/odds_board.py, the read side of the odds archive behind
/api/lineshop and /api/line-movements.

The bug it pins (found 2026-09-24, fixed 2026-09-27): the recorders write a
row only when a book's numbers CHANGE, but the Line Shop kept only quotes
CAPTURED in the last 7 days. A line that sat still for a week vanished, so
opening night was missing and the board showed 17 of the ~41 upcoming games.
Line Movement read one book over a 48-hour window and called the window's
first change "open". Here a book's latest row is its price until a newer row
replaces it, and "open" is its first capture over the whole archive.

Synthetic tests use a throwaway database; the archive test is read-only and
skipped when the real OddsData.sqlite is absent.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Utils import odds_board  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")

NOW = datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)
TIP = "2026-10-20T23:30:00Z"

SCHEMA = """
CREATE TABLE odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL, sport TEXT NOT NULL, sportsbook TEXT NOT NULL,
    game_key TEXT NOT NULL, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    home_ml REAL, away_ml REAL, ou_line REAL, game_start_time_utc TEXT,
    spread_home REAL, spread_home_price REAL, spread_away_price REAL,
    ou_over_price REAL, ou_under_price REAL,
    provenance TEXT NOT NULL DEFAULT 'observed')
"""


def _iso(days_ago, naive=False):
    d = NOW - timedelta(days=days_ago)
    return d.replace(tzinfo=None).isoformat() if naive else d.isoformat()


class _Db:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)

    def add(self, days_ago, book="draftkings", key="Houston Rockets:Dallas Mavericks",
            start=TIP, ml=(-310, 250), spread=(-8.5, -102, -118), total=(227.5, -110, -110),
            naive=False, sport="NBA"):
        home, away = key.split(":")
        self.conn.execute(
            "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, home_team, "
            "away_team, home_ml, away_ml, ou_line, game_start_time_utc, spread_home, "
            "spread_home_price, spread_away_price, ou_over_price, ou_under_price) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_iso(days_ago, naive), sport, book, key, home, away, ml[0], ml[1], total[0], start,
             spread[0], spread[1], spread[2], total[1], total[2]))

    def board(self, **kw):
        return odds_board.build_board(self.conn, now=NOW, **kw)


class TestUnchangedQuotesStay(unittest.TestCase):

    def test_a_price_unchanged_for_a_month_is_still_on_the_board(self):
        # The regression: one row, 38 days old, never changed since.
        db = _Db()
        db.add(38)
        board = db.board()
        self.assertEqual(len(board["games"]), 1)
        q = board["games"][0]["books"]["draftkings"]
        self.assertEqual((q["home_ml"], q["away_ml"]), (-310, 250))
        self.assertEqual(q["since"]["ml"], _iso(38))
        self.assertEqual(q["changes"], {"ml": 0, "spread": 0, "total": 0})

    def test_since_spans_rows_written_because_another_market_moved(self):
        # DraftKings MAV@HOU from the audit: the spread moved while the
        # moneyline sat at -310. The ML's "since" is its first appearance.
        db = _Db()
        db.add(38, spread=(-8.5, -102, -118))
        db.add(33, spread=(-7.5, -118, -102))
        db.add(4, spread=(-7.5, -118, -102), total=(226.5, -110, -110))
        q = db.board()["games"][0]["books"]["draftkings"]
        self.assertEqual(q["since"]["ml"], _iso(38))
        self.assertEqual(q["since"]["spread"], _iso(33))
        self.assertEqual(q["since"]["total"], _iso(4))
        self.assertEqual(q["last_change_at"], _iso(4))

    def test_open_is_the_first_capture_ever_not_the_first_in_a_window(self):
        db = _Db()
        db.add(38, spread=(-8.5, -102, -118))
        db.add(30, spread=(-7.5, -118, -102))
        db.add(1, spread=(-7.0, -110, -110))
        q = db.board()["games"][0]["books"]["draftkings"]
        self.assertEqual(q["open"]["spread"]["spread_home"], -8.5)
        self.assertEqual(q["open"]["spread"]["captured_at"], _iso(38))
        self.assertEqual(q["spread_home"], -7.0)
        self.assertEqual(q["changes"]["spread"], 2)
        self.assertEqual(q["first_seen_at"], _iso(38))

    def test_a_line_that_moved_and_came_back_counts_its_changes(self):
        db = _Db()
        db.add(10, spread=(-8.5, -102, -118))
        db.add(9, spread=(-7.5, -118, -102))
        db.add(8, spread=(-8.5, -102, -118))
        q = db.board()["games"][0]["books"]["draftkings"]
        self.assertEqual(q["open"]["spread"]["spread_home"], q["spread_home"])
        self.assertEqual(q["changes"]["spread"], 2)
        self.assertEqual(q["since"]["spread"], _iso(8))

    def test_open_for_a_market_first_posted_later_is_its_own_first_capture(self):
        db = _Db()
        db.add(20, ml=(None, None))
        db.add(12, ml=(-150, 130))
        q = db.board()["games"][0]["books"]["draftkings"]
        self.assertEqual(q["open"]["ml"]["captured_at"], _iso(12))
        self.assertEqual(q["open"]["spread"]["captured_at"], _iso(20))

    def test_mixed_timestamp_formats_sort_by_time_not_by_string(self):
        # Naive UTC (legacy writer) sorts after '+00:00' strings of the same
        # prefix only by accident; the newer row must win on time.
        db = _Db()
        db.add(5, ml=(-200, 170))
        db.add(2, ml=(-220, 180), naive=True)
        q = db.board()["games"][0]["books"]["draftkings"]
        self.assertEqual(q["home_ml"], -220)


class TestWhatDrops(unittest.TestCase):

    def test_started_games_and_missing_start_times_drop(self):
        db = _Db()
        db.add(3, key="Boston Celtics:Miami Heat", start="2026-09-26T23:00:00Z")
        db.add(3, key="New York Liberty:Dallas Wings", start=None)
        db.add(3)
        board = db.board()
        self.assertEqual([g["game_key"] for g in board["games"]], ["Houston Rockets:Dallas Mavericks"])
        self.assertEqual(board["dropped"]["started"], 1)

    def test_a_book_that_took_every_market_down_drops(self):
        db = _Db()
        db.add(10, book="bovada")
        db.add(2, book="bovada", ml=(None, None), spread=(None, None, None), total=(None, None, None))
        db.add(10, book="draftkings")
        board = db.board()
        self.assertEqual(sorted(board["games"][0]["books"]), ["draftkings"])
        self.assertEqual(board["dropped"]["no_price"], 1)

    def test_a_pulled_moneyline_drops_but_the_spread_stays(self):
        db = _Db()
        db.add(10)
        db.add(2, ml=(None, None))
        q = db.board()["games"][0]["books"]["draftkings"]
        self.assertIsNone(q["home_ml"])
        self.assertIsNone(q["since"]["ml"])
        self.assertEqual(q["spread_home"], -8.5)
        self.assertEqual(q["since"]["spread"], _iso(10))
        # A book with no moneyline takes no part in the fair price.
        self.assertIsNone(db.board()["games"][0]["fair"])

    def test_other_sports_and_book_filter(self):
        db = _Db()
        db.add(3, sport="WNBA", key="New York Liberty:Dallas Wings")
        db.add(3, book="draftkings")
        db.add(3, book="fanduel")
        self.assertEqual(len(db.board()["games"]), 1)
        self.assertEqual(sorted(db.board(sportsbook="fanduel")["games"][0]["books"]), ["fanduel"])


class TestFairAndBest(unittest.TestCase):

    def test_fair_is_the_median_de_vig_and_best_is_the_longest_price(self):
        db = _Db()
        db.add(3, book="draftkings", ml=(-310, 250))
        db.add(3, book="bovada", ml=(-310, 255))
        g = db.board()["games"][0]
        self.assertEqual(g["fair"]["books_used"], 2)
        self.assertAlmostEqual(g["fair"]["home_prob"] + g["fair"]["away_prob"], 1.0, places=3)
        self.assertEqual(g["best"]["away_ml"]["book"], "bovada")
        self.assertEqual(g["best"]["away_ml"]["price"], 255)


class TestEndpoints(unittest.TestCase):

    def setUp(self):
        import main_api
        from fastapi.testclient import TestClient
        self.tmp = tempfile.mkdtemp()
        path = os.path.join(self.tmp, "OddsData.sqlite")
        conn = sqlite3.connect(path)
        conn.execute(SCHEMA)
        conn.execute(
            "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, home_team, "
            "away_team, home_ml, away_ml, ou_line, game_start_time_utc, spread_home) "
            "VALUES ('2026-08-20T17:08:10+00:00','NBA','draftkings','A B:C D','A B','C D',"
            "-150,130,220.5,'2099-10-20T23:30:00Z',-3.5)")
        conn.commit()
        conn.close()
        self.patches = [mock.patch.object(main_api, "ODDS_DB_PATH", path),
                        mock.patch.object(main_api, "API_KEY", "")]
        for p in self.patches:
            p.start()
        self.client = TestClient(main_api.app)

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_lineshop_keeps_an_old_unchanged_quote(self):
        body = self.client.get("/api/lineshop").json()
        self.assertEqual(len(body["games"]), 1)
        q = body["games"][0]["books"]["draftkings"]
        self.assertEqual(q["since"]["ml"], "2026-08-20T17:08:10+00:00")
        self.assertNotIn("as_of", body["games"][0])

    def test_line_movements_reports_open_for_every_book(self):
        body = self.client.get("/api/line-movements").json()
        self.assertIsNone(body["sportsbook"])
        m = body["movements"][0]
        self.assertEqual(m["sportsbook"], "draftkings")
        self.assertEqual(m["open"]["spread"]["spread_home"], -3.5)
        self.assertEqual(m["current"]["ml"], {"home_ml": -150, "away_ml": 130})


@unittest.skipUnless(os.path.exists(ODDS_DB), "odds archive not present")
class TestAgainstTheArchive(unittest.TestCase):

    def setUp(self):
        self.conn = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.now = datetime.now(timezone.utc)
        self.board = odds_board.build_board(self.conn, now=self.now)

    def tearDown(self):
        self.conn.close()

    def test_every_upcoming_priced_game_in_the_archive_is_on_the_board(self):
        now_z = self.now.strftime("%Y-%m-%dT%H:%M:%SZ")
        expected = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT game_key FROM odds_snapshots WHERE sport='NBA' "
            "AND game_start_time_utc > ? AND (home_ml IS NOT NULL OR spread_home IS NOT NULL "
            "OR ou_line IS NOT NULL)", (now_z,))}
        self.assertEqual({g["game_key"] for g in self.board["games"]}, expected)

    def test_open_and_now_match_the_first_and_last_rows(self):
        for g in self.board["games"]:
            for book, q in g["books"].items():
                rows = self.conn.execute(
                    "SELECT * FROM odds_snapshots WHERE sport='NBA' AND game_key=? AND sportsbook=?",
                    (g["game_key"], book)).fetchall()
                rows = sorted(rows, key=lambda r: (odds_board.parse_ts(r["captured_at"]), r["id"]))
                last = rows[-1]
                self.assertEqual(q["last_change_at"], last["captured_at"])
                if last["spread_home"] is not None:
                    first = next(r for r in rows if r["spread_home"] is not None
                                 or r["spread_home_price"] is not None)
                    self.assertEqual(q["spread_home"], last["spread_home"])
                    self.assertEqual(q["open"]["spread"]["spread_home"], first["spread_home"])


if __name__ == "__main__":
    unittest.main()
