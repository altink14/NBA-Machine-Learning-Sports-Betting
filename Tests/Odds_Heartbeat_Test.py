"""
Odds_Heartbeat_Test.py
======================
Guards the odds "last seen" heartbeat: odds_api_client.record_poll /
snapshot_nba_board (write side) and the withdrawn/confirmed rule in
odds_board.build_board (read side).

The bug it pins (found in the 2026-09-24 nav audit, fixed 2026-09-27): the
recorders write an odds_snapshots row only when a line CHANGES, so a book that
stopped listing a game wrote nothing and its last price stayed on the Line
Shop forever (146 book-prices on the board vs 144 in the last poll). Every
poll now records what it saw; a quote a later SUCCESSFUL whole-board poll did
not list leaves the board. A failed or empty poll must never pull anything.

No network: fetch_nba_odds is mocked with Odds API-shaped fixtures, and every
database is a throwaway file.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Utils import odds_api_client as oc  # noqa: E402
from src.Utils import odds_board  # noqa: E402

T0 = datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc)
BOARD_NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
TIP = "2026-10-20T23:30:00Z"
HOU, DAL = "Houston Rockets", "Dallas Mavericks"
KEY = f"{HOU}:{DAL}"
BOS, MIA = "Boston Celtics", "Miami Heat"
_TMP = tempfile.TemporaryDirectory()


def tearDownModule():
    _TMP.cleanup()

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


def book(key, ml=(-310, 250), spread=(-8.5, -110, -110), total=(227.5, -110, -110),
         home=HOU, away=DAL):
    """One bookmaker entry in The Odds API's /odds shape."""
    return {"key": key, "markets": [
        {"key": "h2h", "outcomes": [{"name": home, "price": ml[0]},
                                    {"name": away, "price": ml[1]}]},
        {"key": "spreads", "outcomes": [{"name": home, "point": spread[0], "price": spread[1]},
                                        {"name": away, "point": -spread[0], "price": spread[2]}]},
        {"key": "totals", "outcomes": [{"name": "Over", "point": total[0], "price": total[1]},
                                       {"name": "Under", "point": total[0], "price": total[2]}]},
    ]}


def event(*books, home=HOU, away=DAL, start=TIP):
    return {"home_team": home, "away_team": away, "commence_time": start,
            "bookmakers": list(books)}


class _Archive:
    """A throwaway OddsData.sqlite and a clock-controlled recorder."""

    def __init__(self):
        fd, self.path = tempfile.mkstemp(suffix=".sqlite", dir=_TMP.name)
        os.close(fd)
        conn = sqlite3.connect(self.path)
        conn.execute(SCHEMA)
        conn.commit()
        conn.close()

    def poll(self, at, events=None, error=None, bookmakers=None, write_error=None):
        """Run snapshot_nba_board at `at` against a mocked feed."""
        fake_dt = mock.Mock(wraps=datetime)
        fake_dt.now = lambda tz=None: at
        fetch = (mock.Mock(side_effect=error) if error
                 else mock.Mock(return_value=(events, {"remaining": "400", "used": "100"})))
        patches = [mock.patch.object(oc, "datetime", fake_dt),
                   mock.patch.object(oc, "fetch_nba_odds", fetch)]
        if write_error:
            patches.append(mock.patch.object(oc, "write_snapshot_rows",
                                             mock.Mock(side_effect=write_error)))
        for p in patches:
            p.start()
        try:
            return oc.snapshot_nba_board(self.path, bookmakers=bookmakers)
        finally:
            for p in patches:
                p.stop()

    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def board(self, now=BOARD_NOW):
        c = self.conn()
        try:
            return odds_board.build_board(c, now=now)
        finally:
            c.close()

    def game(self, key=KEY):
        return next(g for g in self.board()["games"] if g["game_key"] == key)

    def polls(self):
        c = self.conn()
        try:
            return [dict(r) for r in c.execute("SELECT * FROM odds_polls ORDER BY id")]
        finally:
            c.close()

    def count(self, table):
        c = self.conn()
        try:
            return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            c.close()

    def add_legacy_row(self, captured_at, sportsbook, key=KEY, ml=(-300, 240)):
        """A row written before the heartbeat existed (no poll, no sighting)."""
        home, away = key.split(":")
        c = sqlite3.connect(self.path)
        c.execute("INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                  "home_team, away_team, home_ml, away_ml, game_start_time_utc) "
                  "VALUES (?, 'NBA', ?, ?, ?, ?, ?, ?, ?)",
                  (captured_at.isoformat(), sportsbook, key, home, away, ml[0], ml[1], TIP))
        c.commit()
        c.close()


def t(hours):
    return T0 + timedelta(hours=hours)


class TestPresentPulledReturned(unittest.TestCase):

    def setUp(self):
        self.a = _Archive()
        # Poll 1: both books list the game.
        self.a.poll(t(0), [event(book("draftkings"), book("bovada", ml=(-305, 255)))])

    def test_present_both_books_confirmed(self):
        g = self.a.game()
        self.assertEqual(sorted(g["books"]), ["bovada", "draftkings"])
        q = g["books"]["bovada"]
        self.assertEqual(q["feed"], {"ml": "confirmed", "spread": "confirmed", "total": "confirmed"})
        self.assertEqual(q["last_seen_at"]["ml"], t(0).isoformat())
        self.assertEqual(g["withdrawn"], {})

    def test_an_unchanged_price_is_reconfirmed_without_a_new_row(self):
        rows_before = self.a.count("odds_snapshots")
        self.a.poll(t(12), [event(book("draftkings"), book("bovada", ml=(-305, 255)))])
        self.assertEqual(self.a.count("odds_snapshots"), rows_before)  # write-on-change intact
        q = self.a.game()["books"]["bovada"]
        self.assertEqual(q["last_seen_at"]["ml"], t(12).isoformat())
        self.assertEqual(q["since"]["ml"], t(0).isoformat())  # "since" still = first appearance

    def test_pulled_then_returned(self):
        # Poll 2: Bovada stops listing the game. It writes no snapshot row.
        rows_before = self.a.count("odds_snapshots")
        self.a.poll(t(12), [event(book("draftkings"))])
        self.assertEqual(self.a.count("odds_snapshots"), rows_before)
        board = self.a.board()
        g = board["games"][0]
        self.assertEqual(sorted(g["books"]), ["draftkings"])
        self.assertEqual(sorted(g["withdrawn"]["bovada"]), ["ml", "spread", "total"])
        w = g["withdrawn"]["bovada"]["ml"]
        self.assertEqual((w["home_ml"], w["away_ml"]), (-305, 255))
        self.assertEqual(w["last_seen_at"], t(0).isoformat())
        self.assertEqual(w["missing_since"], t(12).isoformat())
        self.assertEqual(board["dropped"]["withdrawn"], 1)
        self.assertEqual(board["dropped"]["withdrawn_markets"], 3)
        # A withdrawn price takes no part in the consensus or the best price.
        self.assertEqual(g["fair"]["books_used"], 1)
        self.assertEqual(g["best"]["away_ml"]["book"], "draftkings")

        # Poll 3: Bovada lists it again at the same price (still no new row).
        self.a.poll(t(24), [event(book("draftkings"), book("bovada", ml=(-305, 255)))])
        g = self.a.game()
        self.assertEqual(sorted(g["books"]), ["bovada", "draftkings"])
        self.assertEqual(g["books"]["bovada"]["last_seen_at"]["ml"], t(24).isoformat())
        self.assertEqual(g["withdrawn"], {})

    def test_a_game_gone_from_the_feed_entirely_drops_every_book(self):
        self.a.poll(t(12), [event(book("draftkings", home=BOS, away=MIA), home=BOS, away=MIA)])
        board = self.a.board()
        self.assertEqual([g["game_key"] for g in board["games"]], [f"{BOS}:{MIA}"])
        # It does not vanish without a trace.
        gone = board["withdrawn_games"]
        self.assertEqual([g["game_key"] for g in gone], [KEY])
        self.assertEqual(sorted(gone[0]["withdrawn"]), ["bovada", "draftkings"])
        self.assertEqual(gone[0]["withdrawn"]["draftkings"]["ml"]["missing_since"],
                         t(12).isoformat())

    def test_one_market_pulled_keeps_the_rest(self):
        # The book keeps listing the game but drops its moneyline. That is a
        # change, so the recorder writes a row with NULLs; the heartbeat agrees.
        bk = book("bovada")
        bk["markets"] = [m for m in bk["markets"] if m["key"] != "h2h"]
        self.a.poll(t(12), [event(book("draftkings"), bk)])
        q = self.a.game()["books"]["bovada"]
        self.assertIsNone(q["home_ml"])
        self.assertIsNone(q["feed"]["ml"])
        self.assertEqual(q["feed"]["spread"], "confirmed")


class TestAFailedPollPullsNothing(unittest.TestCase):

    def setUp(self):
        self.a = _Archive()
        self.a.poll(t(0), [event(book("draftkings"), book("bovada"))])

    def _assert_board_unchanged(self):
        g = self.a.game()
        self.assertEqual(sorted(g["books"]), ["bovada", "draftkings"])
        self.assertEqual(g["withdrawn"], {})
        self.assertEqual(g["books"]["bovada"]["last_seen_at"]["ml"], t(0).isoformat())

    def test_a_fetch_error_is_recorded_as_failed_and_re_raised(self):
        with self.assertRaises(oc.OddsApiError):
            self.a.poll(t(12), error=oc.OddsApiError("The Odds API returned 503"))
        last = self.a.polls()[-1]
        self.assertEqual((last["status"], last["polled_at"]), ("failed", t(12).isoformat()))
        self.assertIn("503", last["error"])
        self._assert_board_unchanged()
        self.assertEqual(self.a.board()["heartbeat"]["last_poll"]["status"], "failed")
        self.assertEqual(self.a.board()["heartbeat"]["last_ok_poll_at"], t(0).isoformat())

    def test_an_empty_board_is_recorded_as_empty_not_ok(self):
        self.a.poll(t(12), [])
        self.assertEqual(self.a.polls()[-1]["status"], "empty")
        self._assert_board_unchanged()

    def test_events_with_no_bookmakers_count_as_empty(self):
        self.a.poll(t(12), [event()])
        self.assertEqual(self.a.polls()[-1]["status"], "empty")
        self._assert_board_unchanged()

    def test_an_archive_write_failure_records_no_sightings(self):
        # We fetched the board but could not save it: confirming the OLD price
        # would be a lie, and the missing books prove nothing.
        with self.assertRaises(sqlite3.OperationalError):
            self.a.poll(t(12), [event(book("draftkings"))],
                        write_error=sqlite3.OperationalError("database is locked"))
        self.assertEqual(self.a.polls()[-1]["status"], "failed")
        self._assert_board_unchanged()

    def test_a_single_book_run_does_not_pull_other_books(self):
        self.a.poll(t(12), [event(book("draftkings"))], bookmakers="draftkings")
        self.assertEqual(self.a.polls()[-1]["books"], "draftkings")
        self._assert_board_unchanged()


class TestQuotesFromBeforeTheHeartbeat(unittest.TestCase):

    def test_no_heartbeat_tables_means_today_rule_labelled_unconfirmed(self):
        a = _Archive()
        a.add_legacy_row(t(-100), "betmgm")
        board = a.board()
        q = board["games"][0]["books"]["betmgm"]
        self.assertEqual(q["home_ml"], -300)
        self.assertEqual(q["feed"]["ml"], "unconfirmed")
        self.assertIsNone(q["last_seen_at"]["ml"])
        self.assertEqual(board["heartbeat"], {"since": None, "last_ok_poll_at": None,
                                              "last_poll": None})

    def test_a_book_the_feed_has_never_shown_is_not_pulled_by_it(self):
        a = _Archive()
        a.add_legacy_row(t(-100), "caesars")  # an SBR-only name
        a.poll(t(0), [event(book("draftkings"))])
        q = a.game()["books"]["caesars"]
        self.assertEqual(q["feed"]["ml"], "unconfirmed")

    def test_a_legacy_quote_the_feed_no_longer_lists_is_withdrawn(self):
        # betmgm is still carried (it prices another game), but not this one.
        a = _Archive()
        a.add_legacy_row(t(-100), "betmgm")
        a.poll(t(0), [event(book("draftkings")),
                      event(book("betmgm", home=BOS, away=MIA), home=BOS, away=MIA)])
        g = a.game()
        self.assertNotIn("betmgm", g["books"])
        w = g["withdrawn"]["betmgm"]["ml"]
        self.assertEqual(w["last_seen_at"], t(-100).isoformat())  # its row, never a sighting
        self.assertEqual(w["missing_since"], t(0).isoformat())

    def test_a_legacy_quote_the_feed_still_lists_becomes_confirmed(self):
        a = _Archive()
        a.add_legacy_row(t(-100), "draftkings", ml=(-310, 250))
        a.poll(t(0), [event(book("draftkings"))])
        q = a.game()["books"]["draftkings"]
        self.assertEqual(q["feed"]["ml"], "confirmed")


class TestSbrSightings(unittest.TestCase):
    """The SBR scraper sees one book on one date; its absence proves nothing."""

    def test_an_sbr_poll_confirms_but_never_pulls(self):
        a = _Archive()
        a.poll(t(0), [event(book("fanduel")),
                      event(book("fanduel", home=BOS, away=MIA), home=BOS, away=MIA)])
        c = a.conn()
        try:
            oc.record_poll(c, polled_at=t(6).replace(tzinfo=None).isoformat(), sport="NBA",
                           source="sbr", status="ok", covers_board=False, books="fanduel",
                           markets="ml,total", rows=[{"game_key": KEY, "sportsbook": "fanduel",
                                                      "home_ml": -310, "away_ml": 250,
                                                      "ou_line": 227.5}])
        finally:
            c.close()
        board = a.board()
        by_key = {g["game_key"]: g for g in board["games"]}
        self.assertEqual(sorted(by_key), [f"{BOS}:{MIA}", KEY])  # BOS:MIA not pulled
        q = by_key[KEY]["books"]["fanduel"]
        self.assertEqual(q["last_seen_at"]["ml"], t(6).isoformat())       # SBR's naive stamp
        self.assertEqual(q["last_seen_at"]["spread"], t(0).isoformat())   # SBR has no spreads

    def test_main_api_snapshot_odds_leaves_a_heartbeat(self):
        import main_api
        a = _Archive()
        with mock.patch.object(main_api, "ODDS_DB_PATH", a.path):
            main_api.snapshot_odds({KEY: {HOU: {"money_line_odds": -310},
                                          DAL: {"money_line_odds": 250},
                                          "under_over_odds": 227.5,
                                          "game_start_time_utc": TIP}}, "fanduel", "NBA")
        p = a.polls()[-1]
        self.assertEqual((p["source"], p["status"], p["covers_board"], p["books"]),
                         ("sbr", "ok", 0, "fanduel"))
        q = a.game()["books"]["fanduel"]
        self.assertEqual(q["feed"]["ml"], "confirmed")
        self.assertIsNone(q["feed"]["spread"])


class TestRecordPoll(unittest.TestCase):

    def test_first_seen_is_kept_and_last_seen_moves(self):
        c = sqlite3.connect(":memory:")
        row = {"game_key": KEY, "sportsbook": "draftkings", "home_ml": -310, "away_ml": 250,
               "spread_home": None, "spread_home_price": None, "spread_away_price": None,
               "ou_line": None, "ou_over_price": None, "ou_under_price": None}
        for at in (t(0), t(5)):
            oc.record_poll(c, polled_at=at.isoformat(), sport="NBA", source="odds_api",
                           status="ok", covers_board=True, markets="ml,spread,total", rows=[row])
        seen = c.execute("SELECT market, first_seen_at, last_seen_at FROM odds_seen").fetchall()
        c.close()
        # Only the priced market is a sighting; NULL spread/total are not.
        self.assertEqual(seen, [("ml", t(0).isoformat(), t(5).isoformat())])

    def test_the_status_check_refuses_nonsense(self):
        c = sqlite3.connect(":memory:")
        with self.assertRaises(sqlite3.IntegrityError):
            oc.record_poll(c, polled_at=t(0).isoformat(), sport="NBA", source="odds_api",
                           status="maybe", covers_board=True, markets="ml")
        c.close()


if __name__ == "__main__":
    unittest.main()
