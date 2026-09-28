"""
Closing_Lines_Test.py
=====================
Guards src/Utils/closing_lines.py, which puts closing line value next to the
bets a MEMBER logs on /bets (POST /api/closing-lines/lookup).

What it pins:
  * the close is the SAME book's last pre-tip price, never another book's;
  * nothing is priced before tip, and a row captured after tip is never the close;
  * spreads/totals are compared on price only at the same number, otherwise
    the result is "line_moved" with the difference in points;
  * the away spread is the home spread negated;
  * how close to tip the price was last confirmed travels with it;
  * unknown stays None, never 0.

Synthetic tests use an in-memory database; the archive test is read-only and
skipped when the real OddsData.sqlite is absent.
"""

import os
import sqlite3
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Utils import closing_lines  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")

TIP = "2026-10-20T23:30:00Z"            # 7:30 pm ET on 2026-10-20
AFTER = datetime(2026, 10, 21, 5, 0, tzinfo=timezone.utc)
BEFORE = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
KEY = "Houston Rockets:Dallas Mavericks"

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

HEARTBEAT = """
CREATE TABLE odds_seen (
    source TEXT NOT NULL, sport TEXT NOT NULL, game_key TEXT NOT NULL,
    sportsbook TEXT NOT NULL, market TEXT NOT NULL, first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL, last_poll_id INTEGER)
"""


class _Db:
    def __init__(self, heartbeat=False):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        if heartbeat:
            self.conn.execute(HEARTBEAT)

    def add(self, captured, book="draftkings", key=KEY, start=TIP, ml=(-310, 250),
            spread=(-8.5, -110, -110), total=(227.5, -110, -110), sport="NBA",
            provenance="observed"):
        home, away = key.split(":")
        self.conn.execute(
            "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, home_team, "
            "away_team, home_ml, away_ml, ou_line, game_start_time_utc, spread_home, "
            "spread_home_price, spread_away_price, ou_over_price, ou_under_price, provenance) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (captured, sport, book, key, home, away, ml[0], ml[1], total[0], start,
             spread[0], spread[1], spread[2], total[1], total[2], provenance))

    def seen(self, at, market="ml", book="draftkings", key=KEY):
        self.conn.execute(
            "INSERT INTO odds_seen VALUES ('odds_api','NBA',?,?,?,?,?,1)",
            (key, book, market, at, at))

    def one(self, now=AFTER, **bet):
        base = {"id": "b1", "game_date": "2026-10-20", "home_team": "Houston Rockets",
                "away_team": "Dallas Mavericks", "sportsbook": "DraftKings",
                "market": "moneyline", "side": "away", "odds_american": 260}
        base.update(bet)
        return closing_lines.lookup(self.conn, [base], now=now)["results"][0]


class TestSameBookLastPreTipPrice(unittest.TestCase):

    def test_moneyline_clv_against_the_same_books_last_pre_tip_price(self):
        db = _Db()
        db.add("2026-10-19T12:00:00+00:00", ml=(-300, 245))
        db.add("2026-10-20T22:50:00+00:00", ml=(-320, 255))
        db.add("2026-10-20T22:00:00+00:00", book="fanduel", ml=(-340, 280))
        r = db.one()
        self.assertEqual(r["status"], "priced")
        self.assertEqual(r["close"]["book"], "draftkings")
        self.assertEqual(r["close"]["price"], 255)
        # decimal 3.60 taken vs 3.55 close
        self.assertAlmostEqual(r["clv_pct"], round((3.60 / 3.55 - 1) * 100, 2))
        self.assertEqual(r["close"]["minutes_before_tip"], 40.0)
        self.assertEqual(r["close"]["confirmed_by"], "captured")
        self.assertEqual(r["books_with_close"], ["draftkings", "fanduel"])

    def test_a_row_captured_after_tip_is_never_the_close(self):
        db = _Db()
        db.add("2026-10-20T23:00:00+00:00", ml=(-320, 255))
        db.add("2026-10-21T00:10:00+00:00", ml=(-1000, 650))   # in-play
        r = db.one()
        self.assertEqual(r["close"]["price"], 255)

    def test_nothing_is_priced_before_tip(self):
        db = _Db()
        db.add("2026-10-20T10:00:00+00:00")
        r = db.one(now=BEFORE)
        self.assertEqual(r["status"], "not_started")
        self.assertIsNone(r["close"])
        self.assertIsNone(r["clv_pct"])

    def test_another_book_is_listed_never_substituted(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00", book="fanduel")
        r = db.one(sportsbook="Caesars")
        self.assertEqual(r["status"], "book_not_recorded")
        self.assertIsNone(r["close"])
        self.assertEqual(r["books_with_close"], ["fanduel"])

    def test_book_names_as_members_type_them(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00", book="betonlineag")
        db.add("2026-10-20T22:00:00+00:00", book="williamhill_us")
        self.assertEqual(db.one(sportsbook="BetOnline.ag")["status"], "priced")
        self.assertEqual(db.one(sportsbook="Caesars Sportsbook")["status"], "priced")

    def test_home_and_away_either_way_round_and_clippers_spelling(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00", key="Los Angeles Clippers:Dallas Mavericks",
               ml=(-150, 130))
        # The member typed the teams the other way round; their side is the Clippers.
        r = db.one(home_team="Dallas Mavericks", away_team="LA Clippers", side="away", odds_american=-140)
        self.assertEqual(r["status"], "priced")
        self.assertEqual(r["close"]["price"], -150)

    def test_wrong_date_is_no_game_and_says_where_the_archive_starts(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00")
        r = db.one(game_date="2026-10-21")
        self.assertEqual(r["status"], "no_game")
        self.assertTrue(r["archive_first_tip"].startswith("2026-10-20T23:30"))

    def test_wnba_rows_never_answer_an_nba_bet(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00", sport="WNBA")
        self.assertEqual(db.one()["status"], "no_game")

    def test_props_and_parlays_have_no_archived_close(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00")
        self.assertEqual(db.one(market="player_prop")["status"], "unsupported_market")

    def test_side_unknown_is_reported_not_guessed(self):
        db = _Db()
        db.add("2026-10-20T22:00:00+00:00")
        r = db.one(side=None)
        self.assertEqual(r["status"], "side_unknown")
        self.assertIsNone(r["clv_pct"])


class TestSpreadsAndTotals(unittest.TestCase):

    def test_same_number_compares_price(self):
        db = _Db()
        db.add("2026-10-20T23:00:00+00:00", spread=(-8.5, -115, -105))
        r = db.one(market="spread", side="home", line=-8.5, odds_american=-110)
        self.assertEqual(r["status"], "priced")
        self.assertEqual(r["close"]["line"], -8.5)
        self.assertAlmostEqual(r["clv_pct"], round(((1 + 100 / 110) / (1 + 100 / 115) - 1) * 100, 2))

    def test_away_spread_is_the_home_spread_negated(self):
        db = _Db()
        db.add("2026-10-20T23:00:00+00:00", spread=(-8.5, -115, -105))
        r = db.one(market="spread", side="away", line=8.5, odds_american=-110)
        self.assertEqual(r["close"]["line"], 8.5)
        self.assertEqual(r["close"]["price"], -105)
        self.assertEqual(r["status"], "priced")

    def test_moved_number_reports_points_not_a_percentage(self):
        db = _Db()
        db.add("2026-10-20T23:00:00+00:00", spread=(-8.5, -110, -110), total=(229.5, -110, -110))
        r = db.one(market="spread", side="away", line=9.5, odds_american=-110)
        self.assertEqual((r["status"], r["line_diff"], r["clv_pct"]), ("line_moved", 1.0, None))
        r = db.one(market="total", side="over", line=227.5, odds_american=-110)
        self.assertEqual((r["status"], r["line_diff"]), ("line_moved", 2.0))
        r = db.one(market="total", side="under", line=227.5, odds_american=-110)
        self.assertEqual((r["status"], r["line_diff"]), ("line_moved", -2.0))

    def test_missing_member_line_is_unknown_not_zero(self):
        db = _Db()
        db.add("2026-10-20T23:00:00+00:00")
        r = db.one(market="spread", side="home", line=None, odds_american=-110)
        self.assertEqual(r["status"], "line_unknown")
        self.assertIsNotNone(r["close"])
        self.assertIsNone(r["line_diff"])

    def test_a_pulled_market_is_no_market(self):
        db = _Db()
        db.add("2026-10-20T23:00:00+00:00", spread=(None, None, None))
        r = db.one(market="spread", side="home", line=-8.5, odds_american=-110)
        self.assertEqual(r["status"], "no_market")


class TestHowCloseToTip(unittest.TestCase):

    def test_a_heartbeat_sighting_before_tip_confirms_the_price(self):
        db = _Db(heartbeat=True)
        db.add("2026-10-20T12:00:00+00:00")
        db.seen("2026-10-20T23:20:00+00:00")
        r = db.one()
        self.assertEqual(r["close"]["confirmed_by"], "heartbeat")
        self.assertEqual(r["close"]["minutes_before_tip"], 10.0)

    def test_a_sighting_after_tip_with_no_change_means_it_held_to_tip(self):
        db = _Db(heartbeat=True)
        db.add("2026-10-20T12:00:00+00:00")
        db.seen("2026-10-21T00:30:00+00:00")
        self.assertEqual(db.one()["close"]["minutes_before_tip"], 0.0)

    def test_a_post_tip_change_leaves_only_the_capture_time(self):
        db = _Db(heartbeat=True)
        db.add("2026-10-20T12:00:00+00:00")
        db.add("2026-10-21T00:10:00+00:00", ml=(-900, 600))
        db.seen("2026-10-21T00:30:00+00:00")
        r = db.one()
        self.assertEqual(r["close"]["confirmed_by"], "captured")
        self.assertEqual(r["close"]["minutes_before_tip"], 690.0)

    def test_no_heartbeat_means_only_the_capture_time(self):
        db = _Db()
        db.add("2026-10-20T12:00:00+00:00")
        r = db.one()
        self.assertEqual(r["close"]["confirmed_by"], "captured")
        self.assertEqual(r["close"]["minutes_before_tip"], 690.0)


@unittest.skipUnless(os.path.exists(ODDS_DB), "real odds archive not present")
class TestAgainstTheArchive(unittest.TestCase):

    def test_reads_the_real_archive_without_writing(self):
        conn = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            out = closing_lines.lookup(conn, [{
                "id": "x", "game_date": "2026-10-20", "home_team": "Oklahoma City Thunder",
                "away_team": "Houston Rockets", "sportsbook": "DraftKings",
                "market": "moneyline", "side": "home", "odds_american": -200}])
        finally:
            conn.close()
        self.assertEqual(len(out["results"]), 1)
        # Before opening night nothing can be priced; after it, a result exists.
        self.assertIn(out["results"][0]["status"],
                      {"not_started", "no_game", "priced", "book_not_recorded", "no_market"})


if __name__ == "__main__":
    unittest.main()
