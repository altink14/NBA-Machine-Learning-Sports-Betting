"""
Nba_Clv_Test.py
===============
Closing line value for the NBA prediction log, end to end, through the real
write paths: main_api.log_predictions logs the pick, the recorder's own
write_snapshot_rows / record_poll lay down the captures and the heartbeat,
grade_predictions.grade grades it, grade_predictions.price_clv prices it.
Only the clock and the fixture data are synthetic. No network.

WHY THESE CASES. The arithmetic is easy. What decides whether a published CLV
means anything is WHICH price counts as the close, and every way that choice
can go quietly wrong is a case here: a line that moved toward us or away, a
price that never moved (so the recorder wrote nothing near tip), a missed
capture, a book that pulled the market, a postponed game, a playoff rematch
sharing the game key, a rebuilt price, and a second run trying to re-price.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime as real_datetime, timedelta, timezone
from unittest import mock

import main_api
import grade_predictions as gp
from src.Utils import nba_clv
from src.Utils.odds_api_client import record_poll, write_snapshot_rows

HOME, AWAY = "Boston Celtics", "Utah Jazz"
BOOKS = ("fanduel", "draftkings", "betmgm", "williamhill_us")


def _iso(d):
    return d.isoformat()


class _ClvCase(unittest.TestCase):
    """A scratch OddsData + TeamData, one logged pick on the home side."""

    #: What FanDuel offered when the pick was logged: home -150 / away +130.
    LOGGED = (-150, 130)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="nba_clv_test_")
        self.odds_db = os.path.join(self.dir, "OddsData.sqlite")
        self.team_db = os.path.join(self.dir, "TeamData.sqlite")
        self.patches = [
            mock.patch.object(main_api, "ODDS_DB_PATH", self.odds_db),
            # Tomorrow must be a regular-season date or the preseason guard
            # (correctly) refuses the pick.
            mock.patch("preflight_opening_night.OPENING_NIGHT",
                       (real_datetime.now(timezone.utc) - timedelta(days=30)).date()),
        ]
        for p in self.patches:
            p.start()
        self.tip = (real_datetime.now(timezone.utc) + timedelta(hours=3)).replace(
            second=0, microsecond=0)
        pick = {"home_team": HOME, "away_team": AWAY,
                "home_odds": self.LOGGED[0], "away_odds": self.LOGGED[1],
                "predicted_winner": HOME, "winner_confidence": 64.0,
                "model": "xgboost_cand_2026-08", "expected_value": {},
                "game_start_time_utc": _iso(self.tip)}
        counts = main_api.log_predictions({"predictions": [pick]}, "fanduel", "NBA")
        self.assertEqual(counts["written"], 1)
        self.conn = sqlite3.connect(self.odds_db)
        self.conn.row_factory = sqlite3.Row
        self.frozen_pick = self._pick_columns()
        self._team_db()

    def tearDown(self):
        self.conn.close()
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- fixtures ---------------------------------------------------------
    def _team_db(self, played_date=None):
        c = sqlite3.connect(self.team_db)
        c.execute("CREATE TABLE IF NOT EXISTS team_metadata (team_id INTEGER, full_name TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS team_game_advanced (team_id INTEGER, "
                  "opp_team_id INTEGER, pts INTEGER, opp_pts INTEGER, game_date TEXT)")
        c.execute("DELETE FROM team_metadata")
        c.executemany("INSERT INTO team_metadata VALUES (?, ?)", [(1, HOME), (2, AWAY)])
        c.commit()
        c.close()

    def box_score(self, days_late=0):
        """The game's final, on the logged tip's Eastern date (+ days_late)."""
        d = real_datetime.fromisoformat(gp._et_date(_iso(self.tip))) + timedelta(days=days_late)
        c = sqlite3.connect(self.team_db)
        c.execute("INSERT INTO team_game_advanced VALUES (1, 2, 112, 104, ?)",
                  (d.date().isoformat(),))
        c.commit()
        c.close()

    def row(self, book, home_ml, away_ml, start=None):
        return {"sportsbook": book, "game_key": f"{HOME}:{AWAY}", "home_team": HOME,
                "away_team": AWAY,
                "game_start_time_utc": (start or self.tip).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_ml": home_ml, "away_ml": away_ml,
                "spread_home": None if home_ml is not None else -3.5,
                "spread_home_price": None if home_ml is not None else -110,
                "spread_away_price": None if home_ml is not None else -110,
                "ou_line": None, "ou_over_price": None, "ou_under_price": None}

    def poll(self, minutes_before_tip, rows):
        """One recorder poll: changed rows written, heartbeat recorded."""
        at = _iso(self.tip - timedelta(minutes=minutes_before_tip))
        write_snapshot_rows(self.conn, rows, sport="NBA", captured_at=at)
        record_poll(self.conn, polled_at=at, sport="NBA", source="odds_api", status="ok",
                    covers_board=True, markets="ml,spread,total", events=1,
                    book_rows=len(rows), rows=rows)

    def board(self, fd=None, dk=(-160, 136), mgm=(-158, 134), wh=(-162, 138)):
        prices = {"fanduel": fd if fd is not None else self.LOGGED, "draftkings": dk,
                  "betmgm": mgm, "williamhill_us": wh}
        return [self.row(b, *p) for b, p in prices.items() if p is not False]

    def settle(self, hours_after_tip=14):
        """Grade, then price, as the 9 a.m. job does the next morning."""
        now = self.tip + timedelta(hours=hours_after_tip)

        class Frozen(real_datetime):
            @classmethod
            def utcnow(cls):
                return now.replace(tzinfo=None)

        with mock.patch.object(gp, "datetime", Frozen):
            gp.grade(odds_db=self.odds_db, team_db=self.team_db)
        counts = gp.price_clv(odds_db=self.odds_db, team_db=self.team_db, now=now)
        return counts, self.conn.execute("SELECT * FROM predictions_log").fetchone()

    def _pick_columns(self):
        return tuple(self.conn.execute(
            "SELECT predicted_winner, winner_confidence, home_ml, away_ml, logged_at, "
            "game_start_time_utc, game_key, model FROM predictions_log").fetchone())

    def assertPickUntouched(self):
        self.assertEqual(self._pick_columns(), self.frozen_pick)


class TestLineMovement(_ClvCase):

    def test_line_moved_toward_the_pick_is_positive_everywhere(self):
        self.poll(300, self.board())
        # The market came to Boston: FanDuel -150 -> -175 at ten minutes out.
        self.poll(10, self.board(fd=(-175, 150), dk=(-172, 146), mgm=(-170, 145),
                                 wh=(-178, 150)))
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "priced")
        self.assertEqual(r["clv_consensus_status"], "priced")
        self.assertEqual((r["closing_home_ml"], r["closing_away_ml"]), (-175, 150))
        self.assertEqual(r["closing_confirmed_by"], "heartbeat")
        self.assertAlmostEqual(r["closing_minutes_before_tip"], 10.0)
        # Price: 1/(150/250)... decimal 1.6667 against 1.5714.
        self.assertAlmostEqual(r["clv"], (1 + 100 / 150) / (1 + 100 / 175) - 1, places=9)
        self.assertGreater(r["clv_prob"], 0)
        self.assertEqual(r["consensus_books"], 4)
        self.assertEqual(r["consensus_provenance"], "observed")
        self.assertGreater(r["clv_consensus_prob"], 0)
        self.assertTrue(r["clv_method"].startswith("bb-nba-clv-v1/"))
        self.assertPickUntouched()

    def test_line_moved_away_is_negative(self):
        self.poll(300, self.board())
        self.poll(12, self.board(fd=(-130, 110), dk=(-128, 108), mgm=(-132, 112),
                                 wh=(-130, 110)))
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "priced")
        self.assertLess(r["clv"], 0)
        self.assertLess(r["clv_prob"], 0)
        self.assertLess(r["clv_consensus_prob"], 0)

    def test_unmoved_price_is_confirmed_by_the_heartbeat_not_its_old_row(self):
        # The recorder writes nothing when nothing moves, so FanDuel's only row
        # is from the morning. The polls near tip saw it standing.
        self.poll(300, self.board())
        self.poll(20, self.board())
        self.poll(5, self.board())
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "priced")
        self.assertEqual(r["closing_confirmed_by"], "heartbeat")
        self.assertAlmostEqual(r["closing_minutes_before_tip"], 5.0)
        self.assertEqual(r["closing_captured_at"],
                         _iso(self.tip - timedelta(minutes=300)))
        self.assertAlmostEqual(r["clv"], 0.0, places=12)
        self.assertAlmostEqual(r["clv_prob"], 0.0, places=12)

    def test_listed_in_play_confirms_at_the_last_poll_before_tip(self):
        self.poll(300, self.board())
        self.poll(20, self.board())
        self.poll(5, self.board())
        # In-play: the feed keeps the game and prices move after tip. Post-tip
        # rows are never the close; the last pre-tip poll is.
        self.poll(-30, self.board(fd=(-400, 300)))
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "priced")
        self.assertEqual((r["closing_home_ml"], r["closing_away_ml"]), (-150, 130))
        self.assertEqual(r["closing_confirmed_by"], "poll")
        self.assertAlmostEqual(r["closing_minutes_before_tip"], 5.0)


class TestMissingAndStale(_ClvCase):

    def test_missed_capture_is_null_with_a_reason_and_never_estimated(self):
        self.poll(300, self.board())   # and nothing after: the PC slept
        self.box_score()
        counts, r = self.settle()
        # A repair could still buy the close back, so it waits...
        self.assertEqual(counts, {"waiting_close_too_early": 1})
        self.assertIsNone(r["clv_status"])
        # ...and after the repair window it settles, with no number.
        counts, r = self.settle(hours_after_tip=nba_clv.REPAIR_GRACE_HOURS + 1)
        self.assertEqual(r["clv_status"], "close_too_early")
        self.assertEqual(r["clv_consensus_status"], "too_few_books")
        self.assertIsNone(r["clv"])
        self.assertIsNone(r["clv_prob"])
        self.assertIsNone(r["clv_consensus_prob"])
        # The distance travels with the NULL.
        self.assertAlmostEqual(r["closing_minutes_before_tip"], 300.0)
        self.assertPickUntouched()

    def test_book_never_captured_leaves_same_book_null_but_consensus_stands(self):
        self.poll(15, self.board(fd=False))
        self.box_score()
        _, r = self.settle(hours_after_tip=nba_clv.REPAIR_GRACE_HOURS + 1)
        self.assertEqual(r["clv_status"], "no_capture")
        self.assertIsNone(r["clv"])
        # Never another book standing in for FanDuel.
        self.assertIsNone(r["closing_home_ml"])
        self.assertEqual(r["clv_consensus_status"], "priced")
        self.assertEqual(r["consensus_books"], 3)

    def test_too_few_books_for_a_consensus(self):
        self.poll(10, self.board(mgm=False, wh=False))
        self.box_score()
        _, r = self.settle(hours_after_tip=nba_clv.REPAIR_GRACE_HOURS + 1)
        self.assertEqual(r["clv_status"], "priced")
        self.assertEqual(r["clv_consensus_status"], "too_few_books")
        self.assertEqual(r["consensus_books"], 2)
        self.assertIsNone(r["clv_consensus_prob"])

    def test_not_started_is_not_priced(self):
        self.poll(10, self.board())
        counts = gp.price_clv(odds_db=self.odds_db, team_db=self.team_db,
                              now=self.tip - timedelta(minutes=1))
        self.assertEqual(counts, {"waiting_not_started": 1})


class TestPulledAndPostponed(_ClvCase):

    def test_book_pulled_before_tip(self):
        self.poll(300, self.board())
        self.poll(60, self.board())
        # A whole-board poll ten minutes out lists every book but FanDuel.
        self.poll(10, self.board(fd=False))
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "book_pulled")
        self.assertIsNone(r["clv"])
        self.assertEqual(r["clv_consensus_status"], "priced")
        self.assertEqual(r["consensus_books"], 3)   # FanDuel is not in it

    def test_book_took_the_moneyline_down_but_kept_the_game(self):
        self.poll(300, self.board())
        self.poll(10, [self.row("fanduel", None, None)] + self.board(fd=False))
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "book_pulled")
        self.assertIsNone(r["clv"])

    def test_game_moved_a_day_is_not_priced(self):
        # The grader accepts +/-1 day, so the pick gets a result; the market
        # we logged against was for the original night.
        self.poll(10, self.board())
        self.box_score(days_late=1)
        _, r = self.settle(hours_after_tip=30)
        self.assertIsNotNone(r["actual_winner"])
        self.assertEqual(r["clv_status"], "tip_moved")
        self.assertEqual(r["clv_consensus_status"], "tip_moved")
        self.assertIsNone(r["clv"])

    def test_postponed_and_unplayed_waits_forever_without_a_number(self):
        self.poll(10, self.board())
        counts, r = self.settle(hours_after_tip=24 * 10)
        self.assertEqual(counts, {"waiting_awaiting_result": 1})
        self.assertIsNone(r["clv_status"])
        self.assertIsNone(r["clv"])

    def test_playoff_rematch_on_the_same_key_is_not_the_close(self):
        # Game 2 at the same arena, two days later, is on the board from the
        # morning of game 1. Its rows and sightings share our game_key.
        game2 = self.tip + timedelta(days=2)
        self.poll(300, self.board())
        self.poll(300, [self.row(b, -120, 100, start=game2) for b in BOOKS])
        self.poll(10, [self.row(b, -115, -105, start=game2) for b in BOOKS])
        self.box_score()
        _, r = self.settle(hours_after_tip=nba_clv.REPAIR_GRACE_HOURS + 1)
        # The heartbeat saw "Boston:Utah" ten minutes out, but that was game
        # 2; game 1's last known price is from the morning.
        self.assertEqual(r["clv_status"], "close_too_early")
        self.assertEqual((r["closing_home_ml"], r["closing_away_ml"]), (-150, 130))
        self.assertIsNone(r["clv"])


class TestProvenanceAndOnce(_ClvCase):

    def test_reconstructed_close_is_stored_as_such_and_kept_out_of_the_headline(self):
        self.poll(300, self.board())
        at = _iso(self.tip - timedelta(minutes=5))
        for b, (h, a) in zip(BOOKS, [(-170, 145), (-168, 144), (-165, 140), (-170, 145)]):
            self.conn.execute(
                "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                "home_team, away_team, home_ml, away_ml, game_start_time_utc, provenance) "
                "VALUES (?, 'NBA', ?, ?, ?, ?, ?, ?, ?, 'reconstructed')",
                (at, b, f"{HOME}:{AWAY}", HOME, AWAY, h, a, _iso(self.tip)))
        self.conn.commit()
        self.box_score()
        _, r = self.settle()
        self.assertEqual(r["clv_status"], "priced")
        self.assertEqual(r["closing_provenance"], "reconstructed")
        self.assertEqual(r["consensus_provenance"], "mixed")
        s = nba_clv.summary(self.conn, now=self.tip + timedelta(days=1))
        self.assertEqual(s["consensus"]["n"], 0)
        self.assertEqual(s["same_book"]["price"]["n"], 0)
        self.assertEqual(s["reconstructed"], {"consensus_n": 1, "same_book_n": 1})

    def test_settled_once_and_the_pick_is_never_written(self):
        self.poll(300, self.board())
        self.poll(10, self.board(fd=(-175, 150)))
        self.box_score()
        _, first = self.settle()
        # A later price turns up (a repair, a stray write). Nothing moves.
        self.poll(2, self.board(fd=(-110, -110)))
        counts, again = self.settle(hours_after_tip=40)
        self.assertEqual(counts, {})
        self.assertEqual(dict(first), dict(again))
        # And the guard triggers are still there: the pick cannot change.
        with self.assertRaises(sqlite3.DatabaseError):
            self.conn.execute("UPDATE predictions_log SET predicted_winner = ?", (AWAY,))
        self.assertPickUntouched()


class TestReadSide(_ClvCase):

    def test_summary_and_endpoint(self):
        from fastapi.testclient import TestClient
        client = TestClient(main_api.app)
        empty = client.get("/api/track-record/clv").json()
        self.assertEqual(empty["consensus"]["n"], 0)
        self.assertIsNone(empty["consensus"]["mean"])
        self.assertEqual(empty["definition"]["close_window_minutes"], 30.0)

        self.poll(300, self.board())
        self.poll(10, self.board(fd=(-175, 150), dk=(-172, 146), mgm=(-170, 145),
                                 wh=(-178, 150)))
        self.box_score()
        _, r = self.settle()
        s = nba_clv.summary(self.conn, now=self.tip + timedelta(days=1))
        self.assertEqual(s["settled"], 1)
        self.assertEqual(s["consensus"]["n"], 1)
        self.assertAlmostEqual(s["consensus"]["mean"], round(100 * r["clv_consensus_prob"], 2))
        self.assertIsNone(s["consensus"]["ci95"])          # one pick has no interval
        self.assertEqual(s["consensus"]["beat_close"], 1)
        self.assertEqual(s["same_book"]["price"]["n"], 1)
        self.assertEqual(s["median_minutes_before_tip"], 10.0)

    def test_intervals(self):
        b = nba_clv._block([0.01, 0.03, -0.01, 0.02], 100, 2)
        self.assertEqual(b["n"], 4)
        self.assertAlmostEqual(b["mean"], 1.25)
        lo, hi = b["ci95"]
        self.assertLess(lo, 1.25)
        self.assertGreater(hi, 1.25)
        self.assertEqual(b["beat_close"], 3)
        lo, hi = b["beat_share_ci95"]
        self.assertLess(lo, 75.0)
        self.assertGreater(hi, 75.0)


class TestPinnedToTheRecorder(unittest.TestCase):

    def test_close_window_is_the_recorders_closing_window(self):
        import snapshot_odds_api
        self.assertEqual(nba_clv.CLOSE_WINDOW_MINUTES, snapshot_odds_api.CAPTURE_LADDER[0][0])


if __name__ == "__main__":
    unittest.main()
