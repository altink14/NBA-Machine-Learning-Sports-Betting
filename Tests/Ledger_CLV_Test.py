"""
Ledger_CLV_Test.py
==================
Guards the two rules that decide what a published CLV number means.

WHY THESE AND NOT THE ARITHMETIC. The division is obvious and hard to get
wrong. What is easy to get wrong is WHICH closing price the division runs
against, and that choice is invisible in the output: a CLV of -9% and a CLV
of +5% look equally like numbers. These tests pin the choice down.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime as real_datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.Sports.ledger as L


#: A kickoff in the past, and a prediction written two hours before it. The
#: ledger refuses a prediction written after kickoff, as it should, so the
#: clock moves rather than the constraint.
KICKOFF = "2026-09-20T17:00:00+00:00"
WRITTEN = real_datetime(2026, 9, 20, 15, 0, tzinfo=timezone.utc)

OBSERVED = {"book": "draftkings", "provenance": "observed", "line": None, "price_home": -110}
RECONSTRUCTED = {"book": "fanduel", "provenance": "reconstructed", "line": None, "price_home": 105}
#: Longer than either, and quoted by nobody: nflverse publishes a consensus
#: close for every finished game, with no book and no capture time.
THIRD_PARTY = {"book": "consensus", "provenance": "third_party", "line": None, "price_home": 120}


class _Frozen(real_datetime):
    @classmethod
    def now(cls, tz=None):
        return WRITTEN if tz else WRITTEN.replace(tzinfo=None)


class TestClosePreference(unittest.TestCase):

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="clv_test_"), "ledger.sqlite")
        self.conn = sqlite3.connect(self.db)
        L.ensure_ledger(self.conn)
        L.datetime = _Frozen
        try:
            L.write_prediction(
                self.conn, sport="nfl", league="NFL", competition_id="nfl-2026-w03",
                game_id="G1", market_type="moneyline", side="home",
                model_version="test", model_prob=0.5, price_taken=100,
                event_start_utc=KICKOFF)
        finally:
            L.datetime = real_datetime

    def tearDown(self):
        self.conn.close()

    def _settle(self, candidates):
        self.conn.execute("UPDATE ledger SET clv=NULL, closing_book=NULL, "
                          "closing_provenance=NULL")
        self.conn.commit()
        L.apply_clv(self.conn, {("G1", "moneyline"): candidates})
        return self.conn.execute(
            "SELECT closing_book, closing_provenance, closing_price, clv FROM ledger"
        ).fetchone()

    def test_watched_close_beats_a_longer_consensus(self):
        # The consensus price (+120) is longer than the book's (-110), so a
        # plain "best price wins" rule takes it. It must not: settling a bet
        # against a number no book quoted turns +4.8% CLV into -9.1%, and the
        # sign of a CLV figure is the whole claim.
        book, prov, price, clv = self._settle([OBSERVED, THIRD_PARTY])
        self.assertEqual(prov, "observed")
        self.assertEqual(book, "draftkings")
        self.assertEqual(price, -110)
        self.assertGreater(clv, 0)

    def test_watched_close_beats_a_longer_reconstruction(self):
        # Same rule one tier down: a price bought back out of the archive
        # does not displace one we watched, however good it looks.
        book, prov, _, _ = self._settle([RECONSTRUCTED, OBSERVED])
        self.assertEqual(prov, "observed")
        self.assertEqual(book, "draftkings")

    def test_reconstructed_beats_consensus(self):
        # With nothing watched, a timestamped archive price still outranks a
        # book-unspecified one.
        _, prov, _, _ = self._settle([THIRD_PARTY, RECONSTRUCTED])
        self.assertEqual(prov, "reconstructed")

    def test_falls_back_rather_than_refusing(self):
        # Tiering is a preference, not a filter. A consensus close is worth
        # having as long as the row says that is what it is.
        book, prov, _, clv = self._settle([THIRD_PARTY])
        self.assertEqual(prov, "third_party")
        self.assertEqual(book, "consensus")
        self.assertIsNotNone(clv)

    def test_best_book_still_wins_inside_a_tier(self):
        # The conservative rule survives: among prices of the same kind we
        # settle against the longest, which makes our own CLV smaller.
        short = dict(OBSERVED, book="bovada", price_home=-130)
        book, prov, price, _ = self._settle([short, OBSERVED])
        self.assertEqual(prov, "observed")
        self.assertEqual(book, "draftkings")     # -110 is longer than -130
        self.assertEqual(price, -110)


class TestKickoffGuard(unittest.TestCase):
    """A game that has not started has no close, whatever a flag says."""

    def test_unstarted_game_is_not_priced(self):
        db = os.path.join(tempfile.mkdtemp(prefix="clv_test_"), "ledger.sqlite")
        conn = sqlite3.connect(db)
        L.ensure_ledger(conn)
        future = real_datetime(2030, 1, 1, tzinfo=timezone.utc).isoformat()
        L.write_prediction(conn, sport="nfl", league="NFL", competition_id="nfl-2030-w01",
                           game_id="G2", market_type="moneyline", side="home",
                           model_version="test", model_prob=0.5, price_taken=100,
                           event_start_utc=future)
        counts = L.apply_clv(conn, {("G2", "moneyline"): [OBSERVED]})
        self.assertEqual(counts["not_started"], 1)
        self.assertEqual(counts["priced"], 0)
        self.assertIsNone(conn.execute("SELECT clv FROM ledger").fetchone()[0])
        conn.close()


if __name__ == "__main__":
    unittest.main()
