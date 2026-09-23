"""
Fix_WNBA_Snapshots_Test.py
==========================
The one-off relabel in fix_wnba_snapshots_tagged_nba.py, against a temp copy.
A dry run must write nothing; --apply must change the sport column of exactly
the both-teams-WNBA rows and nothing else.
"""

import io
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fix_wnba_snapshots_tagged_nba as F  # noqa: E402

ROWS = [
    ("2026-07-07T04:21:39", "NBA", "New York Liberty", "Dallas Wings"),     # fix
    ("2026-07-07T05:25:18", "NBA", "Phoenix Mercury", "Chicago Sky"),       # fix
    ("2026-08-20T17:08:10", "NBA", "Detroit Pistons", "Boston Celtics"),    # real NBA
    ("2026-08-21T00:00:00", "WNBA", "Seattle Storm", "Dallas Wings"),       # already right
    ("2026-08-22T00:00:00", "NBA", "Phoenix Suns", "Phoenix Mercury"),      # ambiguous: left alone
]


class TestRelabel(unittest.TestCase):

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="fix_wnba_"), "OddsData.sqlite")
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "captured_at TEXT, sport TEXT, sportsbook TEXT, game_key TEXT, home_team TEXT, "
                  "away_team TEXT, home_ml REAL, away_ml REAL, ou_line REAL)")
        for cap, sport, h, a in ROWS:
            c.execute("INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                      "home_team, away_team) VALUES (?,?,?,?,?,?)",
                      (cap, sport, "fanduel", f"{h}:{a}", h, a))
        c.commit()
        c.close()

    def state(self):
        c = sqlite3.connect(self.db)
        try:
            return c.execute("SELECT * FROM odds_snapshots ORDER BY id").fetchall()
        finally:
            c.close()

    def run_script(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = F.main(["--db", self.db, *argv])
        return code, out.getvalue()

    def test_dry_run_prints_the_rows_and_writes_nothing(self):
        before = self.state()
        code, out = self.run_script()
        self.assertEqual(code, 0)
        self.assertIn("2 row(s)", out)
        self.assertIn("New York Liberty:Dallas Wings", out)
        self.assertIn("NOT touched: 1", out)
        self.assertEqual(self.state(), before)

    def test_apply_changes_only_the_sport_of_the_wnba_rows(self):
        before = self.state()
        code, _ = self.run_script("--apply")
        self.assertEqual(code, 0)
        after = self.state()
        self.assertEqual([r[2] for r in after], ["WNBA", "WNBA", "NBA", "WNBA", "NBA"])
        for b, a in zip(before, after):     # every other column untouched
            self.assertEqual(b[:2] + b[3:], a[:2] + a[3:])
        code, out = self.run_script("--apply")
        self.assertEqual(code, 0)
        self.assertIn("nothing to do", out)


if __name__ == "__main__":
    unittest.main()
