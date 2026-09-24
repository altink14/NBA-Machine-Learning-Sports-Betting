"""Award winners and career leaders read the backfilled reference tables.

Added 2026-09-24 with backfill_reference.py. The answers are only complete
once every player has been asked about, so each carries its coverage. Temp
database only.
"""
import sqlite3
import unittest
from unittest import mock

from fastapi import HTTPException

import main_api


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _db(checked=3):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT, is_active INTEGER, from_year INTEGER, to_year INTEGER)")
    c.executemany("INSERT INTO players VALUES (?, ?, ?, ?, ?)", [
        (1, "Big Scorer", 0, 1990, 2010), (2, "Short Career", 0, 2000, 2002), (3, "Active Star", 1, 2015, 2026),
    ])
    c.execute("CREATE TABLE player_awards (player_id INTEGER, description TEXT, all_nba_team_number TEXT, season TEXT, team TEXT, fetched_at TEXT)")
    c.executemany("INSERT INTO player_awards VALUES (?, ?, ?, ?, ?, 'x')", [
        (1, "NBA Most Valuable Player", "", "2004-05", "AAA"),
        (3, "NBA Most Valuable Player", "", "2023-24", "CCC"),
        (1, "All-NBA", "1", "2004-05", "AAA"),
        (3, "All-NBA", "(null)", "2023-24", "CCC"),
    ])
    main_api._ensure_career_official_table(c)
    for pid, gp, pts in ((1, 1200, 30000), (2, 100, 3000), (3, 700, 18000)):
        c.execute(
            "INSERT INTO player_career_official (player_id, season, season_type, team_abbr, gp, pts, is_career_total, fetched_at) "
            "VALUES (?, 'CAREER', 'Regular Season', 'TOT', ?, ?, 1, 'x')", (pid, gp, pts))
    c.execute("CREATE TABLE reference_fetch (kind TEXT, key TEXT, fetched_at TEXT, n_rows INTEGER, error TEXT, PRIMARY KEY (kind, key))")
    for k in ("awards", "careers"):
        for pid in range(1, checked + 1):
            c.execute("INSERT INTO reference_fetch VALUES (?, ?, 'x', 1, NULL)", (k, str(pid)))
    c.commit()
    return c


class TestLeagueReference(unittest.TestCase):

    def _call(self, fn, checked=3, **kw):
        with mock.patch.object(main_api, "get_db_conn", return_value=_KeepOpen(_db(checked))):
            return fn(**kw)

    def test_mvp_alias_and_season(self):
        out = self._call(main_api.get_award_winners, award="MVP")
        self.assertEqual(out["award"], "NBA Most Valuable Player")
        self.assertEqual([w["full_name"] for w in out["winners"]], ["Active Star", "Big Scorer"])
        one = self._call(main_api.get_award_winners, award="mvp", season="2004-05")
        self.assertEqual([w["full_name"] for w in one["winners"]], ["Big Scorer"])

    def test_team_number_null_markers_become_none(self):
        out = self._call(main_api.get_award_winners, award="all-nba")
        numbers = {w["full_name"]: w["team_number"] for w in out["winners"]}
        self.assertEqual(numbers["Big Scorer"], "1")
        self.assertIsNone(numbers["Active Star"])

    def test_unknown_award_lists_known(self):
        with self.assertRaises(HTTPException) as ctx:
            self._call(main_api.get_award_winners, award="best haircut")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_coverage_says_incomplete(self):
        out = self._call(main_api.get_award_winners, checked=2, award="MVP")
        self.assertFalse(out["coverage"]["complete"])
        self.assertIn("2 of 3", out["coverage"]["note"])
        full = self._call(main_api.get_award_winners, checked=3, award="MVP")
        self.assertTrue(full["coverage"]["complete"])
        self.assertIsNone(full["coverage"]["note"])

    def test_career_totals_and_per_game_minimum(self):
        tot = self._call(main_api.get_career_leaders, stat="pts")
        self.assertEqual([l["full_name"] for l in tot["leaders"]], ["Big Scorer", "Active Star", "Short Career"])
        pg = self._call(main_api.get_career_leaders, stat="pts", per_game=True)
        # Short Career (30.0 in 100 games) is below the 400-game bar; 18,000 in
        # 700 (25.7) outranks 30,000 in 1,200 (25.0).
        self.assertEqual([l["full_name"] for l in pg["leaders"]], ["Active Star", "Big Scorer"])
        self.assertEqual(pg["leaders"][0]["value"], 25.7)
        act = self._call(main_api.get_career_leaders, stat="pts", active_only=True)
        self.assertEqual([l["full_name"] for l in act["leaders"]], ["Active Star"])

    def test_career_bad_stat(self):
        with self.assertRaises(HTTPException):
            self._call(main_api.get_career_leaders, stat="dunks")


if __name__ == "__main__":
    unittest.main()
