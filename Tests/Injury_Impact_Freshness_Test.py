"""Injury impact: old ESPN notes are dated and left out of totals; the on/off season is named.

Found 2026-09-24 (nav audit): ESPN keeps a player on its report until a new
note posts, so 39 of 70 entries were over 60 days old, and the Impact tab
summed them into team totals it called "this season" while pricing them from
2025-26 on/off against rosters that had changed over the summer. Temp
database and a mocked wire only; no network.
"""
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import main_api
from src.Utils import espn_injuries


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _db(last_game: str):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE team_metadata (team_id INTEGER, abbreviation TEXT)")
    c.execute("CREATE TABLE team_season_advanced (team_id INTEGER, season TEXT, season_type TEXT, pace REAL)")
    c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT, game_date TEXT)")
    c.execute("INSERT INTO team_metadata VALUES (1, 'MIN')")
    c.execute("INSERT INTO team_season_advanced VALUES (1, 'S', 'Regular Season', 100.0)")
    c.execute("INSERT INTO box_scores VALUES ('g1', 'S', 'Regular Season', '2026-04-12')")
    c.execute("INSERT INTO box_scores VALUES ('g2', 'S', 'Playoffs', ?)", (last_game,))
    return _KeepOpen(c)


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%MZ")


ONOFF = {
    "games_processed": 10,
    "players": [
        {"player_id": 1, "diff_per100": 10.0, "se_diff_per100": 2.0, "min_on": 240.0},
        {"player_id": 2, "diff_per100": 5.0, "se_diff_per100": 2.0, "min_on": 240.0},
    ],
}


def _run(entries, last_game="2026-06-13"):
    absences = {"by_team": {"MIN": entries}, "source": "espn", "fetched_at": "x",
                "total_counted": len(entries), "match_rate": 1.0, "counted_statuses": ["Out", "Doubtful"]}
    with mock.patch.object(main_api.espn_injuries, "get_absences", return_value=absences), \
         mock.patch.object(main_api, "get_team_onoff", return_value=ONOFF), \
         mock.patch.object(main_api, "get_db_conn", return_value=_db(last_game)):
        return main_api.get_injury_impact(season="S")


class StaleNotes(unittest.TestCase):
    def test_old_note_is_priced_but_left_out_of_the_team_total(self):
        res = _run([
            {"player_id": 1, "name": "Current", "status": "Out", "detail": "", "date": _iso(3)},
            {"player_id": 2, "name": "Old", "status": "Out", "detail": "", "date": _iso(154)},
        ])
        team = res["teams"][0]
        cur, old = team["players"]
        self.assertFalse(cur["stale"])
        self.assertEqual(cur["age_days"], 3)
        self.assertTrue(old["stale"])
        self.assertEqual(old["age_days"], 154)
        self.assertTrue(old["measured"])                          # still priced, for reference
        self.assertAlmostEqual(team["team_impact_pts"], cur["impact_pts"], places=2)   # ...not summed
        self.assertEqual(team["stale_left_out"], 1)
        self.assertEqual(res["stale_after_days"], main_api.INJURY_STALE_DAYS)

    def test_a_team_with_only_old_notes_has_no_total(self):
        res = _run([{"player_id": 1, "name": "Old", "status": "Out", "detail": "", "date": _iso(90)}])
        self.assertIsNone(res["teams"][0]["team_impact_pts"])

    def test_the_threshold_is_exclusive_and_missing_dates_are_not_stale(self):
        res = _run([
            {"player_id": 1, "name": "Edge", "status": "Out", "detail": "", "date": _iso(main_api.INJURY_STALE_DAYS)},
            {"player_id": 2, "name": "Undated", "status": "Out", "detail": "", "date": ""},
        ])
        edge, undated = res["teams"][0]["players"]
        self.assertFalse(edge["stale"])
        self.assertIsNone(undated["age_days"])
        self.assertFalse(undated["stale"])

    def test_the_wire_keeps_the_note_date(self):
        entries = [{"team_name": "Minnesota Timberwolves", "player_name": "X", "status": "Out",
                    "detail": "", "date": "2026-04-26T17:52Z"}]
        with mock.patch.object(espn_injuries, "resolve_team_abbr", return_value="MIN"), \
             mock.patch.object(espn_injuries, "_player_index", return_value={"x": 1}):
            out = espn_injuries.build_absences(entries)
        self.assertEqual(out["by_team"]["MIN"][0]["date"], "2026-04-26T17:52Z")


class SeasonNamed(unittest.TestCase):
    def test_a_finished_season_says_so(self):
        res = _run([{"player_id": 1, "name": "P", "status": "Out", "detail": "", "date": _iso(1)}],
                   last_game="2026-06-13")
        self.assertTrue(res["season_over"])
        self.assertEqual(res["onoff_last_game"], "2026-04-12")     # regular-season evidence
        self.assertEqual(res["season_last_game"], "2026-06-13")
        self.assertEqual(res["season"], "S")

    def test_a_season_in_progress_is_not_over(self):
        today = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        res = _run([{"player_id": 1, "name": "P", "status": "Out", "detail": "", "date": _iso(1)}],
                   last_game=today)
        self.assertFalse(res["season_over"])

    def test_unmeasured_reason_names_the_season_and_team(self):
        res = _run([{"player_id": 99, "name": "Rookie", "status": "Out", "detail": "", "date": _iso(1)}])
        why = res["teams"][0]["players"][0]["why_unmeasured"]
        self.assertIn("S", why)
        self.assertIn("MIN", why)
        self.assertNotIn("this season", why)


if __name__ == "__main__":
    unittest.main()
