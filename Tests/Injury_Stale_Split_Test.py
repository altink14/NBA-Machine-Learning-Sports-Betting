"""Stale injury notes never move a prediction (owner's decision 2026-09-28).

espn_injuries.split_stale() removes notes older than STALE_AFTER_DAYS from
the absences the prediction path and availability.matchup_availability read,
keeps undated notes (their age is unknown), and reports what it removed.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Utils import espn_injuries  # noqa: E402

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def absences(*players):
    by_team = {}
    for team, name, date in players:
        by_team.setdefault(team, []).append({"player_id": 1, "name": name, "status": "Out", "detail": "", "date": date})
    return {"by_team": by_team, "source": "espn", "total_counted": len(players)}


class SplitStaleTest(unittest.TestCase):
    def test_old_note_is_removed_and_reported(self):
        fresh, stale = espn_injuries.split_stale(absences(("MIN", "Old Note", "2026-04-27T15:00Z")), now=NOW)
        self.assertEqual(fresh["by_team"], {})
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["team"], "MIN")
        self.assertEqual(stale[0]["age_days"], 153)
        self.assertEqual(fresh["stale_ignored"], 1)

    def test_recent_note_is_kept(self):
        fresh, stale = espn_injuries.split_stale(absences(("PHX", "Recent", "2026-09-20T15:00Z")), now=NOW)
        self.assertEqual([p["name"] for p in fresh["by_team"]["PHX"]], ["Recent"])
        self.assertEqual(stale, [])

    def test_boundary_thirty_days_is_kept(self):
        fresh, stale = espn_injuries.split_stale(absences(("BOS", "Edge", "2026-08-29T12:00Z")), now=NOW)
        self.assertIn("BOS", fresh["by_team"])
        self.assertEqual(stale, [])

    def test_undated_note_is_kept(self):
        fresh, stale = espn_injuries.split_stale(absences(("NYK", "No Date", "")), now=NOW)
        self.assertIn("NYK", fresh["by_team"])
        self.assertEqual(stale, [])

    def test_mixed_team_keeps_only_fresh(self):
        fresh, stale = espn_injuries.split_stale(
            absences(("DEN", "Fresh", "2026-09-25T00:00Z"), ("DEN", "Stale", "2026-05-01T00:00Z")), now=NOW)
        self.assertEqual([p["name"] for p in fresh["by_team"]["DEN"]], ["Fresh"])
        self.assertEqual([s["name"] for s in stale], ["Stale"])

    def test_idempotent(self):
        once, _ = espn_injuries.split_stale(absences(("DEN", "Fresh", "2026-09-25T00:00Z")), now=NOW)
        twice, stale = espn_injuries.split_stale(once, now=NOW)
        self.assertEqual(once["by_team"], twice["by_team"])
        self.assertEqual(stale, [])

    def test_unavailable_feed_passes_through(self):
        fresh, stale = espn_injuries.split_stale({"by_team": {}, "source": "unavailable"}, now=NOW)
        self.assertEqual(fresh["source"], "unavailable")
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
