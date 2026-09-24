"""Season records come from the league game log, ratings from the box scores.

Found 2026-09-24 by the chat eval. Four games nba.com never serves an advanced
box score for are missing from team_game_advanced, and wins were counted
there, so seven team-seasons read a game or two short (the 1996-97 Sonics
55-25 instead of 57-25). compute_and_save_season_stats now takes W-L from
game_results when it has the season. Throwaway database only.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "backfill_under_test_records", os.path.join(REPO, "src", "Process-Data", "backfill.py"))
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)

from src.Utils.nba_db_schema import ensure_schema, get_connection  # noqa: E402

A, B = 1610612760, 1610612744
SEASON = "1996-97"


class TestSeasonRecord(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.sqlite")
        ensure_schema(self.db)
        c = get_connection(self.db)
        c.execute("INSERT OR REPLACE INTO team_metadata (team_id, abbreviation, full_name) VALUES (?, 'AAA', 'A'), (?, 'BBB', 'B')", (A, B))
        # Three games happened; A won all three. Only two have box scores.
        for i, (a_pts, b_pts) in enumerate([(100, 90), (101, 95)]):
            gid = f"00296000{i}"
            for tid, opp, pts, opp_pts in ((A, B, a_pts, b_pts), (B, A, b_pts, a_pts)):
                c.execute(
                    "INSERT INTO team_game_advanced (game_id, team_id, opp_team_id, season, season_type, pts, opp_pts, "
                    "pace, off_rating, def_rating, net_rating, efg_pct, tov_pct, orb_pct, ft_rate, ts_pct) "
                    "VALUES (?, ?, ?, ?, 'Regular Season', ?, ?, 95, 110, 100, 10, 0.5, 0.1, 0.3, 0.2, 0.55)",
                    (gid, tid, opp, SEASON, pts, opp_pts))
        c.execute("CREATE TABLE IF NOT EXISTS game_results (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT, "
                  "game_date TEXT, team_abbr TEXT, team_name TEXT, matchup TEXT, wl TEXT, pts INTEGER)")
        for i in range(3):
            c.execute("INSERT INTO game_results (game_id, team_id, season, season_type, wl) VALUES (?, ?, ?, 'Regular Season', 'W')", (f"g{i}", A, SEASON))
            c.execute("INSERT INTO game_results (game_id, team_id, season, season_type, wl) VALUES (?, ?, ?, 'Regular Season', 'L')", (f"g{i}", B, SEASON))
        c.commit()
        c.close()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _row(self, tid):
        c = get_connection(self.db)
        try:
            return dict(c.execute(
                "SELECT games, wins, losses, win_pct FROM team_season_advanced WHERE team_id = ? AND season = ?", (tid, SEASON)
            ).fetchone())
        finally:
            c.close()

    def test_record_from_game_results(self):
        backfill.compute_and_save_season_stats(SEASON, "Regular Season", self.db)
        self.assertEqual(self._row(A), {"games": 3, "wins": 3, "losses": 0, "win_pct": 1.0})
        self.assertEqual(self._row(B)["losses"], 3)

    def test_without_game_results_counts_box_scores(self):
        c = get_connection(self.db)
        c.execute("DROP TABLE game_results")
        c.commit()
        c.close()
        backfill.compute_and_save_season_stats(SEASON, "Regular Season", self.db)
        self.assertEqual(self._row(A)["wins"], 2)


if __name__ == "__main__":
    unittest.main()
