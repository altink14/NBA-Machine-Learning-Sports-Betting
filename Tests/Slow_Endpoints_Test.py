"""Speed fixes from 2026-09-23 that must not change what the endpoints say.

- career-official freshness is keyed on the archive (has he played since the
  fetch?) rather than on the calendar alone, and the milestone page no longer
  waits on stats.nba.com for players whose cached career is merely stale;
- the win-probability lookup is memoised per table;
- the 2K DNA card reads rebounding tracking through _rebounding_for (it used
  to call the rate-limited route with the season in the `request` slot, which
  always failed and was swallowed into rebounding: null).

All on temp databases or mocks; nothing touches production.
"""
import random
import sqlite3
import unittest
from datetime import datetime, timedelta
from unittest import mock

import main_api


class _KeepOpen:
    """The handlers close their connection; the test still needs it."""

    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _db():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE player_game_log (player_id INTEGER, game_date TEXT)")
    main_api._ensure_career_official_table(c)
    return c


def _career_row(c, pid, fetched_at, season="2024-25"):
    c.execute("INSERT INTO player_career_official (player_id, season, season_type, team_abbr, "
              "is_career_total, fetched_at, pts) VALUES (?, ?, 'Regular Season', 'TOT', 0, ?, 100)",
              (pid, season, fetched_at))


class TestCareerOfficialFreshness(unittest.TestCase):

    def setUp(self):
        self.c = _db()
        self.addCleanup(self.c.close)
        self.old = (datetime.utcnow() - timedelta(days=30)).isoformat()

    def test_never_fetched_is_missing(self):
        self.assertEqual(main_api._career_official_freshness(self.c, 1), "missing")

    def test_inside_the_week_is_fresh(self):
        _career_row(self.c, 1, datetime.utcnow().isoformat())
        self.c.execute("INSERT INTO player_game_log VALUES (1, ?)", (datetime.utcnow().date().isoformat(),))
        self.assertEqual(main_api._career_official_freshness(self.c, 1), "fresh")

    def test_old_fetch_with_no_game_since_is_fresh(self):
        _career_row(self.c, 1, self.old)
        before = (datetime.utcnow() - timedelta(days=40)).date().isoformat()
        self.c.execute("INSERT INTO player_game_log VALUES (1, ?)", (before,))
        self.assertEqual(main_api._career_official_freshness(self.c, 1), "fresh")

    def test_old_fetch_with_a_game_since_is_stale(self):
        _career_row(self.c, 1, self.old)
        since = (datetime.utcnow() - timedelta(days=2)).date().isoformat()
        self.c.execute("INSERT INTO player_game_log VALUES (1, ?)", (since,))
        self.assertEqual(main_api._career_official_freshness(self.c, 1), "stale")

    def test_game_on_the_fetch_day_counts_as_played_since(self):
        _career_row(self.c, 1, self.old)
        self.c.execute("INSERT INTO player_game_log VALUES (1, ?)", (self.old[:10],))
        self.assertEqual(main_api._career_official_freshness(self.c, 1), "stale")

    def test_old_empty_marker_is_retried(self):
        _career_row(self.c, 1, self.old, season="__EMPTY__")
        self.assertEqual(main_api._career_official_freshness(self.c, 1), "missing")

    def test_ensure_does_not_fetch_when_fresh(self):
        _career_row(self.c, 1, self.old)
        with mock.patch("nba_api.stats.endpoints.playercareerstats.PlayerCareerStats",
                        side_effect=AssertionError("must not fetch")):
            main_api._ensure_career_official(self.c, 1)


class TestMilestonesDoNotWaitOnStaleCareers(unittest.TestCase):

    def test_stale_players_go_to_the_background_and_are_still_shown(self):
        c = _db()
        self.addCleanup(c.close)
        c.execute("CREATE TABLE players (player_id INTEGER, full_name TEXT)")
        c.execute("CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, "
                  "season_type TEXT, gp INTEGER, pts INTEGER)")
        c.execute("INSERT INTO players VALUES (7, 'Near Milestone')")
        c.execute("INSERT INTO player_season_totals VALUES (7, ?, 'Regular Season', 70, 2000)",
                  (main_api.CURRENT_SEASON,))
        old = (datetime.utcnow() - timedelta(days=30)).isoformat()
        c.execute("INSERT INTO player_career_official (player_id, season, season_type, team_abbr, "
                  "is_career_total, fetched_at, pts, ast, reb, fg3m, stl, blk) VALUES "
                  "(7, 'CAREER', 'Regular Season', 'TOT', 1, ?, 19900, 10, 10, 10, 10, 10)", (old,))
        c.execute("INSERT INTO player_game_log VALUES (7, ?)", (datetime.utcnow().date().isoformat(),))
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)), \
             mock.patch.object(main_api, "_ensure_career_official",
                               side_effect=AssertionError("must not fetch in the request")), \
             mock.patch.object(main_api, "_refresh_career_official_async") as bg:
            out = main_api.get_milestone_watch(limit=25)
        bg.assert_called_once_with([7])
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["milestones"][0]["next_milestone"], 20000)


class TestWinProbabilityMemo(unittest.TestCase):

    def test_memo_matches_the_pooled_computation_and_resets_per_table(self):
        rnd = random.Random(7)
        table = {(t, m): [rnd.randint(0, 40), 0] for t in range(0, 2881, 30) for m in range(-30, 31)}
        for cell in table.values():
            cell[1] = rnd.randint(0, cell[0])
        for _ in range(500):
            secs = rnd.uniform(-100, 3200)
            margin = rnd.randint(-45, 45)
            sl = max(0, min(main_api.WP_REGULATION,
                            int(round(secs / main_api.WP_STEP) * main_api.WP_STEP)))
            m = max(-main_api.WP_MARGIN_CAP, min(main_api.WP_MARGIN_CAP, margin))
            self.assertEqual(main_api._wp_lookup(table, secs, margin),
                             main_api._wp_pooled(table, sl, m))
        other = {k: [v[0], v[0]] for k, v in table.items()}   # every home team won
        p, _ = main_api._wp_lookup(other, 600, 3)
        self.assertEqual(p, 1.0)


class TestBuildDnaReadsRebounding(unittest.TestCase):

    def test_dna_uses_the_rebounding_payload(self):
        c = sqlite3.connect(":memory:")
        self.addCleanup(c.close)
        c.row_factory = sqlite3.Row
        c.execute("CREATE TABLE players (player_id INTEGER, full_name TEXT, height TEXT, weight TEXT, "
                  "position TEXT, from_year INTEGER, to_year INTEGER)")
        c.execute("INSERT INTO players VALUES (9, 'Glass Cleaner', '7-0', '250', 'C', 2020, 2026)")
        c.execute("CREATE TABLE player_season_totals (player_id INTEGER, season TEXT, season_type TEXT, "
                  "gp INTEGER, min REAL, pts INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, "
                  "tov INTEGER, fgm INTEGER, fga INTEGER, fg_pct REAL, fg3m INTEGER, fg3a INTEGER, "
                  "fg3_pct REAL, ftm INTEGER, fta INTEGER, ft_pct REAL, oreb INTEGER, dreb INTEGER, pf INTEGER)")
        c.execute("INSERT INTO player_season_totals VALUES (9, '2024-25', 'Regular Season', 70, 2100, "
                  "1000, 900, 100, 50, 150, 80, 400, 700, 0.57, 5, 20, 0.25, 195, 300, 0.65, 300, 600, 180)")
        reb = {"players": [{"player_id": 9, "reb": {"total": 900}, "oreb": {"total": 300}}]}
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)), \
             mock.patch.object(main_api, "get_shot_quality", return_value={"players": []}), \
             mock.patch.object(main_api, "_rebounding_for", return_value=reb) as rf, \
             mock.patch.object(main_api, "_clutch_for", return_value={"players": []}):
            out = main_api.get_build_dna(9, season="2024-25")
        rf.assert_called_once_with("2024-25", "Regular Season")
        self.assertEqual(out["rebounding"], {"reb": {"total": 900}, "oreb": {"total": 300}})


if __name__ == "__main__":
    unittest.main()
