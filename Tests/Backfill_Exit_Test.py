"""backfill.py says what happened to every game, and exits non-zero when one failed.

No network and no production data: the league game log and process_game are
mocked, and every database is a throwaway file in a temp directory.

Run from the repo root:
    venv/Scripts/python.exe -m unittest Tests.Backfill_Exit_Test
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "backfill_under_test", os.path.join(REPO, "src", "Process-Data", "backfill.py"))
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)

from src.Utils.nba_db_schema import ensure_schema, get_connection  # noqa: E402

HOME, AWAY = 1610612738, 1610612754


def log_rows(game_id, home_pts, away_pts, date_="2024-04-16", wl=True):
    """The two team rows the league game log returns for one game."""
    hw = ("W" if home_pts > away_pts else "L") if wl else None
    aw = ("L" if home_pts > away_pts else "W") if wl else None
    return [
        {"GAME_ID": game_id, "TEAM_ID": HOME, "GAME_DATE": date_, "WL": hw, "PTS": home_pts},
        {"GAME_ID": game_id, "TEAM_ID": AWAY, "GAME_DATE": date_, "WL": aw, "PTS": away_pts},
    ]


class _Harness(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_backfill_test_")
        self.db = os.path.join(self.tmp, "TeamData.sqlite")
        ensure_schema(self.db)
        #: what the fake process_game stores per game: (home_pts, away_pts, date)
        #: or an Exception to raise
        self.box = {}
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_process_game(self, game_id, season, season_type, db_path, overwrite=False,
                          game_date_hint=None):
        self.calls.append(game_id)
        conn = get_connection(db_path)
        try:
            n = conn.execute("SELECT COUNT(*) FROM team_game_advanced WHERE game_id=?",
                             (game_id,)).fetchone()[0]
        finally:
            conn.close()
        if n >= 2 and not overwrite:
            return {"game_id": game_id, "status": "cached"}
        spec = self.box[game_id]
        if isinstance(spec, Exception):
            raise spec
        home_pts, away_pts, *rest = spec
        stored_date = rest[0] if rest else str(game_date_hint).split("T")[0]
        self.store(game_id, home_pts, away_pts, stored_date, season, season_type)
        return {"game_id": game_id, "status": "processed"}

    def store(self, game_id, home_pts, away_pts, game_date, season="2023-24",
              season_type="PlayIn"):
        conn = get_connection(self.db)
        try:
            for tid, opp, pts, opp_pts in ((HOME, AWAY, home_pts, away_pts),
                                           (AWAY, HOME, away_pts, home_pts)):
                conn.execute(
                    "INSERT INTO team_game_advanced (game_id, team_id, opp_team_id, season, "
                    "season_type, game_date, pts, opp_pts, computed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (game_id, tid, opp, season, season_type, game_date, pts, opp_pts, "t"))
            conn.commit()
        finally:
            conn.close()

    def run_backfill(self, rows, season_type="PlayIn", **kw):
        client = mock.Mock()
        client.league_game_log.return_value = rows
        with mock.patch.object(backfill, "get_client", return_value=client), \
             mock.patch.object(backfill, "process_game", side_effect=self.fake_process_game):
            return backfill.backfill_games("2023-24", season_type, self.db, **kw)


class BackfillSummaryTest(_Harness):

    def test_clean_run_exits_zero(self):
        self.box = {"0052300101": (110, 104), "0052300111": (99, 101)}
        s = self.run_backfill(log_rows("0052300101", 110, 104) + log_rows("0052300111", 99, 101))
        self.assertEqual(sorted(s.ingested), ["0052300101", "0052300111"])
        self.assertEqual(s.failed, {})
        self.assertEqual(backfill.exit_code_for(s, expect_games=True), backfill.EXIT_OK)

    def test_a_failed_game_fails_the_run(self):
        self.box = {"0052300101": (110, 104), "0052300111": RuntimeError("boxscore timed out")}
        s = self.run_backfill(log_rows("0052300101", 110, 104) + log_rows("0052300111", 99, 101))
        self.assertEqual(s.ingested, ["0052300101"])
        self.assertIn("0052300111", s.failed)
        self.assertIn("boxscore timed out", s.failed["0052300111"])
        self.assertEqual(backfill.exit_code_for(s, expect_games=False), backfill.EXIT_GAMES_FAILED)
        self.assertTrue(any("FAILED     0052300111" in ln for ln in s.lines()))

    def test_second_run_counts_already_present(self):
        self.box = {"0052300101": (110, 104)}
        self.run_backfill(log_rows("0052300101", 110, 104))
        s = self.run_backfill(log_rows("0052300101", 110, 104))
        self.assertEqual(s.already_present, ["0052300101"])
        self.assertEqual(s.ingested, [])
        self.assertEqual(backfill.exit_code_for(s, False), backfill.EXIT_OK)

    def test_known_holes_are_reported_not_requested_and_do_not_fail(self):
        for gid in ("0029600332", "0029600370", "0029800661", "0020300778"):
            with self.subTest(gid=gid):
                self.calls = []
                s = self.run_backfill(log_rows(gid, 123, 83), season_type="Regular Season")
                self.assertEqual(s.known_holes, [gid])
                self.assertEqual(self.calls, [], "a known hole must not be re-requested")
                self.assertEqual(s.failed, {})
                self.assertEqual(backfill.exit_code_for(s, True), backfill.EXIT_OK)
                self.assertTrue(any(f"known hole {gid}" in ln for ln in s.lines()))

    def test_the_cancelled_2013_game_is_a_known_hole(self):
        # In the 2012-13 log with WL None and 0 points: it was never played.
        s = self.run_backfill(log_rows("0021201214", 0, 0, wl=False), season_type="Regular Season")
        self.assertEqual(s.known_holes, ["0021201214"])
        self.assertEqual(backfill.exit_code_for(s, True), backfill.EXIT_OK)

    def test_retrying_a_known_hole_that_still_fails_is_still_known(self):
        self.box = {"0029600332": AttributeError("'NoneType' object has no attribute 'get'")}
        s = self.run_backfill(log_rows("0029600332", 123, 83), season_type="Regular Season",
                              retry_known_holes=True)
        self.assertEqual(self.calls, ["0029600332"])
        self.assertEqual(s.known_holes, ["0029600332"])
        self.assertEqual(s.failed, {})

    def test_a_known_hole_that_has_since_landed_counts_as_present(self):
        self.store("0029600332", 123, 83, "1996-12-17", "1996-97", "Regular Season")
        s = self.run_backfill(log_rows("0029600332", 123, 83, "1996-12-17"),
                              season_type="Regular Season")
        self.assertEqual(s.already_present, ["0029600332"])
        self.assertEqual(s.known_holes, [])

    def test_a_game_without_a_result_in_the_log_is_not_ingested(self):
        s = self.run_backfill(log_rows("0052300101", 0, 0, wl=False))
        self.assertEqual(self.calls, [])
        self.assertEqual(s.not_final, ["0052300101"])
        self.assertEqual(backfill.exit_code_for(s, False), backfill.EXIT_OK)

    def test_a_zero_zero_box_score_is_a_failure_not_a_result(self):
        self.box = {"0052300101": (0, 0)}
        s = self.run_backfill(log_rows("0052300101", 110, 104))
        self.assertIn("0052300101", s.failed)
        self.assertEqual(s.ingested, [])
        self.assertEqual(backfill.exit_code_for(s, False), backfill.EXIT_GAMES_FAILED)

    def test_a_score_that_disagrees_with_the_log_is_a_failure(self):
        self.box = {"0052300101": (110, 105)}
        s = self.run_backfill(log_rows("0052300101", 110, 104))
        self.assertIn("league game log", s.failed["0052300101"])

    def test_a_newly_written_wrong_date_is_a_failure(self):
        self.box = {"0052300101": (110, 104, "2024-04-01")}
        s = self.run_backfill(log_rows("0052300101", 110, 104, "2024-04-16"))
        self.assertIn("game_date", s.failed["0052300101"])

    def test_an_old_wrong_date_is_a_warning_not_a_failure(self):
        self.store("0052300101", 110, 104, "2024-04-01")
        s = self.run_backfill(log_rows("0052300101", 110, 104, "2024-04-16"))
        self.assertEqual(s.date_disagreements, {"0052300101": ("2024-04-01", "2024-04-16")})
        self.assertEqual(s.failed, {})
        self.assertEqual(backfill.exit_code_for(s, False), backfill.EXIT_OK)
        self.assertTrue(any(ln.lstrip().startswith("WARNING") for ln in s.lines()))

    def test_an_empty_log_fails_only_when_games_are_expected(self):
        s = self.run_backfill([])
        self.assertEqual(s.in_game_log, 0)
        self.assertEqual(backfill.exit_code_for(s, expect_games=False), backfill.EXIT_OK)
        self.assertEqual(backfill.exit_code_for(s, expect_games=True), backfill.EXIT_EMPTY_GAME_LOG)


class BackfillMainTest(_Harness):
    """main() turns the summary into the process exit code."""

    def _main(self, argv, summary):
        with mock.patch.object(sys, "argv", ["backfill.py", "--db", self.db] + argv), \
             mock.patch.object(backfill, "backfill_metadata"), \
             mock.patch.object(backfill, "backfill_games", return_value=summary) as bg, \
             mock.patch.object(backfill, "compute_and_save_season_stats") as season_stats, \
             mock.patch.object(backfill, "backfill_players") as players, \
             mock.patch.object(backfill, "compute_and_save_player_season_aggregates") as aggs:
            code = backfill.main()
        return code, bg, season_stats, players, aggs

    def test_failed_games_exit_three(self):
        s = backfill.BackfillSummary("2023-24", "Regular Season", in_game_log=2,
                                     ingested=["a"], failed={"b": "boom"})
        code, *_ = self._main(["--season", "2023-24"], s)
        self.assertEqual(code, backfill.EXIT_GAMES_FAILED)

    def test_clean_exit_zero(self):
        s = backfill.BackfillSummary("2023-24", "Regular Season", in_game_log=1, ingested=["a"])
        code, *_ = self._main(["--season", "2023-24"], s)
        self.assertEqual(code, backfill.EXIT_OK)

    def test_play_in_skips_season_aggregates(self):
        s = backfill.BackfillSummary("2023-24", "PlayIn", in_game_log=6, ingested=list("abcdef"))
        code, bg, season_stats, players, aggs = self._main(
            ["--season", "2023-24", "--season-type", "PlayIn"], s)
        self.assertEqual(code, backfill.EXIT_OK)
        self.assertEqual(bg.call_args[0][1], "PlayIn")
        season_stats.assert_not_called()
        players.assert_not_called()
        aggs.assert_not_called()

    def test_playoffs_still_compute_season_aggregates(self):
        s = backfill.BackfillSummary("2023-24", "Playoffs", in_game_log=1, ingested=["a"])
        _, _, season_stats, players, aggs = self._main(
            ["--season", "2023-24", "--season-type", "Playoffs"], s)
        season_stats.assert_called_once()
        players.assert_called_once()
        aggs.assert_called_once()

    def test_unknown_season_type_is_refused(self):
        with mock.patch.object(sys, "argv", ["backfill.py", "--season-type", "Play-In"]):
            with self.assertRaises(SystemExit) as cm:
                backfill.main()
        self.assertEqual(cm.exception.code, 2)


import daily_update  # noqa: E402  (after sys.path is set)


class _FixedDate(date):
    fixed = date(2027, 4, 20)

    @classmethod
    def today(cls):
        return cls.fixed


class DailyUpdateBackfillTest(unittest.TestCase):

    def test_non_zero_exit_is_a_failed_backfill(self):
        for code in (1, 3, 4):
            with self.subTest(code=code), \
                 mock.patch.object(daily_update.subprocess, "run",
                                   return_value=subprocess.CompletedProcess([], code)):
                self.assertFalse(daily_update.run_backfill("2025-26"))
        with mock.patch.object(daily_update.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0)):
            self.assertTrue(daily_update.run_backfill("2025-26"))

    def test_expect_games_is_passed_through(self):
        with mock.patch.object(daily_update.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0)) as run:
            daily_update.run_backfill("2026-27", expect_games=True)
            self.assertIn("--expect-games", run.call_args[0][0])
            daily_update.run_backfill("2026-27", "PlayIn")
            self.assertNotIn("--expect-games", run.call_args[0][0])

    def _main_with(self, today, backfill_results):
        """Run daily_update.main() with every step stubbed; return (code, backfill calls)."""
        calls = []

        def fake_backfill(season, season_type="Regular Season", expect_games=False):
            calls.append((season, season_type, expect_games))
            return backfill_results.get(season_type, True)

        _FixedDate.fixed = today
        with mock.patch.object(daily_update, "date", _FixedDate), \
             mock.patch.object(daily_update, "run_backfill", side_effect=fake_backfill), \
             mock.patch.object(daily_update, "refresh_team_stats_snapshot", return_value=True), \
             mock.patch.object(daily_update, "grade_logged_predictions", return_value=True), \
             mock.patch.object(daily_update, "log_todays_predictions", return_value="logged"), \
             mock.patch.object(daily_update, "snapshot_odds_board", return_value="ok"), \
             mock.patch.object(daily_update, "refresh_periodic_ingests", return_value=True), \
             mock.patch.object(daily_update, "publish_ledger", return_value="skipped"), \
             mock.patch.object(daily_update, "run_preflight", return_value=0), \
             mock.patch.object(daily_update.logger, "error") as err:
            code = daily_update.main()
        return code, calls, err

    def test_a_failed_backfill_turns_the_run_red_and_says_backfill(self):
        code, _, err = self._main_with(date(2027, 1, 15), {"Regular Season": False})
        self.assertEqual(code, 1)
        self.assertIn("backfill", err.call_args[0][1])

    def test_a_failed_play_in_backfill_turns_the_run_red(self):
        code, _, err = self._main_with(date(2027, 4, 20), {"PlayIn": False})
        self.assertEqual(code, 1)
        self.assertIn("backfill", err.call_args[0][1])

    def test_which_season_types_run_when(self):
        cases = {
            date(2027, 1, 15): ["Regular Season"],
            date(2027, 4, 20): ["Regular Season", "PlayIn", "Playoffs"],
            date(2027, 5, 20): ["Regular Season", "PlayIn", "Playoffs"],
            date(2027, 6, 10): ["Regular Season", "Playoffs"],
            date(2027, 8, 1): ["Regular Season"],
        }
        for today, expected in cases.items():
            with self.subTest(today=today):
                code, calls, _ = self._main_with(today, {})
                self.assertEqual(code, 0)
                self.assertEqual([c[1] for c in calls], expected)
                self.assertEqual(calls[0][0], daily_update.current_season(today))
                # Only the regular season ever demands a non-empty log.
                self.assertTrue(all(not c[2] for c in calls[1:]))

    def test_regular_season_games_expected(self):
        from preflight_opening_night import OPENING_NIGHT
        f = daily_update._regular_season_games_expected
        self.assertFalse(f(date(OPENING_NIGHT.year, 10, 1)))
        self.assertFalse(f(OPENING_NIGHT))
        self.assertTrue(f(date.fromordinal(OPENING_NIGHT.toordinal() + 2)))
        self.assertTrue(f(date(OPENING_NIGHT.year, 11, 1)))
        self.assertTrue(f(date(OPENING_NIGHT.year + 1, 8, 15)))   # finished season's log
        # A stale constant from last year must not turn October red.
        self.assertFalse(f(date(OPENING_NIGHT.year + 1, 10, 5)))
        self.assertTrue(f(date(OPENING_NIGHT.year + 1, 10, 28)))


if __name__ == "__main__":
    unittest.main()
