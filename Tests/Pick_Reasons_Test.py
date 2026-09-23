"""'Why this pick' says only what the model's own arithmetic says.

No model load and no network: contributions are synthetic, and the ledger
tests run against a temp OddsData file.
"""

import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import main_api
from src.Utils import pick_reasons as pr

COLS = ["W_PCT", "PLUS_MINUS", "W_PCT.1", "PLUS_MINUS.1", "Days-Rest-Home", "Days-Rest-Away",
        "R10_HOME_WIN_PCT", "R10_AWAY_WIN_PCT", "ELO_HOME", "ELO_AWAY", "ELO_DIFF", "REST_HOME_B2B"]
X = [0.683, 7.7, 0.268, -8.4, 2, 1, 0.8, float("nan"), 1640, 1450, 190, 0]
#            home season  away season  rest      form10      elo            rest   bias
DELTA = [0.30, 0.10, 0.25, 0.05, 0.02, -0.30, 0.08, 0.00, 0.20, 0.15, 0.25, -0.01, 0.05]


def reasons():
    return pr.group_contributions(COLS, DELTA, X, 6, "Boston Celtics", "Utah Jazz")


class GroupingTest(unittest.TestCase):

    def test_columns_land_in_the_right_groups(self):
        self.assertEqual(pr.group_of("W_PCT"), "season_home")
        self.assertEqual(pr.group_of("W_PCT.1"), "season_away")
        self.assertEqual(pr.group_of("R10_HOME_FGM"), "form10_home")
        self.assertEqual(pr.group_of("R20_AWAY_WIN_PCT"), "form20_away")
        self.assertEqual(pr.group_of("ELO_DIFF"), "elo")
        self.assertEqual(pr.group_of("Days-Rest-Away"), "rest")
        self.assertEqual(pr.group_of("REST_AWAY_3IN4"), "rest")

    def test_group_pushes_add_up_to_the_score_less_the_baseline(self):
        r = reasons()
        self.assertAlmostEqual(sum(g["push"] for g in r["groups"]), r["score"] - r["baseline"], places=3)
        self.assertAlmostEqual(r["baseline"], 0.05)

    def test_shares_are_of_the_total_absolute_push_and_sum_to_one(self):
        self.assertAlmostEqual(sum(g["share"] for g in reasons()["groups"]), 1.0, places=2)

    def test_direction_follows_the_sign(self):
        by = {g["key"]: g for g in reasons()["groups"]}
        self.assertEqual(by["elo"]["toward"], "Boston Celtics")          # +0.60
        self.assertEqual(by["rest"]["toward"], "Utah Jazz")              # 0.02 - 0.30 - 0.01

    def test_inputs_are_the_real_values_and_missing_ones_are_left_out(self):
        by = {g["key"]: g for g in reasons()["groups"]}
        self.assertEqual(by["season_home"]["inputs"],
                         [{"label": "season win %", "value": "68.3%"}, {"label": "avg margin", "value": "+7.7"}])
        self.assertEqual(by["elo"]["inputs"][0], {"label": "home Elo", "value": "1640"})
        # R10_AWAY_WIN_PCT is NaN (a season opener): shown as nothing, not as 0.
        self.assertEqual(by["form10_away"]["inputs"], [])

    def test_mismatched_lengths_refuse(self):
        with self.assertRaises(AssertionError):
            pr.group_contributions(COLS, DELTA[:-1], X, 6, "A", "B")

    def test_possessives(self):
        self.assertEqual(pr.possessive("Boston Celtics"), "Boston Celtics'")
        self.assertEqual(pr.possessive("Utah Jazz"), "Utah Jazz's")


class WordingTest(unittest.TestCase):

    def why(self, winner="Boston Celtics", conf=71.4):
        return pr.build_why(reasons(), winner, conf)

    def test_summary_names_the_pick_and_its_published_confidence(self):
        s = self.why()["summary"]
        self.assertTrue(s.startswith("The model gives Boston Celtics 71.4% to win."), s)

    def test_every_number_in_the_text_is_the_confidence_or_a_share(self):
        why = self.why()
        allowed = {"71.4"} | {f"{g['share'] * 100:.0f}" for g in why["factors"]}
        for n in re.findall(r"\d+(?:\.\d+)?", why["summary"]):
            self.assertIn(n, allowed, f"{n} in {why['summary']!r} is not an input")

    def test_it_never_oversells(self):
        text = (self.why()["summary"] + " " + self.why()["note"]).lower()
        for word in ("lock", "guarantee", "sure thing", "can't lose", "free money"):
            self.assertNotIn(word, text)
        self.assertIn("afford to lose", text)

    def test_a_factor_pulling_against_the_pick_is_named(self):
        self.assertIn("Pulling the other way", self.why()["summary"])

    def test_small_factors_are_not_given_a_sentence(self):
        s = self.why()["summary"]
        tiny = [g for g in self.why()["factors"] if g["share"] < pr.MIN_SHARE_TO_NAME]
        self.assertTrue(tiny)
        for g in tiny:
            self.assertNotIn(g["label"], s)


class AttachTest(unittest.TestCase):

    def test_a_pick_only_gets_its_own_explanation(self):
        preds = [{"home_team": "Boston Celtics", "away_team": "Utah Jazz",
                  "predicted_winner": "Boston Celtics", "winner_confidence": 71.4},
                 {"home_team": "Miami Heat", "away_team": "Orlando Magic",
                  "predicted_winner": "Miami Heat", "winner_confidence": 55.0}]
        main_api._attach_why(preds, [reasons()])
        self.assertIn("why", preds[0])
        self.assertNotIn("why", preds[1], "a wrong reason is worse than none")


class LedgerTest(unittest.TestCase):
    """The why is recorded with the pick and cannot be edited afterwards."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pick_reasons_test_")
        self.db = os.path.join(self.dir, "OddsData.sqlite")
        self.patch = mock.patch.object(main_api, "ODDS_DB_PATH", self.db)
        self.patch.start()
        tip = (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0).isoformat()
        self.pick = {"home_team": "Boston Celtics", "away_team": "Utah Jazz", "home_odds": -300,
                     "away_odds": 240, "under_over_line": 221.5, "predicted_winner": "Boston Celtics",
                     "winner_confidence": 71.4, "model": "xgboost_cand_2026-08",
                     "expected_value": {"home_team": 1.2, "away_team": -3.1},
                     "game_start_time_utc": tip}
        self.pick["why"] = pr.build_why(reasons(), "Boston Celtics", 71.4)

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_logged_with_the_pick_and_immutable(self):
        counts = main_api.log_predictions({"predictions": [self.pick]}, "fanduel", "NBA")
        self.assertEqual(counts["written"], 1)
        c = sqlite3.connect(self.db)
        try:
            stored = json.loads(c.execute("SELECT why_json FROM predictions_log").fetchone()[0])
            self.assertEqual(stored["summary"], self.pick["why"]["summary"])
            with self.assertRaises(sqlite3.DatabaseError) as ctx:
                c.execute("UPDATE predictions_log SET why_json = '{}'")
            self.assertIn("immutable", str(ctx.exception))
            # Grading still works: only the explanation is frozen.
            c.execute("UPDATE predictions_log SET actual_winner = 'Boston Celtics'")
        finally:
            c.close()

    def test_ledger_mode_serves_the_recorded_why(self):
        main_api.log_predictions({"predictions": [self.pick]}, "fanduel", "NBA")
        out = main_api._predictions_from_ledger("fanduel", True, "NBA")
        self.assertEqual(out["predictions"][0]["why"]["summary"], self.pick["why"]["summary"])

    def test_a_pick_without_a_why_still_logs(self):
        self.pick.pop("why")
        self.assertEqual(main_api.log_predictions({"predictions": [self.pick]}, "fanduel", "NBA")["written"], 1)
        self.assertIsNone(main_api._predictions_from_ledger("fanduel", True, "NBA")["predictions"][0]["why"])


if __name__ == "__main__":
    unittest.main()
