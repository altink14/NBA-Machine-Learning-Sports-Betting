"""The Model Diary says what each gate measured, records the over/under
withdrawal, and speaks plain English; Comeback Odds counts what it leaves out.

Found in the 2026-09-24 nav audit: the diary showed a failed gate as an X and
raw JSON (never "+1.69pp against a +2.5pp target"), never recorded the
2026-09-19 over/under withdrawal, and printed the retired model's caveats with
file and function names in them. Comeback Odds said "every game" while the
play-in games it holds no play-by-play for were missing from its counts.

Reads the sealed artifacts read-only; the comeback count uses an in-memory DB.
"""
import sqlite3
import unittest

from fastapi.testclient import TestClient

import main_api


class ModelDiaryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        r = TestClient(main_api.app).get("/api/model/diary")
        assert r.status_code == 200, r.text
        cls.d = r.json()
        cls.serving = next(e for e in cls.d["entries"] if e["status"] == "serving")
        cls.gates = {g["id"]: g for g in cls.serving["gates"]}

    def test_every_gate_states_its_figure_against_its_bar(self):
        for g in self.serving["gates"]:
            self.assertTrue(g.get("result"), f"{g['id']} has no sentence")

    def test_failed_early_season_gate_shows_the_miss(self):
        g = self.gates["t4_oct_dec"]
        self.assertFalse(g["passed"])
        self.assertIn("+1.69 points", g["result"])
        self.assertIn("target was +2.5", g["result"])
        self.assertIn("p = 0.195", g["result"])

    def test_failed_calibration_gate_names_the_bucket(self):
        g = self.gates["t3_calibration"]
        self.assertFalse(g["passed"])
        for part in ("40–50%", "-5.15 points", "216 games", "bound was ±5"):
            self.assertIn(part, g["result"])

    def test_passed_gates_carry_their_p_values(self):
        self.assertIn("p = 0.041", self.gates["t1_beats_old_model"]["result"])
        self.assertIn("p = 0.039", self.gates["t2_beats_better_record"]["result"])

    def test_the_over_under_withdrawal_is_recorded(self):
        ou = [e for e in self.d["entries"] if e.get("version") == "over/under pick"]
        self.assertEqual(len(ou), 1)
        e = ou[0]
        self.assertEqual(e["status"], "withdrawn")
        self.assertEqual(e["retired_on"], "2026-09-19")
        self.assertIn("XGBoost_54.8%_UO-8.json", e["why_wrong"])
        self.assertIn("never had a sealed evaluation", e["why_wrong"])
        self.assertIn("market's total is still", e["why_wrong"])
        # Newest event first: straight after the serving model.
        self.assertEqual(self.d["entries"][1]["version"], "over/under pick")

    def test_retired_caveats_have_a_plain_version_without_code_names(self):
        old = next(e for e in self.d["entries"] if e["status"] == "retired")
        self.assertEqual(len(old["caveats_plain"]), len(old["caveats"]))
        joined = " ".join(old["caveats_plain"])
        for jargon in ("main_api.py", "days_rest_sensitivity", "OddsData.sqlite",
                       "validation block", "McNemar"):
            self.assertNotIn(jargon, joined)
        # Figures survive the translation.
        self.assertIn("65.34%", joined)
        self.assertIn("2024-04-28", joined)

    def test_the_sealed_note_is_still_served(self):
        self.assertTrue(self.d["sealed_note"].startswith("The test set is spent."))


class BacktestBetterRecordPTest(unittest.TestCase):

    def test_p_value_comes_from_the_artifact(self):
        main_api._backtest_summary_cache.clear()
        r = TestClient(main_api.app).get("/api/model/backtest")
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(r.json()["baselines"]["better_record_p_value"], 0.039364)


class ComebackMissingPbpTest(unittest.TestCase):

    def setUp(self):
        self.c = sqlite3.connect(":memory:")
        self.c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT)")
        self.c.execute("CREATE TABLE pbp_events (game_id TEXT, action_id INTEGER)")
        self.c.executemany("INSERT INTO box_scores VALUES (?,?,?)", [
            ("a", "2019-20", "Regular Season"), ("b", "2019-20", "PlayIn"),
            ("c", "2020-21", "PlayIn"), ("d", "2020-21", "Playoffs"),
            ("e", "2018-19", "Regular Season"),   # before the grid's seasons
        ])
        self.c.executemany("INSERT INTO pbp_events VALUES (?,?)", [("a", 1), ("a", 2), ("d", 1)])

    def test_counts_box_scores_with_no_play_by_play(self):
        m = main_api._comeback_missing_pbp(self.c, ["2019-20", "2020-21"])
        self.assertEqual(m["games"], 2)
        self.assertEqual(m["by_season_type"], {"PlayIn": 2})
        self.assertEqual(m["seasons"], ["2019-20", "2020-21"])

    def test_no_seasons_means_nothing_missing(self):
        self.assertEqual(main_api._comeback_missing_pbp(self.c, [])["games"], 0)


if __name__ == "__main__":
    unittest.main()
