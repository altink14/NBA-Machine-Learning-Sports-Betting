import unittest
from src.Utils import Parlay as parlay


def leg(home, away, market="moneyline", pick=None, odds=-110, model_prob=None):
    return {
        "home_team": home,
        "away_team": away,
        "market": market,
        "pick": pick or home,
        "odds": odds,
        "model_prob": model_prob,
    }


class TestOddsConversion(unittest.TestCase):

    def test_negative_american_to_decimal(self):
        self.assertAlmostEqual(parlay.american_to_true_decimal(-110), 1.9091, places=4)

    def test_positive_american_to_decimal(self):
        self.assertAlmostEqual(parlay.american_to_true_decimal(150), 2.5, places=4)

    def test_implied_probability(self):
        self.assertAlmostEqual(parlay.implied_probability(-110), 0.5238, places=4)
        self.assertAlmostEqual(parlay.implied_probability(100), 0.5, places=4)


class TestEvaluateParlay(unittest.TestCase):

    def test_two_leg_independent_parlay(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=-150, model_prob=0.65),
            leg("Denver Nuggets", "Utah Jazz", pick="Denver Nuggets", odds=-120, model_prob=0.60),
        ])
        # decimal: (1+100/150) * (1+100/120) = 1.6667 * 1.8333 = 3.0556
        self.assertAlmostEqual(result["combined_decimal_odds"], 3.0556, places=3)
        self.assertAlmostEqual(result["combined_model_prob"], 0.39, places=4)
        self.assertTrue(result["independent_legs"])
        self.assertEqual(result["warnings"], [])
        # EV per 100: 0.39 * 205.56 - 0.61 * 100 = 80.17 - 61 = ~19.17
        self.assertAlmostEqual(result["expected_value_per_100"], 19.17, delta=0.2)
        self.assertEqual(result["verdict"], "POSITIVE_EV")

    def test_negative_ev_parlay(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=-110, model_prob=0.50),
            leg("Denver Nuggets", "Utah Jazz", pick="Denver Nuggets", odds=-110, model_prob=0.50),
        ])
        self.assertEqual(result["verdict"], "NEGATIVE_EV")
        self.assertLess(result["expected_value_per_100"], 0)
        self.assertEqual(result["kelly_pct_of_bankroll"], 0)

    def test_same_game_legs_flagged(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", market="moneyline", pick="Boston Celtics",
                odds=-150, model_prob=0.65),
            leg("Boston Celtics", "Miami Heat", market="over_under", pick="under",
                odds=-110, model_prob=0.55),
        ])
        self.assertFalse(result["independent_legs"])
        types = [w["type"] for w in result["warnings"]]
        self.assertIn("SAME_GAME", types)
        self.assertTrue(result["verdict"].startswith("UNRELIABLE"))

    def test_repeated_team_across_games_flagged(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=-150, model_prob=0.65),
            leg("Orlando Magic", "Boston Celtics", pick="Boston Celtics", odds=-130, model_prob=0.60),
        ])
        self.assertTrue(result["independent_legs"])  # different games, so product still shown
        types = [w["type"] for w in result["warnings"]]
        self.assertIn("REPEATED_TEAM", types)

    def test_market_implied_fallback(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=-110),
        ])
        self.assertEqual(result["legs"][0]["prob_source"], "market_implied")
        # Implied prob at the book's own line is EV-negative by exactly zero minus nothing
        # (single leg at implied prob nets EV 0 by construction).
        self.assertAlmostEqual(result["expected_value_per_100"], 0.0, delta=0.01)

    def test_break_even_prob(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=100, model_prob=0.55),
        ])
        self.assertAlmostEqual(result["break_even_prob"], 0.5, places=4)
        self.assertAlmostEqual(result["edge_pct"], 5.0, places=1)

    def test_rejects_empty_parlay(self):
        with self.assertRaises(ValueError):
            parlay.evaluate_parlay([])

    def test_rejects_invalid_american_odds(self):
        with self.assertRaises(ValueError):
            parlay.evaluate_parlay([
                leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=50, model_prob=0.6),
            ])

    def test_rejects_invalid_model_prob(self):
        with self.assertRaises(ValueError):
            parlay.evaluate_parlay([
                leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=-110, model_prob=1.5),
            ])

    def test_kelly_capped(self):
        result = parlay.evaluate_parlay([
            leg("Boston Celtics", "Miami Heat", pick="Boston Celtics", odds=300, model_prob=0.9),
        ])
        self.assertLessEqual(result["kelly_pct_of_bankroll"], 10.0)


if __name__ == "__main__":
    unittest.main()
