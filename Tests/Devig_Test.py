import unittest

from src.Utils import devig
from src.Utils import Parlay as parlay


# Decimal odds for a standard -110 / -110 market: 1 + 100/110.
D_110 = 1.0 + 100.0 / 110.0


class TestMultiplicative(unittest.TestCase):

    def test_symmetric_pair_is_even(self):
        # -110/-110: both implied probs are 0.5238; normalising the identical
        # pair must give exactly 50/50.
        probs = devig.fair_probs_multiplicative([D_110, D_110])
        self.assertAlmostEqual(probs[0], 0.5, places=10)
        self.assertAlmostEqual(probs[1], 0.5, places=10)

    def test_asymmetric_pair_hand_checked(self):
        # 1.50 / 2.80: implied 0.666667 and 0.357143, overround 1.023810.
        # Normalised: 0.666667/1.023810 = 0.651163 and 0.357143/1.023810 = 0.348837.
        probs = devig.fair_probs_multiplicative([1.5, 2.8])
        self.assertAlmostEqual(probs[0], 0.651163, places=5)
        self.assertAlmostEqual(probs[1], 0.348837, places=5)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)


@unittest.skipUnless(devig.SHIN_AVAILABLE, "shin package not installed")
class TestShin(unittest.TestCase):

    def test_symmetric_pair_is_even(self):
        # A symmetric market has no favourite-longshot asymmetry, so Shin must
        # also give exactly 50/50 on -110/-110.
        probs = devig.fair_probs_shin([D_110, D_110])
        self.assertAlmostEqual(probs[0], 0.5, places=6)
        self.assertAlmostEqual(probs[1], 0.5, places=6)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_asymmetric_pair_differs_from_multiplicative(self):
        # 1.50 / 2.80: Shin puts more of the margin on the longshot, so the
        # favourite's fair probability lands ABOVE the multiplicative estimate.
        # Hand-checked against the shin package: [0.654762, 0.345238].
        shin_probs = devig.fair_probs_shin([1.5, 2.8])
        mult_probs = devig.fair_probs_multiplicative([1.5, 2.8])
        self.assertAlmostEqual(shin_probs[0], 0.654762, places=5)
        self.assertAlmostEqual(shin_probs[1], 0.345238, places=5)
        self.assertGreater(shin_probs[0], mult_probs[0])
        self.assertLess(shin_probs[1], mult_probs[1])
        self.assertAlmostEqual(sum(shin_probs), 1.0, places=6)


class TestDispatchAndValidation(unittest.TestCase):

    def test_fair_probs_uses_active_method(self):
        probs = devig.fair_probs([1.5, 2.8])
        expected = (devig.fair_probs_shin([1.5, 2.8]) if devig.SHIN_AVAILABLE
                    else devig.fair_probs_multiplicative([1.5, 2.8]))
        self.assertAlmostEqual(probs[0], expected[0], places=10)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_invalid_inputs_raise(self):
        for bad in ([], [1.9], [1.0, 2.0], [0.8, 2.0], None):
            with self.assertRaises(ValueError):
                devig.fair_probs(bad)


class TestParlayFairFallback(unittest.TestCase):

    def _leg(self, **overrides):
        base = {
            "home_team": "Boston Celtics", "away_team": "Miami Heat",
            "market": "moneyline", "pick": "Boston Celtics",
            "odds": -150, "model_prob": None,
        }
        base.update(overrides)
        return base

    def test_opp_odds_triggers_devigged_fallback(self):
        # -150/+130 pair: implied on -150 is 0.6; the de-vigged fair prob is
        # lower, and EV at the QUOTED odds becomes negative (as it should for
        # a market-opinion bet: you pay the vig).
        result = parlay.evaluate_parlay([self._leg(opp_odds=130)])
        evaluated = result["legs"][0]
        self.assertEqual(evaluated["prob_source"], "market_fair")
        self.assertLess(evaluated["model_prob"], evaluated["implied_prob"])
        self.assertLess(result["expected_value_per_100"], 0)

    def test_without_opp_odds_keeps_raw_implied(self):
        result = parlay.evaluate_parlay([self._leg()])
        evaluated = result["legs"][0]
        self.assertEqual(evaluated["prob_source"], "market_implied")
        self.assertAlmostEqual(evaluated["model_prob"], evaluated["implied_prob"], places=4)


if __name__ == "__main__":
    unittest.main()
