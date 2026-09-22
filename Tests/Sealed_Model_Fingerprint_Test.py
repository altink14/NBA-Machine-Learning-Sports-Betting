"""
Sealed_Model_Fingerprint_Test.py
================================
The sealed model must keep scoring the same way after any dependency change.

WHY A FINGERPRINT AND NOT A TOLERANCE. The published 67.2% describes
`candidate_2026-08` as it was evaluated on 2026-08-11. That claim survives
only as long as the code underneath it keeps producing the same numbers, and
the things most likely to move it are invisible: an unpinned xgboost resolving
a version higher, a numpy float change, a different wheel on the deploy host.
None of those raise an exception. They shift a probability by a hair and the
site goes on quoting a figure that no longer describes what is running.

So this pins an exact hash of the booster's raw margins on a fixed matrix. It
is deliberately brittle: if it fails, something about how the model scores has
changed, and that is always worth a human looking rather than a tolerance
quietly absorbing it.

WHAT IT CAUGHT. Added 2026-09-22, when `requirements.txt` had no pins at all
and `docker build` was installing xgboost 3.2.0, pandas 3.0.6 and numpy 2.4.6
against a model sealed on 3.0.2 / 2.3.0 / 2.3.0. The same check then proved
`xgboost-cpu` (which drops 469 MB of CUDA libraries the deploy cannot use)
scores identically to the GPU build -- same hash, so the 1.59 GB saving costs
nothing.

IF THIS FAILS, do not update the hash to make it pass. Find out what changed.
"""

import hashlib
import json
import os
import unittest

import numpy as np
import xgboost as xgb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "Models", "candidate_2026-08")

#: Fixed seed, so the matrix is the same on every machine and in every image.
SEED = 20260922
N_ROWS = 64

#: Recorded 2026-09-22 on xgboost 3.0.2, and reproduced exactly by
#: xgboost-cpu 3.0.2 inside the Docker image.
EXPECTED_SHA256 = "9ad990c85787fdc5b4641313c7b0016c921f856df118dfefd15221f4a9aada83"
EXPECTED_SUM = 63.999996185303


class TestSealedModelFingerprint(unittest.TestCase):

    def setUp(self):
        manifest_path = os.path.join(MODEL_DIR, "feature_manifest.json")
        if not os.path.exists(manifest_path):
            self.skipTest("candidate_2026-08 artifact not present")
        with open(manifest_path, encoding="utf-8") as fh:
            self.cols = json.load(fh)["feature_columns"]
        self.booster = xgb.Booster()
        self.booster.load_model(os.path.join(MODEL_DIR, "model.json"))

    def _margins(self):
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(N_ROWS, len(self.cols))).astype(np.float64)
        dm = xgb.DMatrix(X, feature_names=self.cols)
        return np.asarray(self.booster.predict(dm, output_margin=True)).ravel()

    def test_feature_count_matches_the_manifest(self):
        self.assertEqual(len(self.cols), 207)

    def test_raw_margins_are_bit_for_bit_unchanged(self):
        p = self._margins()
        digest = hashlib.sha256(p.astype("<f8").tobytes()).hexdigest()
        self.assertEqual(
            digest, EXPECTED_SHA256,
            "The sealed model scores differently than it did on 2026-09-22. "
            "Something in the numerical stack moved — check xgboost and numpy "
            "versions against requirements.txt. Do NOT update this hash to make "
            "the test pass.")
        self.assertAlmostEqual(float(p.sum()), EXPECTED_SUM, places=5)

    def test_scoring_is_deterministic_within_a_run(self):
        # A fingerprint is only meaningful if repeated scoring agrees with
        # itself; otherwise a failure says nothing about what changed.
        np.testing.assert_array_equal(self._margins(), self._margins())


if __name__ == "__main__":
    unittest.main()
