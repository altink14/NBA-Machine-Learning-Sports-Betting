"""The withdrawn over/under pick is never emitted, on any path.

Withdrawn 2026-09-19. Until 2026-09-24 the model path of PredictionRunner still
computed it and returned it in /predictions: every signed-in browser got it
through /api/get-nba-odds, the chat got it in its tool payload, the ledger
stored it, and the parlay grader priced totals legs with it. The existing test
covered only the market-only branch, which already returned None. This guards
every place the response dict names the field, so a new path cannot bring it
back without failing here.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NoOverUnderPickTest(unittest.TestCase):

    def test_every_response_sets_the_pick_to_none(self):
        src = open(os.path.join(HERE, "main_api.py"), encoding="utf-8").read()
        found = re.findall(r'"under_over_(prediction|confidence)"\s*:\s*([^,\n]+)', src)
        self.assertTrue(found, "expected the field to be named in the response")
        for field, value in found:
            self.assertEqual(value.strip(), "None", f"under_over_{field} is set to {value.strip()!r}")


if __name__ == "__main__":
    unittest.main()
