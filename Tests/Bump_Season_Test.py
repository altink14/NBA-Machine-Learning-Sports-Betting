"""bump_season.py moves all three season constants together, forward only.

Runs on copies of the three files' relevant lines in a temp layout.
"""

import os
import shutil
import tempfile
import unittest

import bump_season as bs
import ingest_player_bios_bulk as bios
import ingest_players as players
from datetime import date


class BumpSeasonTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bump_season_")
        self.be = os.path.join(self.tmp, "backend")
        self.fe = os.path.join(self.tmp, "frontend")
        os.makedirs(self.be)
        os.makedirs(os.path.join(self.fe, "src", "lib"))
        self.write("backend/main_api.py", 'import os\nCURRENT_SEASON = "2025-26"\nOTHER = "2025-26"\n')
        self.write("frontend/src/lib/nba-api.ts", "export const CURRENT_SEASON = '2025-26';\nconst x = 1;\n")
        self.write("frontend/src/lib/archive-seasons.ts",
                   "const ARCHIVE_START_END_YEAR = 1997;\nconst CURRENT_END_YEAR = 2026; // 2025-26\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel, text):
        with open(os.path.join(self.tmp, rel), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)

    def read(self, rel):
        return open(os.path.join(self.tmp, rel), encoding="utf-8").read()

    def run_bump(self, *extra):
        return bs.main(["--backend", self.be, "--frontend", self.fe, *extra])

    def test_dry_run_writes_nothing(self):
        before = self.read("backend/main_api.py")
        self.assertEqual(self.run_bump(), 0)
        self.assertEqual(self.read("backend/main_api.py"), before)

    def test_apply_moves_all_three_and_nothing_else(self):
        self.assertEqual(self.run_bump("--apply"), 0)
        self.assertIn('CURRENT_SEASON = "2026-27"', self.read("backend/main_api.py"))
        self.assertIn('OTHER = "2025-26"', self.read("backend/main_api.py"), "only the constant changes")
        self.assertIn("export const CURRENT_SEASON = '2026-27';", self.read("frontend/src/lib/nba-api.ts"))
        self.assertIn("const CURRENT_END_YEAR = 2027; // 2026-27", self.read("frontend/src/lib/archive-seasons.ts"))
        self.assertIn("ARCHIVE_START_END_YEAR = 1997", self.read("frontend/src/lib/archive-seasons.ts"))

    def test_refuses_to_go_backwards(self):
        self.assertEqual(self.run_bump("--to", "2024-25", "--apply"), 1)
        self.assertIn('CURRENT_SEASON = "2025-26"', self.read("backend/main_api.py"))

    def test_refuses_when_the_constants_disagree(self):
        self.write("frontend/src/lib/nba-api.ts", "export const CURRENT_SEASON = '2024-25';\n")
        self.assertEqual(self.run_bump("--apply"), 1)
        self.assertIn('CURRENT_SEASON = "2025-26"', self.read("backend/main_api.py"))

    def test_refuses_a_file_in_an_unexpected_shape(self):
        self.write("frontend/src/lib/archive-seasons.ts", "const CURRENT_END_YEAR = 2026;\n")  # comment gone
        with self.assertRaises(SystemExit):
            self.run_bump("--apply")

    def test_malformed_target(self):
        self.assertEqual(self.run_bump("--to", "2026-28", "--apply"), 1)


class DerivedSeasonTest(unittest.TestCase):

    def test_ingest_scripts_follow_the_calendar(self):
        self.assertEqual(players._season_start_year(date(2026, 9, 30)), 2025)
        self.assertEqual(players._season_start_year(date(2026, 10, 1)), 2026)
        self.assertEqual(bios._season_start_year(date(2027, 6, 20)), 2026)


if __name__ == "__main__":
    unittest.main()
