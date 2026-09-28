"""The officials pages correct for running hundreds of tests at once (nav audit bug 19).

The default /stats/officials view checks 128 officials on five measures: 618
separate 95% tests, so ~31 would clear a single test by luck. Every flag the
page shows must now survive a Benjamini-Hochberg false-discovery-rate
correction across that whole family. Also covered: coverage is stated per
season from the data (the page once said nba.com's feed "goes spotty after
Apr 2025" long after the v3 fallback filled those games in).

No network and no real database: a synthetic archive in memory.
"""

import math
import random
import sqlite3
import unittest

from src.Utils import Officials as O


def bh_reference(pvals, alpha):
    """Textbook BH: reject the k smallest, k = max{i : p(i) <= i/m * alpha}."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    k = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / m * alpha:
            k = rank
    return {order[j] for j in range(k)}


class BHAdjustTest(unittest.TestCase):

    def test_known_values(self):
        q = O._bh_adjust([0.01, 0.04, 0.03, 0.005])
        for got, want in zip(q, [0.02, 0.04, 0.04, 0.02]):
            self.assertAlmostEqual(got, want)

    def test_matches_the_textbook_procedure(self):
        rng = random.Random(7)
        for _ in range(200):
            m = rng.randint(1, 60)
            p = [rng.random() ** rng.choice([1, 3, 6]) for _ in range(m)]
            q = O._bh_adjust(p)
            self.assertEqual({i for i, v in enumerate(q) if v <= 0.05}, bh_reference(p, 0.05))
            self.assertTrue(all(0 <= v <= 1 for v in q))
            self.assertTrue(all(q[i] >= p[i] - 1e-12 for i in range(m)))


class PValueAgreesWithIntervalTest(unittest.TestCase):
    """The page used to flag on the interval; p < .05 must mean the same thing."""

    def test_mean_test_matches_mean_ci(self):
        rng = random.Random(3)
        for _ in range(300):
            vals = [rng.gauss(40, 6) for _ in range(rng.randint(5, 400))]
            base = 40 + rng.uniform(-2, 2)
            ci = O._mean_ci(vals)
            p = O._p_mean(vals, base)
            outside = ci[0] > base or ci[1] < base
            # Skip knife-edge cases where the 2-decimal rounding of the CI decides.
            if min(abs(ci[0] - base), abs(ci[1] - base)) < 0.01:
                continue
            self.assertEqual(p < 0.05, outside)

    def test_proportion_test_matches_wilson(self):
        for n in (25, 80, 400):
            for k in range(0, n + 1, max(1, n // 20)):
                for p0 in (0.3, 0.5, 0.58):
                    lo, hi = O._wilson(k, n)
                    p = O._p_prop(k, n, p0)
                    if min(abs(lo - p0 * 100), abs(hi - p0 * 100)) < 0.2:
                        continue
                    self.assertEqual(p < 0.05, lo > p0 * 100 or hi < p0 * 100, (k, n, p0))

    def test_unknown_baseline_gives_no_test(self):
        self.assertIsNone(O._p_mean([1.0, 2.0, 3.0], None))
        self.assertIsNone(O._p_prop(5, 10, None))
        self.assertIsNone(O._p_prop(0, 0, 0.5))


def build_archive(n_officials=40, games_per_season=300, seasons=("2023-24", "2024-25"),
                  loud_official=7, crewless=()):
    """Three officials per game drawn at random; one official's games carry
    8 more fouls. `crewless` game ids get no crew row."""
    rng = random.Random(11)
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT, game_date TEXT, home_team_id INT);
        CREATE TABLE team_game_advanced (game_id TEXT, team_id INT, season TEXT, season_type TEXT,
                                         pts INT, opp_pts INT, pace REAL, ft_rate REAL);
        CREATE TABLE team_metadata (team_id INT, full_name TEXT, abbreviation TEXT);
        CREATE TABLE player_game_log (game_id TEXT, pf INT);
        CREATE TABLE officials (official_id INT, first_name TEXT, last_name TEXT, jersey_num TEXT);
        CREATE TABLE game_officials (game_id TEXT, official_id INT);
        CREATE TABLE officials_fetch (game_id TEXT, fetched_at TEXT, n_officials INT, source TEXT);
    """)
    c.execute("INSERT INTO team_metadata VALUES (1, 'Home Town', 'HOM'), (2, 'Away Town', 'AWY')")
    for o in range(n_officials):
        c.execute("INSERT INTO officials VALUES (?,?,?,?)", (o, "Scott", f"Official{o}", str(o)))
    gid = 0
    for s in seasons:
        for _ in range(games_per_season):
            gid += 1
            g = f"00{gid:08d}"
            crew = rng.sample(range(n_officials), 3)
            hp, ap = rng.randint(95, 125), rng.randint(95, 125)
            if hp == ap:
                hp += 1
            c.execute("INSERT INTO box_scores VALUES (?,?,?,?,1)", (g, s, "Regular Season", f"{s[:4]}-12-01"))
            c.execute("INSERT INTO team_game_advanced VALUES (?,1,?,?,?,?,?,?)",
                      (g, s, "Regular Season", hp, ap, 99.0, 0.25 + rng.gauss(0, 0.03)))
            c.execute("INSERT INTO team_game_advanced VALUES (?,2,?,?,?,?,?,?)",
                      (g, s, "Regular Season", ap, hp, 99.0, 0.25))
            pf = 40 + rng.gauss(0, 5) + (8 if loud_official in crew else 0)
            c.execute("INSERT INTO player_game_log VALUES (?,?)", (g, int(round(pf))))
            if g in crewless:
                c.execute("INSERT INTO officials_fetch VALUES (?, 'x', 0, 'v3')", (g,))
                continue
            c.execute("INSERT INTO officials_fetch VALUES (?, 'x', 3, NULL)", (g,))
            for o in crew:
                c.execute("INSERT INTO game_officials VALUES (?,?)", (g, o))
    return c


class ComputeOfficialsFDRTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res = O.compute_officials(build_archive(crewless=("0000000005", "0000000450")),
                                      min_games=25)

    def test_the_family_is_every_flagged_measure_of_every_official(self):
        t = self.res["testing"]
        expected = sum(1 for r in self.res["officials"] for k in O.FDR_FAMILY
                       if r.get(k) and r[k].get("p") is not None)
        self.assertEqual(t["tests"], expected)
        # No odds connection here, so the two market measures are absent.
        self.assertEqual(t["tests"], 3 * len(self.res["officials"]))

    def test_a_real_gap_survives_and_flags_follow_q(self):
        loud = next(r for r in self.res["officials"] if r["official_id"] == 7)
        self.assertTrue(loud["fouls"]["fdr_significant"])
        for r in self.res["officials"]:
            for k in O.FDR_FAMILY:
                m = r.get(k)
                if m and m.get("q") is not None:
                    self.assertEqual(m["fdr_significant"], m["q"] <= O.FDR_ALPHA)

    def test_correction_never_flags_more_than_the_single_test(self):
        t = self.res["testing"]
        self.assertLessEqual(t["surviving"], t["uncorrected_hits"])
        self.assertIn(str(t["tests"]), t["note"])

    def test_home_win_is_not_flagged(self):
        # The page shows Home W% without color; it must not carry a flag.
        for r in self.res["officials"]:
            self.assertNotIn("fdr_significant", r["home_win"])

    def test_coverage_is_per_season_from_the_data(self):
        cov = self.res["coverage"]
        self.assertEqual(cov["view_games"], 600)
        self.assertEqual(cov["view_games_with_crew"], 598)
        self.assertEqual(cov["view_games_without_crew"], 2)
        self.assertEqual(cov["view_first_season"], "2023-24")
        self.assertEqual(cov["view_seasons_with_gaps"], [
            {"season": "2023-24", "games": 300, "with_crew": 299},
            {"season": "2024-25", "games": 300, "with_crew": 299},
        ])

    def test_pure_noise_mostly_does_not_survive(self):
        res = O.compute_officials(build_archive(loud_official=-1), min_games=25)
        t = res["testing"]
        self.assertLessEqual(t["surviving"], 1)
        self.assertIn("none survives" if t["surviving"] == 0 else "survive", t["note"])


class TeamOfficialsFDRTest(unittest.TestCase):

    def test_team_pairs_carry_the_correction(self):
        res = O.compute_team_officials(build_archive(), "HOM", min_games=10)
        t = res["testing"]
        self.assertEqual(t["tests"], 3 * len(res["officials"]))
        for r in res["officials"]:
            for cell in (r["win_test"], r["pts_for"], r["pts_against"]):
                self.assertIn("fdr_significant", cell)
                if cell.get("q") is not None:
                    self.assertEqual(cell["fdr_significant"], cell["q"] <= O.FDR_ALPHA)


class NoteTest(unittest.TestCase):

    def test_expected_flukes_is_five_percent_of_tests(self):
        note = O._fdr_note(618, 102, 41, "x")
        self.assertIn("about 31 of them", note)
        self.assertIn("of the 102 gaps", note)
        self.assertIn("41 survive", note)
        self.assertIn("1 survives", O._fdr_note(311, 20, 1, "x"))


if __name__ == "__main__":
    unittest.main()
