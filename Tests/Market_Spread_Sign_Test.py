"""
The historical odds dataset stores the spread UNSIGNED in every season before
2022-23 (the favourite's line, never negative). Market.py used to read it as
the home side's signed margin, so every one of those games looked
home-favoured and about a third of each season was graded against the wrong
side of the line. These tests pin the recovery rule and check it against the
real archive when it is present.
"""

import os
import sqlite3
import unittest

from src.Utils import Market as market

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")
TEAM_DB = os.path.join(REPO, "Data", "TeamData.sqlite")


class TestSignedHomeSpread(unittest.TestCase):

    def test_signed_season_is_taken_as_is(self):
        self.assertEqual(market.signed_home_spread("2022-23", -6.5, 250, -300), -6.5)
        self.assertEqual(market.signed_home_spread("2022-23", 4.0, -180, 150), 4.0)

    def test_unsigned_season_home_favourite(self):
        # home shorter on the moneyline: home favoured, positive
        self.assertEqual(market.signed_home_spread("2015-16", 9.5, -600, 450), 9.5)

    def test_unsigned_season_away_favourite(self):
        # away shorter: the same unsigned 9.5 belongs to the away side
        self.assertEqual(market.signed_home_spread("2015-16", 9.5, 450, -600), -9.5)

    def test_stray_negative_in_unsigned_season_is_treated_as_magnitude(self):
        # 2019-20 has one negative row; the column is still a magnitude there
        self.assertEqual(market.signed_home_spread("2019-20", -3.0, 130, -150), -3.0)
        self.assertEqual(market.signed_home_spread("2019-20", -3.0, -150, 130), 3.0)

    def test_pickem_and_unknown(self):
        self.assertEqual(market.signed_home_spread("2015-16", 0.0, -110, -110), 0.0)
        # equal moneylines and a real line: no recoverable side, never guessed
        self.assertIsNone(market.signed_home_spread("2015-16", 1.0, -110, -110))
        self.assertIsNone(market.signed_home_spread("2015-16", 3.0, None, -150))
        self.assertIsNone(market.signed_home_spread("2015-16", None, -150, 130))


@unittest.skipUnless(os.path.exists(ODDS_DB) and os.path.exists(TEAM_DB), "archive databases not present")
class TestAgainstTheArchive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        market._lines = None  # the module caches the loaded lines
        cls.odds = sqlite3.connect(ODDS_DB)
        cls.team = sqlite3.connect(TEAM_DB)

    @classmethod
    def tearDownClass(cls):
        cls.odds.close()
        cls.team.close()

    def test_warriors_2015_16_are_heavy_favourites(self):
        # 73-9 and favoured in 70 games: the average own line must be a big
        # negative number. Read unsigned it came out -2.4.
        data = market.season_market(self.team, self.odds, "2015-16")
        gsw = next(t for t in data["teams"] if t["abbr"] == "GSW")
        self.assertLess(gsw["avg_line"], -7.0)
        self.assertLess(gsw["avg_cover_margin"], 5.0)

    def test_home_ats_is_near_half_across_unsigned_seasons(self):
        # Home teams cover a bit under half the time; with the sign ignored it
        # read about 42.7%. Every unsigned season must land near 50%.
        for season in ("2008-09", "2012-13", "2015-16", "2019-20", "2021-22"):
            lg = market.season_market(self.team, self.odds, season)["league"]
            covers, fails, _ = lg["home_ats"]
            share = covers / (covers + fails)
            self.assertGreater(share, 0.44, season)
            self.assertLess(share, 0.56, season)

    def test_average_own_lines_net_to_zero(self):
        # Every line is one team's -x and the other's +x, so across the league
        # the averages cancel. Unsigned, every home team carried a negative line.
        data = market.season_market(self.team, self.odds, "2015-16")
        total = sum(t["avg_line"] * t["graded"] for t in data["teams"] if t["avg_line"] is not None)
        games = sum(t["graded"] for t in data["teams"])
        self.assertLess(abs(total / games), 0.5)


if __name__ == "__main__":
    unittest.main()
