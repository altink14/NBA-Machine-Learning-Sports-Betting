import unittest

from src.Utils import player_impact as pi


def make_player(**overrides):
    """A league-average-ish 30-mpg player."""
    base = {
        "player_id": 1,
        "name": "Test Player",
        "team_abbr": "AAA",
        "position": None,
        "gp": 70,
        "min": 30.0,
        "pts": 15.0,
        "fga": 12.0,
        "fta": 3.0,
        "ast": 3.5,
        "tov": 1.8,
        "oreb": 1.0,
        "dreb": 4.0,
        "stl": 1.0,
        "blk": 0.5,
        "pf": 2.0,
        "off_rating": 113.0,
        "def_rating": 113.0,
        "pace": 100.0,
    }
    base.update(overrides)
    return base


def make_teams(**overrides):
    teams = {
        "AAA": {"off_rating": 113.0, "def_rating": 113.0, "games": 82},
        "BBB": {"off_rating": 113.0, "def_rating": 113.0, "games": 82},
    }
    teams.update(overrides)
    return teams


class TestComputeImpactsBasics(unittest.TestCase):

    def test_zero_minute_players_are_skipped(self):
        players = [
            make_player(player_id=1),
            make_player(player_id=2, min=0.0),
            make_player(player_id=3, gp=0),
            make_player(player_id=4, pace=0.0),
        ]
        rows = pi.compute_impacts(players, make_teams())
        self.assertEqual([r["player_id"] for r in rows], [1])

    def test_output_has_required_fields(self):
        rows = pi.compute_impacts([make_player()], make_teams())
        row = rows[0]
        for key in ("player_id", "name", "team_abbr", "gp", "min_per_g",
                    "off_impact", "def_impact", "total_impact",
                    "off_raw", "def_raw", "total_raw", "total_minutes"):
            self.assertIn(key, row)
        self.assertAlmostEqual(
            row["total_impact"], row["off_impact"] + row["def_impact"], places=9
        )

    def test_more_efficient_scoring_scores_higher_on_offense(self):
        # Same team, same shot volume - the player converting more points
        # from the same true-shooting attempts must rate higher on offense.
        efficient = make_player(player_id=1, pts=22.0)
        inefficient = make_player(player_id=2, pts=15.0)
        rows = pi.compute_impacts([efficient, inefficient], make_teams())
        by_id = {r["player_id"]: r for r in rows}
        self.assertGreater(by_id[1]["off_impact"], by_id[2]["off_impact"])
        # Team adjustment shifts teammates equally, so the raw gap survives.
        self.assertAlmostEqual(
            by_id[1]["off_impact"] - by_id[2]["off_impact"],
            by_id[1]["off_raw"] - by_id[2]["off_raw"],
            places=9,
        )

    def test_stocks_help_defense(self):
        # Steals and blocks carry the largest defensive weights.
        stopper = make_player(player_id=1, stl=2.5, blk=2.0)
        cone = make_player(player_id=2, stl=0.3, blk=0.1)
        rows = pi.compute_impacts([stopper, cone], make_teams())
        by_id = {r["player_id"]: r for r in rows}
        self.assertGreater(by_id[1]["def_impact"], by_id[2]["def_impact"])

    def test_turnovers_hurt_offense(self):
        careful = make_player(player_id=1, tov=1.0)
        sloppy = make_player(player_id=2, tov=4.5)
        rows = pi.compute_impacts([careful, sloppy], make_teams())
        by_id = {r["player_id"]: r for r in rows}
        self.assertGreater(by_id[1]["off_impact"], by_id[2]["off_impact"])


class TestTeamAdjustment(unittest.TestCase):

    def test_team_constraint_holds(self):
        """Paine's rule: 4.5 x minute-weighted average player rating must
        equal the team's rating relative to league average."""
        teams = make_teams(
            AAA={"off_rating": 118.0, "def_rating": 110.0, "games": 82},
            BBB={"off_rating": 108.0, "def_rating": 116.0, "games": 82},
        )
        players = [
            make_player(player_id=1, team_abbr="AAA", pts=25.0, min=36.0),
            make_player(player_id=2, team_abbr="AAA", pts=10.0, min=20.0),
            make_player(player_id=3, team_abbr="BBB", pts=18.0, min=34.0),
            make_player(player_id=4, team_abbr="BBB", pts=8.0, min=15.0),
        ]
        league_ortg = 113.0
        rows = pi.compute_impacts(players, teams, league_ortg=league_ortg)
        for abbr in ("AAA", "BBB"):
            members = [r for r in rows if r["team_abbr"] == abbr]
            w = sum(m["total_minutes"] for m in members)
            avg_off = sum(m["off_impact"] * m["total_minutes"] for m in members) / w
            avg_def = sum(m["def_impact"] * m["total_minutes"] for m in members) / w
            self.assertAlmostEqual(4.5 * avg_off, teams[abbr]["off_rating"] - league_ortg, places=6)
            self.assertAlmostEqual(4.5 * avg_def, league_ortg - teams[abbr]["def_rating"], places=6)

    def test_unknown_team_skips_adjustment_but_still_rates(self):
        rows = pi.compute_impacts([make_player(team_abbr="ZZZ")], make_teams())
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["off_impact"], rows[0]["off_raw"], places=9)


class TestOnOffClamping(unittest.TestCase):

    def test_extreme_on_court_rating_is_clamped(self):
        # A tiny-minutes player with an absurd on-court rating must not
        # produce an unbounded on-off input (it is clamped to +/-25).
        players = [
            make_player(player_id=1, gp=10, min=5.0, off_rating=160.0),
            make_player(player_id=2, gp=10, min=5.0, off_rating=113.0),
        ]
        rows = pi.compute_impacts(players, make_teams())
        by_id = {r["player_id"]: r for r in rows}
        gap = by_id[1]["off_raw"] - by_id[2]["off_raw"]
        # oncourt weight (0.018381 * 47) + clamped onoff (0.032054 * 25 max)
        # bounds the gap; without the clamp it would be far larger.
        self.assertLess(gap, 2.0)


class TestAgainstRealDatabase(unittest.TestCase):
    """Sanity checks on the shipped Data/TeamData.sqlite. These assert the
    metric is *sane*, not exact values (data updates shift decimals)."""

    def test_top10_2024_25_contains_recognizable_stars(self):
        top10 = {r["name"] for r in pi.get_impact_ratings("2024-25", min_gp=20)[:10]}
        stars = {
            "Shai Gilgeous-Alexander", "Nikola Jokić", "Giannis Antetokounmpo",
            "Luka Dončić", "Tyrese Haliburton", "Jayson Tatum",
            "Victor Wembanyama", "Donovan Mitchell", "Stephen Curry",
            "Anthony Edwards", "Jimmy Butler III", "Kawhi Leonard",
        }
        self.assertGreaterEqual(
            len(top10 & stars), 4,
            f"Top-10 should be star-dominated, got: {sorted(top10)}",
        )

    def test_top10_2025_26_contains_recognizable_stars(self):
        top10 = {r["name"] for r in pi.get_impact_ratings("2025-26", min_gp=20)[:10]}
        stars = {
            "Shai Gilgeous-Alexander", "Nikola Jokić", "Giannis Antetokounmpo",
            "Luka Dončić", "Victor Wembanyama", "Kawhi Leonard",
            "Jayson Tatum", "Donovan Mitchell", "Anthony Edwards",
            "Tyrese Maxey", "Jimmy Butler III", "Tyrese Haliburton",
        }
        self.assertGreaterEqual(
            len(top10 & stars), 4,
            f"Top-10 should be star-dominated, got: {sorted(top10)}",
        )

    def test_ratings_sorted_and_ranked(self):
        rows = pi.get_impact_ratings("2024-25", min_gp=20)
        self.assertGreater(len(rows), 200)
        totals = [r["total_impact"] for r in rows]
        self.assertEqual(totals, sorted(totals, reverse=True))
        self.assertEqual([r["impact_rank"] for r in rows[:3]], [1, 2, 3])

    def test_raw_team_sums_track_net_rating(self):
        """Pre-team-adjustment ratings must already correlate strongly with
        team net rating (post-adjustment matches by construction)."""
        import sqlite3
        rows = pi._season_impacts("2024-25")
        teams = {}
        for r in rows:
            t = teams.setdefault(r["team_abbr"], [0.0, 0.0])
            t[0] += r["total_raw"] * r["total_minutes"]
            t[1] += r["total_minutes"]
        conn = sqlite3.connect(pi._DB_PATH)
        try:
            net = {
                abbr: nr for abbr, nr in conn.execute(
                    "SELECT m.abbreviation, t.net_rating FROM team_season_advanced t "
                    "JOIN team_metadata m ON m.team_id = t.team_id "
                    "WHERE t.season = '2024-25' AND t.season_type = 'Regular Season'"
                )
            }
        finally:
            conn.close()
        common = [a for a in teams if a in net and teams[a][1] > 0]
        self.assertEqual(len(common), 30)
        xs = [4.5 * teams[a][0] / teams[a][1] for a in common]
        ys = [net[a] for a in common]
        self.assertGreater(_spearman(xs, ys), 0.6)

    def test_player_series_covers_seasons(self):
        series = pi.get_player_impact(1628983)  # Shai Gilgeous-Alexander
        self.assertGreaterEqual(len(series), 3)
        self.assertTrue(all(s["player_id"] == 1628983 for s in series))
        seasons = [s["season"] for s in series]
        self.assertEqual(seasons, sorted(seasons))


def _spearman(xs, ys):
    """Spearman rank correlation without scipy (ties broken by order,
    fine for continuous inputs)."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mean = (n - 1) / 2.0
    cov = sum((a - mean) * (b - mean) for a, b in zip(rx, ry))
    var = sum((a - mean) ** 2 for a in rx)
    return cov / var if var else 0.0


if __name__ == "__main__":
    unittest.main()
