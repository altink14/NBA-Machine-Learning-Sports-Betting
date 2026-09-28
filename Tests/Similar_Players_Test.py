"""Similar players: per-season z-scores, missing is missing, whole-season lines.

Found 2026-09-24 (nav audit): the page said "the ten closest stat lines in
the archive" but searched one season, min-max scaling squeezed every match
into 91-94%, missing stats became 0, and "12 stats" sat beside a list of six.
The matcher now lives in src/Utils/similar_players.py. Synthetic pools except
the last class, which reads the real archive read-only and skips without it.
"""
import os
import sqlite3
import unittest

import numpy as np

from src.Utils import similar_players as sp

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "TeamData.sqlite")


def _row(pid, season, gp=60, scale=1.0, **over):
    """A whole-season row shaped like load_pool's output (season totals)."""
    base = dict(pts=15, ast=3, reb=5, min=28, fg3a=3, fta=3, stl=1, blk=0.5, tov=1.5)
    rng = np.random.default_rng(pid)
    per_game = {k: v * (0.5 + rng.random()) for k, v in base.items()}
    per_game.update({k: v for k, v in over.items() if k in base})
    row = {k: per_game[k] * gp * scale for k in base}
    row["min"] = per_game["min"] * gp      # minutes do not inflate with the era
    row.update(player_id=pid, season=season, gp=gp, n_teams=1, team_id=1, age=None,
               full_name=f"P{pid}", fga=row["pts"] / 2.2, fta=row["fta"],
               usg_pct=over.get("usg_pct", 0.15 + 0.1 * rng.random()))
    return row


def _pool(rows, games=82):
    return sp.Pool(rows, {r["season"]: games for r in rows})


class PerSeasonScaling(unittest.TestCase):
    def test_same_line_relative_to_its_era_is_a_zero_gap(self):
        # Season B is season A with every counting stat doubled (a faster,
        # higher-scoring era). Relative to its season, each B line equals its
        # A twin, so the gap is 0 and the twin is the closest match.
        a = [_row(i, "A") for i in range(1, 41)]
        b = [dict(r, player_id=r["player_id"] + 100, season="B", full_name=f"B{r['player_id']}",
                  **{k: r[k] * 2 for k in ("pts", "ast", "reb", "fg3a", "fta", "stl", "blk", "tov", "fga")})
             for r in a]
        pool = _pool(a + b)
        res = sp.find_similar(pool, 7, "A", scope="all", limit=3)
        self.assertEqual(res["matches"][0]["player_id"], 107)
        self.assertAlmostEqual(res["matches"][0]["gap"], 0.0, places=6)
        self.assertEqual(res["seasons_searched"], {"first": "A", "last": "B", "count": 2})

    def test_scope_season_stays_inside_the_season(self):
        a = [_row(i, "A") for i in range(1, 41)]
        b = [_row(i + 100, "B") for i in range(1, 41)]
        res = sp.find_similar(_pool(a + b), 7, "A", scope="season")
        self.assertTrue(res["matches"])
        self.assertEqual({m["season"] for m in res["matches"]}, {"A"})

    def test_one_outlier_does_not_squeeze_everyone_together(self):
        # Min-max scaling put one 60-point outlier at the top of the range and
        # every ordinary player in a sliver of it; z-scores keep ordinary
        # players spread apart.
        rows = [_row(i, "A") for i in range(1, 41)] + [_row(99, "A", pts=60)]
        res = sp.find_similar(_pool(rows), 5, "A", scope="season", limit=10)
        gaps = [m["gap"] for m in res["matches"]]
        self.assertGreater(max(gaps) - min(gaps), 0.1)


class MissingIsMissing(unittest.TestCase):
    def test_unknown_usage_is_left_out_not_counted_as_zero(self):
        rows = [_row(i, "A") for i in range(1, 41)]
        twin = dict(rows[4], player_id=500, full_name="twin", usg_pct=None)   # rows[4] is player 5
        pool = _pool(rows + [twin])
        res = sp.find_similar(pool, 5, "A", scope="season", limit=1)
        self.assertEqual(res["matches"][0]["player_id"], 500)
        self.assertAlmostEqual(res["matches"][0]["gap"], 0.0, places=6)
        self.assertIsNone(res["matches"][0]["stats"]["usg_pct"])
        self.assertIsNone(res["matches"][0]["z"]["usg_pct"])
        w_usg = dict((k, w) for k, _, w, _ in sp.STATS)["usg_pct"]
        self.assertAlmostEqual(res["matches"][0]["shared_weight"], 1 - w_usg / sp.WEIGHTS.sum(), places=3)

    def test_zero_usage_placeholder_reads_as_unknown(self):
        self.assertIsNone(sp._per_game(dict(_row(1, "A"), usg_pct=0.0))["usg_pct"])

    def test_a_pair_sharing_too_little_is_not_compared(self):
        rows = [_row(i, "A") for i in range(1, 41)]
        blank = dict(rows[4], player_id=501, full_name="blank")
        for k in ("pts", "ast", "reb", "fg3a", "fta"):
            blank[k] = None
        pool = _pool(rows + [blank])
        res = sp.find_similar(pool, 5, "A", scope="season", limit=40)
        self.assertNotIn(501, [m["player_id"] for m in res["matches"]])


class PoolRules(unittest.TestCase):
    def test_short_stints_are_not_candidates_but_can_be_searched(self):
        rows = [_row(i, "A") for i in range(1, 41)] + [_row(77, "A", gp=10)]   # 10 < 25% of 82
        pool = _pool(rows)
        self.assertFalse(pool.qualified[pool.index[(77, "A")]])
        res = sp.find_similar(pool, 77, "A", scope="season")
        self.assertFalse(res["target"]["qualified"])
        self.assertEqual(len(res["matches"]), 10)
        other = sp.find_similar(pool, 3, "A", scope="season", limit=50)
        self.assertNotIn(77, [m["player_id"] for m in other["matches"]])

    def test_games_floor_scales_with_the_season(self):
        rows = [_row(i, "L", gp=40) for i in range(1, 41)] + [_row(88, "L", gp=13)]
        pool = sp.Pool(rows, {"L": 50})           # 1998-99: 50 games, floor 13
        self.assertTrue(pool.qualified[pool.index[(88, "L")]])

    def test_one_entry_per_player_and_never_the_player_himself(self):
        a = [_row(i, "A") for i in range(1, 41)]
        b = [dict(_row(i, "B"), player_id=i) for i in range(1, 41)]   # same players, next season
        res = sp.find_similar(_pool(a + b), 5, "A", scope="all", limit=15)
        ids = [m["player_id"] for m in res["matches"]]
        self.assertNotIn(5, ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_method_lists_exactly_the_stats_used(self):
        m = sp.method()
        self.assertEqual([s["key"] for s in m["stats"]], sp.KEYS)
        self.assertEqual(len(m["stats"]), len(sp.WEIGHTS))
        uppers = [b[0] for b in sp.BANDS]
        self.assertEqual(uppers, sorted(uppers))
        self.assertEqual(sp.band(0.0), sp.BANDS[0][1])
        self.assertEqual(sp.band(5.0), sp.BANDS[-1][1])


@unittest.skipUnless(os.path.exists(DB), "archive not present")
class RealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        cls.conn.row_factory = sqlite3.Row
        cls.pool = sp.load_pool(cls.conn)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_traded_seasons_are_the_whole_season_line(self):
        import main_api
        traded = self.conn.execute(
            "SELECT player_id, season FROM player_season_totals WHERE season_type = 'Regular Season' "
            "GROUP BY player_id, season HAVING COUNT(*) > 1 ORDER BY season DESC, player_id LIMIT 5").fetchall()
        self.assertTrue(traded)
        for pid, season in traded:
            totals, _, n = main_api._player_season_line(self.conn, pid, season)
            self.assertGreater(n, 1)
            row = self.pool.rows[self.pool.index[(pid, season)]]
            self.assertEqual(row["gp"], totals["gp"])
            self.assertAlmostEqual(row["per_game"]["pts"], totals["pts"] / totals["gp"], places=9)

    def test_every_archived_season_is_searched(self):
        seasons = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT season FROM player_season_totals WHERE season_type = 'Regular Season' ORDER BY season")]
        self.assertEqual(self.pool.seasons, seasons)
