"""Teams pages: standings SRS cap, team roster, on/off names and minutes, lineups feed cap.

Fix round 3 (2026-09-28), from the nav audit's TEAMS section:
- SRS clips every game margin at +/-30 and nothing said so. The standings
  rows now carry srs_margin_cap, capped_games and srs_uncapped so the page can
  state the cap and its effect (2025-26: 87 games past 30, OKC 10.34 -> 11.04).
- The team roster read jersey/position from player_bio only, which is empty
  for ~3 in 4 players (2025-26: 487 of 661 rows printed "--"). It now falls
  back to the league index in `players`, gives a number only while that
  listing is with THIS team, and no longer returns every player in franchise
  history for a season with no totals yet.
- On/off named three OKC players "J./K. Williams"; the Trade Machine built its
  floor share from stint minutes, which leave out dropped periods. The engine
  now returns full names and box-score minutes (player and team).
- The lineups feed is capped at 2,000 rows by nba.com, which the page called
  "combinations on record". The response says when the cap is hit.
Real-archive checks skip when Data/TeamData.sqlite is absent. No network.
"""
import os
import sqlite3
import unittest
from unittest import mock

import main_api
from src.Utils.nba_computed_derivatives import SRS_BLOWOUT_CAP, TeamRecord, compute_srs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_DB = os.path.join(REPO, "Data", "TeamData.sqlite")
HAVE_DB = os.path.exists(TEAM_DB)


def _ro():
    c = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


class SrsCapTest(unittest.TestCase):
    def test_cap_is_the_named_constant(self):
        self.assertEqual(SRS_BLOWOUT_CAP, 30.0)
        # A 50-point rout counts exactly as a 30-point one under the cap.
        def league(rout):
            games = [(1, 2, rout), (1, 3, 8.0), (2, 3, 4.0), (2, 1, -6.0), (3, 1, 3.0), (3, 2, -2.0)]
            recs = {t: TeamRecord(t, str(t)) for t in (1, 2, 3)}
            for a, b, m in games:
                recs[a].point_diffs.append(m); recs[a].opponent_ids.append(b)
                recs[b].point_diffs.append(-m); recs[b].opponent_ids.append(a)
            return recs
        capped, _ = compute_srs(league(50.0))
        as_thirty, _ = compute_srs(league(30.0), blowout_cap=float("inf"))
        uncapped, _ = compute_srs(league(50.0), blowout_cap=float("inf"))
        for t in (1, 2, 3):
            self.assertAlmostEqual(capped[t], as_thirty[t], places=4)
        self.assertGreater(uncapped[1], capped[1] + 1.0)

    @unittest.skipUnless(HAVE_DB, "needs the real archive")
    def test_standings_report_the_cap_and_its_effect(self):
        rows = main_api.get_stats_standings(season="2025-26")
        self.assertEqual(len(rows), 30)
        for r in rows:
            self.assertEqual(r["srs_margin_cap"], SRS_BLOWOUT_CAP)
            self.assertIsNotNone(r["capped_games"])
            self.assertIsNotNone(r["srs_uncapped"])
        c = _ro()
        try:
            beyond = c.execute(
                "SELECT COUNT(*) FROM team_game_advanced WHERE season='2025-26' "
                "AND season_type='Regular Season' AND ABS(pts - opp_pts) > 30"
            ).fetchone()[0]
        finally:
            c.close()
        # Each game is two team rows.
        self.assertEqual(sum(r["capped_games"] for r in rows), beyond)
        self.assertEqual(beyond // 2, 87)
        okc = next(r for r in rows if r["abbreviation"] == "OKC")
        self.assertAlmostEqual(okc["srs"], 10.34, places=2)
        self.assertAlmostEqual(okc["srs_uncapped"], 11.04, places=2)
        # Uncapped SRS still centres on zero.
        self.assertAlmostEqual(sum(r["srs_uncapped"] for r in rows), 0.0, places=6)


class RosterRowTest(unittest.TestCase):
    def test_number_only_from_a_listing_with_this_team(self):
        base = {"player_id": 1, "full_name": "A B", "first_name": "A", "last_name": "B",
                "gp": 10, "min": 300.0, "pts": 100, "reb": 40, "ast": 20}
        # Bio lists him with another team now; the league index still has him here.
        r = main_api._roster_row({**base, "bio_jersey": "4", "bio_team_id": 99, "bio_position": "Center-Forward",
                                  "idx_jersey": "23", "idx_team_id": 7, "idx_position": "C-F"}, 7)
        self.assertEqual(r["jersey"], "23")
        self.assertEqual(r["position"], "C-F")
        # Both listings elsewhere: his number here is unknown, never borrowed.
        r = main_api._roster_row({**base, "bio_jersey": "4", "bio_team_id": 99, "bio_position": None,
                                  "idx_jersey": "5", "idx_team_id": 98, "idx_position": "F"}, 7)
        self.assertIsNone(r["jersey"])
        self.assertEqual(r["position"], "F")
        # Empty strings are unknown too.
        r = main_api._roster_row({**base, "bio_jersey": "", "bio_team_id": 7, "bio_position": "",
                                  "idx_jersey": None, "idx_team_id": 7, "idx_position": None}, 7)
        self.assertIsNone(r["jersey"])
        self.assertIsNone(r["position"])

    @unittest.skipUnless(HAVE_DB, "needs the real archive")
    def test_real_rosters(self):
        total = unknown_pos = unknown_num = 0
        c = _ro()
        try:
            abbrs = [r[0] for r in c.execute("SELECT abbreviation FROM team_metadata")]
        finally:
            c.close()
        for a in abbrs:
            rows = main_api.get_team_roster(a, "2025-26")
            total += len(rows)
            unknown_pos += sum(r["position"] is None for r in rows)
            unknown_num += sum(r["jersey"] is None for r in rows)
        self.assertEqual(total, 661)
        self.assertEqual(unknown_pos, 0)          # was 483 before the fallback
        self.assertLess(unknown_num, 487)         # was 487; traded players stay blank
        nyk = {r["full_name"]: r for r in main_api.get_team_roster("NYK", "2025-26")}
        self.assertEqual(nyk["Mitchell Robinson"]["jersey"], "23")   # not his new team's 4
        self.assertIsNone(nyk["Guerschon Yabusele"]["jersey"])       # traded to CHI
        # A season with no totals yet is empty, not the franchise's history.
        self.assertEqual(main_api.get_team_roster("NYK", "2031-32"), [])


@unittest.skipUnless(HAVE_DB, "needs the real archive")
class OnOffNamesAndMinutesTest(unittest.TestCase):
    def test_full_names_and_box_minutes(self):
        c = _ro()
        try:
            from src.Utils import LineupEngine
            r = LineupEngine.compute_team_onoff(c, "OKC", "2025-26")
            team_min = c.execute(
                "SELECT SUM(min) FROM player_season_totals WHERE team_id=1610612760 "
                "AND season='2025-26' AND season_type='Regular Season'").fetchone()[0]
            sga_box = c.execute(
                "SELECT gp, min FROM player_season_totals WHERE player_id=1628983 AND team_id=1610612760 "
                "AND season='2025-26' AND season_type='Regular Season'").fetchone()
        finally:
            c.close()
        williams = sorted(p["full_name"] for p in r["players"] if p["name"] in ("J. Williams", "K. Williams"))
        self.assertEqual(williams, ["Jalen Williams", "Jaylin Williams", "Kenrich Williams"])
        self.assertAlmostEqual(r["box_team_minutes"], round(team_min / 5.0, 1), places=1)
        # 82 games x 48 plus overtime.
        self.assertGreaterEqual(r["box_team_minutes"], 82 * 48)
        sga = next(p for p in r["players"] if p["player_id"] == 1628983)
        self.assertEqual(sga["box_gp"], sga_box["gp"])
        self.assertAlmostEqual(sga["box_min"], round(sga_box["min"], 1), places=1)
        # Stint minutes miss dropped periods; the box score does not.
        self.assertLess(sga["min_on"], sga["box_min"])
        self.assertAlmostEqual(sga["box_min"] / sga["box_gp"], 33.2, places=1)
        self.assertTrue(all(len(l["player_names"]) == 5 for l in r["lineups"]))


class LineupsFeedCapTest(unittest.TestCase):
    def _rows(self, n):
        return [{"GROUP_ID": f"g{i}", "GROUP_NAME": "A - B - C - D - E", "TEAM_ABBREVIATION": "OKC",
                 "TEAM_ID": 1, "GP": 3, "W": 2, "L": 1, "MIN": 500.0 - i * 0.24,
                 "NET_RATING": 1.0} for i in range(n)]

    def _call(self, rows):
        from fastapi.testclient import TestClient
        client = mock.Mock()
        client.league_dash_lineups.return_value = rows
        with mock.patch("src.Utils.nba_stats_client.get_client", return_value=client):
            return TestClient(main_api.app).get("/api/lineups?season=2025-26").json()

    def test_capped_feed_is_labelled(self):
        out = self._call(self._rows(main_api.LINEUP_FEED_ROW_CAP))
        self.assertTrue(out["feed_capped"])
        self.assertEqual(out["total_lineups"], 2000)
        self.assertAlmostEqual(out["feed_min_minutes"], round(500.0 - 1999 * 0.24, 1), places=1)

    def test_short_feed_is_not(self):
        out = self._call(self._rows(150))
        self.assertFalse(out["feed_capped"])


if __name__ == "__main__":
    unittest.main()
