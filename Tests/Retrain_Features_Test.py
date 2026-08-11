"""
Retrain_Features_Test.py
========================
Validation gates for STEP 1 of the pre-registered retrain protocol
(src/Process-Data/retrain_features.py). Run from the repo root:

    venv/Scripts/python.exe -m pytest Tests/Retrain_Features_Test.py -q

Gates:
  1. Overlap parity (critical): snapshot-diff rolling-20 vs direct
     per-game rolling-20 from box_scores, 2022-23 and 2023-24 regular
     seasons. PASS = 99%+ of counting-stat cells within the theoretical
     rounding bound 0.05*(GP + GP')/K.
  2. Elo cross-check: odds-table Elo vs src/Utils/elo.py (both clean-start
     at 2022-23) compared at the aligned cutoff 2024-04-14; Pearson
     r >= 0.98.
  3. W/L exactness: per-game W/L reconstructed from integer snapshot W
     diffs must exactly match odds-table results (2015-16 and 2018-19).
  4. Rest features: synthetic-schedule unit tests.
  5. Leakage guard: a game's own result cannot enter its features.

Heavy fixtures are built once per class. Sampling decisions are documented
inline; the parity gate itself is exhaustive over both overlap seasons.
"""

import datetime as dt
import importlib.util
import json
import os
import sqlite3
import sys
import time
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SPEC = importlib.util.spec_from_file_location(
    "retrain_features",
    os.path.join(REPO_ROOT, "src", "Process-Data", "retrain_features.py"))
rf = importlib.util.module_from_spec(_SPEC)
sys.modules["retrain_features"] = rf  # dataclasses need sys.modules entry
_SPEC.loader.exec_module(rf)

from src.Utils import elo as elo_mod  # noqa: E402

TEAM_DATA_DB = rf.TEAM_DATA_DB
ODDS_DATA_DB = rf.ODDS_DATA_DB

# box_scores traditional_json statistics key -> snapshot column
BOX_KEY_MAP = {
    "fieldGoalsMade": "FGM", "fieldGoalsAttempted": "FGA",
    "threePointersMade": "FG3M", "threePointersAttempted": "FG3A",
    "freeThrowsMade": "FTM", "freeThrowsAttempted": "FTA",
    "reboundsOffensive": "OREB", "reboundsDefensive": "DREB",
    "reboundsTotal": "REB", "assists": "AST", "steals": "STL",
    "blocks": "BLK", "turnovers": "TOV", "foulsPersonal": "PF",
    "points": "PTS", "plusMinusPoints": "PLUS_MINUS",
}
PARITY_STATS = tuple(BOX_KEY_MAP.values())

# TOV is collected but gated separately: the snapshot tables
# (stats.nba.com leaguedashteamstats) count TOTAL team turnovers,
# including team turnovers (shot-clock / 8-second / 5-second violations),
# while the box_scores traditional team row is exactly the PLAYER-SUMMED
# turnovers (verified: Boston 2023-24 cumulative TOV is 133/284/520 in the
# snapshots vs 125/268/492 player-summed at GP 10/20/40). The mismatch is
# in the validation oracle, not the builder; no total-TOV source exists in
# box_scores (pbp_json is populated for only 3 games ever). TOV therefore
# gets a one-sided bias test instead of the rounding-bound gate.
GATED_STATS = tuple(s for s in PARITY_STATS if s != "TOV")


def _load_box_games(seasons):
    """Per-team chronological regular-season game logs from box_scores.

    Returns {season: {canonical_team: [(date, {stat: value}, won), ...]}}.
    """
    conn = sqlite3.connect(TEAM_DATA_DB)
    try:
        id_to_name = {r[0]: r[1] for r in conn.execute(
            "SELECT team_id, full_name FROM team_metadata")}
        out = {s: {} for s in seasons}
        q = ("SELECT season, game_date, home_team_id, away_team_id, "
             "traditional_json FROM box_scores WHERE season IN ({}) AND "
             "season_type='Regular Season' ORDER BY game_date, game_id"
             ).format(",".join("?" * len(seasons)))
        for season, date, home_id, away_id, blob in conn.execute(q, seasons):
            box = json.loads(blob)["boxScoreTraditional"]
            for side, team_id in (("homeTeam", home_id), ("awayTeam", away_id)):
                stats_raw = box[side]["statistics"]
                stats = {col: float(stats_raw[k])
                         for k, col in BOX_KEY_MAP.items()}
                team = rf.normalize_team(id_to_name[team_id])
                won = stats["PLUS_MINUS"] > 0
                out[season].setdefault(team, []).append((date, stats, won))
        for season in out:
            for team in out[season]:
                out[season][team].sort(key=lambda g: g[0])
        return out
    finally:
        conn.close()


class TestTeamNameMap(unittest.TestCase):
    def test_aliases_resolve(self):
        self.assertEqual(rf.normalize_team("Seattle SuperSonics"),
                         "Oklahoma City Thunder")
        self.assertEqual(rf.normalize_team("New Jersey Nets"),
                         "Brooklyn Nets")
        self.assertEqual(rf.normalize_team("New Orleans Hornets"),
                         "New Orleans Pelicans")
        self.assertEqual(rf.normalize_team("Charlotte Bobcats"),
                         "Charlotte Hornets")
        self.assertEqual(rf.normalize_team("LA Clippers"),
                         "Los Angeles Clippers")
        self.assertEqual(rf.normalize_team("Boston Celtics"),
                         "Boston Celtics")

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            rf.normalize_team("Vancouver Grizzlies")

    def test_every_db_name_normalizes_to_30_teams(self):
        """Every name in every odds table and a spread of snapshot tables
        must normalize, and each season must contain exactly 30 franchises."""
        conn = sqlite3.connect(ODDS_DATA_DB)
        try:
            for season in rf.ODDS_SEASONS:
                names = {r[0] for r in conn.execute(
                    f'SELECT DISTINCT Home FROM "odds_{season}_new"')}
                names |= {r[0] for r in conn.execute(
                    f'SELECT DISTINCT Away FROM "odds_{season}_new"')}
                canon = {rf.normalize_team(n) for n in names}
                self.assertEqual(len(canon), 30, season)
        finally:
            conn.close()
        conn = sqlite3.connect(TEAM_DATA_DB)
        try:
            # One late-season snapshot per era covers every alias epoch.
            for snap in ("2008-04-16", "2012-04-26", "2013-04-17",
                         "2014-04-16", "2015-04-15", "2016-04-13",
                         "2024-04-14"):
                names = [r[0] for r in conn.execute(
                    f'SELECT TEAM_NAME FROM "{snap}"')]
                canon = {rf.normalize_team(n) for n in names}
                self.assertEqual(len(canon), 30, snap)
        finally:
            conn.close()


class TestOverlapParity(unittest.TestCase):
    """THE critical gate: rolling-20 built from snapshot diffs vs built
    directly from box_scores per-game data, exhaustively for every
    team-date in the 2022-23 and 2023-24 regular seasons where GP >= 20.
    """
    K = 20
    SEASONS = ("2022-23", "2023-24")

    @classmethod
    def setUpClass(cls):
        cls.store = rf.SnapshotStore()
        t0 = time.perf_counter()
        cls.box = _load_box_games(list(cls.SEASONS))
        cls.box_load_s = time.perf_counter() - t0

        cls.results = {s: [] for s in PARITY_STATS}  # (|diff|, bound)
        cls.tov_signed = []  # snapshot-diff TOV minus player-sum TOV
        cls.pct_results = {s: [] for s in rf.PCT_STATS}
        cls.win_pct_diffs = []
        cls.misaligned = 0
        cls.inexact_window = 0
        cls.n_points = 0

        t0 = time.perf_counter()
        for season in cls.SEASONS:
            for team, games in cls.box[season].items():
                for i in range(cls.K, len(games)):
                    date = games[i][0]
                    feat = rf.build_rolling_features(cls.K, date, team,
                                                     cls.store)
                    if feat is None or feat["gp"] != i:
                        cls.misaligned += 1
                        continue
                    if not feat["exact_window"] or feat["is_partial"]:
                        cls.inexact_window += 1
                        continue
                    window = games[i - cls.K:i]
                    gp_now, gp_prev = i, i - cls.K
                    bound = 0.05 * (gp_now + gp_prev) / cls.K + 1e-9
                    cls.n_points += 1
                    direct_tot = {s: sum(g[1][s] for g in window)
                                  for s in PARITY_STATS}
                    for s in PARITY_STATS:
                        direct = direct_tot[s] / cls.K
                        cls.results[s].append(
                            (abs(feat["stats"][s] - direct), bound))
                        if s == "TOV":
                            cls.tov_signed.append(
                                feat["stats"][s] - direct)
                    for pct, (num, den) in rf.PCT_STATS.items():
                        d = direct_tot[den]
                        if d == 0:
                            continue
                        direct = direct_tot[num] / d
                        # propagated rounding bound for a ratio of two
                        # reconstructed totals
                        err_tot = 0.05 * (gp_now + gp_prev)
                        pbound = err_tot * (1.0 + direct) / d + 1e-9
                        cls.pct_results[pct].append(
                            (abs(feat["stats"][pct] - direct), pbound))
                    direct_wpct = sum(1 for g in window if g[2]) / cls.K
                    cls.win_pct_diffs.append(
                        abs(feat["stats"]["WIN_PCT"] - direct_wpct))
        cls.build_s = time.perf_counter() - t0

    def test_alignment(self):
        """Snapshot GP must equal the direct game count at every point;
        misalignments would mean the snapshot-date convention is wrong.
        (82-20)*30 teams*2 seasons = 3720 comparable team-dates."""
        self.assertEqual(self.n_points, 3720)
        self.assertEqual(self.misaligned, 0)
        self.assertEqual(self.inexact_window, 0)

    def test_counting_stats_within_rounding_bound(self):
        """PASS = 99%+ of cells within 0.05*(GP+GP')/K, per stat (all
        stats where both sources measure the same quantity)."""
        summary = []
        for s in GATED_STATS:
            cells = self.results[s]
            within = sum(1 for d, b in cells if d <= b) / len(cells)
            mx = max(d for d, _ in cells)
            mean = sum(d for d, _ in cells) / len(cells)
            summary.append(f"{s}: within={within:.4%} max={mx:.4f} "
                           f"mean={mean:.4f}")
            self.assertGreaterEqual(
                within, 0.99,
                f"{s} parity gate FAILED ({within:.4%} within bound); "
                f"max|diff|={mx:.4f}\n" + "\n".join(summary))
        print("\n[overlap parity, K=20, n=%d team-dates, %d cells/stat]"
              % (self.n_points, len(self.results[PARITY_STATS[0]])))
        for line in summary:
            print("  " + line)

    def test_tov_definition_mismatch_is_one_sided(self):
        """TOV cannot meet the rounding bound against this oracle because
        the sources define it differently (see GATED_STATS comment). The
        diagnosis is pinned here: the snapshot-diff TOV must NEVER fall
        below the player-sum TOV by more than the rounding bound (the
        extra team turnovers are non-negative), and the mean gap must look
        like a plausible league team-turnover rate. A formula bug in the
        builder would produce two-sided errors and fail this test."""
        bound = 0.05 * (2 * 82) / self.K  # most generous rounding bound
        mn, mx = min(self.tov_signed), max(self.tov_signed)
        mean = sum(self.tov_signed) / len(self.tov_signed)
        print(f"  TOV signed gap (snapshot minus player-sum): "
              f"min={mn:.3f} mean={mean:.3f} max={mx:.3f}")
        self.assertGreaterEqual(mn, -bound)      # one-sided
        self.assertGreater(mean, 0.3)            # team TOs exist
        self.assertLess(mean, 1.2)               # ...but are ~0.7/game
        self.assertLess(mx, 3.5)

    def test_pct_stats_close(self):
        """Percentages are recomputed from reconstructed totals; check the
        propagated bound and that mean error is tiny."""
        for pct in rf.PCT_STATS:
            cells = self.pct_results[pct]
            within = sum(1 for d, b in cells if d <= b) / len(cells)
            mean = sum(d for d, _ in cells) / len(cells)
            print(f"  {pct}: within_propagated_bound={within:.4%} "
                  f"mean|diff|={mean:.5f}")
            self.assertGreaterEqual(within, 0.99, pct)
            self.assertLess(mean, 0.01, pct)

    def test_rolling_win_pct_exact(self):
        """W diffs are exact integers: rolling win% must match exactly."""
        self.assertLess(max(self.win_pct_diffs), 1e-9)

    def test_runtime_sane(self):
        """Snapshot-diff builds for two full seasons of team-dates must be
        fast enough for step 2 (well under 2 minutes)."""
        print(f"  [runtime] box_scores load+parse {self.box_load_s:.1f}s, "
              f"{self.n_points} rolling builds in {self.build_s:.1f}s")
        self.assertLess(self.build_s, 120)


class TestEloCrossCheck(unittest.TestCase):
    """Odds-table Elo vs src/Utils/elo.py, both starting clean at 2022-23
    (the production build burns in from 2007-08; this comparison isolates
    engine agreement from burn-in). The two game feeds differ slightly --
    the odds archive is missing 2024-02-15, 2024-02-22..28 and 2024-03-29
    entirely (72 games) plus one 2022-23 game, and stops on 2024-04-28
    while box_scores carries the full 2023-24 playoffs -- so ratings are
    compared at the aligned cutoff 2024-04-14 (end of the 2023-24 regular
    season, before the odds feed's playoff truncation can diverge)."""
    CUTOFF = "2024-04-14"

    @classmethod
    def setUpClass(cls):
        cls.mine = rf.build_elo_odds(start_season="2022-23",
                                     end_season="2023-24")
        cls.theirs = elo_mod.compute_elo_history(TEAM_DATA_DB)
        store = rf.SnapshotStore()
        store._load_season("2023-24")
        cls.name_to_id = dict(store.team_ids)

    def _my_rating_asof(self, team, cutoff):
        rating = rf.ELO_BASE
        entries = []
        for (date, home, away), v in self.mine["post_game"].items():
            if date > cutoff:
                continue
            if home == team:
                entries.append((date, v["home_elo"]))
            elif away == team:
                entries.append((date, v["away_elo"]))
        entries.sort()
        return entries[-1][1] if entries else rating

    def test_pearson_r_at_aligned_cutoff(self):
        xs, ys, diffs = [], [], {}
        for team in rf.CANONICAL_TEAMS:
            mine = self._my_rating_asof(team, self.CUTOFF)
            tid = self.name_to_id[team]
            theirs = elo_mod.elo_as_of(self.theirs["timelines"][tid],
                                       self.CUTOFF)
            xs.append(mine)
            ys.append(theirs)
            diffs[team] = mine - theirs
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        r = cov / (vx * vy)
        worst = max(diffs.items(), key=lambda kv: abs(kv[1]))
        print(f"\n[elo cross-check @ {self.CUTOFF}] r={r:.4f} "
              f"mean|diff|={sum(abs(d) for d in diffs.values())/n:.1f} "
              f"max|diff|={abs(worst[1]):.1f} ({worst[0]})")
        self.assertGreaterEqual(r, 0.98,
                                f"Elo cross-check r={r:.4f} below 0.98; "
                                f"per-team diffs: {diffs}")

    def test_game_feed_discrepancies_documented(self):
        """Quantify the feed gap so it is on the record: the odds archive
        must be missing exactly the known dates for 2023-24 RS."""
        conn = sqlite3.connect(ODDS_DATA_DB)
        odd_dates = {r[0] for r in conn.execute(
            "SELECT DISTINCT Date FROM \"odds_2023-24_new\" "
            "WHERE Date <= '2024-04-14'")}
        conn.close()
        conn = sqlite3.connect(TEAM_DATA_DB)
        box_dates = {r[0] for r in conn.execute(
            "SELECT DISTINCT game_date FROM box_scores WHERE "
            "season='2023-24' AND season_type='Regular Season'")}
        conn.close()
        missing = sorted(box_dates - odd_dates)
        print(f"  odds feed missing {len(missing)} 2023-24 RS dates: "
              f"{missing}")
        self.assertEqual(missing, [
            "2024-02-15", "2024-02-22", "2024-02-23", "2024-02-24",
            "2024-02-25", "2024-02-26", "2024-02-27", "2024-02-28",
            "2024-03-29"])

    def test_sign_convention(self):
        """Win_Margin must be home-minus-away: a team's post-game Elo rises
        after a home win in our replay (spot-check the first game)."""
        games = rf.load_odds_games(seasons=["2023-24"])
        g = games[0]
        key = (g["date"], g["home"], g["away"])
        pre = self.mine["pre_game"].get(key)
        post = self.mine["post_game"].get(key)
        if pre is None:
            self.skipTest("first 2023-24 game not in 2022-24 replay keys")
        if g["win_margin"] > 0:
            self.assertGreater(post["home_elo"], pre["home_elo"])
        else:
            self.assertLess(post["home_elo"], pre["home_elo"])


class TestWLExactness(unittest.TestCase):
    """Reconstructed per-game W/L (integer snapshot W diffs) must exactly
    match the odds tables' results. Sample seasons: 2015-16 and 2018-19
    (full 82-game seasons, no known bad rows)."""
    SEASONS = ("2015-16", "2018-19")

    @classmethod
    def setUpClass(cls):
        cls.store = rf.SnapshotStore()

    def _odds_results(self, season):
        """Per-team chronological (date, won) from the odds table."""
        res = {}
        for g in rf.load_odds_games(seasons=[season]):
            won_home = g["win_margin"] > 0
            res.setdefault(g["home"], []).append((g["date"], won_home))
            res.setdefault(g["away"], []).append((g["date"], not won_home))
        for team in res:
            res[team].sort()
        return res

    def test_wl_sequences_exact(self):
        for season in self.SEASONS:
            odds = self._odds_results(season)
            data = self.store._load_season(season)
            checked = 0
            for team, ts in data.items():
                # snapshot W sequence indexed by GP -> per-game results
                final_gp = ts.entries[-1].gp
                w_by_gp = {e.gp: e.wins for e in ts.entries}
                self.assertEqual(len(w_by_gp), final_gp,
                                 f"{season} {team}: snapshot GP gaps")
                prev_w = 0
                snap_seq = []
                for gp in range(1, final_gp + 1):
                    w = w_by_gp[gp]
                    snap_seq.append(w - prev_w == 1)
                    prev_w = w
                # first final_gp odds games are the regular season
                odds_seq = [won for _, won in odds[team][:final_gp]]
                self.assertEqual(snap_seq, odds_seq,
                                 f"{season} {team}: W/L mismatch")
                checked += final_gp
            self.assertGreaterEqual(checked, 30 * 82 - 5, season)
            print(f"\n  [W/L exactness] {season}: {checked} games exact")

    def test_rolling_win_pct_matches_direct(self):
        """Rolling win% (K=10) from the builder vs direct count over odds
        results, sampled at every 10th game-date of 2015-16."""
        season = "2015-16"
        odds = self._odds_results(season)
        for team, seq in odds.items():
            for i in range(10, min(len(seq), 82), 10):
                date = seq[i][0]
                feat = rf.build_rolling_features(10, date, team, self.store)
                if feat is None or feat["is_partial"] or feat["gp"] != i:
                    continue
                direct = sum(1 for _, w in seq[i - 10:i] if w) / 10
                self.assertAlmostEqual(feat["stats"]["WIN_PCT"], direct,
                                       places=12, msg=f"{team} {date}")


class TestRestFeatures(unittest.TestCase):
    """Synthetic-schedule unit tests for the unified rest convention."""

    @staticmethod
    def _game(date, home, away, season="2023-24"):
        return {"date": date, "home": home, "away": away, "season": season}

    def test_season_opener_is_seven(self):
        out = rf.build_rest_features([
            self._game("2024-01-01", "Boston Celtics", "Miami Heat")])
        r = out[("2024-01-01", "Boston Celtics", "Miami Heat")]
        self.assertEqual(r["home_rest"], 7)
        self.assertEqual(r["away_rest"], 7)
        self.assertFalse(r["home_b2b"])
        self.assertEqual(r["rest_diff"], 0)

    def test_back_to_back(self):
        games = [
            self._game("2024-01-01", "Boston Celtics", "Miami Heat"),
            self._game("2024-01-02", "Boston Celtics", "New York Knicks"),
        ]
        out = rf.build_rest_features(games)
        r = out[("2024-01-02", "Boston Celtics", "New York Knicks")]
        self.assertEqual(r["home_rest"], 1)
        self.assertTrue(r["home_b2b"])
        self.assertFalse(r["away_b2b"])
        self.assertEqual(r["away_rest"], 7)  # Knicks opener
        self.assertEqual(r["rest_diff"], -6)

    def test_rest_never_zero_and_min_one(self):
        games = [
            self._game("2024-01-01", "Boston Celtics", "Miami Heat"),
            self._game("2024-01-02", "Miami Heat", "Boston Celtics"),
        ]
        out = rf.build_rest_features(games)
        r = out[("2024-01-02", "Miami Heat", "Boston Celtics")]
        self.assertEqual(r["home_rest"], 1)
        self.assertEqual(r["away_rest"], 1)

    def test_three_in_four_boundaries(self):
        # Jan 1, 3, 4: the Jan 4 game is the 3rd in the window Jan 1..4.
        games = [
            self._game("2024-01-01", "Boston Celtics", "Miami Heat"),
            self._game("2024-01-03", "Boston Celtics", "Orlando Magic"),
            self._game("2024-01-04", "Boston Celtics", "Chicago Bulls"),
        ]
        out = rf.build_rest_features(games)
        self.assertFalse(
            out[("2024-01-03", "Boston Celtics", "Orlando Magic")]["home_3in4"])
        self.assertTrue(
            out[("2024-01-04", "Boston Celtics", "Chicago Bulls")]["home_3in4"])

    def test_three_in_four_window_excludes_day_minus_four(self):
        # Jan 1, 4, 5: window for Jan 5 is Jan 2..5 -> games on 4 and 5
        # only -> NOT a 3-in-4.
        games = [
            self._game("2024-01-01", "Boston Celtics", "Miami Heat"),
            self._game("2024-01-04", "Boston Celtics", "Orlando Magic"),
            self._game("2024-01-05", "Boston Celtics", "Chicago Bulls"),
        ]
        out = rf.build_rest_features(games)
        self.assertFalse(
            out[("2024-01-05", "Boston Celtics", "Chicago Bulls")]["home_3in4"])

    def test_all_star_gap_capped_at_seven(self):
        games = [
            self._game("2024-02-14", "Boston Celtics", "Miami Heat"),
            self._game("2024-02-23", "Boston Celtics", "Chicago Bulls"),
        ]
        out = rf.build_rest_features(games)
        r = out[("2024-02-23", "Boston Celtics", "Chicago Bulls")]
        self.assertEqual(r["home_rest"], 7)  # 9 days, capped
        self.assertFalse(r["home_b2b"])

    def test_season_boundary_resets(self):
        games = [
            self._game("2024-04-14", "Boston Celtics", "Miami Heat",
                       season="2023-24"),
            self._game("2024-10-22", "Boston Celtics", "Miami Heat",
                       season="2024-25"),
        ]
        out = rf.build_rest_features(games)
        r = out[("2024-10-22", "Boston Celtics", "Miami Heat")]
        self.assertEqual(r["home_rest"], 7)

    def test_matches_unified_convention_on_real_data_sample(self):
        """On 2023-24 odds games, rest must always be in [1, 7] and B2B
        only when the date diff is exactly 1."""
        games = rf.load_odds_games(seasons=["2023-24"])
        out = rf.build_rest_features(games)
        for v in out.values():
            self.assertGreaterEqual(v["home_rest"], 1)
            self.assertLessEqual(v["home_rest"], 7)
            self.assertGreaterEqual(v["away_rest"], 1)
            self.assertLessEqual(v["away_rest"], 7)


class TestLeakageGuard(unittest.TestCase):
    """A game's own result must not be able to enter its features."""

    @classmethod
    def setUpClass(cls):
        cls.store = rf.SnapshotStore()

    def test_snapshot_never_after_asof(self):
        """The builder reports which snapshots it used; they must be
        <= as_of_date. Checked over a full month of 2023-24 game-dates."""
        games = [g for g in rf.load_odds_games(seasons=["2023-24"])
                 if g["date"].startswith("2023-12")]
        self.assertGreater(len(games), 100)
        for g in games:
            for team in (g["home"], g["away"]):
                feat = rf.build_rolling_features(20, g["date"], team,
                                                 self.store)
                if feat is None:
                    continue
                self.assertLessEqual(feat["snapshot_date"], g["date"])
                if feat["prev_snapshot_date"]:
                    self.assertLessEqual(feat["prev_snapshot_date"],
                                         g["date"])

    def test_own_game_excluded(self):
        """Construct the case where snapshot D+1 (which contains the game
        on D) would change the feature, and verify the builder, asked
        as-of D, produces the D version -- i.e. it refuses the leaky
        snapshot. Uses every Boston Celtics 2023-24 game after GP=20."""
        team = "Boston Celtics"
        games = [g for g in rf.load_odds_games(seasons=["2023-24"])
                 if team in (g["home"], g["away"])]
        dates = sorted({g["date"] for g in games})
        changed = 0
        for date in dates[25:60]:
            day_after = (dt.date.fromisoformat(date)
                         + dt.timedelta(days=1)).isoformat()
            f_clean = rf.build_rolling_features(20, date, team, self.store)
            f_leaky = rf.build_rolling_features(20, day_after, team,
                                                self.store)
            if f_clean is None or f_leaky is None:
                continue
            self.assertLessEqual(f_clean["snapshot_date"], date)
            # The day-after build must have absorbed exactly one more game.
            self.assertEqual(f_leaky["gp"], f_clean["gp"] + 1)
            if any(abs(f_clean["stats"][s] - f_leaky["stats"][s]) > 1e-9
                   for s in rf.COUNT_STATS):
                changed += 1
        # The leaky snapshot demonstrably changes features, so the as-of
        # rule is load-bearing -- and f_clean never used it.
        self.assertGreater(changed, 20)

    def test_cache_refuses_protected_dbs(self):
        for name in ("TeamData.sqlite", "OddsData.sqlite", "dataset.sqlite"):
            with self.assertRaises(ValueError):
                rf._open_cache(os.path.join(REPO_ROOT, "Data", name))


if __name__ == "__main__":
    unittest.main()
