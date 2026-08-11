"""Tests for src/Predict/candidate_live.py — the live serving path of the
sealed 2026-08 candidate model.

The live module reuses the sealed harness (backtest_model.py) for loading,
Elo continuation, and calibration; those pieces carry the harness's own
assertions. What needs proving here is the live-only code:

  * rolling blocks built 'as of a date' reproduce the harness's
    rolling_from_box on every (team, date) it produced,
  * rest vectors reproduce the harness's rest_from_box on real games,
  * current Elo ratings equal each team's post-rating from its last played
    game (and the season reversion is applied exactly once),
  * end-to-end predict() returns calibrated probabilities in (0, 1) and
    matches raw-booster-through-isotonic arithmetic.

Run:  venv/Scripts/python.exe -m unittest Tests.Candidate_Live_Test -v
"""

import os
import sqlite3
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.Predict import candidate_live  # noqa: E402

_live = None


def setUpModule():
    global _live
    _live = candidate_live.get_candidate()
    if _live is None:
        raise unittest.SkipTest(
            f"candidate failed to load: {candidate_live._instance_error}")


def _live_frame_for(home_name, away_name):
    """Build the base-106 frame the way PredictionRunner does: latest
    TeamData snapshot row per team, away columns suffixed '.1', plus rest."""
    db = os.path.join(ROOT, 'Data', 'TeamData.sqlite')
    conn = sqlite3.connect(db)
    tbl = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' "
        "ORDER BY name DESC LIMIT 1").fetchone()[0]
    df = pd.read_sql_query(f"SELECT * FROM `{tbl}`", conn, index_col="index")
    conn.close()
    home = df[df['TEAM_NAME'] == home_name].iloc[0].copy()
    away = df[df['TEAM_NAME'] == away_name].iloc[0].copy()
    row = pd.concat([home, away.rename(index=lambda x: x + '.1')])
    row['Days-Rest-Home'] = 2.0
    row['Days-Rest-Away'] = 1.0
    return pd.DataFrame([row])


class ArtifactTest(unittest.TestCase):
    def test_artifact_loaded_with_assertions(self):
        # load_candidate already asserts manifest/booster/column-order
        # agreement; reaching here means they passed. Sanity-check shape.
        self.assertEqual(_live.booster.num_features(), 207)
        self.assertEqual(len(_live.cols), 207)
        self.assertEqual(_live.cols, _live.manifest['feature_columns'])


class RollingParityTest(unittest.TestCase):
    def test_rolling_parity_with_harness(self):
        """_rolling_blocks(as_of=D) must reproduce rolling_from_box for every
        (team, date) key the harness produced, on the most recent season."""
        season = sorted(_live.tg.season.unique())[-1]
        harness = _live.bm.rolling_from_box(_live.tg, _live.rf, season)
        checked = 0
        for (tid, date), blocks in harness.items():
            mine = _live._rolling_blocks(season, int(tid), date)
            if blocks is None:
                self.assertIsNone(mine, f"harness=None, live!=None for {tid} {date}")
                continue
            self.assertIsNotNone(mine, f"live=None, harness!=None for {tid} {date}")
            for k in _live.bm.ROLLING_KS:
                np.testing.assert_allclose(
                    mine[k], blocks[k][0], rtol=0, atol=1e-9, equal_nan=True,
                    err_msg=f"rolling mismatch team {tid} date {date} k={k}")
            checked += 1
        self.assertGreater(checked, 1000,
                           f"only {checked} keys checked — season data missing?")


class RestParityTest(unittest.TestCase):
    def test_rest_parity_with_harness(self):
        """_rest_vectors must reproduce rest_from_box for real games (the
        dedupe guard keeps ingested games from being double-counted)."""
        season = sorted(_live.tg.season.unique())[-1]
        harness = _live.bm.rest_from_box(_live.rf, _live.tg, [season])
        frame = _live.bm.box_game_frame(_live.tg[_live.tg.season == season])
        sample = frame.iloc[:: max(1, len(frame) // 200)]
        for g in sample.itertuples():
            home = _live.canon[int(g.home_id)]
            away = _live.canon[int(g.away_id)]
            mine = _live._rest_vectors(season, [(home, away)], g.game_date)[(home, away)]
            self.assertEqual(
                mine, harness[g.game_id],
                f"rest mismatch {home} vs {away} on {g.game_date}")


class EloStateTest(unittest.TestCase):
    def test_current_elo_is_last_post_rating(self):
        """Each team's serving rating equals the post-rating of its last
        played game in the continuation replay."""
        cont = _live.bm.continue_elo(_live.rf, _live.tg)
        games = cont['games']
        last_post = {}
        for g in games.itertuples():
            last_post[_live.canon[int(g.home_id)]] = cont['post'][g.game_id]['home_elo']
            last_post[_live.canon[int(g.away_id)]] = cont['post'][g.game_id]['away_elo']
        ratings = _live._current_elo(_live.last_played_season)
        for team, r in last_post.items():
            self.assertAlmostEqual(ratings[team], r, places=9,
                                   msg=f"Elo mismatch for {team}")

    def test_season_reversion_applied_once(self):
        """Ratings for a future season are 0.75*r + 0.25*1505, applied once
        even when asked repeatedly."""
        base = dict(_live._current_elo(_live.last_played_season))
        future = "2099-00"
        expected = {t: 0.75 * r + 0.25 * 1505.0 for t, r in base.items()}
        reverted = _live._current_elo(future)
        for t in expected:
            self.assertAlmostEqual(reverted[t], expected[t], places=9)
        again = _live._current_elo(future)
        for t in expected:
            self.assertAlmostEqual(again[t], expected[t], places=9,
                                   msg="reversion applied twice")
        # restore state for other tests
        _live._refresh_locked()


class PredictTest(unittest.TestCase):
    def tearDown(self):
        # predict() for a future season mutates Elo state; restore.
        _live._refresh_locked()

    def test_predict_end_to_end(self):
        frame = _live_frame_for("Boston Celtics", "New York Knicks")
        p = _live.predict(frame, [("Boston Celtics", "New York Knicks")],
                          ["2026-10-25"])
        self.assertEqual(p.shape, (1,))
        self.assertGreater(p[0], 0.0)
        self.assertLess(p[0], 1.0)

    def test_predict_matches_manual_isotonic(self):
        """predict() output equals raw booster prob passed through the
        stored isotonic (up to the serving clip)."""
        frame = _live_frame_for("Denver Nuggets", "Utah Jazz")
        games = [("Denver Nuggets", "Utah Jazz")]
        dates = ["2026-10-25"]
        p = _live.predict(frame, games, dates)
        _live._refresh_locked()

        base = frame[_live.bm.FEATURE_ORDER].astype(float).values
        home = _live.rf.normalize_team(games[0][0])
        away = _live.rf.normalize_team(games[0][1])
        hid, aid = _live.inv_canon[home], _live.inv_canon[away]
        season = candidate_live.season_for_date(dates[0])
        stat_n = len(_live.bm.roll_stat_cols(_live.rf))
        vec = [base[0]]
        for k in _live.bm.ROLLING_KS:
            for tid in (hid, aid):
                blocks = _live._rolling_blocks(season, tid, dates[0])
                vec.append(np.full(stat_n, np.nan) if blocks is None else blocks[k])
        elo = _live._current_elo(season)
        h_elo, a_elo = elo.get(home, 1500.0), elo.get(away, 1500.0)
        vec.append(np.array([h_elo, a_elo, h_elo - a_elo,
                             _live.rf._elo_expected_home(h_elo, a_elo)]))
        rest = _live._rest_vectors(season, [(home, away)], dates[0])
        vec.append(np.array(rest[(home, away)], dtype=float))
        X = np.vstack([np.concatenate(vec)])
        p_raw = _live.bm.predict_candidate(_live.booster, X, _live.cols)
        p_manual = np.clip(_live.bm.apply_isotonic(p_raw, _live._iso),
                           1e-3, 1 - 1e-3)
        np.testing.assert_allclose(p, p_manual, atol=1e-12)

    def test_unknown_team_raises(self):
        frame = _live_frame_for("Boston Celtics", "New York Knicks")
        with self.assertRaises(Exception):
            _live.predict(frame, [("Springfield Ghosts", "New York Knicks")],
                          ["2026-10-25"])


class SeasonForDateTest(unittest.TestCase):
    def test_season_for_date(self):
        self.assertEqual(candidate_live.season_for_date("2026-10-25"), "2026-27")
        self.assertEqual(candidate_live.season_for_date("2027-02-14"), "2026-27")
        self.assertEqual(candidate_live.season_for_date("2026-06-15"), "2025-26")
        self.assertEqual(candidate_live.season_for_date("2026-08-11"), "2026-27")


if __name__ == '__main__':
    unittest.main()
