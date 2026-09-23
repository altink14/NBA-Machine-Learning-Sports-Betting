"""Live serving of the retrained candidate model (Models/candidate_2026-08).

The candidate was evaluated once, sealed, on 2024-26 (67.15% [65.32-68.93],
n=2,597) by backtest_model.py. This module serves that exact artifact for
today's games by REUSING the harness's own functions -- model loading with
its hard column-order assertions, the Elo continuation replay, the stored
isotonic calibrator -- so the live path cannot silently drift from what was
measured.

The only live-specific code here is building features for games that have
not been played yet (today's slate):

  base 106   -- taken from the same TeamData snapshot frame the production
                model uses, reordered to the manifest's column order,
  rolling 92 -- same cumulative-sum window math as the harness's
                rolling_from_box, evaluated as of today (strictly earlier
                game dates only; season openers get NaN blocks, which
                XGBoost handles as missing),
  elo 4      -- current pre-game ratings: the harness's continuation replay
                through the last played game, plus the standard 25%-to-1505
                between-season reversion when today's season has no games
                in box_scores yet,
  rest 5     -- retrain_features.build_rest_features over the current
                season's real game dates with today's matchups appended.

Every prediction is raw booster output passed through the STORED isotonic
calibrator, exactly as the sealed evaluation applied it. No power-rating
blending is applied to these probabilities anywhere -- serving must match
what was measured.

Fail-closed: any assertion or data problem raises; the caller falls back to
the old production model and says so in the response's `model` field.
"""

from __future__ import annotations

import bisect
import importlib.util
import logging
import os
import sqlite3
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKTEST_PY = os.path.join(_ROOT, 'backtest_model.py')
_TEAMDATA_RO = "file:" + os.path.join(_ROOT, "Data", "TeamData.sqlite").replace("\\", "/") + "?mode=ro"


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def season_for_date(date_str: str) -> str:
    """NBA season a date belongs to: Aug-Dec -> YYYY-(YY+1), Jan-Jul -> (YYYY-1)-YY."""
    y, m = int(date_str[:4]), int(date_str[5:7])
    start = y if m >= 8 else y - 1
    return f"{start}-{str(start + 1)[-2:]}"


class CandidateLive:
    """Loads the sealed candidate artifact and serves calibrated home-win
    probabilities for today's games. Thread-safe; refreshes its box-score
    state when new games appear in the database."""

    def __init__(self):
        self._lock = threading.Lock()
        self.bm = _load_module("bb_backtest_model", _BACKTEST_PY)
        self.rf = self.bm.load_rf()
        self.booster, self.manifest, self.config = self.bm.load_candidate(self.rf)
        self.cols = self.bm.candidate_column_order(self.rf)
        self.n_base = len(self.bm.FEATURE_ORDER)
        self._iso = self.config['calibrator']['isotonic']
        self._tg_max_date: Optional[str] = None
        self._refresh_locked()

    # ------------------------------------------------------------------ state

    def _refresh_locked(self) -> None:
        con = sqlite3.connect(_TEAMDATA_RO, uri=True)
        try:
            tg = self.bm.load_team_games(con)
        finally:
            con.close()
        canon = self.bm.team_canonical_map(tg, self.rf)
        base_elo = self.rf.build_elo_odds()
        cont = self.bm.continue_elo(self.rf, tg, base_elo=base_elo)

        # Current ratings entering the NEXT game: base final ratings advanced
        # by each continuation game's post-game rating, in replay order.
        ratings = dict(base_elo['final_ratings'])
        games = cont['games']
        for g in games.itertuples():
            ratings[canon[int(g.home_id)]] = cont['post'][g.game_id]['home_elo']
            ratings[canon[int(g.away_id)]] = cont['post'][g.game_id]['away_elo']

        self.tg = tg
        self.canon = canon
        self.inv_canon: Dict[str, int] = {v: k for k, v in canon.items()}
        self.ratings = ratings
        self.last_played_season: Optional[str] = (
            games.season.iloc[-1] if len(games) else '2023-24')
        self._reverted_for: Optional[str] = None
        self._tg_max_date = str(tg.game_date.max())
        logger.info(
            "CandidateLive state: %d team-game rows through %s; Elo continuation %d games",
            len(tg), self._tg_max_date, cont['n_continuation_games'])

    def _maybe_refresh(self) -> None:
        con = sqlite3.connect(_TEAMDATA_RO, uri=True)
        try:
            row = con.execute("SELECT MAX(game_date) FROM box_scores").fetchone()
        finally:
            con.close()
        if row and row[0] and str(row[0]) != self._tg_max_date:
            logger.info("box_scores advanced (%s -> %s); rebuilding candidate state",
                        self._tg_max_date, row[0])
            self._refresh_locked()

    # ------------------------------------------------------------- components

    def _current_elo(self, game_season: str) -> Dict[str, float]:
        """Ratings entering a game in `game_season`, applying the standard
        between-season reversion once if the season has no played games yet."""
        if game_season == self.last_played_season:
            return self.ratings
        if self._reverted_for != game_season:
            rf = self.rf
            self.ratings = {
                t: rf.ELO_SEASON_CARRYOVER * r
                   + (1.0 - rf.ELO_SEASON_CARRYOVER) * rf.ELO_MEAN_REVERT_TARGET
                for t, r in self.ratings.items()}
            self._reverted_for = game_season
            self.last_played_season = game_season
        return self.ratings

    def _rolling_blocks(self, season: str, team_id: int, as_of_date: str):
        """R{10,20} stat blocks for `team_id` as of `as_of_date` (strictly
        earlier regular-season games only). Mirrors the harness's
        rolling_from_box math exactly: per-game means over the window from
        cumulative sums, MIN/5, recomputed shooting percentages from window
        totals, WIN_PCT from window wins. Returns {k: vector} or None when
        the team has no prior games this season."""
        rf, tg = self.rf, self.tg
        stat_names = list(rf.COUNT_STATS)
        reg = tg[(tg.season == season) & (tg.season_type == 'Regular Season')
                 & (tg.team_id == team_id)].sort_values(['game_date', 'game_id'])
        if reg.empty:
            return None
        dates = reg.game_date.tolist()
        idx = bisect.bisect_left(dates, as_of_date)
        if idx == 0:
            return None
        cols = []
        for s in stat_names:
            v = reg[s].values.astype(float)
            if s == 'MIN':
                v = v / 5.0
            cols.append(v)
        cols.append(reg.W.values.astype(float))
        arr = np.column_stack(cols)
        cums = np.vstack([np.zeros(arr.shape[1]), np.cumsum(arr, axis=0)])

        i_fgm, i_fga = stat_names.index('FGM'), stat_names.index('FGA')
        i_f3m, i_f3a = stat_names.index('FG3M'), stat_names.index('FG3A')
        i_ftm, i_fta = stat_names.index('FTM'), stat_names.index('FTA')
        i_w = len(stat_names)

        blocks = {}
        for k in self.bm.ROLLING_KS:
            w = min(k, idx)
            tot = cums[idx] - cums[idx - w]
            means = tot[:len(stat_names)] / w

            def _pct(n, d):
                return (tot[n] / tot[d]) if tot[d] else np.nan

            blocks[k] = np.concatenate([
                means,
                [_pct(i_fgm, i_fga), _pct(i_f3m, i_f3a), _pct(i_ftm, i_fta),
                 tot[i_w] / w]])
        return blocks

    def _rest_vectors(self, season: str, matchups: List[Tuple[str, str]],
                      game_date: str) -> Dict[Tuple[str, str], Tuple[float, ...]]:
        """(b2b_h, b2b_a, 3in4_h, 3in4_a, rest_diff) per canonical matchup,
        from the season's real game dates with today's slate appended."""
        rf = self.rf
        season_games = self.tg[self.tg.season == season]
        glist = []
        frame = self.bm.box_game_frame(season_games) if len(season_games) else None
        seen = set()
        if frame is not None:
            for g in frame.itertuples():
                row = dict(date=g.game_date,
                           home=self.canon[int(g.home_id)],
                           away=self.canon[int(g.away_id)],
                           season=season)
                glist.append(row)
                seen.add((row['date'], row['home'], row['away']))
        for home, away in matchups:
            # A matchup already in box_scores (re-request after the game was
            # ingested) must not be appended twice — a duplicate date would
            # inflate the 3-in-4 counts.
            if (game_date, home, away) not in seen:
                glist.append(dict(date=game_date, home=home, away=away, season=season))
        rest = rf.build_rest_features(glist)
        out = {}
        for home, away in matchups:
            v = rest[(game_date, home, away)]
            out[(home, away)] = (float(v['home_b2b']), float(v['away_b2b']),
                                 float(v['home_3in4']), float(v['away_3in4']),
                                 float(v['rest_diff']))
        return out

    # -------------------------------------------------------------- interface

    def predict(self, frame_ml: pd.DataFrame,
                games: List[Tuple[str, str]],
                game_dates: List[str]) -> np.ndarray:
        """Calibrated home-win probability per game.

        `frame_ml`: the PredictionRunner's per-game frame (home snapshot cols
        + away '.1' cols + Days-Rest-Home/Away), one row per game, in the
        same order as `games`. `game_dates`: ISO dates (US Eastern) per game.
        Raises on any inconsistency -- the caller falls back to the old model.
        """
        with self._lock:
            self._maybe_refresh()
            X = self._build_matrix(frame_ml, games, game_dates)
            return self._calibrated(X)

    def predict_explained(self, frame_ml: pd.DataFrame,
                          games: List[Tuple[str, str]],
                          game_dates: List[str]) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        """predict(), plus what moved each prediction, from the booster itself.

        The probabilities are computed by exactly the same code as predict()
        on exactly the same feature matrix, so asking for reasons cannot
        change a pick. The reasons are XGBoost's own per-feature contributions
        (pred_contribs) for the home-win class, summed into plain groups by
        src/Utils/pick_reasons.py. They explain the RAW booster score; the
        stored isotonic calibrator then maps that score to the published
        probability, which the note on every explanation says.
        """
        from src.Utils import pick_reasons
        with self._lock:
            self._maybe_refresh()
            X = self._build_matrix(frame_ml, games, game_dates)
            p_cal = self._calibrated(X)
            d = self.bm.xgb.DMatrix(X, feature_names=self.cols)
            contribs = self.booster.predict(d, pred_contribs=True)
            margin = self.booster.predict(d, output_margin=True)
        if contribs.ndim == 3:
            # Two-class softmax: p(home) = sigmoid(margin_home - margin_away),
            # so a feature's push toward home is its class-1 minus class-0 part.
            delta = contribs[:, 1, :] - contribs[:, 0, :]
            logit = margin[:, 1] - margin[:, 0]
        else:
            delta, logit = contribs, margin
        # The contributions must add up to the booster's own score, or they
        # are not an explanation of it.
        assert np.allclose(delta.sum(axis=1), logit, atol=1e-3), "contributions do not sum to the margin"
        reasons = []
        for i, (home_name, away_name) in enumerate(games):
            reasons.append(pick_reasons.group_contributions(
                self.cols, delta[i], X[i], self.n_base, home_name, away_name))
        return p_cal, reasons

    def _calibrated(self, X: np.ndarray) -> np.ndarray:
        p_raw = self.bm.predict_candidate(self.booster, X, self.cols)
        p_cal = self.bm.apply_isotonic(p_raw, self._iso)
        # The isotonic can output exactly 0/1 at its extremes; clip a hair
        # so EV / Kelly arithmetic downstream stays finite.
        return np.clip(p_cal, 1e-3, 1.0 - 1e-3)

    def _build_matrix(self, frame_ml: pd.DataFrame,
                      games: List[Tuple[str, str]],
                      game_dates: List[str]) -> np.ndarray:
        """The 207-column feature matrix, one row per game. Caller holds the lock."""
        missing = [c for c in self.bm.FEATURE_ORDER if c not in frame_ml.columns]
        assert not missing, f"base feature columns missing from live frame: {missing[:5]}"
        base = frame_ml[self.bm.FEATURE_ORDER].astype(float).values
        assert base.shape == (len(games), self.n_base)

        stat_n = len(self.bm.roll_stat_cols(self.rf))
        nan_block = np.full(stat_n, np.nan)

        rows = []
        for i, (home_name, away_name) in enumerate(games):
            date = game_dates[i]
            season = season_for_date(date)
            home = self.rf.normalize_team(home_name)
            away = self.rf.normalize_team(away_name)
            assert home in self.inv_canon, f"unknown home team: {home_name!r} -> {home!r}"
            assert away in self.inv_canon, f"unknown away team: {away_name!r} -> {away!r}"
            hid, aid = self.inv_canon[home], self.inv_canon[away]

            vec = [base[i]]
            for k in self.bm.ROLLING_KS:
                for tid in (hid, aid):
                    blocks = self._rolling_blocks(season, tid, date)
                    vec.append(nan_block if blocks is None else blocks[k])

            elo = self._current_elo(season)
            h_elo = elo.get(home, self.rf.ELO_BASE)
            a_elo = elo.get(away, self.rf.ELO_BASE)
            vec.append(np.array([h_elo, a_elo, h_elo - a_elo,
                                 self.rf._elo_expected_home(h_elo, a_elo)]))

            rest = self._rest_vectors(season, [(home, away)], date)
            vec.append(np.array(rest[(home, away)], dtype=float))

            row = np.concatenate(vec)
            assert row.shape[0] == len(self.cols)
            rows.append(row)

        return np.vstack(rows)


_instance: Optional[CandidateLive] = None
_instance_lock = threading.Lock()
_instance_error: Optional[str] = None

MODEL_TAG = "xgboost_cand_2026-08"


def status() -> Dict[str, object]:
    """What the model is doing right now, WITHOUT triggering a load.

    A cold load walks every box score in the archive and takes about a minute,
    so a health check must never call get_candidate(). This reports only what
    is already known, which is what a health check should do anyway: say what
    is true, not go and make it true.

    `loaded` false with `error` None means nobody has asked yet. `loaded`
    false with an `error` means every prediction is being served by the old
    model, and the published accuracy figure does not describe it.
    """
    return {
        "loaded": _instance is not None,
        "tag": MODEL_TAG if _instance is not None else None,
        "error": _instance_error,
        "serving": ("candidate" if _instance is not None
                    else "old model fallback" if _instance_error
                    else "not yet loaded"),
    }


def warm() -> bool:
    """Load the model now. Returns whether it came up.

    Called at startup so a deployed container pays the ~55s cold load during
    boot rather than on the first user's request, which most gateways would
    time out. Safe to call more than once; the singleton handles that.
    """
    return get_candidate() is not None


def get_candidate() -> Optional[CandidateLive]:
    """Singleton accessor. Returns None (and remembers why) if the artifact
    or its supporting data can't be loaded -- callers then use the old model."""
    global _instance, _instance_error
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None and _instance_error is None:
            try:
                _instance = CandidateLive()
            except Exception as exc:
                _instance_error = str(exc)
                logger.error("Candidate model unavailable, will serve old model: %s",
                             exc, exc_info=True)
    return _instance
