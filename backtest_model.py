"""
Honest out-of-sample backtest of the production XGBoost moneyline model.

WHAT THIS DOES
--------------
The production model (Models/XGBoost_Models/XGBoost_68.9%_ML-3.json) was trained by
src/Train-Models/XGBoost_Model_ML.py on the table `dataset_2012-24_new` in
Data/dataset.sqlite -- 15,115 games from 2012-11-04 through 2024-04-28. The "68.9%"
in the filename is the *maximum* test accuracy over 300 random train_test_split runs,
i.e. a max over 300 noisy estimates using random (not chronological) splits. That is
an optimistically biased number and is not an out-of-sample estimate.

This script measures the model on data it has never seen: the 2024-25 and 2025-26
seasons, whose box scores live in Data/TeamData.sqlite.

Those seasons have no ready-made feature rows, because the as-of-date team-stat
snapshots the training rows were built from (the ~3,996 date-keyed tables in
TeamData.sqlite, produced by src/Process-Data/Get_Data.py hitting
stats.nba.com/leaguedashteamstats) stop on 2024-04-29. So we reconstruct those
snapshots locally from `box_scores`, and validate the reconstruction against the
real snapshots on seasons where both exist (2022-23 and 2023-24).

NO-LEAKAGE RULE
---------------
Get_Data.py fetched leaguedashteamstats with DateTo = D and then stored the result
under the table name D+1. So table "2024-01-15" holds cumulative stats through games
played on 2024-01-14. Create_Games.py reads the table named for the game's own date.
Features for a game on date D therefore use only games played strictly before D.
This script reproduces that exactly: `game_date < as_of`.

The snapshots are also SeasonType=Regular+Season with DateFrom=10/01/<start year>,
so they are season-to-date REGULAR SEASON totals only; during the playoffs they are
frozen at the final regular-season numbers. Reconstruction matches that.

USAGE
-----
    venv/Scripts/python.exe backtest_model.py --validate    # reconstruction check only
    venv/Scripts/python.exe backtest_model.py               # validate + backtest + write JSON

Deterministic: no randomness anywhere. Read-only on every database.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.abspath(__file__))
TEAMDATA = "file:" + os.path.join(ROOT, "Data", "TeamData.sqlite").replace("\\", "/") + "?mode=ro"
DATASET = "file:" + os.path.join(ROOT, "Data", "dataset.sqlite").replace("\\", "/") + "?mode=ro"
ODDSDATA = "file:" + os.path.join(ROOT, "Data", "OddsData.sqlite").replace("\\", "/") + "?mode=ro"
ML_MODEL = os.path.join(ROOT, "Models", "XGBoost_Models", "XGBoost_68.9%_ML-3.json")
OUT_JSON = os.path.join(ROOT, "backtest_results.json")

TRAIN_TABLE = "dataset_2012-24_new"
HELD_OUT_SEASONS = ["2024-25", "2025-26"]
VALIDATION_SEASONS = ["2022-23", "2023-24"]

# ---------------------------------------------------------------------------
# Column order. This is the exact order `dataset_2012-24_new` has after the
# training script's drops, and it is what the 106-feature booster expects.
# ---------------------------------------------------------------------------
BASE = ['GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
        'PF', 'PFD', 'PTS', 'PLUS_MINUS']
RANKS = [c + '_RANK' for c in BASE]
TEAM_BLOCK = BASE + RANKS                     # 52 columns per team
FEATURE_ORDER = TEAM_BLOCK + [c + '.1' for c in TEAM_BLOCK] + ['Days-Rest-Home', 'Days-Rest-Away']

# NBA leaguedash ranks: 1 = best. For these four, lower raw value is better.
RANK_ASCENDING = {'L', 'TOV', 'PF', 'BLKA'}

PER_GAME = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
            'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
SUM_COLS = PER_GAME + ['MIN', 'W', 'L']


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def hround(x, nd):
    """Half-up rounding, which is what the NBA stats API display uses."""
    f = 10.0 ** nd
    return np.floor(np.asarray(x, dtype=float) * f + 0.5) / f


def parse_minutes(s):
    if isinstance(s, (int, float)):
        return float(s)
    if ':' in s:
        a, b = s.split(':')[:2]
        return float(a) + float(b) / 60.0
    return float(s)


def load_team_games(con) -> pd.DataFrame:
    """One row per (game, team) with the counting stats leaguedashteamstats reports.

    Two stats are not directly in the traditional team box score:
      * BLKA (blocked attempts)  = the opponent's BLK.
      * PFD  (personal fouls drawn) = the opponent's PF.
    And one needs correcting:
      * The traditional box score's team `turnovers` is the sum over players and
        omits team turnovers (shot-clock violations etc.), running ~0.7/game low
        versus the league dashboard. We recover the official team total from the
        advanced box score: estimatedTeamTurnoverPercentage * possessions / 100.
    """
    df = pd.read_sql_query(
        "select game_id, season, season_type, game_date, traditional_json, advanced_json "
        "from box_scores", con)
    rows = []
    for r in df.itertuples():
        t = json.loads(r.traditional_json)['boxScoreTraditional']
        a = json.loads(r.advanced_json)['boxScoreAdvanced']
        for side, opp in (('homeTeam', 'awayTeam'), ('awayTeam', 'homeTeam')):
            s = t[side]['statistics']
            o = t[opp]['statistics']
            adv = a[side]['statistics']
            tov = int(np.floor(adv['estimatedTeamTurnoverPercentage'] * adv['possessions'] / 100.0 + 0.5))
            rows.append(dict(
                game_id=r.game_id, season=r.season, season_type=r.season_type,
                game_date=r.game_date, team_id=t[side]['teamId'],
                team_name=f"{t[side]['teamCity']} {t[side]['teamName']}",
                is_home=(side == 'homeTeam'),
                MIN=parse_minutes(s['minutes']),
                FGM=s['fieldGoalsMade'], FGA=s['fieldGoalsAttempted'],
                FG3M=s['threePointersMade'], FG3A=s['threePointersAttempted'],
                FTM=s['freeThrowsMade'], FTA=s['freeThrowsAttempted'],
                OREB=s['reboundsOffensive'], DREB=s['reboundsDefensive'], REB=s['reboundsTotal'],
                AST=s['assists'], TOV=tov, STL=s['steals'], BLK=s['blocks'], BLKA=o['blocks'],
                PF=s['foulsPersonal'], PFD=o['foulsPersonal'],
                PTS=s['points'], OPP_PTS=o['points']))
    tg = pd.DataFrame(rows)
    tg['W'] = (tg.PTS > tg.OPP_PTS).astype(int)
    tg['L'] = 1 - tg.W
    tg['PLUS_MINUS'] = tg.PTS - tg.OPP_PTS
    return tg.sort_values(['game_date', 'game_id']).reset_index(drop=True)


def build_snapshots(tg: pd.DataFrame, season: str, leak: bool = False) -> dict:
    """as-of-date snapshot for every date on which that season had a game.

    Returns {date_str: DataFrame indexed by team_id with the 52 model columns},
    where the snapshot for date D covers only regular-season games before D.

    `leak=True` deliberately includes same-day games. It is never used for the reported
    result -- it exists as a positive control, to show the measurement would visibly
    change if the strict cutoff were not being applied.
    """
    reg = tg[(tg.season == season) & (tg.season_type == 'Regular Season')]
    all_dates = sorted(tg[tg.season == season].game_date.unique())
    snaps = {}
    for d in all_dates:
        sub = reg[reg.game_date <= d] if leak else reg[reg.game_date < d]
        if sub.empty:
            snaps[d] = None
            continue
        g = sub.groupby('team_id')[SUM_COLS].sum()
        g['GP'] = sub.groupby('team_id').size()
        raw = pd.DataFrame(index=g.index)
        raw['GP'] = g.GP.astype(float)
        raw['W'] = g.W.astype(float)
        raw['L'] = g.L.astype(float)
        raw['W_PCT'] = g.W / g.GP
        raw['MIN'] = g.MIN / 5.0 / g.GP          # team minutes -> the API's 48.x per game
        for c in PER_GAME:
            raw[c] = g[c] / g.GP
        raw['FG_PCT'] = g.FGM / g.FGA
        raw['FG3_PCT'] = g.FG3M / g.FG3A
        raw['FT_PCT'] = g.FTM / g.FTA
        raw = raw[BASE]

        out = pd.DataFrame(index=raw.index)
        # Ranks are computed on the UNROUNDED values, ties share the best rank.
        for c in BASE:
            out[c + '_RANK'] = raw[c].rank(ascending=(c in RANK_ASCENDING), method='min').astype(float)
        # Displayed values are rounded the way the API rounds them.
        for c in BASE:
            nd = 3 if c.endswith('_PCT') else (0 if c in ('GP', 'W', 'L') else 1)
            out[c] = hround(raw[c], nd)
        snaps[d] = out[TEAM_BLOCK]
    return snaps


def days_rest_table(tg: pd.DataFrame, season: str, convention: str = 'training') -> dict:
    """{(game_id, team_id): days_rest} replicating the training-time computation.

    Training convention (src/Process-Data/Get_Odds_Data.py, used for the most recent
    training season 2023-24): plain (game_date - previous_game_date).days, 7 for a
    team's first game of the season, no clamping.
    'legacy'     -- src/Process-Data/Add_Days_Rest.py, used for 2012-13..2022-23:
                    same but 10 for the first game and anything outside (0, 9) -> 9.
    'production' -- the training difference then clipped to [1, 7], which is what the
                    corrected main_api.py path produces. (The pre-fix code computed a
                    naive-local-vs-UTC datetime difference plus one day; those two errors
                    cancel during the daytime call window, so it agreed with 'training'
                    97.6% of the time and its only systematic deviation was this clip.)
    """
    sub = tg[tg.season == season].sort_values(['game_date', 'game_id'])
    last = {}
    out = {}
    for r in sub.itertuples():
        d = datetime.strptime(r.game_date, '%Y-%m-%d').date()
        prev = last.get(r.team_id)
        if prev is None:
            val = {'training': 7, 'legacy': 10, 'production': 7}[convention]
        else:
            diff = (d - prev).days
            if convention == 'training':
                val = diff
            elif convention == 'legacy':
                val = diff if 0 < diff < 9 else 9
            else:
                val = max(1, min(diff, 7))
        last[r.team_id] = d
        out[(r.game_id, r.team_id)] = float(val)
    return out


def build_games(tg: pd.DataFrame, season: str, convention: str = 'training', leak: bool = False):
    """Feature matrix + labels for every game of `season`."""
    snaps = build_snapshots(tg, season, leak=leak)
    rest = days_rest_table(tg, season, convention)
    games = tg[tg.season == season].drop_duplicates('game_id')[
        ['game_id', 'game_date', 'season_type']].sort_values(['game_date', 'game_id'])
    by_game = {gid: grp for gid, grp in tg[tg.season == season].groupby('game_id')}

    X, meta, skipped = [], [], []
    for r in games.itertuples():
        snap = snaps.get(r.game_date)
        grp = by_game[r.game_id]
        home = grp[grp.is_home].iloc[0]
        away = grp[~grp.is_home].iloc[0]
        if snap is None or home.team_id not in snap.index or away.team_id not in snap.index:
            skipped.append((r.game_id, r.game_date, 'no prior-game snapshot for one/both teams'))
            continue
        hs, as_ = snap.loc[home.team_id], snap.loc[away.team_id]
        vec = np.concatenate([
            hs.values, as_.values,
            [rest[(r.game_id, home.team_id)], rest[(r.game_id, away.team_id)]]])
        X.append(vec.astype(float))
        meta.append(dict(game_id=r.game_id, game_date=r.game_date, season=season,
                         season_type=r.season_type,
                         home_team=home.team_name, away_team=away.team_name,
                         home_id=int(home.team_id), away_id=int(away.team_id),
                         home_pts=int(home.PTS), away_pts=int(away.PTS),
                         home_win=int(home.PTS > away.PTS),
                         total_points=int(home.PTS) + int(away.PTS),
                         # naive baselines, from the same pre-game snapshot the model saw
                         home_wpct=float(hs['W_PCT']), away_wpct=float(as_['W_PCT']),
                         home_pm=float(hs['PLUS_MINUS']), away_pm=float(as_['PLUS_MINUS'])))
    return np.asarray(X, dtype=float), pd.DataFrame(meta), skipped


# ---------------------------------------------------------------------------
# validation of the reconstruction
# ---------------------------------------------------------------------------
TRAIN_DROPS = ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1',
               'OU-Cover', 'OU']


def assert_feature_order(con_ds):
    """Hard check: FEATURE_ORDER must be exactly the column order the training script fed
    to xgboost, i.e. dataset_2012-24_new minus TRAIN_DROPS, in table order."""
    head = pd.read_sql_query(f'select * from "{TRAIN_TABLE}" limit 1', con_ds, index_col='index')
    expected = [c for c in head.columns if c not in TRAIN_DROPS]
    if expected != FEATURE_ORDER:
        raise AssertionError(
            'FEATURE_ORDER does not match the training table.\n'
            f'  first difference at position '
            f'{next(i for i, (a, b) in enumerate(zip(expected, FEATURE_ORDER)) if a != b)}\n'
            f'  table: {expected}\n  script: {FEATURE_ORDER}')
    return expected


def in_sample_sanity(con_ds, booster):
    """Score the model on its own training table. If the feature order were wrong this
    would collapse toward 50%; a high number confirms the vector is being assembled the
    way the model expects. It is NOT an accuracy claim -- it is in-sample."""
    ds = pd.read_sql_query(f'select * from "{TRAIN_TABLE}"', con_ds, index_col='index')
    X = ds[FEATURE_ORDER].astype(float).values
    y = ds['Home-Team-Win'].astype(int).values
    p = booster.predict(xgb.DMatrix(X))[:, 1]
    acc = float(np.mean((p > 0.5).astype(int) == y))
    shuf = np.concatenate([X[:, 1:], X[:, :1]], axis=1)   # order-sensitivity control
    ps = booster.predict(xgb.DMatrix(shuf))[:, 1]
    return dict(
        n=int(len(y)),
        in_sample_accuracy_pct=round(acc * 100, 2),
        home_win_rate_in_training_pct=round(float(np.mean(y)) * 100, 2),
        control_accuracy_with_shifted_columns_pct=round(
            float(np.mean((ps > 0.5).astype(int) == y)) * 100, 2),
        note=('In-sample only -- the model was fit on these rows. Reported purely to prove the '
              '106-feature vector is assembled in the order the booster expects. The control '
              'row shifts every column by one position; it collapses toward the base rate, '
              'which is what makes the check meaningful.'),
    )


def validate_snapshots(tg, con_team, seasons):
    """Cell-by-cell comparison of reconstructed snapshots vs the real date-keyed tables."""
    existing = set(pd.read_sql_query(
        "select name from sqlite_master where type='table'", con_team)['name'])
    per_col = defaultdict(lambda: [0, 0])
    n_dates = 0
    for season in seasons:
        snaps = build_snapshots(tg, season)
        for d, rec in sorted(snaps.items()):
            if rec is None or d not in existing:
                continue
            orig = pd.read_sql_query(f'select * from "{d}"', con_team, index_col='index')
            if len(orig) != 30 or len(rec) != 30:
                continue
            orig = orig.set_index('TEAM_ID')
            common = rec.index.intersection(orig.index)
            if len(common) != 30:
                continue
            n_dates += 1
            for c in TEAM_BLOCK:
                a = orig.loc[common, c].astype(float).values
                b = rec.loc[common, c].astype(float).values
                per_col[c][0] += int(np.sum(np.abs(a - b) < 1e-9))
                per_col[c][1] += 30
    return n_dates, {c: per_col[c] for c in TEAM_BLOCK}


def validate_feature_rows(tg, con_team, con_ds, season):
    """End-to-end: rebuild the 106-d vector for games that are IN the training table
    and compare against the stored row. Also compares model output on both."""
    ds = pd.read_sql_query(f'select * from "{TRAIN_TABLE}"', con_ds, index_col='index')
    ds = ds[ds.Date.astype(str).str.slice(0, 10).between(
        '2023-10-01' if season == '2023-24' else '2022-10-01',
        '2024-09-30' if season == '2023-24' else '2023-09-30')]
    if ds.empty:
        return None
    X, meta, _ = build_games(tg, season, convention='training')
    if len(meta) == 0:
        return None
    rec = {(m.game_date, m.home_team, m.away_team): i for i, m in enumerate(meta.itertuples())}

    stored, rebuilt = [], []
    unmatched = 0
    feat = ds[FEATURE_ORDER].astype(float)
    for idx, d, hn, an in zip(ds.index, ds['Date'].astype(str),
                              ds['TEAM_NAME'], ds['TEAM_NAME.1']):
        i = rec.get((d[:10], hn, an))
        if i is None:
            unmatched += 1
            continue
        stored.append(feat.loc[idx].values)
        rebuilt.append(X[i])
    if not stored:
        return None
    S = np.asarray(stored, dtype=float)
    R = np.asarray(rebuilt, dtype=float)
    diff = np.abs(S - R)

    stat_idx = [FEATURE_ORDER.index(c) for c in FEATURE_ORDER
                if c not in ('Days-Rest-Home', 'Days-Rest-Away')]
    rest_idx = [FEATURE_ORDER.index('Days-Rest-Home'), FEATURE_ORDER.index('Days-Rest-Away')]

    booster = load_model()
    ps = booster.predict(xgb.DMatrix(S))[:, 1]
    pr = booster.predict(xgb.DMatrix(R))[:, 1]
    stored_pick = (ps > 0.5).astype(int)
    rebuilt_pick = (pr > 0.5).astype(int)

    return dict(
        season=season,
        n_training_rows_in_season=int(len(ds)),
        n_matched_to_box_scores=int(len(S)),
        n_unmatched=int(unmatched),
        stat_features_exact_pct=float(np.mean(diff[:, stat_idx] < 1e-9) * 100),
        stat_features_within_0p1_pct=float(np.mean(diff[:, stat_idx] <= 0.1 + 1e-9) * 100),
        stat_features_within_1_rank_pct=float(np.mean(diff[:, stat_idx] <= 1.0 + 1e-9) * 100),
        rows_with_all_stat_features_exact_pct=float(
            np.mean(np.all(diff[:, stat_idx] < 1e-9, axis=1)) * 100),
        days_rest_exact_pct=float(np.mean(diff[:, rest_idx] < 1e-9) * 100),
        max_abs_stat_diff=float(diff[:, stat_idx].max()),
        mean_abs_prob_diff=float(np.mean(np.abs(ps - pr))),
        max_abs_prob_diff=float(np.max(np.abs(ps - pr))),
        same_pick_pct=float(np.mean(stored_pick == rebuilt_pick) * 100),
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def load_model():
    b = xgb.Booster()
    b.load_model(ML_MODEL)
    assert b.num_features() == 106, f"expected 106 features, booster wants {b.num_features()}"
    assert len(FEATURE_ORDER) == 106
    return b


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h) * 100, 2), round(float(c + h) * 100, 2)]


def mcnemar(df, col):
    """Paired test: is the model better than this baseline on the SAME games?
    Exact binomial two-sided p on the discordant pairs."""
    from math import comb
    model_ok = (df.pred == df.home_win).values
    base_ok = (df[col] == df.home_win).values
    b = int(np.sum(model_ok & ~base_ok))   # model right, baseline wrong
    c = int(np.sum(~model_ok & base_ok))   # baseline right, model wrong
    n = b + c
    if n == 0:
        return dict(model_only_correct=b, baseline_only_correct=c, p_value=1.0)
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return dict(model_only_correct=b, baseline_only_correct=c, n_discordant=n,
                p_value=round(min(1.0, 2 * tail), 4))


def add_baselines(df):
    """Two naive rules a user could apply with no model at all, computed from the same
    pre-game snapshot the model saw. Ties go to the home team."""
    df = df.copy()
    df['pick_home'] = 1
    df['pick_better_record'] = (df.home_wpct >= df.away_wpct).astype(int)
    df['pick_better_margin'] = (df.home_pm >= df.away_pm).astype(int)
    return df


def acc_block(df):
    n = len(df)
    if n == 0:
        return dict(n=0)
    k = int((df.pred == df.home_win).sum())
    hk = int(df.home_win.sum())
    bR = int((df.pick_better_record == df.home_win).sum())
    bM = int((df.pick_better_margin == df.home_win).sum())
    return dict(
        n=int(n),
        model_correct=k,
        model_accuracy_pct=round(k / n * 100, 2),
        model_accuracy_95ci=wilson(k, n),
        home_wins=hk,
        home_team_baseline_pct=round(hk / n * 100, 2),
        home_team_baseline_95ci=wilson(hk, n),
        edge_over_home_baseline_pp=round((k - hk) / n * 100, 2),
        better_record_baseline_pct=round(bR / n * 100, 2),
        better_record_baseline_95ci=wilson(bR, n),
        edge_over_better_record_pp=round((k - bR) / n * 100, 2),
        better_point_margin_baseline_pct=round(bM / n * 100, 2),
        edge_over_better_point_margin_pp=round((k - bM) / n * 100, 2),
        mean_confidence_pct=round(float(df.conf.mean()) * 100, 2),
        brier_score=round(float(np.mean((df.p_home - df.home_win) ** 2)), 4),
        log_loss=round(float(-np.mean(
            df.home_win * np.log(np.clip(df.p_home, 1e-9, 1)) +
            (1 - df.home_win) * np.log(np.clip(1 - df.p_home, 1e-9, 1)))), 4),
    )


def calibration(df, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = df[(df.conf >= lo) & (df.conf < hi)] if hi < 1.0 else df[df.conf >= lo]
        n = len(sel)
        if n == 0:
            out.append(dict(bucket=f"{int(lo*100)}-{int(hi*100)}%", n=0))
            continue
        k = int((sel.pred == sel.home_win).sum())
        out.append(dict(
            bucket=f"{int(lo*100)}-{int(hi*100)}%",
            n=n,
            mean_predicted_pct=round(float(sel.conf.mean()) * 100, 2),
            actual_win_pct=round(k / n * 100, 2),
            actual_95ci=wilson(k, n),
            calibration_error_pp=round(k / n * 100 - float(sel.conf.mean()) * 100, 2),
        ))
    return out


def by_threshold(df, thresholds):
    """If you only surfaced picks at or above a confidence floor, what would you get?"""
    out = []
    for t in thresholds:
        sel = df[df.conf >= t]
        n = len(sel)
        if n == 0:
            out.append(dict(min_confidence_pct=round(t * 100, 1), n=0))
            continue
        k = int((sel.pred == sel.home_win).sum())
        hk = int(sel.home_win.sum())
        out.append(dict(
            min_confidence_pct=round(t * 100, 1),
            n=n,
            pct_of_all_games=round(n / len(df) * 100, 1),
            accuracy_pct=round(k / n * 100, 2),
            accuracy_95ci=wilson(k, n),
            home_baseline_pct=round(hk / n * 100, 2),
            better_record_baseline_pct=round(
                float((sel.pick_better_record == sel.home_win).mean()) * 100, 2),
        ))
    return out


def reliability_home(df, edges):
    """Calibration of P(home win) itself, which is the number the UI derives Kelly from."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = df[(df.p_home >= lo) & (df.p_home < hi)]
        n = len(sel)
        if n == 0:
            out.append(dict(bucket=f"{int(lo*100)}-{int(hi*100)}%", n=0))
            continue
        k = int(sel.home_win.sum())
        out.append(dict(
            bucket=f"{int(lo*100)}-{int(hi*100)}%",
            n=n,
            mean_predicted_home_win_pct=round(float(sel.p_home.mean()) * 100, 2),
            actual_home_win_pct=round(k / n * 100, 2),
            actual_95ci=wilson(k, n),
            calibration_error_pp=round(k / n * 100 - float(sel.p_home.mean()) * 100, 2),
        ))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true', help='run reconstruction validation only')
    ap.add_argument('--out', default=OUT_JSON)
    args = ap.parse_args()

    con_team = sqlite3.connect(TEAMDATA, uri=True)
    con_ds = sqlite3.connect(DATASET, uri=True)

    print('Checking feature order against the training table ...')
    assert_feature_order(con_ds)
    print(f'  OK: {len(FEATURE_ORDER)} columns, {TRAIN_TABLE} minus {TRAIN_DROPS}')

    print('Loading box scores ...')
    tg = load_team_games(con_team)
    print(f'  {len(tg)} team-game rows, {tg.game_id.nunique()} games, '
          f'seasons {sorted(tg.season.unique())}')

    # ---- validation -------------------------------------------------------
    print('\nValidating reconstructed snapshots against the original date-keyed tables ...')
    n_dates, per_col = validate_snapshots(tg, con_team, VALIDATION_SEASONS)
    tot_ok = sum(v[0] for v in per_col.values())
    tot_n = sum(v[1] for v in per_col.values())
    print(f'  {n_dates} snapshot dates compared, {tot_n} cells, '
          f'{tot_ok / tot_n * 100:.3f}% exact')
    worst = sorted(per_col.items(), key=lambda kv: kv[1][0] / kv[1][1])[:8]
    for c, (ok, n) in worst:
        print(f'    {c:20s} {ok}/{n}  ({ok / n * 100:.3f}%)')

    row_val = []
    for s in VALIDATION_SEASONS:
        r = validate_feature_rows(tg, con_team, con_ds, s)
        if r:
            row_val.append(r)
            print(f"\n  end-to-end {s}: matched {r['n_matched_to_box_scores']}/"
                  f"{r['n_training_rows_in_season']} training rows; "
                  f"stat features exact {r['stat_features_exact_pct']:.3f}%, "
                  f"within 0.1 {r['stat_features_within_0p1_pct']:.3f}%; "
                  f"same model pick {r['same_pick_pct']:.2f}%; "
                  f"mean |dP| {r['mean_abs_prob_diff']:.5f}; "
                  f"days-rest exact {r['days_rest_exact_pct']:.2f}%")

    snapshot_validation = dict(
        seasons=VALIDATION_SEASONS,
        n_snapshot_dates_compared=n_dates,
        n_cells_compared=int(tot_n),
        exact_match_pct=round(tot_ok / tot_n * 100, 4),
        per_column={c: dict(exact=int(v[0]), n=int(v[1]),
                            pct=round(v[0] / v[1] * 100, 4)) for c, v in per_col.items()},
    )

    if args.validate:
        con_team.close()
        con_ds.close()
        return

    # ---- backtest ---------------------------------------------------------
    booster = load_model()
    sanity = in_sample_sanity(con_ds, booster)
    print(f"\nIn-sample sanity: {sanity['in_sample_accuracy_pct']}% on the training table "
          f"(n={sanity['n']}); shifted-column control "
          f"{sanity['control_accuracy_with_shifted_columns_pct']}%")

    print('\nBacktesting held-out seasons ...')
    frames = {}
    skipped_all = []
    for conv in ('training', 'legacy', 'production'):
        parts = []
        for s in HELD_OUT_SEASONS:
            X, meta, skipped = build_games(tg, s, convention=conv)
            if conv == 'training':
                skipped_all += skipped
            p = booster.predict(xgb.DMatrix(X))
            meta = meta.copy()
            meta['p_home'] = p[:, 1]
            meta['pred'] = (p[:, 1] > p[:, 0]).astype(int)
            meta['conf'] = np.maximum(p[:, 0], p[:, 1])
            parts.append(meta)
        frames[conv] = add_baselines(pd.concat(parts, ignore_index=True))

    df = frames['training']
    print(f'  {len(df)} games scored '
          f'({df.season_type.value_counts().to_dict()}); {len(skipped_all)} skipped')

    conf_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    prob_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]

    results = dict(
        overall=acc_block(df),
        by_season={s: acc_block(df[df.season == s]) for s in HELD_OUT_SEASONS},
        by_season_type={t: acc_block(df[df.season_type == t])
                        for t in ['Regular Season', 'Playoffs']},
        by_season_and_type={
            f"{s} {t}": acc_block(df[(df.season == s) & (df.season_type == t)])
            for s in HELD_OUT_SEASONS for t in ['Regular Season', 'Playoffs']},
        by_month={str(m): acc_block(df[df.game_date.str.slice(0, 7) == m])
                  for m in sorted(df.game_date.str.slice(0, 7).unique())},
        calibration_by_confidence=calibration(df, conf_edges),
        calibration_by_confidence_regular_season=calibration(
            df[df.season_type == 'Regular Season'], conf_edges),
        accuracy_at_confidence_threshold=by_threshold(
            df, [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]),
        reliability_home_win_probability=reliability_home(df, prob_edges),
        pick_side_split=dict(
            picked_home=acc_block(df[df.pred == 1]),
            picked_away=acc_block(df[df.pred == 0]),
        ),
    )

    sensitivity = {conf: dict(n=len(f),
                              accuracy_pct=round(float((f.pred == f.home_win).mean()) * 100, 2))
                   for conf, f in frames.items()}

    # positive control: same pipeline, but same-day results allowed into the features
    leak_parts = []
    for s in HELD_OUT_SEASONS:
        Xl, ml, _ = build_games(tg, s, convention='training', leak=True)
        pl = booster.predict(xgb.DMatrix(Xl))
        ml = ml.copy()
        ml['pred'] = (pl[:, 1] > pl[:, 0]).astype(int)
        leak_parts.append(ml)
    ldf = pd.concat(leak_parts, ignore_index=True)
    ldf = ldf[ldf.game_id.isin(set(df.game_id))]   # same games as the strict run
    leak_control = dict(
        purpose=('Positive control. Identical pipeline except the as-of cutoff is relaxed from '
                 '"strictly before the game date" to "including the game date", so each game\'s '
                 'own result leaks into its own features. The reported result is the strict one; '
                 'this row exists to show the strict cutoff is actually binding and that the '
                 'harness would surface leakage if it were present.'),
        strict_accuracy_pct=round(float((df.pred == df.home_win).mean()) * 100, 2),
        leaky_accuracy_pct=round(float((ldf.pred == ldf.home_win).mean()) * 100, 2),
        n=int(len(ldf)),
    )
    print(f"\nLeakage control: strict {leak_control['strict_accuracy_pct']}% vs "
          f"deliberately-leaky {leak_control['leaky_accuracy_pct']}% (n={leak_control['n']})")

    # ---- over/under -------------------------------------------------------
    ou = ou_availability(con_team, tg)

    o = results['overall']
    headline = dict(
        number_pct=o['model_accuracy_pct'],
        n_games=o['n'],
        ci95_pct=o['model_accuracy_95ci'],
        seasons=HELD_OUT_SEASONS,
        recommended_public_wording=(
            f"{o['model_accuracy_pct']}% moneyline accuracy across "
            f"{o['n']:,} NBA games in the {HELD_OUT_SEASONS[0]} and {HELD_OUT_SEASONS[1]} "
            f"seasons -- games played entirely after the model was trained. "
            f"Picking the home team every time would have gone "
            f"{o['home_team_baseline_pct']}%."),
        required_footnote=(
            f"Measured on {o['n']:,} games from {HELD_OUT_SEASONS[0]} and "
            f"{HELD_OUT_SEASONS[1]}, none of which were in the training data "
            f"(training ends 2024-04-28). 95% confidence interval "
            f"{o['model_accuracy_95ci'][0]}-{o['model_accuracy_95ci'][1]}%. "
            f"Accuracy is not profitability: no historical odds exist for these seasons, "
            f"so no ROI has been measured. A simple 'pick the better record' rule scored "
            f"{o['better_record_baseline_pct']}% on the same games."),
        replaces_claim='68.9% test accuracy on held-out games',
        why_the_old_claim_is_wrong=(
            'The 68.9% figure is the best of 300 random train/test splits taken during '
            'training, not an out-of-sample result. The splits were random rather than '
            'chronological and the model saved was the single luckiest of the 300. The model '
            f"scores only {sanity['in_sample_accuracy_pct']}% on its own training rows, so "
            '68.9% was never a stable estimate of anything. The phrase "held-out games" is '
            'also misleading: until this backtest, no games outside the training window had '
            'ever been scored.'),
        do_not_say=[
            'Do not quote 68.9% as held-out or out-of-sample accuracy.',
            'Do not quote the confidence-floor numbers (e.g. "79% on high-confidence picks") '
            'without stating that they cover only a selected subset of games and that the '
            'threshold was chosen after seeing the data.',
            'Do not imply profitability, ROI, or edge against the closing line. None of that '
            'has been measured for these seasons.',
        ],
    )

    artifact = dict(
        schema_version=1,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        generated_by='backtest_model.py',
        headline=headline,
        model=dict(
            file='Models/XGBoost_Models/XGBoost_68.9%_ML-3.json',
            type='XGBoost multi:softprob, 2 classes, 750 boosting rounds, max_depth 3, eta 0.01',
            n_features=106,
            target='home team wins',
            trained_by='src/Train-Models/XGBoost_Model_ML.py',
            training_table=f'Data/dataset.sqlite :: {TRAIN_TABLE}',
            training_rows=15115,
            training_date_range=['2012-11-04', '2024-04-28'],
            filename_accuracy_claim_68_9=(
                'The 68.9% in the filename is the MAXIMUM test accuracy observed across 300 '
                'random 90/10 train_test_split runs; the model saved is the single best of those '
                'runs. Splits were random rather than chronological, so test rows sat between '
                'train rows in time. It is a selected maximum over noisy estimates, not an '
                'out-of-sample estimate, and it should not be quoted as held-out accuracy.'),
        ),
        evaluation=dict(
            seasons=HELD_OUT_SEASONS,
            why_held_out=('Neither season appears in dataset_2012-24_new, which ends 2024-04-28. '
                          'No 2024-25 or 2025-26 game influenced training in any way.'),
            n_games_scored=int(len(df)),
            n_games_available=int(tg[tg.season.isin(HELD_OUT_SEASONS)].game_id.nunique()),
            n_games_skipped=len(skipped_all),
            skipped_reason='season-opening games have no prior-game stats to build features from',
            games_by_season={s: int((df.season == s).sum()) for s in HELD_OUT_SEASONS},
            games_by_season_type=df.season_type.value_counts().to_dict(),
            date_range=[df.game_date.min(), df.game_date.max()],
        ),
        results=results,
        baseline=dict(
            description=('Rules a user could apply with no model at all, scored on exactly the '
                         'same games from exactly the same pre-game snapshot. These are the '
                         'numbers that make the model accuracy meaningful or not.'),
            n=int(len(df)),
            model_accuracy_pct=round(float((df.pred == df.home_win).mean()) * 100, 2),
            always_pick_home=dict(
                accuracy_pct=round(float(df.home_win.mean()) * 100, 2),
                accuracy_95ci=wilson(int(df.home_win.sum()), len(df)),
                model_minus_baseline_pp=round(
                    float((df.pred == df.home_win).mean() - df.home_win.mean()) * 100, 2)),
            pick_better_win_pct=dict(
                description='pick the team with the better season-to-date record; ties to home',
                accuracy_pct=round(float((df.pick_better_record == df.home_win).mean()) * 100, 2),
                accuracy_95ci=wilson(int((df.pick_better_record == df.home_win).sum()), len(df)),
                model_minus_baseline_pp=round(float(
                    (df.pred == df.home_win).mean()
                    - (df.pick_better_record == df.home_win).mean()) * 100, 2),
                model_agrees_with_baseline_pct=round(
                    float((df.pred == df.pick_better_record).mean()) * 100, 2),
                mcnemar_vs_model=mcnemar(df, 'pick_better_record')),
            pick_better_point_margin=dict(
                description='pick the team with the better season-to-date point differential',
                accuracy_pct=round(float((df.pick_better_margin == df.home_win).mean()) * 100, 2),
                accuracy_95ci=wilson(int((df.pick_better_margin == df.home_win).sum()), len(df)),
                model_minus_baseline_pp=round(float(
                    (df.pred == df.home_win).mean()
                    - (df.pick_better_margin == df.home_win).mean()) * 100, 2),
                model_agrees_with_baseline_pct=round(
                    float((df.pred == df.pick_better_margin).mean()) * 100, 2),
                mcnemar_vs_model=mcnemar(df, 'pick_better_margin')),
            interpretation=(
                'The model beats "always pick home" by a wide, clearly significant margin. It '
                'does NOT meaningfully beat the two record-based heuristics: the differences are '
                'under half a percentage point and not statistically significant. The model\'s '
                'defensible added value over those heuristics is that it emits a calibrated '
                'probability rather than a bare pick -- see results.calibration_by_confidence.'),
        ),
        validation=dict(
            feature_order_check=(
                f'PASS -- the 106 columns and their order are exactly {TRAIN_TABLE} minus '
                f'{TRAIN_DROPS}, asserted at runtime against the live table.'),
            in_sample_sanity=sanity,
            snapshot_reconstruction=snapshot_validation,
            end_to_end_feature_rows=row_val,
            leakage_positive_control=leak_control,
        ),
        days_rest_sensitivity=dict(
            note=('Days rest is the only feature whose definition is ambiguous. The training data '
                  'mixes two conventions and production uses a third. Headline numbers use the '
                  '"training" convention (plain day difference, 7 for a first game), matching '
                  'src/Process-Data/Get_Odds_Data.py which produced the most recent training '
                  'season. All three are reported so the reader can see the number is not '
                  'sensitive to the choice.'),
            conventions=dict(
                training='(game_date - previous_game_date).days; 7 on a first game; no clamp',
                legacy='same but 10 on a first game and values outside (0,9) clamped to 9 '
                       '(src/Process-Data/Add_Days_Rest.py, used for 2012-13..2022-23)',
                production='training difference, then clipped to [1,7] -- what the corrected '
                           'main_api.py live path produces. The clip is the only systematic '
                           'difference from "training", and it costs nothing measurable.',
            ),
            accuracy=sensitivity,
        ),
        over_under=ou,
        methodology_notes=[
            'Features are as-of-date cumulative REGULAR SEASON team stats for both teams '
            '(26 counting/percentage stats + their 26 league rank columns, home block then away '
            'block) plus days rest for each team: 106 features, in exactly the order '
            'dataset_2012-24_new has after the training script drops '
            "['Score','Home-Team-Win','TEAM_NAME','Date','TEAM_NAME.1','Date.1','OU-Cover','OU'].",
            'No odds are used as model inputs. The model never sees a betting line.',
            'NO LEAKAGE: features for a game on date D aggregate only games played strictly '
            'before D. This reproduces the original pipeline, where Get_Data.py stored the '
            'stats-through-D snapshot under the table name D+1 and Create_Games.py read the '
            "table named for the game's own date.",
            'Playoff games use the frozen final regular-season stats, exactly as the original '
            'snapshots did (the source URL pins SeasonType=Regular+Season).',
            'Reconstructed from Data/TeamData.sqlite box_scores; BLKA = opponent blocks, '
            'PFD = opponent personal fouls, and team turnovers are recovered from the advanced '
            'box score because the traditional team block omits non-player turnovers.',
            'Rank columns are computed on unrounded values with ties taking the best rank; '
            'ranks are ascending (1 = lowest) for L, TOV, PF and BLKA and descending otherwise.',
            'Deterministic: the model is only used for inference, nothing is retrained or '
            'resampled, and every database is opened read-only.',
        ],
        caveats=[
            'THE BIGGEST ONE: the model does not beat a trivial heuristic. "Pick the team with '
            'the better season-to-date record" scores 65.34% and "pick the better point '
            'differential" scores 65.58% on exactly these games. The model is at 65.81%. A '
            'paired McNemar test puts p at 0.61 and 0.82 respectively -- the difference is '
            'noise. Beating the home-field baseline by 10.7 points is real; beating a fan with '
            'a standings page is not demonstrated. What the model does add over those rules is '
            'a calibrated probability rather than a bare pick.',
            'The confidence-floor table is a post-hoc slice: the thresholds were chosen after '
            'seeing the data, and each row covers only a subset of games. Those numbers are '
            'diagnostics, not headline claims.',
            'Historical closing odds in Data/OddsData.sqlite stop at 2024-04-28, so no ROI, '
            'expected value, Kelly or closing-line-value result can be computed for these '
            'seasons. Accuracy alone says nothing about profitability: a moneyline bettor needs '
            'to beat the price, not the coin flip.',
            'The reconstruction is validated against the original snapshots on 2022-23 and '
            '2023-24 only, because those are the seasons where both the box scores and the '
            'original date-keyed tables exist.',
            'Roughly one game in 500 has a reconstructed per-game average that differs from the '
            'original by 0.1 due to half-up rounding at a boundary; the effect on model output '
            'is measured in the validation block and is negligible.',
            'The live days-rest path in main_api.py was found to be clock-dependent: it compared '
            'UTC schedule timestamps against a naive local datetime, so the value it produced '
            'varied with the server timezone and the hour of the call (97.6% correct during the '
            'daytime window that the scheduled 9 AM job falls in, degrading to 41% agreement at '
            'midnight local). It was corrected to key off the game\'s own US Eastern date. The '
            'corrected path reproduces the training convention exactly, up to the existing '
            '[1,7] clip, and days_rest_sensitivity shows that clip costs nothing measurable. '
            'This does not affect any number in this artifact, which builds days rest from the '
            'schedule directly and never called the API path.',
        ],
    )

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2)
    print(f'\nWrote {args.out}')
    print_report(artifact, df)
    con_team.close()
    con_ds.close()


def ou_availability(con_team, tg):
    """Can the over/under model be backtested? Only with real, pre-game total lines."""
    try:
        con = sqlite3.connect(ODDSDATA, uri=True)
        tabs = pd.read_sql_query(
            "select name from sqlite_master where type='table'", con)['name'].tolist()
        latest = None
        for t in tabs:
            if t.startswith('odds_'):
                try:
                    d = pd.read_sql_query(f'select max(Date) m from "{t}"', con)['m'][0]
                    if d and (latest is None or str(d) > latest):
                        latest = str(d)
                except Exception:
                    pass
        snap_n = int(pd.read_sql_query('select count(*) n from odds_snapshots', con)['n'][0])
        snap_rng = pd.read_sql_query(
            'select min(captured_at) a, max(captured_at) b from odds_snapshots',
            con).iloc[0].tolist()
        plog_n = int(pd.read_sql_query('select count(*) n from predictions_log', con)['n'][0])
        con.close()
    except Exception as e:  # pragma: no cover
        return dict(backtested=False, reason=f'could not read OddsData.sqlite: {e}')
    return dict(
        backtested=False,
        reason=('Not backtested. The over/under model takes the sportsbook total itself as its '
                '107th input feature, and no historical total lines exist for 2024-25 or 2025-26 '
                'anywhere in this repo. Back-filling or estimating a line would make the result '
                'meaningless, so it was skipped rather than approximated.'),
        latest_historical_odds_date=latest,
        odds_snapshots_rows=snap_n,
        odds_snapshots_captured_range=snap_rng,
        predictions_log_rows=plog_n,
        model_file='Models/XGBoost_Models/XGBoost_54.8%_UO-8.json',
        model_n_features=107,
    )


def print_report(a, df):
    r = a['results']
    print('\n================ HELD-OUT RESULTS ================')
    o = r['overall']
    print(f"Overall: {o['model_accuracy_pct']}%  (n={o['n']}, 95% CI "
          f"{o['model_accuracy_95ci'][0]}-{o['model_accuracy_95ci'][1]})")
    print(f"Baseline  always pick home  : {o['home_team_baseline_pct']}%  -> edge "
          f"{o['edge_over_home_baseline_pp']} pp")
    print(f"Baseline  better record     : {o['better_record_baseline_pct']}%  -> edge "
          f"{o['edge_over_better_record_pp']} pp")
    print(f"Baseline  better point diff : {o['better_point_margin_baseline_pct']}%  -> edge "
          f"{o['edge_over_better_point_margin_pp']} pp")
    print(f"Brier {o['brier_score']}  LogLoss {o['log_loss']}  "
          f"mean confidence {o['mean_confidence_pct']}%")
    print('\nBy season:')
    for k, v in r['by_season'].items():
        print(f"  {k}: {v['model_accuracy_pct']}% (n={v['n']}), "
              f"home baseline {v['home_team_baseline_pct']}%")
    print('\nBy season type:')
    for k, v in r['by_season_type'].items():
        print(f"  {k}: {v['model_accuracy_pct']}% (n={v['n']}), "
              f"home baseline {v['home_team_baseline_pct']}%")
    print('\nCalibration by displayed confidence:')
    print(f"  {'bucket':>10} {'n':>6} {'predicted':>10} {'actual':>8} {'err(pp)':>8}")
    for b in r['calibration_by_confidence']:
        if b['n'] == 0:
            print(f"  {b['bucket']:>10} {0:>6}")
            continue
        print(f"  {b['bucket']:>10} {b['n']:>6} {b['mean_predicted_pct']:>9.2f}% "
              f"{b['actual_win_pct']:>7.2f}% {b['calibration_error_pp']:>8.2f}")
    print('\nAccuracy at a confidence floor:')
    for b in r['accuracy_at_confidence_threshold']:
        if b['n'] == 0:
            continue
        print(f"  >={b['min_confidence_pct']:>5}%  n={b['n']:>5} "
              f"({b['pct_of_all_games']:>5}% of games)  acc {b['accuracy_pct']:>6.2f}%  "
              f"[{b['accuracy_95ci'][0]}-{b['accuracy_95ci'][1]}]  "
              f"home baseline {b['home_baseline_pct']:>6.2f}%")
    print('\nReliability of P(home win):')
    for b in r['reliability_home_win_probability']:
        if b['n'] == 0:
            continue
        print(f"  {b['bucket']:>10} n={b['n']:>5} pred {b['mean_predicted_home_win_pct']:>6.2f}% "
              f"actual {b['actual_home_win_pct']:>6.2f}%  err {b['calibration_error_pp']:>6.2f}pp")
    print('\nPaired McNemar vs baselines:')
    for k in ('pick_better_win_pct', 'pick_better_point_margin'):
        m = a['baseline'][k]['mcnemar_vs_model']
        print(f"  {k}: model-only-right {m['model_only_correct']}, "
              f"baseline-only-right {m['baseline_only_correct']}, p={m['p_value']}")
    print('\nDays-rest sensitivity:', a['days_rest_sensitivity']['accuracy'])


# ===========================================================================
# CANDIDATE MODEL -- STEP 3 of the pre-registered retrain protocol.
#
# Everything below is ADDITIVE. The production backtest above is unchanged
# and backtest_results.json is never rewritten by the candidate path; the
# candidate writes backtest_results_candidate.json only.
#
# The candidate (Models/candidate_2026-08) uses 207 features:
#   base 106  -- identical to the production model's inputs, reconstructed
#                here by exactly the same build_games() path,
#   rolling 92 -- K in {10,20} per-side blocks. In training these came from
#                snapshot DIFFS (step 1); the snapshots stop 2024-04-29, so
#                for the eval seasons they are rebuilt from box_scores with
#                the same strict game_date < d cutoff, and the two
#                construction paths are cross-validated against each other
#                on 2022-24 where both exist,
#   elo 4      -- pre-game Elo. Training burned in over the odds archive
#                (2007-08 .. 2024-04-28); the eval continues that replay
#                over box_scores games AFTER the archive's last date with
#                identical constants and (game_date, game_id) ordering,
#   rest 5     -- b2b/3-in-4 flags + rest_diff, built by the step-1 builder
#                itself (retrain_features.build_rest_features).
# ===========================================================================
CAND_DIR = os.path.join(ROOT, 'Models', 'candidate_2026-08')
CAND_MODEL = os.path.join(CAND_DIR, 'model.json')
CAND_OUT_JSON = os.path.join(ROOT, 'backtest_results_candidate.json')
RETRAIN_FEATURES_PY = os.path.join(ROOT, 'src', 'Process-Data', 'retrain_features.py')
FEATURE_CACHE_RO = ("file:" + os.path.join(ROOT, 'Data', 'retrain_features.sqlite')
                    .replace("\\", "/") + "?mode=ro")
TRAINING_RO = ("file:" + os.path.join(ROOT, 'Data', 'retrain_training.sqlite')
               .replace("\\", "/") + "?mode=ro")
ROLLING_KS = (10, 20)
CROSSVAL_SEASONS = ["2022-23", "2023-24"]   # both snapshot-diff and box paths exist


def load_rf():
    """Import the step-1 feature builders from the hyphenated directory."""
    spec = importlib.util.spec_from_file_location("retrain_features", RETRAIN_FEATURES_PY)
    rf = importlib.util.module_from_spec(spec)
    sys.modules["retrain_features"] = rf
    spec.loader.exec_module(rf)
    return rf


def roll_stat_cols(rf):
    return list(rf.COUNT_STATS) + list(rf.PCT_STATS) + ["WIN_PCT"]


def candidate_column_order(rf):
    rolling = []
    for k in ROLLING_KS:
        for side in ("HOME", "AWAY"):
            rolling += [f"R{k}_{side}_{s}" for s in roll_stat_cols(rf)]
    elo = ["ELO_HOME", "ELO_AWAY", "ELO_DIFF", "ELO_HOME_EXPECTED"]
    rest = ["REST_HOME_B2B", "REST_AWAY_B2B", "REST_HOME_3IN4",
            "REST_AWAY_3IN4", "REST_DIFF"]
    return FEATURE_ORDER + rolling + elo + rest


def load_candidate(rf):
    """Booster + manifest + config, with hard column-order assertions."""
    with open(os.path.join(CAND_DIR, 'feature_manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(os.path.join(CAND_DIR, 'training_config.json'), encoding='utf-8') as f:
        config = json.load(f)
    booster = xgb.Booster()
    booster.load_model(CAND_MODEL)
    cols = candidate_column_order(rf)
    assert manifest['n_features'] == 207 and len(cols) == 207
    assert manifest['feature_columns'] == cols, \
        'harness column order does not reproduce the manifest'
    assert booster.num_features() == 207
    assert booster.feature_names == cols, \
        'booster feature names do not match the manifest'
    assert config['calibrator']['chosen'] == 'isotonic'
    return booster, manifest, config


def apply_isotonic(p, iso_cfg):
    """Apply the stored isotonic calibrator exactly as sklearn would:
    linear interpolation between thresholds, clipped outside the range
    (out_of_bounds='clip'). np.interp implements precisely that."""
    x = np.asarray(iso_cfg['x_thresholds'], dtype=float)
    y = np.asarray(iso_cfg['y_thresholds'], dtype=float)
    return np.interp(np.asarray(p, dtype=float), x, y)


def team_canonical_map(tg, rf):
    """{team_id: canonical franchise name} from box_scores, verified 30 teams."""
    pairs = tg[['team_id', 'team_name']].drop_duplicates()
    canon = {int(r.team_id): rf.normalize_team(r.team_name) for r in pairs.itertuples()}
    assert len(canon) == 30 and len(set(canon.values())) == 30
    return canon


# ---------------------------------------------------------------------------
# Rolling-K from box_scores (eval-side construction of the step-1 feature)
# ---------------------------------------------------------------------------
def rolling_from_box(tg, rf, season, leak=False):
    """{(team_id, game_date): {k: (stats_vector, window, gp)}} for every
    (team, date) the team plays on in `season`, built from that season's
    REGULAR-SEASON games with game_date strictly before the game date
    (snapshots freeze during the playoffs, so playoff games use the last K
    regular-season games -- same as the training-side snapshot diffs).

    Stats vector order = COUNT_STATS + FG_PCT/FG3_PCT/FT_PCT + WIN_PCT,
    matching retrain_features. TOV in `tg` is already the recovered TOTAL
    team turnovers (advanced-box method), matching the snapshots' TOV.

    `leak=True` relaxes the cutoff to game_date <= d (a game's own stats
    enter its own rolling window). Positive control only.
    """
    stat_names = list(rf.COUNT_STATS)
    reg = tg[(tg.season == season) & (tg.season_type == 'Regular Season')] \
        .sort_values(['game_date', 'game_id'])
    per_team = {}
    for tid, grp in reg.groupby('team_id'):
        dates = grp.game_date.tolist()
        cols = []
        for s in stat_names:
            v = grp[s].values.astype(float)
            if s == 'MIN':
                v = v / 5.0          # snapshot MIN is team minutes / 5 per game
            cols.append(v)
        cols.append(grp.W.values.astype(float))
        arr = np.column_stack(cols)
        cums = np.vstack([np.zeros(arr.shape[1]), np.cumsum(arr, axis=0)])
        per_team[int(tid)] = (dates, cums)

    i_fgm, i_fga = stat_names.index('FGM'), stat_names.index('FGA')
    i_f3m, i_f3a = stat_names.index('FG3M'), stat_names.index('FG3A')
    i_ftm, i_fta = stat_names.index('FTM'), stat_names.index('FTA')
    i_w = len(stat_names)

    out = {}
    for r in tg[tg.season == season].itertuples():
        key = (int(r.team_id), r.game_date)
        if key in out:
            continue
        dates, cums = per_team.get(int(r.team_id), ([], None))
        idx = (bisect.bisect_right(dates, r.game_date) if leak
               else bisect.bisect_left(dates, r.game_date))
        if idx == 0:
            out[key] = None            # season opener: no rolling block (NaN)
            continue
        blocks = {}
        for k in ROLLING_KS:
            w = min(k, idx)
            tot = cums[idx] - cums[idx - w]
            means = tot[:len(stat_names)] / w
            def _pct(n, d):
                return (tot[n] / tot[d]) if tot[d] else np.nan
            vec = np.concatenate([
                means,
                [_pct(i_fgm, i_fga), _pct(i_f3m, i_f3a), _pct(i_ftm, i_fta),
                 tot[i_w] / w]])
            blocks[k] = (vec, w, idx)
        out[key] = blocks
    return out


def crossval_rolling(tg, rf, seasons=CROSSVAL_SEASONS):
    """Cross-validate the box-score rolling construction against the step-1
    snapshot-diff cache on seasons where BOTH exist. Pre-registered bound
    (from step 1): >=99% of counting-stat cells within 0.05*(GP+GP')/K.

    Tolerances by family:
      counting stats : 0.05*(GP+GP')/window  (snapshot avgs are rounded 0.1)
      WIN_PCT        : 1e-6 (both paths are exact integer wins / window)
      shooting PCTs  : 0.1*(GP+GP')/max(denominator totals, 1)  (propagated)
    """
    canon = team_canonical_map(tg, rf)
    inv = {v: k for k, v in canon.items()}
    stat_cols = roll_stat_cols(rf)
    count_set = set(rf.COUNT_STATS)
    pct_den = {'FG_PCT': 'FGA', 'FG3_PCT': 'FG3A', 'FT_PCT': 'FTA'}
    box = {s: rolling_from_box(tg, rf, s) for s in seasons}

    con = sqlite3.connect(FEATURE_CACHE_RO, uri=True)
    q = ("SELECT season, date, team, k, window, is_partial, exact_window, gp, "
         + ", ".join(f'"{c}"' for c in stat_cols)
         + f" FROM rolling_features WHERE season IN ({','.join('?' * len(seasons))})"
         f" AND k IN ({','.join('?' * len(ROLLING_KS))})")
    rows = con.execute(q, list(seasons) + list(ROLLING_KS)).fetchall()
    con.close()

    fam = {f: [0, 0, 0.0] for f in ('counting', 'win_pct', 'pct')}  # ok, n, worst ratio
    n_rows = n_missing = n_window_mismatch = n_gp_mismatch = n_inexact = 0
    for (season, date, team, k, window, is_partial, exact_window, gp, *vals) in rows:
        n_rows += 1
        if not exact_window:
            n_inexact += 1
            continue
        tid = inv[team]
        mine = box[season].get((tid, date))
        if mine is None:
            n_missing += 1
            continue
        vec, w, idx = mine[k]
        if idx != gp:
            n_gp_mismatch += 1
            continue
        if w != window:
            n_window_mismatch += 1
            continue
        gp_prev = gp - window
        base_tol = 0.05 * (gp + gp_prev)
        for j, sname in enumerate(stat_cols):
            cache_v, my_v = vals[j], vec[j]
            if cache_v is None or (isinstance(my_v, float) and np.isnan(my_v)):
                continue
            d = abs(float(cache_v) - float(my_v))
            if sname in count_set:
                tol, f = base_tol / window, 'counting'
            elif sname == 'WIN_PCT':
                tol, f = 1e-6, 'win_pct'
            else:
                den_idx = stat_cols.index(pct_den[sname])
                den_tot = vec[den_idx] * window
                tol, f = 2.0 * base_tol / max(den_tot, 1.0), 'pct'
            fam[f][1] += 1
            if d <= tol + 1e-12:
                fam[f][0] += 1
            fam[f][2] = max(fam[f][2], d / tol if tol > 0 else d)
    result = dict(
        seasons=seasons,
        n_cache_rows=n_rows,
        n_inexact_window_rows_excluded=n_inexact,
        n_missing_in_box_path=n_missing,
        n_gp_mismatch=n_gp_mismatch,
        n_window_mismatch=n_window_mismatch,
        families={f: dict(within_tolerance=v[0], n=v[1],
                          pct=round(v[0] / v[1] * 100, 4) if v[1] else None,
                          worst_diff_over_tolerance=round(v[2], 3))
                  for f, v in fam.items()},
        preregistered_bound='>=99% of counting-stat cells within 0.05*(GP+GP\')/K',
    )
    result['bound_pass'] = bool(
        fam['counting'][1] > 0
        and fam['counting'][0] / fam['counting'][1] >= 0.99
        and fam['win_pct'][0] == fam['win_pct'][1]
        and (fam['pct'][1] == 0 or fam['pct'][0] / fam['pct'][1] >= 0.99))
    return result


# ---------------------------------------------------------------------------
# Elo continuation over box_scores (2024-04-29 onward)
# ---------------------------------------------------------------------------
def box_game_frame(tg):
    """One row per game with home/away ids, names, points."""
    h = tg[tg.is_home][['game_id', 'game_date', 'season', 'season_type',
                        'team_id', 'team_name', 'PTS']].rename(
        columns={'team_id': 'home_id', 'team_name': 'home_name', 'PTS': 'home_pts'})
    a = tg[~tg.is_home][['game_id', 'team_id', 'team_name', 'PTS']].rename(
        columns={'team_id': 'away_id', 'team_name': 'away_name', 'PTS': 'away_pts'})
    g = h.merge(a, on='game_id')
    return g.sort_values(['game_date', 'game_id']).reset_index(drop=True)


def continue_elo(rf, tg, base_elo=None, flip_gid=None):
    """Continue the step-1 odds-table Elo replay over box_scores games played
    AFTER the odds archive's last date (2024-04-28): the remainder of the
    2023-24 playoffs, then (after the standard 25%-to-1505 between-season
    reversion) 2024-25 and 2025-26, ordered by (game_date, game_id) with the
    identical constants. A team plays at most once per date, so within-date
    order cannot change any rating.

    The feature is the PRE-GAME rating: recorded before the game's own
    update is applied. `flip_gid` negates one game's margin -- used by the
    off-by-one probe to show a game's own outcome cannot reach its own
    pre-game feature.

    Returns dict with pre / post ({game_id: {...}}), continuation game list,
    and bookkeeping.
    """
    if base_elo is None:
        base_elo = rf.build_elo_odds()
    last_odds_date = max(k[0] for k in base_elo['pre_game'])
    canon = team_canonical_map(tg, rf)
    games = box_game_frame(tg)
    games = games[games.game_date > last_odds_date]
    seasons_seq = games.season.tolist()
    assert seasons_seq == sorted(seasons_seq), 'continuation seasons out of order'

    ratings = dict(base_elo['final_ratings'])
    current_season = '2023-24'
    pre, post = {}, {}
    for g in games.itertuples():
        if g.season != current_season:
            for t in ratings:
                ratings[t] = (rf.ELO_SEASON_CARRYOVER * ratings[t]
                              + (1.0 - rf.ELO_SEASON_CARRYOVER)
                              * rf.ELO_MEAN_REVERT_TARGET)
            current_season = g.season
        home, away = canon[int(g.home_id)], canon[int(g.away_id)]
        home_elo = ratings.get(home, rf.ELO_BASE)
        away_elo = ratings.get(away, rf.ELO_BASE)
        expected_home = rf._elo_expected_home(home_elo, away_elo)
        pre[g.game_id] = dict(home_elo=home_elo, away_elo=away_elo,
                              home_expected=expected_home)
        margin = float(g.home_pts - g.away_pts)
        if flip_gid is not None and g.game_id == flip_gid:
            margin = -margin
        home_won = margin > 0
        if home_won:
            diff_winner = (home_elo + rf.ELO_HOME_ADVANTAGE) - away_elo
        else:
            diff_winner = away_elo - (home_elo + rf.ELO_HOME_ADVANTAGE)
        shift = (rf.ELO_K * rf._elo_mov_multiplier(margin, diff_winner)
                 * ((1.0 if home_won else 0.0) - expected_home))
        ratings[home] = home_elo + shift
        ratings[away] = away_elo - shift
        post[g.game_id] = dict(
            home_elo=ratings[home], away_elo=ratings[away],
            home_expected=rf._elo_expected_home(ratings[home], ratings[away]))
    return dict(pre=pre, post=post, games=games, last_odds_date=last_odds_date,
                n_continuation_games=len(games))


def elo_offbyone_probe(rf, tg, base_elo, elo_cont, n_samples=5):
    """Explicit off-by-one test on the highest-risk code path: flip one
    game's outcome and assert (a) that game's own PRE-GAME feature is
    bit-identical, (b) its post-game rating changes, (c) the home team's
    next game's pre-game rating changes. Run on games spread across the
    continuation, restricted to the eval seasons."""
    games = elo_cont['games']
    ev = games[games.season.isin(HELD_OUT_SEASONS)].reset_index(drop=True)
    picks = sorted({0, len(ev) // 4, len(ev) // 2, 3 * len(ev) // 4, len(ev) - 1})[:n_samples]
    checks = []
    for i in picks:
        g = ev.iloc[i]
        flipped = continue_elo(rf, tg, base_elo=base_elo, flip_gid=g.game_id)
        same_pre = (flipped['pre'][g.game_id] == elo_cont['pre'][g.game_id])
        post_changed = (flipped['post'][g.game_id]['home_elo']
                        != elo_cont['post'][g.game_id]['home_elo'])
        canon = team_canonical_map(tg, rf)
        home_c = canon[int(g.home_id)]
        later = games[games.game_date > g.game_date]
        nxt = None
        for r in later.itertuples():
            if canon[int(r.home_id)] == home_c or canon[int(r.away_id)] == home_c:
                nxt = r
                break
        next_changed = None
        if nxt is not None:
            side = 'home_elo' if canon[int(nxt.home_id)] == home_c else 'away_elo'
            next_changed = (flipped['pre'][nxt.game_id][side]
                            != elo_cont['pre'][nxt.game_id][side])
        ok = bool(same_pre and post_changed and (next_changed is not False))
        checks.append(dict(game_id=str(g.game_id), game_date=g.game_date,
                           season=g.season, own_pre_unchanged=bool(same_pre),
                           own_post_changed=bool(post_changed),
                           next_game_pre_changed=next_changed, ok=ok))
    return dict(n_probed=len(checks), all_ok=all(c['ok'] for c in checks),
                probes=checks)


def elo_replay_consistency(rf, base_elo):
    """The step-1 cache's elo_pre table must be reproduced exactly by a fresh
    replay (proves the replay this eval continues from is the same one the
    training features came from)."""
    con = sqlite3.connect(FEATURE_CACHE_RO, uri=True)
    rows = con.execute('SELECT date, home, away, home_elo, away_elo FROM elo_pre').fetchall()
    con.close()
    worst = 0.0
    n_missing = 0
    for d, h, a, he, ae in rows:
        v = base_elo['pre_game'].get((d, h, a))
        if v is None:
            n_missing += 1
            continue
        worst = max(worst, abs(v['home_elo'] - he), abs(v['away_elo'] - ae))
    return dict(n_cache_rows=len(rows), n_missing=n_missing,
                max_abs_diff=worst, ok=bool(n_missing == 0 and worst < 1e-9))


# ---------------------------------------------------------------------------
# Rest features (built by the step-1 builder itself)
# ---------------------------------------------------------------------------
def rest_from_box(rf, tg, seasons, leak=False):
    """{game_id: (b2b_h, b2b_a, 3in4_h, 3in4_a, rest_diff)} via
    retrain_features.build_rest_features on box_scores game dates.

    `leak=True` is the positive control: rest measured INCLUDING the game's
    own date (previous game := the game itself), i.e. rest=0 both sides, so
    b2b flags collapse to False and rest_diff to 0. Rest is built from the
    schedule and cannot leak outcomes; the control demonstrates the columns
    are live inputs (predictions must move when they are perturbed)."""
    canon = team_canonical_map(tg, rf)
    games = box_game_frame(tg)
    games = games[games.season.isin(seasons)]
    out = {}
    if leak:
        rest34 = rest_from_box(rf, tg, seasons, leak=False)
        for g in games.itertuples():
            _, _, h3, a3, _ = rest34[g.game_id]
            out[g.game_id] = (0.0, 0.0, h3, a3, 0.0)
        return out
    glist = [dict(date=g.game_date, home=canon[int(g.home_id)],
                  away=canon[int(g.away_id)], season=g.season)
             for g in games.itertuples()]
    rest = rf.build_rest_features(glist)
    for g in games.itertuples():
        v = rest[(g.game_date, canon[int(g.home_id)], canon[int(g.away_id)])]
        out[g.game_id] = (float(v['home_b2b']), float(v['away_b2b']),
                          float(v['home_3in4']), float(v['away_3in4']),
                          float(v['rest_diff']))
    return out


# ---------------------------------------------------------------------------
# 207-column assembly
# ---------------------------------------------------------------------------
def build_candidate_matrix(tg, rf, elo_pre, rest_map, leak_rolling=False,
                           elo_source=None):
    """Assemble the 207-column matrix for the held-out seasons on EXACTLY
    the game list build_games() produces (same skip rule as the production
    backtest: season-opening games with no prior-game snapshot are skipped).
    Returns (X207, meta_df, skipped)."""
    stat_n = len(roll_stat_cols(rf))
    nan_block = np.full(stat_n, np.nan)
    X_parts, metas, skipped_all = [], [], []
    for season in HELD_OUT_SEASONS:
        Xb, meta, skipped = build_games(tg, season, convention='training')
        skipped_all += skipped
        roll = rolling_from_box(tg, rf, season, leak=leak_rolling)
        rows = []
        for i, m in enumerate(meta.itertuples()):
            vec = [Xb[i]]
            for k in ROLLING_KS:
                for tid in (m.home_id, m.away_id):
                    blocks = roll.get((tid, m.game_date))
                    vec.append(nan_block if blocks is None else blocks[k][0])
            e = (elo_source or elo_pre)[m.game_id]
            vec.append(np.array([e['home_elo'], e['away_elo'],
                                 e['home_elo'] - e['away_elo'],
                                 e['home_expected']]))
            vec.append(np.array(rest_map[m.game_id], dtype=float))
            row = np.concatenate(vec)
            assert row.shape[0] == 207
            rows.append(row)
        X_parts.append(np.vstack(rows))
        metas.append(meta)
    X = np.vstack(X_parts)
    meta = pd.concat(metas, ignore_index=True)
    assert X.shape == (len(meta), 207)
    return X, meta, skipped_all


def predict_candidate(booster, X, cols):
    d = xgb.DMatrix(X, feature_names=cols)
    return booster.predict(d)[:, 1]


# ---------------------------------------------------------------------------
# Task-A checks that do not touch sealed outcomes
# ---------------------------------------------------------------------------
def calibrator_replication_check(config):
    """Applying the stored isotonic to the stored validation predictions must
    reproduce the step-2 validation Brier (0.2167397466181913). Proves the
    calibrator is being applied exactly as training_config specifies, using
    only non-sealed (2022-24 validation) data."""
    preds = np.load(os.path.join(CAND_DIR, 'work', 'valpreds_full_depth3_eta0.05.npy'))
    con = sqlite3.connect(TRAINING_RO, uri=True)
    yv = pd.read_sql_query(
        "select \"Home-Team-Win\" y from training where season in ('2022-23','2023-24') "
        "order by Date, home, away", con)['y'].values.astype(float)
    con.close()
    assert len(preds) == len(yv), 'validation prediction/label length mismatch'
    p_cal = apply_isotonic(preds, config['calibrator']['isotonic'])
    brier = float(np.mean((p_cal - yv) ** 2))
    expected = 0.2167397466181913
    return dict(n=int(len(yv)), brier_reproduced=brier, brier_expected=expected,
                abs_diff=abs(brier - expected), ok=bool(abs(brier - expected) < 5e-7))


def candidate_in_sample_sanity(booster, rf):
    """Score the candidate on its own training table (2012-24). In-sample
    only -- proves the 207-vector layout is what the booster expects, with
    the shifted-column control collapsing toward the base rate."""
    cols = candidate_column_order(rf)
    con = sqlite3.connect(TRAINING_RO, uri=True)
    df = pd.read_sql_query('select * from training order by Date, home, away', con)
    con.close()
    X = df[cols].astype(float).values
    y = df['Home-Team-Win'].astype(int).values
    p = predict_candidate(booster, X, cols)
    acc = float(np.mean((p > 0.5).astype(int) == y))
    shuf = np.concatenate([X[:, 1:], X[:, :1]], axis=1)
    ps = predict_candidate(booster, shuf, cols)
    return dict(
        n=int(len(y)),
        in_sample_accuracy_pct=round(acc * 100, 2),
        home_win_rate_pct=round(float(np.mean(y)) * 100, 2),
        control_accuracy_with_shifted_columns_pct=round(
            float(np.mean((ps > 0.5).astype(int) == y)) * 100, 2),
        step2_validation_accuracy_pct=64.44,
        note=('In-sample on the candidate\'s own 15,110 training rows; not an accuracy '
              'claim. Expect it modestly above the 64.44% step-2 validation accuracy; '
              'the shifted-column control must collapse toward the base rate.'),
    )


def run_candidate_checks(tg, rf, booster, config):
    """All Task-A gates. Returns (checks_dict, all_pass)."""
    print('\n=============== CANDIDATE TASK-A CHECKS ===============')
    print('Column order / booster feature-name assertions ... OK (asserted at load)')

    cal = calibrator_replication_check(config)
    print(f"Calibrator replication: Brier {cal['brier_reproduced']:.10f} vs "
          f"expected {cal['brier_expected']:.10f} -> {'OK' if cal['ok'] else 'FAIL'}")

    sanity = candidate_in_sample_sanity(booster, rf)
    print(f"In-sample sanity: {sanity['in_sample_accuracy_pct']}% on n={sanity['n']} "
          f"training rows; shifted-column control "
          f"{sanity['control_accuracy_with_shifted_columns_pct']}%")
    sanity_ok = (sanity['in_sample_accuracy_pct'] >= sanity['step2_validation_accuracy_pct']
                 and sanity['control_accuracy_with_shifted_columns_pct']
                 < sanity['in_sample_accuracy_pct'] - 5)

    print('Cross-validating box-score rolling vs step-1 snapshot-diff rolling '
          f'on {CROSSVAL_SEASONS} ...')
    cv = crossval_rolling(tg, rf)
    for f, v in cv['families'].items():
        print(f"  {f:9s}: {v['within_tolerance']}/{v['n']} within tolerance "
              f"({v['pct']}%), worst diff/tol {v['worst_diff_over_tolerance']}")
    print(f"  rows: {cv['n_cache_rows']} cache rows, {cv['n_missing_in_box_path']} missing, "
          f"{cv['n_gp_mismatch']} gp-mismatch, {cv['n_window_mismatch']} window-mismatch, "
          f"{cv['n_inexact_window_rows_excluded']} inexact-window excluded "
          f"-> bound {'PASS' if cv['bound_pass'] else 'FAIL'}")

    print('Replaying step-1 odds Elo and checking it reproduces the cache ...')
    base_elo = rf.build_elo_odds()
    elo_ok = elo_replay_consistency(rf, base_elo)
    print(f"  {elo_ok['n_cache_rows']} cache rows, max |diff| {elo_ok['max_abs_diff']:.2e} "
          f"-> {'OK' if elo_ok['ok'] else 'FAIL'}")

    print('Continuing Elo over box_scores and running the off-by-one probe ...')
    elo_cont = continue_elo(rf, tg, base_elo=base_elo)
    print(f"  continuation: {elo_cont['n_continuation_games']} games after "
          f"{elo_cont['last_odds_date']}")
    probe = elo_offbyone_probe(rf, tg, base_elo, elo_cont)
    print(f"  off-by-one probe on {probe['n_probed']} games: "
          f"{'ALL OK' if probe['all_ok'] else 'FAIL'}")

    checks = dict(
        column_order_assertions='PASS (hard asserts at model load)',
        calibrator_replication=cal,
        in_sample_sanity=sanity,
        rolling_crossval=cv,
        elo_replay_consistency=elo_ok,
        elo_continuation=dict(last_odds_date=elo_cont['last_odds_date'],
                              n_continuation_games=elo_cont['n_continuation_games'],
                              note=('Continuation includes the 2023-24 playoff games after '
                                    '2024-04-28 (better burn-in, strictly pre-eval); the 72 '
                                    'regular-season games missing from the odds archive '
                                    'remain missing, exactly as in the training-side replay.')),
        elo_offbyone_probe=probe,
    )
    all_pass = bool(cal['ok'] and sanity_ok and cv['bound_pass']
                    and elo_ok['ok'] and probe['all_ok'])
    print(f"TASK-A GATES: {'ALL PASS' if all_pass else 'FAILED -- sealed eval must not run'}")
    return checks, all_pass, base_elo, elo_cont


# ---------------------------------------------------------------------------
# Paired McNemar (generic) and calibration threshold helpers
# ---------------------------------------------------------------------------
def paired_mcnemar(ok_a, ok_b):
    """Exact two-sided binomial McNemar on the discordant pairs.
    ok_a / ok_b are boolean arrays: 'pick was correct' for each side."""
    from math import comb
    ok_a = np.asarray(ok_a, dtype=bool)
    ok_b = np.asarray(ok_b, dtype=bool)
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    n = b + c
    if n == 0:
        return dict(a_only_correct=b, b_only_correct=c, n_discordant=0, p_value=1.0)
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return dict(a_only_correct=b, b_only_correct=c, n_discordant=n,
                p_value=round(min(1.0, 2 * tail), 6))


def calibration_thresholds(df, prob_edges, brier_max=0.2142, logloss_max=0.6177):
    """Threshold 3: reliability of calibrated P(home win), n>=100 buckets."""
    rel = reliability_home(df, prob_edges)
    big = [b for b in rel if b['n'] >= 100]
    errs = [abs(b['calibration_error_pp']) for b in big]
    brier = float(np.mean((df.p_home - df.home_win) ** 2))
    logloss = float(-np.mean(
        df.home_win * np.log(np.clip(df.p_home, 1e-9, 1)) +
        (1 - df.home_win) * np.log(np.clip(1 - df.p_home, 1e-9, 1))))
    return dict(
        reliability=rel,
        n_buckets_ge_100=len(big),
        all_ge100_buckets_within_5pp=bool(all(e <= 5.0 for e in errs)),
        max_abs_bucket_error_pp=round(max(errs), 2) if errs else None,
        mean_abs_bucket_error_pp=round(float(np.mean(errs)), 2) if errs else None,
        mean_within_3pp=bool(errs and float(np.mean(errs)) <= 3.0),
        brier=round(brier, 4), brier_max=brier_max, brier_ok=bool(brier <= brier_max),
        log_loss=round(logloss, 4), logloss_max=logloss_max,
        logloss_ok=bool(logloss <= logloss_max),
    )


# ---------------------------------------------------------------------------
# The sealed one-shot evaluation
# ---------------------------------------------------------------------------
def candidate_main(args):
    con_team = sqlite3.connect(TEAMDATA, uri=True)
    con_ds = sqlite3.connect(DATASET, uri=True)
    print('Checking production feature order against the training table ...')
    assert_feature_order(con_ds)

    rf = load_rf()
    booster_c, manifest, config = load_candidate(rf)
    cols = candidate_column_order(rf)

    print('Loading box scores ...')
    tg = load_team_games(con_team)
    print(f'  {len(tg)} team-game rows, {tg.game_id.nunique()} games')

    checks, all_pass, base_elo, elo_cont = run_candidate_checks(tg, rf, booster_c, config)
    if args.candidate_checks:
        con_team.close(); con_ds.close()
        return
    if not all_pass:
        print('\nABORT: Task-A gates failed; the sealed evaluation was NOT run.')
        con_team.close(); con_ds.close()
        sys.exit(1)

    # ---- sealed evaluation (one shot) -------------------------------------
    print('\n=============== SEALED ONE-SHOT EVALUATION ===============')
    rest_map = rest_from_box(rf, tg, HELD_OUT_SEASONS)
    X, meta, skipped = build_candidate_matrix(tg, rf, elo_cont['pre'], rest_map)

    # candidate: raw -> stored isotonic -> pick
    p_raw = predict_candidate(booster_c, X, cols)
    p_cal = apply_isotonic(p_raw, config['calibrator']['isotonic'])
    n_cal_ties = int(np.sum(p_cal == 0.5))
    pred_c = np.where(p_cal == 0.5, (p_raw > 0.5), (p_cal > 0.5)).astype(int)

    # production model on the IDENTICAL games via the existing harness path
    booster_old = load_model()
    frames_old = []
    for s in HELD_OUT_SEASONS:
        Xo, mo, _ = build_games(tg, s, convention='training')
        po = booster_old.predict(xgb.DMatrix(Xo))
        mo = mo.copy()
        mo['p_home'] = po[:, 1]
        mo['pred'] = (po[:, 1] > po[:, 0]).astype(int)
        frames_old.append(mo)
    old = pd.concat(frames_old, ignore_index=True)
    assert list(old.game_id) == list(meta.game_id), \
        'old/candidate game lists diverged -- reconciliation failed'

    df = add_baselines(meta.copy())
    df['p_raw'] = p_raw
    df['p_home'] = p_cal                    # calibrated prob is the reported prob
    df['pred'] = pred_c
    df['conf'] = np.maximum(p_cal, 1 - p_cal)
    df['old_pred'] = old['pred'].values
    df['old_p_home'] = old['p_home'].values

    y = df.home_win.values
    ok_c = (df.pred.values == y)
    ok_old = (df.old_pred.values == y)
    ok_rec = (df.pick_better_record.values == y)
    month = df.game_date.str.slice(5, 7).astype(int)
    octdec = month.isin([10, 11, 12]).values

    print(f'games evaluated: {len(df)} (skipped {len(skipped)} openers), '
          f'candidate {ok_c.mean()*100:.2f}% vs old {ok_old.mean()*100:.2f}%')

    # ---- leakage positive controls on the new families ---------------------
    print('Running leakage positive controls ...')
    controls = {}
    variants = dict(
        rolling_leq_cutoff=dict(leak_rolling=True),
        elo_post_game=dict(elo_source=elo_cont['post']),
    )
    for name, kw in variants.items():
        Xv, mv, _ = build_candidate_matrix(tg, rf, elo_cont['pre'], rest_map, **kw)
        assert list(mv.game_id) == list(meta.game_id)
        pv = apply_isotonic(predict_candidate(booster_c, Xv, cols),
                            config['calibrator']['isotonic'])
        predv = np.where(pv == 0.5, (predict_candidate(booster_c, Xv, cols) > 0.5),
                         (pv > 0.5)).astype(int)
        controls[name] = dict(
            accuracy_pct=round(float((predv == y).mean()) * 100, 2),
            picks_changed=int(np.sum(predv != df.pred.values)),
            delta_vs_strict_pp=round(float((predv == y).mean() - ok_c.mean()) * 100, 2))
    rest_leak = rest_from_box(rf, tg, HELD_OUT_SEASONS, leak=True)
    Xr, mr, _ = build_candidate_matrix(tg, rf, elo_cont['pre'], rest_leak)
    pr = apply_isotonic(predict_candidate(booster_c, Xr, cols),
                        config['calibrator']['isotonic'])
    predr = (pr > 0.5).astype(int)
    controls['rest_including_same_day'] = dict(
        accuracy_pct=round(float((predr == y).mean()) * 100, 2),
        picks_changed=int(np.sum(predr != df.pred.values)),
        delta_vs_strict_pp=round(float((predr == y).mean() - ok_c.mean()) * 100, 2),
        note=('Rest is schedule-derived and cannot leak outcomes; this control shows '
              'the rest columns are live inputs (picks move when perturbed).'))
    shuf = np.concatenate([X[:, 1:], X[:, :1]], axis=1)
    psh = apply_isotonic(predict_candidate(booster_c, shuf, cols),
                         config['calibrator']['isotonic'])
    controls['column_shift'] = dict(
        accuracy_pct=round(float(((psh > 0.5).astype(int) == y).mean()) * 100, 2),
        note='every column shifted one position; must collapse toward the base rate')
    controls['strict_accuracy_pct'] = round(float(ok_c.mean()) * 100, 2)
    for k, v in controls.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['accuracy_pct']}%"
                  + (f" ({v.get('picks_changed')} picks changed)" if 'picks_changed' in v else ''))

    # ---- results tables -----------------------------------------------------
    conf_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    prob_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]

    old_df = df.copy()
    old_df['pred'] = df.old_pred
    old_df['p_home'] = df.old_p_home
    old_df['conf'] = np.maximum(old_df.p_home, 1 - old_df.p_home)

    def _slices(frame):
        return dict(
            overall=acc_block(frame),
            by_season={s: acc_block(frame[frame.season == s]) for s in HELD_OUT_SEASONS},
            by_season_type={t: acc_block(frame[frame.season_type == t])
                            for t in ['Regular Season', 'Playoffs']},
            oct_dec=acc_block(frame[octdec]),
        )

    results_c = _slices(df)
    results_old = _slices(old_df)
    results_c['calibration_by_confidence'] = calibration(df, conf_edges)
    cal3 = calibration_thresholds(df, prob_edges)

    m_old = paired_mcnemar(ok_c, ok_old)
    m_rec = paired_mcnemar(ok_c, ok_rec)
    m_old_octdec = paired_mcnemar(ok_c[octdec], ok_old[octdec])

    acc_c = float(ok_c.mean()) * 100
    acc_old = float(ok_old.mean()) * 100
    acc_rec = float(ok_rec.mean()) * 100
    acc_c_od = float(ok_c[octdec].mean()) * 100
    acc_old_od = float(ok_old[octdec].mean()) * 100

    verdicts = dict(
        t1_beats_old_model=dict(
            requirement='paired McNemar vs production model p<0.05 AND candidate better',
            candidate_pct=round(acc_c, 2), old_model_pct=round(acc_old, 2),
            mcnemar=m_old,
            PASS=bool(acc_c > acc_old and m_old['p_value'] < 0.05)),
        t2_beats_better_record=dict(
            requirement='paired McNemar vs better-record baseline p<0.05 AND candidate better',
            candidate_pct=round(acc_c, 2), better_record_pct=round(acc_rec, 2),
            mcnemar=m_rec,
            PASS=bool(acc_c > acc_rec and m_rec['p_value'] < 0.05)),
        t3_calibration=dict(
            requirement=('every n>=100 bucket within +/-5pp, mean |bucket error| <=3pp, '
                         'Brier <= 0.2142, log-loss <= 0.6177 (stored isotonic applied)'),
            detail={k: v for k, v in cal3.items() if k != 'reliability'},
            PASS=bool(cal3['all_ge100_buckets_within_5pp'] and cal3['mean_within_3pp']
                      and cal3['brier_ok'] and cal3['logloss_ok'])),
        t4_oct_dec=dict(
            requirement='Oct-Dec improvement >= +2.5pp over the old model (paired slice)',
            candidate_oct_dec_pct=round(acc_c_od, 2),
            old_model_oct_dec_pct=round(acc_old_od, 2),
            n_oct_dec=int(octdec.sum()),
            improvement_pp=round(acc_c_od - acc_old_od, 2),
            mcnemar=m_old_octdec,
            PASS=bool(acc_c_od - acc_old_od >= 2.5)),
    )

    artifact = dict(
        schema_version=1,
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        generated_by='backtest_model.py --candidate (protocol step 3, sealed one-shot)',
        rerun_reason=None,
        model=dict(
            file='Models/candidate_2026-08/model.json',
            n_features=207,
            architecture=('XGBoost multi:softprob 2-class, max_depth 3, eta 0.05, 67 rounds '
                          '(frozen from chronological validation early stopping), trained on '
                          '2012-24, stored isotonic calibrator applied to P(home win)'),
            manifest='Models/candidate_2026-08/feature_manifest.json',
            training_config='Models/candidate_2026-08/training_config.json'),
        evaluation=dict(
            seasons=HELD_OUT_SEASONS,
            n_games_scored=int(len(df)),
            n_games_available=int(tg[tg.season.isin(HELD_OUT_SEASONS)].game_id.nunique()),
            n_games_skipped=len(skipped),
            skipped_reason='season-opening games (no prior-game snapshot); identical rule for both models',
            game_list_reconciliation=('IDENTICAL: both models evaluated on exactly the same '
                                      f'{len(df)} games, asserted by game_id at runtime'),
            games_by_season={s: int((df.season == s).sum()) for s in HELD_OUT_SEASONS},
            games_by_season_type=df.season_type.value_counts().to_dict(),
            date_range=[df.game_date.min(), df.game_date.max()],
            calibrated_prob_exact_ties_at_0p5=n_cal_ties,
            tie_rule='calibrated 0.5 exactly -> fall back to raw-probability argmax'),
        task_a_checks=checks,
        leakage_positive_controls=controls,
        results_candidate=results_c,
        results_old_model_same_games=results_old,
        paired_tests=dict(
            candidate_vs_old_model=m_old,
            candidate_vs_better_record_baseline=m_rec,
            candidate_vs_old_model_oct_dec=m_old_octdec),
        calibration_reliability_home_prob=cal3['reliability'],
        preregistered_thresholds=verdicts,
        methodology_notes=[
            'Base 106 features reconstructed by the identical build_games() path the '
            'production backtest uses (validated 99.667% cell-exact elsewhere in this file).',
            'Rolling K10/K20 rebuilt from box_scores with strict game_date < d cutoffs; '
            'total team turnovers recovered from the advanced box score; playoff games use '
            'the frozen last-K regular-season games, matching the training-side snapshots.',
            'Elo continued from the step-1 odds-archive replay (verified to reproduce the '
            'training cache exactly) over box_scores games after 2024-04-28, same constants, '
            '(game_date, game_id) order, 25% between-season reversion; features are strictly '
            'pre-game ratings (off-by-one probe in task_a_checks).',
            'Rest features built by retrain_features.build_rest_features itself.',
            'Stored isotonic calibrator applied exactly as training_config specifies; picks '
            'are argmax of the calibrated probability.',
            'No odds are model inputs. Accuracy is not profitability; no ROI is measured.',
            'The sealed set was spent by this run. Thresholds were pre-registered before '
            'any sealed number was seen.',
        ],
    )
    with open(args.out_candidate, 'w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2)
    print(f'\nWrote {args.out_candidate}')

    # ---- console report ----------------------------------------------------
    print('\n================ SEALED RESULTS (CANDIDATE) ================')
    o = results_c['overall']
    print(f"Candidate overall: {o['model_accuracy_pct']}% (n={o['n']}, "
          f"95% CI {o['model_accuracy_95ci'][0]}-{o['model_accuracy_95ci'][1]})")
    print(f"Old model  overall: {results_old['overall']['model_accuracy_pct']}%")
    print(f"Better-record baseline: {round(acc_rec, 2)}%")
    for s in HELD_OUT_SEASONS:
        print(f"  {s}: candidate {results_c['by_season'][s]['model_accuracy_pct']}% "
              f"vs old {results_old['by_season'][s]['model_accuracy_pct']}% "
              f"(n={results_c['by_season'][s]['n']})")
    for t in ('Regular Season', 'Playoffs'):
        print(f"  {t}: candidate {results_c['by_season_type'][t]['model_accuracy_pct']}% "
              f"vs old {results_old['by_season_type'][t]['model_accuracy_pct']}% "
              f"(n={results_c['by_season_type'][t]['n']})")
    print(f"  Oct-Dec: candidate {round(acc_c_od, 2)}% vs old {round(acc_old_od, 2)}% "
          f"(n={int(octdec.sum())})")
    print(f"Candidate Brier {cal3['brier']}  log-loss {cal3['log_loss']}")
    print('\nReliability of calibrated P(home win):')
    for b in cal3['reliability']:
        if b['n'] == 0:
            continue
        print(f"  {b['bucket']:>10} n={b['n']:>5} pred {b['mean_predicted_home_win_pct']:>6.2f}% "
              f"actual {b['actual_home_win_pct']:>6.2f}%  err {b['calibration_error_pp']:>6.2f}pp")
    print('\nPre-registered threshold verdicts:')
    for k, v in verdicts.items():
        print(f"  {k}: {'PASS' if v['PASS'] else 'FAIL'}")
    con_team.close()
    con_ds.close()


def _dispatch():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true', help='run reconstruction validation only')
    ap.add_argument('--out', default=OUT_JSON)
    ap.add_argument('--candidate-checks', action='store_true',
                    help='candidate Task-A checks only (no sealed outcomes revealed)')
    ap.add_argument('--candidate', action='store_true',
                    help='ONE-SHOT sealed evaluation of Models/candidate_2026-08')
    ap.add_argument('--out-candidate', default=CAND_OUT_JSON)
    args = ap.parse_args()
    if args.candidate or args.candidate_checks:
        candidate_main(args)
    else:
        sys.argv = [sys.argv[0]] + (['--validate'] if args.validate else []) \
            + ['--out', args.out]
        main()


if __name__ == '__main__':
    _dispatch()
