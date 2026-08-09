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


if __name__ == '__main__':
    main()
