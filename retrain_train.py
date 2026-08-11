"""
retrain_train.py
================
STEP 2 of the pre-registered model-retrain protocol: build the training
table, run the CHRONOLOGICAL train/validation ablations, calibrate, and
save the final candidate model.

Pre-registered design (no deviations without documenting them):

  * Training table = [base 106 columns from dataset_2012-24_new]
    + [rolling-K blocks, home+away, K in {10, 20}]
    + [pre-game Elo: home, away, diff, home_expected]
    + [rest: b2b both sides, 3-in-4 both sides, rest_diff]
    + label Home-Team-Win.
    K=30 is intentionally skipped: a 30-game window is ~2 months of
    schedule and overlaps heavily with the season-to-date averages already
    present in the base block, while adding 46 more columns to a 15k-row
    table; K=10/20 capture the recent-form signal the base block lacks.
  * Split: train = 2012-13 .. 2021-22, validation = 2022-23 .. 2023-24.
    NEVER shuffled. All tuning decisions come from validation only.
    2024-25 / 2025-26 are the sealed test set and are not touched here.
  * Ablations (train on train span, score on validation):
      base            - the honest old-architecture reference
      base+rolling
      base+rolling+elo
      full (+rest)
    Grid per ablation: max_depth {3,4,5} x eta {0.01, 0.05}, XGBoost
    multi:softprob 2-class (current production objective), early stopping
    on validation logloss.
  * Calibration: Platt (sigmoid) and isotonic fit on the chosen model's
    validation predictions; chosen by validation Brier.
  * Final candidate: chosen config refit ONCE on all of 2012-24 with the
    round count frozen from validation early stopping. Saved to
    Models/candidate_2026-08/ (a NEW directory; no existing Models/ file
    is touched). The sealed test evaluation happens in a later step.

Season openers have no rolling features -> NaN (XGBoost handles natively).

Determinism: SEED=42 everywhere, tree_method='hist' on CPU (deterministic
in xgboost 3.x), chronological split (no shuffling), canonical game order.

Usage:
    venv/Scripts/python.exe retrain_train.py --phase build
    venv/Scripts/python.exe retrain_train.py --phase ablate
    venv/Scripts/python.exe retrain_train.py --phase finalize
    venv/Scripts/python.exe retrain_train.py --phase all
The ablate phase checkpoints after every fit and resumes automatically.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

REPO = os.path.abspath(os.path.dirname(__file__))
DATASET_DB = os.path.join(REPO, "Data", "dataset.sqlite")
DATASET_TABLE = "dataset_2012-24_new"
FEATURE_CACHE_DB = os.path.join(REPO, "Data", "retrain_features.sqlite")
TRAINING_DB = os.path.join(REPO, "Data", "retrain_training.sqlite")
TRAINING_TABLE = "training"
CAND_DIR = os.path.join(REPO, "Models", "candidate_2026-08")
WORK_DIR = os.path.join(CAND_DIR, "work")

SEED = 42
TRAIN_SEASONS = ("2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
                 "2017-18", "2018-19", "2019-20", "2020-21", "2021-22")
VAL_SEASONS = ("2022-23", "2023-24")

#: Columns the current production training script drops (label + metadata).
DROP_COLS = ["Score", "Home-Team-Win", "TEAM_NAME", "Date", "TEAM_NAME.1",
             "Date.1", "OU-Cover", "OU"]

ROLLING_KS = (10, 20)
GRID = [{"max_depth": d, "eta": e} for d in (3, 4, 5) for e in (0.01, 0.05)]
MAX_ROUNDS = 6000
EARLY_STOP = 200

ABLATIONS = ("base", "base+rolling", "base+rolling+elo", "full")


def _load_rf():
    spec = importlib.util.spec_from_file_location(
        "retrain_features", os.path.join(REPO, "src", "Process-Data",
                                         "retrain_features.py"))
    rf = importlib.util.module_from_spec(spec)
    sys.modules["retrain_features"] = rf
    spec.loader.exec_module(rf)
    return rf


def _rolling_stat_cols(rf):
    return list(rf.COUNT_STATS) + list(rf.PCT_STATS) + ["WIN_PCT"]


def _feature_blocks(rf):
    """Feature column names per block, in manifest order."""
    stat_cols = _rolling_stat_cols(rf)
    rolling = []
    for k in ROLLING_KS:
        for side in ("HOME", "AWAY"):
            rolling += [f"R{k}_{side}_{s}" for s in stat_cols]
    elo = ["ELO_HOME", "ELO_AWAY", "ELO_DIFF", "ELO_HOME_EXPECTED"]
    rest = ["REST_HOME_B2B", "REST_AWAY_B2B", "REST_HOME_3IN4",
            "REST_AWAY_3IN4", "REST_DIFF"]
    return rolling, elo, rest


# ---------------------------------------------------------------------------
# Phase 1: build the training table
# ---------------------------------------------------------------------------
def build_table():
    rf = _load_rf()
    stat_cols = _rolling_stat_cols(rf)
    rolling_cols, elo_cols, rest_cols = _feature_blocks(rf)

    con = sqlite3.connect(DATASET_DB)
    data = pd.read_sql_query(f'select * from "{DATASET_TABLE}"', con,
                             index_col="index")
    con.close()
    base_cols = [c for c in data.columns if c not in DROP_COLS]
    assert len(base_cols) == 106, f"expected 106 base cols, got {len(base_cols)}"

    data["home"] = data["TEAM_NAME"].map(rf.normalize_team)
    data["away"] = data["TEAM_NAME.1"].map(rf.normalize_team)

    # Canonical game list -> season labels and the join universe.
    seasons = [s for s in rf.ODDS_SEASONS if s >= "2012-13"]
    odds_season = {(g["date"], g["home"], g["away"]): g["season"]
                   for g in rf.load_odds_games(seasons=seasons)}

    # Step-1 feature cache (rebuild if missing).
    if not os.path.exists(FEATURE_CACHE_DB):
        print("feature cache missing; rebuilding via retrain_features ...")
        rf.build_cache()
    cache = sqlite3.connect(FEATURE_CACHE_DB)
    roll = {}
    q = ("SELECT date, team, k, " + ", ".join(f'"{c}"' for c in stat_cols) +
         f" FROM rolling_features WHERE k IN ({','.join('?' * len(ROLLING_KS))})")
    for row in cache.execute(q, list(ROLLING_KS)):
        roll[(row[0], row[1], row[2])] = row[3:]
    elo = {(d, h, a): (he, ae, he - ae, exp) for d, h, a, he, ae, exp in
           cache.execute("SELECT date, home, away, home_elo, away_elo, "
                         "home_expected FROM elo_pre")}
    rest = {(d, h, a): (hb, ab, h3, a3, rd) for d, h, a, _, _, hb, ab, h3, a3,
            rd in cache.execute("SELECT * FROM rest_features")}
    cache.close()

    nan_block = (float("nan"),) * len(stat_cols)
    rows, dropped, opener_sides = [], [], 0
    for idx, r in data.iterrows():
        key = (r["Date"], r["home"], r["away"])
        season = odds_season.get(key)
        if season is None or key not in elo or key not in rest:
            dropped.append({"index": int(idx), "date": r["Date"],
                            "home": r["home"], "away": r["away"],
                            "reason": "not in canonical odds game list "
                                      "(known corrupt/ungraded rows)"})
            continue
        feat = [r[c] for c in base_cols]
        for k in ROLLING_KS:
            for team in (r["home"], r["away"]):
                block = roll.get((r["Date"], team, k))
                if block is None:
                    opener_sides += 1
                    block = nan_block
                feat.extend(block)
        feat.extend(elo[key])
        feat.extend(rest[key])
        rows.append([r["Date"], r["home"], r["away"], season,
                     int(r["Home-Team-Win"])] + feat)

    feature_cols = base_cols + rolling_cols + elo_cols + rest_cols
    df = pd.DataFrame(rows, columns=["Date", "home", "away", "season",
                                     "Home-Team-Win"] + feature_cols)
    df = df.sort_values(["Date", "home", "away"]).reset_index(drop=True)

    out = sqlite3.connect(TRAINING_DB)
    df.to_sql(TRAINING_TABLE, out, if_exists="replace", index=False)
    out.close()

    os.makedirs(CAND_DIR, exist_ok=True)
    manifest = {
        "protocol_step": 2,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "label": "Home-Team-Win",
        "key_columns": ["Date", "home", "away", "season"],
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "blocks": {
            "base": {"n": len(base_cols),
                     "source": f"{DATASET_TABLE} (Data/dataset.sqlite), the "
                               "current model's exact inputs, joined not "
                               "rebuilt"},
            "rolling": {"n": len(rolling_cols), "ks": list(ROLLING_KS),
                        "source": "Data/retrain_features.sqlite "
                                  "rolling_features (step 1 snapshot diffs)"},
            "elo": {"n": len(elo_cols),
                    "source": "Data/retrain_features.sqlite elo_pre "
                              "(pre-game ratings, 2007-08 burn-in)"},
            "rest": {"n": len(rest_cols),
                     "source": "Data/retrain_features.sqlite rest_features"},
        },
        "conventions": {
            "season_openers": "No rolling features exist for a team's first "
                              "game of a season; all rolling columns for "
                              "that side are NaN. XGBoost handles NaN "
                              "natively; do NOT impute at eval time either.",
            "rolling_partial_windows": "When a team has played fewer than K "
                                       "games, the rolling value is the "
                                       "season-to-date average (is_partial "
                                       "in the step-1 cache).",
            "tov_definition": "Rolling TOV comes from the TeamData snapshot "
                              "tables and is TOTAL team turnovers including "
                              "team turnovers (shot-clock/8-second/5-second "
                              "violations), consistent with the base "
                              "block's season-to-date TOV. Eval-side "
                              "feature construction MUST recover total TOV "
                              "from the advanced box score "
                              "(estimatedTeamTurnoverPercentage * "
                              "possessions / 100, rounded) exactly as "
                              "backtest_model.py already does; the "
                              "traditional box team row is player-summed "
                              "and runs ~0.7/game low.",
            "rest": "rest = calendar-day gap to previous same-season game, "
                    "capped at 7; season opener = 7; b2b = (rest == 1); "
                    "3in4 = 3rd-or-more game in the 4-day window ending on "
                    "game date; rest_diff = home_rest - away_rest.",
            "elo": "K=20, home advantage 70, 538 margin-of-victory "
                   "multiplier, 25% between-season reversion toward 1505, "
                   "burn-in from 2007-08 over the odds archive. ELO_DIFF = "
                   "ELO_HOME - ELO_AWAY (raw, without the +70).",
            "k30_skipped": "Pre-registered choice: K=30 (~2 months) overlaps "
                           "heavily with the base block's season-to-date "
                           "averages and would add 46 columns to a 15k-row "
                           "table for little independent signal.",
        },
        "join_report": {
            "dataset_rows": int(len(data)),
            "rows_kept": int(len(df)),
            "rows_dropped": dropped,
            "note_2023_24": "The odds archive (and hence the dataset) is "
                            "missing 72 regular-season games after the 2024 "
                            "All-Star break - pre-existing gap, not filled.",
            "opener_nan_side_blocks": int(opener_sides),
        },
    }
    with open(os.path.join(CAND_DIR, "feature_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"training table: {len(df)} rows x {len(feature_cols)} features "
          f"-> {TRAINING_DB}")
    print(f"dropped {len(dropped)} rows; opener NaN side-blocks: "
          f"{opener_sides}")
    print("season counts:")
    print(df["season"].value_counts().sort_index().to_string())
    return df


# ---------------------------------------------------------------------------
# Phase 2: chronological ablations
# ---------------------------------------------------------------------------
def _load_training():
    con = sqlite3.connect(TRAINING_DB)
    df = pd.read_sql_query(f'select * from "{TRAINING_TABLE}"', con)
    con.close()
    return df


def _metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    acc = float(np.mean((p > 0.5).astype(float) == y))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    return {"n": int(len(y)), "accuracy": acc, "logloss": logloss,
            "brier": brier}


def _ablation_cols(name, rf):
    rolling_cols, elo_cols, rest_cols = _feature_blocks(rf)
    con = sqlite3.connect(TRAINING_DB)
    all_cols = [r[1] for r in con.execute(
        f'PRAGMA table_info("{TRAINING_TABLE}")')]
    con.close()
    base_cols = [c for c in all_cols
                 if c not in ("Date", "home", "away", "season",
                              "Home-Team-Win")
                 and c not in rolling_cols + elo_cols + rest_cols]
    if name == "base":
        return base_cols
    if name == "base+rolling":
        return base_cols + rolling_cols
    if name == "base+rolling+elo":
        return base_cols + rolling_cols + elo_cols
    if name == "full":
        return base_cols + rolling_cols + elo_cols + rest_cols
    raise ValueError(name)


def _fit(df_train, df_val, cols, params, seed=SEED):
    """Train multi:softprob 2-class with early stopping on validation
    logloss. Returns (booster, best_iteration, val_probs_home_win)."""
    xt = df_train[cols].astype(float)
    xv = df_val[cols].astype(float)
    yt = df_train["Home-Team-Win"].values
    yv = df_val["Home-Team-Win"].values
    dtrain = xgb.DMatrix(xt, label=yt, feature_names=list(cols))
    dval = xgb.DMatrix(xv, label=yv, feature_names=list(cols))
    full_params = {
        "max_depth": params["max_depth"],
        "eta": params["eta"],
        "objective": "multi:softprob",
        "num_class": 2,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": seed,
    }
    booster = xgb.train(full_params, dtrain, MAX_ROUNDS,
                        evals=[(dval, "val")],
                        early_stopping_rounds=EARLY_STOP,
                        verbose_eval=False)
    best_it = booster.best_iteration
    probs = booster.predict(dval, iteration_range=(0, best_it + 1))[:, 1]
    return booster, best_it, probs


def run_ablations():
    rf = _load_rf()
    df = _load_training()
    df_train = df[df["season"].isin(TRAIN_SEASONS)]
    df_val = df[df["season"].isin(VAL_SEASONS)]
    val_month = pd.to_datetime(df_val["Date"]).dt.month
    oct_dec = val_month.isin([10, 11, 12]).values
    print(f"train {len(df_train)} rows ({TRAIN_SEASONS[0]}..{TRAIN_SEASONS[-1]}), "
          f"val {len(df_val)} rows ({', '.join(VAL_SEASONS)}), "
          f"Oct-Dec slice {int(oct_dec.sum())} rows")

    os.makedirs(WORK_DIR, exist_ok=True)
    ckpt_path = os.path.join(WORK_DIR, "ablation_checkpoint.json")
    results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
        print(f"resuming: {len(results)} fits already done")

    for ablation in ABLATIONS:
        cols = _ablation_cols(ablation, rf)
        for params in GRID:
            key = f"{ablation}|depth={params['max_depth']}|eta={params['eta']}"
            if key in results:
                continue
            booster, best_it, probs = _fit(df_train, df_val, cols, params)
            entry = {
                "ablation": ablation,
                "n_features": len(cols),
                "params": params,
                "best_iteration": int(best_it),
                "n_rounds_used": int(best_it) + 1,
                "validation": _metrics(df_val["Home-Team-Win"], probs),
                "validation_oct_dec": _metrics(
                    df_val["Home-Team-Win"].values[oct_dec], probs[oct_dec]),
            }
            results[key] = entry
            np.save(os.path.join(WORK_DIR, f"valpreds_{key.replace('|', '_').replace('=', '')}.npy"), probs)
            # Keep the booster for the full ablation (importances + reuse).
            booster.save_model(os.path.join(
                WORK_DIR, f"model_{key.replace('|', '_').replace('=', '')}.json"))
            with open(ckpt_path, "w") as f:
                json.dump(results, f, indent=2)
            v, o = entry["validation"], entry["validation_oct_dec"]
            print(f"{key}: rounds={entry['n_rounds_used']} "
                  f"acc={v['accuracy']:.4f} ll={v['logloss']:.4f} "
                  f"brier={v['brier']:.4f} | OctDec acc={o['accuracy']:.4f}",
                  flush=True)
    print("ablations complete")
    return results


# ---------------------------------------------------------------------------
# Phase 3: choose, calibrate, finalize
# ---------------------------------------------------------------------------
def _platt_fit(p, y):
    """Standard Platt scaling on the logit of the predicted probability."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z.reshape(-1, 1), y)
    return lr


def _platt_apply(lr, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return lr.predict_proba(z.reshape(-1, 1))[:, 1]


def _reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            out.append({"bucket": f"[{lo:.1f},{hi:.1f})", "n": 0})
            continue
        out.append({"bucket": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                    "mean_pred": float(np.mean(p[m])),
                    "empirical": float(np.mean(y[m]))})
    return out


def finalize():
    from sklearn.isotonic import IsotonicRegression
    rf = _load_rf()
    df = _load_training()
    df_train = df[df["season"].isin(TRAIN_SEASONS)]
    df_val = df[df["season"].isin(VAL_SEASONS)]
    yv = df_val["Home-Team-Win"].values.astype(float)
    val_month = pd.to_datetime(df_val["Date"]).dt.month
    oct_dec = val_month.isin([10, 11, 12]).values

    ckpt_path = os.path.join(WORK_DIR, "ablation_checkpoint.json")
    with open(ckpt_path) as f:
        results = json.load(f)
    assert len(results) == len(ABLATIONS) * len(GRID), "ablations incomplete"

    # ---- model selection: validation logloss (proper score), pre-registered.
    chosen_key = min(results, key=lambda k: results[k]["validation"]["logloss"])
    chosen = results[chosen_key]
    best_per_ablation = {
        a: min((r for r in results.values() if r["ablation"] == a),
               key=lambda r: r["validation"]["logloss"])
        for a in ABLATIONS}

    # ---- decision rule: full vs base at each one's best config.
    fb, bb = best_per_ablation["full"], best_per_ablation["base"]
    advance = (
        fb["validation"]["accuracy"] >= bb["validation"]["accuracy"]
        and fb["validation_oct_dec"]["accuracy"] >= bb["validation_oct_dec"]["accuracy"]
        and fb["validation"]["brier"] <= bb["validation"]["brier"] + 1e-9)

    # ---- calibration on the chosen model's validation predictions.
    fname = chosen_key.replace("|", "_").replace("=", "")
    probs = np.load(os.path.join(WORK_DIR, f"valpreds_{fname}.npy"))
    platt = _platt_fit(probs, yv)
    p_platt = _platt_apply(platt, probs)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(probs, yv)
    p_iso = iso.predict(probs)
    cal_brier = {"none": _metrics(yv, probs)["brier"],
                 "platt": _metrics(yv, p_platt)["brier"],
                 "isotonic": _metrics(yv, p_iso)["brier"]}
    cal_choice = min(cal_brier, key=cal_brier.get)
    calibrator = {
        "chosen": cal_choice,
        "selection_metric": "validation Brier (pre-registered)",
        "caveat": "Calibrators are fit AND selected on the same validation "
                  "predictions, so the isotonic in-sample Brier is "
                  "optimistic; the sealed-test step must apply the stored "
                  "calibrator as-is and re-measure reliability.",
        "brier": cal_brier,
        "platt": {"coef": float(platt.coef_[0][0]),
                  "intercept": float(platt.intercept_[0]),
                  "input": "logit of raw model probability, clipped to "
                           "[1e-6, 1-1e-6]"},
        "isotonic": {"x_thresholds": [float(x) for x in iso.X_thresholds_],
                     "y_thresholds": [float(y) for y in iso.y_thresholds_],
                     "out_of_bounds": "clip"},
        "reliability_before": _reliability(probs, yv),
        "reliability_after": _reliability(
            {"none": probs, "platt": p_platt, "isotonic": p_iso}[cal_choice],
            yv),
    }

    # ---- feature importances (top 20, gain) for the FULL model's best config.
    fk = next(k for k, r in results.items()
              if r is best_per_ablation["full"])
    full_booster = xgb.Booster()
    full_booster.load_model(os.path.join(
        WORK_DIR, f"model_{fk.replace('|', '_').replace('=', '')}.json"))
    gain = full_booster.get_score(importance_type="gain")
    total_gain = full_booster.get_score(importance_type="total_gain")
    top20 = sorted(total_gain.items(), key=lambda kv: -kv[1])[:20]
    importances = [{"feature": f, "total_gain": round(v, 2),
                    "gain": round(gain[f], 4)} for f, v in top20]
    rolling_cols, elo_cols, rest_cols = _feature_blocks(rf)
    new_cols = set(rolling_cols + elo_cols + rest_cols)
    new_in_top20 = [i["feature"] for i in importances
                    if i["feature"] in new_cols]

    # ---- final refit on ALL 2012-24, rounds frozen from early stopping.
    cols = _ablation_cols(chosen["ablation"], rf)
    x_all = df[cols].astype(float)
    y_all = df["Home-Team-Win"].values
    d_all = xgb.DMatrix(x_all, label=y_all, feature_names=list(cols))
    final_params = {
        "max_depth": chosen["params"]["max_depth"],
        "eta": chosen["params"]["eta"],
        "objective": "multi:softprob",
        "num_class": 2,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": SEED,
    }
    n_rounds = chosen["n_rounds_used"]
    final = xgb.train(final_params, d_all, n_rounds, verbose_eval=False)
    os.makedirs(CAND_DIR, exist_ok=True)
    final.save_model(os.path.join(CAND_DIR, "model.json"))

    training_config = {
        "protocol_step": 2,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "chosen_ablation": chosen["ablation"],
        "chosen_params": final_params,
        "n_boost_rounds": n_rounds,
        "rounds_note": "Frozen from validation early stopping "
                       f"(best_iteration={chosen['best_iteration']} on the "
                       "2022-24 validation span when trained on 2012-22); "
                       "the final model is refit once on all 2012-24 with "
                       "this fixed round count and never early-stopped.",
        "selection_metric": "validation logloss",
        "train_span": list(TRAIN_SEASONS),
        "validation_span": list(VAL_SEASONS),
        "final_fit_span": "2012-13 .. 2023-24 (train + validation)",
        "sealed_test": "2024-25 and 2025-26 - NOT evaluated in this step",
        "seed": SEED,
        "max_rounds_cap": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOP,
        "grid": GRID,
        "versions": {"python": sys.version.split()[0],
                     "xgboost": xgb.__version__,
                     "pandas": pd.__version__,
                     "numpy": np.__version__},
        "n_features": len(cols),
        "training_table": f"{TRAINING_DB}::{TRAINING_TABLE}",
        "calibrator": {k: calibrator[k] for k in
                       ("chosen", "platt", "isotonic", "caveat")},
    }
    with open(os.path.join(CAND_DIR, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    report = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "split": {"train": list(TRAIN_SEASONS), "validation": list(VAL_SEASONS),
                  "n_train": int(len(df_train)), "n_val": int(len(df_val)),
                  "n_val_oct_dec": int(oct_dec.sum()),
                  "shuffled": False},
        "ablation_results": results,
        "best_per_ablation": {a: {"params": r["params"],
                                  "n_rounds_used": r["n_rounds_used"],
                                  "validation": r["validation"],
                                  "validation_oct_dec": r["validation_oct_dec"]}
                              for a, r in best_per_ablation.items()},
        "chosen": {"key": chosen_key, "ablation": chosen["ablation"],
                   "params": chosen["params"],
                   "n_rounds_used": chosen["n_rounds_used"],
                   "validation": chosen["validation"],
                   "validation_oct_dec": chosen["validation_oct_dec"]},
        "calibration": calibrator,
        "full_model_top20_importances_total_gain": importances,
        "new_features_in_top20": new_in_top20,
        "decision_rule": {
            "rule": "advance to sealed test ONLY if full-model validation "
                    ">= base-only on BOTH overall accuracy AND Oct-Dec "
                    "accuracy, calibration (Brier) no worse",
            "full_best": {"params": fb["params"],
                          "validation": fb["validation"],
                          "validation_oct_dec": fb["validation_oct_dec"]},
            "base_best": {"params": bb["params"],
                          "validation": bb["validation"],
                          "validation_oct_dec": bb["validation_oct_dec"]},
            "advance": bool(advance),
        },
    }
    with open(os.path.join(CAND_DIR, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"chosen": report["chosen"],
                      "decision": report["decision_rule"]["advance"],
                      "calibration": cal_brier,
                      "new_in_top20": new_in_top20}, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "ablate", "finalize", "all"],
                    default="all")
    args = ap.parse_args()
    if args.phase in ("build", "all"):
        build_table()
    if args.phase in ("ablate", "all"):
        run_ablations()
    if args.phase in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
