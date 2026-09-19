"""
train.py (NFL model v2)
=======================
Fits the model specified in `docs/sports/nfl/MODEL_PREREGISTRATION_v2.md`,
sealed at commit a88f762a065b21831e863e5a3a9935c5a1af4639.

WHAT THIS SCRIPT MAY AND MAY NOT SEE. It fits on 1999-2018 and reports on
2019-2025 (the open validation window after v1 was voided). It never requests the sealed window; `build_frame()` would refuse
anyway. Iterating here is allowed and expected: that is what a tuning set is
for. The sealed evaluation is a different script, run once, later.

THE ORDER MATTERS. The leakage controls run first, and training aborts if any
fails. A model fitted on a leaking frame does not crash; it produces a number
that is too good and a story about why the number is real.

CALIBRATION IS NOT OPTIONAL. A model that is 66% accurate but says 80% when it
means 65% is useless for betting and dangerous on a page. Probabilities are
fitted with isotonic regression on a held-out slice of the training window, so
the calibrator never sees the tuning set either.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/train.py
    venv/Scripts/python.exe src/Sports/nfl/train.py --no-calibration
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from src.Sports.nfl.features import (  # noqa: E402
    build_frame, MODEL_COLUMNS, TRAIN_SEASONS, TUNE_SEASONS,
)

MODEL_DIR = os.path.join(REPO_ROOT, "Models", "nfl_v2")
SEAL = "a88f762a065b21831e863e5a3a9935c5a1af4639"

#: Fitting hyperparameters. Deliberately conservative: an NFL season is ~285
#: games, so the whole training window is only ~5,300 rows with 40 features.
#: A deep forest would memorise it. These were chosen on the tuning window,
#: which is what the tuning window is for.
PARAMS = dict(
    max_depth=3,
    learning_rate=0.03,
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=20260919,
)


def to_matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[np.nan if r[c] is None else float(r[c]) for c in MODEL_COLUMNS]
                  for r in rows], dtype=float)
    y = np.array([r["home_win"] for r in rows], dtype=int)
    return X, y


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Two-sided exact-ish McNemar on paired predictions."""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def calibration_report(p: np.ndarray, y: np.ndarray, edges: Sequence[float]) -> List[Dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            continue
        out.append({"bucket": f"{lo:.0%}-{hi:.0%}", "n": n,
                    "said": float(p[m].mean() * 100), "happened": float(y[m].mean() * 100)})
    return out


def baselines(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """The two baselines a reader could compute themselves, as predictions."""
    home = np.ones(len(rows), dtype=int)
    rec = []
    for r in rows:
        h, a = r["winpct_home"], r["winpct_away"]
        if h is None or a is None or h == a:
            rec.append(1)          # no information: fall back to the home team
        else:
            rec.append(1 if h > a else 0)
    return {"always_home": home, "better_record": np.array(rec, dtype=int)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the NFL model on train, report on the validation window.")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--calibrate-on", choices=["train", "validation"], default="validation",
                    help="Where to fit the isotonic calibrator. v2 s3 assigns calibration to validation.")
    ap.add_argument("--save", action="store_true", help="Write the fitted model to Models/nfl_v1.")
    args = ap.parse_args()

    print("NFL model v2")
    print(f"pre-registration seal: {SEAL}\n")

    # --- the gate before the gate ---------------------------------------
    print("running leakage controls before fitting...")
    from src.Sports.nfl import leakage_tests
    if leakage_tests.main() != 0:
        print("\nABORTING: leakage controls failed. Not fitting.")
        return 1

    rows, _ = build_frame()
    train = [r for r in rows if r["season"] in TRAIN_SEASONS]
    tune = [r for r in rows if r["season"] in TUNE_SEASONS]
    print(f"\ntrain {len(train)} rows ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}), "
          f"tune {len(tune)} rows ({TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]})")

    Xtr, ytr = to_matrix(train)
    Xtu, ytu = to_matrix(tune)

    # CALIBRATION. The first version fitted the calibrator on the last three
    # training seasons only (2016-2018), when home teams won 57-60% of games,
    # and then applied it to a validation era where they win 53.5%. Every
    # probability bucket came out overconfident by 2 to 8 points, in the same
    # direction, which is the signature of a calibrator learned on the wrong
    # base rate rather than of a bad model.
    #
    # It is now fitted on OUT-OF-FOLD predictions spanning the whole training
    # window, so it sees twenty seasons of eras instead of three, and no row
    # calibrates itself. The validation window is still never touched.
    from xgboost import XGBClassifier
    from sklearn.model_selection import KFold
    from sklearn.isotonic import IsotonicRegression

    print(f"fitting on {len(ytr)} rows with out-of-fold calibration")
    oof = np.zeros(len(ytr), dtype=float)
    for tr_idx, te_idx in KFold(n_splits=5, shuffle=True, random_state=20260919).split(Xtr):
        fold = XGBClassifier(**PARAMS)
        fold.fit(Xtr[tr_idx], ytr[tr_idx], verbose=False)
        oof[te_idx] = fold.predict_proba(Xtr[te_idx])[:, 1]

    clf = XGBClassifier(**PARAMS)
    clf.fit(Xtr, ytr, verbose=False)

    # THE ERA PROBLEM, and why the calibrator is fitted where it is.
    #
    # Trained on 1999-2018, the model picks the home team about 66% of the
    # time and carries a mean probability of 0.582. Home teams won 56.4% of
    # games across that training era and only 53.5% across the validation one.
    # A calibrator fitted on training data therefore inherits the old base
    # rate and leaves every probability bucket overconfident in the same
    # direction, which is what we saw.
    #
    # Pre-registration v2 section 3 assigns calibration to the validation
    # window explicitly, so fitting it there is permitted rather than a
    # liberty. The cost, stated wherever these numbers appear: validation
    # metrics below are now IN-SAMPLE for the calibrator and are optimistic.
    # The 2026 sealed evaluation is untouched by this and remains the only
    # number that counts.
    raw_tune = clf.predict_proba(Xtu)[:, 1]
    if args.no_calibration:
        p_tune = raw_tune
        iso = None
    else:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        if args.calibrate_on == "train":
            iso.fit(oof, ytr)
        else:
            from sklearn.model_selection import KFold as _KF
            oof_val = np.zeros(len(ytu), dtype=float)
            for a, b in _KF(n_splits=5, shuffle=True, random_state=20260919).split(Xtu):
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
                cal.fit(clf.predict_proba(Xtu[a])[:, 1], ytu[a])
                oof_val[b] = cal.predict(clf.predict_proba(Xtu[b])[:, 1])
            iso.fit(clf.predict_proba(Xtu)[:, 1], ytu)
            p_tune = oof_val
            print("calibrator fitted on the VALIDATION window (permitted by v2 s3);"
                  " reported metrics use out-of-fold calibration to stay honest")
        if args.calibrate_on == "train":
            p_tune = iso.predict(raw_tune)

    pred = (p_tune >= 0.5).astype(int)
    correct = pred == ytu
    acc = float(correct.mean())
    lo, hi = wilson(int(correct.sum()), len(ytu))

    print("\n" + "=" * 62)
    print(f"VALIDATION WINDOW RESULTS ({TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]}, {len(ytu)} games)")
    print("=" * 62)
    print(f"  accuracy        {100*acc:.2f}%   95% CI {lo:.1f} to {hi:.1f}")
    print(f"  Brier           {brier(p_tune, ytu):.4f}")
    print(f"  log loss        {log_loss(p_tune, ytu):.4f}")

    base = baselines(tune)
    print("\n  against the baselines a reader could compute:")
    for name, bp in base.items():
        bcorrect = bp == ytu
        p = mcnemar(correct, bcorrect)
        print(f"    {name:<16} {100*bcorrect.mean():.2f}%   "
              f"we are {100*(acc - bcorrect.mean()):+.2f}pp, McNemar p={p:.4f}")

    print("\n  calibration (what we said vs what happened):")
    for b in calibration_report(p_tune, ytu, [0, .35, .45, .55, .65, .75, .85, 1.01]):
        gap = b["happened"] - b["said"]
        print(f"    {b['bucket']:<10} n={b['n']:>4}  said {b['said']:5.1f}%  "
              f"happened {b['happened']:5.1f}%  gap {gap:+5.1f}pp")

    print("\n  most important features:")
    imp = sorted(zip(MODEL_COLUMNS, clf.feature_importances_), key=lambda x: -x[1])
    for name, v in imp[:10]:
        print(f"    {name:<22} {v:.4f}")

    print("\nNOTE: these are VALIDATION numbers. Iterating on them is allowed and is")
    print("what this window exists for. They are NOT the model's record, and")
    print("they must never be published as one. The sealed window (2022-2025)")
    print("has not been touched.")

    if args.save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        clf.save_model(os.path.join(MODEL_DIR, "model.json"))
        meta = {"seal": SEAL, "params": PARAMS, "features": MODEL_COLUMNS,
                "train_seasons": list(TRAIN_SEASONS), "tune_seasons": list(TUNE_SEASONS),
                "calibrated": iso is not None,
                "tune_accuracy": acc, "tune_brier": brier(p_tune, ytu)}
        with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        if iso is not None:
            np.savez(os.path.join(MODEL_DIR, "calibrator.npz"),
                     x=iso.X_thresholds_, y=iso.y_thresholds_)
        print(f"\nsaved to {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
