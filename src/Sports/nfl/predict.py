"""
predict.py (NFL)
================
Writes a prediction to the ledger for every upcoming NFL game, before kickoff,
with the price we could actually have taken at that moment.

THIS IS THE FORWARD TEST. Pre-registration v2 seals the 2026 season, which
cannot be peeked at because it has not been played. The only way that seal
turns into a result is if predictions are written down, one at a time, before
each game, all season. A season we forget to log is a season we cannot claim.

THE GRADING IS SILENT ON PURPOSE. Games get graded into the ledger as they
finish, because the record has to exist when the evaluation runs. But 2026 is
the sealed window, so this script never prints how those picks did: the ledger
reports a sealed competition's wins and losses as one `settled` count. A daily
win-loss tally in a log file is a running read of the sealed window, which is
the same kind of aggregate that voided v1. The number appears once, after the
season, or it is not a seal.

EVERYTHING IS SHADOW. `is_shadow = 1` on every row this writes. The model has
not passed its gates and will not have until the 2026 season ends. Shadow rows
may be shown publicly LABELLED AS SHADOW and must never be presented as a track
record, quoted in marketing, or averaged into a headline number.

THE PRICE MATTERS AS MUCH AS THE PICK. `price_taken` comes from our own most
recent odds snapshot for that game, so it is a price that genuinely existed
when we predicted. It is the anchor for closing line value later. If no
snapshot exists yet the row is still written with a null price: the prediction
is the thing that must not be late, and a missing price costs us one CLV data
point rather than the whole record.

MODEL PROVENANCE. The served model is the one fitted on 1999-2018, exactly as
v2 section 3 specifies. That is narrower than the data we hold, because v2
assigns 2019-2025 to selection rather than fitting. Widening it is a v3
decision and is flagged in the runbook rather than taken quietly here.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/predict.py --dry-run
    venv/Scripts/python.exe src/Sports/nfl/predict.py
    venv/Scripts/python.exe src/Sports/nfl/predict.py --grade
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from src.Sports.ledger import ensure_ledger, write_prediction, grade_pending, health, LEDGER_DB  # noqa: E402
from src.Sports.nfl.features import build_frame, MODEL_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.predict")

NFL_DB = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
MODEL_DIR = os.path.join(REPO_ROOT, "Models", "nfl_v2")
MODEL_VERSION = "nfl_v2"
MODEL_SEAL = "a88f762a065b21831e863e5a3a9935c5a1af4639"

#: Only predict games starting within this horizon. Further out the rosters,
#: the weather and the line are all still moving, and a prediction made in
#: August about a December game is not a useful record of anything.
HORIZON_HOURS = 200


def american_to_prob(price: Optional[int]) -> Optional[float]:
    if price is None:
        return None
    p = float(price)
    return 100.0 / (p + 100.0) if p > 0 else (-p) / ((-p) + 100.0)


def expected_value(prob: float, price: Optional[int]) -> Optional[float]:
    """EV per unit staked at American odds."""
    if price is None:
        return None
    dec = 1.0 + (price / 100.0 if price > 0 else 100.0 / -price)
    return prob * (dec - 1.0) - (1.0 - prob)


def kelly(prob: float, price: Optional[int]) -> Optional[float]:
    if price is None:
        return None
    b = (price / 100.0) if price > 0 else (100.0 / -price)
    f = (b * prob - (1.0 - prob)) / b
    return max(0.0, f)


def load_model():
    from xgboost import XGBClassifier
    from sklearn.isotonic import IsotonicRegression
    clf = XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, "model.json"))
    iso = None
    cal = os.path.join(MODEL_DIR, "calibrator.npz")
    if os.path.exists(cal):
        d = np.load(cal)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        iso.X_thresholds_, iso.y_thresholds_ = d["x"], d["y"]
        iso.X_min_, iso.X_max_ = float(d["x"][0]), float(d["x"][-1])
        iso.increasing_ = True
        iso._necessary_X_, iso._necessary_y_ = d["x"], d["y"]
        from scipy.interpolate import interp1d
        iso.f_ = interp1d(d["x"], d["y"], kind="linear", bounds_error=False,
                          fill_value=(d["y"][0], d["y"][-1]))
    return clf, iso


def latest_prices(conn: sqlite3.Connection, game_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Our most recent moneyline snapshot per game: a price that really existed."""
    if not game_ids:
        return {}
    ph = ",".join("?" * len(game_ids))
    out: Dict[str, Dict[str, Any]] = {}
    try:
        rows = conn.execute(
            f"""SELECT s.game_id, s.book, s.price_home, s.price_away, s.captured_at
                FROM market_line_snapshots s
                WHERE s.market_type = 'moneyline' AND s.game_id IN ({ph})
                  AND s.id = (SELECT MAX(s2.id) FROM market_line_snapshots s2
                              WHERE s2.game_id = s.game_id AND s2.book = s.book
                                AND s2.market_type = 'moneyline')""", game_ids).fetchall()
    except sqlite3.OperationalError:
        return {}
    # Best available price for each side, across books, which is what a bettor
    # shopping lines would actually get.
    for gid, book, ph_, pa_, cap in rows:
        cur = out.setdefault(gid, {"home": None, "away": None, "book_home": None,
                                   "book_away": None, "captured_at": cap})
        if ph_ is not None and (cur["home"] is None or ph_ > cur["home"]):
            cur["home"], cur["book_home"] = ph_, book
        if pa_ is not None and (cur["away"] is None or pa_ > cur["away"]):
            cur["away"], cur["book_away"] = pa_, book
        cur["captured_at"] = max(cur["captured_at"] or "", cap or "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Write NFL predictions to the ledger before kickoff.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be written.")
    ap.add_argument("--grade", action="store_true", help="Grade finished games and exit.")
    ap.add_argument("--horizon", type=int, default=HORIZON_HOURS)
    args = ap.parse_args()

    led = sqlite3.connect(LEDGER_DB, timeout=60)
    ensure_ledger(led)
    nfl = sqlite3.connect(f"file:{NFL_DB}?mode=ro", uri=True)
    nfl.row_factory = sqlite3.Row

    if args.grade:
        results = {r["game_id"]: {"home_score": r["home_score"], "away_score": r["away_score"],
                                  "status": r["status"]}
                   for r in nfl.execute("SELECT game_id, home_score, away_score, status FROM games")}
        counts = grade_pending(led, results, grading_source="nflverse archive")
        logger.info("graded: %s", counts)
        logger.info("ledger health: %s", health(led, sport="football"))
        logger.info("2026 is sealed: results are in the ledger, but reported "
                    "only as 'settled' until the single evaluation")
        return 0

    rows, _ = build_frame(for_prediction=True)
    now = datetime.now(timezone.utc)
    horizon = (now + timedelta(hours=args.horizon)).isoformat()
    upcoming = [r for r in rows if r["kickoff_utc"] and now.isoformat() < r["kickoff_utc"] <= horizon]
    upcoming.sort(key=lambda r: r["kickoff_utc"])
    logger.info("%d upcoming game(s) within %dh", len(upcoming), args.horizon)
    if not upcoming:
        logger.info("nothing to predict")
        return 0

    clf, iso = load_model()
    X = np.array([[np.nan if r[c] is None else float(r[c]) for c in MODEL_COLUMNS]
                  for r in upcoming], dtype=float)
    raw = clf.predict_proba(X)[:, 1]
    probs = iso.predict(raw) if iso is not None else raw

    prices = latest_prices(nfl, [r["game_id"] for r in upcoming])
    written = skipped = 0

    for r, p_home in zip(upcoming, probs):
        side = "home" if p_home >= 0.5 else "away"
        prob = float(p_home if side == "home" else 1.0 - p_home)
        pr = prices.get(r["game_id"], {})
        price = pr.get(side)
        book = pr.get(f"book_{side}")
        fair = None
        ph_, pa_ = american_to_prob(pr.get("home")), american_to_prob(pr.get("away"))
        if ph_ and pa_ and (ph_ + pa_) > 0:
            fair = (ph_ if side == "home" else pa_) / (ph_ + pa_)

        line = f"{r['away_team']} @ {r['home_team']}  {r['kickoff_utc'][:16]}  " \
               f"pick {side.upper():<4} {100*prob:5.1f}%"
        if price is not None:
            ev = expected_value(prob, price)
            line += f"  price {price:+5d} ({book})  EV {100*ev:+6.2f}%"
        else:
            line += "   no price captured yet"
        print("  " + line)

        if args.dry_run:
            continue
        rid = write_prediction(
            led, sport="football", league="NFL", competition_id=f"nfl-{r['season']}-{r['season_type']}",
            game_id=r["game_id"], market_type="moneyline", side=side,
            model_version=MODEL_VERSION, model_seal=MODEL_SEAL, model_prob=prob,
            event_start_utc=r["kickoff_utc"], price_taken=price, book=book,
            fair_prob=fair, ev=expected_value(prob, price), kelly_fraction=kelly(prob, price),
            is_shadow=True,
            notes="shadow: model has not passed its gates; sealed evaluation after the 2026 season")
        if rid:
            written += 1
        else:
            skipped += 1

    if not args.dry_run:
        logger.info("written=%d already_present=%d", written, skipped)
        logger.info("ledger health: %s", health(led, sport="football"))
    led.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
