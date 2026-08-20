"""
ParlayCorrelation.py
====================
Measured same-game correlation for parlay pricing.

A same-game parlay's legs are not independent, and until now the engine's only
honest move was to refuse to multiply ("UNRELIABLE"). This module replaces the
refusal with a measurement: across the historical odds archive (every game
with a closing spread, total and moneylines), how often does "the favorite
wins" actually co-occur with "the game goes over"?

METHOD. Games are bucketed by the favorite's moneyline-implied probability
(the observable a live parlay leg also carries). Within each bucket the four
joint outcomes (fav win / dog win x over / under) are counted, total-line
pushes excluded and reported. The correction factor for a leg pair is the
LIFT: P(A and B) / (P(A) * P(B)). Multiplying a pair's independent product by
the archive lift for its orientation and bucket gives the correlation-adjusted
joint probability, clamped to the Frechet bounds so no adjustment can ever
produce an impossible number.

HONESTY. Every cell carries its n; buckets under MIN_CELL_GAMES never adjust
anything (the reply says so instead of guessing). This measures the HISTORICAL
league-wide association only - it knows nothing about tonight's teams, and the
response text must never suggest otherwise.
"""

from typing import Any, Dict, List, Optional


# Favorite ML-implied probability buckets. Edges chosen so a -110/-110
# pick'em lands in the first bucket and a -300 favorite in the last.
BUCKETS = [
    (0.50, 0.575, "coin flip (fav 50-57.5%)"),
    (0.575, 0.65, "modest fav (57.5-65%)"),
    (0.65, 0.725, "solid fav (65-72.5%)"),
    (0.725, 1.0, "heavy fav (72.5%+)"),
]

# A bucket cell must hold at least this many games to adjust a price.
MIN_CELL_GAMES = 500


def _implied(american: float) -> float:
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def _bucket_label(fav_implied: float) -> Optional[str]:
    for lo, hi, label in BUCKETS:
        if lo <= fav_implied < hi:
            return label
    return None


def build_correlation_matrix(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    games: rows carrying ml_home, ml_away, margin (home - away, nonzero),
    points (total scored) and total_line. Rows missing any of these are
    excluded and counted.
    """
    excluded = {"missing_ml": 0, "missing_total": 0, "total_push": 0}
    cells: Dict[str, Dict[str, int]] = {
        label: {"n": 0, "fw_o": 0, "fw_u": 0, "dw_o": 0, "dw_u": 0}
        for _, _, label in BUCKETS
    }

    for g in games:
        ml_home = g.get("ml_home")
        ml_away = g.get("ml_away")
        if ml_home is None or ml_away is None:
            excluded["missing_ml"] += 1
            continue
        total_line = g.get("total_line")
        points = g.get("points")
        if total_line is None or points is None:
            excluded["missing_total"] += 1
            continue
        if points == total_line:
            excluded["total_push"] += 1
            continue

        home_implied = _implied(float(ml_home))
        away_implied = _implied(float(ml_away))
        # On an exact tie the home side is called the favorite; those games
        # sit in the first bucket where the fav/dog distinction is weakest.
        home_fav = home_implied >= away_implied
        fav_implied_fair = (
            max(home_implied, away_implied) / (home_implied + away_implied)
        )  # de-vigged multiplicatively, so buckets mean the same across eras

        label = _bucket_label(fav_implied_fair)
        if label is None:
            continue

        fav_won = (g["margin"] > 0) == home_fav
        over = points > total_line

        cell = cells[label]
        cell["n"] += 1
        if fav_won and over:
            cell["fw_o"] += 1
        elif fav_won:
            cell["fw_u"] += 1
        elif over:
            cell["dw_o"] += 1
        else:
            cell["dw_u"] += 1

    buckets_out = []
    for lo, hi, label in BUCKETS:
        c = cells[label]
        n = c["n"]
        if n == 0:
            buckets_out.append({"bucket": label, "n": 0})
            continue
        p_fav = (c["fw_o"] + c["fw_u"]) / n
        p_over = (c["fw_o"] + c["dw_o"]) / n
        joints = {
            "fav_over": c["fw_o"] / n,
            "fav_under": c["fw_u"] / n,
            "dog_over": c["dw_o"] / n,
            "dog_under": c["dw_u"] / n,
        }
        independents = {
            "fav_over": p_fav * p_over,
            "fav_under": p_fav * (1 - p_over),
            "dog_over": (1 - p_fav) * p_over,
            "dog_under": (1 - p_fav) * (1 - p_over),
        }
        lifts = {
            k: (joints[k] / independents[k]) if independents[k] > 0 else None
            for k in joints
        }
        denom = (p_fav * (1 - p_fav) * p_over * (1 - p_over)) ** 0.5
        phi = ((joints["fav_over"] - independents["fav_over"]) / denom) if denom > 0 else None

        buckets_out.append({
            "bucket": label,
            "fav_implied_range": [lo, hi],
            "n": n,
            "p_fav_win": round(p_fav, 4),
            "p_over": round(p_over, 4),
            "joint": {k: round(v, 4) for k, v in joints.items()},
            "independent": {k: round(v, 4) for k, v in independents.items()},
            "lift": {k: (round(v, 4) if v is not None else None) for k, v in lifts.items()},
            "phi": round(phi, 4) if phi is not None else None,
            "usable": n >= MIN_CELL_GAMES,
        })

    return {
        "buckets": buckets_out,
        "excluded": excluded,
        "min_cell_games": MIN_CELL_GAMES,
        "method": (
            "Games bucketed by the favorite's de-vigged moneyline probability. "
            "Lift = P(both) / (P(A) x P(B)) measured on archived final scores "
            "against closing lines; total-line pushes excluded. League-wide "
            "association only - it knows nothing about specific teams."
        ),
    }


def _orientation(leg_ml: Dict[str, Any], leg_total: Dict[str, Any]) -> str:
    fav_pick = float(leg_ml["odds"]) < 0
    over_pick = str(leg_total["pick"]).strip().lower() == "over"
    side = "fav" if fav_pick else "dog"
    total = "over" if over_pick else "under"
    return f"{side}_{total}"


def adjust_same_game_pairs(
    result: Dict[str, Any],
    matrix: Dict[str, Any],
) -> None:
    """
    Enrich an evaluate_parlay() result in place with correlation-adjusted
    pricing, when every same-game group is exactly one moneyline leg plus one
    total leg. Anything else stays UNRELIABLE, as before.
    """
    legs = result.get("legs", [])
    by_game: Dict[str, List[Dict[str, Any]]] = {}
    for leg in legs:
        key = f"{str(leg.get('away_team','')).strip().lower()}@{str(leg.get('home_team','')).strip().lower()}"
        by_game.setdefault(key, []).append(leg)

    groups = {k: v for k, v in by_game.items() if len(v) > 1}
    if not groups:
        return  # nothing correlated; the independent math already stands

    def is_ml(leg):
        return str(leg.get("market", "")).lower().replace("-", "_") == "moneyline"

    def is_total(leg):
        return str(leg.get("market", "")).lower().replace("-", "_") in ("over_under", "total")

    pairs = []
    for key, group in groups.items():
        if len(group) != 2:
            return  # 3+ legs on one game: no measured matrix for that
        ml = next((l for l in group if is_ml(l)), None)
        tot = next((l for l in group if is_total(l)), None)
        if ml is None or tot is None:
            return  # e.g. two totals; not measurable here
        pairs.append((key, ml, tot))

    bucket_by_label = {b["bucket"]: b for b in matrix.get("buckets", [])}

    adjusted_pairs = []
    combined = 1.0
    in_pairs = set()
    for key, ml, tot in pairs:
        p_ml = float(ml["model_prob"])
        p_tot = float(tot["model_prob"])
        # The favorite's implied probability, from the quoted leg itself.
        implied = float(ml["implied_prob"])
        fav_implied = implied if float(ml["odds"]) < 0 else 1.0 - implied
        label = _bucket_label(max(0.5, min(fav_implied, 0.9999)))
        cell = bucket_by_label.get(label)
        orientation = _orientation(ml, tot)

        if not cell or not cell.get("usable"):
            return  # thin bucket: better no number than a made-up one

        lift = (cell.get("lift") or {}).get(orientation)
        if lift is None:
            return

        independent = p_ml * p_tot
        joint = independent * lift
        # Frechet bounds: a joint probability can never exceed either leg nor
        # undercut their overlap minimum.
        joint = max(max(0.0, p_ml + p_tot - 1.0), min(joint, min(p_ml, p_tot)))

        combined *= joint
        in_pairs.update((ml["leg"], tot["leg"]))
        adjusted_pairs.append({
            "game": key,
            "legs": [ml["leg"], tot["leg"]],
            "orientation": orientation,
            "bucket": label,
            "bucket_n": cell["n"],
            "lift": lift,
            "independent_joint": round(independent, 4),
            "correlated_joint": round(joint, 4),
        })

    for leg in legs:
        if leg["leg"] not in in_pairs:
            combined *= float(leg["model_prob"])

    decimal = float(result["combined_decimal_odds"])
    net = decimal - 1.0
    ev = combined * net * 100.0 - (1.0 - combined) * 100.0
    kelly = (net * combined - (1.0 - combined)) / net * 100.0 if net > 0 else 0.0
    kelly = max(0.0, min(kelly, 10.0))

    result["correlation_adjusted"] = {
        "applied": True,
        "pairs": adjusted_pairs,
        "combined_model_prob": round(combined, 4),
        "expected_value_per_100": round(ev, 2),
        "edge_pct": round((combined - float(result["break_even_prob"])) * 100, 2),
        "kelly_pct_of_bankroll": round(kelly, 2),
        "note": (
            "Same-game moneyline+total pairs priced with the measured historical "
            "lift for this favorite strength and orientation - a league-wide "
            "average, not a read on these specific teams."
        ),
    }
    result["verdict"] = "POSITIVE_EV_CORRELATED" if ev > 0 else "NEGATIVE_EV_CORRELATED"
