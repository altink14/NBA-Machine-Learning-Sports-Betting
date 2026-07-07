"""
Parlay.py
=========
Parlay evaluation: combined probability, payout, expected value, Kelly sizing,
and correlation warnings.

A parlay's fair price is the product of its legs' true probabilities ONLY when
the legs are independent. Legs from the same game (e.g. moneyline + total) are
correlated, so the naive product misprices them — we surface warnings instead
of silently multiplying.
"""

from typing import Any, Dict, List


def american_to_true_decimal(american_odds: float) -> float:
    """American odds -> true decimal odds (stake included, e.g. -110 -> 1.909)."""
    if american_odds is None:
        raise ValueError("Leg is missing odds.")
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    return 1.0 + 100.0 / abs(american_odds)


def implied_probability(american_odds: float) -> float:
    """Bookmaker implied probability (includes vig)."""
    return 1.0 / american_to_true_decimal(american_odds)


def _game_key(leg: Dict[str, Any]) -> str:
    home = str(leg.get("home_team", "")).strip().lower()
    away = str(leg.get("away_team", "")).strip().lower()
    return f"{away}@{home}"


def detect_correlations(legs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Flag leg combinations whose outcomes are not independent.

    - SAME_GAME: two or more legs on the same game (strongest correlation;
      many books restrict these combinations outright).
    - REPEATED_TEAM: the same team appears in multiple legs across different
      games (mild schedule/rest correlation, mostly a diversification note).
    """
    warnings: List[Dict[str, str]] = []

    by_game: Dict[str, List[int]] = {}
    for i, leg in enumerate(legs):
        by_game.setdefault(_game_key(leg), []).append(i)

    for game, indexes in by_game.items():
        if len(indexes) > 1:
            markets = ", ".join(str(legs[i].get("market", "?")) for i in indexes)
            warnings.append({
                "type": "SAME_GAME",
                "severity": "high",
                "legs": indexes,
                "message": (
                    f"Legs {[i + 1 for i in indexes]} are on the same game ({markets}). "
                    "Their outcomes are correlated, so the combined probability shown "
                    "here (a simple product) is unreliable — and sportsbooks price or "
                    "block correlated same-game legs for exactly that reason."
                ),
            })

    team_legs: Dict[str, List[int]] = {}
    for i, leg in enumerate(legs):
        pick = str(leg.get("pick", "")).strip().lower()
        for team_field in ("home_team", "away_team"):
            team = str(leg.get(team_field, "")).strip().lower()
            if team and pick == team:
                team_legs.setdefault(team, []).append(i)

    for team, indexes in team_legs.items():
        if len(indexes) > 1 and len({_game_key(legs[i]) for i in indexes}) > 1:
            warnings.append({
                "type": "REPEATED_TEAM",
                "severity": "low",
                "legs": indexes,
                "message": (
                    f"Legs {[i + 1 for i in indexes]} all ride on {team.title()}. "
                    "One bad night for that team sinks every one of these legs."
                ),
            })

    return warnings


def evaluate_parlay(legs: List[Dict[str, Any]], kelly_fraction_cap: float = 10.0) -> Dict[str, Any]:
    """
    Evaluate a parlay ticket.

    Each leg dict:
      home_team, away_team, market ('moneyline' | 'over_under'), pick,
      odds (American), model_prob (0-1, optional — falls back to the
      bookmaker's implied probability, which makes that leg EV-neutral
      minus vig and is flagged in prob_source).

    Returns combined odds/probability, EV per $100, Kelly stake, break-even
    probability, and correlation warnings.
    """
    if not legs:
        raise ValueError("A parlay needs at least one leg.")
    if len(legs) > 12:
        raise ValueError("Parlays over 12 legs are not supported.")

    combined_decimal = 1.0
    combined_prob = 1.0
    evaluated_legs: List[Dict[str, Any]] = []

    for i, leg in enumerate(legs):
        odds = leg.get("odds")
        if odds is None:
            raise ValueError(f"Leg {i + 1} is missing American odds.")
        odds = float(odds)
        if -100 < odds < 100:
            raise ValueError(f"Leg {i + 1}: American odds must be <= -100 or >= +100 (got {odds:g}).")

        decimal = american_to_true_decimal(odds)
        implied = implied_probability(odds)

        model_prob = leg.get("model_prob")
        prob_source = "model"
        if model_prob is None:
            model_prob = implied
            prob_source = "market_implied"
        model_prob = float(model_prob)
        if not 0.0 < model_prob < 1.0:
            raise ValueError(f"Leg {i + 1}: model_prob must be strictly between 0 and 1.")

        combined_decimal *= decimal
        combined_prob *= model_prob

        evaluated_legs.append({
            "leg": i + 1,
            "market": leg.get("market"),
            "pick": leg.get("pick"),
            "home_team": leg.get("home_team"),
            "away_team": leg.get("away_team"),
            "odds": odds,
            "decimal_odds": round(decimal, 4),
            "implied_prob": round(implied, 4),
            "model_prob": round(model_prob, 4),
            "prob_source": prob_source,
            "leg_edge_pct": round((model_prob - implied) * 100, 2),
        })

    warnings = detect_correlations(legs)
    independent = not any(w["type"] == "SAME_GAME" for w in warnings)

    # Per $100 staked: win nets (d-1)*100, lose nets -100.
    net_multiplier = combined_decimal - 1.0
    ev_per_100 = combined_prob * net_multiplier * 100.0 - (1.0 - combined_prob) * 100.0
    break_even_prob = 1.0 / combined_decimal

    # Kelly: f* = (b*p - q) / b, expressed as % of bankroll and capped —
    # parlay variance is brutal, so full Kelly is rarely advisable.
    kelly_pct = (net_multiplier * combined_prob - (1.0 - combined_prob)) / net_multiplier * 100.0
    kelly_pct = max(0.0, min(kelly_pct, kelly_fraction_cap))

    if not independent:
        verdict = "UNRELIABLE — correlated same-game legs; combined probability is not a simple product."
    elif ev_per_100 > 0:
        verdict = "POSITIVE_EV"
    else:
        verdict = "NEGATIVE_EV"

    return {
        "legs": evaluated_legs,
        "combined_decimal_odds": round(combined_decimal, 4),
        "combined_american_odds": round((combined_decimal - 1) * 100) if combined_decimal >= 2
            else round(-100 / (combined_decimal - 1)),
        "combined_model_prob": round(combined_prob, 4),
        "break_even_prob": round(break_even_prob, 4),
        "edge_pct": round((combined_prob - break_even_prob) * 100, 2),
        "expected_value_per_100": round(ev_per_100, 2),
        "kelly_pct_of_bankroll": round(kelly_pct, 2),
        "independent_legs": independent,
        "warnings": warnings,
        "verdict": verdict,
    }
