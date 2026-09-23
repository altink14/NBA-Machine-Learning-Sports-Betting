"""
"Why this pick", from the model's own arithmetic.

The sealed candidate is an XGBoost booster over 207 inputs. For any single
game, XGBoost can say exactly how much each input moved its score
(`pred_contribs`, the tree SHAP values), and those amounts add up to the
score itself. This module sums them into the handful of groups a person can
read (each team's season numbers, each team's last 10 and last 20 games, the
Elo ratings, rest and schedule) and writes two or three sentences from them.

WHY NOT A LANGUAGE MODEL. A language model shown the odds and a confidence
has no access to why the booster decided anything, so whatever it wrote under
"why this pick" would be a plausible story, not the reason. It would also
invent support ("they've been hot at home") the model never saw. Everything
here is the model's own weighting, and every number in the text is an input
the model was actually given for this game.

WHAT IT DOES NOT CLAIM. The contributions explain the RAW booster score in
log-odds. The stored isotonic calibrator then maps that score to the
published probability, so a group's share of the push is exact but it is not
"percentage points of win probability", and the text never says it is.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

#: How each group reads in a sentence. {home}/{away} are team names.
GROUP_LABELS = {
    "season_home": "{home_pos} season numbers",
    "season_away": "{away_pos} season numbers",
    "form10_home": "{home_pos} last 10 games",
    "form10_away": "{away_pos} last 10 games",
    "form20_home": "{home_pos} last 20 games",
    "form20_away": "{away_pos} last 20 games",
    "elo": "the Elo ratings",
    "rest": "rest and schedule",
}

#: The inputs shown beside each group: real values the model was given.
#: (column, label, kind) where kind formats the number.
GROUP_INPUTS = {
    "season_home": [("W_PCT", "season win %", "pct"), ("PLUS_MINUS", "avg margin", "signed")],
    "season_away": [("W_PCT.1", "season win %", "pct"), ("PLUS_MINUS.1", "avg margin", "signed")],
    "form10_home": [("R10_HOME_WIN_PCT", "win % last 10", "pct"), ("R10_HOME_PLUS_MINUS", "avg margin last 10", "signed")],
    "form10_away": [("R10_AWAY_WIN_PCT", "win % last 10", "pct"), ("R10_AWAY_PLUS_MINUS", "avg margin last 10", "signed")],
    "form20_home": [("R20_HOME_WIN_PCT", "win % last 20", "pct"), ("R20_HOME_PLUS_MINUS", "avg margin last 20", "signed")],
    "form20_away": [("R20_AWAY_WIN_PCT", "win % last 20", "pct"), ("R20_AWAY_PLUS_MINUS", "avg margin last 20", "signed")],
    "elo": [("ELO_HOME", "home Elo", "int"), ("ELO_AWAY", "away Elo", "int")],
    # Days-Rest is the whole-day gap since the team's previous game (the
    # convention the model was trained on), so 1 means it played yesterday.
    "rest": [("Days-Rest-Home", "home: days since last game", "int"), ("Days-Rest-Away", "away: days since last game", "int"),
             ("REST_HOME_B2B", "home on a back-to-back", "bool"), ("REST_AWAY_B2B", "away on a back-to-back", "bool")],
}

#: A factor under this share of the total push is not worth a sentence.
MIN_SHARE_TO_NAME = 0.08

NOTE = ("These are the model's own inputs and how much each one moved its score for this game, "
        "before the calibration step that sets the final percentage. They explain the pick; "
        "they do not make it certain. Never wager money you cannot afford to lose.")


def possessive(name: str) -> str:
    """"Boston Celtics'" and "Utah Jazz's"."""
    return name + ("'" if name.endswith("s") else "'s")


def group_of(col: str) -> str:
    if col in ("Days-Rest-Home", "Days-Rest-Away") or col.startswith("REST_"):
        return "rest"
    if col.startswith("ELO_"):
        return "elo"
    for k in ("10", "20"):
        if col.startswith(f"R{k}_HOME_"):
            return f"form{k}_home"
        if col.startswith(f"R{k}_AWAY_"):
            return f"form{k}_away"
    return "season_away" if col.endswith(".1") else "season_home"


def _clean(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _fmt(v: Optional[float], kind: str) -> Optional[str]:
    if v is None:
        return None
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "signed":
        return f"{v:+.1f}"
    if kind == "bool":
        return "yes" if v >= 0.5 else "no"
    return f"{v:.0f}"


def group_contributions(cols: List[str], delta_row, x_row, n_base: int,
                        home: str, away: str) -> Dict[str, object]:
    """Per-group push toward the HOME team, from one game's contributions.

    `delta_row` has one entry per column in `cols` plus the bias last, in
    log-odds toward a home win. Positive pushes toward `home`.
    """
    assert len(delta_row) == len(cols) + 1, "contributions must be columns + bias"
    sums: Dict[str, float] = {}
    for col, c in zip(cols, delta_row[:-1]):
        g = group_of(col)
        sums[g] = sums.get(g, 0.0) + float(c)
    value = {col: _clean(v) for col, v in zip(cols, x_row)}
    total_abs = sum(abs(v) for v in sums.values()) or 1.0
    groups = []
    for key, push in sorted(sums.items(), key=lambda kv: -abs(kv[1])):
        inputs = []
        for col, label, kind in GROUP_INPUTS.get(key, []):
            shown = _fmt(value.get(col), kind)
            if shown is not None:
                inputs.append({"label": label, "value": shown})
        groups.append({
            "key": key,
            "label": GROUP_LABELS[key].format(home_pos=possessive(home), away_pos=possessive(away)),
            "toward": home if push >= 0 else away,
            "push": round(push, 4),
            "share": round(abs(push) / total_abs, 3),
            "inputs": inputs,
        })
    return {
        "home": home,
        "away": away,
        "baseline": round(float(delta_row[-1]), 4),
        "score": round(float(sum(delta_row)), 4),
        "groups": groups,
    }


def _join(parts: List[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def build_why(reasons: Dict[str, object], winner: str, confidence_pct: float) -> Dict[str, object]:
    """Two or three sentences plus the factor list, for one pick.

    `confidence_pct` is the published (calibrated) confidence in `winner`.
    Every team name and number in the text comes from `reasons` or from the
    two arguments; nothing is looked up or added.
    """
    groups = [g for g in reasons["groups"] if g["share"] >= MIN_SHARE_TO_NAME]
    toward = [g for g in groups if g["toward"] == winner][:2]
    against = [g for g in groups if g["toward"] != winner][:1]
    sentences = [f"The model gives {winner} {confidence_pct:.1f}% to win."]
    if toward:
        sentences.append(
            "The biggest push toward " + winner + " came from "
            + _join([f"{g['label']} ({g['share'] * 100:.0f}%" + (" of the model's total push)" if i == 0 else ")")
                     for i, g in enumerate(toward)]) + ".")
    if against:
        g = against[0]
        sentences.append(
            f"Pulling the other way: {g['label']}, toward {g['toward']} ({g['share'] * 100:.0f}%).")
    return {
        "summary": " ".join(sentences),
        "factors": reasons["groups"],
        "note": NOTE,
        "method": "xgboost pred_contribs (tree SHAP), grouped; raw score before isotonic calibration",
    }
