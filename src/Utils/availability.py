"""Availability adjustment: how much a team's strength drops when
specific players are out.

Built on src/Utils/player_impact.py ("Estimated impact, box-score based" -
our implementation of Neil Paine's Estimated RAPTOR weights, NOT official
RAPTOR/DARKO; see that module's docstring for methodology and limits).

Core idea: a player's absence costs his team roughly
    contribution = total_impact (pts / 100 poss) x minutes_share
where minutes_share = min(avg minutes, 40) / 48 - i.e. impact is
per-possession, so the team only loses it for the fraction of the game
he would have played. Minutes are capped at 40 (nobody replaces an
absent star with 48 minutes of nothing - the replacement is implicit in
"league-average player" being the zero point of the impact scale, and
uncapped season averages already never exceed ~40).

Each player's contribution is additionally clamped to +/-8 pts/100 and
the summed team delta to +/-15 pts/100: beyond that the linear
approximation has no support (and 15 is already a bigger swing than any
realistic single-team injury report).

The delta is EXPRESSED as the change to the team's net rating per 100
possessions: negative when good players are out. This is informational -
it is deliberately NOT folded into model win probabilities anywhere
(that would require backtest validation first; see honesty policy).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from src.Utils import player_impact

MINUTES_CAP = 40.0
PLAYER_CONTRIBUTION_CAP = 8.0
TEAM_DELTA_CAP = 15.0

NOTE = (
    "Estimated impact, box-score based (our implementation; not official "
    "RAPTOR/DARKO). Informational only - model probabilities are unchanged."
)

RatingsInput = Union[Dict[int, Dict[str, Any]], Iterable[Dict[str, Any]], None]


def _ratings_index(ratings: RatingsInput, season: str) -> Dict[int, Dict[str, Any]]:
    if ratings is None:
        ratings = player_impact.get_impact_ratings(season, min_gp=0)
    if isinstance(ratings, dict):
        return ratings
    return {r["player_id"]: r for r in ratings}


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(value, limit))


def adjusted_rating_delta(
    team_abbr: str,
    out_player_ids: List[int],
    season: str,
    ratings: RatingsInput = None,
) -> Dict[str, Any]:
    """Net-rating delta (per 100 possessions) for `team_abbr` with the
    given players out, plus the per-player breakdown.

    Pure function of the ratings: pass `ratings` (list of impact records
    or {player_id: record}) to make it fully deterministic in tests; by
    default it uses the cached season ratings from player_impact.

    Players without a rating for the season (rookies who haven't played,
    bad ID, etc.) contribute 0 and are flagged "rated": False - an
    unknown player must never move the number.
    """
    index = _ratings_index(ratings, season)
    players: List[Dict[str, Any]] = []
    delta = 0.0
    for player_id in out_player_ids or []:
        record = index.get(player_id)
        if not record or record.get("total_impact") is None:
            players.append({
                "player_id": player_id,
                "name": record.get("name") if record else None,
                "impact": None,
                "min_share": 0.0,
                "contribution": 0.0,
                "rated": False,
            })
            continue
        impact = float(record["total_impact"])
        minutes = float(record.get("min_per_g") or 0.0)
        min_share = min(minutes, MINUTES_CAP) / 48.0
        contribution = _clamp(impact * min_share, PLAYER_CONTRIBUTION_CAP)
        delta -= contribution
        players.append({
            "player_id": player_id,
            "name": record.get("name"),
            "impact": round(impact, 2),
            "min_share": round(min_share, 3),
            "contribution": round(contribution, 2),
            "rated": True,
        })
    delta = _clamp(delta, TEAM_DELTA_CAP)
    return {"delta_per_100": round(delta, 2), "players": players}


def matchup_availability(
    home_abbr: str,
    away_abbr: str,
    season: str,
    absences: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Current OUT/DOUBTFUL availability picture for one matchup.

    `absences` is the structure from espn_injuries.get_absences();
    fetched live when omitted. Never raises on feed trouble - an empty
    injury list is a valid (and in the offseason, expected) answer.
    """
    from src.Utils import espn_injuries  # local import: keep module load light

    if absences is None:
        absences = espn_injuries.get_absences()
    by_team = absences.get("by_team", {}) if isinstance(absences, dict) else {}

    def side(abbr: str) -> Dict[str, Any]:
        listed = by_team.get(abbr, [])
        ids = [p["player_id"] for p in listed if p.get("player_id") is not None]
        result = adjusted_rating_delta(abbr, ids, season)
        impact_by_id = {p["player_id"]: p for p in result["players"]}
        players_out = []
        for entry in listed:
            impact_rec = impact_by_id.get(entry.get("player_id")) or {}
            players_out.append({
                "player_id": entry.get("player_id"),
                "name": entry.get("name"),
                "status": entry.get("status"),
                "detail": entry.get("detail"),
                "impact": impact_rec.get("impact"),
                "min_share": impact_rec.get("min_share", 0.0),
                "contribution": impact_rec.get("contribution", 0.0),
            })
        return {
            "team": abbr,
            "players_out": players_out,
            "delta_per_100": result["delta_per_100"],
        }

    return {
        "season": season,
        "home": side(home_abbr),
        "away": side(away_abbr),
        "statuses_counted": absences.get("counted_statuses", ["Doubtful", "Out"]) if isinstance(absences, dict) else ["Doubtful", "Out"],
        "injury_feed": {
            "source": absences.get("source") if isinstance(absences, dict) else None,
            "fetched_at": absences.get("fetched_at") if isinstance(absences, dict) else None,
            "name_match_rate": absences.get("match_rate") if isinstance(absences, dict) else None,
        },
        "note": NOTE,
    }
