"""Estimated player impact - box-score based ("Estimated RAPTOR"-style).

WHAT THIS IS (honesty label)
----------------------------
This is OUR box-score implementation of a player-impact metric:
"Estimated impact, box-score based". It is NOT FiveThirtyEight's real
RAPTOR (which used player tracking data), NOT DARKO, and must never be
presented as either. Ratings are estimates of points per 100 possessions
a player added to (or subtracted from) his team relative to a
league-average player, split into offense and defense.

METHODOLOGY & ATTRIBUTION
-------------------------
The regression weights come verbatim from Neil Paine's "Estimated RAPTOR"
(https://github.com/Neil-Paine-1/NBA-elo, MIT license). Paine fit linear
regressions predicting full RAPTOR from box-score actions per 100 team
possessions plus on-court/on-off plus-minus components; among 1,000+ minute
seasons his estimates correlated 0.913 (off) / 0.784 (def) / 0.890 (overall)
with real RAPTOR over 2014-2023. We use his post-1997 weight table exactly
(intercept, MPG, PTS/100, TSA/100, AST/100, TOV/100, ORB/100, DRB/100,
STL/100, BLK/100, PF/100, OnCourt, On-Off, for offense and defense).

Where our data forces adaptations, we adapted (each one documented):

1. Per-100 rates are derived from per-game averages in
   `player_season_stats` (NBA.com leaguedashplayerstats) using the
   player's on-court pace: stat/100 = stat_pg * 100 / (pace * min_pg / 48).
   TSA (true-shooting attempts) = FGA + 0.44 * FTA.
2. OnCourt = the player's NBA.com on-court team rating relative to the
   league average (offense: off_rating - league ORtg; defense:
   league DRtg - def_rating, so positive is always "good").
3. On-Off: we do not store off-court ratings, so the off-court rating is
   *estimated* by decomposing the team's season rating using the player's
   share of team lineup-minutes (share = GP*MIN / (team_games*240)):
   off_court = (team_rating - share * on_court) / (1 - share), share
   clamped to <= 0.85 and the resulting on-off clamped to +/-25 to keep
   low-minute noise out. This is the noisiest input; its regression
   weights are small (0.032 off / 0.022 def), which limits the damage.
4. Position adjustment: OMITTED (deliberate deviation from Paine).
   Paine pins each position's minute-weighted average to PG/SG/SF/PF/C
   targets. Our only position source (`player_bio.position`) is NULL for
   ~97% of player-season minutes, and applying the adjustment to the few
   players who do have a position shifts them by their own tiny group's
   star-dominated mean (measured effect: it moved 2024-25 SGA from +10.0
   raw offense to +3.5, while position-less teammates were untouched).
   Omitting it costs a little offense/defense-split calibration for
   guards vs centers; total impact is still anchored by the team
   constraint below. Measured against Paine's published 2024-25
   Estimated RAPTOR (1000+ minute players, n=212): Pearson r = 0.956
   offense, 0.946 defense, 0.934 total after the omission.
5. Team adjustment: exactly Paine's rule - 4.5 x the minute-weighted
   average of a team's player ratings must equal the team's offensive or
   defensive rating relative to league average; the shortfall is spread
   equally ((target - current) / 4.5 added to every player on the team).
6. WAR is not computed (not needed for availability adjustments).

KNOWN LIMITATIONS (do not hide these)
-------------------------------------
- Box-score only: defense is the weak side, exactly as Paine reports
  (r=0.784 vs real RAPTOR). Elite team defenders with empty steal/block
  columns will be underrated.
- Traded players carry a single season row attributed to their final
  team; their team adjustment uses that team only.
- The on-off input is an estimate (see #3), not measured on-off.
- Derived team ratings in `team_season_advanced` use Dean Oliver
  estimated possessions (1-3 pt differences vs NBA.com official).
- Ratings are descriptive of the season played, not a forward projection
  (no aging curve, no priors - unlike DARKO/RAPTOR predictive mode).
- Per-possession ratings for low-minute players are noisy (a 10-mpg
  bench defender can post an eye-popping per-100 number). Consumers
  should weight by minutes share - the availability adjustment does.

Data source: Data/TeamData.sqlite (read-only). Everything is computed
lazily in memory and cached per (season, season_type).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

# --- Paine's post-1997 regression weights (verbatim from his README) ---
OFFENSE_WEIGHTS = {
    "intercept": -3.88704,
    "mpg": 0.026112,
    "pts100": 0.662784,
    "tsa100": -0.51622,
    "ast100": 0.430454,
    "tov100": -0.893465,
    "orb100": 0.303023,
    "drb100": -0.085637,
    "stl100": 0.418092,
    "blk100": -0.230734,
    "pf100": -0.108369,
    "oncourt": 0.018381,
    "onoff": 0.032054,
}
DEFENSE_WEIGHTS = {
    "intercept": -3.079144,
    "mpg": 0.033637,
    "pts100": -0.081412,
    "tsa100": 0.025422,
    "ast100": -0.025109,
    "tov100": -0.055809,
    "orb100": -0.099034,
    "drb100": 0.191569,
    "stl100": 1.150891,
    "blk100": 0.611107,
    "pf100": 0.010649,
    "oncourt": 0.089391,
    "onoff": 0.021717,
}

MAX_MINUTE_SHARE = 0.85   # clamp for the off-court decomposition
ON_OFF_CLAMP = 25.0       # clamp for the estimated on-off input
DEFAULT_MIN_GP = 20

METHODOLOGY_LABEL = (
    "Estimated impact, box-score based - our implementation of Neil Paine's "
    "'Estimated RAPTOR' regression weights (github.com/Neil-Paine-1/NBA-elo, MIT). "
    "Not official RAPTOR or DARKO."
)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data",
    "TeamData.sqlite",
)

_cache: Dict[tuple, List[Dict[str, Any]]] = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pure computation (unit-testable without the database)
# ---------------------------------------------------------------------------

def compute_impacts(
    players: List[Dict[str, Any]],
    teams: Dict[str, Dict[str, Any]],
    league_ortg: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Compute offensive/defensive impact for one season of players.

    players: dicts with keys player_id, name, team_abbr, position, gp, min,
             pts, fga, fta, ast, tov, oreb, dreb, stl, blk, pf,
             off_rating, def_rating, pace (per-game averages; ratings are
             the player's on-court team ratings).
    teams:   team_abbr -> {"off_rating", "def_rating", "games"} season values.
    league_ortg: league average points per 100 possessions; defaults to the
             unweighted mean of team off_ratings (league ORtg == league DRtg).

    Returns one dict per usable player with raw and adjusted impacts.
    """
    if league_ortg is None:
        if teams:
            league_ortg = sum(t["off_rating"] for t in teams.values()) / len(teams)
        else:
            league_ortg = 113.0

    rows: List[Dict[str, Any]] = []
    for p in players:
        gp = p.get("gp") or 0
        mpg = p.get("min") or 0.0
        pace = p.get("pace") or 0.0
        if gp <= 0 or mpg <= 0 or pace <= 0:
            continue

        poss_per_game = pace * mpg / 48.0
        if poss_per_game <= 0:
            continue
        per100 = 100.0 / poss_per_game

        tsa_pg = (p.get("fga") or 0.0) + 0.44 * (p.get("fta") or 0.0)
        feats = {
            "mpg": mpg,
            "pts100": (p.get("pts") or 0.0) * per100,
            "tsa100": tsa_pg * per100,
            "ast100": (p.get("ast") or 0.0) * per100,
            "tov100": (p.get("tov") or 0.0) * per100,
            "orb100": (p.get("oreb") or 0.0) * per100,
            "drb100": (p.get("dreb") or 0.0) * per100,
            "stl100": (p.get("stl") or 0.0) * per100,
            "blk100": (p.get("blk") or 0.0) * per100,
            "pf100": (p.get("pf") or 0.0) * per100,
        }

        team = teams.get(p.get("team_abbr"))
        on_ortg = p.get("off_rating")
        on_drtg = p.get("def_rating")

        # OnCourt: on-court team performance relative to league (good = +).
        oncourt_off = (on_ortg - league_ortg) if on_ortg is not None else 0.0
        oncourt_def = (league_ortg - on_drtg) if on_drtg is not None else 0.0

        # On-Off: estimated by decomposing the team's season rating with the
        # player's share of lineup-minutes (see module docstring, item 3).
        onoff_off = 0.0
        onoff_def = 0.0
        if team and team.get("games"):
            share = (gp * mpg) / (team["games"] * 240.0)
            share = max(0.0, min(share, MAX_MINUTE_SHARE))
            if share > 0.0 and on_ortg is not None:
                off_court_ortg = (team["off_rating"] - share * on_ortg) / (1.0 - share)
                onoff_off = _clamp(on_ortg - off_court_ortg, ON_OFF_CLAMP)
            if share > 0.0 and on_drtg is not None:
                off_court_drtg = (team["def_rating"] - share * on_drtg) / (1.0 - share)
                # Positive = the team defends better with him on the floor.
                onoff_def = _clamp(off_court_drtg - on_drtg, ON_OFF_CLAMP)

        off_feats = dict(feats, oncourt=oncourt_off, onoff=onoff_off)
        def_feats = dict(feats, oncourt=oncourt_def, onoff=onoff_def)

        off_raw = _apply_weights(off_feats, OFFENSE_WEIGHTS)
        def_raw = _apply_weights(def_feats, DEFENSE_WEIGHTS)

        rows.append({
            "player_id": p.get("player_id"),
            "name": p.get("name"),
            "team_abbr": p.get("team_abbr"),
            "position": p.get("position"),
            "gp": gp,
            "min_per_g": mpg,
            "total_minutes": gp * mpg,
            "off_raw": off_raw,
            "def_raw": def_raw,
            "off_impact": off_raw,
            "def_impact": def_raw,
        })

    # NOTE: Paine's position adjustment is deliberately omitted here - see
    # module docstring, item 4 (our position data is ~97% NULL and applying
    # it non-uniformly broke the ratings of exactly the players who had it).
    _apply_team_adjustment(rows, teams, league_ortg)

    for r in rows:
        r["total_impact"] = r["off_impact"] + r["def_impact"]
        r["total_raw"] = r["off_raw"] + r["def_raw"]
    return rows


def _apply_weights(feats: Dict[str, float], weights: Dict[str, float]) -> float:
    total = weights["intercept"]
    for key, w in weights.items():
        if key == "intercept":
            continue
        total += w * feats.get(key, 0.0)
    return total


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(value, limit))


def _apply_team_adjustment(
    rows: List[Dict[str, Any]],
    teams: Dict[str, Dict[str, Any]],
    league_ortg: float,
) -> None:
    """Paine's team constraint: 4.5 x minute-weighted average player rating
    must equal the team's rating relative to league average."""
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_team.setdefault(r.get("team_abbr"), []).append(r)
    for abbr, members in by_team.items():
        team = teams.get(abbr)
        if not team:
            continue
        total_w = sum(m["total_minutes"] for m in members)
        if total_w <= 0:
            continue
        target_off = team["off_rating"] - league_ortg
        target_def = league_ortg - team["def_rating"]
        cur_off = 4.5 * sum(m["off_impact"] * m["total_minutes"] for m in members) / total_w
        cur_def = 4.5 * sum(m["def_impact"] * m["total_minutes"] for m in members) / total_w
        off_shift = (target_off - cur_off) / 4.5
        def_shift = (target_def - cur_def) / 4.5
        for m in members:
            m["off_impact"] += off_shift
            m["def_impact"] += def_shift


# ---------------------------------------------------------------------------
# Database loading + caching
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_season_inputs(season: str, season_type: str):
    conn = _connect()
    try:
        players = [
            {
                "player_id": r["player_id"],
                "name": r["full_name"],
                "team_abbr": r["team_abbr"],
                "position": r["position"],
                "gp": r["gp"],
                "min": r["min"],
                "pts": r["pts"],
                "fga": r["fga"],
                "fta": r["fta"],
                "ast": r["ast"],
                "tov": r["tov"],
                "oreb": r["oreb"],
                "dreb": r["dreb"],
                "stl": r["stl"],
                "blk": r["blk"],
                "pf": r["pf"],
                "off_rating": r["off_rating"],
                "def_rating": r["def_rating"],
                "pace": r["pace"],
            }
            for r in conn.execute(
                """
                SELECT s.*, p.full_name, b.position
                FROM player_season_stats s
                LEFT JOIN players p ON p.player_id = s.player_id
                LEFT JOIN player_bio b ON b.player_id = s.player_id
                WHERE s.season = ? AND s.season_type = ?
                """,
                (season, season_type),
            )
        ]
        teams = {
            r["abbreviation"]: {
                "off_rating": r["off_rating"],
                "def_rating": r["def_rating"],
                "games": r["games"],
            }
            for r in conn.execute(
                """
                SELECT m.abbreviation, t.off_rating, t.def_rating, t.games
                FROM team_season_advanced t
                JOIN team_metadata m ON m.team_id = t.team_id
                WHERE t.season = ? AND t.season_type = ?
                """,
                (season, season_type),
            )
        }
        return players, teams
    finally:
        conn.close()


def _season_impacts(season: str, season_type: str = "Regular Season") -> List[Dict[str, Any]]:
    """Full (unfiltered) impact list for a season, cached in memory."""
    key = (season, season_type)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    players, teams = _load_season_inputs(season, season_type)
    rows = compute_impacts(players, teams)
    rows.sort(key=lambda r: r["total_impact"], reverse=True)
    with _cache_lock:
        _cache[key] = rows
    return rows


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def available_seasons() -> List[str]:
    conn = _connect()
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT season FROM player_season_stats "
                "WHERE season_type = 'Regular Season' ORDER BY season"
            )
        ]
    finally:
        conn.close()


def _public_record(r: Dict[str, Any], season: str, rank: Optional[int]) -> Dict[str, Any]:
    return {
        "player_id": r["player_id"],
        "name": r["name"],
        "team_abbr": r["team_abbr"],
        "season": season,
        "gp": r["gp"],
        "min_per_g": round(r["min_per_g"], 1) if r["min_per_g"] is not None else None,
        "off_impact": round(r["off_impact"], 2),
        "def_impact": round(r["def_impact"], 2),
        "total_impact": round(r["total_impact"], 2),
        "impact_rank": rank,
    }


def get_impact_ratings(
    season: str,
    min_gp: int = DEFAULT_MIN_GP,
    season_type: str = "Regular Season",
) -> List[Dict[str, Any]]:
    """Qualified (gp >= min_gp) players ranked by total impact, best first."""
    rows = _season_impacts(season, season_type)
    qualified = [r for r in rows if r["gp"] >= min_gp]
    return [
        _public_record(r, season, i + 1)
        for i, r in enumerate(qualified)
    ]


def get_player_impact(player_id: int) -> List[Dict[str, Any]]:
    """Career series (within our data window) for one player.

    impact_rank is the player's rank among default-qualified players
    (gp >= DEFAULT_MIN_GP) for that season, or None if he did not qualify.
    """
    series: List[Dict[str, Any]] = []
    for season in available_seasons():
        rows = _season_impacts(season)
        qualified = [r for r in rows if r["gp"] >= DEFAULT_MIN_GP]
        rank_by_id = {r["player_id"]: i + 1 for i, r in enumerate(qualified)}
        for r in rows:
            if r["player_id"] == player_id:
                series.append(_public_record(r, season, rank_by_id.get(player_id)))
                break
    return series
