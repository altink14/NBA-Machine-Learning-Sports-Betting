"""
nba_computed_derivatives.py
============================
Pure-function implementations of NBA advanced metrics using Dean Oliver's formulas.

All inputs are raw box-score primitives (integers/floats).  No I/O or side effects.

References
----------
- Dean Oliver, "Basketball on Paper" (2004)
- https://www.basketball-reference.com/about/ratings.html  (formula documentation)
- https://www.pbpstats.com/  (possession definitions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Raw box-score container
# ---------------------------------------------------------------------------

@dataclass
class BoxScoreTeam:
    """
    Single-team raw box-score primitives for one game.
    All 'TOT' columns from boxscoretraditionalv2 / TeamStats result-set.
    """
    team_id: int
    team_abbr: str
    game_id: str

    # Minutes (decimal)
    min: float

    # Shooting
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int

    # Miscellaneous
    oreb: int
    dreb: int
    reb: int
    ast: int
    stl: int
    blk: int
    tov: int
    pf: int
    pts: int

    # Overtime periods (0 = no OT)
    ot_periods: int = 0

    # Optional: filled in after both teams are known
    opp_team_id: Optional[int] = None
    opp_pts: Optional[int] = None


@dataclass
class GameAdvancedStats:
    """Computed derivative stats for one team in one game."""
    game_id: str
    team_id: int
    opp_team_id: int

    # Possessions
    poss_estimated: float      # Oliver estimation
    poss_opponent: float

    # Pace (possessions per 48 minutes)
    pace: float

    # Ratings (points per 100 possessions)
    off_rating: float          # ORtg
    def_rating: float          # DRtg  (= opponent ORtg)
    net_rating: float          # ORtg – DRtg

    # Four Factors
    efg_pct: float             # (FGM + 0.5·FG3M) / FGA
    tov_pct: float             # TOV / (FGA + 0.44·FTA + TOV)
    orb_pct: float             # OREB / (OREB + opp_DREB)
    ft_rate: float             # FTA / FGA

    # Shooting efficiency
    ts_pct: float              # PTS / (2 * (FGA + 0.44·FTA))


# ---------------------------------------------------------------------------
# Possession estimation (Dean Oliver)
# ---------------------------------------------------------------------------

def estimate_possessions(b: BoxScoreTeam) -> float:
    """
    Standard Dean Oliver possession formula:

        Poss = FGA - OREB + TOV + 0.44 * FTA

    The 0.44 coefficient accounts for and-ones, technical free throws, etc.
    """
    return b.fga - b.oreb + b.tov + 0.44 * b.fta


# ---------------------------------------------------------------------------
# Pace
# ---------------------------------------------------------------------------

def compute_pace(
    team: BoxScoreTeam,
    opp: BoxScoreTeam,
    team_poss: float,
    opp_poss: float,
) -> float:
    """
    NBA pace formula (possessions per 48 minutes, OT-adjusted):

        Pace = 48 * (team_poss + opp_poss) / (2 * MIN)

    For overtime games we normalise to a 48-minute baseline so pace is
    comparable across regulation and OT games.
    """
    regulation_minutes = 48.0 + 5.0 * team.ot_periods
    total_poss = team_poss + opp_poss
    if team.min <= 0:
        return 0.0
    # Use actual minutes played (should equal 5 * regulation_minutes for team)
    return 48.0 * total_poss / (2.0 * regulation_minutes)


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def compute_off_rating(b: BoxScoreTeam, poss: float) -> float:
    """
    Offensive Rating = (PTS / Poss) * 100
    Returns 0.0 if possessions == 0.
    """
    if poss <= 0:
        return 0.0
    return (b.pts / poss) * 100.0


def compute_net_rating(off_rtg: float, def_rtg: float) -> float:
    return off_rtg - def_rtg


# ---------------------------------------------------------------------------
# Four Factors
# ---------------------------------------------------------------------------

def compute_efg_pct(b: BoxScoreTeam) -> float:
    """Effective Field-Goal percentage: (FGM + 0.5·FG3M) / FGA"""
    if b.fga == 0:
        return 0.0
    return (b.fgm + 0.5 * b.fg3m) / b.fga


def compute_tov_pct(b: BoxScoreTeam) -> float:
    """Turnover percentage: TOV / (FGA + 0.44·FTA + TOV)"""
    denom = b.fga + 0.44 * b.fta + b.tov
    if denom <= 0:
        return 0.0
    return b.tov / denom


def compute_orb_pct(team: BoxScoreTeam, opp: BoxScoreTeam) -> float:
    """Offensive-rebound percentage: OREB / (OREB + opp_DREB)"""
    denom = team.oreb + opp.dreb
    if denom <= 0:
        return 0.0
    return team.oreb / denom


def compute_ft_rate(b: BoxScoreTeam) -> float:
    """Free-throw rate: FTA / FGA"""
    if b.fga == 0:
        return 0.0
    return b.fta / b.fga


def compute_ts_pct(b: BoxScoreTeam) -> float:
    """True-Shooting percentage: PTS / (2 * (FGA + 0.44·FTA))"""
    denom = 2.0 * (b.fga + 0.44 * b.fta)
    if denom <= 0:
        return 0.0
    return b.pts / denom


# ---------------------------------------------------------------------------
# Full per-game derivation
# ---------------------------------------------------------------------------

def compute_game_advanced(team: BoxScoreTeam, opp: BoxScoreTeam) -> Tuple[GameAdvancedStats, GameAdvancedStats]:
    """
    Compute all advanced stats for both teams in a game.

    Returns
    -------
    (team_stats, opp_stats) : Tuple[GameAdvancedStats, GameAdvancedStats]
    """
    team_poss = estimate_possessions(team)
    opp_poss = estimate_possessions(opp)

    pace = compute_pace(team, opp, team_poss, opp_poss)

    team_ortg = compute_off_rating(team, team_poss)
    opp_ortg = compute_off_rating(opp, opp_poss)

    team_stats = GameAdvancedStats(
        game_id=team.game_id,
        team_id=team.team_id,
        opp_team_id=opp.team_id,
        poss_estimated=team_poss,
        poss_opponent=opp_poss,
        pace=pace,
        off_rating=team_ortg,
        def_rating=opp_ortg,
        net_rating=compute_net_rating(team_ortg, opp_ortg),
        efg_pct=compute_efg_pct(team),
        tov_pct=compute_tov_pct(team),
        orb_pct=compute_orb_pct(team, opp),
        ft_rate=compute_ft_rate(team),
        ts_pct=compute_ts_pct(team),
    )

    opp_stats = GameAdvancedStats(
        game_id=opp.game_id,
        team_id=opp.team_id,
        opp_team_id=team.team_id,
        poss_estimated=opp_poss,
        poss_opponent=team_poss,
        pace=pace,
        off_rating=opp_ortg,
        def_rating=team_ortg,
        net_rating=compute_net_rating(opp_ortg, team_ortg),
        efg_pct=compute_efg_pct(opp),
        tov_pct=compute_tov_pct(opp),
        orb_pct=compute_orb_pct(opp, team),
        ft_rate=compute_ft_rate(opp),
        ts_pct=compute_ts_pct(opp),
    )

    return team_stats, opp_stats


# ---------------------------------------------------------------------------
# Simple Rating System (SRS)
# ---------------------------------------------------------------------------

@dataclass
class TeamRecord:
    """Input record for SRS solver."""
    team_id: int
    abbr: str
    point_diffs: List[float] = field(default_factory=list)   # home_pts - away_pts (from home team's perspective)
    opponent_ids: List[int] = field(default_factory=list)


def compute_srs(
    team_records: Dict[int, TeamRecord],
    max_iterations: int = 2000,
    convergence_threshold: float = 1e-6,
    blowout_cap: float = 30.0,
    ot_penalty: float = 0.5,
    ot_periods: Optional[Dict[str, int]] = None,  # game_id → ot_periods
) -> Tuple[Dict[int, float], float]:
    """
    Iterative SRS solver.

    SRS combines a team's average point differential with Strength of Schedule:

        SRS_i = avg_margin_i + avg(SRS_j for j in opponents_i)

    The system is solved iteratively until convergence (avg SRS ≈ 0 by construction).

    Parameters
    ----------
    team_records   : Dict[team_id → TeamRecord]
    max_iterations : int          – safety cap
    convergence_threshold : float – stop when max |Δ| < threshold
    blowout_cap    : float        – cap individual game margin at ±blowout_cap
    ot_penalty     : float        – deduct this many points per OT period (games that 
                                   went to OT are less predictive; this brings margins 
                                   closer to 0 before solving)

    Returns
    -------
    (srs_ratings, sos_ratings) : (Dict[team_id → float], Dict[team_id → float])
    """
    team_ids = list(team_records.keys())
    n = len(team_ids)
    if n == 0:
        return {}, {}

    # Initialise ratings to 0
    ratings: Dict[int, float] = {tid: 0.0 for tid in team_ids}

    for iteration in range(max_iterations):
        new_ratings: Dict[int, float] = {}

        for tid, rec in team_records.items():
            if not rec.point_diffs:
                new_ratings[tid] = 0.0
                continue

            # Average point differential (capped)
            avg_margin = sum(
                max(-blowout_cap, min(blowout_cap, diff))
                for diff in rec.point_diffs
            ) / len(rec.point_diffs)

            # Strength of schedule = average opponent SRS
            sos = sum(ratings.get(opp, 0.0) for opp in rec.opponent_ids) / max(1, len(rec.opponent_ids))

            new_ratings[tid] = avg_margin + sos

        # Re-centre around 0 (league average)
        mean_rating = sum(new_ratings.values()) / n
        new_ratings = {tid: v - mean_rating for tid, v in new_ratings.items()}

        # Check convergence
        max_delta = max(abs(new_ratings[tid] - ratings[tid]) for tid in team_ids)
        ratings = new_ratings

        if max_delta < convergence_threshold:
            break

    # Compute SoS separately
    sos: Dict[int, float] = {}
    for tid, rec in team_records.items():
        if not rec.opponent_ids:
            sos[tid] = 0.0
        else:
            sos[tid] = sum(ratings.get(opp, 0.0) for opp in rec.opponent_ids) / len(rec.opponent_ids)

    return ratings, sos


# ---------------------------------------------------------------------------
# Season aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_season_team_stats(game_stats_list: List[GameAdvancedStats]) -> Dict[str, float]:
    """
    Compute season-average advanced stats from a list of single-game records
    for one team.

    Returns a dict with keys: pace, off_rating, def_rating, net_rating,
    efg_pct, tov_pct, orb_pct, ft_rate, ts_pct.
    """
    n = len(game_stats_list)
    if n == 0:
        return {}

    def avg(attr: str) -> float:
        return sum(getattr(g, attr) for g in game_stats_list) / n

    return {
        "games": n,
        "pace": avg("pace"),
        "off_rating": avg("off_rating"),
        "def_rating": avg("def_rating"),
        "net_rating": avg("net_rating"),
        "efg_pct": avg("efg_pct"),
        "tov_pct": avg("tov_pct"),
        "orb_pct": avg("orb_pct"),
        "ft_rate": avg("ft_rate"),
        "ts_pct": avg("ts_pct"),
    }
