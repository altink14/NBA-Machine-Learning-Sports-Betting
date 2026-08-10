"""
elo.py
======
Game-by-game NBA Elo ratings computed from Data/TeamData.sqlite, in the style
of FiveThirtyEight's NBA Elo.

Methodology and constants
-------------------------
- BASE_RATING = 1500. Every team starts at 1500 at the beginning of 2022-23.
  That is the earliest season in the database — we have no pre-2022-23 games,
  so unlike 538 (which carries ratings back to 1946) the first few months of
  2022-23 are a cold start and early-2022-23 numbers should be read as rough.
- K_FACTOR = 20 (538's NBA value). Published sensitivity checks (and 538's own
  notes) show final orderings are insensitive to K anywhere in the 20-50 range,
  so the exact choice is not load-bearing.
- HOME_ADVANTAGE = 70 Elo points, added to the home team's rating ONLY when
  computing the expected score. Results are similarly insensitive to home
  advantage anywhere in the 50-150 range.
- Margin-of-victory multiplier, 538's published formula:
      mov_mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)
  where elo_diff_winner is the WINNER's pre-game rating minus the loser's,
  including the home-advantage adjustment. This damps blowout wins by heavy
  favourites (autocorrelation correction).
- SEASON_CARRYOVER = 0.75: between seasons every rating reverts 25% toward
  1505 (new = 0.75 * old + 0.25 * 1505), matching 538. Elo therefore flows
  continuously across seasons; a season parameter selects which season's
  final (or current) state to report.
- Playoff games are included and rated identically to regular-season games.

Home-court resolution
---------------------
team_game_advanced holds TWO rows per game (one per team) and has no
home/away flag. box_scores (same database, one row per game) stores
home_team_id and away_team_id for every game in 2022-23 through 2025-26, so
one row per game is derived by joining team_game_advanced to box_scores on
(game_id, team_id = home_team_id): pts is then the home score and opp_pts the
away score.

The full history (~5,255 games) computes in well under a second; callers
should cache the result in memory (main_api.py does, per process).
"""

import logging
import sqlite3
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

BASE_RATING = 1500.0
MEAN_REVERT_TARGET = 1505.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 70.0
SEASON_CARRYOVER = 0.75
HISTORY_START_SEASON = "2022-23"


def _expected_home_score(home_elo: float, away_elo: float) -> float:
    """Expected score (win probability) for the home team, incl. home court."""
    return 1.0 / (1.0 + 10.0 ** (-((home_elo + HOME_ADVANTAGE) - away_elo) / 400.0))


def _mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    """538 margin-of-victory multiplier; elo_diff_winner includes home court."""
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)


def compute_elo_history(db_path: str) -> Dict[str, Any]:
    """
    Run the full Elo history and return:
      {
        "timelines":         {team_id: [(game_date, elo_after), ...]},  # chronological
        "season_end_ratings": {season: {team_id: elo}},  # state after that season's
                                                         # last processed game, BEFORE
                                                         # the between-season reversion
        "season_last_date":  {season: 'YYYY-MM-DD'},
        "seasons":           [season, ...],  # chronological
        "n_games":           int,
        "elapsed_ms":        float,
      }
    """
    started = time.perf_counter()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT b.game_id, b.game_date, b.season, b.home_team_id, b.away_team_id,
                   t.pts AS home_pts, t.opp_pts AS away_pts
            FROM box_scores b
            JOIN team_game_advanced t
              ON t.game_id = b.game_id AND t.team_id = b.home_team_id
            ORDER BY b.game_date, b.game_id
            """
        )
        games = cursor.fetchall()
    finally:
        conn.close()

    ratings: Dict[int, float] = {}
    timelines: Dict[int, List[Tuple[str, float]]] = {}
    season_end_ratings: Dict[str, Dict[int, float]] = {}
    season_last_date: Dict[str, str] = {}
    seasons: List[str] = []
    current_season = None

    for game_id, game_date, season, home_id, away_id, home_pts, away_pts in games:
        if home_pts is None or away_pts is None or home_pts == away_pts:
            logger.warning(f"Elo: skipping game {game_id} with unusable score {home_pts}-{away_pts}")
            continue

        if season != current_season:
            if current_season is not None:
                # Snapshot the outgoing season, then revert 25% toward 1505.
                season_end_ratings[current_season] = dict(ratings)
                for team_id in ratings:
                    ratings[team_id] = (SEASON_CARRYOVER * ratings[team_id]
                                        + (1.0 - SEASON_CARRYOVER) * MEAN_REVERT_TARGET)
            current_season = season
            seasons.append(season)

        home_elo = ratings.get(home_id, BASE_RATING)
        away_elo = ratings.get(away_id, BASE_RATING)

        expected_home = _expected_home_score(home_elo, away_elo)
        home_won = home_pts > away_pts
        margin = home_pts - away_pts

        # Winner's Elo advantage including the home-court adjustment.
        if home_won:
            elo_diff_winner = (home_elo + HOME_ADVANTAGE) - away_elo
        else:
            elo_diff_winner = away_elo - (home_elo + HOME_ADVANTAGE)

        shift = K_FACTOR * _mov_multiplier(margin, elo_diff_winner) * ((1.0 if home_won else 0.0) - expected_home)
        ratings[home_id] = home_elo + shift
        ratings[away_id] = away_elo - shift

        timelines.setdefault(home_id, []).append((game_date, ratings[home_id]))
        timelines.setdefault(away_id, []).append((game_date, ratings[away_id]))
        season_last_date[season] = game_date

    if current_season is not None:
        season_end_ratings[current_season] = dict(ratings)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        f"Elo history computed: {len(games)} games, seasons {seasons[0] if seasons else '?'}"
        f"..{seasons[-1] if seasons else '?'}, {len(ratings)} teams, {elapsed_ms:.0f} ms"
    )

    return {
        "timelines": timelines,
        "season_end_ratings": season_end_ratings,
        "season_last_date": season_last_date,
        "seasons": seasons,
        "n_games": len(games),
        "elapsed_ms": elapsed_ms,
    }


def elo_as_of(timeline: List[Tuple[str, float]], date_str: str) -> float:
    """
    Rating from a team's timeline as of end-of-day `date_str` (last entry with
    date <= date_str). Falls back to BASE_RATING before any game was played.
    Timelines contain game entries only (no synthetic between-season points);
    that is fine for the 7-day windows this backs, since no 7-day window can
    straddle an offseason gap that has games on both sides.
    """
    rating = BASE_RATING
    for entry_date, elo in timeline:
        if entry_date > date_str:
            break
        rating = elo
    return rating
