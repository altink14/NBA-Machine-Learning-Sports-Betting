"""
Hustle.py
=========
Shapes nba.com's leaguehustlestatsplayer rows for /api/stats/hustle.

UNTRACKED IS NULL, NOT ZERO (nav audit bug 16, 2026-09-27). nba.com did not
count box-outs until 2017-18, but its feed still returns BOX_OUTS = 0 (and
OFF_BOXOUTS = DEF_BOXOUTS = 0) for every player in 2015-16 (147 of 147) and
2016-17 (485 of 485). A zero there means "not measured", and the Hustle
Builder was ranking players on it and printing "0.0" as if every player in
the league had put a body on nobody. The rule here is data-driven rather than
a hand-typed first season: a stat that is zero for EVERY player in a season
was not tracked that season, so it becomes None for all of them and is named
in `untracked`. No real tracked season comes close to all-zero (2017-18:
512 of 536 players have a box-out; 2025-26: 533 of 581).

The raw nba.com payload in Data/nba_cache is left exactly as nba.com sent it;
this is the single place those rows are read, so the correction lives here
and there is no stored copy to repair.
"""

from typing import Any, Dict, List, Optional, Tuple

# Our field -> nba.com's column. The order is the order the page shows them.
HUSTLE_FIELDS = (
    ("deflections", "DEFLECTIONS"),
    ("screen_assists", "SCREEN_ASSISTS"),
    ("screen_assist_pts", "SCREEN_AST_PTS"),
    ("loose_balls", "LOOSE_BALLS_RECOVERED"),
    ("charges_drawn", "CHARGES_DRAWN"),
    ("contested_shots", "CONTESTED_SHOTS"),
    ("box_outs", "BOX_OUTS"),
)


def _is_counted(v: Any) -> bool:
    return isinstance(v, (int, float)) and v != 0


def untracked_fields(rows: List[Dict[str, Any]]) -> List[str]:
    """Our field names whose nba.com column is zero or missing for every row."""
    if not rows:
        return []
    return [ours for ours, theirs in HUSTLE_FIELDS
            if not any(_is_counted(r.get(theirs)) for r in rows)]


def shape_players(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(players, untracked): one dict per player, untracked stats as None."""
    untracked = set(untracked_fields(rows))
    players = []
    for r in rows:
        p: Dict[str, Any] = {
            "player_id": r.get("PLAYER_ID"),
            "name": r.get("PLAYER_NAME"),
            "team": r.get("TEAM_ABBREVIATION"),
            "gp": r.get("G"),
            "min": r.get("MIN"),
        }
        for ours, theirs in HUSTLE_FIELDS:
            p[ours] = None if ours in untracked else r.get(theirs)
        players.append(p)
    return players, [f for f, _ in HUSTLE_FIELDS if f in untracked]


def season_coverage(rows: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """How much of the season the feed covers. 2015-16's regular season has
    147 players and nobody past 2 games: nba.com tracked only its last few
    nights, which a page must not present as a season."""
    games = [r.get("G") for r in rows if isinstance(r.get("G"), (int, float))]
    return {
        "players": len(rows),
        "max_games_played": int(max(games)) if games else None,
    }
