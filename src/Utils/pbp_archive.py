"""
pbp_archive.py
==============
Play-by-play and shot charts served from our own `pbp_events` table instead
of a request-time stats.nba.com call. Written 2026-09-28 for the public
server, which cannot reach stats.nba.com (NBA_STATS_LIVE=off), and used on
the home PC too, where it saves a request per page.

WHY THIS IS THE SAME DATA. pbp_events is playbyplayv3 stored row for row
(backfill_pbp.py), 2019-20 onward, kept current by the daily job. Checked
2026-09-28 against the league-wide shotchartdetail answers cached on disk:
for 11,354 field goals in 65 games of 2024-25 the coordinates (LOC_X/LOC_Y
= xLegacy/yLegacy), make/miss, shooter, period, action type and shot type
agreed on every one, and shotchartdetail's SHOT_DISTANCE is exactly
floor(hypot(x, y) / 10) (all 17,670 checked), which is what is served here.

ZONES. shotchartdetail labels each shot with nba.com's zone (basic / area /
range); play-by-play does not. `classify_zone` puts a shot in a zone from
its coordinates with the rule below, which reproduced nba.com's own triple on
99.89-99.96% of 409,600 shots (2019-20, 2022-23, 2023-24, 2024-25; the misses
are mostly nba.com calling a shot typed as a three "Mid-Range", or a heave
from just inside half court "Back Court"). Responses built here say so in
`zones_method`.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

ZONES_METHOD = (
    "Zones classified from shot coordinates by our rule; it matches nba.com's "
    "own zone labels on 99.9% of 409,600 shots checked (2019-20, 2022-23 to 2024-25)."
)
SOURCE = "Betting Buddy play-by-play archive (stats.nba.com playbyplayv3, stored in pbp_events)"

_HALF_COURT_Y = 422.5     # tenths of feet from the basket to the half-court line
_CORNER_Y = 87            # the corner three runs straight up to here
_CORNER_X = 220           # 22 ft, the corner three's distance
_RA_RADIUS = 40           # restricted area, 4 ft
_PAINT_HALF_WIDTH = 80    # the lane is 16 ft wide
_PAINT_TOP_Y = 138        # nba.com's paint ends here (measured, not the 142.5 of the drawn line)


def shot_distance(x: float, y: float) -> int:
    """shotchartdetail's SHOT_DISTANCE: whole feet, rounded down."""
    return int(math.hypot(x, y) / 10)


def classify_zone(x: float, y: float, is_three: bool) -> Tuple[str, str, str]:
    """(zone_basic, zone_area, zone_range) as nba.com names them."""
    if y >= _HALF_COURT_Y:
        return "Backcourt", "Back Court(BC)", "Back Court Shot"
    dist = shot_distance(x, y)
    if is_three:
        if y <= _CORNER_Y and abs(x) >= _CORNER_X:
            return (("Left Corner 3", "Left Side(L)", "24+ ft.") if x < 0
                    else ("Right Corner 3", "Right Side(R)", "24+ ft."))
        basic, rng = "Above the Break 3", "24+ ft."
    elif math.hypot(x, y) <= _RA_RADIUS:
        return "Restricted Area", "Center(C)", "Less Than 8 ft."
    else:
        basic = ("In The Paint (Non-RA)"
                 if abs(x) <= _PAINT_HALF_WIDTH and y <= _PAINT_TOP_Y else "Mid-Range")
        rng = ("Less Than 8 ft." if dist < 8 else "8-16 ft." if dist < 16
               else "16-24 ft." if dist < 24 else "24+ ft.")
    ang = math.degrees(math.atan2(y, x))    # 90 = straight out from the basket
    if rng == "Less Than 8 ft.":
        area = "Center(C)"
    elif rng == "8-16 ft.":
        area = "Center(C)" if 60 <= ang <= 120 else ("Left Side(L)" if x < 0 else "Right Side(R)")
    elif 72 <= ang <= 108:
        area = "Center(C)"
    elif 108 < ang <= 144 or (is_three and x < 0):
        area = "Left Side Center(LC)"
    elif 36 <= ang < 72 or is_three:
        area = "Right Side Center(RC)"
    else:
        area = "Left Side(L)" if x < 0 else "Right Side(R)"
    return basic, area, rng


def _clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    m, s = divmod(float(seconds), 60)
    return f"PT{int(m):02d}M{s:05.2f}S"


def pbp_actions(conn: sqlite3.Connection, game_id: str) -> Optional[List[Dict[str, Any]]]:
    """A game's play-by-play in playbyplayv3's action shape, or None if we do
    not hold it. Only fields we store are present; a score is the empty
    string on actions that did not change it, as in the feed."""
    rows = conn.execute(
        """
        SELECT action_number, action_id, period, clock_seconds, team_id, team_tricode,
               person_id, player_name, action_type, sub_type, description,
               loc_x, loc_y, shot_distance, shot_value, shot_result, is_field_goal,
               score_home, score_away
        FROM pbp_events WHERE game_id = ? ORDER BY action_id
        """,
        (game_id,),
    ).fetchall()
    if not rows:
        return None
    out = []
    for r in rows:
        out.append({
            "gameId": game_id,
            "actionNumber": r[0], "actionId": r[1], "period": r[2],
            "clock": _clock(r[3]),
            "teamId": r[4] or 0, "teamTricode": r[5] or "",
            "personId": r[6] or 0, "playerName": r[7] or "",
            "actionType": r[8] or "", "subType": r[9] or "", "description": r[10] or "",
            "xLegacy": r[11], "yLegacy": r[12], "shotDistance": r[13],
            "shotValue": r[14], "shotResult": r[15] or "", "isFieldGoal": r[16] or 0,
            "scoreHome": "" if r[17] is None else str(r[17]),
            "scoreAway": "" if r[18] is None else str(r[18]),
        })
    return out


def game_shots(conn: sqlite3.Connection, game_id: str) -> Optional[List[Dict[str, Any]]]:
    """The game shot chart's `shots` list, or None if we hold no play-by-play
    for the game (so the caller can say "not available", not "no shots")."""
    if not conn.execute("SELECT 1 FROM pbp_events WHERE game_id = ? LIMIT 1", (game_id,)).fetchone():
        return None
    shots = []
    for x, y, result, value, name, full in conn.execute(
        """
        SELECT e.loc_x, e.loc_y, e.shot_result, e.shot_value, e.player_name, p.full_name
        FROM pbp_events e LEFT JOIN players p ON p.player_id = e.person_id
        WHERE e.game_id = ? AND e.is_field_goal = 1 AND e.loc_x IS NOT NULL AND e.loc_y IS NOT NULL
        ORDER BY e.action_id
        """,
        (game_id,),
    ):
        made = result == "Made"
        who = full or name or "Unknown"
        shots.append({
            "player": who, "x": x, "y": y,
            "result": "made" if made else "missed",
            "description": f"{who} {'makes' if made else 'misses'} {value}PT Field Goal "
                           f"from {shot_distance(x, y)} ft",
        })
    return shots


def player_shots(conn: sqlite3.Connection, player_id: int, season: str,
                 season_type: str) -> Optional[Dict[str, Any]]:
    """A player-season's shots in /api/player-shot-chart's shape, plus coverage.

    None when we hold no play-by-play for any game of that season-type (before
    2019-20, or a season-type not yet played). `coverage` counts the player's
    games we have play-by-play for, so a hole is stated, never silent.
    """
    held = conn.execute(
        """
        SELECT COUNT(*) FROM box_scores b
        WHERE b.season = ? AND b.season_type = ?
          AND EXISTS (SELECT 1 FROM pbp_events e WHERE e.game_id = b.game_id)
        """,
        (season, season_type),
    ).fetchone()[0]
    if not held:
        return None
    games = conn.execute(
        """
        SELECT g.game_id, (SELECT 1 FROM pbp_events e WHERE e.game_id = g.game_id LIMIT 1)
        FROM player_game_log g JOIN box_scores b ON b.game_id = g.game_id
        WHERE g.player_id = ? AND b.season = ? AND b.season_type = ?
        """,
        (player_id, season, season_type),
    ).fetchall()
    rows = conn.execute(
        """
        SELECT e.game_id, b.game_date, e.action_type, e.sub_type, e.shot_value,
               e.loc_x, e.loc_y, e.shot_result, e.period, e.action_number, e.action_id
        FROM pbp_events e JOIN box_scores b ON b.game_id = e.game_id
        WHERE e.person_id = ? AND e.is_field_goal = 1
          AND b.season = ? AND b.season_type = ?
          AND e.loc_x IS NOT NULL AND e.loc_y IS NOT NULL
        ORDER BY e.game_id, e.action_id
        """,
        (player_id, season, season_type),
    ).fetchall()
    shots = []
    for gid, gdate, atype, sub, value, x, y, result, period, number, _aid in rows:
        basic, area, rng = classify_zone(x, y, value == 3)
        shots.append({
            "game_id": gid,
            "game_date": str(gdate or "")[:10].replace("-", ""),
            "event_type": atype,
            "action_type": sub,
            "shot_type": f"{value}PT Field Goal",
            "zone_basic": basic, "zone_area": area, "zone_range": rng,
            "distance": shot_distance(x, y),
            "x": x, "y": y,
            "made": result == "Made",
            "period": period,
            "game_event_id": number,
        })
    # The run coming into each shot, within its game (same rule as the live
    # route: +n straight makes, -n straight misses, 0 for the first attempt).
    streak, last_game = 0, None
    for s in shots:
        if s["game_id"] != last_game:
            streak, last_game = 0, s["game_id"]
        s["streak_before"] = streak
        streak = (streak + 1 if streak > 0 else 1) if s["made"] else (streak - 1 if streak < 0 else -1)
    return {
        "shots": shots,
        "coverage": {
            "games_played": len(games),
            "games_with_play_by_play": sum(1 for _, has in games if has),
        },
    }
