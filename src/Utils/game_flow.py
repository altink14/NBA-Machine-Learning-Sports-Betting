"""
game_flow.py
============
Pure, deterministic transform of stats.nba.com playbyplayv3 actions into a
compact "game flow" payload: score-margin series, scoring runs, lead changes
and ties. No network, no database - fully unit-testable.

playbyplayv3 action fields used (verified against Data/nba_cache files):
- ``period``     int, 1-4 regulation, 5+ overtime
- ``clock``      ISO-8601-style duration remaining in the period, e.g. "PT11M34.00S"
- ``scoreHome``  running home score as a string; EMPTY STRING on non-scoring events
- ``scoreAway``  running away score as a string; empty on non-scoring events
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

REGULATION_PERIOD_SECONDS = 720   # 12 minutes
OVERTIME_PERIOD_SECONDS = 300     # 5 minutes
MAX_RUNS = 6                      # most significant runs to report
RUN_MIN_POINTS = 8                # team must score at least this many...
RUN_MAX_OPP_POINTS = 2            # ...while the opponent scores at most this many

_CLOCK_RE = re.compile(
    r"^PT(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$"
)


def parse_clock(clock: Optional[str]) -> Optional[float]:
    """Parse a playbyplayv3 clock string ("PT11M34.00S") into seconds remaining
    in the period. Returns None when the string is missing or malformed."""
    if not clock:
        return None
    m = _CLOCK_RE.match(clock.strip())
    if not m or not any(m.group(g) for g in ("h", "m", "s")):
        return None
    hours = float(m.group("h") or 0)
    minutes = float(m.group("m") or 0)
    seconds = float(m.group("s") or 0)
    return hours * 3600 + minutes * 60 + seconds


def period_length(period: int) -> int:
    """Length of a period in seconds (720 regulation, 300 overtime)."""
    return REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS


def period_start_elapsed(period: int) -> int:
    """Seconds elapsed from game start at the beginning of `period`."""
    if period <= 4:
        return (period - 1) * REGULATION_PERIOD_SECONDS
    return 4 * REGULATION_PERIOD_SECONDS + (period - 5) * OVERTIME_PERIOD_SECONDS


def elapsed_seconds(period: int, clock: Optional[str]) -> Optional[int]:
    """Seconds elapsed from game start for a given period + clock-remaining.
    Regulation periods are 720 s, overtime periods 300 s."""
    remaining = parse_clock(clock)
    if remaining is None:
        return None
    return int(round(period_start_elapsed(period) + (period_length(period) - remaining)))


def _detect_runs(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect scoring runs from an ordered list of scoring events
    ({"t": int, "team": "home"|"away", "pts": int}).

    A run = one team scoring >= RUN_MIN_POINTS while the opponent scores
    <= RUN_MAX_OPP_POINTS (classic broadcast definition). Windows are grown
    greedily to be maximal, overlapping/adjacent same-team windows are merged
    when the merged window still qualifies, and at most MAX_RUNS runs are
    returned ranked by point differential, output in chronological order.
    """
    runs: List[Dict[str, Any]] = []
    n = len(events)

    for team in ("home", "away"):
        windows: List[List[int]] = []  # [start_idx, end_idx, team_pts, opp_pts]
        i = 0
        while i < n:
            if events[i]["team"] != team:
                i += 1
                continue
            # Grow a window starting at this team score until the opponent
            # would exceed RUN_MAX_OPP_POINTS.
            team_pts = 0
            opp_pts = 0
            last_team_idx = i
            k = i
            while k < n:
                e = events[k]
                if e["team"] == team:
                    team_pts += e["pts"]
                    last_team_idx = k
                else:
                    if opp_pts + e["pts"] > RUN_MAX_OPP_POINTS:
                        break
                    opp_pts += e["pts"]
                k += 1
            # A run ends on the team's last score - trim trailing opponent events.
            window_opp = sum(
                e["pts"] for e in events[i:last_team_idx + 1] if e["team"] != team
            )
            if team_pts >= RUN_MIN_POINTS:
                windows.append([i, last_team_idx, team_pts, window_opp])
                i = last_team_idx + 1
            else:
                i += 1

        # Merge overlapping/adjacent windows for the same team when the merged
        # span still satisfies the opponent cap.
        merged: List[List[int]] = []
        for win in windows:
            if merged and win[0] <= merged[-1][1] + 1:
                s = merged[-1][0]
                e2 = max(merged[-1][1], win[1])
                pts = sum(ev["pts"] for ev in events[s:e2 + 1] if ev["team"] == team)
                opp = sum(ev["pts"] for ev in events[s:e2 + 1] if ev["team"] != team)
                if opp <= RUN_MAX_OPP_POINTS:
                    merged[-1] = [s, e2, pts, opp]
                    continue
            merged.append(win)

        for s, e2, pts, opp in merged:
            runs.append({
                "team": team,
                "points": pts,
                "opp_points": opp,
                "start_t": events[s]["t"],
                "end_t": events[e2]["t"],
                "label": f"{pts}-{opp} run",
            })

    # Keep the MAX_RUNS most significant by point differential, then present
    # chronologically.
    runs.sort(key=lambda r: (-(r["points"] - r["opp_points"]), r["start_t"]))
    runs = runs[:MAX_RUNS]
    runs.sort(key=lambda r: r["start_t"])
    return runs


def build_game_flow(
    game_id: str,
    actions: List[Dict[str, Any]],
    home: Dict[str, str],
    away: Dict[str, str],
) -> Dict[str, Any]:
    """
    Transform playbyplayv3 actions into the game-flow payload.

    `home` / `away` are {"abbr": ..., "name": ...} descriptors. `actions` must
    be in chronological order (as returned by playbyplayv3).
    """
    series: List[Dict[str, Any]] = [
        {"t": 0, "period": 1, "margin": 0, "home_score": 0, "away_score": 0}
    ]
    events: List[Dict[str, Any]] = []  # scoring events feeding run detection
    prev_home = 0
    prev_away = 0
    prev_t = 0
    max_period = 1

    for action in actions:
        raw_home = action.get("scoreHome")
        raw_away = action.get("scoreAway")
        if raw_home in (None, "") or raw_away in (None, ""):
            continue  # non-scoring event: scores are empty strings
        try:
            home_score = int(raw_home)
            away_score = int(raw_away)
        except (TypeError, ValueError):
            continue

        try:
            period = int(action.get("period") or max_period)
        except (TypeError, ValueError):
            period = max_period
        if period > max_period:
            max_period = period

        t = elapsed_seconds(period, action.get("clock"))
        if t is None:
            t = prev_t
        t = max(t, prev_t)  # guard against out-of-order clock glitches
        prev_t = t

        if home_score == prev_home and away_score == prev_away:
            continue  # score unchanged (period markers repeat the score)

        delta_home = home_score - prev_home
        delta_away = away_score - prev_away
        prev_home, prev_away = home_score, away_score

        series.append({
            "t": t,
            "period": period,
            "margin": home_score - away_score,
            "home_score": home_score,
            "away_score": away_score,
        })
        if delta_home > 0:
            events.append({"t": t, "team": "home", "pts": delta_home})
        if delta_away > 0:
            events.append({"t": t, "team": "away", "pts": delta_away})

    final = {"home": prev_home, "away": prev_away}

    # Extend the series to the final horn so charts span the whole game.
    end_t = period_start_elapsed(max_period) + period_length(max_period)
    last_point = series[-1]
    if last_point["t"] < end_t:
        series.append({
            "t": end_t,
            "period": max_period,
            "margin": prev_home - prev_away,
            "home_score": prev_home,
            "away_score": prev_away,
        })

    # Lead changes: sign flips of the margin (through-zero touches that return
    # to the same side do NOT count). Ties: score returns to level after being
    # unlevel (the 0-0 start does not count).
    lead_changes = 0
    ties = 0
    last_sign = 0
    for point in series[1:]:
        margin = point["margin"]
        sign = (margin > 0) - (margin < 0)
        if sign == 0:
            if last_sign != 0:
                ties += 1
        else:
            if last_sign != 0 and sign != last_sign:
                lead_changes += 1
            last_sign = sign

    return {
        "game_id": game_id,
        "home": home,
        "away": away,
        "final": final,
        "series": series,
        "runs": _detect_runs(events),
        "lead_changes": lead_changes,
        "ties": ties,
    }
