"""
ClutchLedger.py
===============
Every clutch event in the archive, attributed to players - the possession
list itself, not just a season aggregate.

CLUTCH here is the standard definition: last five minutes of the fourth
quarter or overtime with the margin at five or fewer, evaluated at the
moment of the event using the score BEFORE it (a shot taken up five that
makes it eight still counts - the game was clutch when it left his hand).
Overtime periods are five minutes long, so all of OT qualifies whenever the
margin condition holds.

WHAT GETS ATTRIBUTED: field-goal attempts (with threes split out), free
throws, and turnovers charged to a player. Team turnovers (no player id)
are skipped and counted. Rebounds, fouls, and defense are not in the
ledger - PBP attribution for those is either noisy or team-level.

THE HONEST PART, which is the product: clutch samples are TINY. A heavy
clutch season is ~150 attempts, so every rate ships with a Wilson 95%
interval, and the comparison column (clutch FG% minus the player's own
full-season FG%) should be read through that interval. Clutch
over/underperformance has weak year-to-year stability in the public
research; treat a hot clutch season as description, not prediction.
"""

import math
from typing import Any, Dict, List, Optional

CLUTCH_CLOCK = 300.0  # seconds remaining
CLUTCH_MARGIN = 5


def _wilson(k: int, n: int, z: float = 1.96) -> Optional[Dict[str, float]]:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"lo": round(max(0.0, center - half), 3), "hi": round(min(1.0, center + half), 3)}


def _fmt_clock(seconds: Optional[float]) -> str:
    s = int(seconds or 0)
    return f"{s // 60}:{s % 60:02d}"


def compute_clutch(conn, season: str, season_type: str = "Regular Season") -> Dict[str, Any]:
    # Team id -> abbreviation, for opponent labels.
    abbr = {r["team_id"]: r["abbreviation"] for r in conn.execute(
        "SELECT team_id, abbreviation FROM team_metadata")}

    # PBP player_name is the bare last name; full names come from the
    # directory. Ids missing there keep the PBP surname rather than a blank.
    full_names = {r["player_id"]: r["full_name"] for r in conn.execute(
        "SELECT player_id, full_name FROM players")}

    games = {g["game_id"]: g for g in conn.execute(
        "SELECT game_id, game_date, home_team_id, away_team_id FROM box_scores "
        "WHERE season = ? AND season_type = ?", (season, season_type))}

    if not games:
        return {"season": season, "season_type": season_type, "players": [], "events_total": 0}

    # Full-season shooting per player for the comparison column. This is one
    # SQL aggregate over every FGA of the season (clutch included - the
    # standard baseline).
    overall: Dict[int, Dict[str, int]] = {}
    for r in conn.execute(
        "SELECT e.person_id AS pid, COUNT(*) AS fga, "
        "SUM(CASE WHEN e.action_type = 'Made Shot' THEN 1 ELSE 0 END) AS fgm "
        "FROM pbp_events e JOIN box_scores b ON b.game_id = e.game_id "
        "WHERE b.season = ? AND b.season_type = ? AND e.is_field_goal = 1 "
        "AND e.person_id IS NOT NULL GROUP BY e.person_id",
        (season, season_type),
    ):
        overall[r["pid"]] = {"fga": r["fga"], "fgm": r["fgm"]}

    # The walk: every 4th-quarter-and-later event, in order, tracking the
    # score BEFORE each event.
    events = conn.execute(
        "SELECT e.game_id, e.action_number, e.period, e.clock_seconds, e.team_tricode, "
        "e.person_id, e.player_name, e.action_type, e.sub_type, e.description, "
        "e.shot_value, e.is_field_goal, e.score_home, e.score_away "
        "FROM pbp_events e JOIN box_scores b ON b.game_id = e.game_id "
        "WHERE b.season = ? AND b.season_type = ? AND e.period >= 4 "
        "ORDER BY e.game_id, e.action_number",
        (season, season_type),
    ).fetchall()

    acc: Dict[int, Dict[str, Any]] = {}
    ledger: Dict[int, List[Dict[str, Any]]] = {}
    team_turnovers_skipped = 0
    events_total = 0

    cur_game = None
    h = a = 0
    score_known = False  # never call an event clutch off an assumed 0-0

    for ev in events:
        if ev["game_id"] != cur_game:
            cur_game = ev["game_id"]
            h = a = 0
            score_known = False

        atype = ev["action_type"]

        # Period rows reset the running score at quarter boundaries.
        if atype == "period":
            if ev["score_home"] is not None:
                h, a = int(ev["score_home"]), int(ev["score_away"])
                score_known = True
            continue

        margin_before = abs(h - a)
        clock = ev["clock_seconds"]
        is_clutch = (score_known and clock is not None
                     and clock <= CLUTCH_CLOCK and margin_before <= CLUTCH_MARGIN)

        # Update the running score AFTER evaluating clutchness.
        if ev["score_home"] is not None and ev["score_away"] is not None:
            h, a = int(ev["score_home"]), int(ev["score_away"])
            score_known = True

        if not is_clutch:
            continue
        if atype not in ("Made Shot", "Missed Shot", "Free Throw", "Turnover"):
            continue

        pid = ev["person_id"]
        if not pid:
            if atype == "Turnover":
                team_turnovers_skipped += 1
            continue

        g = games.get(ev["game_id"])
        opp = None
        if g:
            home_abbr = abbr.get(g["home_team_id"])
            away_abbr = abbr.get(g["away_team_id"])
            opp = away_abbr if ev["team_tricode"] == home_abbr else home_abbr

        p = acc.setdefault(pid, {
            "player_id": pid, "name": full_names.get(pid) or ev["player_name"],
            "team": ev["team_tricode"],
            "games": set(), "fga": 0, "fgm": 0, "fg3a": 0, "fg3m": 0,
            "fta": 0, "ftm": 0, "tov": 0, "pts": 0,
        })
        p["team"] = ev["team_tricode"] or p["team"]
        p["games"].add(ev["game_id"])
        events_total += 1

        desc = ev["description"] or ""
        if atype in ("Made Shot", "Missed Shot"):
            p["fga"] += 1
            three = (ev["shot_value"] == 3) or "3PT" in desc
            if three:
                p["fg3a"] += 1
            if atype == "Made Shot":
                p["fgm"] += 1
                p["pts"] += 3 if three else 2
                if three:
                    p["fg3m"] += 1
        elif atype == "Free Throw":
            p["fta"] += 1
            if not desc.startswith("MISS"):
                p["ftm"] += 1
                p["pts"] += 1
        elif atype == "Turnover":
            p["tov"] += 1

        ledger.setdefault(pid, []).append({
            "game_id": ev["game_id"],
            "date": g["game_date"] if g else None,
            "opp": opp,
            "period": ev["period"],
            "clock": _fmt_clock(clock),
            "margin_before": margin_before,
            "action": atype,
            "desc": desc,
        })

    players_out: List[Dict[str, Any]] = []
    league_fgm = sum(p["fgm"] for p in acc.values())
    league_fga = sum(p["fga"] for p in acc.values())
    for pid, p in acc.items():
        ov = overall.get(pid)
        fg_pct = round(p["fgm"] / p["fga"], 3) if p["fga"] else None
        overall_pct = round(ov["fgm"] / ov["fga"], 3) if ov and ov["fga"] else None
        players_out.append({
            "player_id": pid,
            "name": p["name"],
            "team": p["team"],
            "games": len(p["games"]),
            "fga": p["fga"], "fgm": p["fgm"],
            "fg3a": p["fg3a"], "fg3m": p["fg3m"],
            "fta": p["fta"], "ftm": p["ftm"],
            "tov": p["tov"], "pts": p["pts"],
            "fg_pct": fg_pct,
            "fg_ci": _wilson(p["fgm"], p["fga"]),
            "overall_fg_pct": overall_pct,
            "fg_diff": round(fg_pct - overall_pct, 3)
                if fg_pct is not None and overall_pct is not None else None,
        })
    players_out.sort(key=lambda x: -x["pts"])

    return {
        "season": season,
        "season_type": season_type,
        "games_covered": len(games),
        "events_total": events_total,
        "team_turnovers_skipped": team_turnovers_skipped,
        "league_clutch_fg_pct": round(league_fgm / league_fga, 3) if league_fga else None,
        "players": players_out,
        "_ledger": ledger,  # served via the events endpoint, stripped from the list payload
        "method": (
            "Clutch = last five minutes of the 4th quarter or overtime with the margin "
            "at five or fewer, judged by the score BEFORE each event. Attributed: field "
            "goals, free throws, player turnovers; team turnovers are skipped and "
            "counted. FG% intervals are Wilson 95% - clutch samples are tiny and the "
            "bars say so. The comparison column is against the player's own full-season "
            "FG%. Public research finds clutch over/underperformance has weak "
            "year-to-year stability: description, not prediction."
        ),
    }
