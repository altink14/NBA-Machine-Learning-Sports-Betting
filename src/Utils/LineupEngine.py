"""
LineupEngine.py
===============
Lineup on/off engine: substitutions parsed into stints, stints into who
actually moves the score.

WHAT IT COMPUTES, per team-season:
  - per player: the team's net points per 100 possessions with that player ON
    the floor vs OFF it (off-court only counts games the player appeared in,
    so a month on the injured list doesn't pollute the split);
  - per five-man lineup: minutes, possessions, points for/against, net per 100.

HOW STINTS ARE BUILT. pbp_events stores every substitution, but only the
OUTGOING player carries an id - the incoming player exists only as a last name
inside "SUB: Incoming FOR Outgoing". Names resolve against the game's roster
(box_scores.traditional_json). Period-opening fives are inferred the standard
way: within a period, any player who acts before being substituted in must
have started it. Each period is processed independently, so one unresolvable
period costs exactly that period.

HONESTY. Nothing ambiguous is guessed: a period whose starters can't be pinned
to exactly five per team, or that hits an impossible substitution, or a last
name shared by two teammates, is EXCLUDED AND COUNTED, and the API reports the
count. Possessions are the same Dean Oliver estimate used elsewhere on the
site (FGA + 0.44*FTA - OREB + TO), computed per stint; free-throw-interleaved
substitutions attribute points by event order, the standard approximation.

GARBAGE TIME. Every stint is classified against a published rule (see
GARBAGE_DEFINITION); flagged stints are always counted in the output and
removed from every table only when exclude_garbage is requested. The rule is
deliberately conservative and end-anchored: a blowout that turns into a real
comeback retroactively stops being garbage.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

# "SUB: Harper FOR Fox" -> ("Harper", "Fox")
_SUB_RE = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$")

# The published garbage-time rule. A stint is garbage when ALL of:
#   - it starts in the 4th quarter (overtime is never garbage - the game was
#     tied after 48 minutes);
#   - the margin when the stint begins clears a time-scaled bar:
#     25+ at any point in the 4th, 20+ inside 9:00, 15+ inside 6:00,
#     10+ inside 3:00;
#   - the game ends with a double-digit margin. If a comeback brings it back
#     inside 10, nothing in that game counts as garbage.
GARBAGE_DEFINITION = (
    "4th-quarter stints beginning with a margin of 25+ at any time, 20+ inside "
    "9:00, 15+ inside 6:00, or 10+ inside 3:00, in games that end with a "
    "double-digit margin. Overtime is never garbage time, and a game a comeback "
    "brings back inside 10 has none."
)
_GARBAGE_FINAL_MARGIN = 10


def _stint_is_garbage(period: int, clock_start: float, margin_start: int,
                      final_margin: Optional[int]) -> bool:
    if period != 4 or final_margin is None or final_margin < _GARBAGE_FINAL_MARGIN:
        return False
    return (
        margin_start >= 25
        or (margin_start >= 20 and clock_start <= 540.0)
        or (margin_start >= 15 and clock_start <= 360.0)
        or (margin_start >= 10 and clock_start <= 180.0)
    )


def _norm(name: str) -> str:
    """Diacritic-free, case-free, period-free: 'Schröder' == 'Schroder',
    'Jay. Williams' == 'jay williams'. PBP text strips accents; rosters keep
    them - both sides must land on the same key."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.replace(".", " ").lower().split())


def _roster_from_traditional(traditional_json: str) -> Dict[str, Dict[str, Any]]:
    """
    tricode -> {normalized name key -> [person_id]}, plus id -> display name.

    Keys registered per player: the bare last name, and every first-name
    prefix form ('j williams', 'ja williams', ... 'jalen williams'), because
    PBP disambiguates same-surname teammates with initial prefixes
    ('J. Williams' vs 'Jay. Williams'). A key claimed by two teammates stays
    in the map with both ids and resolves as ambiguous - the PBP text for
    such games uses the longer, unique form.
    """
    data = json.loads(traditional_json)
    box = data.get("boxScoreTraditional") or {}
    teams: Dict[str, Dict[str, Any]] = {}
    names: Dict[int, str] = {}
    for side in ("homeTeam", "awayTeam"):
        team = box.get(side) or {}
        tricode = team.get("teamTricode")
        if not tricode:
            continue
        by_key: Dict[str, List[int]] = {}
        for p in team.get("players") or []:
            pid = p.get("personId")
            last = str(p.get("familyName") or "").strip()
            if not pid or not last:
                continue
            first = str(p.get("firstName") or "").strip()
            last_n = _norm(last)
            first_n = _norm(first).replace(" ", "")
            keys = {last_n}
            for k in range(1, len(first_n) + 1):
                keys.add(f"{first_n[:k]} {last_n}")
            for key in keys:
                ids = by_key.setdefault(key, [])
                if pid not in ids:
                    ids.append(pid)
            names[pid] = p.get("nameI") or f"{first[:1]}. {last}"
        teams[tricode] = by_key
    return {"teams": teams, "names": names}


class _PeriodDrop(Exception):
    """Raised when a period cannot be resolved honestly."""


def _infer_period_starters(events: List[Any], roster_teams: Dict[str, Dict[str, List[int]]],
                           tricodes: Tuple[str, str]) -> Dict[str, Set[int]]:
    """
    Who opened the period: every player who acts (including being subbed OUT)
    before they were first subbed IN must have been on the floor at the start.
    """
    started: Dict[str, Set[int]] = {t: set() for t in tricodes}
    entered: Dict[str, Set[int]] = {t: set() for t in tricodes}

    def resolve_in(tricode: str, sub_name: str) -> int:
        ids = (roster_teams.get(tricode) or {}).get(_norm(sub_name)) or []
        if len(ids) != 1:
            raise _PeriodDrop(f"ambiguous or unknown incoming name '{sub_name}' ({tricode})")
        return ids[0]

    for ev in events:
        tricode = ev["team_tricode"]
        if tricode not in started:
            continue
        pid = ev["person_id"]
        if ev["action_type"] == "Substitution":
            m = _SUB_RE.match(ev["description"] or "")
            if not m or not pid:
                raise _PeriodDrop("unparseable substitution")
            out_id = pid
            if out_id not in entered[tricode]:
                started[tricode].add(out_id)
            entered[tricode].add(resolve_in(tricode, m.group(1)))
        elif pid:
            if pid not in entered[tricode]:
                started[tricode].add(pid)

    for t in tricodes:
        if len(started[t]) > 5:
            raise _PeriodDrop(f"{t}: {len(started[t])} inferred starters")
        if len(started[t]) < 5:
            # Someone played the whole period without a single event. There is
            # no honest way to name them from PBP alone.
            raise _PeriodDrop(f"{t}: only {len(started[t])} inferable starters")
    return started


def _period_stints(events: List[Any], starters: Dict[str, Set[int]],
                   roster_teams: Dict[str, Dict[str, List[int]]],
                   tricodes: Tuple[str, str], period_start_clock: float,
                   score_in: Tuple[int, int],
                   period: int) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """
    Walk one period's events into stints. Each stint carries both fives, the
    clock span, the score movement, and per-team possession ingredients.
    """
    home_t, away_t = tricodes
    on = {t: set(starters[t]) for t in tricodes}
    score = list(score_in)  # [home, away] last known
    stints: List[Dict[str, Any]] = []

    def new_counter():
        return {t: {"fga": 0, "fta": 0, "to": 0, "oreb": 0} for t in tricodes}

    stint_open = {
        "clock_start": period_start_clock,
        "score_start": tuple(score),
        "home_on": frozenset(on[home_t]),
        "away_on": frozenset(on[away_t]),
        "counts": new_counter(),
    }
    last_miss_team: Optional[str] = None

    def close(clock_end: float):
        seconds = max(0.0, stint_open["clock_start"] - clock_end)
        stints.append({
            "seconds": seconds,
            "period": period,
            "clock_start": stint_open["clock_start"],
            "margin_start": abs(stint_open["score_start"][0] - stint_open["score_start"][1]),
            "home_on": stint_open["home_on"],
            "away_on": stint_open["away_on"],
            "home_pts": score[0] - stint_open["score_start"][0],
            "away_pts": score[1] - stint_open["score_start"][1],
            "counts": stint_open["counts"],
        })

    def reopen(clock: float):
        stint_open.update({
            "clock_start": clock,
            "score_start": tuple(score),
            "home_on": frozenset(on[home_t]),
            "away_on": frozenset(on[away_t]),
            "counts": new_counter(),
        })

    def resolve_in(tricode: str, sub_name: str) -> int:
        ids = (roster_teams.get(tricode) or {}).get(_norm(sub_name)) or []
        if len(ids) != 1:
            raise _PeriodDrop(f"ambiguous incoming '{sub_name}'")
        return ids[0]

    pending_boundary: Optional[float] = None

    for ev in events:
        atype = ev["action_type"]
        tricode = ev["team_tricode"]

        # A run of substitutions at one stoppage is a single boundary.
        if atype == "Substitution":
            clock = ev["clock_seconds"]
            if clock is None:
                raise _PeriodDrop("substitution without clock")
            if pending_boundary is None:
                close(clock)
                pending_boundary = clock
            m = _SUB_RE.match(ev["description"] or "")
            if not m or not ev["person_id"]:
                raise _PeriodDrop("unparseable substitution")
            out_id, in_id = ev["person_id"], resolve_in(tricode, m.group(1))
            if out_id not in on.get(tricode, set()) or in_id in on.get(tricode, set()):
                raise _PeriodDrop("impossible substitution sequence")
            on[tricode].remove(out_id)
            on[tricode].add(in_id)
            continue

        if pending_boundary is not None:
            reopen(pending_boundary)
            pending_boundary = None

        if ev["score_home"] is not None and ev["score_away"] is not None:
            score[0], score[1] = int(ev["score_home"]), int(ev["score_away"])

        c = stint_open["counts"]
        desc = ev["description"] or ""
        if atype in ("Made Shot", "Missed Shot") and tricode in c:
            c[tricode]["fga"] += 1
            last_miss_team = tricode if atype == "Missed Shot" else None
        elif atype == "Free Throw" and tricode in c:
            c[tricode]["fta"] += 1
            last_miss_team = tricode if desc.startswith("MISS") else None
        elif atype == "Turnover" and tricode in c:
            c[tricode]["to"] += 1
            last_miss_team = None
        elif atype == "Rebound" and tricode in c:
            if last_miss_team == tricode:
                c[tricode]["oreb"] += 1
            last_miss_team = None

    if pending_boundary is not None:
        reopen(pending_boundary)
    close(0.0)
    return stints, (score[0], score[1])


def _possessions(counts: Dict[str, Dict[str, int]], tricodes: Tuple[str, str]) -> float:
    """Dean Oliver estimate, averaged across the two teams' own counts."""
    per_team = []
    for t in tricodes:
        c = counts[t]
        per_team.append(c["fga"] + 0.44 * c["fta"] - c["oreb"] + c["to"])
    return max(0.0, (per_team[0] + per_team[1]) / 2.0)


def compute_team_onoff(conn, team_abbr: str, season: str,
                       season_type: str = "Regular Season",
                       exclude_garbage: bool = False) -> Dict[str, Any]:
    team_abbr = team_abbr.upper()
    row = conn.execute(
        "SELECT team_id FROM team_metadata WHERE abbreviation = ?", (team_abbr,)
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown team abbreviation: {team_abbr}")
    team_id = row[0]

    games = conn.execute(
        "SELECT game_id, home_team_id, away_team_id, traditional_json "
        "FROM box_scores WHERE season = ? AND season_type = ? "
        "AND (home_team_id = ? OR away_team_id = ?) ORDER BY game_date",
        (season, season_type, team_id, team_id),
    ).fetchall()

    excluded = {"games_no_pbp": 0, "games_no_roster": 0, "periods_dropped": 0, "periods_total": 0}
    garbage = {"stints": 0, "min": 0.0, "poss": 0.0}
    names: Dict[int, str] = {}
    # player_id -> per-game presence and on/off accumulators
    on_acc: Dict[int, Dict[str, float]] = {}
    games_with: Dict[int, Set[str]] = {}
    team_stints_by_game: Dict[str, List[Dict[str, Any]]] = {}
    lineup_acc: Dict[frozenset, Dict[str, float]] = {}
    games_processed = 0

    for g in games:
        game_id = g["game_id"]
        try:
            roster = _roster_from_traditional(g["traditional_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            excluded["games_no_roster"] += 1
            continue
        names.update(roster["names"])

        events = conn.execute(
            "SELECT action_number, period, clock_seconds, team_tricode, person_id, "
            "action_type, sub_type, description, score_home, score_away "
            "FROM pbp_events WHERE game_id = ? ORDER BY action_number",
            (game_id,),
        ).fetchall()
        if not events:
            excluded["games_no_pbp"] += 1
            continue

        # The two tricodes, home first (score_home belongs to it).
        tri_home = tri_away = None
        for side_id, attr in ((g["home_team_id"], "tri_home"), (g["away_team_id"], "tri_away")):
            r = conn.execute(
                "SELECT abbreviation FROM team_metadata WHERE team_id = ?", (side_id,)
            ).fetchone()
            if attr == "tri_home":
                tri_home = r[0] if r else None
            else:
                tri_away = r[0] if r else None
        if not tri_home or not tri_away:
            excluded["games_no_roster"] += 1
            continue
        tricodes = (tri_home, tri_away)
        we_are_home = tri_home == team_abbr

        by_period: Dict[int, List[Any]] = {}
        for ev in events:
            if ev["period"] is not None:
                by_period.setdefault(ev["period"], []).append(ev)

        score_carry = (0, 0)
        game_stints: List[Dict[str, Any]] = []
        for period in sorted(by_period):
            evs = [e for e in by_period[period] if e["action_type"] not in ("period",)]
            period_rows = [e for e in by_period[period] if e["action_type"] == "period"]
            start_clock = 720.0 if period <= 4 else 300.0
            for pr in period_rows:
                if pr["sub_type"] == "start" and pr["clock_seconds"] is not None:
                    start_clock = pr["clock_seconds"]
                if pr["sub_type"] == "start" and pr["score_home"] is not None:
                    score_carry = (int(pr["score_home"]), int(pr["score_away"]))
            excluded["periods_total"] += 1
            try:
                starters = _infer_period_starters(evs, roster["teams"], tricodes)
                stints, score_carry = _period_stints(
                    evs, starters, roster["teams"], tricodes, start_clock, score_carry,
                    period
                )
                game_stints.extend(stints)
            except _PeriodDrop:
                excluded["periods_dropped"] += 1
                # Recover the score for the next period from this period's end row.
                for pr in period_rows:
                    if pr["sub_type"] == "end" and pr["score_home"] is not None:
                        score_carry = (int(pr["score_home"]), int(pr["score_away"]))
                continue

        if not game_stints:
            continue

        # Garbage-time classification needs the final margin: the rule only
        # fires in games that END as blowouts. Read it off the last scored
        # event rather than trusting score_carry past dropped periods.
        final_margin = None
        for ev in reversed(events):
            if ev["score_home"] is not None and ev["score_away"] is not None:
                final_margin = abs(int(ev["score_home"]) - int(ev["score_away"]))
                break
        kept = []
        for st in game_stints:
            if _stint_is_garbage(st["period"], st["clock_start"], st["margin_start"], final_margin):
                garbage["stints"] += 1
                garbage["min"] += st["seconds"] / 60.0
                garbage["poss"] += _possessions(st["counts"], tricodes)
                if exclude_garbage:
                    continue
            kept.append(st)
        game_stints = kept
        if not game_stints:
            continue
        games_processed += 1
        team_stints_by_game[game_id] = game_stints

        for st in game_stints:
            ours = st["home_on"] if we_are_home else st["away_on"]
            theirs = st["away_on"] if we_are_home else st["home_on"]
            pts_for = st["home_pts"] if we_are_home else st["away_pts"]
            pts_against = st["away_pts"] if we_are_home else st["home_pts"]
            poss = _possessions(st["counts"], tricodes)
            st["_ours"] = ours
            st["_net"] = pts_for - pts_against
            st["_pts_for"] = pts_for
            st["_pts_against"] = pts_against
            st["_poss"] = poss

            for pid in ours:
                games_with.setdefault(pid, set()).add(game_id)
            key = frozenset(ours)
            if len(key) == 5:
                acc = lineup_acc.setdefault(key, {
                    "seconds": 0.0, "poss": 0.0, "pts_for": 0.0, "pts_against": 0.0, "games": set(),
                })
                acc["seconds"] += st["seconds"]
                acc["poss"] += poss
                acc["pts_for"] += pts_for
                acc["pts_against"] += pts_against
                acc["games"].add(game_id)

    # On/off per player: off-court counts only stints from games they appeared in.
    for pid, gids in games_with.items():
        acc = on_acc.setdefault(pid, {
            "sec_on": 0.0, "poss_on": 0.0, "net_on": 0.0,
            "sec_off": 0.0, "poss_off": 0.0, "net_off": 0.0, "gp": len(gids),
        })
        for gid in gids:
            for st in team_stints_by_game[gid]:
                if pid in st["_ours"]:
                    acc["sec_on"] += st["seconds"]
                    acc["poss_on"] += st["_poss"]
                    acc["net_on"] += st["_net"]
                else:
                    acc["sec_off"] += st["seconds"]
                    acc["poss_off"] += st["_poss"]
                    acc["net_off"] += st["_net"]

    def per100(net: float, poss: float) -> Optional[float]:
        return round(net / poss * 100.0, 2) if poss >= 1 else None

    # Stint-clustered standard error of the on/off diff (delta method on the
    # two ratio estimators, stints as the sampling unit). This is what lets a
    # consumer say "-4.1 ± 2.3" instead of pretending the point estimate is
    # exact. It understates true uncertainty a little (stints within a game
    # correlate), so it is a floor on the error bar, never a ceiling.
    var_acc: Dict[int, Dict[str, float]] = {}
    for pid, gids in games_with.items():
        a = on_acc[pid]
        r_on = a["net_on"] / a["poss_on"] if a["poss_on"] >= 1 else None
        r_off = a["net_off"] / a["poss_off"] if a["poss_off"] >= 1 else None
        v = var_acc.setdefault(pid, {"on": 0.0, "off": 0.0})
        for gid in gids:
            for st in team_stints_by_game[gid]:
                if pid in st["_ours"]:
                    if r_on is not None:
                        v["on"] += (st["_net"] - r_on * st["_poss"]) ** 2
                else:
                    if r_off is not None:
                        v["off"] += (st["_net"] - r_off * st["_poss"]) ** 2

    players_out = []
    for pid, a in on_acc.items():
        net_on = per100(a["net_on"], a["poss_on"])
        net_off = per100(a["net_off"], a["poss_off"])
        se_diff = None
        if net_on is not None and net_off is not None:
            v = var_acc.get(pid, {"on": 0.0, "off": 0.0})
            se_diff = round(100.0 * (
                v["on"] / (a["poss_on"] ** 2) + v["off"] / (a["poss_off"] ** 2)
            ) ** 0.5, 2)
        players_out.append({
            "player_id": pid,
            "name": names.get(pid, f"#{pid}"),
            "gp": a["gp"],
            "min_on": round(a["sec_on"] / 60.0, 1),
            "poss_on": round(a["poss_on"], 1),
            "net_on_per100": net_on,
            "min_off": round(a["sec_off"] / 60.0, 1),
            "poss_off": round(a["poss_off"], 1),
            "net_off_per100": net_off,
            "diff_per100": round(net_on - net_off, 2)
                if net_on is not None and net_off is not None else None,
            "se_diff_per100": se_diff,
        })
    players_out.sort(key=lambda p: (p["diff_per100"] is None, -(p["diff_per100"] or 0)))

    lineups_out = []
    for key, a in lineup_acc.items():
        poss = a["poss"]
        lineups_out.append({
            "players": sorted(names.get(pid, f"#{pid}") for pid in key),
            "games": len(a["games"]),
            "min": round(a["seconds"] / 60.0, 1),
            "poss": round(poss, 1),
            "pts_for": int(a["pts_for"]),
            "pts_against": int(a["pts_against"]),
            "net_per100": per100(a["pts_for"] - a["pts_against"], poss),
        })
    lineups_out.sort(key=lambda l: -l["min"])

    return {
        "team": team_abbr,
        "season": season,
        "season_type": season_type,
        "games_scheduled": len(games),
        "games_processed": games_processed,
        "excluded": excluded,
        "garbage_time": {
            "excluded": exclude_garbage,
            "stints": garbage["stints"],
            "min": round(garbage["min"], 1),
            "poss": round(garbage["poss"], 1),
            "definition": GARBAGE_DEFINITION,
        },
        "players": players_out,
        "lineups": lineups_out[:60],
        "method": (
            "Stints parsed from play-by-play substitutions; period-opening fives "
            "inferred from who acts before being subbed in. Periods that cannot be "
            "resolved to exactly five per team are dropped and counted above, never "
            "guessed. Possessions are the Dean Oliver estimate (FGA + 0.44*FTA - "
            "OREB + TO) per stint; off-court splits only count games the player "
            "appeared in."
        ),
    }
