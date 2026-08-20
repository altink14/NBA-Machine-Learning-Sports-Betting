"""
ShotQuality.py
==============
Shot quality and shooting over expected: is he hot, or is he just getting
good looks?

THE MODEL. Every field-goal attempt is placed in one of 12 cells: shot class
(rim = inside 10 ft / non-rim two / three) x closest-defender distance at
release (0-2 Very Tight / 2-4 Tight / 4-6 Open / 6+ Wide Open). The expected
value of a shot is the LEAGUE'S points per shot in its cell that season -
nothing is fit, tuned, or extrapolated; the "model" is 12 league averages
with their sample sizes attached.

Per player, from the same 12-way split of his own attempts:
  xPPS   - expected points per shot of his diet alone. High xPPS means good
           looks (rim attempts, wide-open threes), independent of makes.
  PPS    - what he actually scored per shot (2*FG2M + 3*FG3M) / FGA.
  SOE    - shooting over expected, (PPS - xPPS) * 100: points added per 100
           shots by shot-MAKING once shot SELECTION is priced out.
Plus rim pressure: how often he gets to the rim and how contested it is
there (share of rim attempts with a defender inside 4 feet).

WHAT THIS CANNOT SEE. Free throws (and the fouls that create them), shot
clock, dribbles into the shot, or who the defender was - a "wide open" three
off a broken play and one created by gravity look identical. Defender
distance is measured at release, so late closeouts read more open than they
felt. Tracking data begins 2013-14.

DATA. nba.com tracking (LeagueDashPlayerPtShot), 8 league-wide fetches per
season-type: 4 defender buckets overall + the same 4 inside 10 ft. Non-rim
twos are the difference, clamped at zero; clamp events are counted in the
output, never hidden.
"""

from typing import Any, Dict, List, Optional

from src.Utils.nba_stats_client import get_client

DEF_BUCKETS = [
    ("0-2 Feet - Very Tight", "very_tight"),
    ("2-4 Feet - Tight", "tight"),
    ("4-6 Feet - Open", "open"),
    ("6+ Feet - Wide Open", "wide_open"),
]
SHOT_CLASSES = ["rim", "mid", "three"]  # mid = any two outside 10 ft
_VALUE = {"rim": 2, "mid": 2, "three": 3}


def _i(v: Any) -> int:
    return int(v) if v is not None else 0


def compute_shot_quality(season: str, season_type: str = "Regular Season") -> Dict[str, Any]:
    client = get_client()

    # attempts/makes per player per cell: acc[player_id][class][bucket] = [fgm, fga]
    acc: Dict[int, Dict[str, Dict[str, List[int]]]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    clamped = 0

    def ensure(pid: int, row: Dict) -> Dict[str, Dict[str, List[int]]]:
        if pid not in acc:
            acc[pid] = {c: {b: [0, 0] for _, b in DEF_BUCKETS} for c in SHOT_CLASSES}
            meta[pid] = {
                "player_id": pid,
                "name": row.get("PLAYER_NAME"),
                "team": row.get("PLAYER_LAST_TEAM_ABBREVIATION"),
                "gp": _i(row.get("GP")),
            }
        return acc[pid]

    for label, bucket in DEF_BUCKETS:
        overall = client.player_pt_shots(season, season_type, close_def_dist_range=label)
        rim = client.player_pt_shots(season, season_type, close_def_dist_range=label,
                                     general_range="Less Than 10 ft")
        rim_by_id = {r["PLAYER_ID"]: r for r in rim}
        for row in overall:
            pid = row["PLAYER_ID"]
            cells = ensure(pid, row)
            r = rim_by_id.get(pid) or {}
            rim_m, rim_a = _i(r.get("FGM")), _i(r.get("FGA"))
            mid_m = _i(row.get("FG2M")) - rim_m
            mid_a = _i(row.get("FG2A")) - rim_a
            if mid_m < 0 or mid_a < 0:
                clamped += 1
                mid_m, mid_a = max(0, mid_m), max(0, mid_a)
            cells["rim"][bucket] = [rim_m, rim_a]
            cells["mid"][bucket] = [mid_m, mid_a]
            cells["three"][bucket] = [_i(row.get("FG3M")), _i(row.get("FG3A"))]

    # League cells: points per shot with sample size, from the same rows.
    league: Dict[str, Dict[str, Any]] = {}
    for cls in SHOT_CLASSES:
        league[cls] = {}
        for _, bucket in DEF_BUCKETS:
            m = sum(acc[pid][cls][bucket][0] for pid in acc)
            a = sum(acc[pid][cls][bucket][1] for pid in acc)
            league[cls][bucket] = {
                "fga": a,
                "pps": round(_VALUE[cls] * m / a, 4) if a else None,
            }

    players: List[Dict[str, Any]] = []
    for pid, cells in acc.items():
        fga = sum(cells[c][b][1] for c in SHOT_CLASSES for _, b in DEF_BUCKETS)
        if fga == 0:
            continue
        pts = sum(_VALUE[c] * cells[c][b][0] for c in SHOT_CLASSES for _, b in DEF_BUCKETS)
        expected = 0.0
        for c in SHOT_CLASSES:
            for _, b in DEF_BUCKETS:
                pps = league[c][b]["pps"]
                if pps is not None:
                    expected += cells[c][b][1] * pps
        rim_a = sum(cells["rim"][b][1] for _, b in DEF_BUCKETS)
        rim_m = sum(cells["rim"][b][0] for _, b in DEF_BUCKETS)
        rim_contested = sum(cells["rim"][b][1] for b in ("very_tight", "tight"))
        wide_open_3a = cells["three"]["wide_open"][1]
        three_a = sum(cells["three"][b][1] for _, b in DEF_BUCKETS)
        players.append({
            **meta[pid],
            "fga": fga,
            "pps": round(pts / fga, 3),
            "xpps": round(expected / fga, 3),
            "soe_per100": round((pts - expected) / fga * 100.0, 1),
            "pts_over_expected": round(pts - expected, 1),
            "rim_share": round(rim_a / fga, 3),
            "rim_fg_pct": round(rim_m / rim_a, 3) if rim_a else None,
            "rim_contested_share": round(rim_contested / rim_a, 3) if rim_a else None,
            "wide_open_3_share": round(wide_open_3a / three_a, 3) if three_a else None,
        })
    players.sort(key=lambda p: -p["pts_over_expected"])

    return {
        "season": season,
        "season_type": season_type,
        "players": players,
        "league": league,
        "clamped_rows": clamped,
        "method": (
            "Every attempt sits in one of 12 cells: rim (inside 10 ft) / non-rim two / "
            "three, by closest-defender distance at release. Expected points per shot is "
            "the league average of the cell - 12 published averages, nothing fitted. xPPS "
            "prices the diet, SOE/100 is shot-making with the diet priced out. Free throws "
            "and fouls drawn are outside the model entirely, and defender distance at "
            "release reads late closeouts as more open than they felt. Tracking begins "
            "2013-14."
        ),
    }
