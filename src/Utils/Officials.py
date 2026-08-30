"""
Officials.py
============
What the game looks like when a given official is on it.

EVERY NUMBER HERE COMES FROM OUR OWN ARCHIVE. The crew for each game is read
from nba.com's Officials feed (see backfill_officials.py); the game itself —
points, fouls, free-throw rate, pace, who won — is computed from box scores we
already hold. Nothing is imported from a third-party referee site and nothing
is estimated. This matters because the previous version of the referee page
was deleted for attributing invented figures to real, named people.

THE BASELINE IS SEASON-MATCHED, which is the only way this comparison is
honest. Scoring and foul rates move a lot between seasons, and officials work
different eras, so an official's average is compared against the league
average of the SAME seasons, weighted by how many games they worked in each.
Comparing to a single all-time mean would turn "worked recently" into a
tendency.

WHAT THIS CANNOT SAY, and the page must repeat it: crews are not assigned at
random. Senior officials get nationally televised games, playoff series and
rivalry matchups, which differ in pace, stakes and foul rate before anyone
blows a whistle. A difference here is an association with the games an
official is given, not proof of how they call them. No causal or betting
claim is made or supported.
"""

import math
from typing import Any, Dict, List, Optional


def _wilson(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% interval for a proportion; small crews deserve visible error bars."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h) * 100, 1), round(min(1.0, c + h) * 100, 1)]


def _mean_ci(values: List[float]) -> Optional[List[float]]:
    """Normal-approx 95% interval on a mean, so spreads are readable."""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return [round(m - 1.96 * se, 2), round(m + 1.96 * se, 2)]


def compute_officials(
    conn,
    season_from: Optional[str] = None,
    min_games: int = 25,
    season_type: str = "Regular Season",
) -> Dict[str, Any]:
    # ---- per-game facts, for every game that has a crew on file ----
    params: List[Any] = [season_type]
    where = "b.season_type = ?"
    if season_from:
        where += " AND b.season >= ?"
        params.append(season_from)

    games = conn.execute(
        f"""
        SELECT b.game_id, b.season, b.game_date,
               t.pts AS home_pts, t.opp_pts AS away_pts,
               t.pace, t.ft_rate
        FROM box_scores b
        JOIN team_game_advanced t
          ON t.game_id = b.game_id AND t.team_id = b.home_team_id
        WHERE {where}
        """,
        params,
    ).fetchall()

    # Fouls live in the player log; one pass, summed per game.
    fouls: Dict[str, int] = {}
    for r in conn.execute(
        "SELECT game_id, SUM(pf) AS pf FROM player_game_log GROUP BY game_id"
    ):
        if r["pf"] is not None:
            fouls[r["game_id"]] = r["pf"]

    facts: Dict[str, Dict[str, Any]] = {}
    for g in games:
        total = (g["home_pts"] or 0) + (g["away_pts"] or 0)
        if total <= 0:
            continue
        facts[g["game_id"]] = {
            "season": g["season"],
            "total_pts": float(total),
            "pace": float(g["pace"]) if g["pace"] is not None else None,
            "ft_rate": float(g["ft_rate"]) if g["ft_rate"] is not None else None,
            "fouls": float(fouls[g["game_id"]]) if g["game_id"] in fouls else None,
            "home_win": 1 if (g["home_pts"] or 0) > (g["away_pts"] or 0) else 0,
        }

    # ---- league means per season, the yardstick each official is held to ----
    per_season: Dict[str, Dict[str, List[float]]] = {}
    for f in facts.values():
        s = per_season.setdefault(f["season"], {"total_pts": [], "pace": [], "ft_rate": [], "fouls": [], "home_win": []})
        s["total_pts"].append(f["total_pts"])
        s["home_win"].append(f["home_win"])
        for k in ("pace", "ft_rate", "fouls"):
            if f[k] is not None:
                s[k].append(f[k])
    season_mean = {
        s: {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
        for s, d in per_season.items()
    }

    # ---- crews ----
    links = conn.execute(
        "SELECT game_id, official_id FROM game_officials"
    ).fetchall()
    names = {
        r["official_id"]: {
            "first_name": r["first_name"],
            "last_name": r["last_name"],
            "jersey_num": r["jersey_num"],
        }
        for r in conn.execute("SELECT * FROM officials")
    }

    by_off: Dict[int, Dict[str, Any]] = {}
    for link in links:
        f = facts.get(link["game_id"])
        if not f:
            continue
        o = by_off.setdefault(link["official_id"], {
            "total_pts": [], "pace": [], "ft_rate": [], "fouls": [],
            "home_win": [], "seasons": {},
        })
        o["total_pts"].append(f["total_pts"])
        o["home_win"].append(f["home_win"])
        for k in ("pace", "ft_rate", "fouls"):
            if f[k] is not None:
                o[k].append(f[k])
        o["seasons"][f["season"]] = o["seasons"].get(f["season"], 0) + 1

    def matched_baseline(seasons: Dict[str, int], key: str) -> Optional[float]:
        """League mean over the same seasons, weighted by games worked there."""
        num = den = 0.0
        for s, n in seasons.items():
            m = season_mean.get(s, {}).get(key)
            if m is not None:
                num += m * n
                den += n
        return num / den if den else None

    out: List[Dict[str, Any]] = []
    for oid, o in by_off.items():
        n = len(o["total_pts"])
        if n < min_games:
            continue
        nm = names.get(oid, {})
        row: Dict[str, Any] = {
            "official_id": oid,
            "name": f"{nm.get('first_name', '')} {nm.get('last_name', '')}".strip() or f"#{oid}",
            "jersey": nm.get("jersey_num") or None,
            "games": n,
            "seasons": sorted(o["seasons"]),
        }
        for key, label in (("total_pts", "total_points"), ("pace", "pace"),
                           ("ft_rate", "ft_rate"), ("fouls", "fouls")):
            vals = o[key]
            if not vals:
                row[label] = None
                continue
            mean = sum(vals) / len(vals)
            base = matched_baseline(o["seasons"], key)
            row[label] = {
                "avg": round(mean, 3 if key == "ft_rate" else 1),
                "baseline": round(base, 3 if key == "ft_rate" else 1) if base is not None else None,
                "diff": round(mean - base, 3 if key == "ft_rate" else 1) if base is not None else None,
                "ci95": _mean_ci(vals),
                "n": len(vals),
            }
        wins = sum(o["home_win"])
        base_hw = matched_baseline(o["seasons"], "home_win")
        row["home_win"] = {
            "pct": round(wins / n * 100, 1),
            "baseline": round(base_hw * 100, 1) if base_hw is not None else None,
            "diff": round(wins / n * 100 - base_hw * 100, 1) if base_hw is not None else None,
            "ci95": _wilson(wins, n),
            "n": n,
        }
        out.append(row)

    out.sort(key=lambda r: -r["games"])

    # ---- coverage, stated plainly ----
    asked = conn.execute("SELECT COUNT(*) FROM officials_fetch").fetchone()[0]
    with_crew = conn.execute(
        "SELECT COUNT(*) FROM officials_fetch WHERE n_officials > 0"
    ).fetchone()[0]
    covered = conn.execute(
        "SELECT MIN(b.game_date), MAX(b.game_date) FROM game_officials g "
        "JOIN box_scores b ON b.game_id = g.game_id"
    ).fetchone()

    return {
        "season_from": season_from,
        "season_type": season_type,
        "min_games": min_games,
        "officials": out,
        "coverage": {
            "games_checked": asked,
            "games_with_crew": with_crew,
            "games_without_crew": asked - with_crew,
            "first_game": covered[0],
            "last_game": covered[1],
            "games_scored": len(facts),
        },
        "method": (
            "Crews come from nba.com's Officials feed; every game statistic is "
            "computed from our own box-score archive. Each official's average is "
            "compared with the league average of the SAME seasons, weighted by how "
            "many games they worked in each, because scoring and foul rates move "
            "between seasons. Intervals are 95%. Crews are NOT assigned at random - "
            "senior officials draw nationally televised, playoff and rivalry games, "
            "which differ before anyone blows a whistle - so these are associations "
            "with the games an official is given, not evidence about how they call "
            "them, and no betting claim is made from them."
        ),
    }
