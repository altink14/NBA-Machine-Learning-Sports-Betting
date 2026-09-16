"""
Market.py
=========
What the betting market said before an archived game, next to what happened.

THE LINES are the closing spread, total and moneylines from the historical
odds dataset this project has always trained on (Data/OddsData.sqlite,
tables odds_2007-08 .. odds_2022-23, one row per regular-season or playoff
game, 20,477 games). They are the market's final word before tip-off; we did
not set them and we do not adjust them. Nothing here is a Betting Buddy
prediction and nothing here says the model would have beaten these lines -
profitability has never been measured, and this module does not measure it.

THE RESULTS come from OUR OWN archive (box_scores / team_game_advanced), not
from the odds dataset's own Points / Win_Margin columns, which are a
third-party transcription. We report whether the two agree so the reader can
see when they do not.

THE JOIN is by game date + home team. The odds dataset spells three defunct
franchises the old way; box_scores carries modern ids, so those are aliased.
A handful of rows sit one calendar day off (late West Coast tips recorded on
the US date); we accept a +/- 1 day match only when the exact date has no
game for that home team. Play-in games are missing from the dataset, so
about 25 games in 2019-20 .. 2022-23 have no line. Coverage: 20,452 of
20,477 rows land on an archived game (99.9%).

Sign conventions in the dataset: Spread is the HOME side's expected margin
(+3.5 = home favoured by 3.5, -2.5 = home a 2.5-point underdog). Win_Margin is
home minus away. OU is the total. Moneylines are American.
"""

import datetime as dt
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

FIRST_SEASON = "2007-08"
LAST_SEASON = "2022-23"

# Odds-dataset spelling -> the modern franchise name used by team_metadata.
_ALIASES = {
    "New Jersey Nets": "Brooklyn Nets",
    "Charlotte Bobcats": "Charlotte Hornets",
    "Seattle SuperSonics": "Oklahoma City Thunder",
}

_lock = threading.Lock()
_lines: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None  # (date, home_full_name) -> row


def _to_iso(date_key: str) -> str:
    """'2015-16-1027' -> '2015-10-27'. Months 10-12 belong to the first year."""
    season, mmdd = date_key.rsplit("-", 1)
    y1 = int(season[:4])
    month = int(mmdd[:2])
    year = y1 if month >= 10 else y1 + 1
    return f"{year:04d}-{month:02d}-{mmdd[2:]}"


def _implied(american: Optional[float]) -> Optional[float]:
    if american is None:
        return None
    a = float(american)
    if a == 0:
        return None
    return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)


def _num(v: Any) -> Optional[float]:
    """The dataset stores some seasons' numbers as text ('-1400', '1'); coerce."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load(odds_conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict[str, Any]]:
    global _lines
    with _lock:
        if _lines is not None:
            return _lines
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        tables = [
            r[0]
            for r in odds_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'odds_2%' AND name NOT LIKE '%_new' ORDER BY name"
            )
        ]
        for tb in tables:
            season = tb[len("odds_"):]
            cur = odds_conn.execute(
                f'SELECT Date, Home, Away, OU, Spread, ML_Home, ML_Away, Points, Win_Margin, '
                f'Days_Rest_Home, Days_Rest_Away FROM "{tb}"'
            )
            for d, home, away, ou, spread, mlh, mla, pts, margin, rest_h, rest_a in cur:
                if not d or not home:
                    continue
                key = (_to_iso(d), _ALIASES.get(home, home))
                out[key] = {
                    "season": season,
                    "home": _ALIASES.get(home, home),
                    "away": _ALIASES.get(away, away),
                    "total": _num(ou),
                    "spread_home": _num(spread),
                    "ml_home": _num(mlh),
                    "ml_away": _num(mla),
                    "dataset_points": _num(pts),
                    "dataset_margin": _num(margin),
                    "rest_home": _num(rest_h),
                    "rest_away": _num(rest_a),
                }
        _lines = out
        return out


def _lookup(lines, game_date: str, home_name: str) -> Optional[Dict[str, Any]]:
    row = lines.get((game_date, home_name))
    if row is not None:
        return row
    try:
        d = dt.date.fromisoformat(game_date)
    except ValueError:
        return None
    for delta in (1, -1):
        row = lines.get(((d + dt.timedelta(days=delta)).isoformat(), home_name))
        if row is not None:
            return row
    return None


def _grade(row: Dict[str, Any], home_pts: Optional[int], away_pts: Optional[int]) -> Dict[str, Any]:
    """Line vs what our archive says happened."""
    spread = row["spread_home"]
    total = row["total"]
    mlh, mla = row["ml_home"], row["ml_away"]

    margin = None if home_pts is None or away_pts is None else home_pts - away_pts
    points = None if home_pts is None or away_pts is None else home_pts + away_pts

    ph, pa = _implied(mlh), _implied(mla)
    fair_home = None
    if ph is not None and pa is not None and (ph + pa) > 0:
        fair_home = ph / (ph + pa)

    favorite = None
    if mlh is not None and mla is not None:
        favorite = "home" if mlh < mla else ("away" if mla < mlh else "pick")
    elif spread is not None:
        favorite = "home" if spread > 0 else ("away" if spread < 0 else "pick")

    spread_result = None
    cover_margin = None
    if spread is not None and margin is not None:
        cover_margin = round(margin - float(spread), 1)
        spread_result = "push" if cover_margin == 0 else ("home" if cover_margin > 0 else "away")

    total_result = None
    total_margin = None
    if total is not None and points is not None:
        total_margin = round(points - float(total), 1)
        total_result = "push" if total_margin == 0 else ("over" if total_margin > 0 else "under")

    winner = None if margin is None or margin == 0 else ("home" if margin > 0 else "away")
    ml_result = None
    if winner and favorite in ("home", "away"):
        ml_result = "favorite" if winner == favorite else "underdog"

    agrees = None
    if margin is not None and row["dataset_margin"] is not None and points is not None and row["dataset_points"] is not None:
        agrees = int(row["dataset_margin"]) == margin and int(row["dataset_points"]) == points

    return {
        "closing": {
            "spread_home": spread,
            "total": total,
            "ml_home": mlh,
            "ml_away": mla,
            "favorite": favorite,
            "implied_home": None if ph is None else round(ph * 100, 1),
            "implied_away": None if pa is None else round(pa * 100, 1),
            "fair_home": None if fair_home is None else round(fair_home * 100, 1),
            "vig_pct": None if ph is None or pa is None else round((ph + pa - 1) * 100, 1),
            "rest_home": row["rest_home"],
            "rest_away": row["rest_away"],
        },
        "result": {
            "home_pts": home_pts,
            "away_pts": away_pts,
            "margin_home": margin,
            "points": points,
            "winner": winner,
            "spread_result": spread_result,
            "cover_margin_home": cover_margin,
            "total_result": total_result,
            "total_margin": total_margin,
            "ml_result": ml_result,
            "dataset_agrees": agrees,
        },
    }


def game_market(team_conn: sqlite3.Connection, odds_conn: sqlite3.Connection, game_id: str) -> Dict[str, Any]:
    """Closing line + graded result for one archived game, or available=False."""
    row = team_conn.execute(
        """
        SELECT b.game_id, b.season, b.season_type, b.game_date, m.full_name AS home_name,
               t.pts AS home_pts, t.opp_pts AS away_pts
        FROM box_scores b
        JOIN team_metadata m ON m.team_id = b.home_team_id
        LEFT JOIN team_game_advanced t ON t.game_id = b.game_id AND t.team_id = b.home_team_id
        WHERE b.game_id = ?
        """,
        (game_id,),
    ).fetchone()
    if row is None:
        return {"game_id": game_id, "available": False, "reason": "not_archived"}
    game_id, season, season_type, game_date, home_name, home_pts, away_pts = row
    base = {
        "game_id": game_id,
        "season": season,
        "season_type": season_type,
        "coverage": {"first_season": FIRST_SEASON, "last_season": LAST_SEASON},
    }
    if not (FIRST_SEASON <= season <= LAST_SEASON):
        return {**base, "available": False, "reason": "season_not_covered"}
    line = _lookup(_load(odds_conn), game_date, home_name)
    if line is None:
        return {**base, "available": False, "reason": "no_line_on_file"}
    graded = _grade(line, home_pts, away_pts)
    return {**base, "available": True, **graded}


def season_market(team_conn: sqlite3.Connection, odds_conn: sqlite3.Connection, season: str,
                  season_type: str = "Regular Season") -> Dict[str, Any]:
    """Every team's record against the closing market for one season.

    ATS = against the spread (wins-losses-pushes), O/U = the team's games over
    or under the total (a team-neutral count), SU = straight up. 'As favourite'
    and 'as underdog' split the SU record by the moneyline favourite. Every
    figure is a count of graded games; nothing is a projection.
    """
    if not (FIRST_SEASON <= season <= LAST_SEASON):
        return {"season": season, "season_type": season_type, "available": False,
                "coverage": {"first_season": FIRST_SEASON, "last_season": LAST_SEASON}, "teams": []}
    lines = _load(odds_conn)
    rows = team_conn.execute(
        """
        SELECT b.game_id, b.game_date, b.home_team_id, b.away_team_id,
               hm.full_name AS home_name, hm.abbreviation AS home_abbr,
               am.full_name AS away_name, am.abbreviation AS away_abbr,
               t.pts AS home_pts, t.opp_pts AS away_pts
        FROM box_scores b
        JOIN team_metadata hm ON hm.team_id = b.home_team_id
        JOIN team_metadata am ON am.team_id = b.away_team_id
        LEFT JOIN team_game_advanced t ON t.game_id = b.game_id AND t.team_id = b.home_team_id
        WHERE b.season = ? AND b.season_type = ?
        ORDER BY b.game_date
        """,
        (season, season_type),
    ).fetchall()

    def blank(name: str, abbr: str) -> Dict[str, Any]:
        return {
            "team": name, "abbr": abbr, "games": 0, "graded": 0,
            "su": [0, 0], "ats": [0, 0, 0], "ou": [0, 0, 0],
            "as_favorite": [0, 0], "as_underdog": [0, 0],
            "spread_sum": 0.0, "cover_sum": 0.0, "total_sum": 0.0, "points_sum": 0.0,
            "biggest_upset": None,  # largest fair-prob deficit that still won
        }

    teams: Dict[str, Dict[str, Any]] = {}
    league = {"games": 0, "graded": 0, "home_ats": [0, 0, 0], "over": [0, 0, 0], "favorite_su": [0, 0], "home_su": [0, 0]}

    for (gid, gdate, _hid, _aid, hname, habbr, aname, aabbr, hpts, apts) in rows:
        th = teams.setdefault(habbr, blank(hname, habbr))
        ta = teams.setdefault(aabbr, blank(aname, aabbr))
        th["games"] += 1
        ta["games"] += 1
        league["games"] += 1
        line = _lookup(lines, gdate, hname)
        if line is None or hpts is None or apts is None:
            continue
        g = _grade(line, hpts, apts)
        c, r = g["closing"], g["result"]
        if r["winner"] is None:
            continue
        league["graded"] += 1
        th["graded"] += 1
        ta["graded"] += 1

        # straight up
        hw = r["winner"] == "home"
        th["su"][0 if hw else 1] += 1
        ta["su"][1 if hw else 0] += 1
        league["home_su"][0 if hw else 1] += 1

        # against the spread
        if r["spread_result"] == "push":
            th["ats"][2] += 1; ta["ats"][2] += 1; league["home_ats"][2] += 1
        elif r["spread_result"] == "home":
            th["ats"][0] += 1; ta["ats"][1] += 1; league["home_ats"][0] += 1
        elif r["spread_result"] == "away":
            th["ats"][1] += 1; ta["ats"][0] += 1; league["home_ats"][1] += 1
        if c["spread_home"] is not None:
            th["spread_sum"] += -float(c["spread_home"])   # the team's own line (negative = favoured)
            ta["spread_sum"] += float(c["spread_home"])
        if r["cover_margin_home"] is not None:
            th["cover_sum"] += r["cover_margin_home"]
            ta["cover_sum"] -= r["cover_margin_home"]

        # totals
        idx = {"over": 0, "under": 1, "push": 2}.get(r["total_result"])
        if idx is not None:
            th["ou"][idx] += 1; ta["ou"][idx] += 1; league["over"][idx] += 1
        if c["total"] is not None:
            th["total_sum"] += float(c["total"]); ta["total_sum"] += float(c["total"])
            th["points_sum"] += r["points"]; ta["points_sum"] += r["points"]

        # favourite / underdog
        if c["favorite"] in ("home", "away"):
            fav, dog = (th, ta) if c["favorite"] == "home" else (ta, th)
            fav_won = r["ml_result"] == "favorite"
            fav["as_favorite"][0 if fav_won else 1] += 1
            dog["as_underdog"][0 if not fav_won else 1] += 1
            league["favorite_su"][0 if fav_won else 1] += 1
            if not fav_won and c["fair_home"] is not None:
                dog_prob = (100 - c["fair_home"]) if c["favorite"] == "home" else c["fair_home"]
                prev = dog["biggest_upset"]
                if prev is None or dog_prob < prev["fair_prob"]:
                    dog["biggest_upset"] = {
                        "game_id": gid, "date": gdate, "opponent": fav["abbr"],
                        "fair_prob": round(dog_prob, 1), "moneyline": c["ml_home"] if c["favorite"] == "away" else c["ml_away"],
                        "score": f"{r['away_pts']}-{r['home_pts']}",
                    }

    out_teams: List[Dict[str, Any]] = []
    for t in teams.values():
        n = t["graded"]
        ats_dec = t["ats"][0] + t["ats"][1]
        out_teams.append({
            "team": t["team"], "abbr": t["abbr"], "games": t["games"], "graded": n,
            "su": t["su"], "ats": t["ats"], "ou": t["ou"],
            "as_favorite": t["as_favorite"], "as_underdog": t["as_underdog"],
            "ats_pct": round(100 * t["ats"][0] / ats_dec, 1) if ats_dec else None,
            "over_pct": round(100 * t["ou"][0] / (t["ou"][0] + t["ou"][1]), 1) if (t["ou"][0] + t["ou"][1]) else None,
            "avg_line": round(t["spread_sum"] / n, 1) if n else None,
            "avg_cover_margin": round(t["cover_sum"] / n, 2) if n else None,
            "avg_total": round(t["total_sum"] / n, 1) if n else None,
            "avg_points": round(t["points_sum"] / n, 1) if n else None,
            "biggest_upset": t["biggest_upset"],
        })
    out_teams.sort(key=lambda x: (-(x["ats_pct"] or 0), x["team"]))

    return {
        "season": season, "season_type": season_type, "available": True,
        "coverage": {"first_season": FIRST_SEASON, "last_season": LAST_SEASON},
        "league": league,
        "teams": out_teams,
    }
