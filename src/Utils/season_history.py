"""Season History: each season's playoff bracket, champion and headline facts.

Everything here is computed from the league game log (`game_results`), which
holds every playoff game back to 1946-47 with both teams' points, and from
the box-score archive's play-in games. Nothing is typed in: the champion is
whoever won four games in the last round of OUR game log, and the test suite
checks that against nba.com's own "NBA Champion" award rows.

Rounds are not in the game ids before 2001-02 (playoff ids were plain
sequence numbers then), so a series' round is the n-th distinct opponent a
team met that postseason. Both teams are asked and the later round kept, the
rule `_night_games` in main_api already uses.

Series length is read, not assumed: the first round was best-of-five through
2001-02 and best-of-seven from 2002-03, so a series is over when the winner
has four wins, or three in a first round before 2002-03. A season whose final
has not reached that is reported as unfinished, never with a guessed champion.

Team names are the ones the league game log recorded THAT season (1996-97
Seattle SuperSonics, Washington Bullets), not today's names: the standings
still show relocated franchises under modern names (documented debt), and
this page should not add to it. `team_id` is the franchise id, for links.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

ROUND_NAMES = {1: "First round", 2: "Conference semifinals", 3: "Conference finals", 4: "NBA Finals"}

#: The first season whose first round was best-of-seven.
FIRST_BEST_OF_SEVEN_R1 = "2002-03"

# Series per round in a 16-team bracket. A season whose rounds do not come out
# as 8/4/2/1 is flagged rather than drawn as if it were complete.
_SERIES_PER_ROUND = {1: 8, 2: 4, 3: 2, 4: 1}


def _wins_needed(season: str, rnd: int) -> int:
    return 3 if rnd == 1 and season < FIRST_BEST_OF_SEVEN_R1 else 4


def _playoff_rows(conn: sqlite3.Connection, season: Optional[str] = None) -> List[sqlite3.Row]:
    sql = (
        "SELECT game_id, team_id, season, DATE(game_date) AS date, team_abbr, team_name, "
        "matchup, wl, pts FROM game_results WHERE season_type = 'Playoffs'"
    )
    params: tuple = ()
    if season:
        sql += " AND season = ?"
        params = (season,)
    return conn.execute(sql + " ORDER BY season, DATE(game_date), game_id", params).fetchall()


def _series_from_rows(season: str, rows: List[Any]) -> Dict[str, Any]:
    """Build one season's bracket from its playoff game-log rows."""
    games: Dict[str, List[Any]] = {}
    names: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        games.setdefault(r["game_id"], []).append(r)
        names[r["team_id"]] = {"team_id": r["team_id"], "abbr": r["team_abbr"], "name": r["team_name"]}

    # Each team's opponents in the order it first met them.
    order: Dict[int, List[int]] = {}
    ordered = sorted(games.items(), key=lambda kv: (kv[1][0]["date"], kv[0]))
    pairs: Dict[frozenset, List[Dict[str, Any]]] = {}
    for gid, sides in ordered:
        if len(sides) != 2:
            # A game with one side on file cannot be scored; it is reported,
            # not silently dropped.
            continue
        a, b = sides
        for me, opp in ((a, b), (b, a)):
            seen = order.setdefault(me["team_id"], [])
            if opp["team_id"] not in seen:
                seen.append(opp["team_id"])
        home = a if "vs." in (a["matchup"] or "") else b if "vs." in (b["matchup"] or "") else None
        winner = a if a["wl"] == "W" else b if b["wl"] == "W" else None
        pairs.setdefault(frozenset((a["team_id"], b["team_id"])), []).append({
            "game_id": gid,
            "date": a["date"],
            "home_team_id": home["team_id"] if home else None,
            "winner_team_id": winner["team_id"] if winner else None,
            "score": {a["team_id"]: a["pts"], b["team_id"]: b["pts"]},
        })

    series = []
    for pair, gs in pairs.items():
        t1, t2 = sorted(pair)
        rnd = max(order[t1].index(t2) + 1, order[t2].index(t1) + 1)
        wins = {t1: 0, t2: 0}
        for g in gs:
            if g["winner_team_id"] in wins:
                wins[g["winner_team_id"]] += 1
        lead, trail = (t1, t2) if wins[t1] >= wins[t2] else (t2, t1)
        need = _wins_needed(season, rnd)
        complete = wins[lead] >= need
        series.append({
            "round": rnd,
            "round_name": ROUND_NAMES.get(rnd, f"Round {rnd}"),
            "best_of": need * 2 - 1,
            "complete": complete,
            # "winner"/"loser" only once the series is decided; until then the
            # two sides are "leader"/"trailer" and the page says in progress.
            "winner": {**names[lead], "wins": wins[lead]},
            "loser": {**names[trail], "wins": wins[trail]},
            "result": f"{wins[lead]}-{wins[trail]}",
            "games": [
                {
                    "game_id": g["game_id"], "date": g["date"],
                    "home_team_id": g["home_team_id"], "winner_team_id": g["winner_team_id"],
                    "winner_pts": g["score"].get(g["winner_team_id"]) if g["winner_team_id"] else None,
                    "loser_pts": next((v for k, v in g["score"].items() if k != g["winner_team_id"]), None)
                    if g["winner_team_id"] else None,
                }
                for g in sorted(gs, key=lambda g: (g["date"], g["game_id"]))
            ],
            "first_date": min(g["date"] for g in gs),
        })
    series.sort(key=lambda s: (s["round"], s["first_date"], s["winner"]["team_id"]))

    by_round: Dict[int, int] = {}
    for s in series:
        by_round[s["round"]] = by_round.get(s["round"], 0) + 1
    shape_ok = by_round == _SERIES_PER_ROUND
    final = next((s for s in series if s["round"] == 4), None)
    decided = bool(shape_ok and final and final["complete"] and all(s["complete"] for s in series))
    return {
        "season": season,
        "series": series,
        "series_per_round": {str(k): v for k, v in sorted(by_round.items())},
        "bracket_complete": decided,
        "champion": final["winner"] if decided else None,
        "runner_up": final["loser"] if decided else None,
        "finals_result": final["result"] if decided else None,
        "incomplete_games": sum(1 for sides in games.values() if len(sides) != 2),
        "playoff_games": len(games),
    }


def _link_abbrs(conn: sqlite3.Connection) -> Dict[int, str]:
    """Franchise id -> today's abbreviation, which is what team-page URLs use."""
    try:
        return {r[0]: r[1] for r in conn.execute("SELECT team_id, abbreviation FROM team_metadata")}
    except sqlite3.OperationalError:
        return {}


def playoff_bracket(conn: sqlite3.Connection, season: str) -> Dict[str, Any]:
    """One season's playoffs: every series by round, and the champion when decided.

    Each side carries `link_abbr` (today's abbreviation, for the team page's
    URL) beside the name and abbreviation of that season.
    """
    out = _series_from_rows(season, _playoff_rows(conn, season))
    links = _link_abbrs(conn)
    for s in out["series"]:
        for side in (s["winner"], s["loser"]):  # champion/runner_up are these same dicts
            side["link_abbr"] = links.get(side["team_id"])
    return out


def play_in_games(conn: sqlite3.Connection, season: str) -> List[Dict[str, Any]]:
    """The play-in games we hold for a season, with the winner from the box score.

    Play-in games are not in the league game log (`game_results`), only in the
    box-score archive, so the score is the sum of each side's player points.
    2019-20 had a single West game in the bubble; the tournament proper began
    in 2020-21.
    """
    out = []
    # The id range (play-in ids start 005) walks the primary key; filtering
    # on season alone scanned every box score, JSON blobs and all, per season.
    rows = conn.execute(
        """
        SELECT b.game_id, DATE(b.game_date) AS date, b.home_team_id, b.away_team_id
        FROM box_scores b
        WHERE b.game_id >= '005' AND b.game_id < '006'
          AND b.season = ? AND b.season_type = 'PlayIn'
        ORDER BY DATE(b.game_date), b.game_id
        """, (season,)).fetchall()
    for r in rows:
        pts = {t: p for t, p in conn.execute(
            "SELECT team_id, SUM(pts) FROM player_game_log WHERE game_id = ? GROUP BY team_id",
            (r["game_id"],))}
        abbr = {t: a for t, a in conn.execute(
            "SELECT team_id, abbreviation FROM team_metadata WHERE team_id IN (?, ?)",
            (r["home_team_id"], r["away_team_id"]))}
        h, a = pts.get(r["home_team_id"]), pts.get(r["away_team_id"])
        winner = None
        if h is not None and a is not None and h != a:
            winner = r["home_team_id"] if h > a else r["away_team_id"]
        out.append({
            "game_id": r["game_id"], "date": r["date"],
            "home": {"team_id": r["home_team_id"], "abbr": abbr.get(r["home_team_id"]), "pts": h},
            "away": {"team_id": r["away_team_id"], "abbr": abbr.get(r["away_team_id"]), "pts": a},
            "winner_team_id": winner,
        })
    return out


def regular_season_shape(conn: sqlite3.Connection, season: str) -> Dict[str, Any]:
    """Games per team and the best record, from the league game log.

    Games per team is what says "lockout" or "bubble" honestly: 1998-99 was 50
    for everyone, 2011-12 66, and 2019-20 anywhere from 63 to 75 because the
    season stopped in March and only 22 teams went to Orlando.
    """
    rows = conn.execute(
        """
        SELECT team_id, MAX(team_name) AS name, MAX(team_abbr) AS abbr,
               SUM(wl = 'W') AS w, SUM(wl = 'L') AS l, MAX(DATE(game_date)) AS last
        FROM game_results WHERE season = ? AND season_type = 'Regular Season'
        GROUP BY team_id
        """, (season,)).fetchall()
    if not rows:
        return {"teams": 0, "games_per_team_min": None, "games_per_team_max": None, "best_record": []}
    gp = [r["w"] + r["l"] for r in rows]
    best_pct = max(r["w"] / (r["w"] + r["l"]) for r in rows if r["w"] + r["l"])
    best = [
        {"team_id": r["team_id"], "abbr": r["abbr"], "name": r["name"], "wins": r["w"], "losses": r["l"]}
        for r in rows if r["w"] + r["l"] and abs(r["w"] / (r["w"] + r["l"]) - best_pct) < 1e-9
    ]
    best.sort(key=lambda t: t["name"] or "")
    links = _link_abbrs(conn)
    for t in best:
        t["link_abbr"] = links.get(t["team_id"])
    return {
        "teams": len(rows),
        "games_per_team_min": min(gp),
        "games_per_team_max": max(gp),
        "best_record": best,
        "_last_dates": {r["team_id"]: r["last"] for r in rows},
    }


def mvp(conn: sqlite3.Connection, season: str) -> Optional[Dict[str, Any]]:
    """The season's MVP from nba.com's award rows (player_awards), or None."""
    try:
        r = conn.execute(
            """
            SELECT a.player_id, p.full_name, a.team FROM player_awards a
            LEFT JOIN players p ON p.player_id = a.player_id
            WHERE a.description = 'NBA Most Valuable Player' AND a.season = ?
            ORDER BY a.player_id LIMIT 1
            """, (season,)).fetchone()
    except sqlite3.OperationalError:  # award table not built yet
        return None
    if not r:
        return None
    return {"player_id": r["player_id"], "name": r["full_name"], "team": r["team"]}


def season_notes(season: str, shape: Dict[str, Any], play_in: List[Dict[str, Any]]) -> List[str]:
    """Plain-English notes for the seasons that do not read like the others.

    The numbers in them come from `shape`; only the facts no table holds (where
    the bubble was) are words.
    """
    notes = []
    lo, hi = shape.get("games_per_team_min"), shape.get("games_per_team_max")
    if lo is None:
        return notes
    if season == "2019-20":
        went = sum(1 for d in (shape.get("_last_dates") or {}).values() if d and d >= "2020-07-01")
        notes.append(
            f"The bubble season. Play stopped in March 2020; {went} of {shape['teams']} teams "
            f"restarted at one site near Orlando with no fans, so teams finished with anywhere from "
            f"{lo} to {hi} games. Every playoff game was played there, so home court was only a label."
        )
        if play_in:
            notes.append("The West's eighth seed was settled by a one-off play-in game; the full "
                         "play-in tournament began the next season.")
    elif lo == hi and lo < 82:
        cause = {"1998-99": "the lockout", "2011-12": "the lockout", "2020-21": "the pandemic"}.get(season)
        notes.append(
            f"A {lo}-game regular season" + (f", shortened by {cause}." if cause else ".")
        )
    elif lo != hi:
        notes.append(f"Teams played between {lo} and {hi} regular-season games.")
    return notes


def season_summary(conn: sqlite3.Connection, season: str, bracket: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The headline facts for one season: champion, finals, best record, MVP."""
    bracket = bracket or playoff_bracket(conn, season)
    shape = regular_season_shape(conn, season)
    play_in = play_in_games(conn, season)
    out = {
        "season": season,
        "champion": bracket["champion"],
        "runner_up": bracket["runner_up"],
        "finals_result": bracket["finals_result"],
        "playoffs_complete": bracket["bracket_complete"],
        "best_record": shape["best_record"],
        "games_per_team": {"min": shape["games_per_team_min"], "max": shape["games_per_team_max"]},
        "teams": shape["teams"],
        "mvp": mvp(conn, season),
        "notes": season_notes(season, shape, play_in),
    }
    return out
