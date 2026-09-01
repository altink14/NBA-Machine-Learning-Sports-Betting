"""
DailyGame.py
============
"The Daily Box Score" — one real game a day, names redacted, guess the
matchup.

Design rules:
  - Determinism: the pool is FROZEN to completed seasons (season <
    CURRENT_SEASON), so a given date always maps to the same game no matter
    when it's asked for or how many new games the pipeline lands overnight.
  - Modern franchises only: games whose historical abbreviations aren't in
    today's 30 (SEA, VAN, NJN, ...) are excluded, because the player guesses
    from the modern team list.
  - Memorability floor: at least one 30-point scorer in the game.
  - The server never ships the answer with the puzzle; grading is a separate
    stateless call, and hints unlock by guess number.

Everything served here is a real archived box score — nothing invented.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

_pool_cache: Dict[str, List[str]] = {}


def _modern_teams(conn) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        "SELECT team_id, abbreviation, full_name, conference, division FROM team_metadata"
    ).fetchall()
    return {
        r[1]: {"team_id": r[0], "abbr": r[1], "name": r[2], "conference": r[3], "division": r[4]}
        for r in rows
    }


def _pool(conn, current_season: str) -> List[str]:
    """Eligible game_ids, ordered — frozen until the season constant bumps."""
    key = current_season
    if key in _pool_cache:
        return _pool_cache[key]
    modern = set(_modern_teams(conn).keys())
    rows = conn.execute(
        """
        SELECT b.game_id
        FROM box_scores b
        WHERE b.season >= '1996-97' AND b.season < ?
          AND b.game_id IN (SELECT DISTINCT game_id FROM player_game_log WHERE pts >= 30)
        ORDER BY b.game_id
        """,
        (current_season,),
    ).fetchall()
    # Keep only games where both historical abbreviations are modern ones.
    ok: List[str] = []
    for (gid,) in rows:
        abbrs = [r[0] for r in conn.execute(
            "SELECT team_abbr FROM game_results WHERE game_id = ?", (gid,)
        ).fetchall()]
        if len(abbrs) == 2 and all(a in modern for a in abbrs):
            ok.append(gid)
    _pool_cache[key] = ok
    return ok


def _pick(conn, date_key: str, current_season: str) -> str:
    pool = _pool(conn, current_season)
    if not pool:
        raise ValueError("Daily game pool is empty.")
    h = int(hashlib.md5(date_key.encode("utf-8")).hexdigest(), 16)
    return pool[h % len(pool)]


def _game_context(conn, game_id: str) -> Dict[str, Any]:
    """Both game_results rows + who was home (matchup 'X vs. Y' = home)."""
    rows = conn.execute(
        """
        SELECT team_id, team_abbr, team_name, matchup, wl, pts, game_date, season, season_type
        FROM game_results WHERE game_id = ?
        """,
        (game_id,),
    ).fetchall()
    if len(rows) != 2:
        raise ValueError("Game results incomplete for this game.")
    ctx: Dict[str, Any] = {"game_id": game_id}
    for team_id, abbr, name, matchup, wl, pts, game_date, season, season_type in rows:
        side = "home" if " vs. " in (matchup or "") else "away"
        ctx[side] = {"team_id": team_id, "abbr": abbr, "name": name, "wl": wl, "pts": pts}
        ctx["date"] = game_date
        ctx["season"] = season
        ctx["season_type"] = season_type
    if "home" not in ctx or "away" not in ctx:
        raise ValueError("Could not resolve home/away for this game.")
    return ctx


def _player_lines(conn, game_id: str, team_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT min, pts, reb, ast, stl, blk, fg3m
        FROM player_game_log
        WHERE game_id = ? AND team_id = ? AND min IS NOT NULL AND min != '' AND min != '0'
        ORDER BY pts DESC, reb DESC
        """,
        (game_id, team_id),
    ).fetchall()
    # The starter flag is unreliable before the tracking era (whole rosters
    # flagged as starters in 2000s rows), so the puzzle omits it entirely.
    out = []
    for m, pts, reb, ast, stl, blk, fg3m in rows:
        out.append({
            "min": m, "pts": pts, "reb": reb, "ast": ast,
            "stl": stl, "blk": blk, "fg3m": fg3m,
        })
    return out


def build_puzzle(conn, date_key: str, current_season: str) -> Dict[str, Any]:
    """The redacted board: stat lines and totals, no names, no cities."""
    gid = _pick(conn, date_key, current_season)
    ctx = _game_context(conn, gid)
    # 'A' is always the WINNER's column so the label leaks nothing positional.
    win_side = "home" if ctx["home"]["wl"] == "W" else "away"
    lose_side = "away" if win_side == "home" else "home"
    return {
        "date_key": date_key,
        "season": ctx["season"],
        "season_type": ctx["season_type"],
        "max_guesses": 6,
        "teams": {
            "A": {"total": ctx[win_side]["pts"],
                  "players": _player_lines(conn, gid, ctx[win_side]["team_id"])},
            "B": {"total": ctx[lose_side]["pts"],
                  "players": _player_lines(conn, gid, ctx[lose_side]["team_id"])},
        },
    }


def _top_scorer(conn, game_id: str) -> Optional[Tuple[str, int]]:
    row = conn.execute(
        """
        SELECT p.full_name, l.pts
        FROM player_game_log l JOIN players p ON p.player_id = l.player_id
        WHERE l.game_id = ?
        ORDER BY l.pts DESC LIMIT 1
        """,
        (game_id,),
    ).fetchone()
    return (row[0], row[1]) if row else None


def _initials(full_name: str) -> str:
    parts = [p for p in full_name.split() if p]
    return ".".join(p[0].upper() for p in parts[:2]) + "."


def _hint(conn, ctx: Dict[str, Any], guess_number: int, modern: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Ladder: month/year -> top-scorer initials -> home division -> home team -> away division."""
    home_meta = modern.get(ctx["home"]["abbr"], {})
    away_meta = modern.get(ctx["away"]["abbr"], {})
    if guess_number == 1:
        month = ctx["date"][:7]
        return f"The game was played in {month}."
    if guess_number == 2:
        top = _top_scorer(conn, ctx["game_id"])
        if top:
            return f"The game's top scorer: {_initials(top[0])} with {top[1]} points."
        return None
    if guess_number == 3:
        return f"The home team plays in the {home_meta.get('division', '?')} division."
    if guess_number == 4:
        return f"The home team is {ctx['home']['name']}."
    if guess_number == 5:
        return f"The road team plays in the {away_meta.get('division', '?')} division."
    return None


def grade_guess(
    conn,
    date_key: str,
    current_season: str,
    guess: List[str],
    guess_number: int,
) -> Dict[str, Any]:
    """
    Order-agnostic grading of a two-team guess.

    Verdict per guessed team: 'exact', 'division' (right division, wrong
    team), 'conference' (right conference), or 'miss'. Exact matches pair
    first; the leftover guessed team is judged against the leftover answer.
    """
    modern = _modern_teams(conn)
    picks = [str(g).upper().strip() for g in (guess or [])]
    if len(picks) != 2 or picks[0] == picks[1] or any(p not in modern for p in picks):
        raise ValueError("Guess must be two different modern team abbreviations.")
    if not (1 <= int(guess_number) <= 6):
        raise ValueError("guess_number must be 1-6.")

    gid = _pick(conn, date_key, current_season)
    ctx = _game_context(conn, gid)
    answers = [ctx["home"]["abbr"], ctx["away"]["abbr"]]

    verdicts: List[Dict[str, str]] = [{"abbr": p, "verdict": ""} for p in picks]
    remaining_answers = list(answers)
    # Pass 1: exact pairs.
    for v in verdicts:
        if v["abbr"] in remaining_answers:
            v["verdict"] = "exact"
            remaining_answers.remove(v["abbr"])
    # Pass 2: leftovers vs leftover answers (division beats conference).
    for v in verdicts:
        if v["verdict"]:
            continue
        gm = modern[v["abbr"]]
        best = "miss"
        for a in remaining_answers:
            am = modern[a]
            if gm["division"] == am["division"]:
                best = "division"
                break
            if gm["conference"] == am["conference"]:
                best = "conference"
        v["verdict"] = best

    correct = all(v["verdict"] == "exact" for v in verdicts)
    done = correct or int(guess_number) >= 6

    result: Dict[str, Any] = {"verdicts": verdicts, "correct": correct}
    if done:
        top = _top_scorer(conn, gid)
        result["reveal"] = {
            "game_id": gid,
            "date": ctx["date"],
            "season": ctx["season"],
            "season_type": ctx["season_type"],
            "home": {"abbr": ctx["home"]["abbr"], "name": ctx["home"]["name"], "pts": ctx["home"]["pts"]},
            "away": {"abbr": ctx["away"]["abbr"], "name": ctx["away"]["name"], "pts": ctx["away"]["pts"]},
            "top_scorer": {"name": top[0], "pts": top[1]} if top else None,
        }
    else:
        result["hint"] = _hint(conn, ctx, int(guess_number), modern)
    return result
