"""
market.py (NFL)
===============
What the closing market said before a game, next to what happened.

THE LINES are the closing spread, total and moneyline that nflverse ships in
its schedules release (CC-BY-4.0), stored as book = 'consensus'. nflverse does
not say which book quoted them or when, and neither do we: every surface built
on this must say "closing line (consensus, book unspecified)". Spreads and
totals run from 1999, moneylines only from 2006, so a season before 2006 has
against-the-spread records and no favourite/underdog split. That is a fact
about the source, not a bug, and the payload reports it rather than filling
the gap with silence.

THE RESULTS are the final scores in our own `games` table, which come from the
same release. Unlike the NBA equivalent there is no independent transcription
to cross-check against, so this module does not claim one.

NOTHING HERE IS A PREDICTION. These are counts of what already happened
against a line we did not set. No figure on this page says our model would
have beaten these lines; profitability has never been measured for either
sport and this module does not measure it.

NOT SERVED TO ANYONE YET. The owner decided on 2026-09-21 that NFL ships as
a PAID feature, so whatever surface consumes this belongs under the
auth-gated `(app)` route group, not the public `(reference)` one that every
NBA stats page lives in. A half-built public route was removed rather than
finished in the wrong place. These endpoints exist and are tested; nothing
links to them. NBA is the focus until opening night.

TIES ARE REAL IN THIS SPORT, unlike basketball, so straight-up records are
win-loss-tie and a tie is never quietly folded into a loss.

WHY 2026 IS MISSING. The 2026 season is the sealed holdout for the NFL model
(pre-registration v2, seal commit a88f762). Publishing a season-long scoreboard
for it would put those outcomes in front of anyone working on the model,
including the people who wrote the seal, which is exactly how the first
pre-registration was voided. It appears here after the single evaluation. The
API says so out loud rather than letting the season list look like an
oversight.

Sign convention: `spread` is the HOME side's expected margin, positive when
the home team is favoured (verified against 1999_01_OAK_GB in ingest_games).
So the home team covers when its actual margin beats that number.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

#: Spreads and totals start here; the archive itself starts here.
FIRST_SEASON = 1999

#: Moneylines are absent before this, so favourite/underdog splits are too.
MONEYLINE_FIRST_SEASON = 2006

#: The NFL model's sealed evaluation window. Not served, and the reason is
#: returned with the refusal. See the module docstring.
SEALED_SEASONS = frozenset({2026})

#: nflverse season types, grouped the way a reader thinks about them.
SEASON_TYPE_GROUPS = {
    "REG": ("REG",),
    "POST": ("WC", "DIV", "CON", "SB"),
}


def available_seasons(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Which seasons this page can serve, and what is deliberately withheld."""
    rows = conn.execute(
        "SELECT DISTINCT CAST(season AS INTEGER) AS s FROM games "
        "WHERE status = 'final' ORDER BY s DESC").fetchall()
    all_seasons = [r[0] for r in rows]
    served = [s for s in all_seasons if s not in SEALED_SEASONS]
    return {
        "seasons": served,
        "moneyline_from": MONEYLINE_FIRST_SEASON,
        "withheld": sorted(s for s in all_seasons if s in SEALED_SEASONS),
        "withheld_reason": (
            "The 2026 season is the sealed holdout for our NFL model. Its results are "
            "in the archive but are not published here until the single pre-registered "
            "evaluation runs, so that nobody working on the model can see them first."
        ),
        "source": "nflverse-data schedules (CC-BY-4.0)",
    }


def _blank(abbr: str) -> Dict[str, Any]:
    return {
        "abbr": abbr,
        "games": 0,            # games played
        "graded": 0,           # games that had a closing spread
        "su": [0, 0, 0],       # win, loss, tie
        "ats": [0, 0, 0],      # cover, fail, push
        "ou": [0, 0, 0],       # over, under, push
        "as_favorite": [0, 0, 0],
        "as_underdog": [0, 0, 0],
        "spread_sum": 0.0,     # mean closing spread faced, from this team's side
        "cover_sum": 0.0,      # mean margin against the spread
        "points_for": 0,
        "points_against": 0,
    }


def _record(bucket: List[int], margin: int) -> None:
    bucket[0 if margin > 0 else 1 if margin < 0 else 2] += 1


def season_market(conn: sqlite3.Connection, season: int,
                  season_type: str = "REG") -> Dict[str, Any]:
    """Every team's record against the closing market for one season.

    ATS is against the spread (cover-fail-push). O/U counts the team's games
    that went over or under the closing total, which is a team-neutral count:
    both sides of a game score the same way. SU is straight up, with ties.
    'As favourite' and 'as underdog' split the straight-up record by who the
    moneyline made the favourite, and are empty before 2006.

    Every figure is a count of games that have been played. Nothing here is
    a projection and nothing here is a claim about our model.
    """
    if season in SEALED_SEASONS:
        return {
            "season": season, "season_type": season_type, "available": False,
            "reason": available_seasons(conn)["withheld_reason"],
            "teams": [], "league": None,
        }
    types = SEASON_TYPE_GROUPS.get(season_type)
    if not types:
        return {"season": season, "season_type": season_type, "available": False,
                "reason": "season_type must be REG or POST", "teams": [], "league": None}

    placeholders = ",".join("?" * len(types))
    rows = conn.execute(
        f"""
        SELECT g.game_id, g.week, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               sp.line  AS spread,
               tt.line  AS total,
               ml.price_home, ml.price_away
        FROM games g
        LEFT JOIN market_lines sp ON sp.game_id = g.game_id
             AND sp.market_type = 'spread'    AND sp.is_closing = 1
        LEFT JOIN market_lines tt ON tt.game_id = g.game_id
             AND tt.market_type = 'total'     AND tt.is_closing = 1
        LEFT JOIN market_lines ml ON ml.game_id = g.game_id
             AND ml.market_type = 'moneyline' AND ml.is_closing = 1
        WHERE CAST(g.season AS INTEGER) = ?
          AND g.season_type IN ({placeholders})
          AND g.status = 'final'
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY g.week
        """,
        (season, *types),
    ).fetchall()

    teams: Dict[str, Dict[str, Any]] = {}
    league = {
        "games": 0, "graded": 0,
        "home_ats": [0, 0, 0], "ou": [0, 0, 0],
        "home_su": [0, 0, 0], "favorite_su": [0, 0, 0],
        "moneyline_games": 0,
    }

    for r in rows:
        home = (r["home_team_id"] or "").replace("nfl-", "")
        away = (r["away_team_id"] or "").replace("nfl-", "")
        if not home or not away:
            continue
        h = teams.setdefault(home, _blank(home))
        a = teams.setdefault(away, _blank(away))
        hs, as_ = r["home_score"], r["away_score"]
        margin = hs - as_          # home minus away

        h["games"] += 1
        a["games"] += 1
        h["points_for"] += hs
        h["points_against"] += as_
        a["points_for"] += as_
        a["points_against"] += hs
        _record(h["su"], margin)
        _record(a["su"], -margin)
        league["games"] += 1
        _record(league["home_su"], margin)

        spread = r["spread"]
        if spread is not None:
            # The home team covers when its margin beats the number it gave.
            # cover_margin is signed from the home side; the away side is the
            # exact negative, which is what makes ATS a zero-sum ledger.
            cover = margin - spread
            h["graded"] += 1
            a["graded"] += 1
            league["graded"] += 1
            _record(h["ats"], cover)
            _record(a["ats"], -cover)
            _record(league["home_ats"], cover)
            h["spread_sum"] += spread
            a["spread_sum"] += -spread
            h["cover_sum"] += cover
            a["cover_sum"] += -cover

        total = r["total"]
        if total is not None:
            over = (hs + as_) - total
            _record(h["ou"], over)
            _record(a["ou"], over)      # team-neutral: both sides went over
            _record(league["ou"], over)

        ph, pa = r["price_home"], r["price_away"]
        if ph is not None and pa is not None and ph != pa:
            league["moneyline_games"] += 1
            home_favored = ph < pa
            fav, dog = (h, a) if home_favored else (a, h)
            fav_margin = margin if home_favored else -margin
            _record(fav["as_favorite"], fav_margin)
            _record(dog["as_underdog"], -fav_margin)
            _record(league["favorite_su"], fav_margin)

    out: List[Dict[str, Any]] = []
    for t in teams.values():
        graded = t["graded"] or 1
        t["avg_spread"] = round(t["spread_sum"] / graded, 2) if t["graded"] else None
        t["avg_cover"] = round(t["cover_sum"] / graded, 2) if t["graded"] else None
        t["point_diff"] = t["points_for"] - t["points_against"]
        del t["spread_sum"], t["cover_sum"]
        out.append(t)
    out.sort(key=lambda t: (-t["su"][0], t["su"][1], t["abbr"]))

    return {
        "season": season,
        "season_type": season_type,
        "available": bool(out),
        "teams": out,
        "league": league,
        "has_moneylines": league["moneyline_games"] > 0,
        "moneyline_from": MONEYLINE_FIRST_SEASON,
        "source": "nflverse-data schedules (CC-BY-4.0)",
        "line_note": ("Closing spread, total and moneyline as published by nflverse: "
                      "consensus, book unspecified, capture time unknown."),
    }


def line_splits(conn: sqlite3.Connection, season_from: Optional[int] = None,
                season_to: Optional[int] = None,
                season_type: str = "REG") -> Dict[str, Any]:
    """How favourites of each size actually did, across a range of seasons.

    The question a spread page exists to answer: when the market makes a team
    a 3-point favourite, how often does that team cover? Buckets are by the
    absolute closing spread, which is the number a bettor sees.

    Key numbers get their own rows because the NFL's scoring grid makes them
    special: 3 is the most common margin of victory in the sport and 7 the
    second, so a line sitting exactly there behaves unlike its neighbours.
    """
    types = SEASON_TYPE_GROUPS.get(season_type)
    if not types:
        return {"available": False, "reason": "season_type must be REG or POST"}
    lo = max(FIRST_SEASON, season_from or FIRST_SEASON)
    hi = season_to or max(
        (s for s in [r[0] for r in conn.execute(
            "SELECT DISTINCT CAST(season AS INTEGER) FROM games WHERE status='final'")]
         if s not in SEALED_SEASONS), default=FIRST_SEASON)
    # Never let a range quietly include the sealed season.
    served = [s for s in range(lo, hi + 1) if s not in SEALED_SEASONS]
    if not served:
        return {"available": False, "reason": "no unsealed season in that range"}

    placeholders = ",".join("?" * len(types))
    rows = conn.execute(
        f"""
        SELECT g.home_score, g.away_score, sp.line AS spread, tt.line AS total
        FROM games g
        JOIN market_lines sp ON sp.game_id = g.game_id
             AND sp.market_type = 'spread' AND sp.is_closing = 1
        LEFT JOIN market_lines tt ON tt.game_id = g.game_id
             AND tt.market_type = 'total'  AND tt.is_closing = 1
        WHERE CAST(g.season AS INTEGER) BETWEEN ? AND ?
          AND CAST(g.season AS INTEGER) NOT IN ({','.join('?' * len(SEALED_SEASONS))})
          AND g.season_type IN ({placeholders})
          AND g.status = 'final'
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        """,
        (served[0], served[-1], *sorted(SEALED_SEASONS), *types),
    ).fetchall()

    def bucket_of(spread: float) -> str:
        a = abs(spread)
        if a == 0:
            return "pick'em"
        if a < 3:
            return "under 3"
        if a == 3:
            return "exactly 3"
        if a < 7:
            return "3.5 to 6.5"
        if a == 7:
            return "exactly 7"
        if a <= 10:
            return "7.5 to 10"
        if a <= 14:
            return "10.5 to 14"
        return "more than 14"

    ORDER = ["pick'em", "under 3", "exactly 3", "3.5 to 6.5", "exactly 7",
             "7.5 to 10", "10.5 to 14", "more than 14"]
    buckets: Dict[str, Dict[str, Any]] = {
        b: {"bucket": b, "games": 0, "fav_ats": [0, 0, 0], "fav_su": [0, 0, 0],
            "ou": [0, 0, 0]} for b in ORDER}
    home_field = {"games": 0, "home_ats": [0, 0, 0], "home_su": [0, 0, 0]}

    for r in rows:
        spread, margin = r["spread"], r["home_score"] - r["away_score"]
        b = buckets[bucket_of(spread)]
        b["games"] += 1
        home_field["games"] += 1
        _record(home_field["home_ats"], margin - spread)
        _record(home_field["home_su"], margin)
        # From the favourite's side. A pick'em has no favourite, so it is
        # scored from the home side and labelled as such.
        home_is_fav = spread > 0
        fav_cover = (margin - spread) if home_is_fav else (spread - margin)
        fav_margin = margin if home_is_fav else -margin
        _record(b["fav_ats"], fav_cover)
        _record(b["fav_su"], fav_margin)
        if r["total"] is not None:
            _record(b["ou"], (r["home_score"] + r["away_score"]) - r["total"])

    return {
        "available": True,
        "season_from": served[0], "season_to": served[-1],
        "season_type": season_type,
        "seasons_counted": len(served),
        "buckets": [buckets[b] for b in ORDER if buckets[b]["games"]],
        "home_field": home_field,
        "sealed_excluded": sorted(SEALED_SEASONS),
        "source": "nflverse-data schedules (CC-BY-4.0)",
        "line_note": ("Closing spread and total as published by nflverse: consensus, "
                      "book unspecified, capture time unknown. 'Pick'em' games are "
                      "scored from the home side, since neither team is favoured."),
    }
