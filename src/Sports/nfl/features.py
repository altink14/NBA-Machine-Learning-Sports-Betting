"""
features.py (NFL)
=================
Builds the model feature frame specified in
`docs/sports/nfl/MODEL_PREREGISTRATION_v2.md` section 6, sealed at commit
a88f762a065b21831e863e5a3a9935c5a1af4639. (v1 was voided for reading its own
sealed window; see MODEL_V1_VOIDED.md.)

THE DESIGN IS THE LEAKAGE CONTROL. Games are walked in kickoff order while a
per-team state object is carried forward. For each game we EMIT features from
the state as it stands, and only then UPDATE the state with that game's result.
A feature therefore cannot see its own game, or any later one, because those
rows have not been folded in yet. This is structural: it does not depend on a
WHERE clause being right, and `leakage_tests.py` proves it by rebuilding
features from truncated inputs and demanding identical output.

THE SEALED WINDOW IS THE FUTURE. `build_frame()` refuses to return the 2026
season unless `sealed_evaluation=True`, but that guard is now the belt rather
than the braces: the real protection is that 2026 has not been played, so there
is nothing to peek at. v1 relied on discipline alone and the discipline failed
within hours.

NO ODDS. Not as a feature, not as a prior, not as a sanity check. The point of
the market comparison in the pre-registration is that the model never saw a
line; a single odds-derived column would make that comparison circular and
worthless.

Usage:
    from src.Sports.nfl.features import build_frame
    rows, cols = build_frame()                      # train + tune only
    rows, cols = build_frame(seasons=("2019","2020","2021"))
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.nfl.teams import NFL_TEAMS, CURRENT_OF  # noqa: E402

DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")

# --- windows, from pre-registration v2 (a88f762) ---------------------------
# v1 sealed 2022-2025 and was VOIDED when that window was read; see
# docs/sports/nfl/MODEL_V1_VOIDED.md. v2's seal is the 2026 season, which
# cannot be peeked at because it has not been played. Those four seasons are
# therefore now open validation data.
TRAIN_SEASONS = tuple(str(y) for y in range(1999, 2019))
TUNE_SEASONS = tuple(str(y) for y in range(2019, 2026))      # 2019-2025, open
SEALED_SEASONS = ("2026",)                                    # the future
LIVE_SEASONS = ("2027",)

# --- Elo constants. Standard values; not tuned on the sealed window. -------
ELO_START = 1500.0
ELO_K = 20.0
# The home-field term. A FIXED value was the single biggest error in the first
# fit: 65 rating points asserts that home teams win 59.2% of even matchups, in
# an era where they win about 53.5%. The result was a model that was
# overconfident in every probability bucket at once.
#
# It is now estimated from a trailing window of completed games, which
# pre-registration v2 section 6 authorised in advance as the one permitted
# change. The estimate uses only games that had already kicked off, so it is
# subject to leakage control 1 like every other feature.
ELO_HOME_EDGE_PRIOR = 45.0   # whole-archive value, used until the window fills
ELO_HOME_WINDOW = 512        # completed games, roughly two seasons
ELO_HOME_EDGE_MIN = 0.0      # a home DISadvantage is not a thing we will assert
ELO_HOME_EDGE_MAX = 90.0
ELO_SEASON_REGRESSION = 1.0 / 3.0   # toward the mean, between seasons

ROLL_SHORT = 8
ROLL_LONG = 16

#: Column-name fragments that must never appear in a feature frame. Scanned by
#: leakage_tests.py. If a future contributor adds a line-derived feature, the
#: test fails before the model is ever fitted.
ODDS_DENYLIST = ("odds", "line", "spread", "total_line", "moneyline", "ml_",
                 "vegas", "implied", "juice", "vig", "price", "book")


class TeamState:
    """Everything we know about a team from games already played."""

    __slots__ = ("elo", "epa_off", "epa_def", "sr_off", "sr_def", "pf", "pa",
                 "wins", "losses", "last_kickoff")

    def __init__(self) -> None:
        self.elo = ELO_START
        self.epa_off: deque = deque(maxlen=ROLL_LONG)
        self.epa_def: deque = deque(maxlen=ROLL_LONG)
        self.sr_off: deque = deque(maxlen=ROLL_LONG)
        self.sr_def: deque = deque(maxlen=ROLL_LONG)
        self.pf: deque = deque(maxlen=ROLL_LONG)
        self.pa: deque = deque(maxlen=ROLL_LONG)
        self.wins = 0
        self.losses = 0
        self.last_kickoff: Optional[str] = None

    def new_season(self) -> None:
        """Carry strength across the boundary, but not the record.

        Ratings regress toward the mean because rosters and coaches change;
        the rolling windows are kept, which is what lets week 1 have any
        signal at all. Win-loss resets because a 2017 record says nothing
        about 2018's standings.
        """
        self.elo = ELO_START + (self.elo - ELO_START) * (1.0 - ELO_SEASON_REGRESSION)
        self.wins = 0
        self.losses = 0


class HomeEdgeEstimator:
    """Turns the recent rate of home wins into Elo rating points.

    If home teams beat evenly matched opponents a fraction p of the time, the
    Elo edge that implies is -400 * log10(1/p - 1). Fed only completed games,
    oldest first, so the value used for any game reflects only earlier ones.
    """

    __slots__ = ("_wins", "_n", "_recent")

    def __init__(self) -> None:
        self._recent: deque = deque(maxlen=ELO_HOME_WINDOW)

    @property
    def points(self) -> float:
        if len(self._recent) < 64:
            return ELO_HOME_EDGE_PRIOR
        p = sum(self._recent) / len(self._recent)
        p = min(max(p, 0.5001), 0.95)      # log10 needs p strictly inside (0, 1)
        edge = -400.0 * math.log10(1.0 / p - 1.0)
        return min(max(edge, ELO_HOME_EDGE_MIN), ELO_HOME_EDGE_MAX)

    def observe(self, home_result: float) -> None:
        self._recent.append(home_result)


def _mean(d: Sequence[float], n: Optional[int] = None) -> Optional[float]:
    vals = list(d)[-n:] if n else list(d)
    return sum(vals) / len(vals) if vals else None


def _conf_div(abbr: str) -> Tuple[str, str]:
    t = NFL_TEAMS.get(abbr)
    return (t[1], t[2]) if t else ("", "")


def _is_divisional(home: str, away: str) -> int:
    hc, hd = _conf_div(CURRENT_OF.get(home, home))
    ac, ad = _conf_div(CURRENT_OF.get(away, away))
    return 1 if hc and hc == ac and hd == ad else 0


def _days_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    try:
        from datetime import datetime
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return None


def _team_game_aggregates(conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Per (game, team) offensive rates from OUR play-by-play.

    Only run and pass plays count, so kneels, spikes, kicks and penalties do
    not dilute the rate. A team's defensive numbers are its opponent's
    offensive numbers in the same game, resolved by the caller.
    """
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in conn.execute(
        """
        SELECT game_id, posteam, COUNT(*) n, AVG(epa) epa, AVG(success) sr
        FROM nfl_plays
        WHERE posteam IS NOT NULL AND play_type IN ('run', 'pass') AND epa IS NOT NULL
        GROUP BY game_id, posteam
        """
    ):
        if r[2] and r[2] >= 10:      # a handful of rows are fragments
            out[(r[0], r[1])] = {"epa": r[3], "sr": r[4] if r[4] is not None else 0.0}
    return out


FEATURE_COLUMNS = [
    # identity, not fed to the model
    "game_id", "season", "week", "season_type", "kickoff_utc", "home_team", "away_team",
    # target
    "home_win",
    # strength
    "elo_home", "elo_away", "elo_diff", "home_edge_pts",
    "epa_off_home_8", "epa_off_away_8", "epa_def_home_8", "epa_def_away_8",
    "epa_off_home_16", "epa_off_away_16", "epa_def_home_16", "epa_def_away_16",
    "sr_off_home_8", "sr_off_away_8", "sr_def_home_8", "sr_def_away_8",
    "pf_home_8", "pf_away_8", "pa_home_8", "pa_away_8",
    "winpct_home", "winpct_away", "games_played_home", "games_played_away",
    # situation
    "is_neutral", "rest_home", "rest_away", "rest_diff",
    "short_week_home", "short_week_away", "bye_home", "bye_away",
    "is_divisional", "week_num", "is_postseason",
    # conditions
    "roof_outdoor", "roof_dome", "roof_closed", "temp_f", "wind_mph", "surface_turf",
]

#: Columns the model may actually see. Everything before "home_win" is
#: identity and everything after is a feature.
MODEL_COLUMNS = FEATURE_COLUMNS[FEATURE_COLUMNS.index("home_win") + 1:]


def build_frame(db_path: str = DB_PATH,
                seasons: Optional[Sequence[str]] = None,
                sealed_evaluation: bool = False,
                include_live: bool = False,
                max_kickoff: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Walk every game in kickoff order, emitting features from prior state.

    State is always built from the FULL history in order, because a team's
    week 1 2019 rating depends on 2018. Only the RETURNED rows are filtered.
    That is the correct separation: using earlier seasons to compute a feature
    is not leakage; using later ones is.

    `max_kickoff` exists for the leakage tests: it truncates the input to games
    starting strictly before that timestamp, so a test can rebuild a row as if
    the rest of history had not happened yet and demand the identical answer.
    Production code never passes it.
    """
    allowed = set(TRAIN_SEASONS) | set(TUNE_SEASONS)
    if sealed_evaluation:
        allowed |= set(SEALED_SEASONS)
    if include_live:
        allowed |= set(LIVE_SEASONS)
    if seasons is not None:
        requested = set(seasons)
        forbidden = requested - allowed
        if forbidden:
            raise PermissionError(
                f"Seasons {sorted(forbidden)} are outside the permitted windows. "
                f"The sealed window {SEALED_SEASONS} may be read only with "
                f"sealed_evaluation=True, which is used exactly once. See "
                f"docs/sports/nfl/MODEL_PREREGISTRATION_v1.md.")
        allowed = requested

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    agg = _team_game_aggregates(conn)

    games = conn.execute(
        """
        SELECT g.game_id, g.season, g.season_type, g.week, g.date_utc, g.local_date,
               g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.neutral_site,
               r1.days_rest AS rest_home, r2.days_rest AS rest_away,
               w.temp_f, w.wind_mph, w.roof_state, v.surface
        FROM games g
        LEFT JOIN rest_travel r1 ON r1.game_id = g.game_id AND r1.team_id = g.home_team_id
        LEFT JOIN rest_travel r2 ON r2.game_id = g.game_id AND r2.team_id = g.away_team_id
        LEFT JOIN weather w      ON w.game_id = g.game_id
        LEFT JOIN venues v       ON v.venue_id = g.venue_id
        WHERE g.status = 'final'
        ORDER BY COALESCE(g.date_utc, g.local_date || 'T17:00:00+00:00'), g.game_id
        """
    ).fetchall()
    if max_kickoff is not None:
        games = [g for g in games
                 if (g["date_utc"] or (g["local_date"] + "T17:00:00+00:00")) < max_kickoff]
    conn.close()

    state: Dict[str, TeamState] = {}
    season_seen: Dict[str, str] = {}
    home_edge = HomeEdgeEstimator()
    rows: List[Dict[str, Any]] = []

    for g in games:
        home = g["home_team_id"].replace("nfl-", "")
        away = g["away_team_id"].replace("nfl-", "")
        season = g["season"]
        kickoff = g["date_utc"] or (g["local_date"] + "T17:00:00+00:00")

        for t in (home, away):
            st = state.setdefault(t, TeamState())
            if season_seen.get(t) != season:
                if season_seen.get(t) is not None:
                    st.new_season()
                season_seen[t] = season

        hs_, as_ = state[home], state[away]

        # ---------- EMIT: everything below reads state, nothing writes it ----
        if season in allowed and g["home_score"] is not None and g["away_score"] is not None:
            roof = (g["roof_state"] or "").lower()
            surface = (g["surface"] or "").strip().lower()
            gp_h, gp_a = hs_.wins + hs_.losses, as_.wins + as_.losses
            rows.append({
                "game_id": g["game_id"], "season": season, "week": g["week"],
                "season_type": g["season_type"], "kickoff_utc": kickoff,
                "home_team": home, "away_team": away,
                "home_win": 1 if g["home_score"] > g["away_score"] else 0,

                "elo_home": hs_.elo, "elo_away": as_.elo, "elo_diff": hs_.elo - as_.elo,
                "home_edge_pts": home_edge.points,
                "epa_off_home_8": _mean(hs_.epa_off, ROLL_SHORT),
                "epa_off_away_8": _mean(as_.epa_off, ROLL_SHORT),
                "epa_def_home_8": _mean(hs_.epa_def, ROLL_SHORT),
                "epa_def_away_8": _mean(as_.epa_def, ROLL_SHORT),
                "epa_off_home_16": _mean(hs_.epa_off), "epa_off_away_16": _mean(as_.epa_off),
                "epa_def_home_16": _mean(hs_.epa_def), "epa_def_away_16": _mean(as_.epa_def),
                "sr_off_home_8": _mean(hs_.sr_off, ROLL_SHORT),
                "sr_off_away_8": _mean(as_.sr_off, ROLL_SHORT),
                "sr_def_home_8": _mean(hs_.sr_def, ROLL_SHORT),
                "sr_def_away_8": _mean(as_.sr_def, ROLL_SHORT),
                "pf_home_8": _mean(hs_.pf, ROLL_SHORT), "pf_away_8": _mean(as_.pf, ROLL_SHORT),
                "pa_home_8": _mean(hs_.pa, ROLL_SHORT), "pa_away_8": _mean(as_.pa, ROLL_SHORT),
                "winpct_home": (hs_.wins / gp_h) if gp_h else None,
                "winpct_away": (as_.wins / gp_a) if gp_a else None,
                "games_played_home": gp_h, "games_played_away": gp_a,

                "is_neutral": g["neutral_site"] or 0,
                "rest_home": g["rest_home"], "rest_away": g["rest_away"],
                "rest_diff": (None if g["rest_home"] is None or g["rest_away"] is None
                              else g["rest_home"] - g["rest_away"]),
                "short_week_home": 1 if (g["rest_home"] or 99) <= 5 else 0,
                "short_week_away": 1 if (g["rest_away"] or 99) <= 5 else 0,
                "bye_home": 1 if (g["rest_home"] or 0) >= 13 else 0,
                "bye_away": 1 if (g["rest_away"] or 0) >= 13 else 0,
                "is_divisional": _is_divisional(home, away),
                "week_num": g["week"],
                "is_postseason": 0 if g["season_type"] == "REG" else 1,

                "roof_outdoor": 1 if roof in ("outdoors", "open") else 0,
                "roof_dome": 1 if roof == "dome" else 0,
                "roof_closed": 1 if roof == "closed" else 0,
                "temp_f": g["temp_f"], "wind_mph": g["wind_mph"],
                "surface_turf": 0 if surface.startswith("grass") else (1 if surface else 0),
            })

        # ---------- UPDATE: only now does this game exist for the future ----
        if g["home_score"] is None or g["away_score"] is None:
            continue
        home_won = g["home_score"] > g["away_score"]

        edge = 0.0 if g["neutral_site"] else home_edge.points
        exp_home = 1.0 / (1.0 + 10 ** (-((hs_.elo + edge) - as_.elo) / 400.0))
        actual = 1.0 if home_won else (0.5 if g["home_score"] == g["away_score"] else 0.0)
        delta = ELO_K * (actual - exp_home)
        hs_.elo += delta
        as_.elo -= delta
        if not g["neutral_site"]:
            home_edge.observe(actual)

        ha = agg.get((g["game_id"], home))
        aa = agg.get((g["game_id"], away))
        if ha:
            hs_.epa_off.append(ha["epa"]); hs_.sr_off.append(ha["sr"])
            as_.epa_def.append(ha["epa"]); as_.sr_def.append(ha["sr"])
        if aa:
            as_.epa_off.append(aa["epa"]); as_.sr_off.append(aa["sr"])
            hs_.epa_def.append(aa["epa"]); hs_.sr_def.append(aa["sr"])

        hs_.pf.append(g["home_score"]); hs_.pa.append(g["away_score"])
        as_.pf.append(g["away_score"]); as_.pa.append(g["home_score"])
        if g["home_score"] != g["away_score"]:
            (hs_ if home_won else as_).wins += 1
            (as_ if home_won else hs_).losses += 1
        hs_.last_kickoff = as_.last_kickoff = kickoff

    return rows, FEATURE_COLUMNS


if __name__ == "__main__":
    rows, cols = build_frame()
    print(f"{len(rows)} rows, {len(MODEL_COLUMNS)} model features")
    by_season: Dict[str, int] = {}
    for r in rows:
        by_season[r["season"]] = by_season.get(r["season"], 0) + 1
    print("seasons:", ", ".join(f"{k}:{v}" for k, v in sorted(by_season.items())))
