"""
retrain_features.py
===================
STEP 1 of the pre-registered model-retrain protocol: leakage-safe feature
builders. This module builds features ONLY -- it trains nothing and never
writes to any existing database file.

Data sources
------------
- Data/TeamData.sqlite : ~3,996 date-keyed snapshot tables (2007-10-31 ..
  2024-04-29). A table named D holds each team's REGULAR-SEASON per-game
  averages through the games of D-1 (verified: the 2023-10-25 table shows
  GP=1 for teams that played on 2023-10-24, and the opening-day table is
  empty). Averages are rounded to 1 decimal (PCT columns to 3 decimals);
  GP / W / L are exact integers.
- Data/OddsData.sqlite : odds_YYYY-YY_new tables, 2007-08 .. 2023-24, one
  row per game with Date, Home, Away, OU, Spread, ML_Home, ML_Away, Points
  (total points), Win_Margin (home score minus away score), Days_Rest_*.
  Rows with Win_Margin == 0 or NULL scores are dropped (5 such rows in the
  whole archive). The Home column IS the home team.

Rolling averages from snapshot diffs
------------------------------------
For a counting stat, snapshot D stores avg_D = total_D / GP_D rounded to
0.1. Two snapshots exactly K team-games apart give

    rolling_K = (avg_D * GP_D - avg_D' * GP_D') / K

with worst-case rounding error 0.05 * (GP_D + GP_D') / K (about +/-0.35 for
K=20 late in a season) -- negligible against real stat variance. Shooting
percentages are NOT averaged directly (a snapshot PCT is total-made /
total-attempted, not a mean of per-game PCTs); they are recomputed as the
ratio of the reconstructed made/attempted totals. Rolling win% comes from
integer W diffs and is EXACT.

Stat-definition note (matters for step 2): the snapshots' TOV is TOTAL
team turnovers, INCLUDING team turnovers (shot-clock/8-second/5-second
violations), whereas the box_scores traditional team row is the
player-summed turnovers only -- about 0.7/game lower on average. Rolling
TOV built here therefore reflects true total turnovers. Any model feature
mixing this TOV with a player-summed source would be inconsistent.

As-of discipline (leakage rule)
-------------------------------
Features for a game played on date D may consult snapshot tables named
<= D only. Because table D itself contains games through D-1, using it for
a game on D is safe; using table D+1 would leak the game's own result.
`build_rolling_features` enforces this and reports the snapshot dates it
used so tests can audit the rule.

Elo over the odds tables
------------------------
`build_elo_odds` replays every odds-table game 2007-08 .. 2023-24 with the
SAME constants as src/Utils/elo.py (K=20, home advantage 70, 538
margin-of-victory multiplier, 25% between-season reversion toward 1505,
base 1500) and stores the PRE-GAME rating each team carries INTO the game.
Processing order is deterministic: games are sorted by (season, date,
canonical home name, canonical away name). Ties within a date cannot
interact (a team plays at most once per day), so the secondary key is for
reproducibility only.

Rest features
-------------
`build_rest_features` recomputes rest from game-date sequences, ignoring
the two inconsistent Days_Rest_* conventions in the odds archive. Unified
convention: rest = calendar-day difference from the team's previous game
in the same season, capped at 7; a season opener gets 7 (well-rested).
The minimum possible value is 1 (a back-to-back); 0 cannot occur. B2B flag
= (rest == 1). 3-in-4 flag = the game is the team's 3rd (or more) game in
any 4-calendar-day window ending on the game date.

Caching
-------
`build_cache` writes to a NEW file, Data/retrain_features.sqlite, and
refuses to open any protected existing database. The cache is derived data:
safe to delete and rebuild at any time.

Importing
---------
This file lives in the hyphenated directory src/Process-Data, which is not
a valid Python package name. Load it with importlib:

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "retrain_features", "src/Process-Data/retrain_features.py")
    rf = importlib.util.module_from_spec(spec)
    sys.modules["retrain_features"] = rf   # required for dataclasses
    spec.loader.exec_module(rf)
"""

from __future__ import annotations

import bisect
import datetime as _dt
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM_DATA_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
ODDS_DATA_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
CACHE_DB = os.path.join(REPO_ROOT, "Data", "retrain_features.sqlite")

#: Existing databases this module must never write to.
PROTECTED_DBS = frozenset(
    os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, "Data", n)))
    for n in ("TeamData.sqlite", "OddsData.sqlite", "dataset.sqlite",
              "test_backfill_aggregates.sqlite", "test_nba_pipeline.sqlite")
)

# ---------------------------------------------------------------------------
# 1. Team-name normalization
# ---------------------------------------------------------------------------
# Canonical identity = the modern franchise name. Franchise renames observed
# in the two databases (verified against the actual tables):
#   snapshots: "Seattle SuperSonics" (2007-08), "New Jersey Nets" (..2011-12),
#              "New Orleans Hornets" (..2012-13), "Charlotte Bobcats"
#              (..2013-14), "LA Clippers" (2015-16..).
#   odds:      "New Orleans Pelicans" retroactively in ALL seasons,
#              "Charlotte Bobcats" through 2021-22 (switches 2022-23),
#              "Los Angeles Clippers" in all seasons,
#              "Seattle SuperSonics" / "New Jersey Nets" as played.
# All aliases collapse to one canonical name, so era does not matter.
TEAM_NAME_MAP: Dict[str, str] = {
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "New Jersey Nets": "Brooklyn Nets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "Charlotte Bobcats": "Charlotte Hornets",
    "LA Clippers": "Los Angeles Clippers",
}

CANONICAL_TEAMS: Tuple[str, ...] = (
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks",
    "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
    "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers",
    "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
    "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
    "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz",
    "Washington Wizards",
)
_CANONICAL_SET = frozenset(CANONICAL_TEAMS)


def normalize_team(name: str) -> str:
    """Map any team name found in either database to its canonical
    franchise name. Raises KeyError for a name that is neither canonical
    nor a known alias (better to fail loudly than silently mis-join)."""
    name = name.strip()
    if name in _CANONICAL_SET:
        return name
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    raise KeyError(f"Unknown team name: {name!r}")


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------
#: Counting-stat columns taken from the snapshot tables (per-game averages).
COUNT_STATS: Tuple[str, ...] = (
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
)
#: Percentage stats recomputed from reconstructed made/attempted totals.
PCT_STATS: Dict[str, Tuple[str, str]] = {
    "FG_PCT": ("FGM", "FGA"),
    "FG3_PCT": ("FG3M", "FG3A"),
    "FT_PCT": ("FTM", "FTA"),
}


@dataclass
class _StateEntry:
    """A team's cumulative regular-season state, first seen at `date`
    (i.e. valid for any game on `date` or later, until the next entry)."""
    date: str            # first snapshot date carrying this GP
    gp: int              # games played (exact)
    wins: int            # wins (exact)
    totals: Dict[str, float]  # reconstructed stat totals = avg * gp
    avgs: Dict[str, float]    # per-game averages exactly as stored


@dataclass
class _TeamSeason:
    entries: List[_StateEntry] = field(default_factory=list)
    by_gp: Dict[int, _StateEntry] = field(default_factory=dict)
    dates: List[str] = field(default_factory=list)  # entry dates, ascending


class SnapshotStore:
    """Lazy per-season loader for the date-keyed snapshot tables.

    Seasons are discovered by clustering the snapshot table dates: a gap of
    more than 45 days starts a new cluster, and a cluster is a NEW season
    only if it starts in Oct-Dec (this keeps the 2020 bubble restart, which
    resumed in July 2020 after a 4-month gap, inside 2019-20)."""

    def __init__(self, db_path: str = TEAM_DATA_DB):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'")]
        finally:
            conn.close()
        self.snapshot_dates: List[str] = sorted(names)
        if not self.snapshot_dates:
            raise RuntimeError(f"No snapshot tables found in {db_path}")
        self._season_windows = self._cluster_seasons(self.snapshot_dates)
        self._season_starts = [w[1] for w in self._season_windows]
        self._loaded: Dict[str, Dict[str, _TeamSeason]] = {}

    @staticmethod
    def _cluster_seasons(dates: List[str]) -> List[Tuple[str, str, str]]:
        """Return [(season_label, first_date, last_date), ...] ascending."""
        clusters: List[List[str]] = [[dates[0]]]
        prev = _dt.date.fromisoformat(dates[0])
        for d in dates[1:]:
            cur = _dt.date.fromisoformat(d)
            if (cur - prev).days > 45 and cur.month in (10, 11, 12):
                clusters.append([])
            clusters[-1].append(d)
            prev = cur
        out = []
        for c in clusters:
            y = int(c[0][:4])
            label = f"{y}-{str(y + 1)[2:]}"
            out.append((label, c[0], c[-1]))
        return out

    # -- season lookup ------------------------------------------------------
    def season_of(self, date: str) -> Optional[str]:
        """Season whose window start is the latest one <= date. Dates before
        the first snapshot return None."""
        i = bisect.bisect_right(self._season_starts, date) - 1
        if i < 0:
            return None
        return self._season_windows[i][0]

    def seasons(self) -> List[str]:
        return [w[0] for w in self._season_windows]

    # -- loading ------------------------------------------------------------
    def _season_dates(self, season: str) -> List[str]:
        for label, first, last in self._season_windows:
            if label == season:
                lo = bisect.bisect_left(self.snapshot_dates, first)
                hi = bisect.bisect_right(self.snapshot_dates, last)
                return self.snapshot_dates[lo:hi]
        raise KeyError(f"Season {season!r} not covered by snapshots")

    def _load_season(self, season: str) -> Dict[str, _TeamSeason]:
        if season in self._loaded:
            return self._loaded[season]
        data: Dict[str, _TeamSeason] = {}
        self.team_ids: Dict[str, int] = getattr(self, "team_ids", {})
        conn = sqlite3.connect(self.db_path)
        try:
            cols = ", ".join(["TEAM_ID", "TEAM_NAME", "GP", "W"] + list(COUNT_STATS))
            for date in self._season_dates(season):
                for row in conn.execute(f'SELECT {cols} FROM "{date}"'):
                    team_id, raw_name, gp, wins = row[0], row[1], row[2], row[3]
                    if gp is None or gp == 0:
                        continue
                    team = normalize_team(raw_name)
                    self.team_ids.setdefault(team, int(team_id))
                    ts = data.setdefault(team, _TeamSeason())
                    if ts.entries and ts.entries[-1].gp == gp:
                        continue  # no new game since previous snapshot
                    avgs = {s: row[4 + i] for i, s in enumerate(COUNT_STATS)}
                    totals = {s: (v * gp if v is not None else 0.0)
                              for s, v in avgs.items()}
                    entry = _StateEntry(date=date, gp=int(gp), wins=int(wins),
                                        totals=totals, avgs=avgs)
                    ts.entries.append(entry)
                    ts.by_gp[int(gp)] = entry
                    ts.dates.append(date)
        finally:
            conn.close()
        self._loaded[season] = data
        return data

    # -- queries ------------------------------------------------------------
    def state_entering(self, team: str, as_of_date: str) -> Optional[_StateEntry]:
        """The team's cumulative state ENTERING a game on `as_of_date`:
        the newest entry whose snapshot date is <= as_of_date within the
        season containing as_of_date. Never consults a table dated after
        as_of_date (as-of discipline)."""
        season = self.season_of(as_of_date)
        if season is None:
            return None
        ts = self._load_season(season).get(normalize_team(team))
        if ts is None:
            return None
        i = bisect.bisect_right(ts.dates, as_of_date) - 1
        if i < 0:
            return None
        entry = ts.entries[i]
        assert entry.date <= as_of_date, "as-of discipline violated"
        return entry

    def state_at_gp(self, team: str, season: str, gp: int) -> Optional[_StateEntry]:
        ts = self._load_season(season).get(normalize_team(team))
        if ts is None:
            return None
        return ts.by_gp.get(gp)


# ---------------------------------------------------------------------------
# 2. Rolling-K features from snapshot diffs
# ---------------------------------------------------------------------------
def build_rolling_features(K: int, as_of_date: str, team: str,
                           store: Optional[SnapshotStore] = None) -> Optional[dict]:
    """Rolling-K per-game averages for `team` entering a game on
    `as_of_date`, built purely from snapshot tables dated <= as_of_date.

    Returns None when the team has played 0 games in the season as of the
    date (season opener), otherwise a dict:

        {
          'stats':       {stat: rolling per-game average, plus FG_PCT/
                          FG3_PCT/FT_PCT recomputed from totals, plus
                          'WIN_PCT' (exact, from integer W diffs)},
          'window':      games actually averaged (== K normally),
          'is_partial':  True when GP < K (season-to-date average used),
          'exact_window':False when a snapshot gap forced a window != K
                          for a full-GP team (rare; reported by tests),
          'gp':          games played entering the game,
          'season':      season label,
          'snapshot_date':      snapshot date used for the current state,
          'prev_snapshot_date': snapshot date used for the K-games-ago
                                state (None when is_partial),
        }
    """
    if store is None:
        store = _default_store()
    if K <= 0:
        raise ValueError("K must be positive")
    cur = store.state_entering(team, as_of_date)
    if cur is None or cur.gp == 0:
        return None
    season = store.season_of(as_of_date)

    if cur.gp < K:
        stats = {s: cur.avgs[s] for s in COUNT_STATS}
        for pct, (num, den) in PCT_STATS.items():
            d = cur.totals[den]
            stats[pct] = (cur.totals[num] / d) if d else None
        stats["WIN_PCT"] = cur.wins / cur.gp
        return {"stats": stats, "window": cur.gp, "is_partial": True,
                "exact_window": True, "gp": cur.gp, "season": season,
                "snapshot_date": cur.date, "prev_snapshot_date": None}

    target_gp = cur.gp - K
    prev = store.state_at_gp(team, season, target_gp)
    exact = True
    if prev is None and target_gp == 0:
        prev = _StateEntry(date="", gp=0, wins=0,
                           totals={s: 0.0 for s in COUNT_STATS}, avgs={})
    if prev is None:
        # A snapshot gap swallowed the exact target GP; use the closest
        # earlier available GP and flag the inexact window.
        ts = store._load_season(season).get(normalize_team(team))
        candidates = [g for g in ts.by_gp if g < target_gp] if ts else []
        if not candidates:
            return None
        prev = ts.by_gp[max(candidates)]
        exact = False
    window = cur.gp - prev.gp
    stats = {s: (cur.totals[s] - prev.totals[s]) / window for s in COUNT_STATS}
    for pct, (num, den) in PCT_STATS.items():
        d = cur.totals[den] - prev.totals[den]
        stats[pct] = ((cur.totals[num] - prev.totals[num]) / d) if d else None
    stats["WIN_PCT"] = (cur.wins - prev.wins) / window
    return {"stats": stats, "window": window, "is_partial": False,
            "exact_window": exact, "gp": cur.gp, "season": season,
            "snapshot_date": cur.date,
            "prev_snapshot_date": prev.date or None}


def build_rolling_features_batch(K: int, games: Iterable[Tuple[str, str]],
                                 store: Optional[SnapshotStore] = None
                                 ) -> Dict[Tuple[str, str], Optional[dict]]:
    """Batch variant: `games` is an iterable of (as_of_date, team_name).
    Returns {(as_of_date, canonical_team): result-or-None}."""
    if store is None:
        store = _default_store()
    out: Dict[Tuple[str, str], Optional[dict]] = {}
    for as_of_date, team in games:
        canonical = normalize_team(team)
        out[(as_of_date, canonical)] = build_rolling_features(
            K, as_of_date, canonical, store)
    return out


_STORE_SINGLETON: Optional[SnapshotStore] = None


def _default_store() -> SnapshotStore:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        _STORE_SINGLETON = SnapshotStore()
    return _STORE_SINGLETON


# ---------------------------------------------------------------------------
# Odds-table game loading (shared by Elo and rest builders)
# ---------------------------------------------------------------------------
ODDS_SEASONS: Tuple[str, ...] = (
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13",
    "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
)


def load_odds_games(db_path: str = ODDS_DATA_DB,
                    seasons: Sequence[str] = ODDS_SEASONS) -> List[dict]:
    """All graded games from the odds_YYYY-YY_new tables, canonical names,
    bad rows dropped (Win_Margin == 0/NULL or Points NULL -- 5 rows in the
    full archive). Deterministic order: (season, date, home, away)."""
    conn = sqlite3.connect(db_path)
    games: List[dict] = []
    try:
        for season in seasons:
            rows = conn.execute(
                f'SELECT Date, Home, Away, OU, Spread, ML_Home, ML_Away, '
                f'Points, Win_Margin FROM "odds_{season}_new"').fetchall()
            for (date, home, away, ou, spread, ml_h, ml_a, points,
                 margin) in rows:
                if margin is None or margin == 0 or points is None:
                    continue  # ungraded / corrupt row
                games.append({
                    "season": season, "date": date,
                    "home": normalize_team(home), "away": normalize_team(away),
                    "ou": ou, "spread": spread,
                    "ml_home": ml_h, "ml_away": ml_a,
                    "points": points, "win_margin": margin,
                })
    finally:
        conn.close()
    games.sort(key=lambda g: (g["season"], g["date"], g["home"], g["away"]))
    return games


# ---------------------------------------------------------------------------
# 3. Elo over the odds tables (same constants as src/Utils/elo.py)
# ---------------------------------------------------------------------------
ELO_BASE = 1500.0
ELO_MEAN_REVERT_TARGET = 1505.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 70.0
ELO_SEASON_CARRYOVER = 0.75


def _elo_expected_home(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (
        -((home_elo + ELO_HOME_ADVANTAGE) - away_elo) / 400.0))


def _elo_mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)


def build_elo_odds(db_path: str = ODDS_DATA_DB,
                   start_season: str = "2007-08",
                   end_season: str = "2023-24",
                   pre_game: bool = True) -> dict:
    """Replay Elo over the odds tables. Ratings burn in from `start_season`
    (production: 2007-08). Between seasons every team reverts 25% toward
    1505. Games are processed in the deterministic (season, date, home,
    away) order from load_odds_games; a team plays at most once per date,
    so same-date ordering cannot change any rating.

    Returns:
      {
        'pre_game':  {(date, home, away): {'home_elo', 'away_elo',
                      'home_expected'}}   # ratings ENTERING the game
        'post_game': same keyed dict with ratings AFTER the game
                     (only when pre_game=False adds nothing; both always
                      returned for convenience),
        'final_ratings':      {team: elo}  # state after last processed game
        'season_end_ratings': {season: {team: elo}}  # BEFORE reversion
        'n_games': int,
      }
    """
    seasons = [s for s in ODDS_SEASONS if start_season <= s <= end_season]
    games = load_odds_games(db_path, seasons)
    ratings: Dict[str, float] = {}
    pre: Dict[Tuple[str, str, str], dict] = {}
    post: Dict[Tuple[str, str, str], dict] = {}
    season_end: Dict[str, Dict[str, float]] = {}
    current_season: Optional[str] = None

    for g in games:
        if g["season"] != current_season:
            if current_season is not None:
                season_end[current_season] = dict(ratings)
                for t in ratings:
                    ratings[t] = (ELO_SEASON_CARRYOVER * ratings[t]
                                  + (1.0 - ELO_SEASON_CARRYOVER)
                                  * ELO_MEAN_REVERT_TARGET)
            current_season = g["season"]

        home, away = g["home"], g["away"]
        home_elo = ratings.get(home, ELO_BASE)
        away_elo = ratings.get(away, ELO_BASE)
        expected_home = _elo_expected_home(home_elo, away_elo)
        key = (g["date"], home, away)
        pre[key] = {"home_elo": home_elo, "away_elo": away_elo,
                    "home_expected": expected_home}

        margin = g["win_margin"]
        home_won = margin > 0
        if home_won:
            elo_diff_winner = (home_elo + ELO_HOME_ADVANTAGE) - away_elo
        else:
            elo_diff_winner = away_elo - (home_elo + ELO_HOME_ADVANTAGE)
        shift = (ELO_K * _elo_mov_multiplier(margin, elo_diff_winner)
                 * ((1.0 if home_won else 0.0) - expected_home))
        ratings[home] = home_elo + shift
        ratings[away] = away_elo - shift
        post[key] = {"home_elo": ratings[home], "away_elo": ratings[away]}

    if current_season is not None:
        season_end[current_season] = dict(ratings)

    return {"pre_game": pre, "post_game": post,
            "final_ratings": dict(ratings),
            "season_end_ratings": season_end,
            "n_games": len(games)}


# ---------------------------------------------------------------------------
# 4. Rest features from game-date sequences
# ---------------------------------------------------------------------------
REST_CAP = 7
SEASON_OPENER_REST = 7


def build_rest_features(games: Iterable[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Rest features recomputed from game dates (the odds tables' own
    Days_Rest_* columns mix two conventions across seasons and are ignored).

    `games`: iterable of dicts with 'date', 'home', 'away' and optionally
    'season'. When 'season' is missing, a team-schedule gap of more than 60
    days starts a new season (only relevant for synthetic inputs; odds rows
    always carry a season).

    Unified convention (documented for step 2):
      rest        = calendar-day diff to the team's previous game in the
                    same season, capped at REST_CAP=7; a season opener
                    gets SEASON_OPENER_REST=7. Minimum value 1; 0 never
                    occurs.
      b2b         = rest == 1
      three_in_four = 3rd-or-more game in the 4-calendar-day window ending
                    on the game date (game dates D-3..D inclusive)
      rest_diff   = home_rest - away_rest

    Returns {(date, home, away): {'home_rest','away_rest','home_b2b',
             'away_b2b','home_3in4','away_3in4','rest_diff'}}.
    """
    rows = []
    for g in games:
        rows.append({"date": g["date"],
                     "home": normalize_team(g["home"]),
                     "away": normalize_team(g["away"]),
                     "season": g.get("season")})
    rows.sort(key=lambda r: (r["date"], r["home"], r["away"]))

    # Per-team chronological date lists with season labels.
    team_dates: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for r in rows:
        for side in ("home", "away"):
            team_dates.setdefault(r[side], []).append((r["date"], r["season"]))

    # (team, date) -> (rest, b2b, three_in_four)
    per_team: Dict[Tuple[str, str], Tuple[int, bool, bool]] = {}
    for team, seq in team_dates.items():
        seq.sort()
        prev_date: Optional[_dt.date] = None
        prev_season: Optional[str] = None
        season_dates: List[_dt.date] = []
        for date_str, season in seq:
            d = _dt.date.fromisoformat(date_str)
            new_season = False
            if prev_date is None:
                new_season = True
            elif season is not None and season != prev_season:
                new_season = True
            elif season is None and (d - prev_date).days > 60:
                new_season = True
            if new_season:
                season_dates = []
                rest = SEASON_OPENER_REST
            else:
                rest = min(REST_CAP, (d - prev_date).days)
            n_in_4 = 1 + sum(1 for x in season_dates if (d - x).days <= 3)
            per_team[(team, date_str)] = (rest, rest == 1, n_in_4 >= 3)
            season_dates.append(d)
            prev_date, prev_season = d, season

    out: Dict[Tuple[str, str, str], dict] = {}
    for r in rows:
        h = per_team[(r["home"], r["date"])]
        a = per_team[(r["away"], r["date"])]
        out[(r["date"], r["home"], r["away"])] = {
            "home_rest": h[0], "away_rest": a[0],
            "home_b2b": h[1], "away_b2b": a[1],
            "home_3in4": h[2], "away_3in4": a[2],
            "rest_diff": h[0] - a[0],
        }
    return out


# ---------------------------------------------------------------------------
# 5. Optional cache (NEW file only; safe to delete and rebuild)
# ---------------------------------------------------------------------------
def _open_cache(path: str) -> sqlite3.Connection:
    resolved = os.path.normcase(os.path.abspath(path))
    if resolved in PROTECTED_DBS:
        raise ValueError(f"Refusing to write cache into protected DB: {path}")
    return sqlite3.connect(path)


def build_cache(cache_path: str = CACHE_DB,
                seasons: Sequence[str] = ODDS_SEASONS,
                ks: Sequence[int] = (10, 20, 30),
                store: Optional[SnapshotStore] = None) -> dict:
    """Materialize rolling features (all K in `ks`, both teams of every
    odds game), pre-game Elo, and rest features into `cache_path` (a NEW
    sqlite file). Existing cache tables are replaced. Returns row counts."""
    if store is None:
        store = _default_store()
    games = load_odds_games(seasons=seasons)
    elo = build_elo_odds(start_season=seasons[0], end_season=seasons[-1])
    rest = build_rest_features(games)

    conn = _open_cache(cache_path)
    try:
        cur = conn.cursor()
        stat_cols = list(COUNT_STATS) + list(PCT_STATS) + ["WIN_PCT"]
        cols_sql = ", ".join(f'"{c}" REAL' for c in stat_cols)
        cur.execute("DROP TABLE IF EXISTS rolling_features")
        cur.execute(
            f"CREATE TABLE rolling_features (season TEXT, date TEXT, "
            f"team TEXT, k INTEGER, window INTEGER, is_partial INTEGER, "
            f"exact_window INTEGER, gp INTEGER, snapshot_date TEXT, "
            f"{cols_sql}, PRIMARY KEY (date, team, k))")
        cur.execute("DROP TABLE IF EXISTS elo_pre")
        cur.execute("CREATE TABLE elo_pre (date TEXT, home TEXT, away TEXT, "
                    "home_elo REAL, away_elo REAL, home_expected REAL, "
                    "PRIMARY KEY (date, home, away))")
        cur.execute("DROP TABLE IF EXISTS rest_features")
        cur.execute("CREATE TABLE rest_features (date TEXT, home TEXT, "
                    "away TEXT, home_rest INTEGER, away_rest INTEGER, "
                    "home_b2b INTEGER, away_b2b INTEGER, home_3in4 INTEGER, "
                    "away_3in4 INTEGER, rest_diff INTEGER, "
                    "PRIMARY KEY (date, home, away))")

        n_roll = 0
        seen = set()
        for g in games:
            for team in (g["home"], g["away"]):
                for k in ks:
                    key = (g["date"], team, k)
                    if key in seen:
                        continue
                    seen.add(key)
                    r = build_rolling_features(k, g["date"], team, store)
                    if r is None:
                        continue
                    vals = [r["stats"].get(c) for c in stat_cols]
                    cur.execute(
                        f"INSERT OR REPLACE INTO rolling_features VALUES "
                        f"({','.join('?' * (9 + len(stat_cols)))})",
                        [r["season"], g["date"], team, k, r["window"],
                         int(r["is_partial"]), int(r["exact_window"]),
                         r["gp"], r["snapshot_date"]] + vals)
                    n_roll += 1
        for (date, home, away), v in elo["pre_game"].items():
            cur.execute("INSERT OR REPLACE INTO elo_pre VALUES (?,?,?,?,?,?)",
                        [date, home, away, v["home_elo"], v["away_elo"],
                         v["home_expected"]])
        for (date, home, away), v in rest.items():
            cur.execute(
                "INSERT OR REPLACE INTO rest_features VALUES "
                "(?,?,?,?,?,?,?,?,?,?)",
                [date, home, away, v["home_rest"], v["away_rest"],
                 int(v["home_b2b"]), int(v["away_b2b"]),
                 int(v["home_3in4"]), int(v["away_3in4"]), v["rest_diff"]])
        conn.commit()
        return {"rolling_rows": n_roll, "elo_rows": len(elo["pre_game"]),
                "rest_rows": len(rest), "games": len(games)}
    finally:
        conn.close()
