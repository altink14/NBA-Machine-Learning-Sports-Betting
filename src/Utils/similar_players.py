"""Similar players: the closest whole-season stat lines in the archive.

Replaces the frontend's in-browser matcher (found 2026-09-24 in the nav
audit), which said "the ten closest stat lines in the archive" but searched
ONE season, min-max scaled every stat so a single outlier squeezed the whole
league into a narrow band (every match read 91-94% "Highly similar"), turned
missing values into 0, and printed "12 stats" beside a list of six.

The method, every piece of which the endpoint returns so the page can print it:

  * One row per player-SEASON, whole season. player_season_totals is stored
    per team (one row per stint), so stints are summed and shooting rates come
    from makes and attempts, the same rule as main_api._player_season_line.
    Usage is nba.com's whole-season row in player_season_stats.
  * Regular season only, every season in the archive.
  * The pool: players with at least 25% of their team's games (scaled to the
    season's length: 21 of 82, 13 of the 50-game 1998-99 season) and 10
    minutes a game. Below that a per-game line is mostly noise.
  * Each stat is turned into a z-score WITHIN ITS SEASON (how many standard
    deviations from that season's pool average). That is what lets a 1998
    guard be compared with a 2024 guard: 3 threes a game meant a high-volume
    shooter then and a low-volume one now.
  * The gap between two player-seasons is the weighted root-mean-square of
    their z-score differences, over the stats BOTH have. A stat that is
    unknown for either side is left out of that pair's gap and the weights
    are renormalised; it is never treated as 0. A pair that shares less than
    80% of the total weight is not compared at all.
  * The gap's unit is one standard deviation: 0 is an identical line relative
    to its season, 1.0 means that on a typical stat the two lines sit a full
    season-SD apart.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEASON_TYPE = "Regular Season"
MIN_GAME_SHARE = 0.25
MIN_MPG = 10.0
MIN_SHARED_WEIGHT = 0.80

# (key, label, weight, how it is computed). The ONE list: the endpoint
# returns it and the page prints it, so the copy cannot drift from the math.
STATS: List[Tuple[str, str, float, str]] = [
    ("pts", "Points per game", 1.5, "total points / games"),
    ("ast", "Assists per game", 1.5, "total assists / games"),
    ("reb", "Rebounds per game", 1.2, "total rebounds / games"),
    ("usg_pct", "Usage rate", 1.2, "share of team plays he finished while on the floor (nba.com, whole season)"),
    ("min", "Minutes per game", 1.0, "total minutes / games"),
    ("ts_pct", "True shooting %", 1.0, "points / (2 x (FGA + 0.44 x FTA)), from season totals"),
    ("fg3a", "3-point attempts per game", 1.0, "total 3PA / games"),
    ("fta", "Free-throw attempts per game", 0.7, "total FTA / games"),
    ("stl", "Steals per game", 0.8, "total steals / games"),
    ("blk", "Blocks per game", 0.8, "total blocks / games"),
    ("tov", "Turnovers per game", 0.6, "total turnovers / games"),
]
KEYS = [s[0] for s in STATS]
WEIGHTS = np.array([s[2] for s in STATS], dtype=float)

# Plain-English bands for the gap (in season standard deviations). Returned
# with the result so the page prints the same thresholds it colours by.
# Set 2026-09-27 from 400 random pool player-seasons: the median one's
# closest match anywhere in the archive sat about 0.23 away and its tenth
# closest about 0.32; stars, being outliers, sit further from everyone. The
# pool's own typical figures are measured live (Pool.typical) and printed.
BANDS: List[Tuple[float, str]] = [
    (0.20, "Near-identical"),
    (0.35, "Very close"),
    (0.50, "Close"),
    (0.75, "Loosely similar"),
    (math.inf, "Different"),
]


def band(gap: float) -> str:
    for upper, label in BANDS:
        if gap < upper:
            return label
    return BANDS[-1][1]


_POOL_SQL = """
    SELECT t.player_id, t.season, SUM(t.gp) AS gp, SUM(t.min) AS min,
           SUM(t.fga) AS fga, SUM(t.fta) AS fta, SUM(t.fg3a) AS fg3a,
           SUM(t.pts) AS pts, SUM(t.reb) AS reb, SUM(t.ast) AS ast,
           SUM(t.stl) AS stl, SUM(t.blk) AS blk, SUM(t.tov) AS tov,
           COUNT(*) AS n_teams, MIN(t.team_id) AS team_id,
           (SELECT s.usg_pct FROM player_season_stats s
             WHERE s.player_id = t.player_id AND s.season = t.season AND s.season_type = t.season_type
             ORDER BY s.gp DESC LIMIT 1) AS usg_pct,
           (SELECT MAX(o.player_age) FROM player_career_official o
             WHERE o.player_id = t.player_id AND o.season = t.season
               AND o.season_type = 'Regular Season' AND o.is_career_total = 0) AS age,
           p.full_name
    FROM player_season_totals t
    JOIN players p ON p.player_id = t.player_id
    WHERE t.season_type = ?
    GROUP BY t.player_id, t.season
"""

_TEAM_GAMES_SQL = """
    SELECT season, MAX(n) FROM (
        SELECT season, team, COUNT(*) AS n FROM (
            SELECT season, home_team_id AS team FROM box_scores WHERE season_type = ?
            UNION ALL
            SELECT season, away_team_id FROM box_scores WHERE season_type = ?
        ) GROUP BY season, team
    ) GROUP BY season
"""


def _per_game(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    gp = row.get("gp") or 0
    out: Dict[str, Optional[float]] = {}
    for k in ("pts", "ast", "reb", "min", "fg3a", "fta", "stl", "blk", "tov"):
        v = row.get(k)
        out[k] = (v / gp) if (gp and v is not None) else None
    fga, fta, pts = row.get("fga"), row.get("fta"), row.get("pts")
    den = 2.0 * ((fga or 0) + 0.44 * (fta or 0))
    out["ts_pct"] = (pts / den) if (pts is not None and den > 0) else None
    usg = row.get("usg_pct")
    # usg 0.0 is the backfill's never-filled placeholder shape (see
    # main_api._placeholder_free); a player who played has a usage above 0.
    out["usg_pct"] = usg if usg else None
    return out


class Pool:
    """Every archived regular-season player-season, with per-season z-scores."""

    def __init__(self, rows: List[Dict[str, Any]], team_games: Dict[str, int]):
        self.rows = rows
        self.team_games = team_games
        n = len(rows)
        self.values = np.full((n, len(KEYS)), np.nan)
        for i, r in enumerate(rows):
            pg = _per_game(r)
            r["per_game"] = pg
            for j, k in enumerate(KEYS):
                if pg[k] is not None:
                    self.values[i, j] = pg[k]
        self.season = np.array([r["season"] for r in rows])
        self.player = np.array([r["player_id"] for r in rows])
        gp = np.array([r["gp"] or 0 for r in rows], dtype=float)
        mpg = np.where(gp > 0, np.array([r["min"] or 0 for r in rows], dtype=float) / np.maximum(gp, 1), 0)
        floor = np.array([math.ceil(MIN_GAME_SHARE * team_games.get(s, 82)) for s in self.season], dtype=float)
        self.qualified = (gp >= floor) & (mpg >= MIN_MPG)
        self.z = np.full_like(self.values, np.nan)
        self.pct = np.full_like(self.values, np.nan)
        self.season_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for s in np.unique(self.season):
            in_s = self.season == s
            ref = in_s & self.qualified
            stats_s: Dict[str, Tuple[float, float]] = {}
            for j, k in enumerate(KEYS):
                col = self.values[ref, j]
                col = col[~np.isnan(col)]
                if len(col) < 30:
                    continue   # too few to define a season's spread: the stat stays unknown
                mu, sd = float(col.mean()), float(col.std(ddof=0))
                if sd <= 0:
                    continue
                stats_s[k] = (mu, sd)
                self.z[in_s, j] = (self.values[in_s, j] - mu) / sd
                srt = np.sort(col)
                vals = self.values[in_s, j]
                ok = ~np.isnan(vals)
                pc = np.full(vals.shape, np.nan)
                pc[ok] = np.searchsorted(srt, vals[ok], side="right") / len(srt)
                self.pct[in_s, j] = pc
            self.season_stats[str(s)] = stats_s
        self.index = {(r["player_id"], r["season"]): i for i, r in enumerate(rows)}
        self.typical = self._typical()

    def _typical(self, n: int = 300) -> Dict[str, Any]:
        """Median closest and tenth-closest gap for a fixed random sample of
        pool player-seasons, measured on this pool, so the page can say what
        an ordinary gap looks like instead of printing a typed number."""
        q = np.where(self.qualified)[0]
        if len(q) == 0:
            return {"sample": 0, "closest": None, "tenth": None}
        pick = np.random.default_rng(20260927).choice(q, min(n, len(q)), replace=False)
        first, tenth = [], []
        for i in pick:
            gap, shared = gaps_from(self, int(i))
            ok = self.qualified & (self.player != self.player[i]) & (shared >= MIN_SHARED_WEIGHT) & ~np.isnan(gap)
            best = _best_per_player(self, np.where(ok)[0], gap, 10)
            if len(best) >= 10:
                first.append(gap[best[0]])
                tenth.append(gap[best[9]])
        return {"sample": len(first),
                "closest": round(float(np.median(first)), 3) if first else None,
                "tenth": round(float(np.median(tenth)), 3) if tenth else None}

    @property
    def seasons(self) -> List[str]:
        return sorted(self.season_stats)


def load_pool(conn) -> Pool:
    conn_rows = conn.execute(_POOL_SQL, (SEASON_TYPE,)).fetchall()
    cols = ["player_id", "season", "gp", "min", "fga", "fta", "fg3a", "pts", "reb", "ast",
            "stl", "blk", "tov", "n_teams", "team_id", "usg_pct", "age", "full_name"]
    rows = [dict(zip(cols, tuple(r))) for r in conn_rows]
    team_games = {s: int(n) for s, n in conn.execute(_TEAM_GAMES_SQL, (SEASON_TYPE, SEASON_TYPE)).fetchall() if n}
    return Pool(rows, team_games)


_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"key": None, "pool": None}


# Ages come from player_career_official, which the reference backfill fills
# for hours at a time; folding its row count into the fingerprint would
# rebuild the pool on nearly every request while it runs. An hourly rebuild
# picks new ages up instead.
AGE_REFRESH_SECONDS = 3600


def _fingerprint(conn) -> tuple:
    # Changes whenever a backfill adds or rewrites season rows, so the pool
    # is rebuilt after new stats land (a build takes about a second).
    a = conn.execute("SELECT COUNT(*), MAX(id) FROM player_season_totals").fetchone()
    b = conn.execute("SELECT COUNT(*), MAX(id) FROM player_season_stats").fetchone()
    return tuple(a) + tuple(b)


def get_pool(conn) -> Pool:
    key = _fingerprint(conn)
    now = time.time()
    with _cache_lock:
        if (_cache["key"] == key and _cache["pool"] is not None
                and now - _cache.get("built", 0.0) < AGE_REFRESH_SECONDS):
            return _cache["pool"]
    pool = load_pool(conn)
    with _cache_lock:
        _cache.update(key=key, pool=pool, built=now)
    return pool


def gaps_from(pool: Pool, i: int) -> Tuple[np.ndarray, np.ndarray]:
    """(gap, shared_weight_share) from player-season i to every row."""
    zt = pool.z[i]
    diff = pool.z - zt
    known = ~np.isnan(diff)
    w = WEIGHTS * known
    wsum = w.sum(axis=1)
    sq = np.where(known, diff * diff, 0.0) @ WEIGHTS
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.sqrt(sq / wsum)
    return gap, wsum / WEIGHTS.sum()


def _best_per_player(pool: Pool, idx: np.ndarray, gap: np.ndarray, limit: int) -> List[int]:
    """Row indices of the `limit` smallest gaps, one (the closest) per player."""
    idx = idx[np.argsort(gap[idx], kind="stable")]
    seen: set = set()
    out: List[int] = []
    for k in idx:
        pid = int(pool.player[k])
        if pid in seen:
            continue
        seen.add(pid)
        out.append(int(k))
        if len(out) >= limit:
            break
    return out


def _line(pool: Pool, i: int) -> Dict[str, Any]:
    r = pool.rows[i]
    pg = r["per_game"]
    return {
        "player_id": int(r["player_id"]),
        "player_name": r["full_name"],
        "season": r["season"],
        "gp": int(r["gp"] or 0),
        "age": int(r["age"]) if r.get("age") is not None else None,
        "n_teams": int(r["n_teams"]),
        "team_id": int(r["team_id"]) if r["n_teams"] == 1 else None,
        "qualified": bool(pool.qualified[i]),
        "stats": {k: (None if pg[k] is None else round(float(pg[k]), 4)) for k in KEYS},
        "z": {k: (None if np.isnan(pool.z[i, j]) else round(float(pool.z[i, j]), 2)) for j, k in enumerate(KEYS)},
        "pct": {k: (None if np.isnan(pool.pct[i, j]) else round(float(pool.pct[i, j]), 3)) for j, k in enumerate(KEYS)},
    }


def find_similar(pool: Pool, player_id: int, season: str, scope: str = "all", limit: int = 10) -> Optional[Dict[str, Any]]:
    """The `limit` closest qualified player-seasons to (player_id, season).

    One entry per other player (his closest season), never the target player
    himself. scope="season" keeps the search inside the target's season.
    Returns None when the player has no regular-season line that season."""
    i = pool.index.get((player_id, season))
    if i is None:
        return None
    gap, shared = gaps_from(pool, i)
    ok = pool.qualified & (pool.player != player_id) & (shared >= MIN_SHARED_WEIGHT) & ~np.isnan(gap)
    if scope == "season":
        ok &= pool.season == season
    matches: List[Dict[str, Any]] = []
    for k in _best_per_player(pool, np.where(ok)[0], gap, limit):
        line = _line(pool, k)
        g = float(gap[k])
        line["gap"] = round(g, 3)
        line["band"] = band(g)
        line["shared_weight"] = round(float(shared[k]), 3)
        matches.append(line)
    searched = pool.seasons if scope == "all" else [season]
    n_pool = int(pool.qualified.sum()) if scope == "all" else int((pool.qualified & (pool.season == season)).sum())
    return {
        "target": _line(pool, i),
        "matches": matches,
        "scope": scope,
        "seasons_searched": {"first": searched[0], "last": searched[-1], "count": len(searched)},
        "pool_size": n_pool,
        # Measured over the whole archive; only meaningful beside scope="all".
        "typical": pool.typical if scope == "all" else None,
    }


def method() -> Dict[str, Any]:
    return {
        "season_type": SEASON_TYPE,
        "stats": [{"key": k, "label": lbl, "weight": w, "formula": f} for k, lbl, w, f in STATS],
        "pool_rule": (f"Regular season only. A player-season is in the pool with at least "
                      f"{int(MIN_GAME_SHARE * 100)}% of his team's games (scaled to the season's length) "
                      f"and {MIN_MPG:g} minutes a game. Traded players count as one whole season."),
        "scaling": ("Each stat becomes a z-score within its own season: how many standard deviations "
                    "above or below that season's pool average. Eras compare fairly: a line is judged "
                    "against the league it played in."),
        "gap": ("Weighted root-mean-square of the z-score differences, over the stats both lines have. "
                "0 = identical relative to their seasons; 1.0 = a typical stat a full season-SD apart. "
                "A stat unknown for either line is left out, never counted as 0."),
        "bands": [{"below": (None if math.isinf(u) else u), "label": lbl} for u, lbl in BANDS],
        "one_per_player": "Each other player appears once, with his closest season. The player himself is excluded.",
        "age_source": "nba.com's career table (player_career_official), where fetched; otherwise unknown.",
    }
