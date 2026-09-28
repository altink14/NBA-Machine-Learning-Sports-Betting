"""
nba_mirror.py
=============
Copies the league-wide stats.nba.com tables that pages read at request time
into `nba_response_mirror` (in TeamData.sqlite), so the public server, which
cannot reach stats.nba.com and is given only the database, can still serve
them. Written 2026-09-28 for the no-nba.com production mode (NBA_STATS_LIVE=off,
see nba_stats_client.py).

WHAT IS COPIED. Exactly the requests the routes make, recorded by calling the
client's own methods against a stub, so a key here can never drift from the
key a route asks for:

  /api/schedule, /api/cup   scheduleleaguev2 (one per season)
  /api/playtypes/teams      synergyplaytypes (11 play types x offense/defense)
  /api/rebounding, 2K DNA   leaguedashptstats (Rebounding)
  /api/shot-quality, 2K DNA leaguedashplayerptshot (8 defender/range cuts)
  /api/stats/hustle         leaguehustlestatsplayer
  /api/lineups              leaguedashlineups (2-, 3-, 4-, 5-man)
  shot-chart zone averages  shotchartleagueavg, TRIMMED to its 20-row
                            LeagueAverages set (the full answer is ~26 MB of
                            every shot in the league; the averages are ~3 KB)

Per-player requests (matchups, a player's own shot chart) are not copied:
hundreds a season, and the server answers those from pbp_events or says
they are live-only.

TWO WAYS IN.
  --from-cache   copy what Data/nba_cache already holds. No network at all.
                 A season's copy is only taken if it was fetched after that
                 season-type's last game (a mid-season copy is a partial
                 table, and would be served as the whole season).
  (default)      the daily refresh on the home PC (refresh_registry.py job
                 "server_mirror"): the current season's tables through the
                 normal client (disk cache first, then stats.nba.com), and the
                 schedule season's schedule. A season-type already stored as
                 final is never asked for again, so the offseason costs nothing.

A mirrored row carries fetched_at; it is as current as the rest of the
snapshot the server was given, and never presented as newer.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.Utils import nba_stats_client as nsc

logger = logging.getLogger(__name__)

# Mirrors main_api._PLAY_TYPES (a test holds the two lists equal).
PLAY_TYPES = [
    "Transition", "Isolation", "PRBallHandler", "PRRollman", "Postup",
    "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc",
]
SEASON_TYPES = ("Regular Season", "Playoffs")
# First season each table exists on nba.com (the frontend's floors in
# src/lib/archive-seasons.ts). --all-seasons asks nothing before them.
FLOORS = {
    "synergyplaytypes": "2012-13",
    "leaguedashptstats": "2013-14",
    "leaguedashplayerptshot": "2013-14",
    "leaguehustlestatsplayer": "2016-17",
    "leaguedashlineups": "2007-08",
    "shotchartleagueavg": "1996-97",
}
LINEUP_SIZES = (2, 3, 4, 5)
# A season-type is over once it has gone this long without a game. Longer
# than the All-Star break (the longest in-season gap, about a week).
FINAL_AFTER_DAYS = 10

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {nsc.MIRROR_TABLE} (
    cache_key   TEXT PRIMARY KEY,   -- nba_stats_client.cache_key(endpoint, params)
    endpoint    TEXT NOT NULL,
    params      TEXT NOT NULL,      -- JSON, for people reading the table
    season      TEXT,
    season_type TEXT,
    fetched_at  TEXT NOT NULL,      -- when stats.nba.com answered (UTC ISO)
    final       INTEGER NOT NULL DEFAULT 0,  -- 1 = fetched after the season-type ended
    source      TEXT NOT NULL,      -- 'live' or 'disk-cache'
    bytes       INTEGER,
    payload     BLOB NOT NULL       -- gzip(JSON), the response as the client returns it
)
"""


class _Recorder(nsc.NBAStatsClient):
    """Calls nothing: records the (endpoint, class, params, ttl) each method asks for."""

    def __init__(self):
        super().__init__()
        self.calls: List[Tuple[str, Any, Dict[str, Any], Optional[int]]] = []

    def _fetch(self, endpoint_name, endpoint_class, params, ttl=nsc._LIVE_TTL_SECONDS):
        self.calls.append((endpoint_name, endpoint_class, dict(params), ttl))
        return {}


def season_targets(season: str, season_type: str) -> List[Tuple[str, Any, Dict[str, Any], Optional[int]]]:
    """Every league-wide request the routes make for one season-type."""
    from src.Utils.ShotQuality import DEF_BUCKETS
    rec = _Recorder()
    for pt in PLAY_TYPES:
        for grouping in ("offensive", "defensive"):
            rec.synergy_play_types(season, season_type, grouping, pt)
    rec.league_pt_stats(season, season_type, "Rebounding")
    for label, _ in DEF_BUCKETS:
        rec.player_pt_shots(season, season_type, close_def_dist_range=label)
        rec.player_pt_shots(season, season_type, close_def_dist_range=label,
                            general_range="Less Than 10 ft")
    rec.league_hustle_stats(season=season, season_type=season_type)
    for n in LINEUP_SIZES:
        rec.league_dash_lineups(season=season, season_type=season_type,
                                group_quantity=n, measure_type="Advanced")
    rec.league_shot_averages(season, season_type)
    return rec.calls


def schedule_target(season: str):
    rec = _Recorder()
    rec.schedule_league_v2(season=season)
    return rec.calls[0]


def trim(endpoint: str, data: Dict) -> Dict:
    """The part of a response the server needs. Only the zone-average call is
    cut: the client parses nothing but its LeagueAverages set."""
    if endpoint == "shotchartleagueavg":
        sets = [rs for rs in (data.get("resultSets") or []) if rs.get("name") == "LeagueAverages"]
        return {"resultSets": sets}
    return data


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)


def last_game_dates(conn: sqlite3.Connection) -> Dict[Tuple[str, str], str]:
    """{(season, season_type): last game date (YYYY-MM-DD)} from our archive."""
    out = {}
    for season, st, last in conn.execute(
        "SELECT season, season_type, MAX(DATE(game_date)) FROM box_scores GROUP BY 1, 2"
    ):
        if last:
            out[(season, st)] = last
    return out


def is_final(last_game: Optional[str], fetched: datetime, today: date) -> bool:
    """Fetched after the season-type's last game, and that game is long enough
    ago that the season-type is over."""
    if not last_game:
        return False
    last = date.fromisoformat(last_game)
    return (today - last).days >= FINAL_AFTER_DAYS and fetched.date() > last


def _upsert(conn, endpoint, params, season, season_type, fetched_at: datetime,
            final: bool, source: str, data: Dict) -> int:
    blob = gzip.compress(json.dumps(trim(endpoint, data)).encode("utf-8"), compresslevel=6)
    conn.execute(
        f"""
        INSERT INTO {nsc.MIRROR_TABLE}
            (cache_key, endpoint, params, season, season_type, fetched_at, final, source, bytes, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            endpoint=excluded.endpoint, params=excluded.params, season=excluded.season,
            season_type=excluded.season_type, fetched_at=excluded.fetched_at,
            final=excluded.final, source=excluded.source, bytes=excluded.bytes,
            payload=excluded.payload
        """,
        (nsc.cache_key(endpoint, params), endpoint, json.dumps(params, sort_keys=True),
         season, season_type, fetched_at.isoformat(), 1 if final else 0, source,
         len(blob), blob),
    )
    return len(blob)


def _stored(conn, endpoint, params) -> Optional[sqlite3.Row]:
    return conn.execute(
        f"SELECT fetched_at, final FROM {nsc.MIRROR_TABLE} WHERE cache_key = ?",
        (nsc.cache_key(endpoint, params),),
    ).fetchone()


def _all_targets(seasons: Iterable[str], schedule_seasons: Iterable[str]):
    for season in seasons:
        for st in SEASON_TYPES:
            for t in season_targets(season, st):
                yield season, st, t
    for season in schedule_seasons:
        yield season, None, schedule_target(season)


def import_from_cache(conn: sqlite3.Connection, seasons: Iterable[str],
                      schedule_seasons: Iterable[str], today: Optional[date] = None) -> Dict[str, int]:
    """Copy what Data/nba_cache holds into the mirror. No network.

    Season tables are copied only when final (see is_final); a schedule is a
    published plan rather than a running total, so the newest copy is taken
    whatever its age. An existing row is replaced only by a newer copy.
    """
    today = today or date.today()
    lasts = last_game_dates(conn)
    ensure_table(conn)
    counts = {"copied": 0, "not_cached": 0, "partial_skipped": 0, "unreadable": 0,
              "kept_newer": 0, "bytes": 0}
    for season, st, (endpoint, _cls, params, _ttl) in _all_targets(seasons, schedule_seasons):
        cands = nsc.cache_candidates(nsc._cache_path(endpoint, params))
        if not cands:
            counts["not_cached"] += 1
            continue
        path, st_ = cands[0]
        fetched = datetime.fromtimestamp(st_.st_mtime, tz=timezone.utc)
        final = is_final(lasts.get((season, st)), fetched, today) if st else False
        if st and not final:
            counts["partial_skipped"] += 1
            continue
        prev = _stored(conn, endpoint, params)
        if prev and prev[0] >= fetched.isoformat():
            counts["kept_newer"] += 1
            continue
        try:
            data = nsc.load_cache_file(path)
        except Exception as exc:
            logger.warning("Unreadable cache entry %s: %s", path, exc)
            counts["unreadable"] += 1
            continue
        counts["bytes"] += _upsert(conn, endpoint, params, season, st, fetched, final, "disk-cache", data)
        counts["copied"] += 1
    conn.commit()
    return counts


def refresh_live(conn: sqlite3.Connection, season, schedule_season: Optional[str],
                 client: Optional[nsc.NBAStatsClient] = None,
                 today: Optional[date] = None) -> Dict[str, int]:
    """The daily step on the home PC: the current season, through the client.

    A season-type with no games yet (the offseason before opening night, the
    regular season before the playoffs) is skipped: there is no table to copy.
    One failed request never stops the rest.
    """
    today = today or date.today()
    client = client or nsc.get_client()
    lasts = last_game_dates(conn)
    ensure_table(conn)
    counts = {"copied": 0, "already_final": 0, "no_games_yet": 0, "failed": 0, "bytes": 0}
    seasons = [season] if isinstance(season, str) else list(season)
    for s, st, (endpoint, cls, params, ttl) in _all_targets(
            seasons, [schedule_season] if schedule_season else []):
        if st and (s, st) not in lasts:
            counts["no_games_yet"] += 1
            continue
        if st and s < FLOORS.get(endpoint, ""):
            continue   # the table did not exist yet: nothing to copy
        prev = _stored(conn, endpoint, params)
        if prev and prev[1]:
            counts["already_final"] += 1
            continue
        try:
            data = client._fetch(endpoint, cls, params, ttl=ttl)
        except Exception as exc:
            logger.warning("Mirror refresh: %s %s failed: %s", endpoint, params, exc)
            counts["failed"] += 1
            continue
        now = datetime.now(timezone.utc)
        final = is_final(lasts.get((s, st)), now, today) if st else False
        counts["bytes"] += _upsert(conn, endpoint, params, s, st, now, final, "live", data)
        counts["copied"] += 1
        conn.commit()   # per row: a later failure must not lose earlier copies
    return counts


def schedule_season_for(today: date, newest_archived: str) -> str:
    """The season whose schedule /api/schedule shows: main_api._schedule_season's
    rule (from July 1 of a season's end year, the next season's), anchored on
    the newest archived season, which is CURRENT_SEASON once its first game lands."""
    start = int(newest_archived[:4])
    if today >= date(start + 1, 7, 1):
        start += 1
    return f"{start}-{(start + 1) % 100:02d}"


def _db_path() -> str:
    return str(nsc._mirror_db_path())


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-cache", action="store_true",
                    help="copy from Data/nba_cache only; makes no network request")
    ap.add_argument("--all-seasons", action="store_true",
                    help="one-off on the home PC: every archived season from each table's "
                         "floor (~900 stats.nba.com requests, resumable: final rows are skipped)")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(_db_path(), timeout=60)
    try:
        seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM box_scores ORDER BY 1")]
        newest = seasons[-1]
        if args.from_cache:
            nxt = int(newest[:4]) + 1
            counts = import_from_cache(conn, seasons,
                                       seasons[-2:] + [f"{nxt}-{(nxt + 1) % 100:02d}"])
        else:
            if not nsc.live_fetch_enabled():
                print("NBA_STATS_LIVE is off: the live refresh runs on the home PC only.")
                return 1
            counts = refresh_live(conn, seasons if args.all_seasons else newest,
                                  schedule_season_for(date.today(), newest))
        print(json.dumps(counts))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
