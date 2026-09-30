"""
ingest_game_summaries.py
========================
Line scores, inactive players and game info from the boxscoresummaryv2
responses ALREADY ON DISK. No network calls: this reads the permanent cache
that backfill_officials.py filled (one JSON per game, 2003-04 onward) and
writes three tables. Run it any time after the officials backfill; it is
idempotent (INSERT OR REPLACE) and takes a few minutes for ~29,000 files.

Tables
------
game_line_scores   one row per team per game: points by quarter and up to
                   ten overtimes, plus the final. This is nba.com's own line
                   score, so it also covers games with no play-by-play.
game_inactives     one row per inactive player per game (the "Inactive:" line
                   on a box score). Empty for most games before 2005-06; the
                   feed simply did not carry it then. Missing means unknown,
                   not "everyone played".
game_info          attendance, game duration ("2:13" = 2 h 13 min), national
                   TV broadcaster, home and visitor ids, and `source`: which
                   summary the game's three tables came from.

v2 went empty on 2025-04-10: for every game after it nba.com returns the
pre-game shell (GAME_STATUS_ID 1, the ORIGINAL schedule date, no attendance,
"0:00" duration, a line score of NULLs, no inactives) -- 1,203 of 2025-26's
regular-season games. boxscoresummaryv3, cached by backfill_officials.py's v3
fall-through, has all of it. So a v2 shell falls through to the cached v3
summary when there is one (`source` = 'boxscoresummaryv3'); otherwise the
shell is written as before and read as unknown. Checked 2026-09-29 on the two
games cached from both endpoints with a real v2 answer (0020700757,
0020701069): attendance, duration, national TV, every quarter and the
inactive list identical.

Usage:
    python src/Process-Data/ingest_game_summaries.py            # everything cached
    python src/Process-Data/ingest_game_summaries.py --limit 500
    python src/Process-Data/ingest_game_summaries.py --dry-run  # count, write nothing
    python src/Process-Data/ingest_game_summaries.py --season 2026-27

The daily job runs the last form for the newest season, right after the
officials step caches each new game's v3 summary (refresh_registry.py,
job "game_summaries").
"""

import argparse
import glob
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Utils.nba_stats_client import NBAStatsClient, cache_candidates, load_cache_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_game_summaries")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
CACHE_DIR = os.path.join(REPO_ROOT, "Data", "nba_cache")
SUMMARY_PREFIX = "boxscoresummaryv2_game_id="
V3_PREFIX = "boxscoresummaryv3_game_id="


def cached_summaries(cache_dir: str = CACHE_DIR, prefix: str = SUMMARY_PREFIX):
    """(game_id, plain .json path) for every cached summary, sorted by game id.

    The cache holds each entry as `<key>.json` (older writes) or
    `<key>.json.gz` (newer writes, and anything compact_nba_cache.py has
    converted); a key can briefly have both. Each game appears once, under its
    plain name - the reader picks the copy to open.
    """
    gids = set()
    for pattern in ("*.json", "*.json.gz"):
        for path in glob.glob(os.path.join(cache_dir, prefix + pattern)):
            name = os.path.basename(path)
            if name.endswith(".gz"):
                name = name[: -len(".gz")]
            gids.add(name[len(prefix):-len(".json")])
    return [(g, os.path.join(cache_dir, f"{prefix}{g}.json")) for g in sorted(gids)]


def load_summary(plain: str):
    """The newest readable copy of one cached summary; raises if none is."""
    last = FileNotFoundError(plain)
    for candidate, _ in cache_candidates(plain):
        try:
            return load_cache_file(candidate)
        except Exception as exc:  # a corrupt .gz falls back to a plain copy, if any
            last = exc
    raise last

SCHEMA = """
CREATE TABLE IF NOT EXISTS game_line_scores (
    game_id    TEXT NOT NULL,
    team_id    INTEGER NOT NULL,
    team_abbr  TEXT,
    q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER,
    ot1 INTEGER, ot2 INTEGER, ot3 INTEGER, ot4 INTEGER, ot5 INTEGER,
    ot6 INTEGER, ot7 INTEGER, ot8 INTEGER, ot9 INTEGER, ot10 INTEGER,
    pts INTEGER,
    PRIMARY KEY (game_id, team_id)
);
CREATE TABLE IF NOT EXISTS game_inactives (
    game_id    TEXT NOT NULL,
    team_id    INTEGER,
    team_abbr  TEXT,
    player_id  INTEGER NOT NULL,
    first_name TEXT,
    last_name  TEXT,
    jersey_num TEXT,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_game_inactives_player ON game_inactives(player_id);
CREATE TABLE IF NOT EXISTS game_info (
    game_id          TEXT PRIMARY KEY,
    attendance       INTEGER,
    game_time        TEXT,
    natl_tv          TEXT,
    home_team_id     INTEGER,
    visitor_team_id  INTEGER,
    ingested_at      TEXT,
    source           TEXT     -- boxscoresummaryv2 | boxscoresummaryv3
);
"""

OT_COLS = [f"PTS_OT{i}" for i in range(1, 11)]
SOURCE_V2 = "boxscoresummaryv2"
SOURCE_V3 = "boxscoresummaryv3"


def _int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def v2_is_shell(parsed) -> bool:
    """A v2 answer that is the pre-game placeholder, not the game: not final,
    or a line score without points. Every game after 2025-04-10 is one."""
    summ = (parsed.get("GameSummary") or [{}])[0]
    lines = parsed.get("LineScore") or []
    return (_int(summ.get("GAME_STATUS_ID")) != 3 or not lines
            or any(_int(r.get("PTS")) is None for r in lines))


def rows_from_v2(gid, parsed, now):
    """(line rows, inactive rows, info row or None) from a parsed v2 summary."""
    lines = []
    for r in parsed.get("LineScore") or []:
        if _int(r.get("TEAM_ID")) is None:
            continue  # a handful of rows carry no team id; nothing to attach them to
        lines.append((
            gid, _int(r.get("TEAM_ID")), r.get("TEAM_ABBREVIATION"),
            _int(r.get("PTS_QTR1")), _int(r.get("PTS_QTR2")), _int(r.get("PTS_QTR3")), _int(r.get("PTS_QTR4")),
            *[_int(r.get(c)) for c in OT_COLS],
            _int(r.get("PTS")),
        ))
    inactives = []
    for r in parsed.get("InactivePlayers") or []:
        pid = _int(r.get("PLAYER_ID"))
        if pid is None:
            continue
        inactives.append((
            gid, _int(r.get("TEAM_ID")), r.get("TEAM_ABBREVIATION"), pid,
            r.get("FIRST_NAME"), r.get("LAST_NAME"),
            (r.get("JERSEY_NUM") or "").strip() or None,
        ))
    info = (parsed.get("GameInfo") or [{}])[0]
    summ = (parsed.get("GameSummary") or [{}])[0]
    info_row = None
    if info or summ:
        info_row = (
            gid, _int(info.get("ATTENDANCE")), (info.get("GAME_TIME") or "").strip() or None,
            summ.get("NATL_TV_BROADCASTER_ABBREVIATION"),
            _int(summ.get("HOME_TEAM_ID")), _int(summ.get("VISITOR_TEAM_ID")), now, SOURCE_V2,
        )
    return lines, inactives, info_row


def rows_from_v3(gid, raw, now, tv_fallback=None):
    """The same three row shapes from a cached v3 summary, or (None, why) when
    it cannot stand in for the game: not final, about another game, or a line
    score whose periods do not add up to the final. A wrong line score is
    worse than none, so nothing is written from such a file.

    National TV is v3's national TV list joined with '/', v2's own style
    ('TNT/truTV/Max'). When v3 lists none, the v2 shell's pre-game value is
    kept (`tv_fallback`): unknown does not overwrite known."""
    s = (raw or {}).get("boxScoreSummary") or {}
    if str(s.get("gameId") or "") != gid:
        return None, f"answers for {s.get('gameId')!r}"
    if _int(s.get("gameStatus")) != 3:
        return None, f"gameStatus {s.get('gameStatus')!r}, not final"
    lines, inactives = [], []
    for t in (s.get("homeTeam") or {}, s.get("awayTeam") or {}):
        tid, score = _int(t.get("teamId")), _int(t.get("score"))
        by_period = {_int(p.get("period")): _int(p.get("score")) for p in t.get("periods") or []}
        if tid is None or score is None or any(by_period.get(i) is None for i in (1, 2, 3, 4)):
            return None, "a team or its four quarters missing"
        ots = sorted(p for p in by_period if p is not None and p > 4)
        if ots != list(range(5, 5 + len(ots))) or len(ots) > 10:
            return None, f"overtime periods {ots}"
        if any(v is None for v in by_period.values()) or sum(by_period.values()) != score:
            return None, f"periods add up to {sum(v or 0 for v in by_period.values())}, final is {score}"
        # v2 writes 0 for an overtime that was not played; so does this.
        lines.append((gid, tid, t.get("teamTricode"), *[by_period[i] for i in (1, 2, 3, 4)],
                      *[by_period.get(4 + i, 0) for i in range(1, 11)], score))
        for p in t.get("inactives") or []:
            pid = _int(p.get("personId"))
            if pid is None:
                continue
            inactives.append((gid, tid, t.get("teamTricode"), pid,
                              (p.get("firstName") or "").strip() or None,
                              (p.get("familyName") or "").strip() or None,
                              (p.get("jerseyNum") or "").strip() or None))
    nat = [b.get("broadcastDisplay") for b in
           (s.get("broadcasters") or {}).get("nationalBroadcasters") or [] if b.get("broadcastDisplay")]
    tv = "/".join(nat) if nat else (tv_fallback if tv_fallback not in (None, "", "TBD") else None)
    info_row = (gid, _int(s.get("attendance")), (s.get("duration") or "").strip() or None, tv,
                _int(s.get("homeTeamId")), _int(s.get("awayTeamId")), now, SOURCE_V3)
    return (lines, inactives, info_row), None


def ensure_tables(conn) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(game_info)")}
    if "source" not in cols:  # a table made before 2026-09-29
        conn.execute("ALTER TABLE game_info ADD COLUMN source TEXT")
        conn.commit()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="Stop after N games (0 = all)")
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--dry-run", action="store_true", help="Read and count; write nothing")
    p.add_argument("--season", help="Only the games box_scores holds for this season, e.g. 2026-27 "
                                    "(the daily job's run; a few seconds instead of minutes)")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db, timeout=60)
    if not args.dry_run:
        ensure_tables(conn)
    if args.season:
        known = {r[0] for r in conn.execute("SELECT game_id FROM box_scores WHERE season = ?",
                                            (args.season,))}
    else:
        known = {r[0] for r in conn.execute("SELECT game_id FROM box_scores")}

    v2_files = dict(cached_summaries(args.cache_dir))
    v3_files = dict(cached_summaries(args.cache_dir, V3_PREFIX))
    gids = sorted(set(v2_files) | set(v3_files))
    if args.season:
        # Other seasons are out of scope, not "missing from the archive".
        gids = [g for g in gids if g in known]
    if args.limit:
        gids = gids[: args.limit]
    logger.info("%d games with a cached summary%s (%d v2, %d v3 in the cache)", len(gids),
                f" in {args.season}" if args.season else "", len(v2_files), len(v3_files))

    now = datetime.now(timezone.utc).isoformat()
    n = {"v2": 0, "v3": 0, "v2_shell_kept": 0, "line_rows": 0, "inactives": 0, "info": 0,
         "skipped_not_in_archive": 0, "unreadable": 0}
    batch_lines, batch_inact, batch_info, resourced = [], [], [], []

    def flush():
        if not args.dry_run:
            # A game re-sourced from v3 takes v3's inactive list whole, not
            # merged into whatever an earlier run wrote for it.
            conn.executemany("DELETE FROM game_inactives WHERE game_id = ?", [(g,) for g in resourced])
            if batch_lines:
                conn.executemany(
                    "INSERT OR REPLACE INTO game_line_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch_lines)
            if batch_inact:
                conn.executemany("INSERT OR REPLACE INTO game_inactives VALUES (?,?,?,?,?,?,?)", batch_inact)
            if batch_info:
                conn.executemany(
                    "INSERT OR REPLACE INTO game_info (game_id, attendance, game_time, natl_tv, "
                    "home_team_id, visitor_team_id, ingested_at, source) VALUES (?,?,?,?,?,?,?,?)", batch_info)
            conn.commit()
        batch_lines.clear(); batch_inact.clear(); batch_info.clear(); resourced.clear()

    for gid in gids:
        if gid not in known:
            n["skipped_not_in_archive"] += 1
            continue
        rows, parsed = None, None
        if gid in v2_files:
            try:
                parsed = NBAStatsClient._parse_all_result_sets(load_summary(v2_files[gid]))
            except Exception as exc:  # a corrupt cache file must not end the run
                n["unreadable"] += 1
                logger.warning("unreadable v2 cache for %s: %s", gid, str(exc)[:80])
        if parsed is not None and not v2_is_shell(parsed):
            rows = rows_from_v2(gid, parsed, now)
            n["v2"] += 1
        elif gid in v3_files:
            tv = None
            if parsed is not None:
                tv = ((parsed.get("GameSummary") or [{}])[0]).get("NATL_TV_BROADCASTER_ABBREVIATION")
            try:
                got, why = rows_from_v3(gid, load_summary(v3_files[gid]), now, tv)
            except Exception as exc:
                got, why = None, f"unreadable: {str(exc)[:80]}"
            if got is not None:
                rows = got
                resourced.append(gid)
                n["v3"] += 1
            else:
                logger.warning("v3 summary for %s not used: %s", gid, why)
        if rows is None and parsed is not None:
            # The shell as before: its NULLs are read as unknown.
            rows = rows_from_v2(gid, parsed, now)
            n["v2_shell_kept"] += 1
        if rows is None:
            continue
        lines, inactives, info_row = rows
        batch_lines.extend(lines)
        batch_inact.extend(inactives)
        n["line_rows"] += len(lines)
        n["inactives"] += len(inactives)
        if info_row:
            batch_info.append(info_row)
            n["info"] += 1

        if len(batch_lines) >= 2000:
            flush()
            logger.info("%s", n)

    flush()
    logger.info("SUMMARY%s %s", " (dry run, nothing written)" if args.dry_run else "",
                " ".join(f"{k}={v}" for k, v in n.items()))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
