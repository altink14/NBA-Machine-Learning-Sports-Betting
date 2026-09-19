"""
ingest_games.py (NFL)
=====================
The spine of the NFL archive: one HTTP download that fills `games`, `teams`,
`market_lines`, `weather` and `rest_travel` for every NFL game since 1999.

SOURCE: the nflverse-data `schedules` release (`games.csv`), licensed
CC-BY-4.0 (https://raw.githubusercontent.com/nflverse/nflverse-data/master/LICENSE.md),
refreshed daily by nflverse automation. We pull from the nflverse-data
RELEASE, not from the upstream `nflverse/nfldata` raw file, because only the
release carries a licence file. Attribution is required and is rendered on
every page that shows this data.

WHAT THE LINES ARE, AND WHAT THEY ARE NOT. `games.csv` carries a closing
spread and total for every game since 1999 and moneylines from 2006, but it
does not say which book quoted them or when. nflverse's own dictionary defines
`spread_line` only as "The spread line for the game". A spot check against an
independent 2021 archive matched the closing column exactly on one game and
within half a point on another, with moneylines from a different book. So we
store them with book = 'consensus' and captured_at = NULL, and every surface
that displays them says "closing line (consensus, book unspecified)". Claiming
a book or a timestamp we do not have would be the same species of dishonesty
as an unbaselined win rate.

SIGN CONVENTION: `spread_line` is the HOME side's expected margin, positive
when the home team is favoured (verified: 1999_01_OAK_GB, home GB, spread_line
9, GB won by 4 and therefore failed to cover). This matches the convention of
the NBA odds dataset in Market.py, so one reader serves both sports.

Idempotent: every write is INSERT OR REPLACE on a natural key, so re-running
after a daily refresh updates changed rows and duplicates nothing.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/ingest_games.py
    venv/Scripts/python.exe src/Sports/nfl/ingest_games.py --season 2024
    venv/Scripts/python.exe src/Sports/nfl/ingest_games.py --cache-only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.ingest_games")

SOURCE = "nflverse-data schedules (CC-BY-4.0)"
ENDPOINT = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
CACHE_PATH = os.path.join(REPO_ROOT, "Data", "nfl_cache", "games.csv")

SPORT = "football"
LEAGUE = "NFL"
EASTERN = ZoneInfo("America/New_York")

#: nflverse `surface` carries trailing spaces and several spellings of the
#: same thing. Normalise on read; keep the raw value in the venues table.
SURFACE_MAP = {
    "grass": "grass", "astroturf": "turf", "a_turf": "turf", "fieldturf": "turf",
    "sportturf": "turf", "matrixturf": "turf", "astroplay": "turf", "dessograss": "hybrid",
}


def _i(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _f(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def team_id(abbrev: str) -> str:
    return f"nfl-{abbrev}"


def to_utc(gameday: Optional[str], gametime: Optional[str]) -> Optional[str]:
    """nflverse `gametime` is Eastern clock time; 1999 rows have none."""
    if not gameday or not gametime:
        return None
    try:
        naive = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return naive.replace(tzinfo=EASTERN).astimezone(timezone.utc).isoformat()


def fetch_csv(cache_only: bool = False) -> str:
    """Download the release asset, caching it so a re-run costs nothing."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if cache_only:
        if not os.path.exists(CACHE_PATH):
            raise SystemExit(f"--cache-only but no cache at {CACHE_PATH}")
        logger.info("Reading cached %s", CACHE_PATH)
        return open(CACHE_PATH, encoding="utf-8").read()

    import urllib.request
    logger.info("Downloading %s", ENDPOINT)
    req = urllib.request.Request(ENDPOINT, headers={"User-Agent": "BettingBuddy/1.0 (archive ingest)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    with open(CACHE_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    logger.info("Cached %d bytes to %s", len(raw), CACHE_PATH)
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the nflverse schedules release into the NFL archive.")
    ap.add_argument("--season", help="Only this season, e.g. 2024.")
    ap.add_argument("--cache-only", action="store_true", help="Use the cached CSV, do not download.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    text = fetch_csv(args.cache_only)
    rows = list(csv.DictReader(io.StringIO(text)))
    if args.season:
        rows = [r for r in rows if r.get("season") == str(args.season)]
    logger.info("%d rows to ingest", len(rows))
    if not rows:
        logger.error("Nothing to ingest.")
        return 1

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row
    ensure_core_schema(conn)

    games: List[tuple] = []
    lines: List[tuple] = []
    wx: List[tuple] = []
    rest: List[tuple] = []
    comps: Dict[str, tuple] = {}
    teams: Dict[str, Dict[str, Any]] = {}
    n_lines = n_wx = n_rest = 0

    for r in rows:
        gid = _s(r.get("game_id"))
        season = _s(r.get("season"))
        stype = _s(r.get("game_type")) or "REG"
        home, away = _s(r.get("home_team")), _s(r.get("away_team"))
        if not (gid and season and home and away):
            continue
        gameday = _s(r.get("gameday"))
        gametime = _s(r.get("gametime"))
        hs, as_ = _i(r.get("home_score")), _i(r.get("away_score"))
        comp_id = f"nfl-{season}-{stype}"

        comps.setdefault(comp_id, (comp_id, SPORT, LEAGUE, season, stype, None, None,
                                   SOURCE, ENDPOINT, fetched_at, INGEST_VERSION))
        for ab in (home, away):
            t = teams.setdefault(ab, {"first": season, "last": season})
            t["first"] = min(t["first"], season)
            t["last"] = max(t["last"], season)

        games.append((
            gid, SPORT, LEAGUE, comp_id, season, stype, _i(r.get("week")),
            to_utc(gameday, gametime), gameday, gametime, "America/New_York",
            team_id(home), team_id(away),
            "final" if hs is not None and as_ is not None else "scheduled",
            hs, as_,
            _s(r.get("stadium_id")),
            1 if (_s(r.get("location")) or "").lower() == "neutral" else 0,
            None,  # attendance: not in this file
            None,  # broadcast: not in this file
            json.dumps({k: _s(r.get(k)) for k in
                        ("old_game_id", "gsis", "nfl_detail_id", "pfr", "pff", "espn", "ftn")
                        if _s(r.get(k))}),
            SOURCE, ENDPOINT, fetched_at, INGEST_VERSION,
        ))

        # --- market lines: closing, consensus, book and capture time unknown ---
        spread, total = _f(r.get("spread_line")), _f(r.get("total_line"))
        ml_h, ml_a = _i(r.get("home_moneyline")), _i(r.get("away_moneyline"))
        if spread is not None:
            lines.append((gid, "consensus", "spread", spread,
                          _i(r.get("home_spread_odds")), _i(r.get("away_spread_odds")),
                          None, None, None, None, 1, SOURCE, ENDPOINT, fetched_at, INGEST_VERSION))
            n_lines += 1
        if total is not None:
            lines.append((gid, "consensus", "total", total, None, None, None,
                          _i(r.get("over_odds")), _i(r.get("under_odds")),
                          None, 1, SOURCE, ENDPOINT, fetched_at, INGEST_VERSION))
            n_lines += 1
        if ml_h is not None or ml_a is not None:
            lines.append((gid, "consensus", "moneyline", None, ml_h, ml_a, None, None, None,
                          None, 1, SOURCE, ENDPOINT, fetched_at, INGEST_VERSION))
            n_lines += 1

        # --- weather: a stadium reading where the file has one, never invented ---
        temp, wind, roof = _f(r.get("temp")), _f(r.get("wind")), _s(r.get("roof"))
        if temp is not None or wind is not None or roof:
            wx.append((gid, temp, wind, None, None, None, roof,
                       "nflverse games.csv (stadium reading)", ENDPOINT, fetched_at, INGEST_VERSION))
            n_wx += 1

        for ab, days in ((home, _i(r.get("home_rest"))), (away, _i(r.get("away_rest")))):
            if days is not None:
                rest.append((gid, team_id(ab), days, None, None, None, None))
                n_rest += 1

    conn.executemany("INSERT OR REPLACE INTO competitions VALUES (?,?,?,?,?,?,?,?,?,?,?)", list(comps.values()))
    conn.executemany(
        "INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(team_id(ab), SPORT, LEAGUE, ab, None, None, None, v["first"],
          None if v["last"] >= max(t["last"] for t in teams.values()) else v["last"],
          json.dumps({"nflverse": ab}), SOURCE, ENDPOINT, fetched_at, INGEST_VERSION)
         for ab, v in sorted(teams.items())])
    conn.executemany(
        "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", games)
    # Columns named explicitly, and provenance stated: these are nflverse's
    # record of the close, not prices we watched. games.csv does not say which
    # book or what time, so 'third_party' is the honest label and the reason
    # the column exists. A positional VALUES here would silently mis-seat every
    # field the next time a column is added.
    conn.executemany(
        "INSERT OR REPLACE INTO market_lines (game_id, book, market_type, line, price_home, "
        "price_away, price_draw, price_over, price_under, captured_at, is_closing, source, "
        "source_endpoint, fetched_at, ingest_version, provenance) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'third_party')", lines)
    conn.executemany("INSERT OR REPLACE INTO weather VALUES (?,?,?,?,?,?,?,?,?,?,?)", wx)
    conn.executemany("INSERT OR REPLACE INTO rest_travel VALUES (?,?,?,?,?,?,?)", rest)
    conn.commit()

    finished_at = datetime.now(timezone.utc).isoformat()
    for table, n in (("games", len(games)), ("market_lines", n_lines), ("weather", n_wx),
                     ("rest_travel", n_rest), ("teams", len(teams)), ("competitions", len(comps))):
        record_run(conn, table, SOURCE, ENDPOINT, started_at, finished_at, n,
                   notes=f"season={args.season or 'all'}")

    logger.info("SUMMARY games=%d market_lines=%d weather=%d rest=%d teams=%d competitions=%d",
                len(games), n_lines, n_wx, n_rest, len(teams), len(comps))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
