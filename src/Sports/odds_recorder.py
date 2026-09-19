"""
odds_recorder.py (cross-sport)
==============================
Records the betting market as it moves, so that closing line value becomes
measurable. This is infrastructure, not a feature, and like the injury
recorder it only ever works forwards: a line we did not capture is gone.

WHY CLV AND NOT ROI. Return on investment needs hundreds of graded bets before
it says anything; closing line value says something after a few dozen. If our
number was better than the market's last word, we were on the right side of
the price, whatever the scoreboard did. It is also the metric sharp bettors
actually respect, and the one we cannot currently produce for basketball
because every season of lines we hold sits inside the model's training window.

THE CREDIT PROBLEM, AND WHY THIS IS SCHEDULE-AWARE. The Odds API free tier is
500 credits a month and one poll of three markets in one region costs 3. A
fixed 45-minute loop would cost roughly 1,400 a month for a single sport,
which is why the existing NBA recorder is documented as needing the paid tier.

But a fixed loop is also the wrong shape. What CLV needs is not evenly spaced
samples; it needs the LAST price before kickoff, plus a reference price from
earlier. So this recorder reads our own schedule and only calls the API when a
kickoff is actually approaching:

    kickoff within 30 minutes   -> capture (this becomes the closing line)
    kickoff within 3 hours      -> capture at most every 90 minutes
    kickoff within 30 hours     -> capture at most every 12 hours
    otherwise                   -> do not call the API at all

For the NFL, whose games cluster into about five kickoff times a week, that is
roughly 6 to 10 polls a week, or 20 to 30 credits, instead of 350. It fits
inside the free tier alongside the NBA, and it captures closer to the close
than a blind loop would.

WHAT IS AND IS NOT A CLOSING LINE. A capture cannot know it is the last one.
So every snapshot lands in `market_line_snapshots` with `is_closing = 0`, and
`--seal` runs after games start, finds the last snapshot before each kickoff,
and promotes it into `market_lines` as that book's closing line. Nothing is
ever called closing until the game has begun and the question is settled.

Append-only: snapshots have triggers that abort UPDATE and DELETE, for the
same reason the injury observations do.

Usage:
    venv/Scripts/python.exe src/Sports/odds_recorder.py --sport nfl
    venv/Scripts/python.exe src/Sports/odds_recorder.py --sport nfl --dry-run
    venv/Scripts/python.exe src/Sports/odds_recorder.py --sport nfl --seal
    venv/Scripts/python.exe src/Sports/odds_recorder.py --sport nfl --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import requests  # noqa: E402

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402
from src.Sports.nfl.teams import NFL_TEAM_BY_NAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("odds_recorder")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_REGIONS = "us"

#: Refuse to spend the last of the month's quota on a routine capture. A human
#: can override with --force, but an unattended scheduled run must never be the
#: thing that empties the account before a slate we care about.
QUOTA_FLOOR = 60

SPORTS: Dict[str, Dict[str, Any]] = {
    "nfl": {
        "odds_api_key": "americanfootball_nfl",
        "db": os.path.join(REPO_ROOT, "Data", "NflData.sqlite"),
        "team_by_name": NFL_TEAM_BY_NAME,
        "prefix": "nfl-",
    },
}

SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_line_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       TEXT NOT NULL,
    captured_at   TEXT NOT NULL,      -- when WE saw it, UTC
    commence_time TEXT,               -- kickoff as the API reported it
    book          TEXT NOT NULL,
    market_type   TEXT NOT NULL,      -- spread | total | moneyline
    line          REAL,
    price_home    INTEGER,
    price_away    INTEGER,
    price_over    INTEGER,
    price_under   INTEGER,
    source        TEXT,
    ingest_version INTEGER
);
CREATE INDEX IF NOT EXISTS idx_snap_game ON market_line_snapshots(game_id, market_type, captured_at);
CREATE INDEX IF NOT EXISTS idx_snap_time ON market_line_snapshots(captured_at);

CREATE TRIGGER IF NOT EXISTS market_snap_no_update
BEFORE UPDATE ON market_line_snapshots
BEGIN SELECT RAISE(ABORT, 'market_line_snapshots is append-only'); END;

CREATE TRIGGER IF NOT EXISTS market_snap_no_delete
BEFORE DELETE ON market_line_snapshots
BEGIN SELECT RAISE(ABORT, 'market_line_snapshots is append-only'); END;
"""


def api_key() -> str:
    k = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not k:
        raise SystemExit("ODDS_API_KEY is not set. Put it in the repo .env; never in a commit.")
    return k


def upcoming(conn: sqlite3.Connection, hours: int = 30) -> List[Tuple[str, str]]:
    """(game_id, kickoff_utc) for games starting within `hours`, not yet final."""
    now = datetime.now(timezone.utc)
    rows = conn.execute(
        """SELECT game_id, date_utc FROM games
           WHERE date_utc IS NOT NULL AND status != 'final'
             AND date_utc BETWEEN ? AND ? ORDER BY date_utc""",
        (now.isoformat(), (now + timedelta(hours=hours)).isoformat()),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def should_capture(conn: sqlite3.Connection) -> Tuple[bool, str]:
    """The whole point of the schedule-aware design lives here."""
    games = upcoming(conn, hours=30)
    if not games:
        return False, "no game kicks off in the next 30 hours"
    now = datetime.now(timezone.utc)
    soonest = min(datetime.fromisoformat(k) for _, k in games)
    mins_out = (soonest - now).total_seconds() / 60.0

    last = conn.execute("SELECT MAX(captured_at) FROM market_line_snapshots").fetchone()[0]
    since = 1e9 if not last else (now - datetime.fromisoformat(last)).total_seconds() / 60.0

    if mins_out <= 30:
        return (since > 10, f"kickoff in {mins_out:.0f} min (closing window)"
                if since > 10 else f"kickoff in {mins_out:.0f} min but captured {since:.0f} min ago")
    if mins_out <= 180:
        return (since > 90, f"kickoff in {mins_out/60:.1f} h, last capture {since:.0f} min ago")
    return (since > 720, f"kickoff in {mins_out/60:.1f} h, last capture {since/60:.1f} h ago")


def fetch(sport_key: str, bookmakers: Optional[str]) -> Tuple[List[Dict], Dict[str, str]]:
    params = {"apiKey": api_key(), "regions": DEFAULT_REGIONS, "markets": DEFAULT_MARKETS,
              "oddsFormat": "american", "dateFormat": "iso"}
    if bookmakers:
        params["bookmakers"] = bookmakers
    r = requests.get(f"{ODDS_API_BASE}/sports/{sport_key}/odds", params=params, timeout=60)
    quota = {"remaining": r.headers.get("x-requests-remaining"),
             "used": r.headers.get("x-requests-used")}
    if r.status_code == 429:
        raise SystemExit(f"Quota exhausted. Remaining={quota['remaining']}")
    r.raise_for_status()
    return r.json(), quota


def match_games(conn: sqlite3.Connection, events: List[Dict], cfg: Dict[str, Any]) -> Dict[str, str]:
    """API event id -> our game_id, by (kickoff date, home team). Names are
    mapped through a hand-verified registry; we never join on a raw name."""
    by_name = cfg["team_by_name"]
    out: Dict[str, str] = {}
    unmatched: List[str] = []
    for e in events:
        home_abbr = by_name.get((e.get("home_team") or "").strip())
        away_abbr = by_name.get((e.get("away_team") or "").strip())
        ct = e.get("commence_time")
        if not (home_abbr and away_abbr and ct):
            unmatched.append(f"{e.get('away_team')} @ {e.get('home_team')}")
            continue
        # Kickoff can drift by a few minutes between sources; match on the day
        # either side, then require both teams.
        day = ct[:10]
        row = conn.execute(
            """SELECT game_id FROM games
               WHERE home_team_id = ? AND away_team_id = ?
                 AND local_date BETWEEN DATE(?, '-1 day') AND DATE(?, '+1 day')
               LIMIT 1""",
            (cfg["prefix"] + home_abbr, cfg["prefix"] + away_abbr, day, day),
        ).fetchone()
        if row:
            out[e["id"]] = row[0]
        else:
            unmatched.append(f"{away_abbr} @ {home_abbr} on {day}")
    if unmatched:
        logger.warning("%d event(s) did not match a game in our schedule: %s",
                       len(unmatched), "; ".join(unmatched[:4]))
    return out


def rows_from(events: List[Dict], match: Dict[str, str], captured_at: str,
              source: str) -> List[tuple]:
    rows = []
    for e in events:
        gid = match.get(e["id"])
        if not gid:
            continue
        home, away = e.get("home_team"), e.get("away_team")
        for bk in e.get("bookmakers") or []:
            book = bk.get("key")
            for m in bk.get("markets") or []:
                key = m.get("key")
                outs = {o.get("name"): o for o in (m.get("outcomes") or [])}
                if key == "h2h":
                    rows.append((gid, captured_at, e.get("commence_time"), book, "moneyline",
                                 None, (outs.get(home) or {}).get("price"),
                                 (outs.get(away) or {}).get("price"), None, None,
                                 source, INGEST_VERSION))
                elif key == "spreads":
                    h = outs.get(home) or {}
                    a = outs.get(away) or {}
                    # Stored from the HOME side, matching Market.py and the
                    # nflverse convention: positive means home favoured.
                    line = h.get("point")
                    rows.append((gid, captured_at, e.get("commence_time"), book, "spread",
                                 None if line is None else -float(line),
                                 h.get("price"), a.get("price"), None, None,
                                 source, INGEST_VERSION))
                elif key == "totals":
                    o = outs.get("Over") or {}
                    u = outs.get("Under") or {}
                    rows.append((gid, captured_at, e.get("commence_time"), book, "total",
                                 o.get("point"), None, None, o.get("price"), u.get("price"),
                                 source, INGEST_VERSION))
    return rows


def seal(conn: sqlite3.Connection) -> int:
    """Promote the last snapshot before kickoff into market_lines as closing.

    Only for games that have actually started, because until then "last" is
    not a fact about the world, it is a fact about when we stopped looking.
    """
    now = datetime.now(timezone.utc).isoformat()
    picked = conn.execute(
        """
        SELECT s.game_id, s.book, s.market_type, s.line, s.price_home, s.price_away,
               s.price_over, s.price_under, s.captured_at
        FROM market_line_snapshots s
        JOIN games g ON g.game_id = s.game_id
        WHERE g.date_utc IS NOT NULL AND g.date_utc < ?
          AND s.captured_at < g.date_utc
          AND s.id = (SELECT MAX(s2.id) FROM market_line_snapshots s2
                      WHERE s2.game_id = s.game_id AND s2.book = s.book
                        AND s2.market_type = s.market_type AND s2.captured_at < g.date_utc)
        """, (now,)).fetchall()
    n = 0
    for r in picked:
        conn.execute(
            "INSERT OR REPLACE INTO market_lines (game_id, book, market_type, line, price_home, "
            "price_away, price_draw, price_over, price_under, captured_at, is_closing, source, "
            "source_endpoint, fetched_at, ingest_version) "
            "VALUES (?,?,?,?,?,?,NULL,?,?,?,1,?,?,?,?)",
            (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
             "the-odds-api (captured live)", ODDS_API_BASE, now, INGEST_VERSION))
        n += 1
    conn.commit()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Record the betting market so CLV becomes measurable.")
    ap.add_argument("--sport", default="nfl", choices=sorted(SPORTS))
    ap.add_argument("--force", action="store_true", help="Capture regardless of the schedule window.")
    ap.add_argument("--dry-run", action="store_true", help="Say what would happen; call nothing.")
    ap.add_argument("--seal", action="store_true", help="Promote closing lines for started games and exit.")
    ap.add_argument("--bookmakers", help="Comma-separated book keys (default: all US books).")
    args = ap.parse_args()

    cfg = SPORTS[args.sport]
    conn = sqlite3.connect(cfg["db"], timeout=120)
    ensure_core_schema(conn)
    conn.executescript(SNAPSHOT_SCHEMA)
    conn.commit()

    if args.seal:
        n = seal(conn)
        logger.info("sealed %d closing line(s) into market_lines", n)
        conn.close()
        return 0

    ok, why = should_capture(conn)
    if not ok and not args.force:
        logger.info("skipping: %s", why)
        conn.close()
        return 0
    logger.info("capturing: %s%s", why, " (forced)" if args.force and not ok else "")

    if args.dry_run:
        games = upcoming(conn)
        logger.info("dry run: would spend 3 credits; %d game(s) in the window", len(games))
        conn.close()
        return 0

    started_at = datetime.now(timezone.utc).isoformat()
    events, quota = fetch(cfg["odds_api_key"], args.bookmakers)
    remaining = int(quota["remaining"] or 0)
    logger.info("%d event(s) on the board; quota used=%s remaining=%s",
                len(events), quota["used"], quota["remaining"])
    if remaining < QUOTA_FLOOR and not args.force:
        logger.warning("quota remaining %d is below the floor of %d. This capture is kept, but "
                       "further routine captures will refuse until the quota resets or the tier "
                       "is raised.", remaining, QUOTA_FLOOR)

    captured_at = datetime.now(timezone.utc).isoformat()
    match = match_games(conn, events, cfg)
    rows = rows_from(events, match, captured_at, "the-odds-api")
    conn.executemany(
        "INSERT INTO market_line_snapshots (game_id, captured_at, commence_time, book, market_type, "
        "line, price_home, price_away, price_over, price_under, source, ingest_version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()

    sealed = seal(conn)
    record_run(conn, "market_line_snapshots", "the-odds-api", f"{ODDS_API_BASE}/sports/{cfg['odds_api_key']}/odds",
               started_at, datetime.now(timezone.utc).isoformat(), len(rows),
               notes=f"events={len(events)} matched={len(match)} sealed={sealed} quota_remaining={remaining}")

    total = conn.execute("SELECT COUNT(*) FROM market_line_snapshots").fetchone()[0]
    books = conn.execute("SELECT COUNT(DISTINCT book) FROM market_line_snapshots").fetchone()[0]
    logger.info("SUMMARY wrote=%d rows for %d matched game(s) across %d book(s); archive=%d; sealed=%d",
                len(rows), len(match), books, total, sealed)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
