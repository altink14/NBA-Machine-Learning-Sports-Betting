"""
backfill_odds_history.py
========================
One-time backfill of historical NBA odds snapshots from The Odds API into
Data/OddsData.sqlite (new table: odds_api_history). Fills the gap left by the
legacy odds tables, which stop at 2024-04-28 — without this, EV and CLV can
never be backtested for 2024-25 / 2025-26.

CREDIT ECONOMICS (read before running --execute):
  Historical snapshot calls cost 10 x markets x regions credits.
  With markets=h2h,spreads,totals and regions=us that is 30 credits PER CALL.
  The free tier (500/mo) cannot run a season backfill. Plans:
    - one snapshot per game day  (~350 days x 30cr =  ~10,500 credits -> $30 tier)
    - snapshot per distinct tip hour (~3.5/day x 30cr = ~36,700 credits -> $59 tier)
  DEFAULT IS --dry-run: enumerates the plan and prints the exact credit cost
  without making a single API call. Nothing is spent until you pass --execute.

SAFETY:
  - Credit guard: aborts when x-requests-remaining drops below --min-remaining
    (default 50), so a misconfigured run can never zero the account.
  - Checkpoint/resume: progress is recorded in the DB; re-running skips
    completed snapshots, so an aborted run continues where it stopped.
  - Rate-limited politely (1 req/sec).

Usage:
  venv/Scripts/python.exe backfill_odds_history.py                  # dry run, full plan
  venv/Scripts/python.exe backfill_odds_history.py --limit 2        # dry run, first 2 snapshots
  venv/Scripts/python.exe backfill_odds_history.py --execute --limit 2   # tiny paid test (60 credits)
  venv/Scripts/python.exe backfill_odds_history.py --execute        # full run (paid tier)
  venv/Scripts/python.exe backfill_odds_history.py --per-tip-hour --execute   # precision mode
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_odds")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TEAM_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKETS = "h2h,spreads,totals"
REGIONS = "us"
CREDITS_PER_CALL = 10 * 3 * 1  # 10x historical multiplier x 3 markets x 1 region

# Seasons the legacy odds tables do NOT cover.
TARGET_SEASONS = ["2024-25", "2025-26"]

# US/Eastern offset by month (NBA season spans DST changes; a fixed table is
# fine at snapshot granularity: Nov-Mar = EST(-5), rest = EDT(-4)).
def eastern_offset_hours(month: int) -> int:
    return -5 if month in (11, 12, 1, 2, 3) else -4


def load_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        # fall back to .env in repo root
        env_path = os.path.join(REPO_ROOT, ".env")
        if os.path.exists(env_path):
            for line in open(env_path, encoding="utf-8", errors="ignore"):
                if line.startswith("ODDS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        logger.error("ODDS_API_KEY not set (env or .env). Aborting.")
        sys.exit(1)
    return key


def game_days(per_tip_hour: bool) -> list:
    """
    Enumerate snapshot instants (UTC ISO strings) from our own game log.
    One per game day by default (19:00 ET - near close for early tips), or one
    per distinct game-day tip-hour block with --per-tip-hour.
    Only PAST days are included (historical endpoint has ~1-5 min lag anyway).
    """
    conn = sqlite3.connect(f"file:{TEAM_DB}?mode=ro", uri=True)
    seasons_q = ",".join("?" for _ in TARGET_SEASONS)
    rows = conn.execute(
        f"SELECT DISTINCT game_date FROM box_scores WHERE season IN ({seasons_q}) ORDER BY game_date",
        TARGET_SEASONS,
    ).fetchall()
    conn.close()

    now = datetime.now(timezone.utc)
    snapshots = []
    for (gd,) in rows:
        d = datetime.strptime(gd, "%Y-%m-%d")
        off = eastern_offset_hours(d.month)
        if per_tip_hour:
            # 3 blocks: early (19:00 ET), mid (20:30 ET), late (22:00 ET)
            hours = [(19, 0), (20, 30), (22, 0)]
        else:
            hours = [(19, 0)]
        for h, m in hours:
            utc_dt = d.replace(hour=h, minute=m) - timedelta(hours=off)
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
            if utc_dt < now:
                snapshots.append(utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return snapshots


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_api_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts TEXT NOT NULL,
            event_id TEXT NOT NULL,
            commence_time TEXT,
            home_team TEXT,
            away_team TEXT,
            book_key TEXT NOT NULL,
            book_title TEXT,
            market TEXT NOT NULL,
            outcome_name TEXT NOT NULL,
            price_american INTEGER,
            point REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE (snapshot_ts, event_id, book_key, market, outcome_name, point)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_oah_event ON odds_api_history (event_id, book_key, market)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_api_backfill_log (
            snapshot_ts TEXT PRIMARY KEY,
            fetched_at TEXT NOT NULL,
            events INTEGER,
            rows_written INTEGER,
            credits_remaining INTEGER
        )
        """
    )
    conn.commit()


def fetch_snapshot(api_key: str, ts: str):
    """One historical odds call. Returns (payload, remaining_credits)."""
    params = urllib.parse.urlencode(
        {
            "apiKey": api_key,
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": "american",
            "date": ts,
        }
    )
    url = f"{API_BASE}/historical/sports/{SPORT}/odds?{params}"
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        payload = json.loads(resp.read().decode("utf-8"))
    return payload, (int(float(remaining)) if remaining is not None else None)


def write_snapshot(conn: sqlite3.Connection, ts: str, payload: dict) -> int:
    data = payload.get("data") or []
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for ev in data:
        for book in ev.get("bookmakers") or []:
            for mk in book.get("markets") or []:
                for oc in mk.get("outcomes") or []:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO odds_api_history
                        (snapshot_ts, event_id, commence_time, home_team, away_team,
                         book_key, book_title, market, outcome_name, price_american,
                         point, fetched_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            ts,
                            ev.get("id"),
                            ev.get("commence_time"),
                            ev.get("home_team"),
                            ev.get("away_team"),
                            book.get("key"),
                            book.get("title"),
                            mk.get("key"),
                            oc.get("name"),
                            oc.get("price"),
                            oc.get("point"),
                            now,
                        ),
                    )
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="actually spend credits (default: dry run)")
    ap.add_argument("--per-tip-hour", action="store_true", help="3 snapshots/game-day instead of 1")
    ap.add_argument("--limit", type=int, default=None, help="cap number of snapshots (for testing)")
    ap.add_argument("--min-remaining", type=int, default=50, help="abort when credits drop below this")
    args = ap.parse_args()

    snapshots = game_days(args.per_tip_hour)
    if args.limit:
        snapshots = snapshots[: args.limit]

    est_credits = len(snapshots) * CREDITS_PER_CALL
    logger.info("Seasons: %s | snapshot instants: %d | est. credit cost: %d (%d per call)",
                ", ".join(TARGET_SEASONS), len(snapshots), est_credits, CREDITS_PER_CALL)

    if not args.execute:
        logger.info("DRY RUN - no API calls made. First 5 planned snapshots: %s", snapshots[:5])
        logger.info("Free tier is 500 credits/mo; this plan needs a paid month "
                    "($30 tier covers 1/day, $59 covers --per-tip-hour). "
                    "Re-run with --execute when ready.")
        return 0

    api_key = load_api_key()
    conn = sqlite3.connect(ODDS_DB)
    ensure_schema(conn)

    done = {r[0] for r in conn.execute("SELECT snapshot_ts FROM odds_api_backfill_log")}
    todo = [s for s in snapshots if s not in done]
    logger.info("%d already done (resume), %d to fetch", len(done & set(snapshots)), len(todo))

    for i, ts in enumerate(todo, 1):
        try:
            payload, remaining = fetch_snapshot(api_key, ts)
        except Exception as exc:
            logger.error("Fetch failed at %s: %s - stopping (resume-safe).", ts, exc)
            break
        rows = write_snapshot(conn, ts, payload)
        conn.execute(
            "INSERT OR REPLACE INTO odds_api_backfill_log VALUES (?,?,?,?,?)",
            (ts, datetime.now(timezone.utc).isoformat(),
             len(payload.get("data") or []), rows, remaining),
        )
        conn.commit()
        logger.info("[%d/%d] %s: %d events, %d rows, credits remaining: %s",
                    i, len(todo), ts, len(payload.get("data") or []), rows, remaining)
        if remaining is not None and remaining < args.min_remaining:
            logger.warning("Credit guard tripped (%s < %d). Stopping - resume later.",
                           remaining, args.min_remaining)
            break
        time.sleep(1.0)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
