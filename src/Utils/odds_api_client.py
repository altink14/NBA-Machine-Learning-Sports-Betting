"""
odds_api_client.py
==================
The Odds API (the-odds-api.com) -> odds_snapshots archive.

WHY THIS EXISTS. Every blocked money-layer feature (CLV Time Machine, Line
Shop, Book Softness) needs a season's worth of odds snapshots across multiple
books, and closing lines are only observable live - this data cannot be
bought back later. This module is the tape recorder: one official API call
per run captures moneylines, spreads and totals for every NBA game on the
board, across every US book the API carries, into the same odds_snapshots
table the line-movement endpoint already reads.

SCHEMA. odds_snapshots gains five additive columns on first write (spread and
the prices on both sides of spread/total); every existing reader keeps
working because the original columns are untouched.

QUOTA. The API is credit-metered (free tier: 500/month). One call with three
markets in one region costs 3 credits, so the daily 9 AM snapshot costs ~90
credits/month - safely inside the free tier. The in-season cadence (every
30-60 min, snapshot_odds_api.py --loop) needs the paid tier; the quota
headers are logged on every call so drift is visible in daily_update.log.

KEY. Set ODDS_API_KEY in the environment. No key -> callers get a clear
error, never a silent no-op that leaves the archive empty while looking fine.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_REGIONS = "us"

_EXTRA_COLUMNS = [
    ("spread_home", "REAL"),          # home side handicap (negative = home favored)
    ("spread_home_price", "REAL"),    # American juice on the home spread
    ("spread_away_price", "REAL"),
    ("ou_over_price", "REAL"),        # American juice on the over/under
    ("ou_under_price", "REAL"),
]


class OddsApiError(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise OddsApiError(
            "ODDS_API_KEY is not set. Create a free key at the-odds-api.com and "
            "set it in the environment (or the scheduled task's environment)."
        )
    return key


def fetch_nba_odds(
    api_key: Optional[str] = None,
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    bookmakers: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """
    One board snapshot: every upcoming NBA event with every carried book's
    h2h/spreads/totals. Returns (events, quota) where quota holds the API's
    x-requests-remaining/used headers.
    """
    params = {
        "apiKey": api_key or get_api_key(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    resp = requests.get(f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds", params=params, timeout=timeout)
    quota = {
        "remaining": resp.headers.get("x-requests-remaining"),
        "used": resp.headers.get("x-requests-used"),
    }
    if resp.status_code == 401:
        raise OddsApiError("The Odds API rejected the key (401). Check ODDS_API_KEY.")
    if resp.status_code == 429:
        raise OddsApiError(f"The Odds API quota is exhausted (429). Remaining={quota['remaining']}.")
    if not resp.ok:
        raise OddsApiError(f"The Odds API returned {resp.status_code}: {resp.text[:200]}")
    events = resp.json()
    if not isinstance(events, list):
        raise OddsApiError(f"Unexpected response shape: {type(events)}")
    logger.info(
        "The Odds API: %d NBA events on the board; quota used=%s remaining=%s",
        len(events), quota["used"], quota["remaining"],
    )
    return events, quota


def events_to_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten API events into one row per (game, bookmaker), in the archive's
    vocabulary: game_key 'Home:Away' with full team names, American prices.
    """
    rows: List[Dict[str, Any]] = []
    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue
        start = ev.get("commence_time")  # ISO8601 Zulu
        for book in ev.get("bookmakers") or []:
            row: Dict[str, Any] = {
                "sportsbook": book.get("key"),
                "game_key": f"{home}:{away}",
                "home_team": home,
                "away_team": away,
                "game_start_time_utc": start,
                "home_ml": None, "away_ml": None,
                "spread_home": None, "spread_home_price": None, "spread_away_price": None,
                "ou_line": None, "ou_over_price": None, "ou_under_price": None,
            }
            for market in book.get("markets") or []:
                mkey = market.get("key")
                outcomes = market.get("outcomes") or []
                if mkey == "h2h":
                    for o in outcomes:
                        if o.get("name") == home:
                            row["home_ml"] = o.get("price")
                        elif o.get("name") == away:
                            row["away_ml"] = o.get("price")
                elif mkey == "spreads":
                    for o in outcomes:
                        if o.get("name") == home:
                            row["spread_home"] = o.get("point")
                            row["spread_home_price"] = o.get("price")
                        elif o.get("name") == away:
                            row["spread_away_price"] = o.get("price")
                elif mkey == "totals":
                    for o in outcomes:
                        if o.get("name") == "Over":
                            row["ou_line"] = o.get("point")
                            row["ou_over_price"] = o.get("price")
                        elif o.get("name") == "Under":
                            row["ou_under_price"] = o.get("price")
            if row["sportsbook"]:
                rows.append(row)
    return rows


def ensure_snapshot_schema(conn: sqlite3.Connection) -> None:
    """Additive migration: the five price/spread columns, if missing."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(odds_snapshots)")}
    for col, ctype in _EXTRA_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE odds_snapshots ADD COLUMN {col} {ctype}")
            logger.info("odds_snapshots: added column %s", col)


_CHANGE_FIELDS = [
    "home_ml", "away_ml", "ou_line",
    "spread_home", "spread_home_price", "spread_away_price",
    "ou_over_price", "ou_under_price",
]


def write_snapshot_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]],
                        sport: str = "NBA") -> Dict[str, int]:
    """
    Change-detected insert, same contract as the legacy snapshot writer: a row
    is written only when any tracked number moved since that book's last
    snapshot of that game. Returns {written, unchanged}.
    """
    ensure_snapshot_schema(conn)
    captured_at = datetime.now(timezone.utc).isoformat()
    written = unchanged = 0
    for row in rows:
        last = conn.execute(
            "SELECT * FROM odds_snapshots WHERE sportsbook=? AND sport=? AND game_key=? "
            "ORDER BY captured_at DESC LIMIT 1",
            (row["sportsbook"], sport, row["game_key"]),
        ).fetchone()
        if last is not None and all(last[f] == row[f] for f in _CHANGE_FIELDS):
            unchanged += 1
            continue
        conn.execute(
            "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, home_team, "
            "away_team, home_ml, away_ml, ou_line, game_start_time_utc, spread_home, "
            "spread_home_price, spread_away_price, ou_over_price, ou_under_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                captured_at, sport, row["sportsbook"], row["game_key"], row["home_team"],
                row["away_team"], row["home_ml"], row["away_ml"], row["ou_line"],
                row["game_start_time_utc"], row["spread_home"], row["spread_home_price"],
                row["spread_away_price"], row["ou_over_price"], row["ou_under_price"],
            ),
        )
        written += 1
    conn.commit()
    return {"written": written, "unchanged": unchanged}


def snapshot_nba_board(db_path: str, bookmakers: Optional[str] = None) -> Dict[str, Any]:
    """Fetch the board once and archive it. The one-call entry point."""
    events, quota = fetch_nba_odds(bookmakers=bookmakers)
    rows = events_to_rows(events)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 15000")
        result = write_snapshot_rows(conn, rows)
    finally:
        conn.close()
    summary = {
        "events": len(events),
        "book_rows": len(rows),
        **result,
        "quota_remaining": quota["remaining"],
        "quota_used": quota["used"],
    }
    logger.info("Odds snapshot: %s", summary)
    return summary
