"""
nba_live.py
===========
Thin client for the official NBA live-data CDN:

    https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json

Free, no auth, and (unlike stats.nba.com) not blocked for cloud-datacenter
IPs, which makes it the right live-scores source for a deployed backend.
NOTE: the Akamai edge can still geo-block or otherwise 403 some networks —
this module treats any failure (HTTP error, timeout, bad JSON) as "no data":
it logs a warning, returns None from the fetch, and the normalized scoreboard
degrades to games: []. It NEVER raises to the caller.

Responses are cached in memory for 30 seconds per process, which is plenty for
a live scoreboard and keeps the endpoint safe to expose keyless.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
REQUEST_TIMEOUT_SECONDS = 5
CACHE_TTL_SECONDS = 30

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# {"at": monotonic seconds, "payload": normalized scoreboard dict}
_cache: Dict[str, Any] = {}


def _fetch_raw_scoreboard() -> Optional[Dict[str, Any]]:
    """GET the raw CDN payload. Returns None (and logs a warning) on ANY failure."""
    try:
        resp = requests.get(SCOREBOARD_URL, headers=_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(f"cdn.nba.com scoreboard fetch failed: {exc}")
        return None


def _team_block(team: Dict[str, Any]) -> Dict[str, Any]:
    city = str(team.get("teamCity") or "").strip()
    nickname = str(team.get("teamName") or "").strip()
    return {
        "abbr": team.get("teamTricode"),
        "name": (f"{city} {nickname}").strip() or None,
        "score": team.get("score"),
    }


def _normalize(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize the CDN payload to a compact, stable shape. An unreachable CDN or
    an empty/off-season scoreboard both come out as games: [] — a valid answer.
    """
    games = []
    if raw:
        for game in (raw.get("scoreboard") or {}).get("games") or []:
            try:
                games.append({
                    "game_id": game.get("gameId"),
                    "status_text": game.get("gameStatusText"),
                    "period": game.get("period"),
                    "clock": game.get("gameClock"),
                    "home": _team_block(game.get("homeTeam") or {}),
                    "away": _team_block(game.get("awayTeam") or {}),
                    "start_time_utc": game.get("gameTimeUTC"),
                })
            except Exception as exc:
                logger.warning(f"cdn.nba.com scoreboard: skipping malformed game entry: {exc}")
    return {
        "source": "cdn.nba.com",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "games": games,
    }


def get_scoreboard() -> Dict[str, Any]:
    """
    Normalized scoreboard, served from a 30-second in-memory cache. Never
    raises; on any upstream failure the result simply has games: [].
    """
    now = time.monotonic()
    if _cache and now - _cache["at"] < CACHE_TTL_SECONDS:
        return _cache["payload"]
    payload = _normalize(_fetch_raw_scoreboard())
    _cache["at"] = now
    _cache["payload"] = payload
    return payload
