"""ESPN public injuries feed client.

Mirrors the frontend's /api/injuries route (same public ESPN endpoint,
same field handling), adapted for the backend's needs:

- 60-second in-memory TTL cache (the frontend caches 15 min; we poll a
  little fresher because availability deltas feed prediction output).
- 5-second timeout; NEVER raises - every failure path returns an empty
  result so a dead ESPN feed can never break predictions.
- Maps ESPN player display names -> our `players` table player_ids using
  normalized names (lowercase, punctuation stripped, Jr/Sr/II/III/IV/V
  suffixes dropped, accents folded). The match rate is reported in the
  returned structure so silent decay is visible.

STATUS RULE (do not change casually): only "Out" and "Doubtful" count as
absences. "Questionable" / "Day-To-Day" players are game-time decisions -
counting them poisons the availability adjustment far more often than it
helps, because most of them play. This is a deliberate policy.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
REQUEST_TIMEOUT_SECONDS = 5.0
CACHE_TTL_SECONDS = 60.0

# Only these statuses count as an absence. See module docstring.
COUNTED_STATUSES = {"out", "doubtful"}

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Data",
    "TeamData.sqlite",
)

_lock = threading.Lock()
_cache: Dict[str, Any] = {"expires": 0.0, "data": None}

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Fold a display name to a matching key: ascii, lowercase, no
    punctuation, no Jr/Sr/II/III/IV/V suffix, collapsed whitespace."""
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    folded = folded.lower()
    # Apostrophes and periods vanish (O'Neal -> oneal, P.J. -> pj);
    # any other punctuation (hyphens etc.) becomes a word break.
    folded = re.sub(r"[.'’]", "", folded)
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    parts = [p for p in folded.split() if p]
    while len(parts) > 1 and parts[-1] in _NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def parse_injuries(payload: Any) -> List[Dict[str, Any]]:
    """Pure parser for the ESPN payload (same handling as the frontend's
    /api/injuries route). Returns one flat entry per listed injury:
    {team_name, player_name, position, status, date, comment, detail}."""
    entries: List[Dict[str, Any]] = []
    team_list = payload.get("injuries") if isinstance(payload, dict) else None
    if not isinstance(team_list, list):
        return entries
    for team_entry in team_list:
        if not isinstance(team_entry, dict):
            continue
        team_name = team_entry.get("displayName") or "Unknown Team"
        injuries = team_entry.get("injuries")
        if not isinstance(injuries, list):
            continue
        for inj in injuries:
            if not isinstance(inj, dict):
                continue
            athlete = inj.get("athlete") or {}
            details = inj.get("details") or {}
            detail_parts: List[str] = []
            if details.get("type"):
                detail_parts.append(str(details["type"]))
            if details.get("location") and details.get("location") != details.get("type"):
                detail_parts.append(f"({details['location']})")
            if details.get("detail") and details.get("detail") != details.get("type"):
                detail_parts.append(f"- {details['detail']}")
            position = athlete.get("position") or {}
            entries.append({
                "team_name": team_name,
                "player_name": athlete.get("displayName") or athlete.get("shortName") or "Unknown Player",
                "position": position.get("abbreviation") or position.get("displayName") or "",
                "status": inj.get("status") or "Unknown",
                "date": inj.get("date") or "",
                "comment": inj.get("shortComment") or inj.get("longComment") or "",
                "detail": " ".join(detail_parts).strip(),
            })
    return entries


def _fetch_payload() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(ESPN_INJURIES_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _load_player_index() -> Dict[str, int]:
    """normalized full name -> player_id. Active players win name clashes."""
    index: Dict[str, int] = {}
    try:
        conn = sqlite3.connect(_DB_PATH)
        try:
            rows = conn.execute(
                "SELECT player_id, full_name, is_active FROM players "
                "ORDER BY is_active ASC"  # actives last so they overwrite
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return index
    for player_id, full_name, _active in rows:
        key = normalize_name(full_name)
        if key:
            index[key] = player_id
    return index


def _load_team_index() -> Dict[str, str]:
    """normalized team name (full name / nickname / abbr) -> abbreviation."""
    index: Dict[str, str] = {}
    try:
        conn = sqlite3.connect(_DB_PATH)
        try:
            rows = conn.execute(
                "SELECT full_name, nickname, abbreviation FROM team_metadata"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return index
    for full_name, nickname, abbr in rows:
        for name in (full_name, nickname, abbr):
            key = normalize_name(name)
            if key:
                index[key] = abbr
    # ESPN says "LA Clippers"; our metadata says "Los Angeles Clippers".
    if "los angeles clippers" in index:
        index.setdefault("la clippers", index["los angeles clippers"])
    return index


def resolve_team_abbr(name: str) -> Optional[str]:
    """Resolve any team spelling (abbr, nickname, full name) to our
    abbreviation, or None."""
    return _team_index().get(normalize_name(name))


_player_index_cache: Optional[Dict[str, int]] = None
_team_index_cache: Optional[Dict[str, str]] = None


def _player_index() -> Dict[str, int]:
    global _player_index_cache
    if _player_index_cache is None:
        _player_index_cache = _load_player_index()
    return _player_index_cache


def _team_index() -> Dict[str, str]:
    global _team_index_cache
    if _team_index_cache is None:
        _team_index_cache = _load_team_index()
    return _team_index_cache


def build_absences(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure aggregation: filter to counted statuses, map names to
    player_ids/team abbrs, report the unmatched rate."""
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    unmatched: List[str] = []
    counted = 0
    for entry in entries:
        status = str(entry.get("status") or "").strip().lower()
        if status not in COUNTED_STATUSES:
            continue
        counted += 1
        abbr = resolve_team_abbr(entry.get("team_name") or "")
        player_id = _player_index().get(normalize_name(entry.get("player_name") or ""))
        if player_id is None:
            unmatched.append(entry.get("player_name") or "?")
        if abbr is None:
            continue
        by_team.setdefault(abbr, []).append({
            "player_id": player_id,
            "name": entry.get("player_name"),
            "status": entry.get("status"),
            "detail": entry.get("detail") or "",
        })
    match_rate = 1.0 if counted == 0 else round((counted - len(unmatched)) / counted, 3)
    return {
        "by_team": by_team,
        "counted_statuses": ["Out", "Doubtful"],
        "total_counted": counted,
        "unmatched_names": unmatched,
        "match_rate": match_rate,
    }


def get_absences(force_refresh: bool = False) -> Dict[str, Any]:
    """Current OUT/DOUBTFUL players by team abbr, cached for 60 seconds.

    Never raises. On any failure returns an empty structure with
    "source": "unavailable" so callers can tell "healthy feed, no
    injuries" from "feed down".
    """
    now = time.time()
    with _lock:
        if not force_refresh and _cache["data"] is not None and now < _cache["expires"]:
            return _cache["data"]
    payload = _fetch_payload()
    if payload is None:
        result = {
            "by_team": {},
            "counted_statuses": ["Out", "Doubtful"],
            "total_counted": 0,
            "unmatched_names": [],
            "match_rate": 1.0,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "unavailable",
        }
        # Do NOT cache failures for the full TTL - retry on next call.
        return result
    try:
        result = build_absences(parse_injuries(payload))
    except Exception:
        return {
            "by_team": {},
            "counted_statuses": ["Out", "Doubtful"],
            "total_counted": 0,
            "unmatched_names": [],
            "match_rate": 1.0,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "unavailable",
        }
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()
    result["source"] = "espn"
    with _lock:
        _cache["data"] = result
        _cache["expires"] = now + CACHE_TTL_SECONDS
    return result


def clear_cache() -> None:
    global _player_index_cache, _team_index_cache
    with _lock:
        _cache["data"] = None
        _cache["expires"] = 0.0
    _player_index_cache = None
    _team_index_cache = None
