"""
nba_stats_client.py
===================
Thread-safe NBA Stats API client wrapper.
Uses official nba_api endpoint classes directly for reliable connection/header handling,
integrated with rate-limiting, exponential-backoff retries, and two-tier disk caching.
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Import nba_api endpoints
from nba_api.stats.endpoints import (
    scoreboardv3,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
    playbyplayv3,
    leaguedashteamstats,
    commonallplayers,
    commonteamroster,
    leaguegamelog,
    boxscoresummaryv2,
    playercareerstats,
    playergamelog,
    shotchartdetail
)

logger = logging.getLogger(__name__)

# Cache configuration
_CACHE_ROOT = Path(os.environ.get("NBA_CACHE_DIR", "Data/nba_cache"))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

_LIVE_TTL_SECONDS = 24 * 3600          # 24 h for live scoreboard
_COMPLETED_GAME_TTL = None              # permanent cache for completed games
_DEFAULT_RATE_DELAY = 1.0              # seconds between requests
_BACKFILL_RATE_DELAY = 2.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

def _cache_path(endpoint: str, params: Dict[str, Any]) -> Path:
    safe_params = {k: v for k, v in sorted(params.items()) if v is not None}
    key = endpoint + "_" + "_".join(f"{k}={v}" for k, v in safe_params.items())
    key = "".join(c if c.isalnum() or c in "-_=." else "_" for c in key)
    return _CACHE_ROOT / f"{key}.json"

def _read_cache(path: Path, ttl_seconds: Optional[int]) -> Optional[Dict]:
    if not path.exists():
        return None
    if ttl_seconds is not None:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug("Cache read error (%s): %s", path, exc)
        return None

def _write_cache(path: Path, data: Dict) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Cache write error (%s): %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

class NBAStatsClient:
    """
    Thread-safe wrapper around official nba_api endpoints.
    """

    def __init__(self, rate_delay: float = _DEFAULT_RATE_DELAY, backfill_mode: bool = False):
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._rate_delay = _BACKFILL_RATE_DELAY if backfill_mode else rate_delay

    def _fetch(
        self,
        endpoint_name: str,
        endpoint_class: Type,
        params: Dict[str, Any],
        ttl: Optional[int] = _LIVE_TTL_SECONDS
    ) -> Dict:
        """Generic method checking cache, rate-limiting, instantiating endpoint class, and caching results."""
        cache_file = _cache_path(endpoint_name, params)
        cached = _read_cache(cache_file, ttl)
        if cached is not None:
            logger.debug("Cache HIT: %s", endpoint_name)
            return cached

        for attempt in range(1, _MAX_RETRIES + 1):
            with self._lock:
                now = time.time()
                wait = self._rate_delay - (now - self._last_request_time)
                if wait > 0:
                    time.sleep(wait)
                try:
                    logger.info("Outbound request to stats.nba.com for %s (attempt %d/%d)", endpoint_name, attempt, _MAX_RETRIES)
                    instance = endpoint_class(**params)
                    data = instance.get_dict()
                    self._last_request_time = time.time()
                    
                    _write_cache(cache_file, data)
                    return data
                except Exception as exc:
                    self._last_request_time = time.time()
                    if attempt < _MAX_RETRIES:
                        sleep_for = _BACKOFF_BASE ** attempt
                        logger.warning(
                            "Request error on %s (attempt %d/%d). Retrying in %.1fs. Error: %s",
                            endpoint_name, attempt, _MAX_RETRIES, sleep_for, exc
                        )
                        time.sleep(sleep_for)
                        continue
                    raise

        raise RuntimeError(f"Failed to fetch {endpoint_name} after {_MAX_RETRIES} attempts")

    @staticmethod
    def _parse_result_set(data: Dict, result_set_name: str) -> List[Dict]:
        result_sets = data.get("resultSets", [])
        rs = next((r for r in result_sets if r.get("name") == result_set_name), None)
        if rs is None:
            single_rs = data.get("resultSet", {})
            if isinstance(single_rs, dict) and single_rs.get("name") == result_set_name:
                rs = single_rs
        
        if rs is None:
            return []
            
        headers: List[str] = rs["headers"]
        return [dict(zip(headers, row)) for row in rs["rowSet"]]

    @staticmethod
    def _parse_all_result_sets(data: Dict) -> Dict[str, List[Dict]]:
        result_sets = data.get("resultSets", [])
        out = {}
        for rs in result_sets:
            headers: List[str] = rs["headers"]
            out[rs["name"]] = [dict(zip(headers, row)) for row in rs["rowSet"]]
            
        if not out and "resultSet" in data:
            rs = data["resultSet"]
            if isinstance(rs, dict) and "headers" in rs and "rowSet" in rs:
                headers = rs["headers"]
                out[rs.get("name", "resultSet")] = [dict(zip(headers, row)) for row in rs["rowSet"]]
        return out

    # --- API Wrappers ---

    def scoreboard(self, game_date: str = None) -> Dict:
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")
        params = {"game_date": game_date, "league_id": "00"}
        return self._fetch("scoreboardv3", scoreboardv3.ScoreboardV3, params, ttl=_LIVE_TTL_SECONDS)

    def boxscore_traditional(self, game_id: str) -> Dict:
        params = {"game_id": game_id}
        # Returns raw dict directly for modern V3
        return self._fetch("boxscoretraditionalv3", boxscoretraditionalv3.BoxScoreTraditionalV3, params, ttl=_COMPLETED_GAME_TTL)

    def boxscore_advanced(self, game_id: str) -> Dict:
        params = {"game_id": game_id}
        # Returns raw dict directly for modern V3
        return self._fetch("boxscoreadvancedv3", boxscoreadvancedv3.BoxScoreAdvancedV3, params, ttl=_COMPLETED_GAME_TTL)

    def play_by_play(self, game_id: str) -> List[Dict]:
        params = {
            "game_id": game_id,
        }
        raw = self._fetch("playbyplayv3", playbyplayv3.PlayByPlayV3, params, ttl=_COMPLETED_GAME_TTL)
        return raw.get("game", {}).get("actions", [])

    def league_dash_team_stats(
        self,
        season: str = "2024-25",
        season_type: str = "Regular Season",
        per_mode: str = "PerGame",
        measure_type: str = "Base",
    ) -> List[Dict]:
        params = {
            "season": season,
            "season_type_all_star": season_type,
            "per_mode_detailed": per_mode,
            "measure_type_detailed_defense": measure_type,
        }
        raw = self._fetch("leaguedashteamstats", leaguedashteamstats.LeagueDashTeamStats, params, ttl=_LIVE_TTL_SECONDS)
        return self._parse_result_set(raw, "LeagueDashTeamStats")

    def common_all_players(
        self, season: str = "2024-25", is_only_current_season: int = 1
    ) -> List[Dict]:
        params = {
            "is_only_current_season": is_only_current_season,
            "league_id": "00",
            "season": season,
        }
        raw = self._fetch("commonallplayers", commonallplayers.CommonAllPlayers, params, ttl=_LIVE_TTL_SECONDS)
        return self._parse_result_set(raw, "CommonAllPlayers")

    def common_team_roster(self, team_id: int, season: str = "2024-25") -> List[Dict]:
        params = {"team_id": team_id, "season": season, "league_id": "00"}
        raw = self._fetch("commonteamroster", commonteamroster.CommonTeamRoster, params, ttl=_LIVE_TTL_SECONDS)
        return self._parse_result_set(raw, "CommonTeamRoster")

    def league_game_log(
        self,
        season: str = "2024-25",
        season_type: str = "Regular Season",
        player_or_team: str = "T",
    ) -> List[Dict]:
        params = {
            "season": season,
            "season_type_all_star": season_type,
            "player_or_team_abbreviation": player_or_team,
            "league_id": "00",
            "sorter": "DATE",
            "direction": "ASC",
        }
        raw = self._fetch("leaguegamelog", leaguegamelog.LeagueGameLog, params, ttl=_LIVE_TTL_SECONDS)
        return self._parse_result_set(raw, "LeagueGameLog")

    def boxscore_summary(self, game_id: str) -> Dict[str, List[Dict]]:
        params = {"game_id": game_id}
        raw = self._fetch("boxscoresummaryv2", boxscoresummaryv2.BoxScoreSummaryV2, params, ttl=_COMPLETED_GAME_TTL)
        return self._parse_all_result_sets(raw)

    def player_career_stats(
        self, player_id: int, per_mode: str = "PerGame"
    ) -> Dict[str, List[Dict]]:
        params = {
            "player_id": player_id,
            "per_mode_36": per_mode,
            "league_id": "00",
        }
        raw = self._fetch("playercareerstats", playercareerstats.PlayerCareerStats, params, ttl=_LIVE_TTL_SECONDS)
        return self._parse_all_result_sets(raw)

    def league_shot_averages(
        self, season: str, season_type: str = "Regular Season"
    ) -> List[Dict]:
        """
        League-wide FG% by shot zone for a season (the "LeagueAverages" result
        set of shotchartdetail). Fetched with player_id=0/team_id=0 so the
        Shot_Chart_Detail set is empty and the payload stays tiny. One call
        serves every player/game for the season; disk-cached in nba_cache.
        """
        params = {
            "team_id": 0,
            "player_id": 0,
            "season_nullable": season,
            "context_measure_simple": "FGA",
            "season_type_all_star": season_type,
        }
        raw = self._fetch(
            "shotchartleagueavg", shotchartdetail.ShotChartDetail, params,
            ttl=_LIVE_TTL_SECONDS
        )
        return self._parse_result_set(raw, "LeagueAverages")

    def player_game_log(
        self,
        player_id: int,
        season: str = "2024-25",
        season_type: str = "Regular Season",
    ) -> List[Dict]:
        params = {
            "player_id": player_id,
            "season": season,
            "season_type_all_star": season_type,
            "league_id": "00",
        }
        raw = self._fetch("playergamelog", playergamelog.PlayerGameLog, params, ttl=_LIVE_TTL_SECONDS)
        return self._parse_result_set(raw, "PlayerGameLog")


# Module-level singleton
_client_instance: Optional[NBAStatsClient] = None
_client_lock = threading.Lock()

def get_client(backfill_mode: bool = False) -> NBAStatsClient:
    global _client_instance
    with _client_lock:
        if _client_instance is None:
            _client_instance = NBAStatsClient(backfill_mode=backfill_mode)
        return _client_instance
