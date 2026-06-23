"""
nba_pipeline.py
================
Orchestrates the NBA Stats API ETL pipeline using modern V3 endpoints:
1. Fetch raw game box scores (traditional + advanced V3).
2. Compute Dean Oliver advanced derivatives.
3. Validate computed pace, ratings, and four factors against official NBA numbers.
4. Save raw and computed metrics to sqlite database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.Utils.nba_stats_client import get_client
from src.Utils.nba_db_schema import ensure_schema, get_connection
from src.Utils.nba_computed_derivatives import (
    BoxScoreTeam,
    compute_game_advanced,
    GameAdvancedStats
)
from src.Utils.nba_validation import validate_game_stats

logger = logging.getLogger(__name__)


def parse_minutes(min_val: Any) -> float:
    """
    Safely convert minutes values (which can be float, int, or string "240:00")
    to decimal float minutes.
    """
    if min_val is None:
        return 0.0
    if isinstance(min_val, (int, float)):
        return float(min_val)
    
    min_str = str(min_val).strip()
    if ":" in min_str:
        parts = min_str.split(":")
        try:
            minutes = float(parts[0])
            seconds = float(parts[1]) if len(parts) > 1 else 0.0
            return minutes + (seconds / 60.0)
        except ValueError:
            pass
    try:
        return float(min_str)
    except ValueError:
        return 0.0


def extract_team_traditional(team_data: Dict[str, Any], game_id: str, ot_periods: int) -> BoxScoreTeam:
    """Construct a BoxScoreTeam dataclass from a V3 team dictionary."""
    stats = team_data.get("statistics", {})
    minutes = parse_minutes(stats.get("minutes", 240.0))
    
    return BoxScoreTeam(
        team_id=int(team_data["teamId"]),
        team_abbr=str(team_data.get("teamTricode", "")),
        game_id=game_id,
        min=minutes,
        fgm=int(stats.get("fieldGoalsMade", 0)),
        fga=int(stats.get("fieldGoalsAttempted", 0)),
        fg3m=int(stats.get("threePointersMade", 0)),
        fg3a=int(stats.get("threePointersAttempted", 0)),
        ftm=int(stats.get("freeThrowsMade", 0)),
        fta=int(stats.get("freeThrowsAttempted", 0)),
        oreb=int(stats.get("reboundsOffensive", 0)),
        dreb=int(stats.get("reboundsDefensive", 0)),
        reb=int(stats.get("reboundsTotal", 0)),
        ast=int(stats.get("assists", 0)),
        stl=int(stats.get("steals", 0)),
        blk=int(stats.get("blocks", 0)),
        tov=int(stats.get("turnovers", 0)),
        pf=int(stats.get("foulsPersonal", 0)),
        pts=int(stats.get("points", 0)),
        ot_periods=ot_periods
    )


def save_players_and_game_log(
    db_conn: sqlite3.Connection,
    game_id: str,
    game_date: str,
    players_list: List[Dict[str, Any]],
    team_id: int
) -> None:
    """Save player records and player game logs to the database."""
    for p in players_list:
        player_id = int(p["personId"])
        first_name = p.get("firstName", "")
        last_name = p.get("familyName", "")
        full_name = f"{first_name} {last_name}".strip()

        # Save/update player in players table
        db_conn.execute(
            """
            INSERT INTO players (player_id, full_name, first_name, last_name, is_active)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(player_id) DO UPDATE SET
                full_name=excluded.full_name,
                first_name=excluded.first_name,
                last_name=excluded.last_name
            """,
            (player_id, full_name, first_name, last_name)
        )

        # Check if the player played
        stats = p.get("statistics", {})
        min_str = stats.get("minutes", "")
        if not min_str or min_str.strip() == "":
            continue

        minutes = parse_minutes(min_str)
        if minutes <= 0.0:
            continue

        fgm = int(stats.get("fieldGoalsMade", 0))
        fga = int(stats.get("fieldGoalsAttempted", 0))
        fg_pct = float(stats.get("fieldGoalsPercentage", 0.0))
        fg3m = int(stats.get("threePointersMade", 0))
        fg3a = int(stats.get("threePointersAttempted", 0))
        fg3_pct = float(stats.get("threePointersPercentage", 0.0))
        ftm = int(stats.get("freeThrowsMade", 0))
        fta = int(stats.get("freeThrowsAttempted", 0))
        ft_pct = float(stats.get("freeThrowsPercentage", 0.0))
        oreb = int(stats.get("reboundsOffensive", 0))
        dreb = int(stats.get("reboundsDefensive", 0))
        reb = int(stats.get("reboundsTotal", 0))
        ast = int(stats.get("assists", 0))
        stl = int(stats.get("steals", 0))
        blk = int(stats.get("blocks", 0))
        tov = int(stats.get("turnovers", 0))
        pf = int(stats.get("foulsPersonal", 0))
        pts = int(stats.get("points", 0))
        plus_minus = float(stats.get("plusMinusPoints", 0.0))
        starter = 1 if p.get("position") else 0

        db_conn.execute(
            """
            INSERT INTO player_game_log (
                game_id, player_id, team_id, game_date, min,
                fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                ftm, fta, ft_pct, oreb, dreb, reb,
                ast, stl, blk, tov, pf, pts, plus_minus, starter
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, player_id) DO UPDATE SET
                team_id=excluded.team_id,
                game_date=excluded.game_date,
                min=excluded.min,
                fgm=excluded.fgm,
                fga=excluded.fga,
                fg_pct=excluded.fg_pct,
                fg3m=excluded.fg3m,
                fg3a=excluded.fg3a,
                fg3_pct=excluded.fg3_pct,
                ftm=excluded.ftm,
                fta=excluded.fta,
                ft_pct=excluded.ft_pct,
                oreb=excluded.oreb,
                dreb=excluded.dreb,
                reb=excluded.reb,
                ast=excluded.ast,
                stl=excluded.stl,
                blk=excluded.blk,
                tov=excluded.tov,
                pf=excluded.pf,
                pts=excluded.pts,
                plus_minus=excluded.plus_minus,
                starter=excluded.starter
            """,
            (
                game_id, player_id, team_id, game_date, minutes,
                fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                ftm, fta, ft_pct, oreb, dreb, reb,
                ast, stl, blk, tov, pf, pts, plus_minus, starter
            )
        )


def save_computed_game_stats(
    db_conn: sqlite3.Connection,
    stats: GameAdvancedStats,
    season: str,
    season_type: str,
    game_date: str,
    pts: int,
    opp_pts: int
) -> None:
    """Save computed GameAdvancedStats to team_game_advanced database table."""
    timestamp = datetime.utcnow().isoformat()
    try:
        db_conn.execute(
            """
            INSERT INTO team_game_advanced (
                game_id, team_id, opp_team_id, season, season_type, game_date,
                pts, opp_pts,
                poss_estimated, poss_opponent, pace, off_rating, def_rating, net_rating,
                efg_pct, tov_pct, orb_pct, ft_rate, ts_pct, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, team_id) DO UPDATE SET
                pts=excluded.pts,
                opp_pts=excluded.opp_pts,
                poss_estimated=excluded.poss_estimated,
                poss_opponent=excluded.poss_opponent,
                pace=excluded.pace,
                off_rating=excluded.off_rating,
                def_rating=excluded.def_rating,
                net_rating=excluded.net_rating,
                efg_pct=excluded.efg_pct,
                tov_pct=excluded.tov_pct,
                orb_pct=excluded.orb_pct,
                ft_rate=excluded.ft_rate,
                ts_pct=excluded.ts_pct,
                computed_at=excluded.computed_at
            """,
            (
                stats.game_id,
                stats.team_id,
                stats.opp_team_id,
                season,
                season_type,
                game_date,
                pts,
                opp_pts,
                stats.poss_estimated,
                stats.poss_opponent,
                stats.pace,
                stats.off_rating,
                stats.def_rating,
                stats.net_rating,
                stats.efg_pct,
                stats.tov_pct,
                stats.orb_pct,
                stats.ft_rate,
                stats.ts_pct,
                timestamp
            )
        )
    except sqlite3.Error as err:
        logger.error("Failed to save computed stats for game %s, team %d: %s", stats.game_id, stats.team_id, err)
        raise


def get_game_from_db(db_conn: sqlite3.Connection, game_id: str) -> Optional[Tuple[Dict, Dict]]:
    """Load cached raw traditional and advanced box scores JSON from box_scores table."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT traditional_json, advanced_json FROM box_scores WHERE game_id = ?", (game_id,))
        row = cursor.fetchone()
        if row and row["traditional_json"] and row["advanced_json"]:
            return json.loads(row["traditional_json"]), json.loads(row["advanced_json"])
    except sqlite3.Error as err:
        logger.error("Error checking box_scores cache for game %s: %s", game_id, err)
    return None


def save_raw_game_to_db(
    db_conn: sqlite3.Connection,
    game_id: str,
    home_team_id: int,
    away_team_id: int,
    season: str,
    season_type: str,
    game_date: str,
    traditional_data: Dict,
    advanced_data: Dict
) -> None:
    """Save raw API JSON responses to box_scores table for offline playback and durability."""
    timestamp = datetime.utcnow().isoformat()
    try:
        db_conn.execute(
            """
            INSERT INTO box_scores (
                game_id, fetched_at, home_team_id, away_team_id, season, season_type, game_date, traditional_json, advanced_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                traditional_json=excluded.traditional_json,
                advanced_json=excluded.advanced_json,
                fetched_at=excluded.fetched_at,
                game_date=excluded.game_date,
                season=excluded.season,
                season_type=excluded.season_type,
                home_team_id=excluded.home_team_id,
                away_team_id=excluded.away_team_id
            """,
            (
                game_id,
                timestamp,
                home_team_id,
                away_team_id,
                season,
                season_type,
                game_date,
                json.dumps(traditional_data),
                json.dumps(advanced_data)
            )
        )
    except sqlite3.Error as err:
        logger.error("Error saving raw JSON for game %s: %s", game_id, err)


def process_game(
    game_id: str,
    season: str = "2024-25",
    season_type: str = "Regular Season",
    db_path: str = "Data/TeamData.sqlite",
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Run pipeline ETL for a single game using V3 box score endpoints.
    Check database cache → Fetch if missing → Compute derivatives → Validate → Save.
    """
    ensure_schema(db_path)
    conn = get_connection(db_path)
    
    try:
        # Check if already processed in team_game_advanced
        if not overwrite:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM team_game_advanced WHERE game_id = ?", (game_id,))
            team_g_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM player_game_log WHERE game_id = ?", (game_id,))
            player_g_count = cursor.fetchone()[0]
            if team_g_count >= 2 and player_g_count > 0:
                logger.info("Game %s already fully computed and stored.", game_id)
                return {"game_id": game_id, "status": "cached"}

        # Check raw cache first
        cached_raw = get_game_from_db(conn, game_id)
        client = get_client()

        if cached_raw:
            trad_data, adv_data = cached_raw
            logger.info("Loaded raw JSON from cache for game %s", game_id)
        else:
            logger.info("Fetching raw box scores from NBA Stats API for game %s", game_id)
            trad_data = client.boxscore_traditional(game_id)
            adv_data = client.boxscore_advanced(game_id)

        # Get Team totals from traditional boxscore (V3 structure)
        box_score_trad = trad_data.get("boxScoreTraditional", {})
        home_team_data = box_score_trad.get("homeTeam", {})
        away_team_data = box_score_trad.get("awayTeam", {})
        
        home_team_id = int(box_score_trad.get("homeTeamId", home_team_data.get("teamId", 0)))
        away_team_id = int(box_score_trad.get("awayTeamId", away_team_data.get("teamId", 0)))

        # Determine if game had overtime periods
        # Calculate from team minutes: normal is 240 min (48 mins * 5 players)
        # 1 OT is 265 min, 2 OT is 290 min, etc.
        # ot_periods = round((total_minutes - 240.0) / 25.0)
        t1_stats = home_team_data.get("statistics", {})
        t1_min = parse_minutes(t1_stats.get("minutes", 240))
        ot_periods = max(0, int(round((t1_min - 240.0) / 25.0)))
        if ot_periods > 0:
            logger.info("Game %s identified with %d OT period(s)", game_id, ot_periods)

        team_a = extract_team_traditional(home_team_data, game_id, ot_periods)
        team_b = extract_team_traditional(away_team_data, game_id, ot_periods)
        
        # Link opponents
        team_a.opp_team_id = team_b.team_id
        team_a.opp_pts = team_b.pts
        team_b.opp_team_id = team_a.team_id
        team_b.opp_pts = team_a.pts

        # Fetch/use game header metadata for date
        summary_sets = client.boxscore_summary(game_id)
        game_date_est = None
        for key in ("GameSummary", "GameHeader"):
            if key in summary_sets and len(summary_sets[key]) > 0:
                game_date_est = summary_sets[key][0].get("GAME_DATE_EST", None)
                if game_date_est:
                    break
        if game_date_est:
            # "2024-10-22T00:00:00" -> "2024-10-22"
            game_date = game_date_est.split("T")[0]
        else:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # Save raw JSONs to database
        if not cached_raw or overwrite:
            save_raw_game_to_db(
                conn, game_id, home_team_id, away_team_id, season, season_type, game_date, trad_data, adv_data
            )

        # Compute advanced ratings
        stats_a, stats_b = compute_game_advanced(team_a, team_b)

        # Validate with official advanced statistics (V3 structure)
        box_score_adv = adv_data.get("boxScoreAdvanced", {})
        home_adv = box_score_adv.get("homeTeam", {})
        away_adv = box_score_adv.get("awayTeam", {})
        
        adv_map = {
            int(home_adv.get("teamId", 0)): home_adv.get("statistics", {}),
            int(away_adv.get("teamId", 0)): away_adv.get("statistics", {})
        }

        # Perform validation for both teams
        for stats in (stats_a, stats_b):
            off_stats = adv_map.get(stats.team_id)
            if off_stats:
                off_pace = float(off_stats.get("pace", 0.0))
                off_ortg = float(off_stats.get("offensiveRating", 0.0))
                off_drtg = float(off_stats.get("defensiveRating", 0.0))
                
                validate_game_stats(
                    db_conn=conn,
                    game_id=game_id,
                    team_id=stats.team_id,
                    our_pace=stats.pace,
                    our_ortg=stats.off_rating,
                    our_drtg=stats.def_rating,
                    official_pace=off_pace,
                    official_ortg=off_ortg,
                    official_drtg=off_drtg,
                    threshold=1.0
                )

        # Save computed advanced stats to database
        save_computed_game_stats(conn, stats_a, season, season_type, game_date, pts=team_a.pts, opp_pts=team_b.pts)
        save_computed_game_stats(conn, stats_b, season, season_type, game_date, pts=team_b.pts, opp_pts=team_a.pts)

        # Save players and player game logs
        save_players_and_game_log(conn, game_id, game_date, home_team_data.get("players", []), home_team_id)
        save_players_and_game_log(conn, game_id, game_date, away_team_data.get("players", []), away_team_id)

        conn.commit()

        logger.info("Successfully processed game %s", game_id)
        return {
            "game_id": game_id,
            "status": "processed",
            "team_a_rating": stats_a.off_rating,
            "team_b_rating": stats_b.off_rating
        }

    except Exception as exc:
        conn.rollback()
        logger.error("Error processing game %s: %s", game_id, exc, exc_info=True)
        # Log error in database raw_scrape_log
        try:
            timestamp = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO raw_scrape_log (logged_at, game_id, endpoint, status, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, game_id, "nba_pipeline", "error", str(exc))
            )
            conn.commit()
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()
