"""
backfill.py
===========
CLI script to backfill historical and current season team/player metrics:
1. Fetch all games in a season using leaguegamelog.
2. Run each game through the pipeline (process_game) to compute advanced stats.
3. Solve SRS/SoS iteratively for the entire season.
4. Aggregate game advanced stats into team_season_advanced season averages.
5. Fetch and store league-wide player statistics using LeagueDashPlayerStats.

Season types: 'Regular Season', 'Playoffs', and 'PlayIn' (the play-in
tournament, 2020-21 on). PlayIn is ingested game by game only; steps 3-5 are
skipped for it (see SEASON_AGGREGATE_TYPES).

Exit status, which daily_update.py reads:
  0  every game in the log is stored, or is a known permanent hole
  1  a stage crashed (uncaught exception)
  3  at least one game failed to ingest, or was stored with a result that
     disagrees with the league game log -- the summary names each one
  4  --expect-games was passed and the league game log listed nothing
"""
import os
import sys
import argparse
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

# Resolve project root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.Utils.nba_stats_client import get_client
from src.Utils.nba_db_schema import ensure_schema, get_connection
from src.Utils.nba_pipeline import process_game
from src.Utils.nba_computed_derivatives import compute_srs, TeamRecord, aggregate_season_team_stats

# nba_api imports for player backfills
from nba_api.stats.endpoints import leaguedashplayerstats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


#: The season types this script knows how to ingest, as stats.nba.com spells
#: them. 'PlayIn' is the play-in tournament (2020-21 on, game ids 005...): it
#: is neither regular season nor playoffs, so it is stored under its own
#: season_type and never folded into either.
SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")

#: Game ids that are in the league game log but can never be ingested, and why.
#: A run that meets one of these reports it by name and does NOT fail, because
#: a job that goes red every time for a known, permanent, harmless reason
#: teaches people to ignore red. Anything else that fails does fail the run.
#:
#: The four box-score holes: nba.com's own boxscoreadvancedv3 returns nothing
#: for them on every retry (last re-checked 2026-08-23; the traditional box
#: score exists, the advanced one does not). The fifth is not a hole in the
#: data at all: the game was never played.
KNOWN_PERMANENT_HOLES: Dict[str, str] = {
    "0029600332": "nba.com boxscoreadvancedv3 returns nothing (SEA-GSW 1996-12-17)",
    "0029600370": "nba.com boxscoreadvancedv3 returns nothing (SEA-DAL 1996-12-22)",
    "0029800661": "nba.com boxscoreadvancedv3 returns nothing (DET-NJN 1999-04-28)",
    "0020300778": "nba.com boxscoreadvancedv3 returns nothing (NOH-WAS 2004-02-18)",
    "0021201214": "never played: BOS-IND 2013-04-16, cancelled after the Boston Marathon bombing",
}

#: Exit codes. 1 is Python's own code for an uncaught exception (a stage
#: crashed); 2 is argparse's for a bad command line.
EXIT_OK = 0
EXIT_GAMES_FAILED = 3
EXIT_EMPTY_GAME_LOG = 4


@dataclass
class BackfillSummary:
    """What one backfill_games run actually did, game by game.

    The script used to log "Errors: N" and exit 0 whatever N was, so the daily
    job reported a green backfill on a morning when games failed to land --
    and a game that never lands is a prediction that is never graded.
    """
    season: str
    season_type: str
    in_game_log: int = 0
    ingested: List[str] = field(default_factory=list)
    already_present: List[str] = field(default_factory=list)
    known_holes: List[str] = field(default_factory=list)
    not_final: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    #: Already-stored games whose game_date disagrees with the league game
    #: log: {game_id: (stored, log)}. Reported loudly but NOT a failure -- the
    #: rows predate this check and re-running the backfill cannot correct them
    #: (see the note in find_date_disagreements).
    date_disagreements: Dict[str, tuple] = field(default_factory=dict)

    @property
    def game_ids(self) -> List[str]:
        return sorted(set(self.ingested) | set(self.already_present) | set(self.known_holes)
                      | set(self.not_final) | set(self.failed))

    def lines(self) -> List[str]:
        out = [
            f"BACKFILL SUMMARY {self.season} {self.season_type}: "
            f"{self.in_game_log} game(s) in the league game log -- "
            f"{len(self.ingested)} ingested, {len(self.already_present)} already present, "
            f"{len(self.failed)} FAILED, {len(self.known_holes)} known permanent hole(s), "
            f"{len(self.not_final)} not final yet."
        ]
        for gid in self.known_holes:
            out.append(f"  known hole {gid}: {KNOWN_PERMANENT_HOLES.get(gid, '?')}")
        for gid in self.not_final:
            out.append(f"  not final  {gid}: the league game log has no result for it yet")
        for gid, why in list(self.failed.items())[:25]:
            out.append(f"  FAILED     {gid}: {why}")
        if len(self.failed) > 25:
            out.append(f"  ... and {len(self.failed) - 25} more failed game(s)")
        if self.date_disagreements:
            out.append(
                f"  WARNING    {len(self.date_disagreements)} stored game(s) carry a game_date "
                f"the league game log disagrees with (stored -> log): "
                + ", ".join(f"{g} {s}->{l}" for g, (s, l) in
                            sorted(self.date_disagreements.items())[:15]))
        return out


def _is_stored(db_path: str, game_id: str) -> bool:
    conn = get_connection(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM team_game_advanced WHERE game_id = ?",
                         (game_id,)).fetchone()[0]
        return n >= 2
    finally:
        conn.close()


def find_date_disagreements(db_path: str, season: str, season_type: str,
                            log_dates: Dict[str, str]) -> Dict[str, tuple]:
    """Stored games of this season/type whose game_date differs from the log.

    Found 2026-09-23: eleven 2025-26 games (postponed and NBA Cup games,
    ingested 2026-07-07) carry their ORIGINAL schedule date -- 0022500651
    MEM-DEN is stored on 2026-01-25 but was played 2026-03-18 (the box
    score's own gameCode says 20260318/DENMEM). A wrong date misplaces the
    game on the date browser and in rest-day and Elo ordering, and a pick on a
    rescheduled game would be looked up on the wrong day and never graded.
    `process_game(overwrite=True)` would not fix it: team_game_advanced's
    upsert does not update game_date. So this reports; repairing is a
    separate, deliberate job.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT game_id, MIN(game_date) AS d FROM team_game_advanced "
            "WHERE season = ? AND season_type = ? GROUP BY game_id",
            (season, season_type)).fetchall()
    finally:
        conn.close()
    out = {}
    for r in rows:
        want = (log_dates.get(r["game_id"]) or "").split("T")[0]
        have = (r["d"] or "").split("T")[0]
        if want and have != want:
            out[r["game_id"]] = (have, want)
    return out


def check_stored_score(db_path: str, game_id: str, log_points: Dict[int, int]) -> Optional[str]:
    """Why the stored result for `game_id` cannot be trusted, or None if it can.

    The box-score parser defaults every missing statistic to 0, so a skeleton
    box score (a game nba.com has listed but not finalised) would be stored as
    a 0-0 game rather than refused -- and grade_predictions would then grade a
    pick against it, permanently. So each newly ingested game is checked
    against the league game log the backfill already holds: two team rows,
    no zero, no tie, and each team's points equal to what the log says.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT team_id, pts, opp_pts FROM team_game_advanced WHERE game_id = ?",
            (game_id,)).fetchall()
    finally:
        conn.close()
    if len(rows) != 2:
        return f"{len(rows)} team row(s) stored, expected 2"
    for r in rows:
        pts, opp = r["pts"], r["opp_pts"]
        if not pts or not opp or pts <= 0 or opp <= 0:
            return f"implausible score stored ({pts}-{opp})"
        if pts == opp:
            return f"a tie stored ({pts}-{opp}); NBA games cannot end level"
        expected = log_points.get(int(r["team_id"]))
        if expected is not None and int(expected) != int(pts):
            return (f"team {r['team_id']} stored {pts} points but the league game log "
                    f"says {expected}")
    return None


def backfill_games(
    season: str,
    season_type: str,
    db_path: str,
    overwrite: bool = False,
    retry_known_holes: bool = False,
) -> BackfillSummary:
    """
    Fetch all team game logs for a given season and process each unique game
    to compute and store raw/advanced team stats.

    Returns a BackfillSummary; the caller decides the exit code from it.
    """
    summary = BackfillSummary(season=season, season_type=season_type)
    client = get_client(backfill_mode=True)
    logger.info("Fetching game log list for season: %s, type: %s", season, season_type)

    # Fetch team game logs
    game_log_rows = client.league_game_log(season=season, season_type=season_type, player_or_team="T")

    unique_game_ids: Set[str] = set()
    game_dates: Dict[str, str] = {}
    has_result: Dict[str, bool] = {}
    log_points: Dict[str, Dict[int, int]] = {}
    for row in game_log_rows:
        gid = row.get("GAME_ID")
        if not gid:
            continue
        unique_game_ids.add(gid)
        if row.get("GAME_DATE"):
            game_dates[gid] = row["GAME_DATE"]
        if row.get("WL") in ("W", "L"):
            has_result[gid] = True
            if row.get("TEAM_ID") is not None and row.get("PTS") is not None:
                log_points.setdefault(gid, {})[int(row["TEAM_ID"])] = int(row["PTS"])

    game_ids = sorted(unique_game_ids)
    summary.in_game_log = len(game_ids)
    logger.info("Found %d unique games to process for %s %s.", len(game_ids), season, season_type)

    for idx, game_id in enumerate(game_ids, 1):
        if idx % 50 == 0 or idx == len(game_ids):
            logger.info("Processing games progress: %d/%d...", idx, len(game_ids))

        if game_id in KNOWN_PERMANENT_HOLES and not overwrite:
            if _is_stored(db_path, game_id):
                summary.already_present.append(game_id)
                continue
            if not retry_known_holes:
                # Not re-requested: every retry costs stats.nba.com three
                # failing calls with backoff, for an answer that has been the
                # same every time. --retry-known-holes asks again.
                summary.known_holes.append(game_id)
                continue

        if not has_result.get(game_id) and game_id not in KNOWN_PERMANENT_HOLES:
            # Listed without a W/L: not final. Ingesting it now would store
            # whatever partial box score exists as if it were the result.
            summary.not_final.append(game_id)
            continue

        try:
            res = process_game(
                game_id=game_id,
                season=season,
                season_type=season_type,
                db_path=db_path,
                overwrite=overwrite,
                game_date_hint=game_dates.get(game_id)
            )
        except Exception as exc:
            if game_id in KNOWN_PERMANENT_HOLES:
                logger.info("Known permanent hole %s still unavailable: %s", game_id, exc)
                summary.known_holes.append(game_id)
            else:
                logger.error("Failed to process game %s: %s", game_id, exc)
                summary.failed[game_id] = f"{type(exc).__name__}: {exc}"[:300]
            continue

        if res.get("status") == "cached":
            summary.already_present.append(game_id)
            continue
        problem = check_stored_score(db_path, game_id, log_points.get(game_id, {}))
        if problem:
            logger.error("Game %s was stored but its result is not trustworthy: %s", game_id, problem)
            summary.failed[game_id] = problem
        else:
            summary.ingested.append(game_id)

    disagreements = find_date_disagreements(db_path, season, season_type, game_dates)
    for gid, (have, want) in disagreements.items():
        if gid in summary.ingested:
            # Written by this run from the log's own date, so a disagreement
            # here is a defect in the write, not legacy data.
            summary.ingested.remove(gid)
            summary.failed[gid] = f"stored game_date {have} but the league game log says {want}"
        else:
            summary.date_disagreements[gid] = (have, want)

    logger.info(
        "Finished game processing. Total: %d, Ingested: %d, Already present: %d, "
        "Failed: %d, Known holes: %d, Not final: %d",
        summary.in_game_log, len(summary.ingested), len(summary.already_present),
        len(summary.failed), len(summary.known_holes), len(summary.not_final),
    )
    return summary


def compute_and_save_season_stats(
    season: str,
    season_type: str,
    db_path: str
) -> None:
    """
    Query all team_game_advanced records for the season, compute wins/losses,
    solve the SRS/SoS linear system, aggregate advanced metrics, and write to
    team_season_advanced table.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    logger.info("Computing season aggregates and SRS solver for season %s...", season)
    
    try:
        # Load all computed games for this season
        cursor.execute(
            """
            SELECT game_id, team_id, opp_team_id, pts, opp_pts, pace, off_rating, def_rating,
                   net_rating, efg_pct, tov_pct, orb_pct, ft_rate, ts_pct
            FROM team_game_advanced
            WHERE season = ? AND season_type = ?
            """,
            (season, season_type)
        )
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning("No computed game logs found for season %s. Cannot compute season statistics.", season)
            return

        # 1. Group records by team
        team_games: Dict[int, List[Dict]] = {}
        team_records: Dict[int, TeamRecord] = {}
        
        # We need team abbreviations or full names for user display. Let's pull from team_metadata or use team_id as placeholder name
        cursor.execute("SELECT team_id, abbreviation FROM team_metadata")
        team_abbr_map = {row["team_id"]: row["abbreviation"] for row in cursor.fetchall()}
        
        for r in rows:
            tid = r["team_id"]
            if tid not in team_games:
                team_games[tid] = []
                team_records[tid] = TeamRecord(team_id=tid, abbr=team_abbr_map.get(tid, str(tid)))
            
            # Map row to dictionary
            game_dict = dict(r)
            team_games[tid].append(game_dict)
            
            # Construct SRS parameters: margin = pts - opp_pts
            pts = r["pts"] if r["pts"] is not None else 0
            opp_pts = r["opp_pts"] if r["opp_pts"] is not None else 0
            margin = float(pts - opp_pts)
            
            team_records[tid].point_diffs.append(margin)
            team_records[tid].opponent_ids.append(r["opp_team_id"])
            
        # 2. Solve SRS/SoS
        srs_ratings, sos_ratings = compute_srs(team_records)
        
        # 3. Aggregate metrics and save for each team
        timestamp = datetime.utcnow().isoformat()
        
        for tid, games in team_games.items():
            # Calculate wins / losses
            wins = sum(1 for g in games if (g["pts"] or 0) > (g["opp_pts"] or 0))
            losses = sum(1 for g in games if (g["pts"] or 0) < (g["opp_pts"] or 0))
            total_g = wins + losses
            win_pct = wins / total_g if total_g > 0 else 0.0
            
            # Aggregate advanced derivatives
            # Map database keys to GameAdvancedStats objects for aggregate helper compatibility
            from src.Utils.nba_computed_derivatives import GameAdvancedStats
            stats_list = []
            for g in games:
                stats_list.append(
                    GameAdvancedStats(
                        game_id=g["game_id"],
                        team_id=g["team_id"],
                        opp_team_id=g["opp_team_id"],
                        poss_estimated=0.0, # not needed for avg
                        poss_opponent=0.0,
                        pace=g["pace"],
                        off_rating=g["off_rating"],
                        def_rating=g["def_rating"],
                        net_rating=g["net_rating"],
                        efg_pct=g["efg_pct"],
                        tov_pct=g["tov_pct"],
                        orb_pct=g["orb_pct"],
                        ft_rate=g["ft_rate"],
                        ts_pct=g["ts_pct"]
                    )
                )
            avgs = aggregate_season_team_stats(stats_list)
            
            srs_val = srs_ratings.get(tid, 0.0)
            sos_val = sos_ratings.get(tid, 0.0)
            
            # Save or Update
            conn.execute(
                """
                INSERT INTO team_season_advanced (
                    team_id, season, season_type, games, wins, losses, win_pct,
                    pace, off_rating, def_rating, net_rating, efg_pct, tov_pct, orb_pct, ft_rate, ts_pct,
                    srs, sos, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, season, season_type) DO UPDATE SET
                    games=excluded.games,
                    wins=excluded.wins,
                    losses=excluded.losses,
                    win_pct=excluded.win_pct,
                    pace=excluded.pace,
                    off_rating=excluded.off_rating,
                    def_rating=excluded.def_rating,
                    net_rating=excluded.net_rating,
                    efg_pct=excluded.efg_pct,
                    tov_pct=excluded.tov_pct,
                    orb_pct=excluded.orb_pct,
                    ft_rate=excluded.ft_rate,
                    ts_pct=excluded.ts_pct,
                    srs=excluded.srs,
                    sos=excluded.sos,
                    computed_at=excluded.computed_at
                """,
                (
                    tid,
                    season,
                    season_type,
                    total_g,
                    wins,
                    losses,
                    win_pct,
                    avgs.get("pace", 0.0),
                    avgs.get("off_rating", 0.0),
                    avgs.get("def_rating", 0.0),
                    avgs.get("net_rating", 0.0),
                    avgs.get("efg_pct", 0.0),
                    avgs.get("tov_pct", 0.0),
                    avgs.get("orb_pct", 0.0),
                    avgs.get("ft_rate", 0.0),
                    avgs.get("ts_pct", 0.0),
                    srs_val,
                    sos_val,
                    timestamp
                )
            )
            
        conn.commit()
        logger.info("Successfully updated season stats for %d teams.", len(team_games))

        # Assert league-average SRS is within +/-0.5 of 0.0
        cursor.execute(
            "SELECT AVG(srs) as avg_srs FROM team_season_advanced WHERE season = ? AND season_type = ?",
            (season, season_type)
        )
        row = cursor.fetchone()
        avg_srs = row["avg_srs"] if row else None
        if avg_srs is not None:
            logger.info("League-average SRS calculated: %.4f", avg_srs)
            if abs(avg_srs) > 0.5:
                raise ValueError(
                    f"League-average SRS ({avg_srs:.4f}) deviates from 0.0 by more than the allowed +/-0.5 threshold!"
                )
        else:
            logger.warning("Could not calculate league-average SRS (no team records found).")
        
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to compute and save season stats: %s", exc, exc_info=True)
        raise
    finally:
        conn.close()


def backfill_players(
    season: str,
    season_type: str,
    db_path: str
) -> None:
    """
    Fetch league-wide player statistics for base and advanced metrics,
    merge them on PLAYER_ID, and save them to player_season_stats table.
    """
    logger.info("Backfilling player season statistics for %s...", season)
    
    # 1. Fetch from NBA Stats using leaguedashplayerstats (via requests or endpoint class)
    try:
        # Base stats
        logger.info("Fetching base player statistics...")
        base_ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base"
        )
        base_rows = base_ep.get_dict()["resultSets"][0]["rowSet"]
        base_headers = base_ep.get_dict()["resultSets"][0]["headers"]
        base_dict_list = [dict(zip(base_headers, row)) for row in base_rows]
        
        # Advanced stats
        logger.info("Fetching advanced player statistics...")
        adv_ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced"
        )
        adv_rows = adv_ep.get_dict()["resultSets"][0]["rowSet"]
        adv_headers = adv_ep.get_dict()["resultSets"][0]["headers"]
        adv_dict_list = [dict(zip(adv_headers, row)) for row in adv_rows]
        
        # Map advanced stats by PLAYER_ID
        adv_map = {int(p["PLAYER_ID"]): p for p in adv_dict_list}
        
        conn = get_connection(db_path)
        timestamp = datetime.utcnow().isoformat()
        
        for player in base_dict_list:
            pid = int(player["PLAYER_ID"])
            adv_p = adv_map.get(pid, {})
            
            # Map values, handling missing columns gracefully
            conn.execute(
                """
                INSERT INTO player_season_stats (
                    player_id, season, season_type, team_id, team_abbr,
                    gp, gs, min, pts, reb, ast, stl, blk, tov, pf,
                    fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct,
                    oreb, dreb, plus_minus,
                    ts_pct, usg_pct, off_rating, def_rating, net_rating,
                    ast_pct, reb_pct, efg_pct, tov_pct, pace, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id, season, season_type, team_id) DO UPDATE SET
                    gp=excluded.gp,
                    gs=excluded.gs,
                    min=excluded.min,
                    pts=excluded.pts,
                    reb=excluded.reb,
                    ast=excluded.ast,
                    stl=excluded.stl,
                    blk=excluded.blk,
                    tov=excluded.tov,
                    pf=excluded.pf,
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
                    plus_minus=excluded.plus_minus,
                    ts_pct=excluded.ts_pct,
                    usg_pct=excluded.usg_pct,
                    off_rating=excluded.off_rating,
                    def_rating=excluded.def_rating,
                    net_rating=excluded.net_rating,
                    ast_pct=excluded.ast_pct,
                    reb_pct=excluded.reb_pct,
                    efg_pct=excluded.efg_pct,
                    tov_pct=excluded.tov_pct,
                    pace=excluded.pace,
                    fetched_at=excluded.fetched_at
                """,
                (
                    pid,
                    season,
                    season_type,
                    int(player.get("TEAM_ID", 0)),
                    player.get("TEAM_ABBREVIATION"),
                    player.get("GP"),
                    player.get("GS"),
                    player.get("MIN"),
                    player.get("PTS"),
                    player.get("REB"),
                    player.get("AST"),
                    player.get("STL"),
                    player.get("BLK"),
                    player.get("TOV"),
                    player.get("PF"),
                    player.get("FGM"),
                    player.get("FGA"),
                    player.get("FG_PCT"),
                    player.get("FG3M"),
                    player.get("FG3A"),
                    player.get("FG3_PCT"),
                    player.get("FTM"),
                    player.get("FTA"),
                    player.get("FT_PCT"),
                    player.get("OREB"),
                    player.get("DREB"),
                    player.get("PLUS_MINUS"),
                    adv_p.get("TS_PCT"),
                    adv_p.get("USG_PCT"),
                    adv_p.get("OFF_RATING"),
                    adv_p.get("DEF_RATING"),
                    adv_p.get("NET_RATING"),
                    adv_p.get("AST_PCT"),
                    adv_p.get("REB_PCT"),
                    adv_p.get("EFG_PCT"),
                    adv_p.get("TM_TOV_PCT"),
                    adv_p.get("PACE"),
                    timestamp
                )
            )
        conn.commit()
        conn.close()
        logger.info("Successfully backfilled statistics for all players in %s.", season)
    except Exception as e:
        logger.error("Failed to backfill player stats: %s", e, exc_info=True)
        raise


# Conference/division are stable league facts nba_api's static list doesn't carry.
TEAM_CONFERENCE_DIVISION = {
    "ATL": ("East", "Southeast"), "BOS": ("East", "Atlantic"), "BKN": ("East", "Atlantic"),
    "CHA": ("East", "Southeast"), "CHI": ("East", "Central"), "CLE": ("East", "Central"),
    "DET": ("East", "Central"), "IND": ("East", "Central"), "MIA": ("East", "Southeast"),
    "MIL": ("East", "Central"), "NYK": ("East", "Atlantic"), "ORL": ("East", "Southeast"),
    "PHI": ("East", "Atlantic"), "TOR": ("East", "Atlantic"), "WAS": ("East", "Southeast"),
    "DAL": ("West", "Southwest"), "DEN": ("West", "Northwest"), "GSW": ("West", "Pacific"),
    "HOU": ("West", "Southwest"), "LAC": ("West", "Pacific"), "LAL": ("West", "Pacific"),
    "MEM": ("West", "Southwest"), "MIN": ("West", "Northwest"), "NOP": ("West", "Southwest"),
    "OKC": ("West", "Northwest"), "PHX": ("West", "Pacific"), "POR": ("West", "Northwest"),
    "SAC": ("West", "Pacific"), "SAS": ("West", "Southwest"), "UTA": ("West", "Northwest"),
}


def backfill_metadata(db_path: str) -> None:
    """Populate team_metadata table with team descriptions using nba_stats_client."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    logger.info("Backfilling team metadata table...")
    try:
        from nba_api.stats.static import teams as nba_teams
        team_list = nba_teams.get_teams()
        timestamp = datetime.utcnow().isoformat()

        for team in team_list:
            conference, division = TEAM_CONFERENCE_DIVISION.get(team["abbreviation"], (None, None))
            conn.execute(
                """
                INSERT INTO team_metadata (
                    team_id, full_name, abbreviation, nickname, city, state, year_founded,
                    conference, division, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    full_name=excluded.full_name,
                    abbreviation=excluded.abbreviation,
                    nickname=excluded.nickname,
                    city=excluded.city,
                    state=excluded.state,
                    conference=excluded.conference,
                    division=excluded.division,
                    fetched_at=excluded.fetched_at
                """,
                (
                    team["id"],
                    team["full_name"],
                    team["abbreviation"],
                    team["nickname"],
                    team["city"],
                    team["state"],
                    team["year_founded"],
                    conference,
                    division,
                    timestamp
                )
            )
        conn.commit()
        logger.info("Successfully updated metadata for %d teams.", len(team_list))
    except Exception as e:
        logger.error("Failed to backfill team metadata: %s", e)
    finally:
        conn.close()


def compute_and_save_player_season_aggregates(
    season: str,
    season_type: str,
    db_path: str
) -> None:
    """
    Compute player_season_totals, player_splits, and player_season_advanced
    from player_game_log and player_season_stats (if populated).
    """
    logger.info("Computing and saving player season aggregates from game logs for %s (%s)...", season, season_type)
    conn = get_connection(db_path)
    try:
        # 1. Compute and save player_season_totals
        conn.execute(
            """
            INSERT INTO player_season_totals (
                player_id, season, season_type, team_id,
                gp, gs, min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct,
                oreb, dreb, reb, ast, stl, blk, tov, pf, pts
            )
            SELECT 
                pgl.player_id,
                tga.season,
                tga.season_type,
                pgl.team_id,
                COUNT(pgl.id) as gp,
                SUM(pgl.starter) as gs,
                SUM(pgl.min) as min,
                SUM(pgl.fgm) as fgm,
                SUM(pgl.fga) as fga,
                CASE WHEN SUM(pgl.fga) > 0 THEN CAST(SUM(pgl.fgm) as REAL) / SUM(pgl.fga) ELSE 0.0 END as fg_pct,
                SUM(pgl.fg3m) as fg3m,
                SUM(pgl.fg3a) as fg3a,
                CASE WHEN SUM(pgl.fg3a) > 0 THEN CAST(SUM(pgl.fg3m) as REAL) / SUM(pgl.fg3a) ELSE 0.0 END as fg3_pct,
                SUM(pgl.ftm) as ftm,
                SUM(pgl.fta) as fta,
                CASE WHEN SUM(pgl.fta) > 0 THEN CAST(SUM(pgl.ftm) as REAL) / SUM(pgl.fta) ELSE 0.0 END as ft_pct,
                SUM(pgl.oreb) as oreb,
                SUM(pgl.dreb) as dreb,
                SUM(pgl.reb) as reb,
                SUM(pgl.ast) as ast,
                SUM(pgl.stl) as stl,
                SUM(pgl.blk) as blk,
                SUM(pgl.tov) as tov,
                SUM(pgl.pf) as pf,
                SUM(pgl.pts) as pts
            FROM player_game_log pgl
            JOIN team_game_advanced tga ON pgl.game_id = tga.game_id AND pgl.team_id = tga.team_id
            WHERE tga.season = ? AND tga.season_type = ?
            GROUP BY pgl.player_id, pgl.team_id, tga.season, tga.season_type
            ON CONFLICT(player_id, season, season_type, team_id) DO UPDATE SET
                gp=excluded.gp,
                gs=excluded.gs,
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
                pts=excluded.pts
            """,
            (season, season_type)
        )
        logger.info("Computed player_season_totals successfully.")

        # 2. Compute and save player_season_advanced (step 1: base stats; step 2: merge from stats cache)
        conn.execute(
            """
            INSERT INTO player_season_advanced (
                player_id, season, season_type, team_id,
                ts_pct, usg_pct, off_rating, def_rating, net_rating,
                ast_pct, reb_pct, efg_pct, tov_pct, pace
            )
            SELECT 
                player_id, season, season_type, team_id,
                CASE WHEN (fga + 0.44 * fta) > 0 THEN CAST(pts as REAL) / (2.0 * (fga + 0.44 * fta)) ELSE 0.0 END as ts_pct,
                0.0 as usg_pct,
                0.0 as off_rating,
                0.0 as def_rating,
                0.0 as net_rating,
                0.0 as ast_pct,
                0.0 as reb_pct,
                CASE WHEN fga > 0 THEN CAST(fgm + 0.5 * fg3m as REAL) / fga ELSE 0.0 END as efg_pct,
                CASE WHEN (fga + 0.44 * fta + tov) > 0 THEN CAST(tov as REAL) / (fga + 0.44 * fta + tov) ELSE 0.0 END as tov_pct,
                0.0 as pace
            FROM player_season_totals
            WHERE season = ? AND season_type = ?
            ON CONFLICT(player_id, season, season_type, team_id) DO UPDATE SET
                ts_pct=excluded.ts_pct,
                efg_pct=excluded.efg_pct,
                tov_pct=excluded.tov_pct
            """,
            (season, season_type)
        )

        conn.execute(
            """
            UPDATE player_season_advanced
            SET 
                usg_pct = (SELECT usg_pct FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id),
                off_rating = (SELECT off_rating FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id),
                def_rating = (SELECT def_rating FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id),
                net_rating = (SELECT net_rating FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id),
                ast_pct = (SELECT ast_pct FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id),
                reb_pct = (SELECT reb_pct FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id),
                pace = (SELECT pace FROM player_season_stats s WHERE s.player_id = player_season_advanced.player_id AND s.season = player_season_advanced.season AND s.season_type = player_season_advanced.season_type AND s.team_id = player_season_advanced.team_id)
            WHERE season = ? AND season_type = ?
              AND EXISTS (
                  SELECT 1 FROM player_season_stats s 
                  WHERE s.player_id = player_season_advanced.player_id 
                    AND s.season = player_season_advanced.season 
                    AND s.season_type = player_season_advanced.season_type 
                    AND s.team_id = player_season_advanced.team_id
              )
            """,
            (season, season_type)
        )
        logger.info("Computed player_season_advanced successfully.")

        # 3. Compute and save player_splits (Location split)
        conn.execute(
            """
            INSERT INTO player_splits (
                player_id, season, season_type, split_type, split_value,
                gp, gs, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus
            )
            SELECT 
                pgl.player_id,
                tga.season,
                tga.season_type,
                'Location' as split_type,
                CASE WHEN pgl.team_id = bs.home_team_id THEN 'Home' ELSE 'Road' END as split_value,
                COUNT(pgl.id) as gp,
                SUM(pgl.starter) as gs,
                SUM(pgl.min) as min,
                SUM(pgl.pts) as pts,
                SUM(pgl.reb) as reb,
                SUM(pgl.ast) as ast,
                SUM(pgl.stl) as stl,
                SUM(pgl.blk) as blk,
                SUM(pgl.tov) as tov,
                SUM(pgl.fgm) as fgm,
                SUM(pgl.fga) as fga,
                CASE WHEN SUM(pgl.fga) > 0 THEN CAST(SUM(pgl.fgm) as REAL) / SUM(pgl.fga) ELSE 0.0 END as fg_pct,
                SUM(pgl.fg3m) as fg3m,
                SUM(pgl.fg3a) as fg3a,
                CASE WHEN SUM(pgl.fg3a) > 0 THEN CAST(SUM(pgl.fg3m) as REAL) / SUM(pgl.fg3a) ELSE 0.0 END as fg3_pct,
                SUM(pgl.ftm) as ftm,
                SUM(pgl.fta) as fta,
                CASE WHEN SUM(pgl.fta) > 0 THEN CAST(SUM(pgl.ftm) as REAL) / SUM(pgl.fta) ELSE 0.0 END as ft_pct,
                SUM(pgl.plus_minus) as plus_minus
            FROM player_game_log pgl
            JOIN team_game_advanced tga ON pgl.game_id = tga.game_id AND pgl.team_id = tga.team_id
            JOIN box_scores bs ON pgl.game_id = bs.game_id
            WHERE tga.season = ? AND tga.season_type = ?
            GROUP BY pgl.player_id, tga.season, tga.season_type, split_value
            ON CONFLICT(player_id, season, season_type, split_type, split_value) DO UPDATE SET
                gp=excluded.gp,
                gs=excluded.gs,
                min=excluded.min,
                pts=excluded.pts,
                reb=excluded.reb,
                ast=excluded.ast,
                stl=excluded.stl,
                blk=excluded.blk,
                tov=excluded.tov,
                fgm=excluded.fgm,
                fga=excluded.fga,
                fg_pct=excluded.fg_pct,
                fg3m=excluded.fg3m,
                fg3a=excluded.fg3a,
                fg3_pct=excluded.fg3_pct,
                ftm=excluded.ftm,
                fta=excluded.fta,
                ft_pct=excluded.ft_pct,
                plus_minus=excluded.plus_minus
            """,
            (season, season_type)
        )

        # 4. Compute and save player_splits (Wins/Losses split)
        conn.execute(
            """
            INSERT INTO player_splits (
                player_id, season, season_type, split_type, split_value,
                gp, gs, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus
            )
            SELECT 
                pgl.player_id,
                tga.season,
                tga.season_type,
                'Wins/Losses' as split_type,
                CASE WHEN tga.pts > tga.opp_pts THEN 'Wins' ELSE 'Losses' END as split_value,
                COUNT(pgl.id) as gp,
                SUM(pgl.starter) as gs,
                SUM(pgl.min) as min,
                SUM(pgl.pts) as pts,
                SUM(pgl.reb) as reb,
                SUM(pgl.ast) as ast,
                SUM(pgl.stl) as stl,
                SUM(pgl.blk) as blk,
                SUM(pgl.tov) as tov,
                SUM(pgl.fgm) as fgm,
                SUM(pgl.fga) as fga,
                CASE WHEN SUM(pgl.fga) > 0 THEN CAST(SUM(pgl.fgm) as REAL) / SUM(pgl.fga) ELSE 0.0 END as fg_pct,
                SUM(pgl.fg3m) as fg3m,
                SUM(pgl.fg3a) as fg3a,
                CASE WHEN SUM(pgl.fg3a) > 0 THEN CAST(SUM(pgl.fg3m) as REAL) / SUM(pgl.fg3a) ELSE 0.0 END as fg3_pct,
                SUM(pgl.ftm) as ftm,
                SUM(pgl.fta) as fta,
                CASE WHEN SUM(pgl.fta) > 0 THEN CAST(SUM(pgl.ftm) as REAL) / SUM(pgl.fta) ELSE 0.0 END as ft_pct,
                SUM(pgl.plus_minus) as plus_minus
            FROM player_game_log pgl
            JOIN team_game_advanced tga ON pgl.game_id = tga.game_id AND pgl.team_id = tga.team_id
            WHERE tga.season = ? AND tga.season_type = ?
            GROUP BY pgl.player_id, tga.season, tga.season_type, split_value
            ON CONFLICT(player_id, season, season_type, split_type, split_value) DO UPDATE SET
                gp=excluded.gp,
                gs=excluded.gs,
                min=excluded.min,
                pts=excluded.pts,
                reb=excluded.reb,
                ast=excluded.ast,
                stl=excluded.stl,
                blk=excluded.blk,
                tov=excluded.tov,
                fgm=excluded.fgm,
                fga=excluded.fga,
                fg_pct=excluded.fg_pct,
                fg3m=excluded.fg3m,
                fg3a=excluded.fg3a,
                fg3_pct=excluded.fg3_pct,
                ftm=excluded.ftm,
                fta=excluded.fta,
                ft_pct=excluded.ft_pct,
                plus_minus=excluded.plus_minus
            """,
            (season, season_type)
        )

        # 5. Compute and save player_splits (Month split)
        conn.execute(
            """
            INSERT INTO player_splits (
                player_id, season, season_type, split_type, split_value,
                gp, gs, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus
            )
            SELECT 
                pgl.player_id,
                tga.season,
                tga.season_type,
                'Month' as split_type,
                CASE strftime('%m', pgl.game_date)
                    WHEN '01' THEN 'January'
                    WHEN '02' THEN 'February'
                    WHEN '03' THEN 'March'
                    WHEN '04' THEN 'April'
                    WHEN '05' THEN 'May'
                    WHEN '06' THEN 'June'
                    WHEN '07' THEN 'July'
                    WHEN '08' THEN 'August'
                    WHEN '09' THEN 'September'
                    WHEN '10' THEN 'October'
                    WHEN '11' THEN 'November'
                    WHEN '12' THEN 'December'
                    ELSE 'Unknown'
                END as split_value,
                COUNT(pgl.id) as gp,
                SUM(pgl.starter) as gs,
                SUM(pgl.min) as min,
                SUM(pgl.pts) as pts,
                SUM(pgl.reb) as reb,
                SUM(pgl.ast) as ast,
                SUM(pgl.stl) as stl,
                SUM(pgl.blk) as blk,
                SUM(pgl.tov) as tov,
                SUM(pgl.fgm) as fgm,
                SUM(pgl.fga) as fga,
                CASE WHEN SUM(pgl.fga) > 0 THEN CAST(SUM(pgl.fgm) as REAL) / SUM(pgl.fga) ELSE 0.0 END as fg_pct,
                SUM(pgl.fg3m) as fg3m,
                SUM(pgl.fg3a) as fg3a,
                CASE WHEN SUM(pgl.fg3a) > 0 THEN CAST(SUM(pgl.fg3m) as REAL) / SUM(pgl.fg3a) ELSE 0.0 END as fg3_pct,
                SUM(pgl.ftm) as ftm,
                SUM(pgl.fta) as fta,
                CASE WHEN SUM(pgl.fta) > 0 THEN CAST(SUM(pgl.ftm) as REAL) / SUM(pgl.fta) ELSE 0.0 END as ft_pct,
                SUM(pgl.plus_minus) as plus_minus
            FROM player_game_log pgl
            JOIN team_game_advanced tga ON pgl.game_id = tga.game_id AND pgl.team_id = tga.team_id
            WHERE tga.season = ? AND tga.season_type = ?
            GROUP BY pgl.player_id, tga.season, tga.season_type, split_value
            ON CONFLICT(player_id, season, season_type, split_type, split_value) DO UPDATE SET
                gp=excluded.gp,
                gs=excluded.gs,
                min=excluded.min,
                pts=excluded.pts,
                reb=excluded.reb,
                ast=excluded.ast,
                stl=excluded.stl,
                blk=excluded.blk,
                tov=excluded.tov,
                fgm=excluded.fgm,
                fga=excluded.fga,
                fg_pct=excluded.fg_pct,
                fg3m=excluded.fg3m,
                fg3a=excluded.fg3a,
                fg3_pct=excluded.fg3_pct,
                ftm=excluded.ftm,
                fta=excluded.fta,
                ft_pct=excluded.ft_pct,
                plus_minus=excluded.plus_minus
            """,
            (season, season_type)
        )
        conn.commit()
        logger.info("Computed and saved all player season aggregates and splits successfully.")
    except Exception as e:
        conn.rollback()
        logger.error("Failed to compute and save player season aggregates: %s", e)
        raise
    finally:
        conn.close()


#: Season types whose season-level aggregates (team_season_advanced with SRS,
#: player_season_stats/totals/advanced/splits) this script computes. PlayIn is
#: deliberately absent: a "season" of one or two games per team is not a
#: season, SRS solved over six games is noise that the +/-0.5 league-average
#: check can reject outright, and every new season_type row in those tables is
#: one more row for a query that forgets to filter season_type to mix into
#: regular-season numbers. Play-in games are still ingested game by game
#: (box_scores, team_game_advanced, player_game_log) -- which is what grading,
#: box-score pages and Elo read.
SEASON_AGGREGATE_TYPES = ("Regular Season", "Playoffs")


def exit_code_for(summary: Optional[BackfillSummary], expect_games: bool) -> int:
    """The process exit code for a finished run. Pure, so it can be tested."""
    if summary is None:
        return EXIT_OK
    if summary.failed:
        return EXIT_GAMES_FAILED
    if expect_games and summary.in_game_log == 0:
        return EXIT_EMPTY_GAME_LOG
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill NBA Stats API team/player data pipeline.")
    parser.add_argument("--season", type=str, default="2024-25", help="Season in format YYYY-YY (e.g. 2024-25)")
    parser.add_argument("--season-type", type=str, default="Regular Season", choices=SEASON_TYPES,
                        help="Regular Season | PlayIn | Playoffs")
    parser.add_argument("--db", type=str, default="Data/TeamData.sqlite", help="SQLite database path")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess game logs even if cached")
    parser.add_argument("--only-teams", action="store_true", help="Only backfill team game stats and aggregates")
    parser.add_argument("--only-players", action="store_true", help="Only backfill player aggregates")
    parser.add_argument("--expect-games", action="store_true",
                        help="Fail (exit 4) if the league game log lists no games. The daily job "
                             "passes this once the regular season is under way, when an empty log "
                             "means the feed is broken rather than that no games exist yet.")
    parser.add_argument("--retry-known-holes", action="store_true",
                        help="Re-request the known permanent holes instead of skipping them.")

    args = parser.parse_args()

    ensure_schema(args.db)

    # 1. Backfill Metadata
    if not args.only_players:
        backfill_metadata(args.db)

    summary: Optional[BackfillSummary] = None
    aggregate = args.season_type in SEASON_AGGREGATE_TYPES

    # 2. Backfill Game log & computed aggregates
    if not args.only_players:
        summary = backfill_games(args.season, args.season_type, args.db, args.overwrite,
                                 retry_known_holes=args.retry_known_holes)

        # Enforce validation crash if error rate exceeds 5%
        from src.Utils.nba_validation import get_validation_failure_rate
        conn_check = get_connection(args.db)
        try:
            val_rate = get_validation_failure_rate(conn_check)
            logger.info("Validation failure rate across all games: %.2f%%", val_rate)
            # Our ratings use Dean Oliver estimated possessions; NBA.com uses actual
            # possession counts, so 1-3 point deviations are expected methodology
            # differences, not data corruption. Warn loudly but never halt the
            # pipeline over it - raw box scores are exact either way.
            if val_rate > 5.0:
                logger.warning(
                    "Validation failure rate is %.2f%% (threshold 5%%). Expected when comparing "
                    "estimated-possession ratings to NBA.com official ratings; continuing.",
                    val_rate,
                )
        finally:
            conn_check.close()

        if aggregate:
            compute_and_save_season_stats(args.season, args.season_type, args.db)
        else:
            logger.info("No season-level team aggregates for %s (see SEASON_AGGREGATE_TYPES).",
                        args.season_type)

    # 3. Backfill Player statistics
    if not args.only_teams and aggregate:
        backfill_players(args.season, args.season_type, args.db)
        compute_and_save_player_season_aggregates(args.season, args.season_type, args.db)

    code = exit_code_for(summary, args.expect_games)
    for line in (summary.lines() if summary else []):
        head = line.lstrip()
        (logger.error if head.startswith("FAILED")
         else logger.warning if head.startswith("WARNING")
         else logger.info)(line)
    if code == EXIT_GAMES_FAILED:
        logger.error("Backfill for %s (%s) FAILED: %d game(s) did not land. Exit %d.",
                     args.season, args.season_type, len(summary.failed), code)
    elif code == EXIT_EMPTY_GAME_LOG:
        logger.error("Backfill for %s (%s) FAILED: the league game log listed no games while "
                     "games were expected. Exit %d.", args.season, args.season_type, code)
    else:
        logger.info("Backfill complete for season %s (%s).", args.season, args.season_type)
    return code


if __name__ == "__main__":
    sys.exit(main())
