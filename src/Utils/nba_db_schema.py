"""
nba_db_schema.py
=================
Idempotent SQLite schema migration for the NBA Stats pipeline.
Run this once before the pipeline starts to ensure all tables exist.

All new tables are ADDITIVE – existing tables (TeamData, Odds, etc.) are
never modified or dropped.

Usage:
    from src.Utils.nba_db_schema import ensure_schema
    ensure_schema("Data/TeamData.sqlite")
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL statements (CREATE TABLE IF NOT EXISTS → fully idempotent)
# ---------------------------------------------------------------------------

_DDL = [
    # -------------------------------------------------------------------
    # 1.  Raw game box scores
    #     Stores the JSON payload from boxscoretraditionalv2 for durability
    #     and replay without hitting the API again.
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS box_scores (
        game_id         TEXT NOT NULL,
        fetched_at      TEXT NOT NULL,          -- ISO-8601 UTC
        home_team_id    INTEGER,
        away_team_id    INTEGER,
        season          TEXT,
        season_type     TEXT,
        game_date       TEXT,
        traditional_json TEXT,                  -- raw JSON blob (boxscoretraditionalv2)
        advanced_json    TEXT,                  -- raw JSON blob (boxscoreadvancedv2)
        pbp_json        TEXT,                   -- cache-first Play-by-Play event JSON
        PRIMARY KEY (game_id)
    )
    """,

    # -------------------------------------------------------------------
    # 2.  Per-game computed advanced stats (one row per team per game)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS team_game_advanced (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id         TEXT NOT NULL,
        team_id         INTEGER NOT NULL,
        opp_team_id     INTEGER NOT NULL,
        season          TEXT,
        season_type     TEXT,
        game_date       TEXT,

        -- Points
        pts             INTEGER,
        opp_pts         INTEGER,

        -- Possessions
        poss_estimated  REAL,
        poss_opponent   REAL,

        -- Pace (possessions per 48 min)
        pace            REAL,

        -- Ratings (points per 100 possessions)
        off_rating      REAL,
        def_rating      REAL,
        net_rating      REAL,

        -- Four Factors
        efg_pct         REAL,
        tov_pct         REAL,
        orb_pct         REAL,
        ft_rate         REAL,

        -- Shooting efficiency
        ts_pct          REAL,

        computed_at     TEXT,                   -- ISO-8601 UTC

        UNIQUE (game_id, team_id)
    )
    """,

    # -------------------------------------------------------------------
    # 3.  Season-level team aggregates
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS team_season_advanced (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id         INTEGER NOT NULL,
        season          TEXT NOT NULL,
        season_type     TEXT NOT NULL DEFAULT 'Regular Season',

        games           INTEGER,
        wins            INTEGER,
        losses          INTEGER,
        win_pct         REAL,

        -- Computed derivatives (season averages)
        pace            REAL,
        off_rating      REAL,
        def_rating      REAL,
        net_rating      REAL,
        efg_pct         REAL,
        tov_pct         REAL,
        orb_pct         REAL,
        ft_rate         REAL,
        ts_pct          REAL,

        -- SRS
        srs             REAL,
        sos             REAL,

        computed_at     TEXT,                   -- ISO-8601 UTC

        UNIQUE (team_id, season, season_type)
    )
    """,

    # -------------------------------------------------------------------
    # 4.  Operational scrape / validation log
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS raw_scrape_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at       TEXT NOT NULL,          -- ISO-8601 UTC
        game_id         TEXT,
        team_id         INTEGER,
        endpoint        TEXT,
        status          TEXT,                   -- 'ok' | 'warning' | 'error'
        message         TEXT,

        -- Validation diff fields (populated when status='warning')
        metric          TEXT,                   -- e.g. 'pace', 'off_rating'
        our_value       REAL,
        official_value  REAL,
        diff            REAL
    )
    """,

    # -------------------------------------------------------------------
    # 5.  Player season stats cache (one row per player per season)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_season_stats (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id       INTEGER NOT NULL,
        season          TEXT NOT NULL,
        season_type     TEXT NOT NULL DEFAULT 'Regular Season',
        team_id         INTEGER,
        team_abbr       TEXT,

        gp              INTEGER,
        gs              INTEGER,
        min             REAL,
        pts             REAL,
        reb             REAL,
        ast             REAL,
        stl             REAL,
        blk             REAL,
        tov             REAL,
        pf              REAL,
        fgm             REAL,
        fga             REAL,
        fg_pct          REAL,
        fg3m            REAL,
        fg3a            REAL,
        fg3_pct         REAL,
        ftm             REAL,
        fta             REAL,
        ft_pct          REAL,
        oreb            REAL,
        dreb            REAL,
        plus_minus      REAL,

        -- Advanced (from LeagueDashPlayerStats Advanced)
        ts_pct          REAL,
        usg_pct         REAL,
        off_rating      REAL,
        def_rating      REAL,
        net_rating      REAL,
        ast_pct         REAL,
        reb_pct         REAL,
        efg_pct         REAL,
        tov_pct         REAL,
        pace            REAL,

        fetched_at      TEXT,

        UNIQUE (player_id, season, season_type, team_id)
    )
    """,

    # -------------------------------------------------------------------
    # 6.  Player bio cache (static data: height, weight, DOB, position…)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_bio (
        player_id       INTEGER PRIMARY KEY,
        full_name       TEXT,
        first_name      TEXT,
        last_name       TEXT,
        team_id         INTEGER,
        team_abbr       TEXT,
        jersey          TEXT,
        position        TEXT,
        height          TEXT,
        weight          TEXT,
        birth_date      TEXT,
        country         TEXT,
        school          TEXT,
        draft_year      INTEGER,
        draft_round     INTEGER,
        draft_number    INTEGER,
        years_experience INTEGER,
        is_active       INTEGER DEFAULT 1,
        fetched_at      TEXT
    )
    """,

    # -------------------------------------------------------------------
    # 7.  Team metadata
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS team_metadata (
        team_id         INTEGER PRIMARY KEY,
        full_name       TEXT,
        abbreviation    TEXT,
        nickname        TEXT,
        city            TEXT,
        state           TEXT,
        year_founded    INTEGER,
        conference      TEXT,
        division        TEXT,
        fetched_at      TEXT
    )
    """,

    # -------------------------------------------------------------------
    # 8.  Players (Phase 1 core player index)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS players (
        player_id       INTEGER PRIMARY KEY,
        full_name       TEXT NOT NULL,
        first_name      TEXT,
        last_name       TEXT,
        is_active       INTEGER DEFAULT 1
    )
    """,

    # -------------------------------------------------------------------
    # 9.  Player Game Log (single row per player per game)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_game_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id         TEXT NOT NULL,
        player_id       INTEGER NOT NULL,
        team_id         INTEGER NOT NULL,
        game_date       TEXT,
        min             REAL,
        fgm             INTEGER,
        fga             INTEGER,
        fg_pct          REAL,
        fg3m            INTEGER,
        fg3a            INTEGER,
        fg3_pct         REAL,
        ftm             INTEGER,
        fta             INTEGER,
        ft_pct          REAL,
        oreb            INTEGER,
        dreb            INTEGER,
        reb             INTEGER,
        ast             INTEGER,
        stl             INTEGER,
        blk             INTEGER,
        tov             INTEGER,
        pf              INTEGER,
        pts             INTEGER,
        plus_minus      REAL,
        starter         INTEGER DEFAULT 0,
        UNIQUE(game_id, player_id)
    )
    """,

    # -------------------------------------------------------------------
    # 10. Player Season Totals (aggregated traditional stats)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_season_totals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id       INTEGER NOT NULL,
        season          TEXT NOT NULL,
        season_type     TEXT NOT NULL DEFAULT 'Regular Season',
        team_id         INTEGER NOT NULL,
        gp              INTEGER,
        gs              INTEGER,
        min             REAL,
        fgm             INTEGER,
        fga             INTEGER,
        fg_pct          REAL,
        fg3m            INTEGER,
        fg3a            INTEGER,
        fg3_pct         REAL,
        ftm             INTEGER,
        fta             INTEGER,
        ft_pct          REAL,
        oreb            INTEGER,
        dreb            INTEGER,
        reb             INTEGER,
        ast             INTEGER,
        stl             INTEGER,
        blk             INTEGER,
        tov             INTEGER,
        pf              INTEGER,
        pts             INTEGER,
        UNIQUE(player_id, season, season_type, team_id)
    )
    """,

    # -------------------------------------------------------------------
    # 11. Player Season Advanced (aggregated advanced stats)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_season_advanced (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id       INTEGER NOT NULL,
        season          TEXT NOT NULL,
        season_type     TEXT NOT NULL DEFAULT 'Regular Season',
        team_id         INTEGER NOT NULL,
        ts_pct          REAL,
        usg_pct         REAL,
        off_rating      REAL,
        def_rating      REAL,
        net_rating      REAL,
        ast_pct         REAL,
        reb_pct         REAL,
        efg_pct         REAL,
        tov_pct         REAL,
        pace            REAL,
        UNIQUE(player_id, season, season_type, team_id)
    )
    """,

    # -------------------------------------------------------------------
    # 12. Player Splits (Home/Road, Win/Loss, Months, etc.)
    # -------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS player_splits (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id       INTEGER NOT NULL,
        season          TEXT NOT NULL,
        season_type     TEXT NOT NULL DEFAULT 'Regular Season',
        split_type      TEXT NOT NULL,
        split_value     TEXT NOT NULL,
        gp              INTEGER,
        gs              INTEGER,
        min             REAL,
        pts             REAL,
        reb             REAL,
        ast             REAL,
        stl             REAL,
        blk             REAL,
        tov             REAL,
        fgm             INTEGER,
        fga             INTEGER,
        fg_pct          REAL,
        fg3m            INTEGER,
        fg3a            INTEGER,
        fg3_pct         REAL,
        ftm             INTEGER,
        fta             INTEGER,
        ft_pct          REAL,
        plus_minus      REAL,
        UNIQUE(player_id, season, season_type, split_type, split_value)
    )
    """,
]

# Indexes to speed up common lookups
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tga_game_team ON team_game_advanced (game_id, team_id)",
    "CREATE INDEX IF NOT EXISTS idx_tga_season ON team_game_advanced (season, team_id)",
    "CREATE INDEX IF NOT EXISTS idx_tsa_season ON team_season_advanced (season, team_id)",
    "CREATE INDEX IF NOT EXISTS idx_pss_player ON player_season_stats (player_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_pgl_player_game ON player_game_log (player_id, game_id)",
    "CREATE INDEX IF NOT EXISTS idx_pst_player_season ON player_season_totals (player_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_psa_player_season ON player_season_advanced (player_id, season)",
    "CREATE INDEX IF NOT EXISTS idx_ps_player_split ON player_splits (player_id, season, split_type)",
    "CREATE INDEX IF NOT EXISTS idx_log_game ON raw_scrape_log (game_id)",
    "CREATE INDEX IF NOT EXISTS idx_log_status ON raw_scrape_log (status)",
]


def ensure_schema(db_path: str) -> None:
    """
    Create all pipeline tables and indexes if they don't already exist.
    Safe to call on every startup — all statements use IF NOT EXISTS.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
        cur.execute("PRAGMA foreign_keys=ON")

        for ddl in _DDL:
            cur.execute(ddl)

        for idx in _INDEXES:
            cur.execute(idx)

        conn.commit()
        logger.info("NBA pipeline schema ensured at: %s", db_path)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open and return a SQLite connection with sensible defaults.
    The caller is responsible for closing the connection.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row        # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Data/TeamData.sqlite"
    logging.basicConfig(level=logging.INFO)
    ensure_schema(path)
    print(f"Schema ensured at: {path}")
