# main_api.py
# FINAL STABLE VERSION - Corrected endpoint routing and data handling.
import collections
import glob
import re
import os
import unicodedata
import secrets
from typing import List, Dict, Any, Optional, Tuple, Union
import uvicorn
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import json
from datetime import datetime, timedelta, timezone

# Rate limiting (slowapi). Declared in requirements.txt; if it is missing from a
# local environment the API still boots, but unthrottled - and says so loudly.
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

# nba_api imports
try:
    from nba_api.stats.static import players as nba_players
    from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats, commonplayerinfo
except ImportError:
    nba_players = None
    shotchartdetail = None
    leaguedashplayerstats = None
    commonplayerinfo = None

# Local Imports
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Utils import Expected_Value, Kelly_Criterion as kc, Parlay as parlay
from src.Utils import devig
from src.Utils import elo as elo_engine
from src.Utils import nba_live
from src.Utils.tools import create_todays_games_from_odds
from src.Utils.Dictionaries import team_index_current
from src.Utils.game_flow import build_game_flow
from src.Predict import candidate_live
from src.Utils import player_impact
from src.Utils import availability as availability_adjust
from src.Utils import espn_injuries



# Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DATABASE-BACKED ADVANCED STATS LOOKUP (REPLACES BBREF) ---
# Latest season with complete data in the stats database.
# Bump each fall once the new season's games start flowing in.
CURRENT_SEASON = "2025-26"

def find_db_team_stats(team_name: str, season: str = CURRENT_SEASON):
    """
    Lookup team season advanced statistics (pace, offensive rating, defensive rating)
    from SQLite database to blend with predictions.
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'TeamData.sqlite')
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Resolve team name to team_id using fuzzy matches
        team_q = "%" + team_name.lower().strip() + "%"
        cursor.execute(
            """
            SELECT s.* FROM team_season_advanced s
            JOIN team_metadata m ON s.team_id = m.team_id
            WHERE (lower(m.full_name) LIKE ? OR lower(m.nickname) LIKE ? OR lower(m.abbreviation) LIKE ?)
              AND s.season = ? AND s.season_type = 'Regular Season'
            LIMIT 1
            """,
            (team_q, team_q, team_q, season)
        )
        row = cursor.fetchone()
        if row:
            res = dict(row)
            # Map DB keys to the properties expected by the blending code: pace, offRating, defRating, netRating
            res["offRating"] = res.get("off_rating", 0.0)
            res["defRating"] = res.get("def_rating", 0.0)
            res["netRating"] = res.get("net_rating", 0.0)
            res["fourFactors"] = {
                "eFG": res.get("efg_pct", 0.0),
                "TOV": res.get("tov_pct", 0.0),
                "ORB": res.get("orb_pct", 0.0),
                "FT": res.get("ft_rate", 0.0)
            }
            conn.close()
            return res
        conn.close()
        return None
    except Exception as e:
        logger.warning(f"Error querying advanced team stats for {team_name}: {e}")
        return None

# --- Security / networking configuration (env-driven) ---
# CORS_ORIGINS: comma-separated list of allowed browser origins. Defaults to the
# local Next.js dev server so a fresh checkout works with no configuration.
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
] or [o.strip() for o in DEFAULT_CORS_ORIGINS.split(",")]

# API_KEY: when unset, auth is disabled (local dev / current frontend keep working).
# When set, only the PROTECTED paths below require the X-API-Key header.
#
# The model is an allowlist of PROTECTED paths, not an exempt-list, on purpose:
# the reference stats site is fetched straight from the browser by client
# components, so any key they could send would have to be a NEXT_PUBLIC_ var -
# i.e. not a secret. Those endpoints are guarded by CORS + rate limiting.
# The paths below are the expensive / non-public ones, and the frontend already
# reaches them through server-side Next API routes that can hold a real secret.
API_KEY = (os.environ.get("API_KEY") or "").strip()
API_KEY_HEADER = "X-API-Key"
PROTECTED_PATH_PREFIXES = (
    "/predictions",           # live sbrscrape scraping + XGBoost inference
    "/api/parlay/evaluate",   # parlay EV engine
    "/api/line-movements",    # odds snapshot history
)
# Deliberately NOT protected: /api/prediction-log. It backs the public
# /track-record grading ledger, is fetched straight from the browser, and is a
# cheap DB read - public readability is the point of that page.
# Never require a key here, whatever else is configured (uptime probes).
ALWAYS_OPEN_PATHS = {"/", "/health"}


def _is_protected_path(path: str) -> bool:
    """True when `path` needs the API key. Exact match or a genuine sub-path."""
    if path in ALWAYS_OPEN_PATHS:
        return False
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in PROTECTED_PATH_PREFIXES
    )

# Rate limits (per client IP). Tunable without a code change / redeploy.
# RATE_LIMIT_DEFAULT is slowapi's per-endpoint default; RATE_LIMIT_GLOBAL is the
# overall per-IP ceiling that stops a crawler from fanning out across every route.
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "").strip() or "60/minute"
RATE_LIMIT_GLOBAL = os.environ.get("RATE_LIMIT_GLOBAL", "").strip() or "240/minute"
RATE_LIMIT_EXPENSIVE = os.environ.get("RATE_LIMIT_EXPENSIVE", "").strip() or "10/minute"
# RATE_LIMIT_UPSTREAM guards the handful of public routes that turn one inbound
# request into one outbound stats.nba.com call. Abuse there gets THIS SERVER's IP
# throttled or banned by nba.com, which breaks the site for everyone and is not
# something a redeploy fixes. Keyless (they back the public reference site), but
# throttled harder than an ordinary DB read.
RATE_LIMIT_UPSTREAM = os.environ.get("RATE_LIMIT_UPSTREAM", "").strip() or "20/minute"


def require_api_key(request: Request) -> None:
    """Global dependency: enforce X-API-Key on protected paths when API_KEY is set."""
    if not API_KEY:
        return
    # CORS preflight never carries custom headers - never block it.
    if request.method == "OPTIONS":
        return
    path = request.url.path.rstrip("/") or "/"
    if not _is_protected_path(path):
        return
    provided = request.headers.get(API_KEY_HEADER)
    if not provided or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(
            status_code=401,
            detail=f"Missing or invalid {API_KEY_HEADER} header.",
        )


class _NoopLimiter:
    """Stand-in used when slowapi is not installed; decorators become no-ops."""

    def limit(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def exempt(self, func):
        return func


if SLOWAPI_AVAILABLE:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[RATE_LIMIT_DEFAULT],
        application_limits=[RATE_LIMIT_GLOBAL],
    )
else:
    limiter = _NoopLimiter()

# FastAPI App Setup
app = FastAPI(
    title="Betting Buddy API",
    version="1.1.1-stable-fixed",
    dependencies=[Depends(require_api_key)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

# Make misconfiguration obvious in deploy logs.
logger.info(f"CORS allowed origins: {CORS_ORIGINS}")
logger.info(
    f"Key-protected paths ({len(PROTECTED_PATH_PREFIXES)}): {', '.join(PROTECTED_PATH_PREFIXES)} "
    "- every other route is public by design (reference stats site)."
)
if API_KEY:
    logger.info(f"API key auth ENABLED: the paths above require the {API_KEY_HEADER} header.")
else:
    logger.warning(
        "API_KEY is not set - the protected paths listed above are OPEN to anyone who can "
        f"reach this host. Set API_KEY in the environment to require the {API_KEY_HEADER} header."
    )
if SLOWAPI_AVAILABLE:
    logger.info(
        f"Rate limiting enabled (per IP): {RATE_LIMIT_DEFAULT} per endpoint, "
        f"{RATE_LIMIT_GLOBAL} overall, {RATE_LIMIT_EXPENSIVE} on /predictions and /api/parlay/evaluate, "
        f"{RATE_LIMIT_UPSTREAM} on the 5 routes that call stats.nba.com live; /health exempt"
    )
else:
    logger.warning("slowapi is not installed - rate limiting is DISABLED. Run: pip install slowapi")

# De-vig method for turning bookmaker quotes into fair probabilities (Task: EV math).
if devig.SHIN_AVAILABLE:
    logger.info("De-vig method active: shin (insider-trading model) for market fair probabilities.")
else:
    logger.warning(
        "De-vig method active: multiplicative FALLBACK - the 'shin' package is not installed. "
        "Run: pip install shin"
    )

# --- Odds snapshots (real line-movement tracking) ---
ODDS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'OddsData.sqlite')

def _odds_snapshot_conn():
    conn = sqlite3.connect(ODDS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            sport TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            game_key TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_ml REAL,
            away_ml REAL,
            ou_line REAL,
            game_start_time_utc TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_snapshots_lookup ON odds_snapshots (sportsbook, sport, game_key, captured_at)"
    )
    return conn

def snapshot_odds(odds_data: Dict[str, Any], sportsbook: str, sport: str) -> None:
    """Persist current lines; only writes a row when a game's lines changed since the last snapshot."""
    if not odds_data:
        return
    conn = _odds_snapshot_conn()
    try:
        captured_at = datetime.utcnow().isoformat()
        for game_key, game in odds_data.items():
            try:
                home_team, away_team = game_key.split(":", 1)
                home_ml = game.get(home_team, {}).get('money_line_odds')
                away_ml = game.get(away_team, {}).get('money_line_odds')
                ou_line = game.get('under_over_odds')
                start = game.get('game_start_time_utc')
                start_str = start.isoformat() if isinstance(start, datetime) else (start or None)

                last = conn.execute(
                    """
                    SELECT home_ml, away_ml, ou_line FROM odds_snapshots
                    WHERE sportsbook = ? AND sport = ? AND game_key = ?
                    ORDER BY captured_at DESC LIMIT 1
                    """,
                    (sportsbook, sport, game_key)
                ).fetchone()
                if last and last["home_ml"] == home_ml and last["away_ml"] == away_ml and last["ou_line"] == ou_line:
                    continue

                conn.execute(
                    """
                    INSERT INTO odds_snapshots
                        (captured_at, sport, sportsbook, game_key, home_team, away_team, home_ml, away_ml, ou_line, game_start_time_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (captured_at, sport, sportsbook, game_key, home_team, away_team, home_ml, away_ml, ou_line, start_str)
                )
            except Exception as exc:
                logger.warning(f"Skipping odds snapshot for {game_key}: {exc}")
        conn.commit()
    finally:
        conn.close()

# --- Prediction track record ---
# Tag for the market-implied fallback in run_predictions(). It is NOT a model:
# it takes the book's own de-vigged probability and adds a hardcoded edge, so a
# row carrying this tag is the market's opinion wearing our name. Such rows must
# never reach predictions_log (the public track record grades what is in there)
# and are filtered again on the way out, in case any were written historically.
SIMULATED_MODEL_TAG = "implied_probability_sim"


def _ensure_prediction_log_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            log_date TEXT NOT NULL,
            sport TEXT NOT NULL,
            sportsbook TEXT NOT NULL,
            game_key TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            game_start_time_utc TEXT,
            home_ml REAL,
            away_ml REAL,
            ou_line REAL,
            predicted_winner TEXT,
            winner_confidence REAL,
            ou_prediction TEXT,
            ou_confidence REAL,
            ev_home REAL,
            ev_away REAL,
            model TEXT,
            actual_winner TEXT,
            actual_total REAL,
            UNIQUE (log_date, sportsbook, game_key)
        )
        """
    )

def log_predictions(result: Dict[str, Any], sportsbook: str, sport: str) -> None:
    """
    Persist each model prediction (one row per game/book/day, latest wins).
    This is the raw material for a public track-record page: predictions are
    recorded BEFORE games happen; actual_winner/actual_total get filled in
    later by a grading pass.
    """
    predictions = result.get("predictions") or []
    if not predictions:
        return
    conn = _odds_snapshot_conn()
    try:
        _ensure_prediction_log_schema(conn)
        now = datetime.utcnow()
        skipped_sim = 0
        for p in predictions:
            home = p.get("home_team")
            away = p.get("away_team")
            if not home or not away:
                continue
            if p.get("model") == SIMULATED_MODEL_TAG:
                skipped_sim += 1
                continue
            ev = p.get("expected_value") or {}
            conn.execute(
                """
                INSERT INTO predictions_log (
                    logged_at, log_date, sport, sportsbook, game_key,
                    home_team, away_team, game_start_time_utc,
                    home_ml, away_ml, ou_line,
                    predicted_winner, winner_confidence, ou_prediction, ou_confidence,
                    ev_home, ev_away, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(log_date, sportsbook, game_key) DO UPDATE SET
                    logged_at=excluded.logged_at,
                    home_ml=excluded.home_ml,
                    away_ml=excluded.away_ml,
                    ou_line=excluded.ou_line,
                    predicted_winner=excluded.predicted_winner,
                    winner_confidence=excluded.winner_confidence,
                    ou_prediction=excluded.ou_prediction,
                    ou_confidence=excluded.ou_confidence,
                    ev_home=excluded.ev_home,
                    ev_away=excluded.ev_away,
                    model=excluded.model
                """,
                (
                    now.isoformat(),
                    now.strftime("%Y-%m-%d"),
                    sport,
                    sportsbook,
                    f"{home}:{away}",
                    home,
                    away,
                    p.get("game_start_time_utc"),
                    p.get("home_odds"),
                    p.get("away_odds"),
                    p.get("under_over_line") if isinstance(p.get("under_over_line"), (int, float)) else None,
                    p.get("predicted_winner"),
                    p.get("winner_confidence"),
                    p.get("under_over_prediction"),
                    p.get("under_over_confidence"),
                    ev.get("home_team"),
                    ev.get("away_team"),
                    p.get("model"),
                )
            )
        if skipped_sim:
            logger.info(
                "Prediction log: skipped %d market-implied row(s) - '%s' output is not a "
                "model prediction and is never graded as one.",
                skipped_sim, SIMULATED_MODEL_TAG
            )
        conn.commit()
    finally:
        conn.close()

# --- Days-rest helpers -------------------------------------------------------
# The schedule CSVs (Data/nba-*-UTC.csv) carry UTC timestamps, but an NBA "game date"
# -- and therefore the days-rest convention the model was trained on -- is the US
# Eastern calendar date. A 7:30pm ET tip-off is stamped 00:30 UTC the FOLLOWING day, so
# comparing UTC timestamps against a naive local clock silently shifts games by a day.
# In season the team-stats snapshot is rewritten every morning; a few days of slack
# covers the All-Star break and a missed run without crying wolf.
TEAM_STATS_MAX_AGE_DAYS = 10

DEFAULT_DAYS_REST = 7          # used when a team has no earlier game on record
MIN_DAYS_REST = 1
MAX_DAYS_REST = 7

try:
    from zoneinfo import ZoneInfo
    _NBA_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - no tz database on this platform
    _NBA_TZ = None


def to_nba_date(ts):
    """UTC instant (datetime, pandas Timestamp, or ISO string) -> US Eastern calendar date.

    Returns None if `ts` cannot be interpreted. Falls back to a fixed UTC-5 offset if no
    timezone database is available; that is correct for the whole regular season and off
    by at most one day for a handful of late-evening playoff tip-offs.
    """
    if ts is None:
        return None
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        if _NBA_TZ is not None:
            return ts.tz_convert(_NBA_TZ).date()
        return (ts.tz_convert('UTC').tz_localize(None) - timedelta(hours=5)).date()
    except Exception:
        return None


# PredictionRunner Class
class PredictionRunner:
    def __init__(self, sportsbook: str, kelly_criterion: bool, sport: str = 'NBA'):
        self.sportsbook = sportsbook
        self.model_name = 'xgboost'
        self.kelly_criterion = kelly_criterion
        self.sport = sport
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.team_stats_df = self._load_team_stats()
        self.schedule_df = self._load_schedule()
        self.odds_provider = SbrOddsProvider(sportsbook=self.sportsbook, sport=self.sport)
        # The provider falls back from NBA to WNBA out of season; record what it
        # actually scraped so snapshots and the prediction log are labeled truthfully.
        try:
            self.resolved_sport = self.odds_provider.get_resolved_sport() or self.sport
        except Exception as exc:
            logger.warning(f"Could not resolve scraped sport, falling back to '{self.sport}': {exc}")
            self.resolved_sport = self.sport
        if self.resolved_sport != self.sport:
            logger.info(f"Requested sport '{self.sport}' resolved to '{self.resolved_sport}' by the odds provider.")
        self.xgb_ml_model, self.xgb_uo_model = self._load_xgboost_models()

    def _load_team_stats(self):
        try:
            db_path = os.path.join(self.project_root, 'Data', 'TeamData.sqlite')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' ORDER BY name DESC LIMIT 1")
            table_row = cursor.fetchone()
            if not table_row:
                raise FileNotFoundError("No team statistics tables found in TeamData.sqlite")
            table_name = table_row[0]
            logger.info(f"Loading team stats from SQLite table: {table_name}")
            # These snapshots are written daily by refresh_team_stats.py. If that
            # stops running the newest table just gets old, and predictions quietly
            # keep being served from stale team form - which is how the serving path
            # ended up on 2024-04-29 data two seasons later. Say so, loudly.
            try:
                snapshot_age = (datetime.now().date() - datetime.strptime(table_name, "%Y-%m-%d").date()).days
                if snapshot_age > TEAM_STATS_MAX_AGE_DAYS:
                    logger.error(
                        "Team stats are %d days old (table '%s'). Predictions are being made "
                        "from stale team form - run refresh_team_stats.py.",
                        snapshot_age, table_name
                    )
                self.team_stats_age_days = snapshot_age
            except ValueError:
                self.team_stats_age_days = None
            self.team_stats_table = table_name
            df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn, index_col="index")
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to load team stats from database: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Server configuration error: Could not load team stats.")

    def _load_schedule(self):
        # Use the most recent season's schedule file (nba-<year>-UTC.csv)
        schedule_files = sorted(glob.glob(os.path.join(self.project_root, 'Data', 'nba-*-UTC.csv')), reverse=True)
        if not schedule_files:
            logger.error("No schedule file (Data/nba-*-UTC.csv) found.")
            return None
        try:
            logger.info(f"Loading schedule from: {os.path.basename(schedule_files[0])}")
            df = pd.read_csv(schedule_files[0], parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
            # Precompute the US Eastern calendar date so days-rest can be worked out in
            # whole days without depending on the server's clock or timezone.
            df['GameDateET'] = df['Date'].map(to_nba_date)
            return df
        except Exception as e:
            logger.error(f"Failed to load schedule file {schedule_files[0]}: {e}")
            return None

    def _days_rest(self, team, game_date):
        """Days of rest a team has going into a game on `game_date` (US Eastern date).

        Matches the convention the model was trained on (src/Process-Data/Add_Days_Rest.py
        and src/Process-Data/Get_Odds_Data.py): the whole-day difference between this
        game's date and the team's most recent previous game date.

        Only games on a STRICTLY EARLIER date are considered, so a game can never
        contribute to its own rest figure no matter when this endpoint is called. The
        previous implementation compared UTC timestamps against a naive local clock,
        which made the answer depend on the server's timezone and the hour of the call.
        """
        if self.schedule_df is None or game_date is None:
            return DEFAULT_DAYS_REST
        if 'GameDateET' not in self.schedule_df.columns:
            return DEFAULT_DAYS_REST
        sched = self.schedule_df
        played = sched.loc[
            ((sched['Home Team'] == team) | (sched['Away Team'] == team))
            & sched['GameDateET'].notna()
            & (sched['GameDateET'] < game_date),
            'GameDateET']
        if played.empty:
            return DEFAULT_DAYS_REST
        return (game_date - max(played)).days

    def _load_xgboost_models(self):
        try:
            ml_path = os.path.join(self.project_root, 'Models', 'XGBoost_Models', 'XGBoost_68.9%_ML-3.json')
            uo_path = os.path.join(self.project_root, 'Models', 'XGBoost_Models', 'XGBoost_54.8%_UO-8.json')
            xgb_ml, xgb_uo = xgb.Booster(), xgb.Booster()
            xgb_ml.load_model(ml_path)
            xgb_uo.load_model(uo_path)
            return xgb_ml, xgb_uo
        except xgb.core.XGBoostError as e:
            logger.error(f"Failed to load XGBoost models: {e}")
            raise HTTPException(status_code=500, detail="Server configuration error: Could not load prediction models.")

    def run_predictions(self):
        odds_data = self.odds_provider.get_odds()
        if not odds_data:
            return {"error": f"No odds data found from {self.sportsbook}.", "predictions": []}
        try:
            snapshot_odds(odds_data, self.sportsbook, getattr(self, 'resolved_sport', self.sport) or self.sport)
        except Exception as exc:
            logger.warning(f"Odds snapshot failed (non-fatal): {exc}")
        games_list = create_todays_games_from_odds(odds_data)
        if not games_list:
            return {"error": "No valid games processed from odds data.", "predictions": []}
        
        # Check if we have database stats for the teams. If not, use bookmaker odds simulation.
        has_stats = False
        for home_team, away_team in games_list:
            # Check name or Clippers variation
            home_stats_rows = self.team_stats_df[self.team_stats_df['TEAM_NAME'] == home_team]
            if home_stats_rows.empty:
                alt_name = "Los Angeles Clippers" if home_team == "LA Clippers" else ("LA Clippers" if home_team == "Los Angeles Clippers" else home_team)
                home_stats_rows = self.team_stats_df[self.team_stats_df['TEAM_NAME'] == alt_name]
            
            if not home_stats_rows.empty:
                has_stats = True
                break

        if not has_stats:
            logger.info("No teams in today's games have stats in database. Using bookmaker implied probability simulation.")
            predictions_list = []
            for home_team, away_team in games_list:
                game_key = f"{home_team}:{away_team}"
                game_odds = odds_data.get(game_key, {})
                home_odd = game_odds.get(home_team, {}).get('money_line_odds')
                away_odd = game_odds.get(away_team, {}).get('money_line_odds')
                uo_line = game_odds.get('under_over_odds', 160.0)
                if uo_line is None:
                    uo_line = 160.0
                game_datetime_obj = game_odds.get('game_start_time_utc')
                game_start_time_str = game_datetime_obj if isinstance(game_datetime_obj, str) else (game_datetime_obj.isoformat() if isinstance(game_datetime_obj, datetime) else None)
                
                # Convert moneyline odds to implied probability
                def ml_to_prob(ml):
                    if ml is None:
                        return 0.5
                    try:
                        ml_val = int(ml)
                    except ValueError:
                        return 0.5
                    if ml_val < 0:
                        return abs(ml_val) / (abs(ml_val) + 100)
                    else:
                        return 100 / (ml_val + 100)
                        
                prob_home = ml_to_prob(home_odd)
                prob_away = ml_to_prob(away_odd)
                # Market opinion = de-vigged fair probability of the moneyline
                # pair (Shin when available), not raw implied normalisation -
                # payouts below still use the quoted odds.
                prob_home_norm = None
                if home_odd is not None and away_odd is not None:
                    try:
                        prob_home_norm = devig.fair_probs([
                            parlay.american_to_true_decimal(float(home_odd)),
                            parlay.american_to_true_decimal(float(away_odd)),
                        ])[0]
                    except (ValueError, TypeError):
                        prob_home_norm = None
                if prob_home_norm is None:
                    total_prob = prob_home + prob_away
                    if total_prob > 0:
                        prob_home_norm = prob_home / total_prob
                    else:
                        prob_home_norm = 0.5
                
                # Simulate a small model edge (e.g. adding 2% to favored team or a slight variation)
                # to show a positive Expected Value and Kelly Criterion suggestion!
                if prob_home_norm >= 0.5:
                    winner_confidence = min(0.99, prob_home_norm + 0.02)
                    winner_idx = 1
                else:
                    winner_confidence = min(0.99, (1 - prob_home_norm) + 0.02)
                    winner_idx = 0
                
                # Under/Over prediction: default to UNDER with a 51% simulated confidence
                ou_idx = 0
                ou_confidence = 0.51
                
                ev_home, ev_away, kelly_home, kelly_away = 0.0, 0.0, "No Bet", "No Bet"
                try:
                    if home_odd is not None and away_odd is not None:
                        # Use the normalized probability (or slightly boosted simulated prob)
                        model_prob_home = winner_confidence if winner_idx == 1 else (1 - winner_confidence)
                        ev_home = round(Expected_Value.expected_value(model_prob_home, int(home_odd)), 2)
                        ev_away = round(Expected_Value.expected_value(1 - model_prob_home, int(away_odd)), 2)
                        if self.kelly_criterion:
                            kelly_home = kc.calculate_kelly_criterion(int(home_odd), model_prob_home)
                            kelly_away = kc.calculate_kelly_criterion(int(away_odd), 1 - model_prob_home)
                except Exception as e:
                    logger.error(f"Error calculating stats in simulation: {e}")
                
                # Resolve team IDs to find the game_id from box_scores
                game_id = None
                db_conn = get_db_conn()
                if db_conn:
                    try:
                        cursor = db_conn.cursor()
                        home_q = "%" + home_team.lower().strip() + "%"
                        away_q = "%" + away_team.lower().strip() + "%"
                        cursor.execute(
                            "SELECT team_id FROM team_metadata WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?",
                            (home_q, home_q, home_q)
                        )
                        h_row = cursor.fetchone()
                        cursor.execute(
                            "SELECT team_id FROM team_metadata WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?",
                            (away_q, away_q, away_q)
                        )
                        a_row = cursor.fetchone()
                        if h_row and a_row:
                            h_id = h_row["team_id"]
                            a_id = a_row["team_id"]
                            cursor.execute(
                                "SELECT game_id FROM box_scores WHERE home_team_id = ? AND away_team_id = ? ORDER BY game_date DESC LIMIT 1",
                                (h_id, a_id)
                            )
                            g_row = cursor.fetchone()
                            if g_row:
                                game_id = g_row["game_id"]
                            else:
                                cursor.execute(
                                    "SELECT game_id FROM team_game_advanced WHERE team_id = ? AND opp_team_id = ? ORDER BY game_date DESC LIMIT 1",
                                    (h_id, a_id)
                                )
                                g_row = cursor.fetchone()
                                if g_row:
                                    game_id = g_row["game_id"]
                    except Exception as ex:
                        logger.warning(f"Error looking up game_id for {home_team} vs {away_team}: {ex}")
                    finally:
                        db_conn.close()

                predictions_list.append({
                    "game_id": game_id,
                    "game_identifier": f"{away_team.replace(' ', '_')}_{home_team.replace(' ', '_')}_{game_start_time_str or 'today'}",
                    "home_team": home_team, "away_team": away_team, "home_odds": home_odd, "away_odds": away_odd,
                    "under_over_line": uo_line, "predicted_winner": home_team if winner_idx == 1 else away_team,
                    "winner_confidence": round(winner_confidence * 100, 2),
                    "under_over_prediction": "OVER" if ou_idx == 1 else "UNDER",
                    "under_over_confidence": round(ou_confidence * 100, 2), "model": SIMULATED_MODEL_TAG,
                    "expected_value": {"home_team": ev_home, "away_team": ev_away},
                    "kelly_criterion": {"home_team": kelly_home, "away_team": kelly_away},
                    "game_start_time_utc": game_start_time_str
                })
            return self._attach_availability({"sportsbook": self.sportsbook, "predictions": predictions_list})

        (data_for_model, todays_games_uo, frame_ml, home_team_odds, away_team_odds,
         game_start_times, processed_games, game_dates) = self._prepare_data_for_model(games_list, odds_data)

        if data_for_model.size == 0:
            return {"error": "Could not prepare valid data for the prediction model.", "predictions": []}

        ml_predictions, ou_predictions = self._run_xgboost_models(data_for_model, frame_ml, todays_games_uo)

        # Serve the sealed 2026-08 candidate for the moneyline when its feature
        # pipeline is healthy; its calibrated probabilities are used AS-IS (the
        # power-rating blend below is skipped — serving must match what the
        # sealed evaluation measured). Any failure falls back to the old model.
        calibrated_ml = False
        cand = candidate_live.get_candidate()
        if cand is not None:
            try:
                p_home = cand.predict(frame_ml, processed_games, game_dates)
                ml_predictions = np.column_stack([1.0 - p_home, p_home])
                self.model_name = candidate_live.MODEL_TAG
                calibrated_ml = True
            except Exception as exc:
                logger.error(f"Candidate model path failed; serving old model: {exc}", exc_info=True)

        return self._attach_availability(
            self._format_predictions(processed_games, ml_predictions, ou_predictions, home_team_odds,
                                     away_team_odds, todays_games_uo, game_start_times,
                                     calibrated_ml=calibrated_ml)
        )

    def _prepare_data_for_model(self, games, odds):
        game_data_list, home_odds_list, away_odds_list, uo_lines_list, game_start_times_list = [], [], [], [], []
        processed_games, game_dates_list = [], []

        # Fallback game date, used only when the odds feed carries no start time.
        today_nba_date = to_nba_date(datetime.now(timezone.utc))

        for home_team, away_team in games:
            game_key = f"{home_team}:{away_team}"
            game_odds = odds.get(game_key, {})
            
            # Find team statistics in team_stats_df by name, with fallback to name variants
            home_stats_rows = self.team_stats_df[self.team_stats_df['TEAM_NAME'] == home_team]
            if home_stats_rows.empty:
                alt_name = "Los Angeles Clippers" if home_team == "LA Clippers" else ("LA Clippers" if home_team == "Los Angeles Clippers" else home_team)
                home_stats_rows = self.team_stats_df[self.team_stats_df['TEAM_NAME'] == alt_name]
            
            away_stats_rows = self.team_stats_df[self.team_stats_df['TEAM_NAME'] == away_team]
            if away_stats_rows.empty:
                alt_name = "Los Angeles Clippers" if away_team == "LA Clippers" else ("LA Clippers" if away_team == "Los Angeles Clippers" else away_team)
                away_stats_rows = self.team_stats_df[self.team_stats_df['TEAM_NAME'] == alt_name]
                
            if home_stats_rows.empty or away_stats_rows.empty:
                logger.warning(f"Skipping game {home_team} vs {away_team}: statistics row not found in database.")
                continue

            home_stats = home_stats_rows.iloc[0].copy()
            away_stats = away_stats_rows.iloc[0].copy()

            # Calculate days rest, anchored to the game's own US Eastern date rather than
            # to the server's wall clock, so the result is identical whenever it is run.
            game_date = to_nba_date(game_odds.get('game_start_time_utc')) or today_nba_date
            home_days_off = self._days_rest(home_team, game_date)
            away_days_off = self._days_rest(away_team, game_date)

            # Clip rest days to a reasonable range of 1-7 days to prevent model outlier issues
            home_days_off = max(MIN_DAYS_REST, min(home_days_off, MAX_DAYS_REST))
            away_days_off = max(MIN_DAYS_REST, min(away_days_off, MAX_DAYS_REST))

            # Concatenate home and away team statistics
            game_data = pd.concat([home_stats, away_stats.rename(index=lambda x: x + '.1')])
            game_data['Days-Rest-Home'] = float(home_days_off)
            game_data['Days-Rest-Away'] = float(away_days_off)
            
            game_data_list.append(game_data)
            processed_games.append((home_team, away_team))
            game_dates_list.append(game_date.isoformat())

            home_odds_list.append(game_odds.get(home_team, {}).get('money_line_odds'))
            away_odds_list.append(game_odds.get(away_team, {}).get('money_line_odds'))
            uo_lines_list.append(game_odds.get('under_over_odds'))
            game_start_times_list.append(game_odds.get('game_start_time_utc'))

        if not game_data_list:
            return np.array([]), [], pd.DataFrame(), [], [], [], [], []
            
        frame_ml = pd.DataFrame(game_data_list)
        
        # Columns to drop to match what XGBoost models expect
        cols_to_drop = [
            'TEAM_ID', 'TEAM_NAME', 'Date', 'index',
            'TEAM_ID.1', 'TEAM_NAME.1', 'Date.1', 'index.1',
            'Score', 'Home-Team-Win', 'OU-Cover', 'OU'
        ]
        frame_for_model = frame_ml.drop(columns=[c for c in cols_to_drop if c in frame_ml.columns], errors='ignore')
        
        # Drop any remaining non-numeric columns just in case
        non_numeric = [col for col in frame_for_model.columns if not pd.api.types.is_numeric_dtype(frame_for_model[col])]
        if non_numeric:
            frame_for_model.drop(columns=non_numeric, inplace=True)
            
        logger.info(f"Prepared model input with shape: {frame_for_model.shape}")
        return (frame_for_model.values.astype(float), uo_lines_list, frame_ml, home_odds_list,
                away_odds_list, game_start_times_list, processed_games, game_dates_list)

    def _run_xgboost_models(self, data_ml, frame_ml, todays_games_uo):
        ml_predictions = self.xgb_ml_model.predict(xgb.DMatrix(data_ml))
        frame_uo = frame_ml.copy()
        
        # Add OU column
        safe_uo = [x if x is not None else 0.0 for x in todays_games_uo]
        frame_uo['OU'] = np.asarray(safe_uo).astype(float)
        
        # Columns to drop to match UO model training features (which includes OU)
        cols_to_drop = [
            'TEAM_ID', 'TEAM_NAME', 'Date', 'index',
            'TEAM_ID.1', 'TEAM_NAME.1', 'Date.1', 'index.1',
            'Score', 'Home-Team-Win', 'OU-Cover'
        ]
        frame_uo_clean = frame_uo.drop(columns=[c for c in cols_to_drop if c in frame_uo.columns], errors='ignore')
        
        # Drop any remaining non-numeric columns
        non_numeric = [col for col in frame_uo_clean.columns if not pd.api.types.is_numeric_dtype(frame_uo_clean[col])]
        if non_numeric:
            frame_uo_clean.drop(columns=non_numeric, inplace=True)
            
        logger.info(f"Prepared UO model input with shape: {frame_uo_clean.shape}")
        
        ou_predictions = self.xgb_uo_model.predict(xgb.DMatrix(frame_uo_clean.values.astype(float)))
        return ml_predictions, ou_predictions

    def _attach_availability(self, result):
        """Attach an informational "availability" field to each prediction.

        ADDITIVE ONLY: model probabilities, EV, and Kelly numbers are never
        altered (folding the delta into win probability would need backtest
        validation first). Any failure - ESPN feed down, WNBA fallback,
        unknown team names - leaves predictions exactly as they were.
        """
        try:
            if (getattr(self, 'resolved_sport', None) or self.sport) != 'NBA':
                return result
            preds = result.get("predictions") if isinstance(result, dict) else None
            if not preds:
                return result
            absences = espn_injuries.get_absences()
            for pred in preds:
                try:
                    home_abbr = espn_injuries.resolve_team_abbr(pred.get("home_team") or "")
                    away_abbr = espn_injuries.resolve_team_abbr(pred.get("away_team") or "")
                    if not home_abbr or not away_abbr:
                        continue
                    info = availability_adjust.matchup_availability(
                        home_abbr, away_abbr, CURRENT_SEASON, absences=absences
                    )
                    pred["availability"] = {
                        "home_delta": info["home"]["delta_per_100"],
                        "away_delta": info["away"]["delta_per_100"],
                        "players_out": (
                            [p["name"] for p in info["home"]["players_out"]]
                            + [p["name"] for p in info["away"]["players_out"]]
                        ),
                        "note": "impact-adjusted",
                    }
                except Exception as ex:
                    logger.warning(f"Availability attach skipped for one game (non-fatal): {ex}")
                    continue
        except Exception as exc:
            logger.warning(f"Availability attach failed (non-fatal): {exc}")
        return result

    def _format_predictions(self, games, ml_preds, ou_preds, home_odds, away_odds, uo_lines, game_start_times, calibrated_ml=False):
        predictions_list = []
        for i, (home_team, away_team) in enumerate(games):
            home_odd, away_odd = home_odds[i], away_odds[i]
            xgb_winner_idx, xgb_ou_idx = np.argmax(ml_preds[i]), np.argmax(ou_preds[i])
            
            # Extract raw probabilities from XGBoost
            xgb_home_prob = float(ml_preds[i][1])
            xgb_away_prob = float(ml_preds[i][0])
            xgb_over_prob = float(ou_preds[i][1])
            xgb_under_prob = float(ou_preds[i][0])
            
            # Query advanced stats for blending
            home_power = find_db_team_stats(home_team)
            away_power = find_db_team_stats(away_team)
            
            blended_home_prob = xgb_home_prob
            blended_away_prob = xgb_away_prob
            blended_over_prob = xgb_over_prob
            blended_under_prob = xgb_under_prob
            
            # Apply Bayesian/Weighted Blending if advanced stats are found.
            # When the calibrated candidate is serving the moneyline, its
            # probabilities are used untouched: the sealed evaluation measured
            # the calibrated model alone, so blending would publish numbers
            # nobody validated. The UO model is unchanged and keeps its blend.
            uo_line = uo_lines[i] if uo_lines[i] is not None else 220.0
            if home_power and away_power:
                try:
                    expected_poss = (home_power["pace"] + away_power["pace"]) / 2.0
                    expected_home_pts = (home_power["offRating"] + away_power["defRating"]) / 2.0 / 100.0 * expected_poss
                    expected_away_pts = (away_power["offRating"] + home_power["defRating"]) / 2.0 / 100.0 * expected_poss

                    expected_margin = expected_home_pts - expected_away_pts
                    expected_total = expected_home_pts + expected_away_pts

                    if not calibrated_ml:
                        # Sigmoid for win probability (k = 0.14 matches standard margin-to-win distribution)
                        power_home_prob = 1.0 / (1.0 + math.exp(-0.14 * expected_margin))

                        # 70/30 weight blend for moneyline
                        blended_home_prob = 0.7 * xgb_home_prob + 0.3 * power_home_prob
                        blended_away_prob = 0.7 * xgb_away_prob + 0.3 * (1.0 - power_home_prob)
                        sum_ml = blended_home_prob + blended_away_prob
                        if sum_ml > 0:
                            blended_home_prob /= sum_ml
                            blended_away_prob /= sum_ml

                    # 70/30 weight blend for under/over
                    power_over_prob = max(0.35, min(0.5 + (expected_total - uo_line) * 0.02, 0.65))
                    blended_over_prob = 0.7 * xgb_over_prob + 0.3 * power_over_prob
                    blended_under_prob = 0.7 * xgb_under_prob + 0.3 * (1.0 - power_over_prob)
                    sum_ou = blended_over_prob + blended_under_prob
                    if sum_ou > 0:
                        blended_over_prob /= sum_ou
                        blended_under_prob /= sum_ou
                except Exception as ex:
                    logger.error(f"Error blending SRS stats for {home_team} vs {away_team}: {ex}")
            
            # Determine predicted winner and confidence based on blended probabilities
            winner_idx = 1 if blended_home_prob >= blended_away_prob else 0
            winner_confidence = blended_home_prob if winner_idx == 1 else blended_away_prob
            
            ou_idx = 1 if blended_over_prob >= blended_under_prob else 0
            ou_confidence = blended_over_prob if ou_idx == 1 else blended_under_prob
            
            ev_home, ev_away, kelly_home, kelly_away = 0.0, 0.0, "No Bet", "No Bet"
            game_datetime_obj = game_start_times[i]
            game_start_time_str = game_datetime_obj.isoformat() if isinstance(game_datetime_obj, datetime) else None

            try:
                if home_odd is not None and away_odd is not None:
                    ev_home = Expected_Value.expected_value(winner_confidence if winner_idx == 1 else (1.0 - winner_confidence), int(home_odd))
                    ev_away = Expected_Value.expected_value(winner_confidence if winner_idx == 0 else (1.0 - winner_confidence), int(away_odd))
                    if self.kelly_criterion:
                        kelly_home = kc.calculate_kelly_criterion(int(home_odd), winner_confidence if winner_idx == 1 else (1.0 - winner_confidence))
                        kelly_away = kc.calculate_kelly_criterion(int(away_odd), winner_confidence if winner_idx == 0 else (1.0 - winner_confidence))
            except (ValueError, TypeError): pass
            
            # Resolve team IDs to find the game_id from box_scores
            game_id = None
            db_conn = get_db_conn()
            if db_conn:
                try:
                    cursor = db_conn.cursor()
                    home_q = "%" + home_team.lower().strip() + "%"
                    away_q = "%" + away_team.lower().strip() + "%"
                    cursor.execute(
                        "SELECT team_id FROM team_metadata WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?",
                        (home_q, home_q, home_q)
                    )
                    h_row = cursor.fetchone()
                    cursor.execute(
                        "SELECT team_id FROM team_metadata WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?",
                        (away_q, away_q, away_q)
                    )
                    a_row = cursor.fetchone()
                    if h_row and a_row:
                        h_id = h_row["team_id"]
                        a_id = a_row["team_id"]
                        cursor.execute(
                            "SELECT game_id FROM box_scores WHERE home_team_id = ? AND away_team_id = ? ORDER BY game_date DESC LIMIT 1",
                            (h_id, a_id)
                        )
                        g_row = cursor.fetchone()
                        if g_row:
                            game_id = g_row["game_id"]
                        else:
                            cursor.execute(
                                "SELECT game_id FROM team_game_advanced WHERE team_id = ? AND opp_team_id = ? ORDER BY game_date DESC LIMIT 1",
                                (h_id, a_id)
                            )
                            g_row = cursor.fetchone()
                            if g_row:
                                game_id = g_row["game_id"]
                except Exception as ex:
                    logger.warning(f"Error looking up game_id for {home_team} vs {away_team}: {ex}")
                finally:
                    db_conn.close()

            predictions_list.append({
                "game_id": game_id,
                "game_identifier": f"{away_team.replace(' ', '_')}_{home_team.replace(' ', '_')}_{game_start_time_str or 'today'}",
                "home_team": home_team, 
                "away_team": away_team, 
                "home_odds": home_odd, 
                "away_odds": away_odd,
                "under_over_line": uo_lines[i], 
                "predicted_winner": home_team if winner_idx == 1 else away_team,
                "winner_confidence": round(winner_confidence * 100, 2),
                "under_over_prediction": "OVER" if ou_idx == 1 else "UNDER",
                "under_over_confidence": round(ou_confidence * 100, 2), 
                "model": self.model_name,
                "expected_value": {"home_team": ev_home, "away_team": ev_away},
                "kelly_criterion": {"home_team": kelly_home, "away_team": kelly_away},
                "game_start_time_utc": game_start_time_str,
                "home_srs_stats": {
                    "ortg": home_power["offRating"],
                    "drtg": home_power["defRating"],
                    "nrtg": home_power["netRating"],
                    "pace": home_power["pace"],
                    "efg": home_power["fourFactors"]["eFG"],
                    "tov": home_power["fourFactors"]["TOV"],
                    "orb": home_power["fourFactors"]["ORB"],
                    "ft": home_power["fourFactors"]["FT"]
                } if home_power else None,
                "away_srs_stats": {
                    "ortg": away_power["offRating"],
                    "drtg": away_power["defRating"],
                    "nrtg": away_power["netRating"],
                    "pace": away_power["pace"],
                    "efg": away_power["fourFactors"]["eFG"],
                    "tov": away_power["fourFactors"]["TOV"],
                    "orb": away_power["fourFactors"]["ORB"],
                    "ft": away_power["fourFactors"]["FT"]
                } if away_power else None
            })
        return {"sportsbook": self.sportsbook, "predictions": predictions_list}

# --- API Endpoints ---

# This is the root endpoint for http://localhost:8000/
@app.get("/")
def read_root():
    return { "message": "Welcome to the Betting Buddy API!", "status": "healthy" }

# Health check endpoint for frontend monitoring (never rate limited - uptime probes)
@app.get("/health")
@limiter.exempt
def health_check():
    return {
        "status": "healthy",
        "version": "1.1.1-stable-fixed",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Supported sportsbooks endpoint for frontend dropdown
@app.get("/sportsbooks")
def get_sportsbooks():
    return {
        "supported_sportsbooks": [
            "fanduel", "draftkings", "betmgm",
            "pointsbet", "caesars", "wynn", "bet_rivers_ny"
        ]
    }

predictions_cache = {}

# This is the predictions endpoint for http://localhost:8000/predictions
@app.get("/predictions")
@limiter.limit(RATE_LIMIT_EXPENSIVE)
def get_predictions_endpoint(request: Request, sportsbook: str = 'fanduel', kelly_criterion: bool = True, sport: str = 'NBA'):
    cache_key = f"{sportsbook}_{kelly_criterion}_{sport}"
    now = datetime.now()
    
    print(f"DEBUG_CACHE: checking cache_key='{cache_key}'. Current keys={list(predictions_cache.keys())}")
    
    if cache_key in predictions_cache:
        cached_data, timestamp = predictions_cache[cache_key]
        if now - timestamp < timedelta(minutes=5):
            print(f"DEBUG_CACHE: HIT for cache_key='{cache_key}'")
            return cached_data
        else:
            print(f"DEBUG_CACHE: EXPIRED for cache_key='{cache_key}'")
    else:
        print(f"DEBUG_CACHE: MISS for cache_key='{cache_key}'")
            
    try:
        runner = PredictionRunner(sportsbook=sportsbook, kelly_criterion=kelly_criterion, sport=sport)
        res = runner.run_predictions()
        predictions_cache[cache_key] = (res, now)
        try:
            # Log the sport the odds provider actually resolved (the NBA->WNBA
            # offseason fallback means it is not always the requested one).
            log_predictions(res, sportsbook, getattr(runner, 'resolved_sport', sport) or sport)
        except Exception as exc:
            logger.warning(f"Prediction logging failed (non-fatal): {exc}")
        return res
    except Exception as e:
        logger.error(f"Error in /predictions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Parlay Evaluation ---
class ParlayLeg(BaseModel):
    home_team: str
    away_team: str
    market: str  # 'moneyline' | 'over_under'
    pick: str    # team name, or 'over' / 'under'
    odds: float  # American odds for this leg
    model_prob: Optional[float] = None  # 0-1; auto-filled from today's predictions when omitted

class ParlayRequest(BaseModel):
    legs: List[ParlayLeg]
    sportsbook: str = 'fanduel'

def _model_prob_from_prediction(leg: ParlayLeg, prediction: Dict[str, Any]) -> Optional[float]:
    """Derive the model's probability for this leg from a /predictions game entry."""
    market = leg.market.lower().replace("-", "_")
    if market == "moneyline":
        conf = prediction.get("winner_confidence")
        winner = prediction.get("predicted_winner")
        if conf is None or not winner:
            return None
        p = float(conf) / 100.0
        return p if leg.pick.strip().lower() == str(winner).strip().lower() else 1.0 - p
    if market in ("over_under", "total"):
        conf = prediction.get("under_over_confidence")
        side = prediction.get("under_over_prediction")
        if conf is None or not side:
            return None
        p = float(conf) / 100.0
        return p if leg.pick.strip().lower() == str(side).strip().lower() else 1.0 - p
    return None

@app.post("/api/parlay/evaluate")
@limiter.limit(RATE_LIMIT_EXPENSIVE)
def evaluate_parlay_endpoint(request: Request, payload: ParlayRequest):
    """
    Evaluate a parlay ticket: combined odds, model probability, EV, Kelly stake,
    and correlation warnings. Legs without an explicit model_prob are enriched
    from the cached model predictions for today's games when available.
    """
    # Reuse a fresh-enough predictions cache entry (any kelly variant) for enrichment.
    cached_predictions: List[Dict[str, Any]] = []
    now = datetime.now()
    for kelly_variant in (True, False):
        entry = predictions_cache.get(f"{payload.sportsbook}_{kelly_variant}_NBA")
        if entry and now - entry[1] < timedelta(minutes=5):
            cached_predictions = entry[0].get("predictions", []) or []
            break

    def find_prediction(leg: ParlayLeg) -> Optional[Dict[str, Any]]:
        for pred in cached_predictions:
            if (str(pred.get("home_team", "")).strip().lower() == leg.home_team.strip().lower()
                    and str(pred.get("away_team", "")).strip().lower() == leg.away_team.strip().lower()):
                return pred
        return None

    legs_payload = []
    for leg in payload.legs:
        model_prob = leg.model_prob
        opp_odds = None
        if model_prob is None:
            pred = find_prediction(leg)
            if pred:
                model_prob = _model_prob_from_prediction(leg, pred)
                if model_prob is None and leg.market.lower().replace("-", "_") == "moneyline":
                    # The model could not price this leg, but the cached game
                    # carries both moneylines - hand the opposite quote to
                    # evaluate_parlay so its market fallback is the de-vigged
                    # fair probability instead of raw implied (vig included).
                    pick = leg.pick.strip().lower()
                    if pick == leg.home_team.strip().lower():
                        opp_odds = pred.get("away_odds")
                    elif pick == leg.away_team.strip().lower():
                        opp_odds = pred.get("home_odds")
        legs_payload.append({
            "home_team": leg.home_team,
            "away_team": leg.away_team,
            "market": leg.market,
            "pick": leg.pick,
            "odds": leg.odds,
            "model_prob": model_prob,
            "opp_odds": opp_odds,
        })

    try:
        result = parlay.evaluate_parlay(legs_payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not cached_predictions and any(l["model_prob"] is None for l in legs_payload):
        result["note"] = (
            "No live model predictions were available (call /predictions first during the "
            "season); legs without an explicit model_prob used bookmaker implied probability."
        )
    return result

# --- Line movements (from real odds snapshots) ---
@app.get("/api/line-movements")
def get_line_movements(sportsbook: str = 'fanduel', sport: str = 'NBA', hours: int = 48):
    """
    Real line movement per upcoming game, computed from stored odds snapshots.
    A game appears once at least one snapshot exists; movement is meaningful
    once the lines have changed at least once.
    """
    conn = _odds_snapshot_conn()
    try:
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            """
            SELECT * FROM odds_snapshots
            WHERE sportsbook = ? AND sport = ? AND captured_at >= ?
            ORDER BY game_key, captured_at ASC
            """,
            (sportsbook, sport, since)
        ).fetchall()

        games: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            games.setdefault(r["game_key"], []).append(r)

        movements = []
        for game_key, snaps in games.items():
            first, last = snaps[0], snaps[-1]
            movements.append({
                "game_key": game_key,
                "home_team": last["home_team"],
                "away_team": last["away_team"],
                "game_start_time_utc": last["game_start_time_utc"],
                "opening": {
                    "captured_at": first["captured_at"],
                    "home_ml": first["home_ml"],
                    "away_ml": first["away_ml"],
                    "ou_line": first["ou_line"],
                },
                "current": {
                    "captured_at": last["captured_at"],
                    "home_ml": last["home_ml"],
                    "away_ml": last["away_ml"],
                    "ou_line": last["ou_line"],
                },
                "snapshots": [
                    {
                        "captured_at": s["captured_at"],
                        "home_ml": s["home_ml"],
                        "away_ml": s["away_ml"],
                        "ou_line": s["ou_line"],
                    }
                    for s in snaps
                ],
            })
        return {"sportsbook": sportsbook, "sport": sport, "window_hours": hours, "movements": movements}
    except Exception as e:
        logger.error(f"Error in /api/line-movements: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Prediction track record endpoint ---
@app.get("/api/prediction-log")
def get_prediction_log(days: int = 30, sportsbook: Optional[str] = None):
    """Model predictions recorded before games, newest first (transparency page source)."""
    conn = _odds_snapshot_conn()
    try:
        _ensure_prediction_log_schema(conn)
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        # Market-implied fallback rows are excluded here as well as on write: this
        # endpoint feeds the public track record, and the record must contain only
        # what the model actually predicted.
        if sportsbook:
            rows = conn.execute(
                "SELECT * FROM predictions_log WHERE log_date >= ? AND sportsbook = ? "
                "AND (model IS NULL OR model != ?) ORDER BY logged_at DESC",
                (since, sportsbook, SIMULATED_MODEL_TAG)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions_log WHERE log_date >= ? "
                "AND (model IS NULL OR model != ?) ORDER BY logged_at DESC",
                (since, SIMULATED_MODEL_TAG)
            ).fetchall()

        predictions = [dict(r) for r in rows]

        # Honest topline: only graded predictions count toward the record
        graded = [p for p in predictions if p.get("actual_winner")]
        ml_correct = sum(1 for p in graded if p.get("predicted_winner") == p.get("actual_winner"))
        ou_graded = [
            p for p in graded
            if p.get("actual_total") is not None and p.get("ou_line") is not None and p.get("ou_prediction")
        ]
        ou_correct = sum(
            1 for p in ou_graded
            if (p["actual_total"] > p["ou_line"] and p["ou_prediction"].upper() == "OVER")
            or (p["actual_total"] < p["ou_line"] and p["ou_prediction"].upper() == "UNDER")
        )

        return {
            "days": days,
            "count": len(predictions),
            "summary": {
                "graded": len(graded),
                "moneyline_correct": ml_correct,
                "moneyline_pct": round(100 * ml_correct / len(graded), 1) if graded else None,
                "ou_graded": len(ou_graded),
                "ou_correct": ou_correct,
                "ou_pct": round(100 * ou_correct / len(ou_graded), 1) if ou_graded else None,
            },
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Error in /api/prediction-log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Box scores by date (the daily scores archive) ---
@app.get("/api/games/by-date/{game_date}")
def get_games_by_date(game_date: str):
    """
    All games on a calendar date (YYYY-MM-DD) with final scores, from the
    local box-score archive. Also returns the nearest earlier/later dates
    that have games so the UI can page through the calendar.
    """
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", game_date):
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT t.game_id, t.game_date, t.season, t.season_type,
                   t.team_id, t.opp_team_id, t.pts, t.opp_pts,
                   m.full_name AS team_name, m.abbreviation AS team_abbr,
                   om.full_name AS opp_name, om.abbreviation AS opp_abbr,
                   b.home_team_id
            FROM team_game_advanced t
            JOIN team_metadata m ON m.team_id = t.team_id
            JOIN team_metadata om ON om.team_id = t.opp_team_id
            JOIN box_scores b ON b.game_id = t.game_id
            WHERE t.game_date = ?
            ORDER BY t.game_id
            """,
            (game_date,),
        ).fetchall()

        games = {}
        for r in rows:
            d = dict(r)
            gid = d["game_id"]
            side = "home" if d["team_id"] == d["home_team_id"] else "away"
            g = games.setdefault(gid, {
                "game_id": gid, "game_date": d["game_date"],
                "season": d["season"], "season_type": d["season_type"],
            })
            g[side] = {
                "team_id": d["team_id"], "name": d["team_name"],
                "abbr": d["team_abbr"], "pts": d["pts"],
            }

        prev_row = conn.execute(
            "SELECT MAX(game_date) FROM team_game_advanced WHERE game_date < ?", (game_date,)
        ).fetchone()
        next_row = conn.execute(
            "SELECT MIN(game_date) FROM team_game_advanced WHERE game_date > ?", (game_date,)
        ).fetchone()

        return {
            "date": game_date,
            "games": sorted(games.values(), key=lambda g: g["game_id"]),
            "prev_date": prev_row[0] if prev_row else None,
            "next_date": next_row[0] if next_row else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching games for {game_date}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Rookies: a draft class's first-season stats ---
@app.get("/api/stats/rookies")
def get_rookies(season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    Players drafted in the season's start year, joined with their stats for
    that season. Undrafted first-year players are not detectable from draft
    records, so this is explicitly the drafted rookie class.
    """
    try:
        draft_year = int(season.split("-")[0])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="Season must look like 2025-26")
    conn = get_db_conn()
    try:
        _ensure_draft_history(conn)
        rows = conn.execute(
            """
            SELECT d.person_id AS player_id, d.player_name AS full_name,
                   d.overall_pick, d.round_number, d.team_abbreviation AS drafted_by,
                   d.organization,
                   t.gp, t.gs, t.min, t.pts, t.reb, t.ast, t.stl, t.blk, t.tov,
                   t.fg_pct, t.fg3_pct, t.ft_pct,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = t.team_id) AS team_abbr
            FROM draft_history d
            LEFT JOIN player_season_totals t
                   ON t.player_id = d.person_id AND t.season = ? AND t.season_type = ?
            WHERE d.season = ?
            ORDER BY d.overall_pick ASC
            """,
            (season, season_type, draft_year),
        ).fetchall()
        return {"season": season, "draft_year": draft_year, "rookies": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Error fetching rookies for {season}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Career highs from the game-log archive ---
@app.get("/api/players/{id}/highs")
def get_player_highs(id: int):
    """
    Career-high performances across every game log we have, per stat:
    the value, the date, and the opponent. Depth grows as more seasons
    are backfilled — honest label handled client-side.
    """
    conn = get_db_conn()
    try:
        stats = ["pts", "reb", "ast", "stl", "blk", "fg3m", "min"]
        highs = {}
        for stat in stats:
            row = conn.execute(
                f"""
                SELECT g.{stat} AS value, g.game_date, g.game_id,
                       (SELECT abbreviation FROM team_metadata WHERE team_id = t.opp_team_id) AS opp_abbr,
                       t.season, t.season_type
                FROM player_game_log g
                JOIN team_game_advanced t ON t.game_id = g.game_id AND t.team_id = g.team_id
                WHERE g.player_id = ? AND g.{stat} IS NOT NULL
                ORDER BY g.{stat} DESC, g.game_date ASC
                LIMIT 1
                """,
                (id,),
            ).fetchone()
            if row and row["value"] is not None:
                highs[stat] = dict(row)
        span = conn.execute(
            "SELECT MIN(game_date), MAX(game_date), COUNT(*) FROM player_game_log WHERE player_id = ?",
            (id,),
        ).fetchone()
        return {
            "player_id": id,
            "highs": highs,
            "games_covered": span[2] if span else 0,
            "from_date": span[0] if span else None,
            "to_date": span[1] if span else None,
        }
    except Exception as e:
        logger.error(f"Error fetching highs for player {id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Official full-career stats (stats.nba.com playercareerstats, cached) ---
CAREER_STAT_COLS = [
    "season", "team_abbr", "player_age", "gp", "gs", "min", "fgm", "fga", "fg_pct",
    "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
    "ast", "stl", "blk", "tov", "pf", "pts",
]


def _ensure_career_official(conn, player_id: int) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_career_official (
            player_id INTEGER,
            season TEXT,
            season_type TEXT,
            team_abbr TEXT,
            player_age REAL,
            gp INTEGER, gs INTEGER, min REAL,
            fgm INTEGER, fga INTEGER, fg_pct REAL,
            fg3m INTEGER, fg3a INTEGER, fg3_pct REAL,
            ftm INTEGER, fta INTEGER, ft_pct REAL,
            oreb INTEGER, dreb INTEGER, reb INTEGER,
            ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER, pts INTEGER,
            is_career_total INTEGER DEFAULT 0,
            fetched_at TEXT,
            PRIMARY KEY (player_id, season, season_type, team_abbr, is_career_total)
        )
        """
    )
    row = conn.execute(
        "SELECT MAX(fetched_at) FROM player_career_official WHERE player_id = ?", (player_id,)
    ).fetchone()
    if row and row[0] and row[0] > (datetime.utcnow() - timedelta(days=7)).isoformat():
        return

    from nba_api.stats.endpoints import playercareerstats
    data = playercareerstats.PlayerCareerStats(player_id=player_id, per_mode36="Totals").get_dict()
    sets = {rs["name"]: rs for rs in data["resultSets"]}
    now = datetime.utcnow().isoformat()

    def rows_from(set_name, season_type, is_total):
        rs = sets.get(set_name)
        if not rs:
            return []
        idx = {h: i for i, h in enumerate(rs["headers"])}
        out = []
        for r in rs["rowSet"]:
            def g(col, default=None):
                i = idx.get(col)
                return r[i] if i is not None else default
            out.append((
                player_id,
                g("SEASON_ID", "CAREER") if not is_total else "CAREER",
                season_type,
                g("TEAM_ABBREVIATION", "TOT") or "TOT",
                g("PLAYER_AGE"),
                g("GP"), g("GS"), g("MIN"),
                g("FGM"), g("FGA"), g("FG_PCT"),
                g("FG3M"), g("FG3A"), g("FG3_PCT"),
                g("FTM"), g("FTA"), g("FT_PCT"),
                g("OREB"), g("DREB"), g("REB"),
                g("AST"), g("STL"), g("BLK"), g("TOV"), g("PF"), g("PTS"),
                1 if is_total else 0,
                now,
            ))
        return out

    all_rows = (
        rows_from("SeasonTotalsRegularSeason", "Regular Season", False)
        + rows_from("SeasonTotalsPostSeason", "Playoffs", False)
        + rows_from("CareerTotalsRegularSeason", "Regular Season", True)
        + rows_from("CareerTotalsPostSeason", "Playoffs", True)
    )
    if not all_rows:
        # Endpoint returned nothing (happens for a few ids) — record the
        # attempt so we don't hammer the API, but keep any old rows.
        conn.execute(
            """
            INSERT OR REPLACE INTO player_career_official
            (player_id, season, season_type, team_abbr, is_career_total, fetched_at, gp)
            VALUES (?, '__EMPTY__', 'None', 'TOT', 0, ?, 0)
            """,
            (player_id, now),
        )
        conn.commit()
        return

    conn.execute("DELETE FROM player_career_official WHERE player_id = ?", (player_id,))
    conn.executemany(
        """
        INSERT OR REPLACE INTO player_career_official VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        all_rows,
    )
    conn.commit()


@app.get("/api/players/{id}/career-official")
def get_player_career_official(id: int):
    """
    Complete official career, season by season (regular + playoffs), plus
    career totals — from stats.nba.com, cached weekly. This covers a
    player's WHOLE career, not just the seasons in our local archive.
    """
    conn = get_db_conn()
    try:
        try:
            _ensure_career_official(conn, id)
        except Exception as exc:
            logger.warning(f"Career-official fetch failed for {id} (serving cache): {exc}")
        rows = conn.execute(
            """
            SELECT * FROM player_career_official
            WHERE player_id = ? AND season != '__EMPTY__'
            ORDER BY season
            """,
            (id,),
        ).fetchall()
        seasons = [dict(r) for r in rows if not r["is_career_total"]]
        totals = [dict(r) for r in rows if r["is_career_total"]]
        return {
            "player_id": id,
            "seasons": seasons,
            "career_totals": {t["season_type"]: t for t in totals},
        }
    except Exception as e:
        logger.error(f"Error fetching official career for {id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Milestone watch: proximity to career milestones ---
MILESTONE_STEPS = {"pts": 1000, "ast": 500, "reb": 500, "fg3m": 250, "stl": 250, "blk": 250}


@app.get("/api/stats/milestones")
def get_milestone_watch(limit: int = 25):
    """
    Active players approaching career milestones. Seeded from this season's
    top scorers/assisters/rebounders (whose official careers get cached on
    first call), then ranked by proximity to their next round number.
    """
    conn = get_db_conn()
    try:
        # Seed pool: top current-season producers across categories.
        seed_rows = conn.execute(
            """
            SELECT DISTINCT t.player_id, p.full_name
            FROM player_season_totals t
            JOIN players p ON p.player_id = t.player_id
            WHERE t.season = ? AND t.season_type = 'Regular Season' AND t.gp >= 30
            ORDER BY t.pts DESC LIMIT 60
            """,
            (CURRENT_SEASON,),
        ).fetchall()

        watch = []
        for pid, name in [(r[0], r[1]) for r in seed_rows]:
            try:
                _ensure_career_official(conn, pid)
            except Exception:
                continue
            tot = conn.execute(
                """
                SELECT pts, ast, reb, fg3m, stl, blk FROM player_career_official
                WHERE player_id = ? AND is_career_total = 1 AND season_type = 'Regular Season'
                """,
                (pid,),
            ).fetchone()
            if not tot:
                continue
            for stat, step in MILESTONE_STEPS.items():
                val = tot[stat]
                if val is None or val < step:
                    continue
                next_ms = ((val // step) + 1) * step
                remaining = next_ms - val
                # Only show genuinely close milestones (within 60% of a step)
                if remaining <= step * 0.6:
                    watch.append({
                        "player_id": pid,
                        "full_name": name,
                        "stat": stat,
                        "career_total": val,
                        "next_milestone": next_ms,
                        "remaining": remaining,
                        "pct_there": round(100 * val / next_ms, 1),
                    })

        watch.sort(key=lambda w: w["remaining"] / MILESTONE_STEPS[w["stat"]])
        return {"count": len(watch), "milestones": watch[:limit]}
    except Exception as e:
        logger.error(f"Error building milestone watch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Game Finder: Stathead-style queries over the game-log archive ---
@app.get("/api/finder/player-games")
def finder_player_games(
    min_pts: Optional[int] = None,
    min_reb: Optional[int] = None,
    min_ast: Optional[int] = None,
    min_stl: Optional[int] = None,
    min_blk: Optional[int] = None,
    min_fg3m: Optional[int] = None,
    season: Optional[str] = None,
    season_type: Optional[str] = None,
    player: Optional[str] = None,
    opponent: Optional[str] = None,
    sort: str = "pts",
    limit: int = 100,
):
    """
    Find individual player games matching stat thresholds across the whole
    archive. Example: min_pts=40&season_type=Playoffs -> every 40-point
    playoff game we have.
    """
    sortable = {"pts", "reb", "ast", "stl", "blk", "fg3m", "game_date"}
    if sort not in sortable:
        raise HTTPException(status_code=400, detail=f"sort must be one of {sorted(sortable)}")
    limit = max(1, min(int(limit), 200))

    where = ["1=1"]
    params: list = []
    for col, val in (("pts", min_pts), ("reb", min_reb), ("ast", min_ast),
                     ("stl", min_stl), ("blk", min_blk), ("fg3m", min_fg3m)):
        if val is not None:
            where.append(f"g.{col} >= ?")
            params.append(int(val))
    if season:
        where.append("t.season = ?")
        params.append(season)
    if season_type:
        where.append("t.season_type = ?")
        params.append(season_type)
    if player:
        where.append("p.full_name LIKE ?")
        params.append(f"%{player}%")
    if opponent:
        where.append("om.abbreviation = ?")
        params.append(opponent.upper())

    conn = get_db_conn()
    try:
        query = f"""
            SELECT g.game_id, g.game_date, g.player_id, p.full_name,
                   g.min, g.pts, g.reb, g.ast, g.stl, g.blk, g.fg3m, g.fgm, g.fga,
                   t.season, t.season_type, t.pts AS team_pts, t.opp_pts,
                   m.abbreviation AS team_abbr, om.abbreviation AS opp_abbr
            FROM player_game_log g
            JOIN players p ON p.player_id = g.player_id
            JOIN team_game_advanced t ON t.game_id = g.game_id AND t.team_id = g.team_id
            JOIN team_metadata m ON m.team_id = g.team_id
            JOIN team_metadata om ON om.team_id = t.opp_team_id
            WHERE {" AND ".join(where)}
            ORDER BY g.{sort} DESC, g.game_date DESC
            LIMIT ?
        """
        rows = conn.execute(query, params + [limit]).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["won"] = (d.pop("team_pts") or 0) > (d.pop("opp_pts") or 0)
            results.append(d)
        return {"count": len(results), "limit": limit, "results": results}
    except Exception as e:
        logger.error(f"Error in game finder: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Player awards (official, via stats.nba.com; cached per player) ---
def _ensure_player_awards(conn, player_id: int) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_awards (
            player_id INTEGER,
            description TEXT,
            all_nba_team_number TEXT,
            season TEXT,
            team TEXT,
            fetched_at TEXT,
            PRIMARY KEY (player_id, description, season, all_nba_team_number)
        )
        """
    )
    row = conn.execute(
        "SELECT MAX(fetched_at) FROM player_awards WHERE player_id = ?", (player_id,)
    ).fetchone()
    # Refresh at most weekly; award histories change rarely.
    if row and row[0] and row[0] > (datetime.utcnow() - timedelta(days=7)).isoformat():
        return
    from nba_api.stats.endpoints import playerawards
    data = playerawards.PlayerAwards(player_id=player_id).get_dict()
    rs = data["resultSets"][0]
    idx = {h: i for i, h in enumerate(rs["headers"])}
    now = datetime.utcnow().isoformat()
    conn.execute("DELETE FROM player_awards WHERE player_id = ?", (player_id,))
    for r in rs["rowSet"]:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_awards
            (player_id, description, all_nba_team_number, season, team, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                player_id,
                r[idx["DESCRIPTION"]],
                str(r[idx["ALL_NBA_TEAM_NUMBER"]] or ""),
                r[idx["SEASON"]],
                r[idx["TEAM"]],
                now,
            ),
        )
    conn.commit()


@app.get("/api/players/{id}/awards")
def get_player_awards(id: int):
    """Official award history: description, season, All-NBA/All-Defense team number."""
    conn = get_db_conn()
    try:
        try:
            _ensure_player_awards(conn, id)
        except Exception as exc:
            logger.warning(f"Award fetch failed for {id} (serving cache if any): {exc}")
        rows = conn.execute(
            """
            SELECT description, all_nba_team_number, season, team
            FROM player_awards WHERE player_id = ?
            ORDER BY season DESC, description
            """,
            (id,),
        ).fetchall()
        awards = [dict(r) for r in rows]
        # Grouped summary for chip rendering: {"All-NBA": ["2024-25 (1st)", ...]}
        summary = {}
        for a in awards:
            key = a["description"]
            label = a["season"] or ""
            n = a["all_nba_team_number"]
            if n and n not in ("0", "", "None", "(null)"):
                suffix = {"1": "1st", "2": "2nd", "3": "3rd"}.get(n, n)
                label = f"{label} ({suffix})" if label else suffix
            summary.setdefault(key, []).append(label)
        return {"player_id": id, "count": len(awards), "awards": awards, "summary": summary}
    except Exception as e:
        logger.error(f"Error fetching awards for {id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Team coaching staff (cached per team+season) ---
@app.get("/api/teams/{abbr}/coaches")
def get_team_coaches(abbr: str, season: str = CURRENT_SEASON):
    conn = get_db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS team_coaches (
                team_id INTEGER, season TEXT, coach_id INTEGER,
                coach_name TEXT, coach_type TEXT, sort_sequence INTEGER,
                fetched_at TEXT,
                PRIMARY KEY (team_id, season, coach_id)
            )
            """
        )
        team_row = conn.execute(
            "SELECT team_id FROM team_metadata WHERE abbreviation = ?", (abbr.upper(),)
        ).fetchone()
        if not team_row:
            raise HTTPException(status_code=404, detail=f"Unknown team {abbr}")
        team_id = team_row[0]

        cached = conn.execute(
            "SELECT MAX(fetched_at) FROM team_coaches WHERE team_id = ? AND season = ?",
            (team_id, season),
        ).fetchone()
        if not (cached and cached[0] and cached[0] > (datetime.utcnow() - timedelta(days=7)).isoformat()):
            try:
                from nba_api.stats.endpoints import commonteamroster
                data = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_dict()
                coaches_rs = next((r for r in data["resultSets"] if r["name"] == "Coaches"), None)
                if coaches_rs:
                    idx = {h: i for i, h in enumerate(coaches_rs["headers"])}
                    now = datetime.utcnow().isoformat()
                    conn.execute(
                        "DELETE FROM team_coaches WHERE team_id = ? AND season = ?", (team_id, season)
                    )
                    for r in coaches_rs["rowSet"]:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO team_coaches
                            (team_id, season, coach_id, coach_name, coach_type, sort_sequence, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                team_id, season,
                                r[idx["COACH_ID"]],
                                r[idx["COACH_NAME"]],
                                r[idx["COACH_TYPE"]] if "COACH_TYPE" in idx else "",
                                r[idx["SORT_SEQUENCE"]] if "SORT_SEQUENCE" in idx else 0,
                                now,
                            ),
                        )
                    conn.commit()
            except Exception as exc:
                logger.warning(f"Coach fetch failed for {abbr} {season} (serving cache): {exc}")

        rows = conn.execute(
            """
            SELECT coach_id, coach_name, coach_type, sort_sequence
            FROM team_coaches WHERE team_id = ? AND season = ?
            ORDER BY sort_sequence, coach_id
            """,
            (team_id, season),
        ).fetchall()
        return {"team": abbr.upper(), "season": season, "coaches": [dict(r) for r in rows]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching coaches for {abbr}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Franchise year-by-year (grows automatically as seasons backfill) ---
@app.get("/api/teams/{abbr}/franchise")
def get_franchise_history(abbr: str):
    """
    Year-by-year record for a franchise across every season in the archive:
    regular-season record + ratings, plus playoff record when applicable.
    """
    conn = get_db_conn()
    try:
        team_row = conn.execute(
            "SELECT team_id, full_name FROM team_metadata WHERE abbreviation = ?",
            (abbr.upper(),),
        ).fetchone()
        if not team_row:
            raise HTTPException(status_code=404, detail=f"Unknown team {abbr}")
        team_id, full_name = team_row[0], team_row[1]

        rows = conn.execute(
            """
            SELECT season, season_type, games, wins, losses, win_pct,
                   pace, off_rating, def_rating, net_rating, srs
            FROM team_season_advanced
            WHERE team_id = ?
            ORDER BY season DESC
            """,
            (team_id,),
        ).fetchall()

        seasons = {}
        for r in rows:
            d = dict(r)
            entry = seasons.setdefault(d["season"], {"season": d["season"]})
            if d["season_type"] == "Playoffs":
                entry["playoffs"] = {"wins": d["wins"], "losses": d["losses"]}
            else:
                entry["regular"] = {
                    k: d[k] for k in
                    ("games", "wins", "losses", "win_pct", "pace",
                     "off_rating", "def_rating", "net_rating", "srs")
                }
        out = sorted(seasons.values(), key=lambda x: x["season"], reverse=True)
        return {"team": abbr.upper(), "full_name": full_name, "seasons": out}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching franchise history for {abbr}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Draft history (official NBA draft records via stats.nba.com) ---
def _ensure_draft_history(conn) -> None:
    """Populate the draft_history table once from the NBA's official feed."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS draft_history (
            person_id INTEGER,
            player_name TEXT,
            season INTEGER,
            round_number INTEGER,
            round_pick INTEGER,
            overall_pick INTEGER,
            team_id INTEGER,
            team_city TEXT,
            team_name TEXT,
            team_abbreviation TEXT,
            organization TEXT,
            organization_type TEXT,
            fetched_at TEXT,
            PRIMARY KEY (season, overall_pick, person_id)
        )
        """
    )
    count = conn.execute("SELECT COUNT(*) FROM draft_history").fetchone()[0]
    if count > 0:
        return
    logger.info("Fetching full NBA draft history from stats.nba.com (one-time)...")
    from nba_api.stats.endpoints import drafthistory
    data = drafthistory.DraftHistory(league_id="00").get_dict()
    rs = data["resultSets"][0]
    idx = {h: i for i, h in enumerate(rs["headers"])}
    now = datetime.utcnow().isoformat()
    for row in rs["rowSet"]:
        conn.execute(
            """
            INSERT OR REPLACE INTO draft_history (
                person_id, player_name, season, round_number, round_pick,
                overall_pick, team_id, team_city, team_name, team_abbreviation,
                organization, organization_type, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row[idx["PERSON_ID"]], row[idx["PLAYER_NAME"]], row[idx["SEASON"]],
                row[idx["ROUND_NUMBER"]], row[idx["ROUND_PICK"]], row[idx["OVERALL_PICK"]],
                row[idx["TEAM_ID"]], row[idx["TEAM_CITY"]], row[idx["TEAM_NAME"]],
                row[idx["TEAM_ABBREVIATION"]], row[idx["ORGANIZATION"]],
                row[idx["ORGANIZATION_TYPE"]], now,
            ),
        )
    conn.commit()
    logger.info("Stored %d historical draft picks.", conn.execute("SELECT COUNT(*) FROM draft_history").fetchone()[0])


@app.get("/api/draft/years")
def get_draft_years():
    """All draft years on record, newest first."""
    conn = get_db_conn()
    try:
        _ensure_draft_history(conn)
        rows = conn.execute(
            "SELECT season, COUNT(*) as picks FROM draft_history GROUP BY season ORDER BY season DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error fetching draft years: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Pick value curve ---------------------------------------------------------
# What a draft slot has actually been worth, from outcomes rather than opinion.
#
# The whole exercise turns on one methodological trap: a pick from 2024 has had two
# seasons to accumulate a career and a pick from 2000 has had twenty, so comparing
# their career totals measures how long ago they were drafted. Every curve of this
# kind that ignores censoring is really a chart of draft year.
#
# So classes are only included when they have had `min_years` of opportunity, and
# the cutoff is a parameter the caller can move rather than a hidden constant. The
# response reports which classes were used and which were excluded for being too
# recent, because that choice changes the answer.
#
# Career totals come from player_career_span, summed from 1996-97 onward, so a
# player who debuted before that window has a clipped career and is excluded too.
DEFAULT_PICK_VALUE_MIN_YEARS = 9


@app.get("/api/draft/pick-value")
def get_pick_value(
    min_years: int = DEFAULT_PICK_VALUE_MIN_YEARS,
    metric: str = "gp",
    max_pick: int = 60,
):
    """
    Career outcome by draft slot, for classes old enough to have finished.

    `metric` is one of gp, min, pts - games played, minutes, points. Games is the
    default because it is the least era-sensitive of the three: scoring rates have
    moved a lot since 1996 and a game is a game.
    """
    METRICS = {"gp": "gp", "min": "min", "pts": "pts"}
    if metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"metric must be one of {sorted(METRICS)}")

    conn = get_db_conn()
    try:
        span = conn.execute(
            "SELECT MIN(window_first), MAX(window_last), COUNT(*) FROM player_career_span"
        ).fetchone()
        if not span or not span[0]:
            raise HTTPException(
                status_code=503,
                detail="Career table not built yet - run ingest_career_totals.py.",
            )
        window_first, window_last, n_players = span[0], span[1], span[2]

        # A class is eligible when it was drafted inside the career window AND has
        # had min_years of seasons since. Drafted-in-window matters because a
        # player whose career began before 1996-97 has totals clipped at the left.
        latest_eligible = window_last - min_years
        rows = conn.execute(
            f"""
            SELECT d.overall_pick AS pick, d.season AS class, d.player_name,
                   d.person_id,
                   COALESCE(c.gp, 0) AS gp, COALESCE(c.min, 0) AS min,
                   COALESCE(c.pts, 0) AS pts, COALESCE(c.seasons, 0) AS seasons,
                   CASE WHEN c.player_id IS NULL THEN 1 ELSE 0 END AS never_played
            FROM draft_history d
            LEFT JOIN player_career_span c ON c.player_id = d.person_id
            WHERE d.season >= ? AND d.season <= ? AND d.overall_pick BETWEEN 1 AND ?
            ORDER BY d.overall_pick, d.season
            """,
            (window_first, latest_eligible, max_pick),
        ).fetchall()

        by_pick: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_pick.setdefault(r["pick"], []).append(dict(r))

        col = METRICS[metric]
        picks = []
        for pick in sorted(by_pick):
            group = by_pick[pick]
            vals = sorted(float(g[col] or 0) for g in group)
            n = len(vals)
            if not n:
                continue
            # Median, not mean: one Hall of Famer at pick 13 would drag a mean up
            # and imply every pick 13 is worth that.
            median = vals[n // 2]
            best = max(group, key=lambda g: g[col] or 0)
            picks.append({
                "pick": pick,
                "n_players": n,
                "median": round(median, 1),
                "p25": round(vals[int(0.25 * n)], 1),
                "p75": round(vals[int(0.75 * n)], 1),
                "mean": round(sum(vals) / n, 1),
                "max": round(vals[-1], 1),
                "best_player": {"name": best["player_name"], "class": best["class"],
                                "value": round(float(best[col] or 0), 1)},
                # Two rates that say more than an average: how often a slot returns
                # a real NBA career, and how often it returns nothing at all.
                "share_400_games": round(
                    sum(1 for g in group if (g["gp"] or 0) >= 400) / n * 100, 1),
                "share_never_played": round(
                    sum(1 for g in group if g["never_played"]) / n * 100, 1),
            })

        excluded = conn.execute(
            "SELECT COUNT(DISTINCT season) FROM draft_history WHERE season > ?",
            (latest_eligible,),
        ).fetchone()[0]

        lottery = [p for p in picks if p["pick"] <= 14]
        second = [p for p in picks if p["pick"] >= 31]

        # Twenty-one players per slot is not enough to rank slot against slot: in
        # this window the median at pick 5 is HIGHER than at pick 1, which is noise,
        # not a finding about pick 5. Bands pool slots until the sample is large
        # enough for the trend to be the signal, and the page leads with these.
        BANDS = [(1, 3), (4, 7), (8, 14), (15, 20), (21, 30), (31, 45), (46, 60)]
        bands = []
        for lo, hi in BANDS:
            group = [g for pick in range(lo, hi + 1) for g in by_pick.get(pick, [])]
            if not group:
                continue
            vals = sorted(float(g[col] or 0) for g in group)
            n = len(vals)
            bands.append({
                "label": f"{lo}-{hi}" if lo != hi else str(lo),
                "from_pick": lo, "to_pick": hi,
                "n_players": n,
                "median": round(vals[n // 2], 1),
                "p25": round(vals[int(0.25 * n)], 1),
                "p75": round(vals[int(0.75 * n)], 1),
                "share_400_games": round(
                    sum(1 for g in group if (g["gp"] or 0) >= 400) / n * 100, 1),
                "share_never_played": round(
                    sum(1 for g in group if g["never_played"]) / n * 100, 1),
            })
        return {
            "metric": metric,
            "metric_label": {"gp": "career games", "min": "career minutes",
                             "pts": "career points"}[metric],
            "min_years": min_years,
            "classes": {"first": window_first, "last": latest_eligible,
                        "excluded_recent": excluded},
            "career_window": {"first_season": window_first, "last_season": window_last,
                              "players": n_players},
            "picks": picks,
            "bands": bands,
            # The two numbers the curve exists to produce.
            "summary": {
                "pick1_median": next((p["median"] for p in picks if p["pick"] == 1), None),
                "pick14_median": next((p["median"] for p in picks if p["pick"] == 14), None),
                "pick30_median": next((p["median"] for p in picks if p["pick"] == 30), None),
                "lottery_median": round(
                    sum(p["median"] for p in lottery) / len(lottery), 1) if lottery else None,
                "second_round_median": round(
                    sum(p["median"] for p in second) / len(second), 1) if second else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pick-value curve: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not compute the pick value curve.")
    finally:
        conn.close()


# NOTE: this route must stay ABOVE /api/draft/{year}. FastAPI matches in
# definition order, and "pipeline" cannot be parsed as an int - below it, this
# path would 422 instead of resolving.
@app.get("/api/draft/pipeline")
def get_draft_pipeline(since: int = 2000, limit: int = 40):
    """
    Where drafted players actually come from: programs ranked by picks produced.

    This is the honest version of a "prospects" page. Pre-draft rankings are
    scouting opinion and no feed we can reach publishes them, but which programs
    have actually produced NBA picks is a matter of record - 8,434 of them.

    Volume and quality are reported separately on purpose. A program can send
    plenty of players to the league without sending high ones, and one number
    covering both would hide the more interesting half.
    """
    conn = get_db_conn()
    try:
        rows = conn.execute(
            """
            SELECT organization AS org,
                   organization_type AS org_type,
                   COUNT(*) AS picks,
                   SUM(CASE WHEN overall_pick <= 14 THEN 1 ELSE 0 END) AS lottery,
                   SUM(CASE WHEN round_number = 1 THEN 1 ELSE 0 END) AS first_round,
                   MIN(overall_pick) AS best_pick,
                   MIN(season) AS first_season,
                   MAX(season) AS last_season
            FROM draft_history
            WHERE season >= ? AND organization IS NOT NULL AND organization != ''
            GROUP BY organization, organization_type
            ORDER BY picks DESC, lottery DESC
            LIMIT ?
            """,
            (since, limit),
        ).fetchall()

        programs = []
        for r in rows:
            d = dict(r)
            picks = d["picks"] or 0
            d["lottery_rate"] = round((d["lottery"] or 0) / picks * 100, 1) if picks else None
            # Who the program's highest pick actually was, since "best pick 1" is
            # a number until it has a name attached.
            top = conn.execute(
                """
                SELECT player_name, season, overall_pick, team_abbreviation
                FROM draft_history
                WHERE organization = ? AND season >= ? AND overall_pick = ?
                ORDER BY season DESC LIMIT 1
                """,
                (d["org"], since, d["best_pick"]),
            ).fetchone()
            d["best_player"] = dict(top) if top else None
            programs.append(d)

        totals = conn.execute(
            """
            SELECT COUNT(*) AS picks,
                   COUNT(DISTINCT organization) AS orgs,
                   MIN(season) AS first_season,
                   MAX(season) AS last_season
            FROM draft_history WHERE season >= ?
            """,
            (since,),
        ).fetchone()

        by_type = [
            dict(r) for r in conn.execute(
                """
                SELECT organization_type AS org_type, COUNT(*) AS picks
                FROM draft_history WHERE season >= ?
                GROUP BY organization_type ORDER BY picks DESC
                """,
                (since,),
            ).fetchall()
        ]

        return {
            "since": since,
            "totals": dict(totals) if totals else {},
            "by_type": by_type,
            "programs": programs,
        }
    except Exception as e:
        logger.error(f"Error in /api/draft/pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not compute the draft pipeline.")
    finally:
        conn.close()


@app.get("/api/draft/{year}")
def get_draft_class(year: int):
    """
    Full draft class for a year: every pick in order, flagged with whether the
    player exists in our stats database (so the UI can link to their page).
    """
    conn = get_db_conn()
    try:
        _ensure_draft_history(conn)
        # position / height / weight / country come from player_bio, filled for
        # recent classes by ingest_draft_bios.py. They are LEFT JOINed so an
        # unfilled class still returns its picks rather than nothing - a board
        # missing a weight column is fine, a board missing its picks is not.
        rows = conn.execute(
            """
            SELECT d.*,
                   CASE WHEN p.player_id IS NOT NULL THEN 1 ELSE 0 END AS in_database,
                   b.position, b.height, b.weight, b.country,
                   b.team_abbr AS current_team
            FROM draft_history d
            LEFT JOIN players p ON p.player_id = d.person_id
            LEFT JOIN player_bio b ON b.player_id = d.person_id
            WHERE d.season = ?
            ORDER BY d.overall_pick ASC
            """,
            (year,),
        ).fetchall()
        picks = [dict(r) for r in rows]

        # Where a pick ended up. No feed maps a draft-night trade to the pick it
        # moved, but the player's current team is on record, and a current team
        # that differs from the drafting team IS the move - 26 of the 2026 class.
        # Called "moved_to" rather than "traded_to" because a later trade or a
        # waiver-and-signing produces the same difference, and the endpoint should
        # not claim to know which happened.
        for p in picks:
            cur = p.get("current_team")
            p["moved_to"] = cur if cur and cur != p.get("team_abbreviation") else None
        return {
            "year": year,
            "count": len(picks),
            "rounds": sorted({p["round_number"] for p in picks if p.get("round_number")}),
            "bios_available": sum(1 for p in picks if p.get("height")),
            "picks": picks,
        }
    except Exception as e:
        logger.error(f"Error fetching draft class {year}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Historical Data Endpoints ---
@app.get("/api/historical/team-stats")
def get_historical_team_stats(team: str, season: int):
    try:
        # season is ending year, e.g. 2025. Convert it to "2024-25"
        season_str = f"{season-1}-{str(season)[2:]}"
        
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'TeamData.sqlite')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            
            # Resolve team_id
            team_q = "%" + team.lower().strip() + "%"
            cursor.execute(
                """
                SELECT team_id, abbreviation, full_name FROM team_metadata
                WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?
                """,
                (team_q, team_q, team_q)
            )
            meta_row = cursor.fetchone()
            if not meta_row:
                br_abbr = TEAM_TO_BR_ABBR.get(team.lower())
                if br_abbr:
                    cursor.execute("SELECT team_id, abbreviation, full_name FROM team_metadata WHERE abbreviation = ?", (br_abbr,))
                    meta_row = cursor.fetchone()
                    
            if not meta_row:
                raise HTTPException(status_code=404, detail=f"Team '{team}' not found in database metadata.")
                
            team_id = meta_row["team_id"]
            
            # Fetch from team_season_advanced
            cursor.execute(
                """
                SELECT * FROM team_season_advanced
                WHERE team_id = ? AND season = ? AND season_type = 'Regular Season'
                """,
                (team_id, season_str)
            )
            stats_row = cursor.fetchone()
            
            if not stats_row:
                return {
                    "team": meta_row["full_name"],
                    "season": season,
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_pct": 0,
                    "pace": 0,
                    "off_rating": 0,
                    "def_rating": 0,
                    "net_rating": 0,
                    "srs": 0,
                    "sos": 0
                }
                
            res = dict(stats_row)
            res["Pace"] = res["pace"]
            res["ORtg"] = res["off_rating"]
            res["DRtg"] = res["def_rating"]
            res["NetRtg"] = res["net_rating"]
            res["SRS"] = res["srs"]
            res["SOS"] = res["sos"]
            res["Win_Pct"] = res["win_pct"]
            res["Wins"] = res["wins"]
            res["Losses"] = res["losses"]
            res["team"] = meta_row["full_name"]
            
            return res
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching historical team stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical/matchup")
def get_historical_matchup(team1: str, team2: str, season: int):
    try:
        season_str = f"{season-1}-{str(season)[2:]}"
        
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'TeamData.sqlite')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            
            def resolve_id(t_name):
                t_q = "%" + t_name.lower().strip() + "%"
                cursor.execute(
                    """
                    SELECT team_id, full_name FROM team_metadata
                    WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?
                    """,
                    (t_q, t_q, t_q)
                )
                r = cursor.fetchone()
                if not r:
                    br = TEAM_TO_BR_ABBR.get(t_name.lower())
                    if br:
                        cursor.execute("SELECT team_id, full_name FROM team_metadata WHERE abbreviation = ?", (br,))
                        r = cursor.fetchone()
                return r
                
            res1 = resolve_id(team1)
            res2 = resolve_id(team2)
            
            if not res1 or not res2:
                return {
                    "team1": team1.upper(),
                    "team2": team2.upper(),
                    "season": season,
                    "total_games": 0,
                    "win_percentage": {team1.upper(): 0, team2.upper(): 0},
                    "wins": {team1.upper(): 0, team2.upper(): 0},
                    "matchups": []
                }
                
            tid1, name1 = res1["team_id"], res1["full_name"]
            tid2, name2 = res2["team_id"], res2["full_name"]
            
            cursor.execute(
                """
                SELECT game_id, team_id, opp_team_id, game_date, pts, opp_pts
                FROM team_game_advanced
                WHERE team_id = ? AND opp_team_id = ? AND season = ?
                """,
                (tid1, tid2, season_str)
            )
            rows = cursor.fetchall()
            
            matchups_list = []
            team1_wins = 0
            team2_wins = 0
            total_games = 0
            
            for r in rows:
                pts1 = r["pts"] if r["pts"] is not None else 0
                pts2 = r["opp_pts"] if r["opp_pts"] is not None else 0
                
                cursor.execute("SELECT home_team_id FROM box_scores WHERE game_id = ?", (r["game_id"],))
                bs = cursor.fetchone()
                is_t1_home = True
                if bs:
                    is_t1_home = int(bs["home_team_id"]) == tid1
                    
                visitor = name2 if is_t1_home else name1
                home = name1 if is_t1_home else name2
                visitor_pts = pts2 if is_t1_home else pts1
                home_pts = pts1 if is_t1_home else pts2
                
                winner = name1 if pts1 > pts2 else name2
                winner_pts = pts1 if pts1 > pts2 else pts2
                loser = name2 if pts1 > pts2 else name1
                loser_pts = pts2 if pts1 > pts2 else pts1
                
                if pts1 > pts2:
                    team1_wins += 1
                else:
                    team2_wins += 1
                    
                total_games += 1
                matchups_list.append({
                    "date": r["game_date"],
                    "visitor": visitor,
                    "visitor_pts": visitor_pts,
                    "home": home,
                    "home_pts": home_pts,
                    "winner": winner,
                    "score_summary": f"{winner} {winner_pts} - {loser_pts} {loser}"
                })
                
            win_pct1 = round((team1_wins / total_games) * 100, 1) if total_games > 0 else 0
            win_pct2 = round((team2_wins / total_games) * 100, 1) if total_games > 0 else 0
            
            return {
                "team1": team1.upper(),
                "team2": team2.upper(),
                "season": season,
                "total_games": total_games,
                "win_percentage": {
                    team1.upper(): win_pct1,
                    team2.upper(): win_pct2
                },
                "wins": {
                    team1.upper(): team1_wins,
                    team2.upper(): team2_wins
                },
                "matchups": matchups_list
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching historical matchup details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Cache for shot charts
shot_chart_cache = {}

# League-wide FG% by shot zone, keyed by "{season}_{season_type}". League
# averages are league-wide (not per player/game), so ONE upstream call per
# season serves every shot-chart request; also disk-cached in Data/nba_cache.
league_shot_averages_cache: Dict[str, List[Dict[str, Any]]] = {}

def _season_from_game_id(game_id: str) -> str:
    """Derive the season string ("2024-25") from an NBA game id: digits 3-4
    are the season start year modulo 100 (e.g. "0022400001" -> 2024-25)."""
    yy = int(game_id[3:5])
    start = 2000 + yy if yy < 90 else 1900 + yy
    return f"{start}-{(start + 1) % 100:02d}"

def _get_league_shot_averages(season: str, season_type: str = "Regular Season") -> List[Dict[str, Any]]:
    """
    Zone-level league-average shooting for a season, in response shape:
    [{"zone_basic", "zone_area", "zone_range", "fga", "fgm", "fg_pct"}].
    Returns [] on failure so the additive field never breaks an endpoint.
    """
    cache_key = f"{season}_{season_type}"
    if cache_key in league_shot_averages_cache:
        return league_shot_averages_cache[cache_key]
    try:
        from src.Utils.nba_stats_client import get_client
        rows = get_client().league_shot_averages(season, season_type)
        averages = [
            {
                "zone_basic": r.get("SHOT_ZONE_BASIC"),
                "zone_area": r.get("SHOT_ZONE_AREA"),
                "zone_range": r.get("SHOT_ZONE_RANGE"),
                "fga": r.get("FGA"),
                "fgm": r.get("FGM"),
                "fg_pct": r.get("FG_PCT"),
            }
            for r in rows
        ]
        if averages:
            league_shot_averages_cache[cache_key] = averages
        return averages
    except Exception as e:
        logger.warning(f"League shot averages fetch failed for {season} ({season_type}): {e}")
        return []

TEAM_TO_BR_ABBR = {
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BRK", "charlotte hornets": "CHO", "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE", "dallas mavericks": "DAL", "denver nuggets": "DEN", "detroit pistons": "DET", "golden state warriors": "GSW",
    "houston rockets": "HOU", "indiana pacers": "IND", "los angeles clippers": "LAC", "los angeles lakers": "LAL", "memphis grizzlies": "MEM",
    "miami heat": "MIA", "milwaukee bucks": "MIL", "minnesota timberwolves": "MIN", "new orleans pelicans": "NOP", "new york knicks": "NYK",
    "oklahoma city thunder": "OKC", "orlando magic": "ORL", "philadelphia 76ers": "PHI", "phoenix suns": "PHO", "portland trail blazers": "POR",
    "sacramento kings": "SAC", "san antonio spurs": "SAS", "toronto raptors": "TOR", "utah jazz": "UTA", "washington wizards": "WAS",
    "atl": "ATL", "bos": "BOS", "brk": "BRK", "bkn": "BRK", "cho": "CHO", "cha": "CHO", "chi": "CHI", "cle": "CLE", "dal": "DAL",
    "den": "DEN", "det": "DET", "gsw": "GSW", "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM", "mia": "MIA",
    "mil": "MIL", "min": "MIN", "nop": "NOP", "nyk": "NYK", "okc": "OKC", "orl": "ORL", "phi": "PHI", "pho": "PHO", "por": "POR",
    "sac": "SAC", "sas": "SAS", "tor": "TOR", "uta": "UTA", "was": "WAS"
}

TATUM_MOCK_SHOTS = [
    {"player": "Jayson Tatum", "x": 250, "y": 50, "result": "made", "description": "1st Q, 10:14 remaining, Jayson Tatum makes 3-pointer from 26 ft"},
    {"player": "Jayson Tatum", "x": 120, "y": 80, "result": "missed", "description": "1st Q, 8:45 remaining, Jayson Tatum misses 2-pointer from 12 ft"},
    {"player": "Jayson Tatum", "x": 260, "y": 45, "result": "made", "description": "1st Q, 5:12 remaining, Jayson Tatum makes 3-pointer from 28 ft"},
    {"player": "Jayson Tatum", "x": 250, "y": 240, "result": "made", "description": "1st Q, 3:20 remaining, Jayson Tatum makes driving layup"},
    {"player": "Jayson Tatum", "x": 380, "y": 120, "result": "missed", "description": "2nd Q, 11:05 remaining, Jayson Tatum misses 3-pointer from 24 ft"},
    {"player": "Jayson Tatum", "x": 255, "y": 235, "result": "made", "description": "2nd Q, 9:40 remaining, Jayson Tatum makes dunk"},
    {"player": "Jayson Tatum", "x": 245, "y": 242, "result": "made", "description": "2nd Q, 6:15 remaining, Jayson Tatum makes running layup"},
    {"player": "Jayson Tatum", "x": 80, "y": 150, "result": "made", "description": "2nd Q, 2:50 remaining, Jayson Tatum makes 2-pointer from 15 ft"},
    {"player": "Jayson Tatum", "x": 250, "y": 55, "result": "missed", "description": "3rd Q, 10:30 remaining, Jayson Tatum misses 3-pointer from 25 ft"},
    {"player": "Jayson Tatum", "x": 350, "y": 180, "result": "made", "description": "3rd Q, 8:15 remaining, Jayson Tatum makes 2-pointer from 18 ft"},
    {"player": "Jayson Tatum", "x": 248, "y": 238, "result": "made", "description": "3rd Q, 5:04 remaining, Jayson Tatum makes layup"},
    {"player": "Jayson Tatum", "x": 100, "y": 70, "result": "made", "description": "3rd Q, 1:40 remaining, Jayson Tatum makes 3-pointer from 25 ft"},
    {"player": "Jayson Tatum", "x": 252, "y": 245, "result": "missed", "description": "4th Q, 11:20 remaining, Jayson Tatum misses tip-in"},
    {"player": "Jayson Tatum", "x": 250, "y": 240, "result": "made", "description": "4th Q, 9:05 remaining, Jayson Tatum makes driving dunk"},
    {"player": "Jayson Tatum", "x": 265, "y": 48, "result": "made", "description": "4th Q, 7:30 remaining, Jayson Tatum makes 3-pointer from 29 ft"},
    {"player": "Jayson Tatum", "x": 150, "y": 110, "result": "missed", "description": "4th Q, 4:10 remaining, Jayson Tatum misses 2-pointer from 10 ft"},
    {"player": "Jayson Tatum", "x": 250, "y": 240, "result": "made", "description": "4th Q, 1:55 remaining, Jayson Tatum makes driving layup"},
    {"player": "Jayson Tatum", "x": 245, "y": 140, "result": "made", "description": "4th Q, 0:45 remaining, Jayson Tatum makes 2-pointer from 11 ft"},
    {"player": "Jaylen Brown", "x": 250, "y": 242, "result": "made", "description": "1st Q, 11:30 remaining, Jaylen Brown makes dunk"},
    {"player": "Jaylen Brown", "x": 100, "y": 60, "result": "missed", "description": "1st Q, 6:40 remaining, Jaylen Brown misses 3-pointer from 26 ft"},
    {"player": "Jaylen Brown", "x": 280, "y": 140, "result": "made", "description": "2nd Q, 4:15 remaining, Jaylen Brown makes 2-pointer from 12 ft"},
    {"player": "Trae Young", "x": 252, "y": 50, "result": "made", "description": "1st Q, 9:55 remaining, Trae Young makes 3-pointer from 27 ft"},
    {"player": "Trae Young", "x": 248, "y": 150, "result": "missed", "description": "2nd Q, 8:20 remaining, Trae Young misses driving floater"},
    {"player": "Trae Young", "x": 250, "y": 240, "result": "made", "description": "3rd Q, 7:10 remaining, Trae Young makes driving layup"}
]

@app.get("/api/shot-chart")
@limiter.limit(RATE_LIMIT_UPSTREAM)
def get_shot_chart(request: Request, game_date: str, home_team: str):
    """
    Fetch shot chart details for a game, resolved via database and retrieved from NBA Stats API.
    """
    # 1. Format date: "20230415" -> "2023-04-15"
    formatted_date = game_date
    if len(game_date) == 8:
        formatted_date = f"{game_date[0:4]}-{game_date[4:6]}-{game_date[6:8]}"
        
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # 2. Resolve team_id
        team_q = "%" + home_team.lower().strip() + "%"
        cursor.execute(
            "SELECT team_id FROM team_metadata WHERE lower(full_name) LIKE ? OR lower(nickname) LIKE ? OR lower(abbreviation) LIKE ?",
            (team_q, team_q, team_q)
        )
        row = cursor.fetchone()
        if not row:
            # Try to match BBRef abbreviations
            abbr = TEAM_TO_BR_ABBR.get(home_team.strip().lower(), home_team.upper())
            cursor.execute("SELECT team_id FROM team_metadata WHERE abbreviation = ?", (abbr,))
            row = cursor.fetchone()
            
        if not row:
            raise HTTPException(status_code=404, detail=f"Home team {home_team} not found.")
            
        team_id = row["team_id"]
        
        # 3. Find game_id in database
        cursor.execute(
            "SELECT game_id FROM team_game_advanced WHERE game_date = ? AND team_id = ? LIMIT 1",
            (formatted_date, team_id)
        )
        game_row = cursor.fetchone()
        if not game_row:
            # Try to look in box_scores
            cursor.execute(
                "SELECT game_id FROM box_scores WHERE game_date = ? AND home_team_id = ? LIMIT 1",
                (formatted_date, team_id)
            )
            game_row = cursor.fetchone()
            
        if not game_row:
            raise HTTPException(
                status_code=404, 
                detail=f"Game for home team {home_team} on {formatted_date} not found in database."
            )
            
        game_id = game_row["game_id"]
        
        # 4. Fetch shots using ShotChartDetail
        if game_id in shot_chart_cache:
            logger.info(f"Returning cached shot chart data for game: {game_id}")
            return shot_chart_cache[game_id]
            
        logger.info(f"Fetching shot chart detail from NBA Stats API for game: {game_id}")
        sc = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            game_id_nullable=game_id,
            context_measure_simple="FGA",
            season_type_all_star="Regular Season"
        )
        data = sc.get_dict()
        
        result_sets = data.get("resultSets", [])
        shots_set = next((rs for rs in result_sets if rs.get("name") == "Shot_Chart_Detail"), None)
        shots = []
        if shots_set:
            headers = shots_set.get("headers", [])
            row_set = shots_set.get("rowSet", [])
            col_map = {h: idx for idx, h in enumerate(headers)}
            
            for row in row_set:
                try:
                    x = row[col_map["LOC_X"]]
                    y = row[col_map["LOC_Y"]]
                    made = row[col_map["SHOT_MADE_FLAG"]] == 1
                    player_name = row[col_map["PLAYER_NAME"]]
                    desc = f"{player_name} {'makes' if made else 'misses'} {row[col_map['SHOT_TYPE']]} from {row[col_map['SHOT_DISTANCE']]} ft"
                    
                    shots.append({
                        "player": player_name,
                        "x": x,
                        "y": y,
                        "result": "made" if made else "missed",
                        "description": desc
                    })
                except Exception:
                    continue
                    
        response_data = {
            "game_id": game_id,
            "shots": shots
        }
        shot_chart_cache[game_id] = response_data
        return response_data
        
    except Exception as e:
        logger.error(f"Error fetching game shot chart: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Caches
active_players_cache = []

@app.get("/api/players")
def get_active_players():
    global active_players_cache
    if active_players_cache:
        return active_players_cache
        
    if not nba_players:
        raise HTTPException(status_code=500, detail="nba_api library not imported")
        
    try:
        raw_players = nba_players.get_active_players()
        active_players_cache = [
            {
                "id": p["id"],
                "name": p["full_name"],
                "first_name": p["first_name"],
                "last_name": p["last_name"]
            }
            for p in raw_players
        ]
        active_players_cache.sort(key=lambda x: x["name"])
        return active_players_cache
    except Exception as e:
        logger.error(f"Error fetching active players: {e}")
        raise HTTPException(status_code=500, detail=str(e))

player_shot_chart_cache = {}

@app.get("/api/player-shot-chart")
@limiter.limit(RATE_LIMIT_UPSTREAM)
def get_player_shot_chart(request: Request, player_id: int, season: str = CURRENT_SEASON):
    """
    Per-shot chart data for a player-season, plus league-average FG% by zone.

    Coordinate space (stats.nba.com shotchartdetail convention): shot x/y are
    LOC_X / LOC_Y in tenths of feet with the basket at the origin -
    x in [-250, 250] (negative = left of the basket from the shooter's view),
    y in [-52, ~890] (increasing toward half court).

    `league_averages` (additive field) carries the same rows as the legacy
    `averages` field, matching the shared shot-chart response contract.
    """
    cache_key = f"{player_id}_{season}"
    if cache_key in player_shot_chart_cache:
        logger.info(f"Returning cached player shot chart for key: {cache_key}")
        return player_shot_chart_cache[cache_key]
        
    if not shotchartdetail:
        raise HTTPException(status_code=500, detail="nba_api library not imported")
        
    try:
        logger.info(f"Fetching shot chart detail from NBA stats API for player: {player_id}, season: {season}")
        shot_chart = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable=season,
            context_measure_simple="FGA",
            season_type_all_star="Regular Season"
        )
        data = shot_chart.get_dict()
        
        result_sets = data.get("resultSets", [])
        
        # 1. Parse individual shots
        shots_set = next((rs for rs in result_sets if rs.get("name") == "Shot_Chart_Detail"), None)
        shots = []
        if shots_set:
            headers = shots_set.get("headers", [])
            row_set = shots_set.get("rowSet", [])
            col_map = {h: idx for idx, h in enumerate(headers)}
            
            for row in row_set:
                try:
                    shots.append({
                        "game_id": row[col_map["GAME_ID"]],
                        "game_date": row[col_map["GAME_DATE"]],
                        "event_type": row[col_map["EVENT_TYPE"]],
                        "action_type": row[col_map["ACTION_TYPE"]],
                        "shot_type": row[col_map["SHOT_TYPE"]],
                        "zone_basic": row[col_map["SHOT_ZONE_BASIC"]],
                        "zone_area": row[col_map["SHOT_ZONE_AREA"]],
                        "zone_range": row[col_map["SHOT_ZONE_RANGE"]],
                        "distance": row[col_map["SHOT_DISTANCE"]],
                        "x": row[col_map["LOC_X"]],
                        "y": row[col_map["LOC_Y"]],
                        "made": row[col_map["SHOT_MADE_FLAG"]] == 1,
                        "period": row[col_map["PERIOD"]],
                        "game_event_id": row[col_map["GAME_EVENT_ID"]]
                    })
                except Exception:
                    continue

        # 1b. Annotate each shot with the run the player was on coming into it,
        # within that game: streak_before > 0 means that many straight makes
        # immediately before this attempt, < 0 straight misses, 0 first attempt
        # of the game. GAME_EVENT_ID is monotonic within a game, so ordering
        # needs no play-by-play join. Field goals only - free throws neither
        # extend nor break a run here.
        by_game = {}
        for s in shots:
            by_game.setdefault(s["game_id"], []).append(s)
        for game_shots in by_game.values():
            game_shots.sort(key=lambda s: s.get("game_event_id") or 0)
            streak = 0
            for s in game_shots:
                s["streak_before"] = streak
                if s["made"]:
                    streak = streak + 1 if streak > 0 else 1
                else:
                    streak = streak - 1 if streak < 0 else -1

        # 2. Parse league averages
        avg_set = next((rs for rs in result_sets if rs.get("name") == "LeagueAverages"), None)
        averages = []
        if avg_set:
            headers = avg_set.get("headers", [])
            row_set = avg_set.get("rowSet", [])
            col_map = {h: idx for idx, h in enumerate(headers)}
            
            for row in row_set:
                try:
                    averages.append({
                        "zone_basic": row[col_map["SHOT_ZONE_BASIC"]],
                        "zone_area": row[col_map["SHOT_ZONE_AREA"]],
                        "zone_range": row[col_map["SHOT_ZONE_RANGE"]],
                        "fga": row[col_map["FGA"]],
                        "fgm": row[col_map["FGM"]],
                        "fg_pct": row[col_map["FG_PCT"]]
                    })
                except Exception:
                    continue
                    
        response_data = {
            "player_id": player_id,
            "season": season,
            "shots": shots,
            "averages": averages,
            # Additive alias: same zone rows under the contract field name.
            "league_averages": averages
        }
        
        player_shot_chart_cache[cache_key] = response_data
        return response_data
        
    except Exception as e:
        logger.error(f"Error in get_player_shot_chart API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

player_stats_cache = {}

@app.get("/api/player-stats")
@limiter.limit(RATE_LIMIT_UPSTREAM)
def get_player_stats(request: Request, season: str = "2025-26", per_mode: str = "PerGame", measure_type: str = "Base"):
    cache_key = f"{season}_{per_mode}_{measure_type}"
    now = datetime.now()
    
    if cache_key in player_stats_cache:
        cached_data, timestamp = player_stats_cache[cache_key]
        if now - timestamp < timedelta(minutes=5):
            logger.info(f"Returning cached player stats for key: {cache_key}")
            return cached_data
            
    # Check SQLite database cache first
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'TeamData.sqlite')
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Join the name in. player_season_stats carries a player_name column
            # that is never populated, so this endpoint had been returning rows
            # of numbers with no way to tell whose they were.
            cursor.execute(
                "SELECT s.*, p.full_name AS joined_name "
                "FROM player_season_stats s "
                "LEFT JOIN players p ON p.player_id = s.player_id "
                "WHERE s.season = ? AND s.season_type = 'Regular Season'",
                (season,)
            )
            rows = cursor.fetchall()
            if rows:
                logger.info(f"Returning cached player stats from SQLite for season: {season}")
                results = []
                for r in rows:
                    rec = dict(r)
                    rec.pop("id", None)
                    rec.pop("fetched_at", None)
                    joined = rec.pop("joined_name", None)
                    if joined and not rec.get("player_name"):
                        rec["player_name"] = joined
                    results.append(rec)
                conn.close()
                player_stats_cache[cache_key] = (results, now)
                return results
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to query player stats from database cache: {e}")

    if not leaguedashplayerstats:
        raise HTTPException(status_code=500, detail="nba_api library not imported")
        
    try:
        logger.info(f"Fetching league-wide player stats for season: {season}, per_mode: {per_mode}")
        
        # 1. Fetch Base stats
        base_endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Base"
        )
        base_df = base_endpoint.get_data_frames()[0]
        
        # 2. Fetch Advanced stats
        adv_endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Advanced"
        )
        adv_df = adv_endpoint.get_data_frames()[0]
        
        if base_df.empty or adv_df.empty:
            return []
            
        # 3. Merge dataframes on PLAYER_ID
        merged = pd.merge(
            base_df,
            adv_df[['PLAYER_ID', 'TS_PCT', 'USG_PCT', 'DEF_RATING', 'NET_RATING']],
            on='PLAYER_ID',
            how='inner'
        )
        
        # 4. Calculate Player Power Index
        # Formula: (BPM * 1.5) + (TS% * 20) + (USG% * 0.5) - (DRtg * 0.8) + (OnOff * 1.2)
        bpm = merged['PLUS_MINUS'].fillna(0)
        ts = merged['TS_PCT'].fillna(0) * 100
        usg = merged['USG_PCT'].fillna(0) * 100
        drtg = merged['DEF_RATING'].fillna(110)
        onoff = merged['NET_RATING'].fillna(0)
        
        merged['power_index'] = (bpm * 1.5) + (ts * 20) + (usg * 0.5) - (drtg * 0.8) + (onoff * 1.2)
        
        # Apply qualification check: subtract 500 penalty if GP < 5 or MIN < 10 to filter outliers
        def apply_qualification_penalty(row):
            gp = row.get('GP', 0)
            min_val = row.get('MIN', 0)
            pi = row.get('power_index', 0.0)
            if gp < 5 or min_val < 10:
                return pi - 500.0
            return pi
            
        merged['power_index'] = merged.apply(apply_qualification_penalty, axis=1)
        
        # 5. Extract results matching the requested measure_type
        target_cols = base_df.columns.tolist() if measure_type == "Base" else adv_df.columns.tolist()
        # Add power_index, ts_pct, usg_pct, def_rating, net_rating so they are always accessible
        extra_cols = ['power_index', 'TS_PCT', 'USG_PCT', 'DEF_RATING', 'NET_RATING']
        all_cols = list(set(target_cols + extra_cols))
        
        final_df = merged[all_cols]
        
        results = []
        for _, row in final_df.iterrows():
            player_record = {}
            for col in all_cols:
                val = row[col]
                if pd.isna(val) or val is None:
                    player_record[col.lower()] = None
                else:
                    # Convert numpy types to native Python types for JSON compatibility
                    if isinstance(val, (np.integer, np.int64)):
                        player_record[col.lower()] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        player_record[col.lower()] = float(val)
                    else:
                        player_record[col.lower()] = val
            results.append(player_record)
            
        player_stats_cache[cache_key] = (results, now)
        return results
        
    except Exception as e:
        logger.error(f"Error in get_player_stats API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# 12 New Database-Backed Endpoints (Replaces BBRef scraping/JSON dependency)
# ---------------------------------------------------------------------------

def get_db_conn():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'TeamData.sqlite')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Wait for a writer instead of failing instantly with "database is locked".
    # The lazy caches (passing, draft, awards, coaches) write from request
    # handlers, so a backfill script running alongside the API is a normal
    # situation rather than an exceptional one.
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn

@app.get("/api/matchups/{game_id}/power-ratings")
def get_matchup_power_ratings(game_id: str):
    """
    Project spreads, margins, and win probabilities based on Simple Rating System (SRS)
    from team_season_advanced table.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # 1. Fetch game details from box_scores
        cursor.execute("SELECT home_team_id, away_team_id, season, game_date FROM box_scores WHERE game_id = ?", (game_id,))
        game_row = cursor.fetchone()
        
        if not game_row:
            # If not in box_scores, try to fetch from team_game_advanced
            cursor.execute("SELECT team_id, opp_team_id, season, game_date FROM team_game_advanced WHERE game_id = ? LIMIT 1", (game_id,))
            game_row = cursor.fetchone()
            if not game_row:
                raise HTTPException(status_code=404, detail=f"Game {game_id} not found in database.")
            home_team_id = game_row["team_id"]
            away_team_id = game_row["opp_team_id"]
            season = game_row["season"]
            game_date = game_row["game_date"]
        else:
            home_team_id = game_row["home_team_id"]
            away_team_id = game_row["away_team_id"]
            season = game_row["season"]
            game_date = game_row["game_date"]

        # 2. Fetch team details
        cursor.execute("SELECT team_id, abbreviation, full_name FROM team_metadata WHERE team_id IN (?, ?)", (home_team_id, away_team_id))
        teams = {row["team_id"]: dict(row) for row in cursor.fetchall()}
        
        home_meta = teams.get(home_team_id, {"abbreviation": "HOME", "full_name": "Home Team"})
        away_meta = teams.get(away_team_id, {"abbreviation": "AWAY", "full_name": "Away Team"})

        # 3. Fetch SRS and Net Rating from team_season_advanced
        cursor.execute(
            "SELECT team_id, srs, sos, net_rating, computed_at FROM team_season_advanced WHERE team_id IN (?, ?) AND season = ? AND season_type = 'Regular Season'",
            (home_team_id, away_team_id, season)
        )
        season_stats = {row["team_id"]: dict(row) for row in cursor.fetchall()}
        
        home_srs = season_stats.get(home_team_id, {}).get("srs", 0.0) or 0.0
        away_srs = season_stats.get(away_team_id, {}).get("srs", 0.0) or 0.0
        
        home_sos = season_stats.get(home_team_id, {}).get("sos", 0.0) or 0.0
        away_sos = season_stats.get(away_team_id, {}).get("sos", 0.0) or 0.0

        home_net = season_stats.get(home_team_id, {}).get("net_rating", 0.0) or 0.0
        away_net = season_stats.get(away_team_id, {}).get("net_rating", 0.0) or 0.0
        computed_at = season_stats.get(home_team_id, {}).get("computed_at", "N/A")

        # Calculate projected margins and spreads
        # Home court advantage is assumed to be 2.5
        home_projected_margin = home_srs - away_srs + 2.5
        projected_spread = -home_projected_margin
        
        # Logistic win probability formula
        import math
        home_win_prob = 1.0 / (1.0 + math.exp(-0.15 * home_projected_margin))
        away_win_prob = 1.0 - home_win_prob

        srs_delta = home_srs - away_srs
        net_delta = home_net - away_net

        # Plain-English read
        if home_projected_margin > 0:
            winner_team = home_meta["full_name"]
            margin_val = home_projected_margin
            spread_str = f"-{margin_val:.1f}"
        else:
            winner_team = away_meta["full_name"]
            margin_val = abs(home_projected_margin)
            spread_str = f"+{margin_val:.1f}"

        read_text = (
            f"Based on Simple Rating System (SRS), the {winner_team} are projected to win by {margin_val:.1f} points "
            f"(projected spread: {spread_str}). The home team has a Net Rating edge of {net_delta:+.1f}."
        )

        return {
            "game_id": game_id,
            "season": season,
            "as_of": game_date,
            "home": {
                "team_id": home_team_id,
                "abbreviation": home_meta["abbreviation"],
                "full_name": home_meta["full_name"],
                "srs": round(home_srs, 2),
                "sos": round(home_sos, 2),
                "net_rating": round(home_net, 2),
                "projected_margin": round(home_projected_margin, 2),
                "win_probability": round(home_win_prob * 100, 1)
            },
            "away": {
                "team_id": away_team_id,
                "abbreviation": away_meta["abbreviation"],
                "full_name": away_meta["full_name"],
                "srs": round(away_srs, 2),
                "sos": round(away_sos, 2),
                "net_rating": round(away_net, 2),
                "projected_margin": round(-home_projected_margin, 2),
                "win_probability": round(away_win_prob * 100, 1)
            },
            "projected_spread": round(projected_spread, 1),
            "srs_delta": round(srs_delta, 2),
            "net_rating_delta": round(net_delta, 2),
            "summary": read_text,
            "citation": "Dean Oliver formulas computed from official NBA Stats API box scores",
            "computed_at": computed_at
        }
    except Exception as e:
        logger.error(f"Error fetching matchup power ratings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Player impact & availability (Estimated impact, box-score based) ---
# Honesty label: these are OUR box-score estimates (Neil Paine's Estimated
# RAPTOR regression weights, MIT-licensed), NOT official RAPTOR or DARKO.
# Full methodology + limitations: src/Utils/player_impact.py docstring.

@app.get("/api/impact-ratings")
def get_impact_ratings_endpoint(season: str = CURRENT_SEASON, min_gp: int = 20):
    """Ranked player-impact ratings (points per 100 possessions above a
    league-average player) for one season, best first."""
    try:
        players = player_impact.get_impact_ratings(season, min_gp=min_gp)
    except Exception as e:
        logger.error(f"Error computing impact ratings for {season}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute impact ratings.")
    if not players:
        raise HTTPException(status_code=404, detail=f"No impact ratings available for season {season}.")
    return {
        "season": season,
        "min_gp": min_gp,
        "count": len(players),
        "methodology": player_impact.METHODOLOGY_LABEL,
        "players": players,
    }


@app.get("/api/players/{id}/impact")
def get_player_impact_endpoint(id: int):
    """One player's estimated impact by season (career series within our
    data window). impact_rank is among gp>=20 qualifiers, or null."""
    try:
        seasons = player_impact.get_player_impact(id)
    except Exception as e:
        logger.error(f"Error computing impact series for player {id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute player impact.")
    name = seasons[0]["name"] if seasons else None
    if name is None:
        conn = get_db_conn()
        if conn:
            try:
                row = conn.execute("SELECT full_name FROM players WHERE player_id = ?", (id,)).fetchone()
                if row:
                    name = row["full_name"]
            finally:
                conn.close()
        if name is None:
            raise HTTPException(status_code=404, detail=f"Player {id} not found.")
    return {
        "player_id": id,
        "name": name,
        "methodology": player_impact.METHODOLOGY_LABEL,
        "seasons": seasons,
    }


@app.get("/api/matchups/availability")
def get_matchup_availability_endpoint(home: str, away: str, season: str = CURRENT_SEASON):
    """Current OUT/DOUBTFUL players (ESPN injury wire) for both teams with
    estimated impact contributions and each side's net-rating delta per
    100 possessions. Questionable/Day-To-Day are deliberately excluded
    (game-time decisions). Empty lists are the correct offseason answer."""
    home_abbr = espn_injuries.resolve_team_abbr(home)
    away_abbr = espn_injuries.resolve_team_abbr(away)
    if not home_abbr or not away_abbr:
        unknown = home if not home_abbr else away
        raise HTTPException(status_code=404, detail=f"Unknown team: {unknown}")
    try:
        return availability_adjust.matchup_availability(home_abbr, away_abbr, season)
    except Exception as e:
        logger.error(f"Error computing availability for {home_abbr} vs {away_abbr}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute matchup availability.")


@app.get("/api/players/browse")
def browse_players(
    letter: Optional[str] = None,
    team: Optional[str] = None,
    position: Optional[str] = None,
    college: Optional[str] = None,
    era: Optional[str] = None,
    active: Optional[bool] = None,
    sort: str = "name",
    limit: int = 120,
    offset: int = 0,
):
    """
    Browse the directory instead of searching it.

    Search only helps a reader who already knows a name. With 5,210 players on
    record that leaves most of the league unreachable, so this exposes the same
    axes nba.com's roster page does - initial, team, position, college - plus era,
    which theirs does not need because theirs is current players only.

    Filter option lists are computed over the whole directory rather than the
    current result, so narrowing by team does not empty the position menu.
    """
    conn = get_db_conn()
    try:
        where, params = [], []
        if letter:
            where.append("UPPER(last_name) LIKE ?")
            params.append(f"{letter[:1].upper()}%")
        if team:
            where.append("UPPER(last_team) = ?")
            params.append(team.upper())
        if position:
            # playerindex stores positions as single letters and hyphenated pairs:
            # C, C-F, F, F-C, F-G, G, G-F. It does NOT spell them out - the
            # draft-board path uses commonplayerinfo, which does, and matching the
            # spelled-out word here returned zero active centres.
            #
            # Matching on the letter catches the combinations, which is what a
            # reader picking "Center" wants: a C-F is a centre who also plays
            # forward, not a different position. The alphabet is only C/F/G so a
            # substring match cannot collide.
            token = {"guard": "G", "forward": "F", "center": "C"}.get(
                position.strip().lower(), position.strip().upper()
            )
            where.append("position LIKE ?")
            params.append(f"%{token}%")
        if college:
            where.append("college = ?")
            params.append(college)
        if active is not None:
            where.append("is_active = ?")
            params.append(1 if active else 0)
        if era:
            # "1990s" style buckets, matched on any overlap with the decade rather
            # than on debut year - a player who spanned 1998-2010 belongs to both.
            try:
                decade = int(str(era)[:4])
                where.append("COALESCE(to_year, 0) >= ? AND COALESCE(from_year, 9999) <= ?")
                params.extend([decade, decade + 9])
            except ValueError:
                pass

        clause = (" WHERE " + " AND ".join(where)) if where else ""
        order = {
            "name": "last_name COLLATE NOCASE, first_name COLLATE NOCASE",
            "recent": "COALESCE(to_year, 0) DESC, last_name COLLATE NOCASE",
            "career": "(COALESCE(to_year,0) - COALESCE(from_year,0)) DESC, last_name COLLATE NOCASE",
            "debut": "COALESCE(from_year, 9999), last_name COLLATE NOCASE",
        }.get(sort, "last_name COLLATE NOCASE, first_name COLLATE NOCASE")

        total = conn.execute(
            f"SELECT COUNT(*) FROM players{clause}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT player_id, full_name, first_name, last_name, is_active,
                   from_year, to_year, position, height, weight, college,
                   country, jersey, last_team, last_team_id
            FROM players{clause}
            ORDER BY {order}
            LIMIT ? OFFSET ?
            """,
            params + [max(1, min(limit, 300)), max(0, offset)],
        ).fetchall()

        # Option lists over the whole directory, and the per-initial counts the
        # A-Z rail needs to grey out letters nobody is filed under.
        letters = {
            r["ltr"]: r["n"] for r in conn.execute(
                "SELECT UPPER(SUBSTR(last_name,1,1)) AS ltr, COUNT(*) AS n FROM players "
                "WHERE last_name IS NOT NULL AND last_name != '' GROUP BY ltr ORDER BY ltr"
            )
        }
        teams = [
            r["last_team"] for r in conn.execute(
                "SELECT DISTINCT last_team FROM players WHERE last_team IS NOT NULL "
                "AND last_team != '' ORDER BY last_team"
            )
        ]
        colleges = [
            {"name": r["college"], "players": r["n"]}
            for r in conn.execute(
                "SELECT college, COUNT(*) AS n FROM players WHERE college IS NOT NULL "
                "AND college != '' GROUP BY college ORDER BY n DESC, college LIMIT 80"
            )
        ]
        bios = conn.execute(
            "SELECT COUNT(*) FROM players WHERE position IS NOT NULL AND position != ''"
        ).fetchone()[0]
        directory_total = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "directory_total": directory_total,
            # How much of the directory has a bio yet, so the page can be honest
            "with_bio": bios,
            "filters": {
                "letter": letter, "team": team, "position": position,
                "college": college, "era": era, "active": active, "sort": sort,
            },
            "options": {
                "letters": letters,
                "teams": teams,
                "positions": ["Guard", "Forward", "Center"],
                # The raw values, so a caller can see that C-F exists and that
                # picking Center includes it.
                "position_codes": [
                    r["position"] for r in conn.execute(
                        "SELECT DISTINCT position FROM players WHERE position IS NOT NULL "
                        "AND position != '' ORDER BY position"
                    )
                ],
                "colleges": colleges,
                "eras": ["2020s", "2010s", "2000s", "1990s", "1980s", "1970s", "1960s", "1950s"],
            },
            "players": [dict(r) for r in rows],
        }
    except Exception as e:
        logger.error(f"Error browsing players: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not browse the directory.")
    finally:
        conn.close()


# NOTE: must be registered before /api/players/{id} or FastAPI matches
# the literal path segment "search" as an id.
@app.get("/api/players/search")
def search_players(q: str):
    """
    Search players by name.
    """
    if not q or len(q) < 2:
        return []

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        # The directory covers all of league history now, so ordering decides
        # whether this is useful. Surname matches outrank forename matches: a
        # search for "jordan" means Michael and DeAndre before Jordan Clarkson,
        # and "james" means LeBron before James Harden. Sorting active-first
        # alone buried every retired great under whoever is on a roster today.
        #
        # Within a tier, current players come first and then the most recent,
        # because a bare surname is ambiguous and recency is the best tiebreak
        # available without a prominence measure we do not have.
        cursor.execute(
            """
            SELECT player_id, full_name, first_name, last_name, is_active,
                   from_year, to_year
            FROM players
            WHERE full_name LIKE ?
            ORDER BY
                CASE
                    WHEN lower(full_name) = lower(?) THEN 0
                    WHEN lower(last_name) = lower(?) THEN 1
                    WHEN lower(last_name) LIKE lower(?) THEN 2
                    WHEN lower(full_name) LIKE lower(?) THEN 3
                    ELSE 4
                END,
                is_active DESC,
                -- Career length as a prominence proxy, which is the best signal
                -- already on the row. Alphabetical put Bronny ahead of LeBron and
                -- Seth ahead of Stephen; twenty-three seasons against two is a
                -- better guess at who was meant than the letter B against L.
                (COALESCE(to_year, 0) - COALESCE(from_year, 0)) DESC,
                COALESCE(to_year, 0) DESC,
                full_name
            LIMIT 25
            """,
            (f"%{q}%", q, q, f"{q}%", f"{q}%")
        )
        rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append(dict(r))

        return results
    except Exception as e:
        logger.error(f"Error searching players: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/players/{id}")
@limiter.limit(RATE_LIMIT_UPSTREAM)
def get_player_by_id(request: Request, id: int, season: str = CURRENT_SEASON):
    """
    Fetch player biography, current season totals, and advanced statistics.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        # 1. Fetch player base details
        cursor.execute("SELECT * FROM players WHERE player_id = ?", (id,))
        player_row = cursor.fetchone()
        if not player_row:
            raise HTTPException(status_code=404, detail=f"Player ID {id} not found.")
        player_info = dict(player_row)

        # 1b. Cache-first player biography lookup from player_bio table
        cursor.execute("SELECT * FROM player_bio WHERE player_id = ?", (id,))
        bio_row = cursor.fetchone()
        
        bio_data = {}
        if bio_row:
            bio_data = dict(bio_row)
            
        # Check if cache miss or fetched_at is NULL
        if not bio_data or not bio_data.get("fetched_at"):
            # TODO: Potential thundering herd at scale — 100 concurrent uncached profile hits = 100 parallel CommonPlayerInfo calls queueing under the rate limiter. Acceptable for launch; add a per-player in-flight lock before scaling.
            if not commonplayerinfo:
                logger.warning("nba_api commonplayerinfo not imported, skipping live bio fetch")
            else:
                try:
                    logger.info(f"Cache miss: Fetching player bio from NBA stats API for player ID {id}")
                    # Fetch from CommonPlayerInfo
                    info = commonplayerinfo.CommonPlayerInfo(player_id=id)
                    df = info.get_data_frames()[0]
                    if not df.empty:
                        row_data = df.iloc[0].to_dict()
                        
                        # Prepare fields
                        jersey = row_data.get("JERSEY")
                        position = row_data.get("POSITION")
                        height = row_data.get("HEIGHT")
                        weight = row_data.get("WEIGHT")
                        birth_date = row_data.get("BIRTHDATE")
                        if birth_date and "T" in birth_date:
                            birth_date = birth_date.split("T")[0]
                        country = row_data.get("COUNTRY")
                        school = row_data.get("SCHOOL")
                        
                        draft_year = row_data.get("DRAFT_YEAR")
                        try:
                            draft_year = int(draft_year) if str(draft_year).isdigit() else None
                        except:
                            draft_year = None
                            
                        draft_round = row_data.get("DRAFT_ROUND")
                        try:
                            draft_round = int(draft_round) if str(draft_round).isdigit() else None
                        except:
                            draft_round = None
                            
                        draft_number = row_data.get("DRAFT_NUMBER")
                        try:
                            draft_number = int(draft_number) if str(draft_number).isdigit() else None
                        except:
                            draft_number = None
                            
                        exp = row_data.get("SEASON_EXP")
                        try:
                            years_experience = int(exp) if str(exp).isdigit() else 0
                        except:
                            years_experience = 0
                            
                        team_id = row_data.get("TEAM_ID")
                        try:
                            team_id = int(team_id) if str(team_id).isdigit() else None
                        except:
                            team_id = None
                            
                        team_abbr = row_data.get("TEAM_ABBREVIATION")
                        team_name = row_data.get("TEAM_NAME")
                        is_active = 1 if row_data.get("ROSTERSTATUS") == "Active" else 0
                        fetched_at = datetime.utcnow().isoformat()
                        
                        # Insert/Update player_bio table (upsert)
                        cursor.execute(
                            """
                            INSERT INTO player_bio (
                                player_id, full_name, first_name, last_name, team_id, team_abbr,
                                jersey, position, height, weight, birth_date, country, school,
                                draft_year, draft_round, draft_number, years_experience, is_active, fetched_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(player_id) DO UPDATE SET
                                full_name=excluded.full_name,
                                first_name=excluded.first_name,
                                last_name=excluded.last_name,
                                team_id=excluded.team_id,
                                team_abbr=excluded.team_abbr,
                                jersey=excluded.jersey,
                                position=excluded.position,
                                height=excluded.height,
                                weight=excluded.weight,
                                birth_date=excluded.birth_date,
                                country=excluded.country,
                                school=excluded.school,
                                draft_year=excluded.draft_year,
                                draft_round=excluded.draft_round,
                                draft_number=excluded.draft_number,
                                years_experience=excluded.years_experience,
                                is_active=excluded.is_active,
                                fetched_at=excluded.fetched_at
                            """,
                            (
                                id, player_info["full_name"], player_info["first_name"], player_info["last_name"],
                                team_id, team_abbr, jersey, position, height, weight, birth_date, country, school,
                                draft_year, draft_round, draft_number, years_experience, is_active, fetched_at
                            )
                        )
                        conn.commit()
                        
                        # Re-read from db
                        cursor.execute("SELECT * FROM player_bio WHERE player_id = ?", (id,))
                        updated_row = cursor.fetchone()
                        if updated_row:
                            bio_data = dict(updated_row)
                except Exception as e:
                    logger.error(f"Error fetching CommonPlayerInfo for player ID {id}: {e}", exc_info=True)
        
        # 2. Fetch current season totals
        cursor.execute(
            """
            SELECT t.*,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = t.team_id) as team_abbr
            FROM player_season_totals t
            WHERE t.player_id = ? AND t.season = ? AND t.season_type = 'Regular Season'
            LIMIT 1
            """,
            (id, season)
        )
        totals_row = cursor.fetchone()
        totals = dict(totals_row) if totals_row else {}
        
        # 3. Fetch current season advanced stats
        cursor.execute(
            """
            SELECT * FROM player_season_advanced
            WHERE player_id = ? AND season = ? AND season_type = 'Regular Season'
            LIMIT 1
            """,
            (id, season)
        )
        adv_row = cursor.fetchone()
        advanced = dict(adv_row) if adv_row else {}
        
        # 4. Construct response
        team_abbr = totals.get("team_abbr") or bio_data.get("team_abbr") or "N/A"
        
        if not totals:
            cursor.execute(
                """
                SELECT COUNT(*) as games, SUM(pts) as pts, SUM(ast) as ast, SUM(reb) as reb, SUM(min) as min
                FROM player_game_log
                WHERE player_id = ? AND game_id IN (SELECT game_id FROM box_scores WHERE season = ?)
                """,
                (id, season)
            )
            agg_row = cursor.fetchone()
            if agg_row and agg_row["games"] > 0:
                totals = {
                    "games": agg_row["games"],
                    "pts": agg_row["pts"],
                    "ast": agg_row["ast"],
                    "reb": agg_row["reb"],
                    "min": agg_row["min"],
                    "pts_per_game": round(agg_row["pts"] / agg_row["games"], 1),
                    "ast_per_game": round(agg_row["ast"] / agg_row["games"], 1),
                    "reb_per_game": round(agg_row["reb"] / agg_row["games"], 1),
                }
                
        # Format properties
        bio_position = bio_data.get("position") or "Forward/Guard"
        bio_height = bio_data.get("height") or "N/A"
        bio_weight = bio_data.get("weight") or "N/A"
        bio_height_weight = f"{bio_height}, {bio_weight}lb" if (bio_height != "N/A" and bio_weight != "N/A") else f"{bio_height}"
        if bio_weight != "N/A" and "lb" not in str(bio_weight).lower() and str(bio_weight).isdigit():
            bio_height_weight = f"{bio_height}, {bio_weight}lb"
            
        bio_born = bio_data.get("birth_date") or "N/A"
        bio_college = bio_data.get("school") or "N/A"
        
        bio_draft_year = bio_data.get("draft_year")
        bio_draft_round = bio_data.get("draft_round")
        bio_draft_number = bio_data.get("draft_number")
        
        exp_val = bio_data.get("years_experience")
        if exp_val is not None:
            bio_exp = f"{exp_val} Years" if exp_val > 0 else "Rookie"
        else:
            bio_exp = "Active" if player_info["is_active"] else "Inactive"
            
        response = {
            "id": id,
            "player_id": id,
            "full_name": player_info["full_name"],
            "first_name": player_info["first_name"],
            "last_name": player_info["last_name"],
            "is_active": player_info["is_active"],
            "bio": {
                "fullName": player_info["full_name"],
                "position": bio_position,
                "heightWeight": bio_height_weight,
                "team": team_abbr,
                "born": bio_born,
                "college": bio_college,
                "experience": bio_exp,
                "jersey": bio_data.get("jersey") or "N/A",
                "country": bio_data.get("country") or "N/A",
                "draft_year": bio_draft_year,
                "draft_round": bio_draft_round,
                "draft_number": bio_draft_number,
                "active": bool(player_info["is_active"]),
                "instagram": player_info["full_name"].lower().replace(" ", ""),
                "nicknames": "None"
            },
            "totals": totals,
            "advanced": advanced
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching player by ID {id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/players/by-slug/{slug}")
def get_player_by_slug(request: Request, slug: str, season: str = CURRENT_SEASON):
    """
    Resolve a player slug of format 'first-last-id' to player_id,
    then return their biography, current season totals, and advanced stats.
    """
    parts = slug.split("-")
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Invalid slug format: {slug}. Must end with player ID.")
    player_id_str = parts[-1]
    if not player_id_str.isdigit():
        raise HTTPException(status_code=400, detail=f"Invalid slug format: {slug}. Last part must be a numeric player ID.")
    player_id = int(player_id_str)
    
    # get_player_by_id gained a leading `request` parameter when the upstream
    # rate limiter was added; forwarding it here keeps the internal call valid
    # (calling it positionally without request put player_id INTO request and
    # 500'd every player page).
    return get_player_by_id(request, player_id, season=season)

# --- Heat calendar ------------------------------------------------------------
# Season and season type are read from the game id rather than the date. The id
# encodes both - prefix 002 is regular season and 004 is playoffs, characters four
# and five are the season's start year - which matters because a season straddles
# the new year, so deriving it from a date means a rule about October, and the
# playoffs would be indistinguishable from late regular-season games either way.
#
# Game Score is Hollinger's, kept exactly as he defined it:
#   PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB
#        + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
# It is a one-number summary of a single game, roughly on a points-like scale, and
# it is the colour on the calendar because points alone call a 30-point night on
# 30 shots a good game.
GAME_SCORE_SQL = (
    "(g.pts + 0.4 * g.fgm - 0.7 * g.fga - 0.4 * (g.fta - g.ftm) "
    "+ 0.7 * g.oreb + 0.3 * g.dreb + g.stl + 0.7 * g.ast + 0.7 * g.blk "
    "- 0.4 * g.pf - g.tov)"
)


@app.get("/api/players/{id}/heat-calendar")
def get_player_heat_calendar(id: int, season: Optional[str] = None):
    """
    Every game a player played in a season, dated, for a calendar heat map.

    Returns the seasons this player actually has games for, so a caller can offer
    only those rather than a season list the player was not in - the game-log
    archive starts in 2022-23, and a calendar rendered empty for a retired player
    would look broken rather than out of range.
    """
    conn = get_db_conn()
    try:
        exists = conn.execute(
            "SELECT full_name FROM players WHERE player_id = ?", (id,)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"Player ID {id} not found.")

        # Seasons this player has games in, newest first.
        season_rows = conn.execute(
            """
            SELECT DISTINCT CAST(SUBSTR(game_id, 4, 2) AS INTEGER) AS yy
            FROM player_game_log WHERE player_id = ? AND LENGTH(game_id) >= 5
            ORDER BY yy DESC
            """,
            (id,),
        ).fetchall()
        seasons = [f"20{r['yy']:02d}-{(r['yy'] + 1) % 100:02d}" for r in season_rows]

        if not seasons:
            return {
                "player_id": id, "player": exists["full_name"], "season": None,
                "seasons": [], "games": 0, "entries": [],
                "note": "No game logs on record for this player.",
            }

        target = season if season in seasons else seasons[0]
        yy = int(target[2:4])

        rows = conn.execute(
            f"""
            SELECT g.game_id, DATE(g.game_date) AS date, g.min, g.pts, g.reb, g.ast,
                   g.stl, g.blk, g.tov, g.fgm, g.fga, g.fg3m, g.ftm, g.fta,
                   g.plus_minus, g.starter,
                   CASE SUBSTR(g.game_id, 1, 3)
                        WHEN '004' THEN 'Playoffs'
                        WHEN '005' THEN 'Play-In'
                        WHEN '001' THEN 'Preseason'
                        ELSE 'Regular Season' END AS season_type,
                   {GAME_SCORE_SQL} AS game_score
            FROM player_game_log g
            WHERE g.player_id = ? AND CAST(SUBSTR(g.game_id, 4, 2) AS INTEGER) = ?
            ORDER BY DATE(g.game_date)
            """,
            (id, yy),
        ).fetchall()

        entries = [dict(r) for r in rows]
        for e in entries:
            e["game_score"] = round(e["game_score"], 1) if e["game_score"] is not None else None

        scores = [e["game_score"] for e in entries if e["game_score"] is not None]
        pts = [e["pts"] for e in entries if e["pts"] is not None]
        scale = {}
        if scores:
            ordered = sorted(scores)
            n = len(ordered)
            # The colour scale is built from THIS player's season rather than a
            # league-wide one, so a role player's good night is visible instead of
            # every cell being cold next to a star's.
            scale = {
                "min": round(ordered[0], 1),
                "p25": round(ordered[int(0.25 * n)], 1),
                "median": round(ordered[n // 2], 1),
                "p75": round(ordered[int(0.75 * n)], 1),
                "max": round(ordered[-1], 1),
            }

        best = max(entries, key=lambda e: e["game_score"] or -99) if entries else None
        return {
            "player_id": id,
            "player": exists["full_name"],
            "season": target,
            "seasons": seasons,
            "games": len(entries),
            "scale": scale,
            "totals": {
                "pts": sum(pts) if pts else 0,
                "mean_game_score": round(sum(scores) / len(scores), 1) if scores else None,
            },
            "best_game": {
                "date": best["date"], "game_id": best["game_id"],
                "game_score": best["game_score"], "pts": best["pts"],
                "reb": best["reb"], "ast": best["ast"],
            } if best else None,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in heat calendar for {id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not build the heat calendar.")
    finally:
        conn.close()


@app.get("/api/players/{id}/game-log")
def get_player_game_log(id: int, season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    Fetch a player's individual game log from player_game_log table.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # Fetch logs
        cursor.execute(
            """
            SELECT pgl.*, bs.game_date, bs.season, bs.season_type,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = pgl.team_id) as team_abbr,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = (CASE WHEN pgl.team_id = bs.home_team_id THEN bs.away_team_id ELSE bs.home_team_id END)) as opp_abbr,
                   (CASE WHEN pgl.team_id = bs.home_team_id THEN 1 ELSE 0 END) as is_home
            FROM player_game_log pgl
            JOIN box_scores bs ON pgl.game_id = bs.game_id
            WHERE pgl.player_id = ? AND bs.season = ? AND bs.season_type = ?
            ORDER BY bs.game_date DESC
            """,
            (id, season, season_type)
        )
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            rec = dict(r)
            results.append(rec)
            
        return results
    except Exception as e:
        logger.error(f"Error fetching player game log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/players/{id}/splits")
def get_player_splits_api(id: int, season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    Fetch player splits (Location, Wins/Losses, Month) from player_splits table.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM player_splits WHERE player_id = ? AND season = ? AND season_type = ?",
            (id, season, season_type)
        )
        rows = cursor.fetchall()
        
        splits = {
            "Location": [],
            "Wins/Losses": [],
            "Month": []
        }
        for r in rows:
            rec = dict(r)
            stype = rec.get("split_type")
            if stype in splits:
                splits[stype].append(rec)
                
        return splits
    except Exception as e:
        logger.error(f"Error fetching player splits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/players/{id}/career")
def get_player_career(id: int):
    """
    Fetch player career aggregates (season-by-season totals and advanced statistics).
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.*, a.ts_pct, a.usg_pct, a.off_rating, a.def_rating, a.net_rating, a.pace,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = t.team_id) as team_abbr
            FROM player_season_totals t
            LEFT JOIN player_season_advanced a ON t.player_id = a.player_id AND t.season = a.season AND t.season_type = a.season_type AND t.team_id = a.team_id
            WHERE t.player_id = ?
            ORDER BY t.season DESC
            """,
            (id,)
        )
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            rec = dict(r)
            results.append(rec)
            
        return results
    except Exception as e:
        logger.error(f"Error fetching player career stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/teams/advanced")
def get_all_teams_advanced(season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    Fetch advanced stats for all teams for a season.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.*, m.full_name, m.abbreviation, m.conference, m.division
            FROM team_season_advanced s
            JOIN team_metadata m ON s.team_id = m.team_id
            WHERE s.season = ? AND s.season_type = ?
            ORDER BY s.net_rating DESC
            """,
            (season, season_type)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error fetching all teams advanced stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/teams/{abbr}/advanced")
def get_team_advanced(abbr: str, season: Optional[str] = None):
    """
    Fetch a team's advanced stats for a season (or all seasons).
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # Get team_id
        cursor.execute("SELECT team_id, full_name, abbreviation FROM team_metadata WHERE abbreviation = ?", (abbr.upper(),))
        team_row = cursor.fetchone()
        if not team_row:
            raise HTTPException(status_code=404, detail=f"Team {abbr} not found.")
            
        team_id = team_row["team_id"]
        
        if season:
            cursor.execute(
                "SELECT * FROM team_season_advanced WHERE team_id = ? AND season = ?",
                (team_id, season)
            )
        else:
            cursor.execute(
                "SELECT * FROM team_season_advanced WHERE team_id = ? ORDER BY season DESC",
                (team_id,)
            )
            
        rows = cursor.fetchall()
        results = []
        for r in rows:
            rec = dict(r)
            rec["team_name"] = team_row["full_name"]
            rec["abbreviation"] = team_row["abbreviation"]
            results.append(rec)
            
        return results
    except Exception as e:
        logger.error(f"Error fetching team advanced stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/teams/{abbr}/roster")
def get_team_roster(abbr: str, season: str = CURRENT_SEASON):
    """
    Fetch the roster for a team including season averages (GP, PTS, REB, AST) from player_season_totals.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # Query players that have season totals for this team and season
        cursor.execute(
            """
            SELECT p.player_id, p.full_name, p.first_name, p.last_name,
                   t.gp, t.min, t.pts, t.reb, t.ast,
                   (SELECT jersey FROM player_bio WHERE player_id = p.player_id) as jersey,
                   (SELECT position FROM player_bio WHERE player_id = p.player_id) as position
            FROM players p
            JOIN player_season_totals t ON p.player_id = t.player_id
            JOIN team_metadata m ON t.team_id = m.team_id
            WHERE m.abbreviation = ? AND t.season = ? AND t.season_type = 'Regular Season'
            ORDER BY t.pts DESC
            """,
            (abbr.upper(), season)
        )
        rows = cursor.fetchall()
        if rows:
            return [dict(r) for r in rows]
            
        # Fallback to player_game_log if season totals aren't computed yet
        cursor.execute(
            """
            SELECT DISTINCT p.player_id, p.full_name, p.first_name, p.last_name,
                            (SELECT jersey FROM player_bio WHERE player_id = p.player_id) as jersey,
                            (SELECT position FROM player_bio WHERE player_id = p.player_id) as position
            FROM players p
            JOIN player_game_log pgl ON p.player_id = pgl.player_id
            JOIN team_metadata m ON pgl.team_id = m.team_id
            WHERE m.abbreviation = ?
            """,
            (abbr.upper(),)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error fetching team roster: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/teams/{abbr}/games")
def get_team_games(abbr: str, season: str = CURRENT_SEASON):
    """
    Fetch all games played by a team in a season, including score, outcome, and location.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        # Resolve team_id
        cursor.execute("SELECT team_id FROM team_metadata WHERE abbreviation = ?", (abbr.upper(),))
        t_row = cursor.fetchone()
        if not t_row:
            raise HTTPException(status_code=404, detail=f"Team abbreviation {abbr} not found.")
        team_id = t_row["team_id"]
        
        # Query games from team_game_advanced joined with box_scores to get home/away info
        cursor.execute(
            """
            SELECT tga.game_id, tga.game_date, tga.pts, tga.opp_pts, tga.season,
                   m.abbreviation as opp_abbr, m.full_name as opp_name,
                   (CASE WHEN bs.home_team_id = ? THEN 1 ELSE 0 END) as is_home
            FROM team_game_advanced tga
            JOIN team_metadata m ON tga.opp_team_id = m.team_id
            JOIN box_scores bs ON tga.game_id = bs.game_id
            WHERE tga.team_id = ? AND tga.season = ?
            ORDER BY tga.game_date DESC
            """,
            (team_id, team_id, season)
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            rec = dict(r)
            # Determine outcome
            rec["wl"] = "W" if rec["pts"] > rec["opp_pts"] else "L"
            results.append(rec)
        return results
    except Exception as e:
        logger.error(f"Error fetching team games for {abbr}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Scorigami: every final score that has and has not happened ---
#
# Built from `game_results`, which covers every NBA game since 1946-47, because
# "this scoreline has never happened" is only honest against the full history.
#
# One real exclusion: a game where BOTH teams scored zero was never played
# (Celtics-Pacers, 2013-04-16, cancelled after the Boston Marathon bombing).
# The 19-18 from 1950 is NOT excluded - that is the genuine lowest-scoring game
# in NBA history and belongs on the board.

_SCORIGAMI_CACHE: Optional[Dict[str, Any]] = None
_SCORIGAMI_CACHE_ROWS: int = -1

# No score floor. An earlier cut at 55 looked tidier but silently dropped 130
# real games - most of the pre-shot-clock era, and a 96-54 from the 1998 Finals.
# Those are genuine scorelines and a board that hides them is lying about what
# has happened. The sparse bottom-left is the shot clock's arrival, which is
# worth seeing.


@app.get("/api/stats/scorigami")
def get_scorigami():
    """
    Every final scoreline in NBA history, with how often it has happened and
    when it first did. Cells absent from `cells` have never happened.
    """
    global _SCORIGAMI_CACHE, _SCORIGAMI_CACHE_ROWS
    conn = get_db_conn()
    try:
        rows_total = conn.execute("SELECT COUNT(*) FROM game_results").fetchone()[0]
        if not rows_total:
            raise HTTPException(
                status_code=503,
                detail="Final scores have not been backfilled yet. Run backfill_results.py.",
            )
        if _SCORIGAMI_CACHE is not None and _SCORIGAMI_CACHE_ROWS == rows_total:
            return _SCORIGAMI_CACHE

        games: Dict[str, Dict[str, Any]] = {}
        for gid, pts, date, season in conn.execute(
            "SELECT game_id, pts, game_date, season FROM game_results ORDER BY game_date"
        ):
            g = games.setdefault(gid, {"pts": [], "date": date, "season": season})
            g["pts"].append(pts)

        cells: Dict[tuple, Dict[str, Any]] = {}
        counted = 0
        excluded: List[Dict[str, Any]] = []

        for gid, g in games.items():
            if len(g["pts"]) != 2:
                continue
            w, l = max(g["pts"]), min(g["pts"])
            if w == 0 and l == 0:
                # Never played: Celtics-Pacers 2013-04-16, cancelled after the
                # Boston Marathon bombing. A 0-0 cell would be a lie.
                excluded.append({"game_id": gid, "date": g["date"], "reason": "game not played"})
                continue
            counted += 1
            key = (w, l)
            c = cells.get(key)
            if c is None:
                cells[key] = {
                    "count": 1,
                    "first_date": g["date"], "first_season": g["season"],
                    "last_date": g["date"], "last_season": g["season"],
                }
            else:
                c["count"] += 1
                # Rows arrive in date order, so the last write is the latest.
                c["last_date"], c["last_season"] = g["date"], g["season"]

        if not cells:
            raise HTTPException(status_code=503, detail="No usable final scores found.")

        hi = max(w for w, _ in cells)
        lo = min(l for _, l in cells)
        # Only winner >= loser is reachable, so that triangle is the real board.
        possible = sum(1 for w in range(lo, hi + 1) for l in range(lo, w + 1))

        ordered = sorted(cells.items(), key=lambda kv: -kv[1]["count"])
        top = ordered[0]
        once = [k for k, v in cells.items() if v["count"] == 1]
        newest = max(cells.items(), key=lambda kv: kv[1]["first_date"])
        highest = max(cells.keys(), key=lambda k: k[0] + k[1])

        _SCORIGAMI_CACHE = {
            "games": counted,
            "seasons": {
                "first": conn.execute("SELECT MIN(season) FROM game_results").fetchone()[0],
                "last": conn.execute("SELECT MAX(season) FROM game_results").fetchone()[0],
            },
            "range": {"lo": lo, "hi": hi},
            "happened": len(cells),
            "possible": possible,
            "never": possible - len(cells),
            # Compact: [winner, loser, count, first_date, last_date]. A dict per
            # cell would triple the payload for 3,000+ entries.
            "cells": [
                [w, l, v["count"], v["first_date"], v["last_date"]]
                for (w, l), v in cells.items()
            ],
            "records": {
                "most_common": {"winner": top[0][0], "loser": top[0][1], "count": top[1]["count"]},
                "once_only": len(once),
                "newest": {
                    "winner": newest[0][0], "loser": newest[0][1],
                    "date": newest[1]["first_date"], "season": newest[1]["first_season"],
                },
                "highest_total": {"winner": highest[0], "loser": highest[1],
                                  "total": highest[0] + highest[1]},
            },
            "lowest_total": {"winner": min(cells, key=lambda k: k[0] + k[1])[0],
                             "loser": min(cells, key=lambda k: k[0] + k[1])[1]},
            "excluded": excluded,
        }
        _SCORIGAMI_CACHE_ROWS = rows_total
        return _SCORIGAMI_CACHE
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building scorigami: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Win probability, counted rather than modelled ---
#
# There is no model here on purpose. For every 30 seconds of every archived
# game we record the score margin and who eventually won, then a curve point is
# simply "of the N times a home team led by this much with this long left, this
# fraction went on to win". It is checkable in a way a fitted model is not, and
# it matches how the comeback tables are built.
#
# Raw cells are too thin to use directly (25% hold fewer than 30 samples), so a
# lookup pools neighbours: margin +-2 points and time +-60 seconds. The pooled
# sample count travels with every point so the UI can show it.
#
# MEASURED, on games the table was not built from: Brier 0.163, and calibration
# buckets land within about 3 points of what they claim. The low-probability
# buckets run slightly optimistic. Home advantage falls out of the data rather
# than being inserted - a tied game at tip-off reads 55.7% for the home side.
#
# Known simplification: possession is ignored. That matters most inside the last
# minute, where having the ball is worth real probability, so the endpoint says
# so rather than pretending otherwise.

_WP_TABLE: Optional[Dict[tuple, List[int]]] = None
_WP_TABLE_ROWS: int = -1

WP_REGULATION = 2880      # seconds
WP_STEP = 30              # sampling resolution
WP_MARGIN_CAP = 30
WP_MIN_POOLED = 50        # below this a point is reported without a probability


def _build_wp_table(conn) -> Dict[tuple, List[int]]:
    finals: Dict[str, tuple] = {}
    for gid, sh, sa in conn.execute(
        "SELECT game_id, score_home, score_away FROM pbp_events "
        "WHERE score_home IS NOT NULL ORDER BY game_id, action_number"
    ):
        finals[gid] = (sh, sa)

    table: Dict[tuple, List[int]] = {}
    marks = list(range(0, WP_REGULATION + 1, WP_STEP))
    cur, last, idx = None, (0, 0), 0

    for gid, es, sh, sa in conn.execute(
        "SELECT game_id, elapsed_seconds, score_home, score_away FROM pbp_events "
        "WHERE elapsed_seconds IS NOT NULL ORDER BY game_id, action_number"
    ):
        if gid != cur:
            cur, last, idx = gid, (0, 0), 0
        if sh is not None and sa is not None:
            last = (sh, sa)
        fin = finals.get(gid)
        if not fin:
            continue
        while idx < len(marks) and es >= marks[idx]:
            secs_left = WP_REGULATION - marks[idx]
            idx += 1
            margin = max(-WP_MARGIN_CAP, min(WP_MARGIN_CAP, last[0] - last[1]))
            cell = table.setdefault((secs_left, margin), [0, 0])
            cell[0] += 1
            cell[1] += 1 if fin[0] > fin[1] else 0
    return table


def _wp_lookup(table: Dict[tuple, List[int]], secs_left: float, margin: int) -> tuple:
    """Pooled (probability, sample_count) for a game state. (None, n) when thin."""
    # Snap to the sampling grid, and treat overtime as "zero seconds left".
    sl = max(0, min(WP_REGULATION, int(round(secs_left / WP_STEP) * WP_STEP)))
    m = max(-WP_MARGIN_CAP, min(WP_MARGIN_CAP, margin))
    n = w = 0
    for dm in (-2, -1, 0, 1, 2):
        for dt in (-60, -30, 0, 30, 60):
            cell = table.get((sl + dt, max(-WP_MARGIN_CAP, min(WP_MARGIN_CAP, m + dm))))
            if cell:
                n += cell[0]
                w += cell[1]
    if n < WP_MIN_POOLED:
        return None, n
    return w / n, n


def _get_wp_table(conn) -> Dict[tuple, List[int]]:
    global _WP_TABLE, _WP_TABLE_ROWS
    rows = conn.execute("SELECT COUNT(*) FROM pbp_events").fetchone()[0]
    if not rows:
        raise HTTPException(
            status_code=503,
            detail="Play-by-play has not been backfilled yet. Run backfill_pbp.py.",
        )
    if _WP_TABLE is not None and _WP_TABLE_ROWS == rows:
        return _WP_TABLE
    _WP_TABLE = _build_wp_table(conn)
    _WP_TABLE_ROWS = rows
    return _WP_TABLE


@app.get("/api/games/{game_id}/win-probability")
def get_win_probability(game_id: str):
    """
    The home team's win probability through one game, plus the swings that
    decided it.

    Each point carries the sample count behind it. Points backed by too few
    comparable situations return a null probability rather than a guess.
    """
    conn = get_db_conn()
    try:
        table = _get_wp_table(conn)

        events = conn.execute(
            "SELECT elapsed_seconds, period, score_home, score_away, description, "
            "       action_type, player_name, team_tricode "
            "FROM pbp_events WHERE game_id = ? ORDER BY action_number",
            (game_id,),
        ).fetchall()
        if not events:
            raise HTTPException(status_code=404, detail=f"No play-by-play stored for game {game_id}.")

        meta = conn.execute(
            "SELECT bs.home_team_id, bs.away_team_id, bs.game_date, bs.season, bs.season_type, "
            "       h.abbreviation AS home_abbr, h.full_name AS home_name, "
            "       a.abbreviation AS away_abbr, a.full_name AS away_name "
            "FROM box_scores bs "
            "JOIN team_metadata h ON h.team_id = bs.home_team_id "
            "JOIN team_metadata a ON a.team_id = bs.away_team_id "
            "WHERE bs.game_id = ?",
            (game_id,),
        ).fetchone()

        series: List[Dict[str, Any]] = []
        last = (0, 0)
        prev_p: Optional[float] = None
        swings: List[Dict[str, Any]] = []

        for e in events:
            d = dict(e)
            if d["score_home"] is not None and d["score_away"] is not None:
                last = (d["score_home"], d["score_away"])
            es = d["elapsed_seconds"]
            if es is None:
                continue
            secs_left = WP_REGULATION - es
            margin = last[0] - last[1]
            p, n = _wp_lookup(table, secs_left, margin)

            point = {
                "t": round(es, 1),
                "period": d["period"],
                "home": last[0],
                "away": last[1],
                "margin": margin,
                "wp": round(p, 4) if p is not None else None,
                "n": n,
            }
            # Only scoring plays move the line, so annotate those.
            if p is not None and prev_p is not None:
                delta = p - prev_p
                if abs(delta) >= 0.06 and d["description"]:
                    swings.append({
                        "t": round(es, 1), "period": d["period"], "delta": round(delta, 4),
                        "wp": round(p, 4), "description": d["description"],
                        "player": d["player_name"], "team": d["team_tricode"],
                        "home": last[0], "away": last[1],
                    })
            if p is not None:
                prev_p = p
            series.append(point)

        # Thin the series: one point per scoring change is plenty for a curve
        # and keeps the payload small on a 500-event game.
        thinned: List[Dict[str, Any]] = []
        for pt in series:
            if not thinned or pt["home"] != thinned[-1]["home"] or pt["away"] != thinned[-1]["away"]:
                thinned.append(pt)
        if thinned and thinned[-1] is not series[-1]:
            thinned.append(series[-1])

        swings.sort(key=lambda s: -abs(s["delta"]))
        final = (last[0], last[1])

        return {
            "game_id": game_id,
            "game_date": meta["game_date"] if meta else None,
            "season": meta["season"] if meta else None,
            "season_type": meta["season_type"] if meta else None,
            "home": {
                "abbr": meta["home_abbr"] if meta else None,
                "name": meta["home_name"] if meta else None,
                "pts": final[0],
            },
            "away": {
                "abbr": meta["away_abbr"] if meta else None,
                "name": meta["away_name"] if meta else None,
                "pts": final[1],
            },
            "regulation_seconds": WP_REGULATION,
            "went_to_overtime": max(e["period"] for e in events) > 4,
            "series": thinned,
            "biggest_swings": swings[:5],
            "method": {
                "basis": "frequency",
                "games": len({r[0] for r in conn.execute("SELECT DISTINCT game_id FROM pbp_events")}),
                "note": "Counted from past games, not modelled. Possession is not accounted for, "
                        "which matters most inside the final minute.",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building win probability for {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Per-game assist network, parsed out of the play descriptions ---
#
# The play-by-play feed carries no assist player id. It carries a description -
# "Queen 4' Driving Layup (2 PTS) (Poole 1 AST)" - and the assisting player has
# to be recovered from that surname against the players actually on the floor.
#
# Three things break a naive match, all found by measuring rather than guessing:
#   accents      the description strips them ("Jokic" vs "Jokić")
#   suffixes     one side carries them ("Butler" vs "Butler III") - this alone
#                accounted for 891 misses
#   umlauts      sometimes transliterated ("Poeltl" vs "Pöltl")
#
# Matching is tiered on purpose: an exact fold is tried before the
# suffix-stripped one, because collapsing suffixes first lets "Jackson Jr."
# collide with a teammate named "Jackson". Getting that order wrong cost 828
# assists to ambiguity.
#
# MEASURED across all 274,538 assisted baskets: 99.90% resolve to exactly one
# teammate, 1 is ambiguous, 264 (0.10%) cannot be resolved. The unresolved count
# is returned per team so the page can say so rather than quietly under-counting.

_SUFFIX_RE = re.compile(r'\s+(jr|sr|ii|iii|iv|v)\.?$')
_INITIAL_RE = re.compile(r'^([A-Za-z]{1,3})\.\s*(.+)$')


def _fold_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", name)
        if unicodedata.category(ch) != "Mn"
    ).lower().strip()


def _loose_name(folded: str) -> str:
    return (_SUFFIX_RE.sub("", folded).strip()
            .replace("oe", "o").replace("ue", "u").replace("ae", "a"))


@app.get("/api/games/{game_id}/passing")
def get_game_passing(game_id: str):
    """
    Who assisted whom, for one game, per team.

    Only assists are available here - the feed records no pass that did not lead
    to a basket - so the payload reports assists, the points they created, and
    how many were threes. It deliberately does not carry pass counts or shooting
    percentages, which would be meaningless for a single game.
    """
    conn = get_db_conn()
    try:
        events = conn.execute(
            "SELECT person_id, player_name, team_id, team_tricode, assist_hint, shot_value "
            "FROM pbp_events WHERE game_id = ? ORDER BY action_number",
            (game_id,),
        ).fetchall()
        if not events:
            raise HTTPException(status_code=404, detail=f"No play-by-play stored for game {game_id}.")

        meta = conn.execute(
            "SELECT bs.game_date, bs.season, bs.season_type, bs.home_team_id, bs.away_team_id, "
            "       h.abbreviation AS home, h.full_name AS home_name, "
            "       a.abbreviation AS away, a.full_name AS away_name "
            "FROM box_scores bs "
            "JOIN team_metadata h ON h.team_id = bs.home_team_id "
            "JOIN team_metadata a ON a.team_id = bs.away_team_id "
            "WHERE bs.game_id = ?",
            (game_id,),
        ).fetchone()

        first_names = {
            r["player_id"]: (r["first_name"] or "")
            for r in conn.execute("SELECT player_id, first_name FROM players")
        }

        # Everyone who appears in this game, by team.
        roster: Dict[int, Dict[str, Any]] = {}
        for e in events:
            pid = e["person_id"]
            if not pid or not e["player_name"] or pid in roster:
                continue
            folded = _fold_name(e["player_name"])
            roster[pid] = {
                "id": pid, "name": e["player_name"], "team_id": e["team_id"],
                "exact": folded, "loose": _loose_name(folded),
            }

        def resolve(hint: str, team_id: int, shooter_id: int) -> Optional[int]:
            m = _INITIAL_RE.match(hint.strip())
            prefix, surname = (m.group(1), m.group(2)) if m else (None, hint)
            exact = _fold_name(surname)
            lo = _loose_name(exact)
            pool = [p for p in roster.values() if p["team_id"] == team_id and p["id"] != shooter_id]
            cands = [p["id"] for p in pool if p["exact"] == exact]
            if not cands:
                cands = [p["id"] for p in pool if p["loose"] == lo]
            if len(cands) > 1 and prefix:
                pk = _fold_name(prefix)
                narrowed = [p for p in cands
                            if (first_names.get(p, "") or "").lower().startswith(pk)]
                if narrowed:
                    cands = narrowed
            return cands[0] if len(cands) == 1 else None

        # team_id -> (passer, receiver) -> stats
        edges: Dict[int, Dict[tuple, Dict[str, int]]] = collections.defaultdict(dict)
        unresolved: Dict[int, int] = collections.defaultdict(int)

        for e in events:
            hint = e["assist_hint"]
            shooter = e["person_id"]
            tid = e["team_id"]
            if not hint or not shooter or not tid:
                continue
            passer = resolve(hint, tid, shooter)
            if passer is None:
                unresolved[tid] += 1
                continue
            key = (passer, shooter)
            slot = edges[tid].setdefault(key, {"assists": 0, "points": 0, "threes": 0})
            slot["assists"] += 1
            sv = e["shot_value"] or 2
            slot["points"] += sv
            if sv == 3:
                slot["threes"] += 1

        def build(team_id: int, abbr: str, name: str) -> Dict[str, Any]:
            conns = []
            players: Dict[int, Dict[str, Any]] = {}

            def slot(pid: int) -> Dict[str, Any]:
                if pid not in players:
                    players[pid] = {
                        "id": pid,
                        "name": roster.get(pid, {}).get("name", f"#{pid}"),
                        "assists_given": 0, "assists_received": 0,
                        "passes_made": 0, "passes_received": 0,
                    }
                return players[pid]

            total_assists = 0
            for (passer, receiver), st in edges.get(team_id, {}).items():
                conns.append({
                    "from": passer, "from_name": roster.get(passer, {}).get("name", f"#{passer}"),
                    "to": receiver, "to_name": roster.get(receiver, {}).get("name", f"#{receiver}"),
                    "assists": st["assists"],
                    "points": st["points"],
                    "threes": st["threes"],
                    # Not available from play-by-play; left at zero rather than invented.
                    "passes": 0, "fgm": 0, "fga": 0, "fg_pct": None, "fg3m": 0, "fg3a": 0,
                })
                slot(passer)["assists_given"] += st["assists"]
                slot(receiver)["assists_received"] += st["assists"]
                total_assists += st["assists"]

            conns.sort(key=lambda x: -x["assists"])
            return {
                "team_id": team_id, "team": abbr, "team_name": name,
                "tracked": len(conns) > 0,
                "totals": {"assists": total_assists, "passes": 0},
                "unresolved_assists": unresolved.get(team_id, 0),
                "top_duo": conns[0] if conns else None,
                "players": sorted(players.values(),
                                  key=lambda p: -(p["assists_given"] + p["assists_received"])),
                "connections": conns,
            }

        teams = []
        if meta:
            teams.append(build(meta["home_team_id"], meta["home"], meta["home_name"]))
            teams.append(build(meta["away_team_id"], meta["away"], meta["away_name"]))

        return {
            "game_id": game_id,
            "game_date": meta["game_date"] if meta else None,
            "season": meta["season"] if meta else None,
            "season_type": meta["season_type"] if meta else None,
            "teams": teams,
            "method": {
                "note": "Assisting players are recovered from the play description, which carries "
                        "a surname rather than an id. 99.9% resolve to exactly one teammate "
                        "across the archive; any that do not are counted, not guessed.",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building game passing for {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Excitement: which games were actually worth watching ---
#
# Score = the total distance the win probability travelled. A blowout barely
# moves it; a game that swings back and forth racks it up. This is the standard
# excitement-index construction and it needs no weights or tuning, which is why
# it is used here rather than something bespoke.
#
# It validates on the extremes: the top of the list is overtime games decided by
# a possession, the bottom is 40-point blowouts, and the top-50 median final
# margin is 4 points against 10 league-wide.
#
# KNOWN BIAS, surfaced in the API rather than hidden: the score sums across
# plays, so a longer game accumulates more. 49 of the top 50 are overtime games.
# That is defensible - overtime really is more exciting - but it buries a
# regulation thriller won at the buzzer, so `regulation_only` exists to let a
# reader take overtime out of the running entirely.

_EXCITEMENT_CACHE: Optional[List[Dict[str, Any]]] = None
_EXCITEMENT_CACHE_ROWS: int = -1

EXCITEMENT_LATE_SECONDS = 300  # "late" = final five minutes of regulation onward


def _build_excitement(conn) -> List[Dict[str, Any]]:
    table = _get_wp_table(conn)

    finals: Dict[str, tuple] = {}
    for gid, sh, sa in conn.execute(
        "SELECT game_id, score_home, score_away FROM pbp_events "
        "WHERE score_home IS NOT NULL ORDER BY game_id, action_number"
    ):
        finals[gid] = (sh, sa)

    meta = {
        r["game_id"]: dict(r)
        for r in conn.execute(
            "SELECT bs.game_id, bs.game_date, bs.season, bs.season_type, "
            "       h.abbreviation AS home, h.full_name AS home_name, "
            "       a.abbreviation AS away, a.full_name AS away_name "
            "FROM box_scores bs "
            "JOIN team_metadata h ON h.team_id = bs.home_team_id "
            "JOIN team_metadata a ON a.team_id = bs.away_team_id"
        )
    }

    out: List[Dict[str, Any]] = []
    cur = None
    last = (0, 0)
    prev_p: Optional[float] = None
    total = late = 0.0
    lead_changes = 0
    prev_sign = 0
    max_period = 1

    def flush(gid: Optional[str]):
        nonlocal total, late, lead_changes, prev_sign, max_period, prev_p, last
        if gid and prev_p is not None and gid in finals and gid in meta:
            fin = finals[gid]
            m = meta[gid]
            out.append({
                "game_id": gid,
                "game_date": (m["game_date"] or "")[:10],
                "season": m["season"],
                "season_type": m["season_type"],
                "home": m["home"], "home_name": m["home_name"], "home_pts": fin[0],
                "away": m["away"], "away_name": m["away_name"], "away_pts": fin[1],
                "score": round(total, 3),
                "late": round(late, 3),
                "lead_changes": lead_changes,
                "overtime": max_period > 4,
                "periods": max_period,
                "margin": abs(fin[0] - fin[1]),
            })
        total = late = 0.0
        lead_changes = 0
        prev_sign = 0
        max_period = 1
        prev_p = None
        last = (0, 0)

    for gid, es, period, sh, sa in conn.execute(
        "SELECT game_id, elapsed_seconds, period, score_home, score_away FROM pbp_events "
        "WHERE elapsed_seconds IS NOT NULL ORDER BY game_id, action_number"
    ):
        if gid != cur:
            flush(cur)
            cur = gid
        if period:
            max_period = max(max_period, period)
        if sh is None or sa is None or (sh, sa) == last:
            continue
        last = (sh, sa)
        p, _n = _wp_lookup(table, WP_REGULATION - es, sh - sa)
        if p is None:
            continue
        if prev_p is not None:
            d = abs(p - prev_p)
            total += d
            if es >= WP_REGULATION - EXCITEMENT_LATE_SECONDS:
                late += d
        prev_p = p
        sign = (sh > sa) - (sh < sa)
        if sign and prev_sign and sign != prev_sign:
            lead_changes += 1
        if sign:
            prev_sign = sign
    flush(cur)

    out.sort(key=lambda g: -g["score"])
    return out


def _get_excitement(conn) -> List[Dict[str, Any]]:
    global _EXCITEMENT_CACHE, _EXCITEMENT_CACHE_ROWS
    rows = conn.execute("SELECT COUNT(*) FROM pbp_events").fetchone()[0]
    if not rows:
        raise HTTPException(
            status_code=503,
            detail="Play-by-play has not been backfilled yet. Run backfill_pbp.py.",
        )
    if _EXCITEMENT_CACHE is not None and _EXCITEMENT_CACHE_ROWS == rows:
        return _EXCITEMENT_CACHE
    _EXCITEMENT_CACHE = _build_excitement(conn)
    _EXCITEMENT_CACHE_ROWS = rows
    return _EXCITEMENT_CACHE


@app.get("/api/stats/exciting-games")
def get_exciting_games(
    season: Optional[str] = None,
    season_type: Optional[str] = None,
    regulation_only: bool = False,
    sort: str = "score",
    limit: int = 100,
):
    """
    Games ranked by how far the win probability travelled.

    `sort=late` ranks by movement in the final five minutes instead, which is
    much less sensitive to game length than the overall score.
    """
    conn = get_db_conn()
    try:
        allg = _get_excitement(conn)
        rows = allg
        if season:
            rows = [g for g in rows if g["season"] == season]
        if season_type:
            rows = [g for g in rows if g["season_type"] == season_type]
        if regulation_only:
            rows = [g for g in rows if not g["overtime"]]
        if sort == "late":
            rows = sorted(rows, key=lambda g: -g["late"])

        seasons = sorted({g["season"] for g in allg}, reverse=True)
        return {
            "games_scored": len(allg),
            "matched": len(rows),
            "seasons": seasons,
            "sort": sort,
            "median_score": round(
                sorted(g["score"] for g in allg)[len(allg) // 2], 3
            ) if allg else None,
            "games": rows[: max(1, min(limit, 250))],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ranking exciting games: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Comeback probabilities from play-by-play ---
#
# "Down 15 entering the fourth, teams win 7.7% of the time (n=479)." Every cell
# is a frequency with its sample count, never a model output.
#
# Built by walking pbp_events once, carrying the last known score forward
# (only scoring plays carry one) and sampling the margin at each minute mark.
# The whole archive computes in about two seconds, so it is done lazily and
# held for the life of the process rather than precomputed into a table.

# Keyed on how many games are in pbp_events, so the grid rebuilds when the
# archive grows. A plain "compute once per process" cache would keep serving a
# stale grid after every nightly backfill until someone restarted the API -
# silently, since a slightly-wrong percentage looks exactly like a right one.
_COMEBACK_CACHE: Optional[Dict[str, Any]] = None
_COMEBACK_CACHE_GAMES: int = -1

# Minutes remaining in regulation. The labelled ones are the moments people
# actually ask about.
COMEBACK_MARKS = [
    (36, "End of Q1"),
    (24, "Halftime"),
    (18, "Mid Q3"),
    (12, "Start of Q4"),
    (9, "9 min left"),
    (6, "6 min left"),
    (3, "3 min left"),
    (1, "1 min left"),
]

COMEBACK_BUCKETS = [
    (1, 3, "1-3"), (4, 6, "4-6"), (7, 9, "7-9"), (10, 12, "10-12"),
    (13, 15, "13-15"), (16, 20, "16-20"), (21, 99, "21+"),
]


def _deficit_bucket(d: int) -> Optional[str]:
    for lo, hi, label in COMEBACK_BUCKETS:
        if lo <= d <= hi:
            return label
    return None


def _build_comeback_grid(conn) -> Dict[str, Any]:
    finals: Dict[str, tuple] = {}
    for gid, sh, sa in conn.execute(
        "SELECT game_id, score_home, score_away FROM pbp_events "
        "WHERE score_home IS NOT NULL ORDER BY game_id, action_number"
    ):
        finals[gid] = (sh, sa)

    marks = [m for m, _ in COMEBACK_MARKS]
    # (minutes_left, bucket) -> [situations, comebacks]
    agg: Dict[tuple, List[int]] = {}

    cur = None
    last = (0, 0)
    idx = 0
    minute_marks = list(range(0, 48))

    for gid, es, sh, sa in conn.execute(
        "SELECT game_id, elapsed_seconds, score_home, score_away FROM pbp_events "
        "WHERE elapsed_seconds IS NOT NULL ORDER BY game_id, action_number"
    ):
        if gid != cur:
            cur, last, idx = gid, (0, 0), 0
        if sh is not None and sa is not None:
            last = (sh, sa)
        final = finals.get(gid)
        if not final:
            continue
        mins = es / 60.0
        while idx < len(minute_marks) and mins >= minute_marks[idx]:
            elapsed_min = minute_marks[idx]
            idx += 1
            mins_left = 48 - elapsed_min
            if mins_left not in marks:
                continue
            margin = last[0] - last[1]
            home_won = final[0] > final[1]
            # Both perspectives: the trailing team is what we are counting, and
            # recording each game from both sides keeps the table symmetric and
            # doubles the sample.
            for deficit, won in ((-margin, home_won), (margin, not home_won)):
                if deficit <= 0:
                    continue
                b = _deficit_bucket(deficit)
                if not b:
                    continue
                cell = agg.setdefault((mins_left, b), [0, 0])
                cell[0] += 1
                cell[1] += 1 if won else 0

    rows = []
    for _, _, b in COMEBACK_BUCKETS:
        cells = []
        for m, label in COMEBACK_MARKS:
            n, w = agg.get((m, b), [0, 0])
            cells.append({
                "mins_left": m, "label": label, "games": n, "comebacks": w,
                "win_pct": round(w / n, 4) if n else None,
            })
        rows.append({"deficit": b, "cells": cells})

    seasons = [r[0] for r in conn.execute(
        "SELECT DISTINCT b.season FROM pbp_events p JOIN box_scores b ON b.game_id = p.game_id "
        "ORDER BY b.season"
    )]

    return {
        "games": len(finals),
        "seasons": seasons,
        "marks": [{"mins_left": m, "label": lbl} for m, lbl in COMEBACK_MARKS],
        "deficits": [b for _, _, b in COMEBACK_BUCKETS],
        "grid": rows,
    }


@app.get("/api/stats/comebacks")
def get_comeback_grid():
    """
    How often a trailing team came back, by deficit and time remaining.

    Pure frequencies over every archived game. A cell with a small `games`
    count means exactly that and callers must show it.
    """
    global _COMEBACK_CACHE, _COMEBACK_CACHE_GAMES
    conn = get_db_conn()
    try:
        games = conn.execute("SELECT COUNT(DISTINCT game_id) FROM pbp_events").fetchone()[0]
        if not games:
            raise HTTPException(
                status_code=503,
                detail="Play-by-play has not been backfilled yet. Run backfill_pbp.py.",
            )
        if _COMEBACK_CACHE is not None and _COMEBACK_CACHE_GAMES == games:
            return _COMEBACK_CACHE
        _COMEBACK_CACHE = _build_comeback_grid(conn)
        _COMEBACK_CACHE_GAMES = games
        return _COMEBACK_CACHE
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building comeback grid: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Schedule spots: rest, back-to-backs, and what they actually cost ---
#
# There is a `rest_features` table in retrain_features.sqlite, but it stops at
# 2024-04-28 and carries no result, so it cannot answer the only question worth
# asking - whether tired teams actually lose. Deriving rest from
# team_game_advanced instead gives current seasons AND the outcome attached.
#
# Rest is counted in days between a team's own consecutive games, never across
# a season boundary or between season types: a team's first playoff game is not
# "120 days rested" in any useful sense.

REST_BUCKETS = [
    (0, 0, "Back-to-back"),
    (1, 1, "1 day off"),
    (2, 2, "2 days off"),
    (3, 99, "3+ days off"),
]


def _rest_bucket(days: Optional[int]) -> Optional[str]:
    if days is None:
        return None
    for lo, hi, label in REST_BUCKETS:
        if lo <= days <= hi:
            return label
    return None


def _load_rest_rows(conn, season: str, season_type: str) -> List[Dict[str, Any]]:
    """One row per team-game with days of rest, the opponent's rest, and the result."""
    rows = conn.execute(
        """
        SELECT tga.team_id, tga.opp_team_id, tga.game_id, tga.game_date,
               tga.pts, tga.opp_pts,
               m.abbreviation AS team, m.full_name AS team_name,
               (CASE WHEN bs.home_team_id = tga.team_id THEN 1 ELSE 0 END) AS is_home
        FROM team_game_advanced tga
        JOIN team_metadata m ON m.team_id = tga.team_id
        JOIN box_scores bs ON bs.game_id = tga.game_id
        WHERE tga.season = ? AND tga.season_type = ?
        ORDER BY tga.team_id, tga.game_date
        """,
        (season, season_type),
    ).fetchall()

    # Days of rest, per team, in date order.
    prev_date: Dict[int, Any] = {}
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        gd = datetime.strptime(d["game_date"][:10], "%Y-%m-%d").date()
        last = prev_date.get(d["team_id"])
        # Days OFF, so consecutive calendar days is 0 - a back-to-back.
        d["rest_days"] = (gd - last).days - 1 if last else None
        prev_date[d["team_id"]] = gd
        out.append(d)

    # Attach each opponent's rest for the same game.
    by_game: Dict[str, List[Dict[str, Any]]] = {}
    for d in out:
        by_game.setdefault(d["game_id"], []).append(d)
    for pair in by_game.values():
        if len(pair) == 2:
            pair[0]["opp_rest"] = pair[1]["rest_days"]
            pair[1]["opp_rest"] = pair[0]["rest_days"]
        else:
            for d in pair:
                d["opp_rest"] = None
    return out


@app.get("/api/stats/rest")
def get_rest_splits(season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    How teams perform by how rested they are, and by how rested they are
    relative to the opponent.

    Every bucket carries its own sample size; small ones are meaningless and the
    caller must show n alongside any rate.
    """
    conn = get_db_conn()
    try:
        rows = _load_rest_rows(conn, season, season_type)
        if not rows:
            return {
                "season": season, "season_type": season_type, "games": 0,
                "by_rest": [], "by_advantage": [], "by_team": [],
            }

        def blank() -> Dict[str, Any]:
            return {"games": 0, "wins": 0, "margin": 0}

        def record(acc: Dict[str, Any], d: Dict[str, Any]) -> None:
            acc["games"] += 1
            acc["wins"] += 1 if d["pts"] > d["opp_pts"] else 0
            acc["margin"] += d["pts"] - d["opp_pts"]

        def finish(acc: Dict[str, Any], **extra) -> Dict[str, Any]:
            g = acc["games"]
            return {
                **extra,
                "games": g,
                "wins": acc["wins"],
                "losses": g - acc["wins"],
                "win_pct": round(acc["wins"] / g, 4) if g else None,
                "avg_margin": round(acc["margin"] / g, 2) if g else None,
            }

        by_rest: Dict[str, Dict[str, Any]] = {}
        by_adv: Dict[str, Dict[str, Any]] = {}
        by_team: Dict[int, Dict[str, Any]] = {}

        for d in rows:
            bucket = _rest_bucket(d["rest_days"])
            if bucket:
                by_rest.setdefault(bucket, blank())
                record(by_rest[bucket], d)

            # Rest advantage against the opponent, which is what a line reacts to.
            if d["rest_days"] is not None and d.get("opp_rest") is not None:
                diff = d["rest_days"] - d["opp_rest"]
                label = "Even rest" if diff == 0 else (
                    f"+{diff} day{'s' if diff > 1 else ''} rested" if diff > 0
                    else f"{diff} day{'s' if diff < -1 else ''} rested"
                )
                by_adv.setdefault(label, blank())
                by_adv[label]["diff"] = diff
                record(by_adv[label], d)

            t = by_team.setdefault(d["team_id"], {
                "team": d["team"], "team_name": d["team_name"],
                "b2b": blank(), "rested": blank(),
            })
            if d["rest_days"] == 0:
                record(t["b2b"], d)
            elif d["rest_days"] is not None and d["rest_days"] >= 2:
                record(t["rested"], d)

        order = [lbl for _, _, lbl in REST_BUCKETS]
        return {
            "season": season,
            "season_type": season_type,
            "games": len({d["game_id"] for d in rows}),
            "by_rest": [finish(by_rest[k], label=k) for k in order if k in by_rest],
            "by_advantage": sorted(
                [finish(v, label=k, diff=v.get("diff", 0)) for k, v in by_adv.items()],
                key=lambda x: x["diff"],
            ),
            "by_team": sorted(
                [
                    {
                        "team": t["team"], "team_name": t["team_name"],
                        "b2b": finish(t["b2b"]), "rested": finish(t["rested"]),
                        "drop": (
                            round((t["b2b"]["wins"] / t["b2b"]["games"]
                                   - t["rested"]["wins"] / t["rested"]["games"]) * 100, 1)
                            if t["b2b"]["games"] and t["rested"]["games"] else None
                        ),
                    }
                    for t in by_team.values()
                ],
                key=lambda x: (x["drop"] is None, x["drop"]),
            ),
        }
    except Exception as e:
        logger.error(f"Error computing rest splits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# --- Team passing wheel (player->player pass tracking, 2013-14 onward) ---
#
# One PlayerDashPtPass call per rostered player yields that player's outgoing
# edges; the roster's worth of calls is the complete directed passer->receiver
# matrix for a team-season. That is ~18 outbound requests, so the result is
# persisted in `team_passing` and only ever fetched once per team-season.
#
# `team_passing_fetch_log` records the attempt itself. Without it, a season with
# no tracking data (anything before 2013-14) would re-run the whole roster on
# every single page view and never store a thing.

def _flip_last_first(name: Optional[str]) -> str:
    """'Tatum, Jayson' -> 'Jayson Tatum'. Leaves already-normal names alone."""
    if not name:
        return ""
    if "," not in name:
        return name.strip()
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}".strip()


def _passing_roster(conn, team_id: int, season: str, season_type: str) -> List[Dict]:
    """
    Who to ask for pass data. Prefers the local season totals (no HTTP), and
    falls back to the official roster endpoint for seasons the archive does not
    cover — that fallback is what makes pre-2022 team-seasons work at all.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT t.player_id, p.full_name
        FROM player_season_totals t
        JOIN players p ON p.player_id = t.player_id
        WHERE t.team_id = ? AND t.season = ? AND t.season_type = ?
        """,
        (team_id, season, season_type),
    ).fetchall()
    if rows:
        return [{"player_id": r["player_id"], "full_name": r["full_name"]} for r in rows]

    from src.Utils.nba_stats_client import get_client
    client = get_client()
    roster = client.common_team_roster(team_id=team_id, season=season)
    return [
        {"player_id": r.get("PLAYER_ID"), "full_name": r.get("PLAYER")}
        for r in roster
        if r.get("PLAYER_ID")
    ]


def _ensure_team_passing(conn, team_id: int, season: str, season_type: str) -> None:
    """Populate `team_passing` for one team-season, once."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_passing (
            team_id INTEGER,
            season TEXT,
            season_type TEXT,
            passer_id INTEGER,
            passer_name TEXT,
            receiver_id INTEGER,
            receiver_name TEXT,
            games INTEGER,
            passes INTEGER,
            assists INTEGER,
            fgm INTEGER,
            fga INTEGER,
            fg_pct REAL,
            fg3m INTEGER,
            fg3a INTEGER,
            fetched_at TEXT,
            PRIMARY KEY (team_id, season, season_type, passer_id, receiver_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS team_passing_fetch_log (
            team_id INTEGER,
            season TEXT,
            season_type TEXT,
            fetched_at TEXT,
            players_queried INTEGER,
            edges_stored INTEGER,
            PRIMARY KEY (team_id, season, season_type)
        )
        """
    )

    already = conn.execute(
        "SELECT edges_stored FROM team_passing_fetch_log WHERE team_id=? AND season=? AND season_type=?",
        (team_id, season, season_type),
    ).fetchone()
    if already is not None:
        return

    roster = _passing_roster(conn, team_id, season, season_type)
    logger.info(
        "Fetching pass tracking for team %s %s %s (%d players)...",
        team_id, season, season_type, len(roster)
    )

    from src.Utils.nba_stats_client import get_client
    client = get_client()
    now = datetime.utcnow().isoformat()
    stored = 0

    def fetch_one(pid: int, fallback_name: Optional[str] = None) -> int:
        """Store one player's outgoing edges. Returns how many were written."""
        try:
            made = client.player_pass_dashboard(
                player_id=pid, team_id=team_id, season=season, season_type=season_type
            )
        except Exception as exc:
            # One dead player call must not lose the other seventeen.
            logger.warning("Pass dashboard failed for player %s (%s): %s", pid, season, exc)
            return 0

        written = 0
        for row in made:
            receiver_id = row.get("PASS_TEAMMATE_PLAYER_ID")
            if not receiver_id or receiver_id == pid:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO team_passing (
                    team_id, season, season_type, passer_id, passer_name,
                    receiver_id, receiver_name, games, passes, assists,
                    fgm, fga, fg_pct, fg3m, fg3a, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team_id, season, season_type, pid,
                    _flip_last_first(row.get("PLAYER_NAME_LAST_FIRST")) or fallback_name,
                    receiver_id, _flip_last_first(row.get("PASS_TO")),
                    row.get("G"), row.get("PASS"), row.get("AST"),
                    row.get("FGM"), row.get("FGA"), row.get("FG_PCT"),
                    row.get("FG3M"), row.get("FG3A"), now,
                ),
            )
            written += 1
        return written

    for player in roster:
        stored += fetch_one(player["player_id"], player.get("full_name"))

    # Second pass: mid-season departures.
    #
    # The roster above is the END-of-season squad, so a player traded away in
    # February is never asked for his outgoing passes - yet his teammates' rows
    # still name him as a receiver. Left there, he lands on the wheel with a
    # zero-width arc and, because arc colour is assists given against received,
    # gets painted as a pure finisher. Measured across five seasons that hit 30%
    # of team-seasons, and the players it caught were ball handlers like
    # McCollum, Harden and Derrick White - precisely the ones whose passing the
    # chart exists to show.
    #
    # Anyone who appears as a receiver but was never queried as a passer is
    # exactly that case, so ask for him too. One extra pass is enough in
    # practice: it would only fall short for a player whose sole receiver was
    # himself another mid-season departure.
    queried = {p["player_id"] for p in roster}
    received = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT receiver_id FROM team_passing WHERE team_id=? AND season=? AND season_type=?",
            (team_id, season, season_type),
        )
    }
    departed = sorted(pid for pid in received if pid not in queried)
    if departed:
        logger.info(
            "Second pass for team %s %s: %d mid-season departure(s) absent from the end-of-season roster.",
            team_id, season, len(departed),
        )
        for pid in departed:
            stored += fetch_one(pid)

    conn.execute(
        """
        INSERT OR REPLACE INTO team_passing_fetch_log
            (team_id, season, season_type, fetched_at, players_queried, edges_stored)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (team_id, season, season_type, now, len(roster) + len(departed), stored),
    )
    conn.commit()
    logger.info("Stored %d passing edges for team %s %s.", stored, team_id, season)


@app.get("/api/teams/{abbr}/passing")
def get_team_passing(
    abbr: str,
    season: str = CURRENT_SEASON,
    season_type: str = "Regular Season",
):
    """
    The passer->receiver matrix for one team-season, for the passing wheel.

    Returns every edge with at least one pass. `players` carries per-player
    totals in both directions so the chord layout can size arcs by assists
    given without a second pass over the edge list.

    An empty `connections` array with `tracked: false` means the season predates
    SecondSpectrum tracking (2013-14) — the caller must say so rather than
    render an empty wheel.
    """
    conn = get_db_conn()
    try:
        t_row = conn.execute(
            "SELECT team_id, full_name, nickname FROM team_metadata WHERE abbreviation = ?",
            (abbr.upper(),),
        ).fetchone()
        if not t_row:
            raise HTTPException(status_code=404, detail=f"Team abbreviation {abbr} not found.")
        team_id = t_row["team_id"]

        _ensure_team_passing(conn, team_id, season, season_type)

        rows = conn.execute(
            """
            SELECT passer_id, passer_name, receiver_id, receiver_name,
                   games, passes, assists, fgm, fga, fg_pct, fg3m, fg3a
            FROM team_passing
            WHERE team_id = ? AND season = ? AND season_type = ?
              AND passes > 0
            ORDER BY assists DESC, passes DESC
            """,
            (team_id, season, season_type),
        ).fetchall()

        connections = []
        players: Dict[int, Dict[str, Any]] = {}
        total_assists = 0
        total_passes = 0

        def slot(pid: int, name: str) -> Dict[str, Any]:
            if pid not in players:
                players[pid] = {
                    "id": pid, "name": name,
                    "assists_given": 0, "assists_received": 0,
                    "passes_made": 0, "passes_received": 0,
                }
            elif name and not players[pid]["name"]:
                players[pid]["name"] = name
            return players[pid]

        for r in rows:
            d = dict(r)
            connections.append({
                "from": d["passer_id"], "from_name": d["passer_name"],
                "to": d["receiver_id"], "to_name": d["receiver_name"],
                "passes": d["passes"] or 0, "assists": d["assists"] or 0,
                "fgm": d["fgm"] or 0, "fga": d["fga"] or 0, "fg_pct": d["fg_pct"],
                "fg3m": d["fg3m"] or 0, "fg3a": d["fg3a"] or 0,
            })
            giver = slot(d["passer_id"], d["passer_name"])
            taker = slot(d["receiver_id"], d["receiver_name"])
            giver["assists_given"] += d["assists"] or 0
            giver["passes_made"] += d["passes"] or 0
            taker["assists_received"] += d["assists"] or 0
            taker["passes_received"] += d["passes"] or 0
            total_assists += d["assists"] or 0
            total_passes += d["passes"] or 0

        # Team games, for the passes-per-game line. The archive is authoritative
        # where it reaches; for seasons it does not cover, fall back to the
        # largest per-player G, since a healthy starter's appearances are a close
        # proxy for the team's schedule.
        games_row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM team_game_advanced
            WHERE team_id = ? AND season = ? AND season_type = ?
            """,
            (team_id, season, season_type),
        ).fetchone()
        games = (games_row["n"] if games_row else 0) or 0
        games_exact = games > 0
        if not games_exact:
            games = max((dict(r)["games"] or 0 for r in rows), default=0)
        top = connections[0] if connections else None

        return {
            "team": abbr.upper(),
            "team_name": t_row["full_name"],
            "team_nickname": t_row["nickname"],
            "season": season,
            "season_type": season_type,
            "tracked": len(connections) > 0,
            "games": games,
            "games_exact": games_exact,
            "totals": {"assists": total_assists, "passes": total_passes},
            "top_duo": top,
            "players": sorted(
                players.values(),
                key=lambda p: p["assists_given"] + p["assists_received"],
                reverse=True,
            ),
            "connections": connections,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching team passing for {abbr}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@app.get("/api/games/{game_id}/advanced-box")
def get_game_advanced_box(game_id: str):
    """
    Fetch computed advanced box score stats for team and players.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # Fetch team stats
        cursor.execute(
            """
            SELECT tga.*, m.abbreviation, m.full_name
            FROM team_game_advanced tga
            JOIN team_metadata m ON tga.team_id = m.team_id
            WHERE tga.game_id = ?
            """,
            (game_id,)
        )
        team_rows = cursor.fetchall()
        
        # Fetch player stats
        cursor.execute(
            """
            SELECT pgl.*, p.full_name
            FROM player_game_log pgl
            JOIN players p ON pgl.player_id = p.player_id
            WHERE pgl.game_id = ?
            """,
            (game_id,)
        )
        player_rows = cursor.fetchall()
        
        return {
            "game_id": game_id,
            "teams": [dict(r) for r in team_rows],
            "players": [dict(r) for r in player_rows]
        }
    except Exception as e:
        logger.error(f"Error fetching advanced game box score: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/games/{game_id}")
def get_game_details(game_id: str):
    """
    Retrieve metadata, teams, and scores for a specific game from the box_scores table.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM box_scores WHERE game_id = ?",
            (game_id,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Game not found")
        
        traditional_json = row["traditional_json"]
        home_team_id = row["home_team_id"]
        away_team_id = row["away_team_id"]
        
        home_name = ""
        home_abbr = ""
        home_score = None
        away_name = ""
        away_abbr = ""
        away_score = None
        
        if traditional_json:
            traditional_data = json.loads(traditional_json)
            box_score = traditional_data.get("boxScoreTraditional", {})
            home_team_json = box_score.get("homeTeam", {})
            away_team_json = box_score.get("awayTeam", {})
            
            home_name = f"{home_team_json.get('teamCity', '')} {home_team_json.get('teamName', '')}".strip()
            home_abbr = home_team_json.get('teamTricode', '')
            home_score = home_team_json.get('statistics', {}).get('points')
            
            away_name = f"{away_team_json.get('teamCity', '')} {away_team_json.get('teamName', '')}".strip()
            away_abbr = away_team_json.get('teamTricode', '')
            away_score = away_team_json.get('statistics', {}).get('points')
        
        # If metadata is missing, fallback to database team_metadata lookup
        if not home_abbr or not away_abbr:
            cursor.execute("SELECT team_id, abbreviation, full_name FROM team_metadata WHERE team_id IN (?, ?)", (home_team_id, away_team_id))
            teams = cursor.fetchall()
            for team in teams:
                if team["team_id"] == home_team_id:
                    home_abbr = team["abbreviation"]
                    home_name = team["full_name"]
                elif team["team_id"] == away_team_id:
                    away_abbr = team["abbreviation"]
                    away_name = team["full_name"]
                    
        status = "Final" if (home_score is not None or away_score is not None) else "Scheduled"
        
        return {
            "game_id": game_id,
            "game_date": row["game_date"],
            "season": row["season"],
            "season_type": row["season_type"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team": {
                "name": home_name,
                "abbreviation": home_abbr,
                "score": home_score
            },
            "away_team": {
                "name": away_name,
                "abbreviation": away_abbr,
                "score": away_score
            },
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching game details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/games/{game_id}/play-by-play")
def get_game_play_by_play(game_id: str):
    """
    Retrieve play-by-play timeline events for a game.
    
    This endpoint enforces a cache-first discipline:
    1. Read cached `pbp_json` from the `box_scores` table.
    2. If the cached JSON is not NULL, return it immediately.
    3. If it is NULL, fall back to a live `nba_stats_client.play_by_play(game_id)` call.
    4. Store the fetched live result back into `box_scores.pbp_json` so it is permanently cached.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # 1. Read pbp_json from the box_scores table
        cursor.execute(
            "SELECT pbp_json FROM box_scores WHERE game_id = ?",
            (game_id,)
        )
        row = cursor.fetchone()
        
        if row and row["pbp_json"]:
            # Cache hit - return the cached JSON directly
            return json.loads(row["pbp_json"])
            
        # 2. Cache miss - fall back to live play_by_play call
        logger.info(f"PBP cache miss for game {game_id}. Querying live stats.nba.com...")
        from src.Utils.nba_stats_client import get_client
        client = get_client()
        events = client.play_by_play(game_id)
        
        # 3. Store back into box_scores.pbp_json if game exists in the table
        if events and row:
            events_json = json.dumps(events)
            cursor.execute(
                "UPDATE box_scores SET pbp_json = ? WHERE game_id = ?",
                (events_json, game_id)
            )
            conn.commit()
            logger.info(f"PBP data cached successfully in box_scores for game {game_id}.")
            
        return events
    except Exception as e:
        logger.error(f"Error fetching play-by-play events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Transformed game-flow payloads, keyed by game_id. Finished games never
# change, so entries are kept for the life of the process.
game_flow_cache: Dict[str, Dict[str, Any]] = {}

def _load_pbp_actions(game_id: str) -> List[Dict[str, Any]]:
    """
    Load playbyplayv3 actions for a game using the same fetch path as
    /api/games/{game_id}/play-by-play: box_scores.pbp_json first, then the
    nba_stats_client (which itself has a permanent Data/nba_cache disk cache
    for completed games).
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT pbp_json FROM box_scores WHERE game_id = ?", (game_id,))
        row = cursor.fetchone()
        if row and row["pbp_json"]:
            return json.loads(row["pbp_json"])
    except Exception as e:
        logger.warning(f"DB pbp_json lookup failed for game {game_id}: {e}")
    finally:
        conn.close()

    from src.Utils.nba_stats_client import get_client
    return get_client().play_by_play(game_id)

def _resolve_game_teams(game_id: str, actions: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Resolve {"abbr", "name"} descriptors for the home and away teams.
    Prefers box_scores + team_metadata; falls back to the pbp actions
    themselves (location 'h'/'v' + teamTricode).
    """
    home_id = None
    away_id = None
    # Tricodes straight from the pbp stream as a fallback.
    tricodes: Dict[str, Tuple[int, str]] = {}
    for action in actions:
        team_id = action.get("teamId")
        tricode = action.get("teamTricode")
        loc = action.get("location")
        if team_id and tricode and loc in ("h", "v") and loc not in tricodes:
            tricodes[loc] = (team_id, tricode)
        if len(tricodes) == 2:
            break

    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT home_team_id, away_team_id FROM box_scores WHERE game_id = ?",
            (game_id,)
        )
        row = cursor.fetchone()
        if row:
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
        if home_id is None and "h" in tricodes:
            home_id = tricodes["h"][0]
        if away_id is None and "v" in tricodes:
            away_id = tricodes["v"][0]

        def team_meta(team_id, fallback_abbr: str) -> Dict[str, str]:
            if team_id:
                cursor.execute(
                    "SELECT abbreviation, full_name FROM team_metadata WHERE team_id = ?",
                    (team_id,)
                )
                r = cursor.fetchone()
                if r:
                    return {"abbr": r["abbreviation"], "name": r["full_name"]}
            return {"abbr": fallback_abbr, "name": fallback_abbr}

        home = team_meta(home_id, tricodes.get("h", (0, ""))[1])
        away = team_meta(away_id, tricodes.get("v", (0, ""))[1])
        return home, away
    finally:
        conn.close()

@app.get("/api/games/{game_id}/game-flow")
@limiter.limit(RATE_LIMIT_UPSTREAM)
def get_game_flow(request: Request, game_id: str):
    """
    Score-margin game flow derived server-side from playbyplayv3:
    compact scoring series (one point per score change), scoring runs
    (>=8-point bursts while the opponent scores <=2), lead changes and ties.

    `t` on series/run points is seconds elapsed from game start
    (regulation periods = 720 s, overtime periods = 300 s);
    `margin` = home_score - away_score.
    """
    if game_id in game_flow_cache:
        return game_flow_cache[game_id]

    if not re.fullmatch(r"\d{10}", game_id):
        raise HTTPException(status_code=404, detail=f"Invalid game_id: {game_id}")

    try:
        actions = _load_pbp_actions(game_id)
    except Exception as e:
        logger.error(f"Upstream play-by-play fetch failed for game {game_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream stats.nba.com play-by-play fetch failed for game {game_id}: {e}"
        )

    if not actions:
        raise HTTPException(status_code=404, detail=f"No play-by-play data found for game {game_id}")

    try:
        home, away = _resolve_game_teams(game_id, actions)
        flow = build_game_flow(game_id, actions, home, away)
    except Exception as e:
        logger.error(f"Error building game flow for game {game_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build game flow: {e}")

    game_flow_cache[game_id] = flow
    return flow

@app.get("/api/games/{game_id}/shot-chart")
@limiter.limit(RATE_LIMIT_UPSTREAM)
def get_game_shot_chart(request: Request, game_id: str):
    """
    Fetch shot chart details for a game resolved directly via game_id.

    Coordinate space (stats.nba.com shotchartdetail convention): shot x/y are
    LOC_X / LOC_Y in tenths of feet with the basket at the origin -
    x in [-250, 250] (negative = left of the basket from the shooter's view),
    y in [-52, ~890] (increasing toward half court).

    `league_averages` (additive field) is league-wide FG% by zone for the
    game's season - one upstream call per season, cached.
    """
    # Check cache
    if game_id in shot_chart_cache:
        logger.info(f"Returning cached shot chart data for game: {game_id}")
        cached = shot_chart_cache[game_id]
        # Entries written by /api/shot-chart (shared cache) may predate the
        # league_averages field - backfill it additively.
        if "league_averages" not in cached:
            try:
                cached["league_averages"] = _get_league_shot_averages(_season_from_game_id(game_id))
            except Exception:
                cached["league_averages"] = []
        return cached

    try:
        logger.info(f"Fetching shot chart detail from NBA Stats API for game: {game_id}")
        sc = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            game_id_nullable=game_id,
            context_measure_simple="FGA",
            season_type_all_star="Regular Season"
        )
        data = sc.get_dict()
        
        result_sets = data.get("resultSets", [])
        shots_set = next((rs for rs in result_sets if rs.get("name") == "Shot_Chart_Detail"), None)
        shots = []
        if shots_set:
            headers = shots_set.get("headers", [])
            row_set = shots_set.get("rowSet", [])
            col_map = {h: idx for idx, h in enumerate(headers)}
            
            for row in row_set:
                try:
                    x = row[col_map["LOC_X"]]
                    y = row[col_map["LOC_Y"]]
                    made = row[col_map["SHOT_MADE_FLAG"]] == 1
                    player_name = row[col_map["PLAYER_NAME"]]
                    desc = f"{player_name} {'makes' if made else 'misses'} {row[col_map['SHOT_TYPE']]} from {row[col_map['SHOT_DISTANCE']]} ft"
                    
                    shots.append({
                        "player": player_name,
                        "x": x,
                        "y": y,
                        "result": "made" if made else "missed",
                        "description": desc
                    })
                except Exception:
                    continue
                    
        # Additive: league-wide FG% by zone for the game's season. Never let
        # this break the shots payload.
        try:
            league_averages = _get_league_shot_averages(_season_from_game_id(game_id))
        except Exception:
            league_averages = []

        response_data = {
            "game_id": game_id,
            "shots": shots,
            "league_averages": league_averages
        }
        shot_chart_cache[game_id] = response_data
        return response_data

    except Exception as e:
        logger.error(f"Error fetching game shot chart: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Streak board -------------------------------------------------------------
# Streak length alone is a bad ranking. A forty-game run of scoring ten points is
# less of an achievement than a twelve-game run of scoring thirty, so this board
# sorts by how rare a streak of that length has been rather than by how long it is.
#
# Rarity is measured against our own archive, 2022-23 onward. That is not
# "historical" in the all-time sense and the response says so rather than
# implying a century of context.
#
# "Active" needs care as well. The last game on record is the 2026 Finals, so a
# streak unbroken when a team's season ended is active in the only sense
# available: nobody has ended it. It is reported as of each team's last game.
STREAK_DEFS = [
    {"key": "team_win", "label": "Team wins", "scope": "team"},
    {"key": "team_loss", "label": "Team losses", "scope": "team"},
    {"key": "pts10", "label": "10+ points", "scope": "player"},
    {"key": "pts20", "label": "20+ points", "scope": "player"},
    {"key": "pts30", "label": "30+ points", "scope": "player"},
    {"key": "three", "label": "Made a three", "scope": "player"},
    {"key": "dd", "label": "Double-doubles", "scope": "player"},
]

# Below this a streak is not a streak, it is two games in a row. Set per type
# because three straight 30-point games is notable and three straight games with
# a three is not.
STREAK_FLOOR = {"team_win": 4, "team_loss": 4, "pts10": 12, "pts20": 6,
                "pts30": 3, "three": 12, "dd": 4}


def _qualifies(row, key: str) -> bool:
    pts = row["pts"] or 0
    if key == "pts10":
        return pts >= 10
    if key == "pts20":
        return pts >= 20
    if key == "pts30":
        return pts >= 30
    if key == "three":
        return (row["fg3m"] or 0) >= 1
    if key == "dd":
        # Any two of the five counting categories, which is the real definition -
        # not points and rebounds only.
        cats = (row["pts"], row["reb"], row["ast"], row["stl"], row["blk"])
        return sum(1 for c in cats if (c or 0) >= 10) >= 2
    return False


_streak_cache: Dict[str, Any] = {"key": None, "payload": None}


def _slice_streaks(payload: Dict[str, Any], kind: Optional[str], mode: str, limit: int):
    """Pick a mode's board out of the cached payload and filter it."""
    board = payload["longest"] if mode == "longest" else payload["streaks"]
    if kind:
        board = [b for b in board if b["kind"] == kind]
    return {
        **{k: v for k, v in payload.items() if k not in ("streaks", "longest")},
        "mode": mode,
        "total": len(board),
        "streaks": board[: max(1, min(limit, 200))],
    }


# --- Run detector -------------------------------------------------------------
# Unanswered scoring runs, precomputed by ingest_scoring_runs.py. A run is
# consecutive points by one team with the opponent scoring nothing in between -
# what "an 8-0 run" means, and a definition with no parameters to argue about.
#
# Runs are stored down to 6 points and filtered upward here, so the threshold is a
# reader's choice rather than baked into the table.
@app.get("/api/stats/runs")
def get_scoring_runs(
    min_points: int = 10,
    season: Optional[str] = None,
    season_type: str = "Regular Season",
    team: Optional[str] = None,
    limit: int = 40,
):
    """
    Biggest runs, who delivers them, who concedes them, and in which quarter.

    Per-team figures are per game rather than totals, because a team that played
    more games would otherwise look more run-prone for having existed longer.
    """
    conn = get_db_conn()
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scoring_runs'"
        ).fetchone()
        if not exists:
            raise HTTPException(
                status_code=503,
                detail="Run table not built yet - run ingest_scoring_runs.py.",
            )

        where = ["points >= ?", "season_type = ?"]
        params: List[Any] = [min_points, season_type]
        if season:
            where.append("season = ?")
            params.append(season)
        clause = " AND ".join(where)

        biggest_sql = f"SELECT * FROM scoring_runs WHERE {clause}"
        biggest_params = list(params)
        if team:
            biggest_sql += " AND (team_tricode = ? OR opp_tricode = ?)"
            biggest_params += [team.upper(), team.upper()]
        biggest = [
            dict(r) for r in conn.execute(
                biggest_sql + " ORDER BY points DESC, game_date DESC LIMIT ?",
                biggest_params + [max(1, min(limit, 200))],
            )
        ]

        # Games per team in scope, so runs can be expressed per game.
        games = {
            r["tri"]: r["n"] for r in conn.execute(
                f"""
                SELECT m.abbreviation AS tri, COUNT(DISTINCT t.game_id) AS n
                FROM team_game_advanced t
                JOIN team_metadata m ON m.team_id = t.team_id
                WHERE t.season_type = ?{" AND t.season = ?" if season else ""}
                GROUP BY m.abbreviation
                """,
                [season_type] + ([season] if season else []),
            )
        }

        delivered = {
            r["team_tricode"]: dict(r) for r in conn.execute(
                f"""
                SELECT team_tricode, COUNT(*) AS runs, SUM(points) AS points,
                       MAX(points) AS biggest
                FROM scoring_runs WHERE {clause} GROUP BY team_tricode
                """,
                params,
            )
        }
        allowed = {
            r["opp_tricode"]: dict(r) for r in conn.execute(
                f"""
                SELECT opp_tricode, COUNT(*) AS runs, SUM(points) AS points,
                       MAX(points) AS biggest
                FROM scoring_runs WHERE {clause} GROUP BY opp_tricode
                """,
                params,
            )
        }

        teams = []
        for tri, n_games in sorted(games.items()):
            d = delivered.get(tri, {})
            a = allowed.get(tri, {})
            if not n_games:
                continue
            teams.append({
                "team": tri,
                "games": n_games,
                "runs_delivered": d.get("runs", 0),
                "runs_allowed": a.get("runs", 0),
                "delivered_per_game": round((d.get("runs", 0)) / n_games, 2),
                "allowed_per_game": round((a.get("runs", 0)) / n_games, 2),
                "biggest_delivered": d.get("biggest"),
                "biggest_allowed": a.get("biggest"),
                "net_per_game": round(
                    ((d.get("runs", 0)) - (a.get("runs", 0))) / n_games, 2),
            })

        # Which quarter a run started in. The vault's interest here is the third,
        # where a team coming out of the break flat is a live-betting tell.
        by_quarter = [
            dict(r) for r in conn.execute(
                f"""
                SELECT start_period AS period, COUNT(*) AS runs,
                       ROUND(AVG(points), 2) AS avg_points, MAX(points) AS biggest
                FROM scoring_runs WHERE {clause}
                GROUP BY start_period ORDER BY start_period
                """,
                params,
            )
        ]

        # Teams that concede most in the third specifically.
        third_allowed = [
            dict(r) for r in conn.execute(
                f"""
                SELECT opp_tricode AS team, COUNT(*) AS runs_allowed_q3,
                       MAX(points) AS biggest
                FROM scoring_runs WHERE {clause} AND start_period = 3
                GROUP BY opp_tricode ORDER BY runs_allowed_q3 DESC LIMIT 10
                """,
                params,
            )
        ]

        totals = conn.execute(
            f"SELECT COUNT(*) n, MAX(points) mx FROM scoring_runs WHERE {clause}", params
        ).fetchone()
        seasons = [
            r["season"] for r in conn.execute(
                "SELECT DISTINCT season FROM scoring_runs ORDER BY season DESC"
            )
        ]

        return {
            "min_points": min_points,
            "season": season,
            "season_type": season_type,
            "seasons": seasons,
            "totals": {"runs": totals["n"], "biggest": totals["mx"]},
            "biggest": biggest,
            "teams": sorted(teams, key=lambda t: -t["allowed_per_game"]),
            "by_quarter": by_quarter,
            "third_quarter_allowed": third_allowed,
            "definition": (
                "A run is consecutive points by one team with the opponent scoring "
                "nothing in between. Runs can cross a period break, because a team "
                "closing one quarter and opening the next without reply has gone on "
                "one run, not two."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read scoring runs.")
    finally:
        conn.close()


# --- Hustle stats ---------------------------------------------------------------
# Raw material for the hustle index builder. The endpoint ships totals plus games
# and minutes and nothing derived: the page computes per-36 and the 0-100 scaling
# itself, the same shape as Build-a-Metric, so a slider move never round-trips.
@app.get("/api/stats/hustle")
def get_hustle_stats(season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    Season hustle totals for every player: the effort plays the box score skips.

    Tracked by the league from 2015-16 onward. Charges drawn are genuinely rare -
    a handful of players a season reach double digits - which the builder page
    should surface rather than smooth over.
    """
    try:
        from src.Utils.nba_stats_client import get_client

        rows = get_client().league_hustle_stats(season=season, season_type=season_type)
    except Exception as e:
        logger.error(f"Error fetching hustle stats: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Could not reach the hustle feed.")

    if not rows:
        return {"season": season, "season_type": season_type, "players": []}

    players = []
    for r in rows:
        players.append({
            "player_id": r.get("PLAYER_ID"),
            "name": r.get("PLAYER_NAME"),
            "team": r.get("TEAM_ABBREVIATION"),
            "gp": r.get("G"),
            "min": r.get("MIN"),
            "deflections": r.get("DEFLECTIONS"),
            "screen_assists": r.get("SCREEN_ASSISTS"),
            "screen_assist_pts": r.get("SCREEN_AST_PTS"),
            "loose_balls": r.get("LOOSE_BALLS_RECOVERED"),
            "charges_drawn": r.get("CHARGES_DRAWN"),
            "contested_shots": r.get("CONTESTED_SHOTS"),
            "box_outs": r.get("BOX_OUTS"),
        })

    return {
        "season": season,
        "season_type": season_type,
        "players": players,
        "source_note": (
            "Hustle tracking is the league's own, recorded from 2015-16. These are "
            "counted events, not estimates."
        ),
    }


@app.get("/api/stats/streaks")
def get_streak_board(limit: int = 40, kind: Optional[str] = None, mode: str = "active"):
    """
    Every unbroken streak in the archive, ranked by how rare its length is.

    Each entry carries how many streaks have reached that length in the archive,
    which is both the sort key and the number a reader needs to judge it. A player
    streak counts consecutive games PLAYED, the league's own convention - a missed
    game does not end it.

    Two modes, because "active" is systematically skewed at a season boundary.
    Every team except the champion ends its season on a loss, so at the end of a
    season there are almost no active team win streaks (five teams, longest two
    games) while losing streaks are inflated by playoff eliminations. `mode=active`
    is the live-season board; `mode=longest` is the rarest streaks in the archive
    however they ended, which is the one worth reading in the offseason.
    """
    conn = get_db_conn()
    try:
        # The whole board is computed at once and cached against the archive's
        # last game, so a backfill invalidates it and a filter change does not
        # rescan 120,000 rows.
        stamp = conn.execute(
            "SELECT MAX(DATE(game_date)) || ':' || COUNT(*) FROM player_game_log"
        ).fetchone()[0]
        if _streak_cache["key"] == stamp and _streak_cache["payload"]:
            return _slice_streaks(_streak_cache["payload"], kind, mode, limit)
        team_rows = conn.execute(
            """
            SELECT t.team_id, DATE(t.game_date) AS date, t.pts, t.opp_pts,
                   m.abbreviation AS team, m.full_name AS team_name
            FROM team_game_advanced t
            LEFT JOIN team_metadata m ON m.team_id = t.team_id
            WHERE t.pts IS NOT NULL AND t.opp_pts IS NOT NULL
            ORDER BY t.team_id, DATE(t.game_date)
            """
        ).fetchall()

        player_rows = conn.execute(
            """
            SELECT g.player_id, DATE(g.game_date) AS date, g.pts, g.reb, g.ast,
                   g.stl, g.blk, g.fg3m, p.full_name,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = g.team_id) AS team
            FROM player_game_log g
            JOIN players p ON p.player_id = g.player_id
            ORDER BY g.player_id, DATE(g.game_date)
            """
        ).fetchall()

        lengths: Dict[str, List[int]] = {d["key"]: [] for d in STREAK_DEFS}
        actives: Dict[str, List[Dict[str, Any]]] = {d["key"]: [] for d in STREAK_DEFS}
        completed: Dict[str, List[Dict[str, Any]]] = {d["key"]: [] for d in STREAK_DEFS}

        def walk(rows, group_key, tests, meta):
            """One pass per entity, recording every streak plus the trailing one.

            The trailing streak is the active one: it ran to the entity's last game
            without being broken.
            """
            current = {k: 0 for k in tests}
            start = {k: None for k in tests}
            prev_group = None
            last_row = None

            def flush():
                for k in tests:
                    if current[k]:
                        lengths[k].append(current[k])
                        actives[k].append(meta(last_row, current[k], start[k]))

            for r in rows:
                g = r[group_key]
                if g != prev_group:
                    if prev_group is not None:
                        flush()
                    for k in tests:
                        current[k] = 0
                        start[k] = None
                    prev_group = g
                for k, test in tests.items():
                    if test(r):
                        if not current[k]:
                            start[k] = r["date"]
                        current[k] += 1
                    else:
                        if current[k]:
                            lengths[k].append(current[k])
                            # Keep the streak itself, not only its length, so the
                            # longest board can name who did it. last_row is the
                            # final game OF THE STREAK, since r broke it.
                            completed[k].append(meta(last_row, current[k], start[k]))
                        current[k] = 0
                        start[k] = None
                last_row = r
            if prev_group is not None:
                flush()

        walk(
            team_rows, "team_id",
            {
                "team_win": lambda r: (r["pts"] or 0) > (r["opp_pts"] or 0),
                "team_loss": lambda r: (r["pts"] or 0) < (r["opp_pts"] or 0),
            },
            lambda r, n, st: {
                "name": r["team_name"] or r["team"], "team": r["team"],
                "team_id": r["team_id"], "length": n, "started": st,
                "last_game": r["date"],
            },
        )
        # Bind the key per lambda, or every test closes over the last one.
        player_tests = {
            d["key"]: (lambda key: (lambda r: _qualifies(r, key)))(d["key"])
            for d in STREAK_DEFS if d["scope"] == "player"
        }
        walk(
            player_rows, "player_id", player_tests,
            lambda r, n, st: {
                "name": r["full_name"], "team": r["team"], "player_id": r["player_id"],
                "length": n, "started": st, "last_game": r["date"],
            },
        )

        def reached(key: str, n: int) -> int:
            return sum(1 for L in lengths[key] if L >= n)

        def decorate(entries, k, d, active):
            out = []
            for a in entries:
                if a["length"] < STREAK_FLOOR[k]:
                    continue
                out.append({
                    **a,
                    "kind": k,
                    "kind_label": d["label"],
                    "scope": d["scope"],
                    "active": active,
                    "as_long_or_longer": reached(k, a["length"]),
                    "longest_ever": max(lengths[k]) if lengths[k] else a["length"],
                })
            return out

        board, longest = [], []
        for d in STREAK_DEFS:
            k = d["key"]
            board.extend(decorate(actives[k], k, d, True))
            # The longest board draws on every streak, ended or not, so it is not
            # hostage to where the season happened to stop.
            longest.extend(decorate(actives[k], k, d, True))
            longest.extend(decorate(completed[k], k, d, False))

        # Rarest first; a longer streak breaks a tie, being further into the tail.
        rank = lambda b: (b["as_long_or_longer"], -b["length"])
        board.sort(key=rank)
        longest.sort(key=rank)

        span = conn.execute(
            "SELECT MIN(season), MAX(season), MAX(DATE(game_date)) FROM team_game_advanced"
        ).fetchone()
        payload = {
            "archive": {"first_season": span[0], "last_season": span[1], "last_game": span[2]},
            "kinds": [
                {
                    "key": d["key"], "label": d["label"], "scope": d["scope"],
                    "floor": STREAK_FLOOR[d["key"]],
                    "longest": max(lengths[d["key"]]) if lengths[d["key"]] else None,
                    "streaks_recorded": len(lengths[d["key"]]),
                }
                for d in STREAK_DEFS
            ],
            "streaks": board,
            "longest": longest,
            # Why an active board looks strange at a season boundary, in numbers
            # rather than as a warning a reader has to take on trust.
            "season_boundary": {
                "teams_ending_on_a_win": sum(
                    1 for a in actives["team_win"]
                ),
                "teams_ending_on_a_loss": sum(
                    1 for a in actives["team_loss"]
                ),
                "longest_active_team_win": max(
                    (a["length"] for a in actives["team_win"]), default=0
                ),
            },
        }
        _streak_cache["key"] = stamp
        _streak_cache["payload"] = payload
        return _slice_streaks(payload, kind, mode, limit)
    except Exception as e:
        logger.error(f"Error building streak board: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not build the streak board.")
    finally:
        conn.close()


@app.get("/api/stats/daily-leaders")
def get_daily_leaders(
    date: Optional[str] = None,
    category: str = "pts",
    season_type: str = "Regular Season",
    limit: int = 10,
):
    """
    Leaders for a single night, from the game log.

    Season leaders reward whoever has played most; a night's leaders are the
    thing people actually talk about. `date` is YYYY-MM-DD and defaults to the
    most recent date on record rather than today, because today is usually the
    offseason or a night that has not been played yet - defaulting to an empty
    board would look broken.
    """
    CATEGORIES = {
        "pts": "g.pts", "reb": "g.reb", "ast": "g.ast", "stl": "g.stl",
        "blk": "g.blk", "fg3m": "g.fg3m", "min": "g.min", "tov": "g.tov",
        "fantasy": "(g.pts + 1.2*g.reb + 1.5*g.ast + 3.0*g.stl + 3.0*g.blk - g.tov)",
    }
    cat = category.lower()
    if cat not in CATEGORIES:
        raise HTTPException(
            status_code=400, detail=f"Category must be one of {sorted(CATEGORIES)}"
        )

    conn = get_db_conn()
    try:
        latest = conn.execute(
            "SELECT MAX(DATE(game_date)) FROM player_game_log"
        ).fetchone()[0]
        day = (date or latest or "")[:10]
        if not day:
            return {"date": None, "category": cat, "games": 0, "leaders": [], "available_dates": []}

        expr = CATEGORIES[cat]
        rows = conn.execute(
            f"""
            SELECT g.player_id, p.full_name, g.game_id, g.team_id, g.min,
                   g.pts, g.reb, g.ast, g.stl, g.blk, g.tov, g.fg3m,
                   (SELECT abbreviation FROM team_metadata WHERE team_id = g.team_id) AS team_abbr,
                   {expr} AS value
            FROM player_game_log g
            JOIN players p ON p.player_id = g.player_id
            WHERE DATE(g.game_date) = ?
            ORDER BY value DESC, g.min DESC
            LIMIT ?
            """,
            (day, max(1, min(limit, 50))),
        ).fetchall()

        games = conn.execute(
            "SELECT COUNT(DISTINCT game_id) FROM player_game_log WHERE DATE(game_date) = ?",
            (day,),
        ).fetchone()[0]

        # Neighbouring dates, so a page can step night by night without guessing
        # which dates exist - the archive has gaps between seasons.
        prev_day = conn.execute(
            "SELECT MAX(DATE(game_date)) FROM player_game_log WHERE DATE(game_date) < ?", (day,)
        ).fetchone()[0]
        next_day = conn.execute(
            "SELECT MIN(DATE(game_date)) FROM player_game_log WHERE DATE(game_date) > ?", (day,)
        ).fetchone()[0]

        return {
            "date": day,
            "is_latest": day == latest,
            "latest_date": latest,
            "prev_date": prev_day,
            "next_date": next_day,
            "category": cat,
            "games": games,
            "categories": sorted(CATEGORIES),
            "leaders": [dict(r) for r in rows],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/stats/daily-leaders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not compute daily leaders.")
    finally:
        conn.close()


@app.get("/api/stats/leaders")
def get_stats_leaders(category: str = "pts", season: str = CURRENT_SEASON, season_type: str = "Regular Season", limit: int = 10):
    """
    Fetch league leaders for a specific stat category.

    Counting stats (pts, ast, ...) are returned as season totals ordered by
    the total — clients divide by gp for per-game boards. Percentage stats
    (fg_pct, fg3_pct, ft_pct) are ordered by the percentage itself with
    basketball-reference-style qualification minimums so a 3-for-3 bench
    stint can't lead the league.
    """
    counting_categories = ["pts", "ast", "reb", "stl", "blk", "min", "fg3m", "tov", "pf"]
    # Fantasy points is the league's own scoring rule, not a house formula:
    #   PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV
    # Verified against NBA_FANTASY_PTS on leaguedashplayerstats for all 582
    # players in 2025-26 - zero difference to four decimal places - so the number
    # here matches the one nba.com prints rather than approximating it.
    FANTASY_SQL = ("(t.pts + 1.2 * t.reb + 1.5 * t.ast + 3.0 * t.stl "
                   "+ 3.0 * t.blk - t.tov)")
    # category -> (attempts column, minimum attempts to qualify, per 82-game season)
    pct_categories = {
        "fg_pct": ("fga", 300),
        "fg3_pct": ("fg3a", 82),
        "ft_pct": ("fta", 125),
    }
    cat = category.lower()
    if cat not in counting_categories and cat not in pct_categories and cat != "fantasy":
        raise HTTPException(
            status_code=400,
            detail=f"Category must be one of {counting_categories + list(pct_categories) + ['fantasy']}"
        )

    conn = get_db_conn()
    try:
        cursor = conn.cursor()

        # Category names are validated against the allowlists above, so the
        # f-string interpolation cannot inject SQL.
        if cat == "fantasy":
            query = f"""
                SELECT t.*, p.full_name,
                       (SELECT abbreviation FROM team_metadata WHERE team_id = t.team_id) as team_abbr,
                       {FANTASY_SQL} AS fantasy
                FROM player_season_totals t
                JOIN players p ON t.player_id = p.player_id
                WHERE t.season = ? AND t.season_type = ?
                ORDER BY fantasy DESC
                LIMIT ?
            """
            cursor.execute(query, (season, season_type, limit))
        elif cat in pct_categories:
            attempts_col, min_attempts = pct_categories[cat]
            query = f"""
                SELECT t.*, p.full_name,
                       (SELECT abbreviation FROM team_metadata WHERE team_id = t.team_id) as team_abbr
                FROM player_season_totals t
                JOIN players p ON t.player_id = p.player_id
                WHERE t.season = ? AND t.season_type = ?
                  AND t.{attempts_col} >= ? AND t.{cat} IS NOT NULL
                ORDER BY t.{cat} DESC
                LIMIT ?
            """
            cursor.execute(query, (season, season_type, min_attempts, limit))
        else:
            query = f"""
                SELECT t.*, p.full_name,
                       (SELECT abbreviation FROM team_metadata WHERE team_id = t.team_id) as team_abbr
                FROM player_season_totals t
                JOIN players p ON t.player_id = p.player_id
                WHERE t.season = ? AND t.season_type = ?
                ORDER BY t.{cat} DESC
                LIMIT ?
            """
            cursor.execute(query, (season, season_type, limit))
        rows = cursor.fetchall()

        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error fetching stats leaders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/stats/standings")
def get_stats_standings(season: str = CURRENT_SEASON, season_type: str = "Regular Season"):
    """
    Fetch league standings compiled from team_season_advanced and team_metadata.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.*, m.full_name, m.abbreviation, m.conference, m.division
            FROM team_season_advanced s
            JOIN team_metadata m ON s.team_id = m.team_id
            WHERE s.season = ? AND s.season_type = ?
            ORDER BY s.win_pct DESC
            """,
            (season, season_type)
        )
        rows = cursor.fetchall()
        
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Error fetching standings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/seasons/{year}")
def get_season_info(year: str):
    """
    Fetch team standings and overview stats for a season.
    """
    return get_stats_standings(season=year)

# --- Key numbers -------------------------------------------------------------
# The historical odds archive stores one table per season. Two quirks, both
# established by checking the data rather than by assumption:
#
# 1. `Spread` is signed (positive = home favoured) only in 2022-23 and 2023-24.
#    Every earlier season stores the FAVOURITE'S MAGNITUDE, unsigned, so the
#    column alone cannot say who was favoured - taken at face value it puts home
#    ATS at 42.7%, about six points below where it belongs. The sign is
#    recovered from the moneyline: whichever side is shorter was favoured. On the
#    two seasons that do carry a sign, that rule reproduces it for 99.56% of
#    games (11 disagreements in 2,510, every one a pick'em with equal or nearly
#    equal moneylines) and the magnitude matches exactly. Recovered, home ATS
#    across the archive is 48.60%, which is where NBA home ATS actually sits.
# 2. A handful of rows are unplayed games (Points = 0) or carry a zero margin,
#    which the NBA cannot produce. Both are excluded and counted, never silently
#    dropped - the caller reports the exclusions.
KEY_NUMBERS_SIGNED_SEASONS = frozenset({"2022-23", "2023-24"})


def _load_key_number_games(conn) -> tuple:
    """Every archived game with a margin, plus the exclusions made getting there."""
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'odds_%_new' "
            "ORDER BY name"
        )
    ]
    games: List[Dict[str, Any]] = []
    excluded = {"unplayed": 0, "zero_margin": 0, "missing_fields": 0}

    for table in tables:
        season = table.replace("odds_", "").replace("_new", "")
        rows = conn.execute(
            f'SELECT Date, Home, Away, OU, Spread, ML_Home, ML_Away, Points, Win_Margin '
            f'FROM "{table}"'
        ).fetchall()
        for r in rows:
            try:
                margin = float(r["Win_Margin"])
                points = float(r["Points"])
                spread = float(r["Spread"])
                ml_home = float(r["ML_Home"])
                ml_away = float(r["ML_Away"])
            except (TypeError, ValueError):
                excluded["missing_fields"] += 1
                continue
            if points <= 0:
                excluded["unplayed"] += 1
                continue
            if margin == 0:
                excluded["zero_margin"] += 1
                continue

            if season in KEY_NUMBERS_SIGNED_SEASONS:
                spread_signed = spread
            else:
                spread_signed = abs(spread) if ml_home < ml_away else -abs(spread)

            games.append({
                "season": season,
                "date": r["Date"],
                "home": r["Home"],
                "away": r["Away"],
                "margin": margin,
                "abs_margin": abs(margin),
                "spread": spread_signed,
                "total_line": float(r["OU"]) if r["OU"] is not None else None,
                "points": points,
            })
    return games, excluded


@app.get("/api/stats/key-numbers")
def get_key_numbers(season_from: Optional[str] = None, season_to: Optional[str] = None):
    """
    How NBA games actually finish, and therefore what a half-point is worth.

    The betting folklore about buying half-points comes from the NFL, where the
    margin distribution spikes hard at 3 and 7. Basketball has no such spikes,
    and this endpoint exists to show that rather than assert it: the margin
    histogram, the share of games landing on each number, and the observed push
    rate at each integer spread - each with its own n.
    """
    conn = _odds_snapshot_conn()
    try:
        games, excluded = _load_key_number_games(conn)
        if not games:
            raise HTTPException(status_code=503, detail="Historical odds archive is unavailable.")

        seasons = sorted({g["season"] for g in games})
        if season_from:
            games = [g for g in games if g["season"] >= season_from]
        if season_to:
            games = [g for g in games if g["season"] <= season_to]
        if not games:
            return {
                "seasons_available": seasons, "season_from": season_from,
                "season_to": season_to, "games": 0, "margins": [],
                "pushes": [], "excluded": excluded,
            }

        total = len(games)

        # Margin histogram. Every entry carries the share of games decided by
        # exactly that many points - which IS the value of moving a line through
        # that number, since those are precisely the games whose result flips.
        counts: Dict[int, int] = {}
        for g in games:
            counts[int(g["abs_margin"])] = counts.get(int(g["abs_margin"]), 0) + 1
        margins = [
            {
                "margin": m,
                "games": counts[m],
                "pct": round(counts[m] / total * 100, 3),
            }
            for m in sorted(counts)
        ]

        # Observed push rate at each integer spread. A push needs an integer
        # line, so half-point spreads are excluded from this table only.
        # A pick'em cannot push in a sport with no ties, so it is not a line the
        # half-point question applies to.
        integer_spread = [
            g for g in games if float(g["spread"]).is_integer() and abs(g["spread"]) >= 1
        ]
        push_by_line: Dict[int, Dict[str, int]] = {}
        for g in integer_spread:
            k = int(abs(g["spread"]))
            acc = push_by_line.setdefault(k, {"games": 0, "pushes": 0})
            acc["games"] += 1
            if g["margin"] == g["spread"]:
                acc["pushes"] += 1
        pushes = [
            {
                "line": k,
                "games": v["games"],
                "pushes": v["pushes"],
                "push_pct": round(v["pushes"] / v["games"] * 100, 3) if v["games"] else None,
            }
            for k, v in sorted(push_by_line.items())
        ]

        home_wins = sum(1 for g in games if g["margin"] > 0)
        covers = sum(1 for g in games if g["margin"] > g["spread"])
        push_count = sum(1 for g in games if g["margin"] == g["spread"])

        return {
            "seasons_available": seasons,
            "season_from": season_from or seasons[0],
            "season_to": season_to or seasons[-1],
            "games": total,
            "margins": margins,
            "pushes": pushes,
            "integer_spread_games": len(integer_spread),
            "summary": {
                "mean_margin": round(sum(g["abs_margin"] for g in games) / total, 2),
                "median_margin": sorted(g["abs_margin"] for g in games)[total // 2],
                "home_win_pct": round(home_wins / total * 100, 2),
                "home_ats_pct": round(covers / total * 100, 2),
                "push_pct": round(push_count / total * 100, 2),
                "most_common_margin": max(margins, key=lambda m: m["games"])["margin"],
            },
            "excluded": excluded,
            "spread_sign_recovered_for": [s for s in seasons if s not in KEY_NUMBERS_SIGNED_SEASONS],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/stats/key-numbers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not compute key numbers.")
    finally:
        conn.close()


# --- League schedule ---------------------------------------------------------
# nba.com/schedule is one page driven by query params, not seven pages: its
# "Preseason Schedule", "Regular Season Schedule", "NBA Cup Schedule" and
# "National TV Games" menu items are all filter states of the same feed. This
# endpoint mirrors that - one route, the same filters.
#
# Season type comes from the game id prefix, which is the league's own encoding:
# 001 preseason, 002 regular season, 003 all-star, 004 playoffs, 005 play-in,
# 006 special events. gameLabel/gameSubtype carry the NBA Cup and global-game
# markers on top of it.
SEASON_TYPE_BY_PREFIX = {
    "001": "Preseason",
    "002": "Regular Season",
    "003": "All-Star",
    "004": "Playoffs",
    "005": "Play-In",
    "006": "Special Event",
}


def _broadcaster_names(game: Dict[str, Any], scope: str, media: str) -> List[str]:
    """Display names for one scope ('national'/'home'/'away') and medium."""
    out = []
    for key, entries in (game.get("broadcasters") or {}).items():
        if not key.startswith(scope):
            continue
        for b in entries or []:
            if media and b.get("broadcasterMedia") != media:
                continue
            name = b.get("broadcasterDisplay") or b.get("broadcasterAbbreviation")
            # "TBD" is a real value in the feed for games whose window is sold
            # but unassigned. It is not a broadcaster, so it is not offered as a
            # filter option, but it is passed through so the row can show it.
            if name and name not in out:
                out.append(name)
    return out


@app.get("/api/schedule")
def get_league_schedule(
    season: str = "2026-27",
    season_type: Optional[str] = None,
    month: Optional[int] = None,
    team: Optional[str] = None,
    broadcaster: Optional[str] = None,
    national_tv_only: bool = False,
    hide_previous: bool = False,
    cup_only: bool = False,
):
    """
    The published league schedule, filtered the way nba.com/schedule filters it.

    `team` is a tricode (BOS). `broadcaster` matches a national TV display name
    (ESPN, ABC, NBC, Peacock, Prime Video). `hide_previous` drops dates before
    today. Every filter is optional and they compose.
    """
    try:
        from src.Utils.nba_stats_client import get_client

        league = get_client().schedule_league_v2(season=season)
    except Exception as e:
        logger.error(f"Error fetching league schedule: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Could not reach the NBA schedule feed.")

    today = datetime.now(timezone.utc).date()
    dates_out = []
    teams_seen: Dict[str, str] = {}
    broadcasters_seen: Dict[str, int] = {}
    season_types_seen: Dict[str, int] = {}
    total = 0

    for gd in league.get("gameDates") or []:
        games = []
        for g in gd.get("games") or []:
            gid = str(g.get("gameId") or "")
            stype = SEASON_TYPE_BY_PREFIX.get(gid[:3], "Other")
            season_types_seen[stype] = season_types_seen.get(stype, 0) + 1

            home = g.get("homeTeam") or {}
            away = g.get("awayTeam") or {}
            for t in (home, away):
                if t.get("teamTricode"):
                    teams_seen[t["teamTricode"]] = f"{t.get('teamCity', '')} {t.get('teamName', '')}".strip()

            natl_tv = _broadcaster_names(g, "national", "tv")
            for name in natl_tv:
                if name != "TBD":
                    broadcasters_seen[name] = broadcasters_seen.get(name, 0) + 1

            label = g.get("gameLabel") or ""
            subtype = g.get("gameSubtype") or ""
            is_cup = "cup" in label.lower() or subtype.startswith("in-season")

            if season_type and stype != season_type:
                continue
            if cup_only and not is_cup:
                continue
            if team and team.upper() not in {home.get("teamTricode"), away.get("teamTricode")}:
                continue
            if broadcaster and broadcaster not in natl_tv:
                continue
            if national_tv_only and not [n for n in natl_tv if n != "TBD"]:
                continue

            est = g.get("gameDateTimeEst") or ""
            if month and est[5:7].isdigit() and int(est[5:7]) != month:
                continue
            if hide_previous and est[:10]:
                try:
                    if datetime.strptime(est[:10], "%Y-%m-%d").date() < today:
                        continue
                except ValueError:
                    pass

            total += 1
            games.append({
                "game_id": gid,
                "season_type": stype,
                "date_est": est,
                "time_est": g.get("gameTimeEst"),
                "date_utc": g.get("gameDateTimeUTC"),
                "status": g.get("gameStatusText"),
                "week": g.get("weekNumber"),
                "week_name": g.get("weekName"),
                "label": label or None,
                "sub_label": g.get("gameSubLabel") or None,
                "is_cup": is_cup,
                "is_neutral": bool(g.get("isNeutral")),
                "arena": g.get("arenaName"),
                "arena_city": g.get("arenaCity"),
                "arena_state": g.get("arenaState"),
                "home": {
                    "tricode": home.get("teamTricode"),
                    "name": f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip(),
                    "team_id": home.get("teamId"),
                },
                "away": {
                    "tricode": away.get("teamTricode"),
                    "name": f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip(),
                    "team_id": away.get("teamId"),
                },
                "national_tv": natl_tv,
                "national_radio": _broadcaster_names(g, "national", "radio"),
                "home_tv": _broadcaster_names(g, "home", "tv"),
                "away_tv": _broadcaster_names(g, "away", "tv"),
            })

        if games:
            # The feed lists a night's games in its own sequence, which is not
            # tip-off order. A schedule that reads 7:30, 8:00, 7:00 down the page
            # is unreadable, so sort by start time.
            games.sort(key=lambda g: g["date_est"] or "")
            dates_out.append({
                "date": (games[0]["date_est"] or "")[:10],
                "label": gd.get("gameDate"),
                "games": games,
            })

    return {
        "season": season,
        "games": total,
        "dates": dates_out,
        "weeks": league.get("weeks") or [],
        "filters": {
            "season_type": season_type,
            "month": month,
            "team": team,
            "broadcaster": broadcaster,
            "national_tv_only": national_tv_only,
            "hide_previous": hide_previous,
            "cup_only": cup_only,
        },
        "options": {
            # Built from the whole feed, before filtering, so the dropdowns do
            # not shrink as the user narrows the view.
            "season_types": [
                {"value": k, "games": v}
                for k, v in sorted(season_types_seen.items(), key=lambda x: -x[1])
            ],
            "teams": [{"tricode": k, "name": v} for k, v in sorted(teams_seen.items())],
            "broadcasters": [
                {"name": k, "games": v}
                for k, v in sorted(broadcasters_seen.items(), key=lambda x: -x[1])
            ],
        },
    }


@app.get("/api/hall-of-fame")
def get_hall_of_fame(category: Optional[str] = None, year: Optional[int] = None, q: Optional[str] = None):
    """
    Naismith Hall of Fame inductees, grouped by enshrinement class.

    Ingested from the Hall's own site by ingest_hall_of_fame.py - there is no API
    for this. Every row carries where it came from and when it was fetched, so a
    stale table is visible rather than assumed current.

    The Hall enshrines players, coaches, referees, contributors and whole teams.
    All of them are here, because narrowing to players would quietly redefine
    what the Hall of Fame is.
    """
    conn = get_db_conn()
    try:
        try:
            rows = conn.execute("SELECT * FROM hof_inductees").fetchall()
        except sqlite3.Error:
            raise HTTPException(
                status_code=503,
                detail="Hall of Fame table not built yet - run ingest_hall_of_fame.py.",
            )
        if not rows:
            raise HTTPException(status_code=503, detail="Hall of Fame table is empty.")

        people = [dict(r) for r in rows]
        categories = sorted({p["category"] for p in people if p["category"]})
        years = sorted({p["class_year"] for p in people if p["class_year"]}, reverse=True)
        fetched = max((p.get("fetched_at") or "" for p in people), default="")

        sel = people
        if category:
            sel = [p for p in sel if (p.get("category") or "").lower() == category.lower()]
        if year:
            sel = [p for p in sel if p.get("class_year") == year]
        if q:
            needle = q.lower()
            sel = [p for p in sel if needle in (p.get("name") or "").lower()]

        classes: Dict[int, List[Dict[str, Any]]] = {}
        for p in sel:
            classes.setdefault(p.get("class_year") or 0, []).append(p)
        for members in classes.values():
            members.sort(key=lambda p: p.get("sort_name") or "")

        return {
            "total": len(people),
            "shown": len(sel),
            "source_url": people[0].get("source_url"),
            "fetched_at": fetched,
            "options": {
                "categories": [
                    {"name": c, "count": sum(1 for p in people if p["category"] == c)}
                    for c in categories
                ],
                "years": years,
            },
            "classes": [
                {"year": y, "members": classes[y]}
                for y in sorted(classes.keys(), reverse=True)
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/hall-of-fame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read the Hall of Fame table.")
    finally:
        conn.close()


@app.get("/api/hall-of-fame/careers")
def get_hof_careers(sort: str = "pts"):
    """
    What a Hall of Fame NBA career actually looks like, in numbers.

    Built from the career totals of the inducted players who played in the NBA.
    The point is descriptive, not predictive: it shows the range the Hall has
    actually accepted, including how low the bottom of that range goes, rather
    than asserting a threshold nobody at the Hall has ever published.

    Three honesty constraints are baked into the response:
      - `nba_players` vs `inducted_players` makes clear this covers a subset. The
        Naismith Hall enshrines WNBA players, Globetrotters and international
        figures with no NBA career, and they are counted but not measured.
      - Percentiles are reported alongside the min, because the interesting fact
        about the Hall is its floor, and a median alone hides it.
      - Counting stats reward longevity and era. The response carries era spans
        so a caller can show that a 1950s career is not comparable to a modern one.
    """
    conn = get_db_conn()
    try:
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM hof_career_totals").fetchall()]
        except sqlite3.Error:
            raise HTTPException(
                status_code=503,
                detail="Career table not built yet - run ingest_hof_careers.py.",
            )
        if not rows:
            raise HTTPException(status_code=503, detail="No Hall of Fame career totals stored.")

        inducted = conn.execute(
            "SELECT COUNT(*) c FROM hof_inductees WHERE category = 'Player'"
        ).fetchone()["c"]

        # Some inductees played most of their career in the ABA or the BAA and
        # only a handful of NBA games - Mel Daniels is in the Hall on an ABA
        # career and has 11 NBA games to his name. Their NBA totals are real but
        # they describe a fragment, so they are listed and flagged while being
        # kept out of the distribution: left in, they would set a "lowest Hall of
        # Famer" mark that is an artefact of which league the stats come from.
        PARTIAL_CAREER_GP = 200
        for r in rows:
            r["partial_nba_career"] = (r.get("gp") or 0) < PARTIAL_CAREER_GP
        full = [r for r in rows if not r["partial_nba_career"]]
        partial = [r for r in rows if r["partial_nba_career"]]

        def stat_summary(key: str) -> Dict[str, Any]:
            rows_for_stat = full
            vals = sorted(r[key] for r in rows_for_stat if r.get(key) is not None)
            if not vals:
                return {}
            n = len(vals)

            def q(p: float):
                return vals[min(n - 1, int(p * n))]

            lowest = min(rows_for_stat, key=lambda r: r.get(key) if r.get(key) is not None else 10**9)
            highest = max(rows_for_stat, key=lambda r: r.get(key) if r.get(key) is not None else -1)
            return {
                "stat": key,
                "n": n,
                "min": vals[0],
                "p10": q(0.10),
                "median": q(0.50),
                "p90": q(0.90),
                "max": vals[-1],
                "lowest_holder": lowest["name"],
                "highest_holder": highest["name"],
            }

        players = sorted(
            rows,
            key=lambda r: (r.get(sort) if r.get(sort) is not None else -1),
            reverse=True,
        )
        eras = [r["from_year"] for r in rows if r.get("from_year")]

        return {
            "nba_players": len(rows),
            "measured_players": len(full),
            "partial_career_players": len(partial),
            "partial_career_names": sorted(r["name"] for r in partial),
            "partial_career_gp_threshold": PARTIAL_CAREER_GP,
            "inducted_players": inducted,
            "never_played_nba": inducted - len(rows),
            "era": {"earliest_debut": min(eras) if eras else None,
                    "latest_debut": max(eras) if eras else None},
            "summaries": [stat_summary(k) for k in ("pts", "reb", "ast", "gp", "seasons", "ppg")],
            "players": players,
            "sorted_by": sort,
            "fetched_at": max((r.get("fetched_at") or "" for r in rows), default=""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/hall-of-fame/careers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read Hall of Fame careers.")
    finally:
        conn.close()


@app.get("/api/cup")
def get_nba_cup(season: str = "2026-27"):
    """
    The Emirates NBA Cup: six groups, the knockout round, and who plays whom.

    Group membership is not published as a roster anywhere in the feed - it is
    derived from the games, since a team's group is whichever group its group
    games carry. Sixty group games across six groups of five teams, each team
    playing the other four once.

    Standings are deliberately absent. Before the tournament starts there are no
    results to stand on, and inventing a table of zeroes would imply the section
    is live when it is not.
    """
    sched = get_league_schedule(season=season, cup_only=True)
    games = [g for d in sched["dates"] for g in d["games"]]
    if not games:
        raise HTTPException(status_code=503, detail="No NBA Cup games in this season's feed.")

    groups: Dict[str, Dict[str, Any]] = {}
    knockout: List[Dict[str, Any]] = []
    KNOCKOUT_ORDER = {"Quarterfinal": 1, "Semifinal": 2, "Championship": 3}

    for g in games:
        stage = g.get("sub_label") or ""
        if stage in KNOCKOUT_ORDER:
            knockout.append(g)
            continue
        if not stage:
            continue
        entry = groups.setdefault(stage, {"group": stage, "teams": {}, "games": []})
        entry["games"].append(g)
        for side in ("home", "away"):
            t = g[side]
            if t.get("tricode"):
                entry["teams"][t["tricode"]] = t["name"]

    group_list = [
        {
            "group": name,
            "conference": "East" if name.startswith("East") else "West",
            "teams": [{"tricode": k, "name": v} for k, v in sorted(data["teams"].items())],
            "games": len(data["games"]),
            "first_game": min(g["date_est"] for g in data["games"])[:10],
            "last_game": max(g["date_est"] for g in data["games"])[:10],
        }
        for name, data in sorted(groups.items())
    ]
    knockout.sort(key=lambda g: (KNOCKOUT_ORDER.get(g.get("sub_label") or "", 9), g["date_est"]))

    return {
        "season": season,
        "total_games": len(games),
        "groups": group_list,
        "knockout": [
            {
                "stage": g.get("sub_label"),
                "date": g["date_est"][:10],
                "date_est": g["date_est"],
                "arena": g["arena"],
                "arena_city": g["arena_city"],
                "national_tv": g["national_tv"],
                # Knockout brackets are seeded by group results, so before the
                # group stage these carry placeholder teams. Say so rather than
                # printing whatever the feed has parked there.
                "teams_decided": bool(g["home"]["tricode"] and g["away"]["tricode"]),
                "home": g["home"],
                "away": g["away"],
            }
            for g in knockout
        ],
        "group_stage": {
            "first_game": min(g["date_est"] for g in games)[:10],
            "last_game": max(
                (g["date_est"] for g in games if (g.get("sub_label") or "") not in KNOCKOUT_ORDER),
                default="",
            )[:10],
        },
    }


# --- Lineups ------------------------------------------------------------------
# Two thousand five-man combinations come back per season and the median one
# played twenty minutes. A leaderboard sorted by net rating with no minutes floor
# is therefore not a leaderboard: the best figure in 2025-26 belongs to a lineup
# that played thirteen minutes and outscored people by 106 per 100 possessions,
# which will not happen again. The floor is the feature here, so it is a
# first-class parameter with a real default rather than a filter nobody finds.
LINEUP_MIN_MINUTES_DEFAULT = 100

LINEUP_SORTS = {
    "net_rating": "NET_RATING", "off_rating": "OFF_RATING", "def_rating": "DEF_RATING",
    "min": "MIN", "gp": "GP", "poss": "POSS", "ts_pct": "TS_PCT", "pace": "PACE",
}


@app.get("/api/lineups")
def get_lineups(
    season: str = CURRENT_SEASON,
    season_type: str = "Regular Season",
    group_quantity: int = 5,
    team: Optional[str] = None,
    min_minutes: float = LINEUP_MIN_MINUTES_DEFAULT,
    sort: str = "net_rating",
    limit: int = 60,
):
    """
    Lineup combinations, with the sample-size problem made explicit.

    `defensive` ratings here are the lineup's own, per 100 possessions. Sorting
    defaults to net rating and the minutes floor defaults to 100, because the
    unfiltered version of this table is actively misleading.

    The response carries what the floor removed - how many lineups exist, how
    many survive, and the best figure below the cut - so a page can show the
    reader why the floor is there instead of asserting it.
    """
    if group_quantity not in (2, 3, 4, 5):
        raise HTTPException(status_code=400, detail="group_quantity must be 2, 3, 4 or 5.")
    if sort not in LINEUP_SORTS:
        raise HTTPException(
            status_code=400,
            detail=f"sort must be one of: {', '.join(sorted(LINEUP_SORTS))}",
        )

    try:
        from src.Utils.nba_stats_client import get_client

        rows = get_client().league_dash_lineups(
            season=season, season_type=season_type,
            group_quantity=group_quantity, measure_type="Advanced",
        )
    except Exception as e:
        logger.error(f"Error fetching lineups: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Could not reach the lineup feed.")

    if not rows:
        return {
            "season": season, "season_type": season_type,
            "group_quantity": group_quantity, "total_lineups": 0, "lineups": [],
        }

    def num(v):
        return float(v) if isinstance(v, (int, float)) else None

    all_lineups = []
    for r in rows:
        mins = num(r.get("MIN")) or 0.0
        all_lineups.append({
            "group_id": r.get("GROUP_ID"),
            "name": r.get("GROUP_NAME"),
            # The feed gives one string of abbreviated names; split it so a page
            # can render them as separate players rather than one long line.
            "players": [p.strip() for p in (r.get("GROUP_NAME") or "").split(" - ") if p.strip()],
            "team": r.get("TEAM_ABBREVIATION"),
            "team_id": r.get("TEAM_ID"),
            "gp": r.get("GP"), "w": r.get("W"), "l": r.get("L"),
            "min": round(mins, 1),
            "poss": num(r.get("POSS")),
            "off_rating": num(r.get("OFF_RATING")),
            "def_rating": num(r.get("DEF_RATING")),
            "net_rating": num(r.get("NET_RATING")),
            "ts_pct": num(r.get("TS_PCT")),
            "pace": num(r.get("PACE")),
            "reb_pct": num(r.get("REB_PCT")),
            "tov_pct": num(r.get("TM_TOV_PCT")),
        })

    total = len(all_lineups)
    if team:
        pool = [l for l in all_lineups if (l["team"] or "").upper() == team.upper()]
    else:
        pool = all_lineups

    qualified = [l for l in pool if (l["min"] or 0) >= min_minutes]
    below = [l for l in pool if (l["min"] or 0) < min_minutes]

    key = LINEUP_SORTS[sort]
    field = {
        "NET_RATING": "net_rating", "OFF_RATING": "off_rating", "DEF_RATING": "def_rating",
        "MIN": "min", "GP": "gp", "POSS": "poss", "TS_PCT": "ts_pct", "PACE": "pace",
    }[key]
    # Defence is the one column where lower is better.
    reverse = field != "def_rating"
    qualified.sort(key=lambda l: (l.get(field) is None, l.get(field) or 0), reverse=reverse)
    if reverse:
        qualified = [l for l in qualified if l.get(field) is not None] +                     [l for l in qualified if l.get(field) is None]

    # What the floor is protecting the reader from, by name.
    noise = None
    if below:
        worst_offender = max(below, key=lambda l: l.get("net_rating") or -999)
        if worst_offender.get("net_rating") is not None:
            noise = {
                "name": worst_offender["name"],
                "team": worst_offender["team"],
                "min": worst_offender["min"],
                "net_rating": worst_offender["net_rating"],
            }

    mins_sorted = sorted((l["min"] or 0) for l in pool)
    n = len(mins_sorted) or 1

    return {
        "season": season,
        "season_type": season_type,
        "group_quantity": group_quantity,
        "team": team,
        "sort": sort,
        "min_minutes": min_minutes,
        "total_lineups": total,
        "in_scope": len(pool),
        "qualified": len(qualified),
        "excluded": len(below),
        "median_minutes": round(mins_sorted[n // 2], 1),
        "max_minutes": round(mins_sorted[-1], 1) if mins_sorted else 0,
        # The single most useful honesty figure on the page: how few combinations
        # have played enough to say anything.
        "minute_bands": [
            {"at_least": t, "lineups": sum(1 for m in mins_sorted if m >= t)}
            for t in (25, 50, 100, 200, 400)
        ],
        "excluded_best": noise,
        "lineups": qualified[:limit],
        "teams": sorted({l["team"] for l in all_lineups if l["team"]}),
    }


@app.get("/api/health/validation")
def get_validation_health():
    """
    Every place our derived numbers disagree with NBA.com, and by how much.

    The pipeline recomputes pace, offensive rating and defensive rating from raw
    box scores and compares each against the official figure, writing the result
    to raw_scrape_log. This publishes that audit trail rather than summarising it
    charitably: the distribution of disagreements, the direction of the bias, and
    the log's own limitations.

    Three things the caller must surface, or the numbers mislead:
      - `ok` and `warning` rows were written under DIFFERENT tolerances. The
        validator's default is 1.0 now; the 'ok' rows predate that and say 3.0.
        Counting one against the other measures a threshold change, not quality.
      - Ratings are double-counted by construction. Our offensive rating for one
        team is our defensive rating for the other, so a single disagreement in a
        game appears as two rows with identical numbers.
      - The log covers the days the archive was built, not every day since. It is
        a record of how the archive was validated, not a live heartbeat.
    """
    conn = get_db_conn()
    try:
        rows = conn.execute(
            "SELECT logged_at, game_id, team_id, endpoint, status, metric, "
            "our_value, official_value, diff FROM raw_scrape_log"
        ).fetchall()
        if not rows:
            raise HTTPException(status_code=503, detail="Validation log is empty.")

        statuses: Dict[str, int] = {}
        days: Dict[str, int] = {}
        by_metric: Dict[str, List[Dict[str, float]]] = {}
        all_games = set()
        warned_games = set()
        error_games = set()
        error_rows = 0

        for r in rows:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            if r["logged_at"]:
                day = r["logged_at"][:10]
                days[day] = days.get(day, 0) + 1
            if r["game_id"]:
                all_games.add(r["game_id"])
                if r["status"] == "warning":
                    warned_games.add(r["game_id"])
                elif r["status"] == "error":
                    error_games.add(r["game_id"])
            if r["status"] == "error":
                error_rows += 1
            if r["metric"] and r["diff"] is not None:
                by_metric.setdefault(r["metric"], []).append({
                    "diff": abs(float(r["diff"])),
                    "signed": float(r["our_value"]) - float(r["official_value"])
                    if r["our_value"] is not None and r["official_value"] is not None else 0.0,
                })

        def pct(values: List[float], q: float) -> float:
            s = sorted(values)
            if not s:
                return 0.0
            return round(s[min(len(s) - 1, int(q * len(s)))], 2)

        # Buckets chosen around the tolerance, so a reader can see how much of the
        # disagreement is the documented 1-3 point band and how much is the tail.
        edges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, None)]
        metrics = []
        for name, vals in sorted(by_metric.items()):
            diffs = [v["diff"] for v in vals]
            signed = [v["signed"] for v in vals]
            n = len(diffs)
            buckets = []
            for lo, hi in edges:
                c = sum(1 for d in diffs if d >= lo and (hi is None or d < hi))
                buckets.append({
                    "from": lo, "to": hi, "games": c,
                    "pct": round(c / n * 100, 2) if n else 0.0,
                })
            metrics.append({
                "metric": name,
                "n": n,
                "mean_abs_diff": round(sum(diffs) / n, 2) if n else None,
                "median_abs_diff": pct(diffs, 0.5),
                "p95_abs_diff": pct(diffs, 0.95),
                "max_abs_diff": round(max(diffs), 2) if diffs else None,
                "bias": round(sum(signed) / n, 2) if n else None,
                "within_1": round(sum(1 for d in diffs if d <= 1) / n * 100, 2) if n else None,
                "within_3": round(sum(1 for d in diffs if d <= 3) / n * 100, 2) if n else None,
                "over_5": round(sum(1 for d in diffs if d > 5) / n * 100, 2) if n else None,
                "buckets": buckets,
            })

        # An error row means a fetch failed at the time. It does not mean the game
        # is missing - the backfill retries - so resolve them against the archive
        # rather than reporting a scary count that has already been fixed.
        recovered = 0
        if error_games:
            placeholders = ",".join("?" * len(error_games))
            ids = list(error_games)
            in_box = {
                r["game_id"] for r in conn.execute(
                    f"SELECT DISTINCT game_id FROM box_scores WHERE game_id IN ({placeholders})",
                    ids,
                ).fetchall()
            }
            in_adv = {
                r["game_id"] for r in conn.execute(
                    f"SELECT DISTINCT game_id FROM team_game_advanced WHERE game_id IN ({placeholders})",
                    ids,
                ).fetchall()
            }
            recovered = len(in_box & in_adv)

        stamps = sorted(r["logged_at"] for r in rows if r["logged_at"])
        return {
            "rows": len(rows),
            "statuses": statuses,
            "first_logged": stamps[0] if stamps else None,
            "last_logged": stamps[-1] if stamps else None,
            "days_logged": sorted(
                [{"day": d, "rows": c} for d, c in days.items()], key=lambda x: x["day"]
            ),
            "games_validated": len(all_games),
            "games_with_warning": len(warned_games),
            "metrics": metrics,
            "errors": {
                "rows": error_rows,
                "games": len(error_games),
                "recovered": recovered,
                "still_missing": len(error_games) - recovered,
            },
            "caveats": {
                "current_threshold": 1.0,
                "legacy_ok_threshold": 3.0,
                "ratings_double_counted": True,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api/health/validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read the validation log.")
    finally:
        conn.close()


@app.get("/api/health/data")
def get_data_health():
    """
    Fetch health metrics of the local database and data pipeline.
    """
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        
        # Get count of raw games cached
        cursor.execute("SELECT COUNT(*) as count FROM box_scores")
        raw_games = cursor.fetchone()["count"]
        
        # Get count of computed games
        cursor.execute("SELECT COUNT(DISTINCT game_id) as count FROM team_game_advanced")
        computed_games = cursor.fetchone()["count"]
        
        # Get count of players
        cursor.execute("SELECT COUNT(*) as count FROM players")
        players_cnt = cursor.fetchone()["count"]
        
        # Get count of game logs
        cursor.execute("SELECT COUNT(*) as count FROM player_game_log")
        logs_cnt = cursor.fetchone()["count"]
        
        # Get validation failure rate
        from src.Utils.nba_validation import get_validation_failure_rate
        failure_rate = get_validation_failure_rate(conn)
        
        # Get latest fetched timestamp
        cursor.execute("SELECT MAX(fetched_at) FROM box_scores")
        latest_fetch = cursor.fetchone()[0]
        
        # Get latest computed timestamp
        cursor.execute("SELECT MAX(computed_at) FROM team_game_advanced")
        latest_compute = cursor.fetchone()[0]
        
        # Get latest validation log timestamp
        cursor.execute("SELECT MAX(logged_at) FROM raw_scrape_log")
        latest_validation = cursor.fetchone()[0]
        
        return {
            "database_status": "healthy",
            "cached_raw_games": raw_games,
            "computed_games": computed_games,
            "indexed_players": players_cnt,
            "player_game_logs": logs_cnt,
            "validation_failure_rate_pct": round(failure_rate, 2),
            "freshness": {
                "box_scores": latest_fetch,
                "team_game_advanced": latest_compute,
                "validation_logs": latest_validation
            }
        }
    except Exception as e:
        logger.error(f"Error fetching data health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Elo power ratings (computed lazily from TeamData.sqlite, cached per process) ---
_elo_history_cache: Dict[str, Any] = {}

def _get_elo_history() -> Dict[str, Any]:
    if "history" not in _elo_history_cache:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'TeamData.sqlite')
        _elo_history_cache["history"] = elo_engine.compute_elo_history(db_path)
    return _elo_history_cache["history"]

@app.get("/api/power-ratings")
def get_power_ratings(season: str = CURRENT_SEASON):
    """
    FiveThirtyEight-style Elo power ratings (see src/Utils/elo.py for the
    methodology). Elo runs continuously from 2022-23 through today, with 25%
    reversion toward 1505 between seasons; `season` selects which season's
    final (or, mid-season, current) state to report.
    """
    try:
        history = _get_elo_history()
    except Exception as e:
        logger.error(f"Error computing Elo history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not compute Elo ratings.")

    if season not in history["season_end_ratings"]:
        available = ", ".join(history["seasons"])
        raise HTTPException(status_code=404, detail=f"No Elo history for season '{season}'. Available: {available}.")

    as_of = history["season_last_date"][season]
    season_ratings = history["season_end_ratings"][season]
    cutoff_7d = (datetime.strptime(as_of, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

    # Team names, regular-season records, and net rating for the season.
    meta: Dict[int, Dict[str, Any]] = {}
    conn = get_db_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT team_id, full_name, abbreviation FROM team_metadata")
        for row in cursor.fetchall():
            meta[row["team_id"]] = {"team": row["full_name"], "abbr": row["abbreviation"],
                                    "wins": None, "losses": None, "net_rtg": None}
        cursor.execute(
            "SELECT team_id, wins, losses, net_rating FROM team_season_advanced "
            "WHERE season = ? AND season_type = 'Regular Season'",
            (season,),
        )
        for row in cursor.fetchall():
            if row["team_id"] in meta:
                meta[row["team_id"]]["wins"] = row["wins"]
                meta[row["team_id"]]["losses"] = row["losses"]
                meta[row["team_id"]]["net_rtg"] = round(row["net_rating"], 1) if row["net_rating"] is not None else None
    finally:
        conn.close()

    ratings = []
    for team_id, rating in sorted(season_ratings.items(), key=lambda kv: kv[1], reverse=True):
        team_meta = meta.get(team_id, {})
        elo_7d_ago = elo_engine.elo_as_of(history["timelines"].get(team_id, []), cutoff_7d)
        ratings.append({
            "team": team_meta.get("team"),
            "abbr": team_meta.get("abbr"),
            "elo": round(rating, 1),
            "rank": len(ratings) + 1,
            "change_7d": round(rating - elo_7d_ago, 1),
            "wins": team_meta.get("wins"),
            "losses": team_meta.get("losses"),
            "net_rtg": team_meta.get("net_rtg"),
        })

    return {
        "season": season,
        "as_of": as_of,
        "params": {
            "k": int(elo_engine.K_FACTOR),
            "home_adv": int(elo_engine.HOME_ADVANTAGE),
            "mov": True,
            "carryover": elo_engine.SEASON_CARRYOVER,
            "history_start": elo_engine.HISTORY_START_SEASON,
        },
        "ratings": ratings,
    }

# --- Model backtest summary (served from the validated artifact at repo root) ---
BACKTEST_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results.json')
CANDIDATE_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results_candidate.json')
_backtest_summary_cache: Dict[str, Any] = {}

def _summary_from_candidate_artifact() -> Dict[str, Any]:
    """Summary of backtest_results_candidate.json — the sealed one-shot
    evaluation of the model now serving predictions (candidate_2026-08)."""
    with open(CANDIDATE_RESULTS_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    overall = artifact["results_candidate"]["overall"]
    old_overall = artifact["results_old_model_same_games"]["overall"]
    seasons = artifact["evaluation"]["seasons"]
    calibration = [
        {
            "bucket": row["bucket"].rstrip("%"),
            "n": row["n"],
            "predicted_pct": row["mean_predicted_pct"],
            "actual_pct": row["actual_win_pct"],
            # Additive: the artifact's own uncertainty figures, so the
            # calibration playground can show error bars instead of implying
            # a 20-game bucket is as settled as a 700-game one.
            "actual_95ci": row.get("actual_95ci"),
            "error_pp": row.get("calibration_error_pp"),
        }
        for row in artifact["results_candidate"]["calibration_by_confidence"]
    ]
    # The second reliability table in the artifact: by predicted HOME win
    # probability rather than by pick confidence. This is the one that carries
    # the failed pre-registration gate (40-50% bucket, -5.15pp) - shown, not
    # hidden.
    calibration_home = [
        {
            "bucket": row["bucket"].rstrip("%"),
            "n": row["n"],
            "predicted_pct": row["mean_predicted_home_win_pct"],
            "actual_pct": row["actual_home_win_pct"],
            "actual_95ci": row.get("actual_95ci"),
            "error_pp": row.get("calibration_error_pp"),
        }
        for row in artifact.get("calibration_reliability_home_prob", [])
    ]
    return {
        "headline": {
            "accuracy_pct": overall["model_accuracy_pct"],
            "n_games": overall["n"],
            "ci95": overall["model_accuracy_95ci"],
            "seasons": seasons,
        },
        "baselines": {
            "home_team_pct": overall["home_team_baseline_pct"],
            "better_record_pct": overall["better_record_baseline_pct"],
        },
        "previous_model": {
            "accuracy_pct": old_overall["model_accuracy_pct"],
            "mcnemar_p_vs_current": artifact["paired_tests"]["candidate_vs_old_model"]["p_value"],
        },
        "calibration": calibration,
        "calibration_home_prob": calibration_home,
        "generated_at": artifact["generated_at_utc"],
        "methodology_note": (
            f"Sealed, pre-registered, one-shot evaluation on {overall['n']:,} games from the "
            f"{' and '.join(seasons)} seasons, all outside the model's 2012-24 training window. "
            f"Statistically ahead of both the previous model ({old_overall['model_accuracy_pct']}%) "
            "and the better-record baseline. Accuracy only - no ROI has been measured."
        ),
    }


def _summary_from_legacy_artifact() -> Dict[str, Any]:
    """Summary of backtest_results.json (the previous production model)."""
    with open(BACKTEST_RESULTS_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    headline = artifact["headline"]
    baseline = artifact["baseline"]
    calibration = [
        {
            "bucket": row["bucket"].rstrip("%"),
            "n": row["n"],
            "predicted_pct": row["mean_predicted_pct"],
            "actual_pct": row["actual_win_pct"],
        }
        for row in artifact["results"]["calibration_by_confidence"]
    ]
    seasons = headline["seasons"]
    training_cutoff = artifact["model"]["training_date_range"][1]
    return {
        "headline": {
            "accuracy_pct": headline["number_pct"],
            "n_games": headline["n_games"],
            "ci95": headline["ci95_pct"],
            "seasons": seasons,
        },
        "baselines": {
            "home_team_pct": baseline["always_pick_home"]["accuracy_pct"],
            "better_record_pct": baseline["pick_better_win_pct"]["accuracy_pct"],
        },
        "calibration": calibration,
        "generated_at": artifact["generated_at_utc"],
        "methodology_note": (
            f"Measured on {headline['n_games']:,} games from the {' and '.join(seasons)} seasons, "
            f"all played after the model's training cutoff ({training_cutoff}); no historical odds "
            "exist for these seasons, so this measures accuracy only - no ROI has been measured."
        ),
    }


# --- Model diary --------------------------------------------------------------
# A public changelog of the model. Every entry is assembled from the sealed
# artifacts on disk rather than written by hand, because a page whose whole point
# is honesty cannot have its numbers typed in from memory.
#
# The two artifacts carry more than their own results: backtest_results.json
# records the claim it replaced ("68.9% test accuracy") and why that claim was
# wrong, so even the retired figure is sourced to a file rather than to a
# recollection. Nothing on this page is authored except the labels.
@app.get("/api/model/diary")
def get_model_diary():
    """
    Version history of the serving model, with each version's sealed result and
    its pre-registered gates - passes and failures alike.

    Read straight from backtest_results_candidate.json and backtest_results.json.
    If an artifact is missing its entry is omitted rather than approximated, and
    the response says which files it found.
    """
    entries: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    def load(path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("Could not read %s: %s", path, exc)
            return None

    cand = load(CANDIDATE_RESULTS_PATH)
    old = load(BACKTEST_RESULTS_PATH)

    # ---- current serving model -------------------------------------------
    if cand:
        sources.append({
            "file": os.path.basename(CANDIDATE_RESULTS_PATH),
            "generated_at": cand.get("generated_at_utc"),
            "role": "sealed evaluation of the serving model",
        })
        res = (cand.get("results_candidate") or {}).get("overall") or {}
        ev = cand.get("evaluation") or {}
        thresholds = cand.get("preregistered_thresholds") or {}

        gates = []
        for key, t in thresholds.items():
            if not isinstance(t, dict):
                continue
            gates.append({
                "id": key,
                "requirement": t.get("requirement"),
                "passed": bool(t.get("PASS")),
                # Whatever numbers the gate recorded, minus the requirement text,
                # so the page can show a gate's own evidence without the endpoint
                # deciding which figures matter.
                "detail": {k: v for k, v in t.items()
                           if k not in ("requirement", "PASS")},
            })

        entries.append({
            "version": "candidate_2026-08",
            "status": "serving",
            "serving_since": "2026-08-11",
            "sealed_at": cand.get("generated_at_utc"),
            "headline_pct": res.get("model_accuracy_pct"),
            "ci95": res.get("model_accuracy_95ci"),
            "n_games": ev.get("n_games_scored"),
            "seasons": ev.get("seasons"),
            "date_range": ev.get("date_range"),
            "baselines": {
                "home_team_pct": res.get("home_team_baseline_pct"),
                "better_record_pct": res.get("better_record_baseline_pct"),
                "better_point_margin_pct": res.get("better_point_margin_baseline_pct"),
            },
            "brier": res.get("brier_score"),
            "log_loss": res.get("log_loss"),
            "architecture": (cand.get("model") or {}).get("architecture"),
            "n_features": (cand.get("model") or {}).get("n_features"),
            "gates": gates,
            "gates_passed": sum(1 for g in gates if g["passed"]),
            "gates_total": len(gates),
            "calibration": cand.get("calibration_reliability_home_prob"),
            "methodology_notes": cand.get("methodology_notes"),
            "by_season": (cand.get("results_candidate") or {}).get("by_season"),
        })

    # ---- previous model ---------------------------------------------------
    if old:
        sources.append({
            "file": os.path.basename(BACKTEST_RESULTS_PATH),
            "generated_at": old.get("generated_at_utc"),
            "role": "evaluation of the previous model",
        })
        head = old.get("headline") or {}
        entries.append({
            "version": "previous production model",
            "status": "retired",
            "retired_on": "2026-08-11",
            "sealed_at": old.get("generated_at_utc"),
            "headline_pct": head.get("number_pct"),
            "ci95": head.get("ci95_pct"),
            "n_games": head.get("n_games"),
            "seasons": head.get("seasons"),
            # This artifact nests its baselines one level deeper and names them
            # differently from the candidate's. Reading the candidate's key names
            # against it returned None for both, which would have shown the
            # previous model with no baseline to judge it against.
            "baselines": {
                "home_team_pct": ((old.get("baseline") or {}).get("always_pick_home") or {}).get("accuracy_pct"),
                "better_record_pct": ((old.get("baseline") or {}).get("pick_better_win_pct") or {}).get("accuracy_pct"),
            },
            "gates": [],
            "gates_passed": None,
            "gates_total": None,
            # The most useful thing this artifact says is what it could NOT show.
            "caveats": old.get("caveats"),
            "methodology_notes": old.get("methodology_notes"),
        })

        # ---- the claim this replaced --------------------------------------
        if head.get("replaces_claim"):
            entries.append({
                "version": "withdrawn claim",
                "status": "withdrawn",
                "retired_on": old.get("generated_at_utc", "")[:10],
                "claim": head.get("replaces_claim"),
                "why_wrong": head.get("why_the_old_claim_is_wrong"),
                "gates": [],
            })

    if not entries:
        raise HTTPException(
            status_code=503,
            detail="No model artifacts on disk - nothing to publish.",
        )

    return {
        "entries": entries,
        "sources": sources,
        # Stated once, here, so no page has to remember it: accuracy is not
        # profitability and this project has never measured the latter.
        "no_roi_note": (
            "No return on investment has ever been measured. There is no closing-odds "
            "archive for these seasons, so profitability is unknown and no figure on "
            "this page implies it."
        ),
        "sealed_note": (
            "The test set is spent. It was scored once, and re-running the model against "
            "it after seeing the result would turn a held-out evaluation into a tuning "
            "set, so it will not be re-used."
        ),
    }


@app.get("/api/model/backtest")
def get_model_backtest():
    """
    Curated summary of the serving model's held-out evaluation. Prefers the
    sealed candidate artifact (the model serving since 2026-08-11); falls back
    to the previous model's backtest_results.json. Read once, cached per process.
    """
    if "summary" in _backtest_summary_cache:
        return _backtest_summary_cache["summary"]
    summary = None
    if os.path.exists(CANDIDATE_RESULTS_PATH):
        try:
            summary = _summary_from_candidate_artifact()
        except Exception as e:
            logger.error(f"Error reading candidate backtest artifact: {e}", exc_info=True)
    if summary is None and os.path.exists(BACKTEST_RESULTS_PATH):
        try:
            summary = _summary_from_legacy_artifact()
        except Exception as e:
            logger.error(f"Error reading backtest artifact: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail="Backtest artifact could not be parsed.")
    if summary is None:
        raise HTTPException(
            status_code=503,
            detail="No backtest artifact is present on this deployment; "
                   "run backtest_model.py to regenerate one.",
        )
    _backtest_summary_cache["summary"] = summary
    return summary

# --- Live scoreboard (official NBA live CDN; cheap, cached, keyless by design) ---
@app.get("/api/live/scoreboard")
def get_live_scoreboard():
    """
    Normalized live scoreboard from cdn.nba.com (30s in-memory cache). An
    off-season day or an unreachable CDN both yield games: [] - never an error.
    """
    return nba_live.get_scoreboard()

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
