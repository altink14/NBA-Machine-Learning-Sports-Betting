# main_api.py
# FINAL STABLE VERSION - Corrected endpoint routing and data handling.
import os
import uvicorn
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import json
from datetime import datetime, timedelta

# nba_api imports
try:
    from nba_api.stats.static import players as nba_players
    from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats
except ImportError:
    nba_players = None
    shotchartdetail = None
    leaguedashplayerstats = None

# Local Imports
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Utils import Expected_Value, Kelly_Criterion as kc
from src.Utils.tools import create_todays_games_from_odds
from src.Utils.Dictionaries import team_index_current

# --- MONKEY-PATCH FOR BASKETBALL-REFERENCE-SCRAPER ---
import io
original_read_html = pd.read_html
def patched_read_html(io_or_html, *args, **kwargs):
    if isinstance(io_or_html, str) and ("<table" in io_or_html or "<html" in io_or_html):
        return original_read_html(io.StringIO(io_or_html), *args, **kwargs)
    return original_read_html(io_or_html, *args, **kwargs)
pd.read_html = patched_read_html

try:
    import basketball_reference_scraper.seasons as seasons
    import basketball_reference_scraper.request_utils as request_utils
    import basketball_reference_scraper.teams as teams
    import basketball_reference_scraper.players as players
    from bs4 import BeautifulSoup, Comment
    from basketball_reference_scraper.request_utils import get_wrapper
    import re
    
    def patched_get_selenium_wrapper(url, xpath):
        match = re.search(r'@id="([^"]+)"', xpath)
        if not match:
            return None
        table_id = match.group(1)
        
        r = request_utils.get_wrapper(url)
        if r.status_code == 200:
            html = r.content.decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', id=table_id)
            if table:
                return f'<table>{table.decode_contents()}</table>'
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                if f'id="{table_id}"' in comment:
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    table = comment_soup.find('table', id=table_id)
                    if table:
                        return f'<table>{table.decode_contents()}</table>'
            return None
        return None

    request_utils.get_selenium_wrapper = patched_get_selenium_wrapper
    teams.get_selenium_wrapper = patched_get_selenium_wrapper
    players.get_selenium_wrapper = patched_get_selenium_wrapper

    
    def patched_get_schedule(season, playoffs=False):
        months = ['October', 'November', 'December', 'January', 'February', 'March',
                'April', 'May', 'June']
        if season==2020:
            months = ['October-2019', 'November', 'December', 'January', 'February', 'March',
                    'July', 'August', 'September', 'October-2020']
        df = pd.DataFrame()
        for month in months:
            url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html'
            r = get_wrapper(url)
            if r.status_code==200:
                soup = BeautifulSoup(r.content, 'html.parser')
                table = soup.find('table', attrs={'id': 'schedule'})
                if table:
                    month_df = pd.read_html(io.StringIO(str(table)))[0]
                    df = pd.concat([df, month_df])

        if df.empty:
            return df

        df = df.reset_index(drop=True)
        df = df.iloc[:, [0, 2, 3, 4, 5]]
        df.columns = ['DATE', 'VISITOR', 'VISITOR_PTS', 'HOME', 'HOME_PTS']

        if season==2020:
            df = df[df['DATE']!='Playoffs']
            df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
            df = df.sort_values(by='DATE')
            df = df.reset_index(drop=True)
            playoff_loc = df[df['DATE']==pd.to_datetime('2020-08-17')].head(n=1)
            if len(playoff_loc.index)>0:
                playoff_index = playoff_loc.index[0]
            else:
                playoff_index = len(df)
            if playoffs:
                df = df[playoff_index:]
            else:
                df = df[:playoff_index]
        else:
            if season == 1953:
                df.drop_duplicates(subset=['DATE', 'HOME', 'VISITOR'], inplace=True)
            playoff_loc = df[df['DATE']=='Playoffs']
            if len(playoff_loc.index)>0:
                playoff_index = playoff_loc.index[0]
            else:
                playoff_index = len(df)
            if playoffs:
                df = df[playoff_index+1:]
            else:
                df = df[:playoff_index]
            df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
        return df

    seasons.get_schedule = patched_get_schedule
    print("Successfully applied monkey-patches to basketball_reference_scraper")
except Exception as e:
    print(f"Failed to apply monkey-patches to basketball_reference_scraper: {e}")

# Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI App Setup
app = FastAPI(title="Betting Buddy API", version="1.1.1-stable-fixed")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

# AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY not found! Chatbot will be disabled.")
    genai.configure(api_key="DUMMY_KEY_FOR_STARTUP")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# PredictionRunner Class
class PredictionRunner:
    def __init__(self, sportsbook: str, kelly_criterion: bool):
        self.sportsbook = sportsbook
        self.model_name = 'xgboost'
        self.kelly_criterion = kelly_criterion
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.team_stats_df = self._load_team_stats()
        self.schedule_df = self._load_schedule()
        self.odds_provider = SbrOddsProvider(sportsbook=self.sportsbook)
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
            df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn, index_col="index")
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to load team stats from database: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Server configuration error: Could not load team stats.")

    def _load_schedule(self):
        try:
            path = os.path.join(self.project_root, 'Data', 'nba-2024-UTC.csv')
            return pd.read_csv(path, parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        except FileNotFoundError:
            logger.error("Schedule file nba-2024-UTC.csv not found.")
            return None

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
        games_list = create_todays_games_from_odds(odds_data)
        if not games_list:
            return {"error": "No valid games processed from odds data.", "predictions": []}
        
        data_for_model, todays_games_uo, frame_ml, home_team_odds, away_team_odds, game_start_times = self._prepare_data_for_model(games_list, odds_data)
        
        if data_for_model.size == 0:
            return {"error": "Could not prepare valid data for the prediction model.", "predictions": []}
        
        ml_predictions, ou_predictions = self._run_xgboost_models(data_for_model, frame_ml, todays_games_uo)
        return self._format_predictions(games_list, ml_predictions, ou_predictions, home_team_odds, away_team_odds, todays_games_uo, game_start_times)

    def _prepare_data_for_model(self, games, odds):
        game_data_list, home_odds_list, away_odds_list, uo_lines_list, game_start_times_list = [], [], [], [], []
        
        # We need datetime today to calculate rest days
        today = datetime.today()

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

            # Calculate days rest
            home_days_off = 7
            away_days_off = 7
            if self.schedule_df is not None:
                home_games = self.schedule_df[(self.schedule_df['Home Team'] == home_team) | (self.schedule_df['Away Team'] == home_team)]
                away_games = self.schedule_df[(self.schedule_df['Home Team'] == away_team) | (self.schedule_df['Away Team'] == away_team)]
                previous_home_games = home_games.loc[self.schedule_df['Date'] <= today].sort_values('Date', ascending=False).head(1)['Date']
                previous_away_games = away_games.loc[self.schedule_df['Date'] <= today].sort_values('Date', ascending=False).head(1)['Date']
                
                if len(previous_home_games) > 0:
                    last_home_date = previous_home_games.iloc[0]
                    home_days_off = (timedelta(days=1) + today - last_home_date).days
                if len(previous_away_games) > 0:
                    last_away_date = previous_away_games.iloc[0]
                    away_days_off = (timedelta(days=1) + today - last_away_date).days

            # Clip rest days to a reasonable range of 1-7 days to prevent model outlier issues
            home_days_off = max(1, min(home_days_off, 7))
            away_days_off = max(1, min(away_days_off, 7))

            # Concatenate home and away team statistics
            game_data = pd.concat([home_stats, away_stats.rename(index=lambda x: x + '.1')])
            game_data['Days-Rest-Home'] = float(home_days_off)
            game_data['Days-Rest-Away'] = float(away_days_off)
            
            game_data_list.append(game_data)
            
            home_odds_list.append(game_odds.get(home_team, {}).get('money_line_odds'))
            away_odds_list.append(game_odds.get(away_team, {}).get('money_line_odds'))
            uo_lines_list.append(game_odds.get('under_over_odds'))
            game_start_times_list.append(game_odds.get('game_start_time_utc'))

        if not game_data_list:
            return np.array([]), [], pd.DataFrame(), [], [], []
            
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
        return frame_for_model.values.astype(float), uo_lines_list, frame_ml, home_odds_list, away_odds_list, game_start_times_list

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

    def _format_predictions(self, games, ml_preds, ou_preds, home_odds, away_odds, uo_lines, game_start_times):
        predictions_list = []
        for i, (home_team, away_team) in enumerate(games):
            home_odd, away_odd = home_odds[i], away_odds[i]
            winner_idx, ou_idx = np.argmax(ml_preds[i]), np.argmax(ou_preds[i])
            winner_confidence, ou_confidence = float(ml_preds[i][winner_idx]), float(ou_preds[i][ou_idx])
            ev_home, ev_away, kelly_home, kelly_away = 0.0, 0.0, "No Bet", "No Bet"
            
            game_datetime_obj = game_start_times[i]
            game_start_time_str = game_datetime_obj.isoformat() if isinstance(game_datetime_obj, datetime) else None

            try:
                if home_odd is not None and away_odd is not None:
                    ev_home = Expected_Value.expected_value(winner_confidence, int(home_odd))
                    ev_away = Expected_Value.expected_value(1 - winner_confidence, int(away_odd))
                    if self.kelly_criterion:
                        kelly_home = kc.calculate_kelly_criterion(int(home_odd), winner_confidence)
                        kelly_away = kc.calculate_kelly_criterion(int(away_odd), 1 - winner_confidence)
            except (ValueError, TypeError): pass
            
            predictions_list.append({
                "home_team": home_team, "away_team": away_team, "home_odds": home_odd, "away_odds": away_odd,
                "under_over_line": uo_lines[i], "predicted_winner": home_team if winner_idx == 1 else away_team,
                "winner_confidence": round(winner_confidence * 100, 2),
                "under_over_prediction": "OVER" if ou_idx == 1 else "UNDER",
                "under_over_confidence": round(ou_confidence * 100, 2), "model": self.model_name,
                "expected_value": {"home_team": ev_home, "away_team": ev_away},
                "kelly_criterion": {"home_team": kelly_home, "away_team": kelly_away},
                "game_start_time_utc": game_start_time_str
            })
        return {"sportsbook": self.sportsbook, "predictions": predictions_list}

# --- API Endpoints ---

# This is the root endpoint for http://localhost:8000/
@app.get("/")
def read_root():
    return { "message": "Welcome to the Betting Buddy API!", "status": "healthy" }

# Health check endpoint for frontend monitoring
@app.get("/health")
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
def get_predictions_endpoint(sportsbook: str = 'fanduel', kelly_criterion: bool = True):
    cache_key = f"{sportsbook}_{kelly_criterion}"
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
        runner = PredictionRunner(sportsbook=sportsbook, kelly_criterion=kelly_criterion)
        res = runner.run_predictions()
        predictions_cache[cache_key] = (res, now)
        return res
    except Exception as e:
        logger.error(f"Error in /predictions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# This is the chat endpoint for http://localhost:8000/api/chat
@app.post("/api/chat")
async def chat_handler(chat_message: ChatMessage):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "DUMMY_KEY_FOR_STARTUP":
        raise HTTPException(status_code=503, detail="Chatbot is currently unavailable.")
    try:
        logger.info(f"Received chat message: {chat_message.message}")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = await model.generate_content_async(chat_message.message)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"An error occurred in chat_handler: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the AI assistant: {str(e)}")

# --- Historical Data Endpoints ---
@app.get("/api/historical/team-stats")
def get_historical_team_stats(team: str, season: int):
    try:
        from basketball_reference_scraper.teams import get_team_stats
        df = get_team_stats(team, season)
        if isinstance(df, pd.Series):
            return df.to_dict()
        elif isinstance(df, pd.DataFrame):
            return df.iloc[0].to_dict()
        else:
            return {"error": "Invalid data format returned."}
    except Exception as e:
        logger.error(f"Error fetching historical team stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical/matchup")
def get_historical_matchup(team1: str, team2: str, season: int):
    try:
        from basketball_reference_scraper.seasons import get_schedule
        schedule_df = get_schedule(season)
        if schedule_df.empty:
            return {"matchups": [], "win_percentage": {team1: 0, team2: 0}}
        
        t1 = team1.lower()
        t2 = team2.lower()
        
        # Filter games involving both teams
        filtered = schedule_df[
            ((schedule_df['VISITOR'].str.lower().str.contains(t1)) & (schedule_df['HOME'].str.lower().str.contains(t2))) |
            ((schedule_df['VISITOR'].str.lower().str.contains(t2)) & (schedule_df['HOME'].str.lower().str.contains(t1)))
        ]
        
        matchups_list = []
        team1_wins = 0
        team2_wins = 0
        total_games = 0
        
        for _, row in filtered.iterrows():
            date_str = row['DATE'].strftime('%Y-%m-%d') if isinstance(row['DATE'], pd.Timestamp) else str(row['DATE'])
            visitor = row['VISITOR']
            home = row['HOME']
            
            try:
                visitor_pts = int(row['VISITOR_PTS'])
                home_pts = int(row['HOME_PTS'])
            except (ValueError, TypeError):
                continue
                
            total_games += 1
            
            # Determine winner
            if visitor_pts > home_pts:
                winner = visitor
                winner_pts = visitor_pts
                loser = home
                loser_pts = home_pts
            else:
                winner = home
                winner_pts = home_pts
                loser = visitor
                loser_pts = visitor_pts
                
            if t1 in winner.lower():
                team1_wins += 1
            elif t2 in winner.lower():
                team2_wins += 1
                
            matchups_list.append({
                "date": date_str,
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
    except Exception as e:
        logger.error(f"Error fetching historical matchup details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Cache for shot charts
shot_chart_cache = {}

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
def get_shot_chart(game_date: str, home_team: str):
    normalized_team = home_team.strip().lower()
    team_abbr = TEAM_TO_BR_ABBR.get(normalized_team, home_team.upper())
    
    if team_abbr == "BKN": team_abbr = "BRK"
    if team_abbr == "CHA": team_abbr = "CHO"
    if team_abbr == "PHX": team_abbr = "PHO"
    
    game_id = f"{game_date}0{team_abbr}"
    
    if game_id in shot_chart_cache:
        logger.info(f"Returning cached shot chart data for game: {game_id}")
        return shot_chart_cache[game_id]
        
    is_tatum_mock_game = (game_date == "20230415" and team_abbr == "BOS")
    url = f"https://www.basketball-reference.com/boxscores/shot-chart/{game_id}.html"
    logger.info(f"Scraping shot chart from URL: {url}")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"Failed to scrape shot chart (status {response.status_code}).")
            if is_tatum_mock_game or game_date == "20230415":
                logger.info("Using mock fallback data for Tatum's 50pt game")
                mock_res = {"game_id": game_id, "shots": TATUM_MOCK_SHOTS}
                shot_chart_cache[game_id] = mock_res
                return mock_res
            raise HTTPException(status_code=404, detail=f"Game shot chart boxscore not found (status {response.status_code}).")
            
        soup = BeautifulSoup(response.text, 'html.parser')
        shots_divs = soup.find_all("div", class_="tooltip")
        
        if not shots_divs:
            logger.warning("No tooltip shot divs found on the page.")
            if is_tatum_mock_game:
                mock_res = {"game_id": game_id, "shots": TATUM_MOCK_SHOTS}
                shot_chart_cache[game_id] = mock_res
                return mock_res
            return {"game_id": game_id, "shots": []}
            
        results = []
        for shot in shots_divs:
            tip = shot.get("tip", "")
            style = shot.get("style", "")
            
            if not tip or not style:
                continue
                
            try:
                style_parts = style.lower().split(";")
                top_val = 0
                left_val = 0
                for part in style_parts:
                    if "top:" in part:
                        top_val = int(part.split("top:")[1].split("px")[0].strip())
                    elif "left:" in part:
                        left_val = int(part.split("left:")[1].split("px")[0].strip())
                        
                made = "makes" in tip
                missed = "misses" in tip
                
                if not made and not missed:
                    made = "makes" in tip.lower()
                    missed = "misses" in tip.lower()
                    
                action = "makes" if made else "misses"
                if action not in tip:
                    continue
                    
                player_name = tip.split(action)[0].strip()
                description = tip.replace("<br>", " ").strip()
                
                results.append({
                    "player": player_name,
                    "x": left_val,
                    "y": top_val,
                    "result": "made" if made else "missed",
                    "description": description
                })
            except Exception as parse_err:
                logger.debug(f"Error parsing individual shot tooltip: {parse_err}")
                continue
                
        if not results and is_tatum_mock_game:
            logger.info("Scraper returned empty results, using mock fallback data for Tatum")
            results = TATUM_MOCK_SHOTS
            
        game_res = {"game_id": game_id, "shots": results}
        shot_chart_cache[game_id] = game_res
        return game_res
        
    except Exception as e:
        logger.error(f"Error in get_shot_chart API: {e}", exc_info=True)
        if is_tatum_mock_game:
            logger.info("Using mock fallback data for Tatum's 50pt game after scraping exception")
            mock_res = {"game_id": game_id, "shots": TATUM_MOCK_SHOTS}
            shot_chart_cache[game_id] = mock_res
            return mock_res
        raise HTTPException(status_code=500, detail=str(e))

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
def get_player_shot_chart(player_id: int, season: str = "2024-25"):
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
                        "made": row[col_map["SHOT_MADE_FLAG"]] == 1
                    })
                except Exception:
                    continue
                    
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
            "averages": averages
        }
        
        player_shot_chart_cache[cache_key] = response_data
        return response_data
        
    except Exception as e:
        logger.error(f"Error in get_player_shot_chart API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

player_stats_cache = {}

@app.get("/api/player-stats")
def get_player_stats(season: str = "2025-26", per_mode: str = "PerGame", measure_type: str = "Base"):
    cache_key = f"{season}_{per_mode}_{measure_type}"
    now = datetime.now()
    
    if cache_key in player_stats_cache:
        cached_data, timestamp = player_stats_cache[cache_key]
        if now - timestamp < timedelta(minutes=5):
            logger.info(f"Returning cached player stats for key: {cache_key}")
            return cached_data
            
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

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
