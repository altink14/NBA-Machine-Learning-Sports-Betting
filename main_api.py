from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Added for ChatMessage
from typing import Optional, Dict, Any, List
import pandas as pd
# import tensorflow as tf # Commented out for now
from datetime import datetime, timedelta
import json
import logging
import os
import numpy as np

# --- Chatbot Specific Imports ---
import google.generativeai as genai
from dotenv import load_dotenv # To load .env file for API keys

# Assuming these src files are in a 'src' subdirectory relative to main_api.py
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import (
    create_todays_games_from_odds,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games
)

# --- Load environment variables for API keys ---
load_dotenv() # Looks for a .env file in the current directory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NBA Sports Betting Predictions API",
    description="Machine learning predictions for NBA games with odds from various sportsbooks and an AI Chatbot",
    version="1.0.1" # Incremented version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables! Chatbot functionality will be disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using the latest flash model
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}", exc_info=True)
        gemini_model = None # Ensure model is None if config fails

# Constants
TODAYS_GAMES_URL = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
DATA_URL = ('https://stats.nba.com/stats/leaguedashteamstats?'
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&'
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&'
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&'
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&'
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&'
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&'
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision=')

SUPPORTED_SPORTSBOOKS = [
    "fanduel", "draftkings", "betmgm", "pointsbet",
    "caesars", "wynn", "bet_rivers_ny"
]

# (create_todays_games_data function remains the same as your last version - I'll omit it here for brevity but ensure it's in your file)
def create_todays_games_data(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    for game_idx, game_teams in enumerate(games):
        home_team = game_teams[0]
        away_team = game_teams[1]

        if home_team not in team_index_current or away_team not in team_index_current:
            logger.warning(f"Team index not found for game: {away_team} @ {home_team}. Skipping.")
            continue

        current_game_odds = None
        if odds is not None:
            game_key = f"{home_team}:{away_team}"
            if game_key in odds:
                current_game_odds = odds[game_key]
                todays_games_uo.append(current_game_odds.get('under_over_odds', "N/A"))
                home_team_odds.append(current_game_odds.get(home_team, {}).get('money_line_odds', "N/A"))
                away_team_odds.append(current_game_odds.get(away_team, {}).get('money_line_odds', "N/A"))
            else:
                logger.warning(f"Odds not found for game_key: {game_key}. Appending N/A for odds.")
                todays_games_uo.append("N/A")
                home_team_odds.append("N/A")
                away_team_odds.append("N/A")
        else:
            logger.warning("Odds object is None. Appending N/A for odds.")
            todays_games_uo.append("N/A")
            home_team_odds.append("N/A")
            away_team_odds.append("N/A")

        home_days_val = 2
        away_days_val = 2
        try:
            project_root_dir = os.path.dirname(os.path.abspath(__file__))
            schedule_path = os.path.join(project_root_dir, 'Data', 'nba-2024-UTC.csv')
            if not os.path.exists(schedule_path):
                 logger.warning(f"Schedule file not found at {os.path.abspath(schedule_path)}. Using default days rest.")
            else:
                schedule_df = pd.read_csv(schedule_path, parse_dates=['Date'], dayfirst=True)
                today_dt = datetime.now()
                home_games_played = schedule_df[((schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)) & (schedule_df['Date'] < today_dt)]
                away_games_played = schedule_df[((schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)) & (schedule_df['Date'] < today_dt)]

                if not home_games_played.empty:
                    last_home_date = home_games_played.sort_values('Date', ascending=False).iloc[0]['Date']
                    home_days_val = (today_dt - last_home_date).days
                else:
                    home_days_val = 7
                if not away_games_played.empty:
                    last_away_date = away_games_played.sort_values('Date', ascending=False).iloc[0]['Date']
                    away_days_val = (today_dt - last_away_date).days
                else:
                    away_days_val = 7
        except Exception as e:
            logger.error(f"Error calculating days rest for {away_team} @ {home_team}: {e}. Using default days rest.", exc_info=True)

        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats_dict = pd.concat([home_team_series, away_team_series]).to_dict()
        stats_dict['Days-Rest-Home'] = home_days_val
        stats_dict['Days-Rest-Away'] = away_days_val
        match_data.append(pd.Series(stats_dict))

    if not match_data:
        logger.info("No match data could be processed after filtering.")
        return np.array([]), [], pd.DataFrame(), [], []

    games_data_frame = pd.DataFrame(match_data)
    cols_to_drop = [col for col in ['TEAM_ID', 'TEAM_NAME'] if col in games_data_frame.columns]
    
    data_np = np.array([])
    frame_ml = pd.DataFrame()
    if not games_data_frame.empty:
        frame_ml = games_data_frame.drop(columns=cols_to_drop, errors='ignore')
        try:
            data_np = frame_ml.values.astype(float)
        except ValueError as ve:
            logger.error(f"Could not convert all frame_ml values to float: {ve}", exc_info=True)
            return np.array([]), todays_games_uo, frame_ml, home_team_odds, away_team_odds
            
    return data_np, todays_games_uo, frame_ml, home_team_odds, away_team_odds

# (PredictionRunner class with run_xgboost and run_neural_network remains the same - I'll omit it here for brevity)
class PredictionRunner:
    @staticmethod
    def run_xgboost(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, use_kc=False):
        import copy
        import xgboost as xgb
        from src.Utils import Expected_Value
        from src.Utils import Kelly_Criterion as kc

        try:
            project_root = os.path.dirname(os.path.abspath(__file__))

            model_path_ml = os.path.join(project_root, 'Models', 'XGBoost_Models', 'XGBoost_68.9%_ML-3.json')
            if not os.path.exists(model_path_ml):
                raise FileNotFoundError(f"ML Model file not found: {model_path_ml}")
            xgb_ml = xgb.Booster()
            xgb_ml.load_model(model_path_ml)

            model_path_uo = os.path.join(project_root, 'Models', 'XGBoost_Models', 'XGBoost_54.8%_UO-8.json')
            if not os.path.exists(model_path_uo):
                raise FileNotFoundError(f"UO Model file not found: {model_path_uo}")
            xgb_uo = xgb.Booster()
            xgb_uo.load_model(model_path_uo)

            if data.size == 0:
                logger.warning("Input data for XGBoost is empty. No predictions will be made.")
                return []

            ml_predictions_array = []
            for row_idx in range(data.shape[0]):
                row_data = data[row_idx, :]
                if not isinstance(row_data, np.ndarray): row_data = np.array(row_data)
                if row_data.ndim == 1: row_data = row_data.reshape(1, -1)
                ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(row_data)))

            uo_data_list = []
            if not frame_ml.empty:
                frame_uo = copy.deepcopy(frame_ml)
                if len(todays_games_uo) == len(frame_uo):
                    frame_uo['OU'] = np.asarray(todays_games_uo)
                else:
                    logger.warning(f"Mismatch UO lines ({len(todays_games_uo)}) and games ({len(frame_uo)}). Using NaN for OU.")
                    frame_uo['OU'] = np.nan
                
                uo_data_from_frame = frame_uo.values
                for row in uo_data_from_frame:
                    uo_data_list.append(row.astype(float))
            else:
                logger.warning("Frame_ml for UO predictions is empty.")

            ou_predictions_array = []
            if uo_data_list:
                for row_data_uo in uo_data_list:
                    if not isinstance(row_data_uo, np.ndarray): row_data_uo = np.array(row_data_uo)
                    if row_data_uo.ndim == 1: row_data_uo = row_data_uo.reshape(1, -1)
                    if np.isnan(row_data_uo).any():
                        logger.warning(f"NaNs found in UO data row. Appending default OU prediction.")
                        ou_predictions_array.append(np.array([[0.5, 0.5]]))
                    else:
                        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(row_data_uo)))
            else:
                 ou_predictions_array = [np.array([[0.5, 0.5]]) for _ in range(len(games))]

            predictions = []
            for count, game_teams in enumerate(games):
                home_team = game_teams[0]
                away_team = game_teams[1]

                if count >= len(ml_predictions_array) or count >= len(ou_predictions_array):
                    logger.warning(f"Prediction array out of bounds for game count {count}. Game: {away_team} @ {home_team}")
                    continue

                winner_confidence_raw = ml_predictions_array[count][0]
                winner_idx = int(np.argmax(winner_confidence_raw))

                un_confidence_raw = ou_predictions_array[count][0]
                under_over_idx = int(np.argmax(un_confidence_raw))

                predicted_winner = home_team if winner_idx == 1 else away_team
                winner_confidence_pct = float(round(winner_confidence_raw[winner_idx] * 100, 1))

                uo_prediction = "OVER" if under_over_idx == 1 else "UNDER"
                uo_confidence_pct = float(round(un_confidence_raw[under_over_idx] * 100, 1))

                ev_home, ev_away, kc_home, kc_away = 0.0, 0.0, 0.0, 0.0
                current_home_odds_str = home_team_odds[count] if count < len(home_team_odds) else "N/A"
                current_away_odds_str = away_team_odds[count] if count < len(away_team_odds) else "N/A"

                try:
                    if current_home_odds_str != "N/A":
                        current_home_odds_int = int(current_home_odds_str)
                        ev_home = float(Expected_Value.expected_value(winner_confidence_raw[1], current_home_odds_int))
                        if use_kc:
                            kc_home = float(round(kc.calculate_kelly_criterion(current_home_odds_int, winner_confidence_raw[1]) * 100, 2))
                    if current_away_odds_str != "N/A":
                        current_away_odds_int = int(current_away_odds_str)
                        ev_away = float(Expected_Value.expected_value(winner_confidence_raw[0], current_away_odds_int))
                        if use_kc:
                            kc_away = float(round(kc.calculate_kelly_criterion(current_away_odds_int, winner_confidence_raw[0]) * 100, 2))
                except ValueError:
                    logger.warning(f"Could not parse odds for EV/KC for game {away_team} @ {home_team}. Odds H:{current_home_odds_str}, A:{current_away_odds_str}")

                prediction_item = {
                    "away_team": away_team, "home_team": home_team,
                    "away_odds": current_away_odds_str, "home_odds": current_home_odds_str,
                    "predicted_winner": predicted_winner, "winner_confidence": winner_confidence_pct,
                    "under_over_line": todays_games_uo[count] if count < len(todays_games_uo) else "N/A",
                    "under_over_prediction": uo_prediction, "under_over_confidence": uo_confidence_pct,
                    "expected_value": {"home_team": ev_home, "away_team": ev_away},
                    "model": "XGBoost"
                }
                if use_kc:
                    prediction_item["kelly_criterion"] = {"home_team": kc_home, "away_team": kc_away}
                predictions.append(prediction_item)
            return predictions
        except xgb.core.XGBoostError as xgb_err:
            logger.error(f"XGBoost core error: {str(xgb_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"XGBoost core error: {str(xgb_err)}")
        except FileNotFoundError as fnf_err:
            logger.error(f"Model file not found: {str(fnf_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Critical model file not found: {str(fnf_err)}")
        except Exception as e:
            logger.error(f"Unexpected XGBoost prediction error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected XGBoost prediction error: {str(e)}")

    @staticmethod
    def run_neural_network(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, use_kc=False):
        logger.warning("run_neural_network called, but NN models are currently unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Neural Network models are currently unavailable or TensorFlow/Keras is not installed. Please use XGBoost model instead."
        )

# (@app.get("/") and @app.get("/health") and @app.get("/predictions") remain the same - omitted for brevity)
@app.get("/")
async def root():
    return {
        "message": "NBA Sports Betting Predictions API",
        "endpoints": {
            "/predictions": "Get predictions for today's games",
            "/health": "Health check",
            "/docs": "API documentation",
            "/api/chat": "Chat with the Betting Buddy AI" # Added chat endpoint
        },
        "supported_sportsbooks": SUPPORTED_SPORTSBOOKS
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/predictions")
async def get_predictions(
    sportsbook: str = Query(..., description="Sportsbook to fetch odds from"),
    model: str = Query("xgboost", description="Model to use: xgboost, neural_network, or all"),
    kelly_criterion: bool = Query(False, description="Include Kelly Criterion calculations")
) -> Dict[str, Any]:

    if sportsbook not in SUPPORTED_SPORTSBOOKS:
        raise HTTPException(status_code=400, detail=f"Unsupported sportsbook. Supported: {', '.join(SUPPORTED_SPORTSBOOKS)}")
    if model not in ["xgboost", "neural_network", "all"]:
        raise HTTPException(status_code=400, detail="Model must be one of: xgboost, neural_network, all")

    try:
        odds_provider = SbrOddsProvider(sportsbook=sportsbook)
        odds = odds_provider.get_odds()
        if not odds:
            logger.warning(f"No odds data found for today from sportsbook: {sportsbook}")
            return {
                 "timestamp": datetime.now().isoformat(), "sportsbook": sportsbook, "total_games": 0,
                 "odds_data": {}, "predictions": [],
                 "message": "No odds data found for today. Cannot generate predictions."
            }

        games = create_todays_games_from_odds(odds)
        if not games:
            logger.warning("No games could be created from the fetched odds data.")
            return {
                 "timestamp": datetime.now().isoformat(), "sportsbook": sportsbook, "total_games": 0,
                 "odds_data": odds, "predictions": [],
                 "message": "No games could be structured from the odds data."
            }

        team_stats_data_json = get_json_data(DATA_URL)
        if not team_stats_data_json:
            logger.error("Failed to fetch team stats data from NBA_API.")
            raise HTTPException(status_code=503, detail="Failed to fetch team statistics data.")
        df_team_stats = to_data_frame(team_stats_data_json)

        data_for_model, todays_games_uo, frame_ml, home_team_odds_list, away_team_odds_list = create_todays_games_data(games, df_team_stats, odds)

        if data_for_model.size == 0 and not frame_ml.empty:
             logger.warning("Data for model is empty after processing, but frame_ml was not. Check type conversions or filtering.")
        elif data_for_model.size == 0:
            logger.warning("No valid game data could be processed for model input (data_for_model is empty).")
            return {
                 "timestamp": datetime.now().isoformat(), "sportsbook": sportsbook, "total_games": len(games),
                 "odds_data": odds, "predictions": [],
                 "message": "No valid game data could be processed for model input."
            }

        api_response = {
            "timestamp": datetime.now().isoformat(), "sportsbook": sportsbook,
            "total_games": len(games), "odds_data": {}, "predictions": []
        }

        if odds:
            for game_key, game_odds_detail in odds.items():
                ht, at = game_key.split(":")
                api_response["odds_data"][game_key] = {
                    "away_team": at, "home_team": ht,
                    "away_money_line": game_odds_detail.get(at, {}).get('money_line_odds', "N/A"),
                    "home_money_line": game_odds_detail.get(ht, {}).get('money_line_odds', "N/A"),
                    "under_over": game_odds_detail.get('under_over_odds', "N/A")
                }
        
        if model == "xgboost" or model == "all":
            xgb_predictions_list = PredictionRunner.run_xgboost(
                data_for_model, todays_games_uo, frame_ml, games,
                home_team_odds_list, away_team_odds_list, kelly_criterion
            )
            if model == "all":
                api_response["predictions"] = {"xgboost": xgb_predictions_list}
            else:
                api_response["predictions"] = xgb_predictions_list

        if model == "neural_network" or model == "all":
            try:
                nn_predictions_list = PredictionRunner.run_neural_network(
                    data_for_model, todays_games_uo, frame_ml, games,
                    home_team_odds_list, away_team_odds_list, kelly_criterion
                )
                if model == "all":
                    if "predictions" not in api_response or not isinstance(api_response["predictions"], dict):
                         api_response["predictions"] = {} # Initialize if xgb wasn't run
                    api_response["predictions"]["neural_network"] = nn_predictions_list
                else:
                    api_response["predictions"] = nn_predictions_list
            except HTTPException as http_exc:
                if model == "all":
                    if "predictions" not in api_response or not isinstance(api_response["predictions"], dict):
                        api_response["predictions"] = {}
                    api_response["predictions"]["neural_network_error"] = http_exc.detail
                else:
                    raise http_exc

        return api_response

    except HTTPException as http_e:
        logger.warning(f"HTTPException in get_predictions: {http_e.detail}")
        raise http_e
    except Exception as e:
        logger.error(f"Internal server error in get_predictions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sportsbooks")
async def get_supported_sportsbooks():
    return {"supported_sportsbooks": SUPPORTED_SPORTSBOOKS}


# --- Chatbot Pydantic Model ---
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

# --- UPDATED Chatbot System Prompt ---
CHATBOT_SYSTEM_PROMPT = """You are Betting Buddy, an expert NBA sports betting advisor.
Your goal is to provide helpful, data-driven insights and recommendations for NBA games.
You are friendly, analytical, and always promote responsible gambling.

When I provide you with 'Current game predictions based on our XGBoost model' in the context, please USE that specific information to answer user questions about those games.
- If the user asks for a general prediction or "best bet," summarize the available predictions, highlighting any with strong positive Expected Value (EV) or clear Kelly Criterion suggestions.
- If the user asks about a specific game (e.g., "Lakers game"), focus your answer on the prediction for that game if it's in the provided context.
- Clearly state the predicted winner and its confidence percentage.
- Clearly state the Over/Under pick, the line, and its confidence percentage.
- Mention the Expected Value (EV) for both teams in the matchup. Explain that positive EV is generally favorable.
- If Kelly Criterion percentages are provided and are not "No Bet", mention them as a suggestion for bankroll allocation, but also advise caution and suggest users might consider fractional Kelly.
- If no predictions are available in the context for a specific query, or if the context says predictions couldn't be fetched, inform the user clearly.

If asked about betting terms (e.g., "What is EV?"), explain them simply.
Always include a disclaimer about the risks of betting if providing any betting-related information or predictions.
If you cannot answer or don't have specific information, state that clearly.
Keep responses concise but informative. Be a 'devil's advocate' by mentioning potential risks or why a bet might not be straightforward. Try to put yourself in the user's shoes, considering they want to make informed betting decisions.
Do not invent games or predictions if they are not provided in the context.
"""

# --- UPDATED Chatbot Endpoint with Prediction Context ---
@app.post("/api/chat")
async def chat_with_bot(chat_message: ChatMessage) -> Dict[str, Any]:
    if not gemini_model:
        logger.error("Chatbot endpoint called, but Gemini model is not available (API key or config issue).")
        raise HTTPException(status_code=503, detail="Chatbot is currently unavailable due to configuration issues.")

    user_input_message = chat_message.message
    user_message_lower = user_input_message.lower()
    
    game_predictions_context_str = "" # Initialize context string

    # Basic intent recognition: Check if the user is asking for predictions or about games
    prediction_keywords = ["predict", "pick", "game", "match", "odds for", "bet on", "tonight", "today", "recommendation"]
    is_prediction_query = any(keyword in user_message_lower for keyword in prediction_keywords)

    if is_prediction_query:
        logger.info(f"Chatbot: User query '{user_input_message}' identified as potentially prediction-related.")
        # For simplicity, always try to fetch predictions if keywords match.
        # More advanced: parse specific teams from user_message to filter predictions.
        requested_sportsbook = "fanduel" # Default, or could be parsed from user_message or session
        
        try:
            logger.info(f"Chatbot: Attempting to fetch predictions for context from sportsbook: {requested_sportsbook}")
            odds_provider = SbrOddsProvider(sportsbook=requested_sportsbook)
            odds = odds_provider.get_odds()

            if not odds:
                game_predictions_context_str = f"\n\nContext: Could not fetch current odds from {requested_sportsbook} to generate predictions."
            else:
                games = create_todays_games_from_odds(odds)
                if not games:
                    game_predictions_context_str = "\n\nContext: No NBA games found from the odds provider for today."
                else:
                    team_stats_data_json = get_json_data(DATA_URL)
                    if not team_stats_data_json:
                         game_predictions_context_str = "\n\nContext: Failed to fetch team statistics, cannot generate detailed predictions."
                    else:
                        df_team_stats = to_data_frame(team_stats_data_json)
                        data_for_model, todays_games_uo, frame_ml, home_team_odds_list, away_team_odds_list = \
                            create_todays_games_data(games, df_team_stats, odds)

                        if data_for_model.size > 0:
                            xgb_predictions = PredictionRunner.run_xgboost(
                                data_for_model, todays_games_uo, frame_ml, games,
                                home_team_odds_list, away_team_odds_list, kelly_criterion=True
                            )
                            if xgb_predictions:
                                formatted_predictions = []
                                # TODO: Later, filter predictions if user mentioned a specific team.
                                for pred_item in xgb_predictions[:3]: # Limit context length, e.g., first 3 games
                                    kc_home_str = str(pred_item.get("kelly_criterion", {}).get("home_team", "No Bet"))
                                    kc_away_str = str(pred_item.get("kelly_criterion", {}).get("away_team", "No Bet"))
                                    formatted_predictions.append(
                                        f"- Game: {pred_item['away_team']} at {pred_item['home_team']}.\n"
                                        f"  Odds: {pred_item['home_team']} ({pred_item['home_odds']}), {pred_item['away_team']} ({pred_item['away_odds']}).\n"
                                        f"  Predicted Winner: {pred_item['predicted_winner']} ({pred_item['winner_confidence']}% confidence).\n"
                                        f"  O/U ({pred_item['under_over_line']}): {pred_item['under_over_prediction']} ({pred_item['under_over_confidence']}% confidence).\n"
                                        f"  EV: {pred_item['home_team']} ({pred_item['expected_value']['home_team']:.2f}), {pred_item['away_team']} ({pred_item['expected_value']['away_team']:.2f}).\n"
                                        f"  Kelly: {pred_item['home_team']} ({kc_home_str}%), {pred_item['away_team']} ({kc_away_str}%)."
                                    )
                                if formatted_predictions:
                                    game_predictions_context_str = "\n\nContext: Current game predictions based on our XGBoost model are:\n" + "\n".join(formatted_predictions)
                                else:
                                    game_predictions_context_str = "\n\nContext: Our XGBoost model didn't produce any specific predictions right now."
                            else:
                                game_predictions_context_str = "\n\nContext: Our XGBoost model didn't produce predictions right now."
                        else:
                            game_predictions_context_str = "\n\nContext: Could not process data for XGBoost predictions at the moment."
        except Exception as e:
            logger.error(f"Chatbot: Error fetching/processing predictions for context: {e}", exc_info=True)
            game_predictions_context_str = "\n\nContext: I encountered an internal error trying to fetch the latest game predictions."

    # Construct the full prompt for Gemini
    # Ensure user_input_message is the original case message from the user
    full_prompt = f"{CHATBOT_SYSTEM_PROMPT}{game_predictions_context_str}\n\nUser: {user_input_message}"
    
    logger.info(f"Sending prompt to Gemini (approx length {len(full_prompt)}). Context snippet: '{game_predictions_context_str[:100]}...' User message: '{user_input_message}'")

    try:
        response = await gemini_model.generate_content_async(full_prompt)
        
        bot_reply = "I'm sorry, I couldn't generate a response right now." # Default
        # Try to extract text, handling potential new API structures
        if hasattr(response, 'text') and response.text:
            bot_reply = response.text
        elif response.parts:
            bot_reply = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        
        if not bot_reply.strip(): # If bot_reply is empty or just whitespace
            bot_reply = "I'm not sure how to respond to that. Can you try asking differently?"
            logger.warning("Gemini returned an empty or whitespace-only response.")

        # Add disclaimer if response seems to contain betting advice (heuristic)
        keywords_for_disclaimer = ["bet", "pick", "odds", "prediction", "wager", "moneyline", "spread", "over/under", "ev", "kelly"]
        if any(keyword in bot_reply.lower() for keyword in keywords_for_disclaimer) or \
           any(keyword in user_input_message.lower() for keyword in keywords_for_disclaimer):
            disclaimer = "\n\nDisclaimer: Please remember that all betting involves risk. These are AI-generated insights, not financial advice. Gamble responsibly and within your limits."
            if disclaimer not in bot_reply: # Avoid duplicate disclaimers
                 bot_reply += disclaimer

        logger.info(f"Gemini raw response text snippet: {bot_reply[:150]}...")
        return {"response": bot_reply}

    except Exception as e:
        logger.error(f"Error calling Gemini API or processing its response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Sorry, I encountered an issue trying to get a response from the AI.")


if __name__ == "__main__":
    import uvicorn
    # The run_api.py script is preferred for running, but this allows direct execution
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
