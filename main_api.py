# main_api.py
# FINAL STABLE VERSION - Corrected endpoint routing and data handling.
import os
import uvicorn
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import json
from datetime import datetime

# Local Imports
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Utils import Expected_Value, Kelly_Criterion as kc
from src.Utils.tools import create_todays_games_from_odds
from src.Utils.Dictionaries import team_index_current

# Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI App Setup
app = FastAPI(title="Betting Buddy API", version="1.1.1-stable-fixed")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
        self.odds_provider = SbrOddsProvider(sportsbook=self.sportsbook)
        self.xgb_ml_model, self.xgb_uo_model = self._load_xgboost_models()

    def _load_team_stats(self):
        try:
            path = os.path.join(self.project_root, 'Data', 'nba-2024-UTC.csv')
            return pd.read_csv(path)
        except FileNotFoundError:
            logger.error(f"Team statistics file not found at expected path: {path}")
            raise HTTPException(status_code=500, detail="Server configuration error: Team statistics file missing.")

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
        
        for home_team, away_team in games:
            game_key = f"{home_team}:{away_team}"
            game_odds = odds.get(game_key, {})
            
            home_team_index = team_index_current.get(home_team)
            away_team_index = team_index_current.get(away_team)
            if home_team_index is None or away_team_index is None:
                continue

            home_stats = self.team_stats_df.iloc[home_team_index]
            away_stats = self.team_stats_df.iloc[away_team_index]
            
            game_data = pd.concat([home_stats, away_stats.rename(index=lambda x: x + '.1')])
            game_data_list.append(game_data)
            
            home_odds_list.append(game_odds.get(home_team, {}).get('money_line_odds'))
            away_odds_list.append(game_odds.get(away_team, {}).get('money_line_odds'))
            uo_lines_list.append(game_odds.get('under_over_odds'))
            game_start_times_list.append(game_odds.get('game_start_time_utc'))

        if not game_data_list:
            return np.array([]), [], pd.DataFrame(), [], [], []
            
        frame_ml = pd.concat(game_data_list, axis=1).T
        
        columns_to_drop = [col for col in frame_ml.columns if not pd.api.types.is_numeric_dtype(frame_ml[col])]
        frame_for_model = frame_ml.drop(columns=columns_to_drop)
        
        return frame_for_model.values.astype(float), uo_lines_list, frame_ml, home_odds_list, away_odds_list, game_start_times_list

    def _run_xgboost_models(self, data_ml, frame_ml, todays_games_uo):
        ml_predictions = self.xgb_ml_model.predict(xgb.DMatrix(data_ml))
        frame_uo = frame_ml.copy()
        safe_uo = [x if x is not None else np.nan for x in todays_games_uo]
        if len(safe_uo) == len(frame_uo):
            frame_uo['OU'] = np.asarray(safe_uo)
        else:
            frame_uo['OU'] = np.nan
        
        columns_to_drop = [col for col in frame_uo.columns if not pd.api.types.is_numeric_dtype(frame_uo[col])]
        frame_uo.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        ou_predictions = self.xgb_uo_model.predict(xgb.DMatrix(frame_uo.values.astype(float)))
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

# This is the predictions endpoint for http://localhost:8000/predictions
@app.get("/predictions")
def get_predictions_endpoint(sportsbook: str = 'fanduel', kelly_criterion: bool = True):
    try:
        runner = PredictionRunner(sportsbook=sportsbook, kelly_criterion=kelly_criterion)
        return runner.run_predictions()
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

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
