from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware  # ADD THIS LINE
from typing import Optional, Dict, Any, List
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import json
import logging
import os

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NBA Sports Betting Predictions API",
    description="Machine learning predictions for NBA games with odds from various sportsbooks",
    version="1.0.0"
)

# ADD CORS MIDDLEWARE - This fixes the frontend connection issue
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

def create_todays_games_data(games, df, odds):
    """Modified version of createTodaysGames that returns structured data"""
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    home_team_days_rest = []
    away_team_days_rest = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
            
        if odds is not None:
            game_key = home_team + ':' + away_team
            if game_key in odds:
                game_odds = odds[game_key]
                todays_games_uo.append(game_odds['under_over_odds'])
                home_team_odds.append(game_odds[home_team]['money_line_odds'])
                away_team_odds.append(game_odds[away_team]['money_line_odds'])
            else:
                continue
        else:
            # Skip games without odds in API mode
            continue

        # Calculate days rest for both teams
        try:
            schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
            home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
            away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
            
            previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
            previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
            
            if len(previous_home_games) > 0:
                last_home_date = previous_home_games.iloc[0]
                home_days_off = timedelta(days=1) + datetime.today() - last_home_date
            else:
                home_days_off = timedelta(days=7)
                
            if len(previous_away_games) > 0:
                last_away_date = previous_away_games.iloc[0]
                away_days_off = timedelta(days=1) + datetime.today() - last_away_date
            else:
                away_days_off = timedelta(days=7)

            home_team_days_rest.append(home_days_off.days)
            away_team_days_rest.append(away_days_off.days)
            
            home_team_series = df.iloc[team_index_current.get(home_team)]
            away_team_series = df.iloc[team_index_current.get(away_team)]
            stats = pd.concat([home_team_series, away_team_series])
            stats['Days-Rest-Home'] = home_days_off.days
            stats['Days-Rest-Away'] = away_days_off.days
            match_data.append(stats)
        except Exception as e:
            # Use default values if schedule data is unavailable
            home_team_days_rest.append(2)
            away_team_days_rest.append(2)
            
            home_team_series = df.iloc[team_index_current.get(home_team)]
            away_team_series = df.iloc[team_index_current.get(away_team)]
            stats = pd.concat([home_team_series, away_team_series])
            stats['Days-Rest-Home'] = 2
            stats['Days-Rest-Away'] = 2
            match_data.append(stats)

    if not match_data:
        return None, None, None, None, None

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T
    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds

class PredictionRunner:
    """Modified prediction runners that return data instead of printing"""
    
    @staticmethod
    def run_xgboost(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, use_kc=False):
        import copy
        import numpy as np
        import xgboost as xgb
        from src.Utils import Expected_Value
        from src.Utils import Kelly_Criterion as kc
        
        try:
            # Load models
            xgb_ml = xgb.Booster()
            xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.9%_ML-3.json')  
            xgb_uo = xgb.Booster()
            xgb_uo.load_model('Models/XGBoost_Models/XGBoost_54.8%_UO-8.json')
            
            # Money Line predictions
            ml_predictions_array = []
            for row in data:
                ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

            # Over/Under predictions
            frame_uo = copy.deepcopy(frame_ml)
            frame_uo['OU'] = np.asarray(todays_games_uo)
            uo_data = frame_uo.values.astype(float)

            ou_predictions_array = []
            for row in uo_data:
                ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

            # Process predictions for each game
            predictions = []
            for count, game in enumerate(games):
                home_team = game[0]
                away_team = game[1]
                
                # Winner prediction
                winner = int(np.argmax(ml_predictions_array[count]))
                winner_confidence = ml_predictions_array[count]
                
                # Over/Under prediction
                under_over = int(np.argmax(ou_predictions_array[count]))
                un_confidence = ou_predictions_array[count]
                
                # Calculate confidences
                if winner == 1:  # Home team wins
                    predicted_winner = home_team
                    winner_confidence_pct = round(winner_confidence[0][1] * 100, 1)
                else:  # Away team wins
                    predicted_winner = away_team
                    winner_confidence_pct = round(winner_confidence[0][0] * 100, 1)
                
                # Over/Under confidence
                if under_over == 0:  # Under
                    uo_prediction = "UNDER"
                    uo_confidence_pct = round(ou_predictions_array[count][0][0] * 100, 1)
                else:  # Over
                    uo_prediction = "OVER"
                    uo_confidence_pct = round(ou_predictions_array[count][0][1] * 100, 1)
                
                # Expected Values
                ev_home = ev_away = 0
                kc_home = kc_away = 0
                
                if count < len(home_team_odds) and count < len(away_team_odds):
                    if home_team_odds[count] and away_team_odds[count]:
                        ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
                        ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
                        
                        if use_kc:
                            kc_home = kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])
                            kc_away = kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])
                
                prediction = {
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_odds": away_team_odds[count] if count < len(away_team_odds) else "N/A",
                    "home_odds": home_team_odds[count] if count < len(home_team_odds) else "N/A",
                    "predicted_winner": predicted_winner,
                    "winner_confidence": winner_confidence_pct,
                    "under_over_line": todays_games_uo[count] if count < len(todays_games_uo) else "N/A",
                    "under_over_prediction": uo_prediction,
                    "under_over_confidence": uo_confidence_pct,
                    "expected_value": {
                        "home_team": ev_home,
                        "away_team": ev_away
                    },
                    "model": "XGBoost"
                }
                
                if use_kc:
                    prediction["kelly_criterion"] = {
                        "home_team": kc_home,
                        "away_team": kc_away
                    }
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"XGBoost prediction error: {str(e)}")
    
    @staticmethod
    def run_neural_network(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, use_kc=False):
        raise HTTPException(
            status_code=503, 
            detail="Neural Network models are corrupted and need to be retrained. Please use XGBoost model instead."
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NBA Sports Betting Predictions API",
        "endpoints": {
            "/predictions": "Get predictions for today's games",
            "/health": "Health check",
            "/docs": "API documentation"
        },
        "supported_sportsbooks": SUPPORTED_SPORTSBOOKS
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/predictions")
async def get_predictions(
    sportsbook: str = Query(..., description="Sportsbook to fetch odds from"),
    model: str = Query("xgboost", description="Model to use: xgboost, neural_network, or all"),
    kelly_criterion: bool = Query(False, description="Include Kelly Criterion calculations")
) -> Dict[str, Any]:
    """
    Get NBA game predictions with odds from specified sportsbook
    
    - **sportsbook**: One of the supported sportsbooks (fanduel, draftkings, etc.)
    - **model**: Model to use for predictions (xgboost, neural_network, or all)
    - **kelly_criterion**: Whether to include Kelly Criterion betting recommendations
    """
    
    # Validate sportsbook
    if sportsbook not in SUPPORTED_SPORTSBOOKS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported sportsbook. Supported: {', '.join(SUPPORTED_SPORTSBOOKS)}"
        )
    
    # Validate model
    if model not in ["xgboost", "neural_network", "all"]:
        raise HTTPException(
            status_code=400,
            detail="Model must be one of: xgboost, neural_network, all"
        )
    
    try:
        # Fetch odds data
        odds_provider = SbrOddsProvider(sportsbook=sportsbook)
        odds = odds_provider.get_odds()
        
        if not odds:
            raise HTTPException(status_code=404, detail="No odds data found for today")
        
        # Create games from odds
        games = create_todays_games_from_odds(odds)
        
        if not games:
            raise HTTPException(status_code=404, detail="No games found for today")
        
        # Validate games against odds
        if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
            raise HTTPException(
                status_code=503, 
                detail="Games list not up to date for today's games"
            )
        
        # Get team stats data
        data = get_json_data(DATA_URL)
        df = to_data_frame(data)
        
        # Create today's games data
        processed_data = create_todays_games_data(games, df, odds)
        
        if processed_data[0] is None:
            raise HTTPException(status_code=404, detail="No valid game data could be processed")
        
        data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = processed_data
        
        # Prepare response
        response = {
            "timestamp": datetime.now().isoformat(),
            "sportsbook": sportsbook,
            "total_games": len(games),
            "odds_data": {}
        }
        
        # Add odds information
        for game_key, game_odds in odds.items():
            home_team, away_team = game_key.split(":")
            response["odds_data"][game_key] = {
                "away_team": away_team,
                "home_team": home_team,
                "away_money_line": game_odds[away_team]['money_line_odds'],
                "home_money_line": game_odds[home_team]['money_line_odds'],
                "under_over": game_odds['under_over_odds']
            }
        
        # Run predictions based on model choice
        if model == "xgboost":
            predictions = PredictionRunner.run_xgboost(
                data, todays_games_uo, frame_ml, games, 
                home_team_odds, away_team_odds, kelly_criterion
            )
            response["predictions"] = predictions
            
        elif model == "neural_network":
            predictions = PredictionRunner.run_neural_network(
                data, todays_games_uo, frame_ml, games, 
                home_team_odds, away_team_odds, kelly_criterion
            )
            response["predictions"] = predictions
            
        elif model == "all":
            xgb_predictions = PredictionRunner.run_xgboost(
                data, todays_games_uo, frame_ml, games, 
                home_team_odds, away_team_odds, kelly_criterion
            )
            
            # Try Neural Network, but handle gracefully if it fails
            try:
                nn_predictions = PredictionRunner.run_neural_network(
                    data, todays_games_uo, frame_ml, games, 
                    home_team_odds, away_team_odds, kelly_criterion
                )
                response["predictions"] = {
                    "xgboost": xgb_predictions,
                    "neural_network": nn_predictions
                }
            except HTTPException:
                response["predictions"] = {
                    "xgboost": xgb_predictions,
                    "neural_network_error": "Neural Network models are corrupted and unavailable"
                }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sportsbooks")
async def get_supported_sportsbooks():
    """Get list of supported sportsbooks"""
    return {"supported_sportsbooks": SUPPORTED_SPORTSBOOKS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
