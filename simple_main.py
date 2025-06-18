import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import os

def get_odds_data(sportsbook="fanduel"):
    try:
        # Simulate odds data for testing
        sample_games = [
            {
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'commence_time': datetime.now().isoformat(),
                'bookmakers': [{
                    'markets': [{
                        'outcomes': [
                            {'name': 'Los Angeles Lakers', 'price': -110},
                            {'name': 'Golden State Warriors', 'price': -110}
                        ]
                    }]
                }]
            },
            {
                'home_team': 'Boston Celtics',
                'away_team': 'Miami Heat',
                'commence_time': datetime.now().isoformat(),
                'bookmakers': [{
                    'markets': [{
                        'outcomes': [
                            {'name': 'Boston Celtics', 'price': -120},
                            {'name': 'Miami Heat', 'price': +100}
                        ]
                    }]
                }]
            }
        ]
        return sample_games
    except:
        return None

def make_predictions():
    # Check if model exists
    if not os.path.exists('Models/XGBoost_ML.json'):
        print("Error: Model file not found. Please run train_model.py first!")
        return

    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model('Models/XGBoost_ML.json')
    
    # Get today's games
    games = get_odds_data()
    
    if games:
        print("\n=== NBA Game Predictions ===\n")
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            print(f"\nGame: {away_team} @ {home_team}")
            
            # Create dummy features (replace with real features in production)
            features = np.random.rand(1, 10)  # 10 features matching our training data
            
            # Make prediction
            prediction = model.predict_proba(features)[0]
            home_win_prob = prediction[1] * 100
            
            print(f"Prediction:")
            print(f"Home Team ({home_team}) Win Probability: {home_win_prob:.1f}%")
            print(f"Away Team ({away_team}) Win Probability: {(100-home_win_prob):.1f}%")
            print("-" * 50)
    else:
        print("Error: Could not fetch game data")

if __name__ == "__main__":
    print("\nNBA Prediction System")
    print("=" * 20)
    make_predictions()