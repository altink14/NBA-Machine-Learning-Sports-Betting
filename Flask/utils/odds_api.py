import requests
import os
from datetime import datetime
import pandas as pd

class OddsAPI:
    def __init__(self):
        self.api_key = "7d74e921d3af1af4a91f6e75fd15219c"  # We'll change this later
        self.base_url = "https://api.the-odds-api.com/v4/sports/"
        self.supported_books = ['fanduel', 'draftkings', 'betmgm']

    def get_nba_odds(self):
        """Fetch current NBA odds from multiple sportsbooks"""
        endpoint = f"{self.base_url}basketball_nba/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american'
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return self.process_odds_data(response.json())
            else:
                print(f"API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching odds: {str(e)}")
            return None

    def process_odds_data(self, data):
        """Process and structure the odds data"""
        processed_data = {book: {} for book in self.supported_books}
        
        for game in data:
            game_key = f"{game['away_team']}:{game['home_team']}"
            
            for bookmaker in game['bookmakers']:
                if bookmaker['key'] in self.supported_books:
                    book_data = {
                        'away_team': game['away_team'],
                        'home_team': game['home_team'],
                        'start_time': game['commence_time'],
                        'away_team_odds': None,
                        'home_team_odds': None,
                        'ou_value': None,
                        'ou_pick': None
                    }
                    
                    # Process different market types
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == game['away_team']:
                                    book_data['away_team_odds'] = outcome['price']
                                else:
                                    book_data['home_team_odds'] = outcome['price']
                        elif market['key'] == 'totals':
                            book_data['ou_value'] = market['outcomes'][0]['point']
                    
                    processed_data[bookmaker['key']][game_key] = book_data
        
        return processed_data