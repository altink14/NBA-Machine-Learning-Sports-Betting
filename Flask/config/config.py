import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ODDS_API_KEY = os.getenv('7d74e921d3af1af4a91f6e75fd15219c')
    NBA_API_KEY = os.getenv('NBA_API_KEY')  # for official NBA stats
    RAPID_API_KEY = os.getenv('RAPID_API_KEY')  # for additional stats

    # API endpoints
    NBA_GAMES_URL = "https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/"
    ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    
    # Update intervals
    UPDATE_INTERVAL = 300  # 5 minutes for odds
    STATS_UPDATE_INTERVAL = 3600  # 1 hour for team stats

    