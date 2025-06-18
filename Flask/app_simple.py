from flask import Flask, render_template, jsonify
from flask_caching import Cache
from datetime import datetime, timedelta
from utils.odds_api import OddsAPI
from utils.nba_data import NBADataManager
from config.config import Config
import threading
import time
import requests
import json

app = Flask(__name__)

# Add custom datetime filter
@app.template_filter('datetime')
def format_datetime(value, format='%I:%M %p'):
    """Format a datetime object or ISO string to a readable time."""
    if value is None:
        return ""
    
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return value
    
    if isinstance(value, datetime):
        return value.strftime(format)
    
    return value

# API Usage Tracker
class APIUsageTracker:
    def __init__(self):
        self.monthly_requests = 0
        self.last_request_time = None
        self.month_start = datetime.now()

    def can_make_request(self):
        now = datetime.now()
        
        # Reset counter on new month
        if now.month != self.month_start.month:
            self.monthly_requests = 0
            self.month_start = now

        # Check monthly limit
        if self.monthly_requests >= 450:  # Warning at 450
            print("WARNING: Approaching monthly API limit!")
            return False

        # Check rate limit (1 per minute)
        if self.last_request_time and (now - self.last_request_time) < timedelta(minutes=1):
            return False

        return True

    def log_request(self):
        self.monthly_requests += 1
        self.last_request_time = datetime.now()
        print(f"API Requests this month: {self.monthly_requests}/500")

api_tracker = APIUsageTracker()

# Configure cache properly
cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}
app.config.from_mapping(cache_config)
cache = Cache(app)

odds_api = OddsAPI()
nba_data_manager = NBADataManager()

def fetch_live_games():
    """Fetch today's NBA games and odds"""
    if not api_tracker.can_make_request():
        print("API rate limit reached or approaching monthly limit")
        return get_game_data()

    try:
        # Get NBA game data first
        nba_games = nba_data_manager.get_todays_games()
        
        # Get odds data
        odds_data = odds_api.get_nba_odds()
        
        if not nba_games:
            print("No NBA games found for today, using fallback data")
            return get_game_data()
            
        if odds_data:
            api_tracker.log_request()
            
            # Create a new structure with real games
            real_games = {
                'fanduel': {},
                'draftkings': {},
                'betmgm': {}
            }
            
            # Process and enhance the odds data with real NBA games
            for game_key, game_info in nba_games.items():
                # Try to find matching odds
                odds_found = False
                
                for sportsbook in odds_data:
                    if game_key in odds_data[sportsbook]:
                        # Copy odds data
                        real_games[sportsbook][game_key] = odds_data[sportsbook][game_key]
                        odds_found = True
                    else:
                        # Try to find a match with different formatting
                        for odds_key in odds_data[sportsbook]:
                            if (game_info['away_team'] in odds_key and game_info['home_team'] in odds_key):
                                real_games[sportsbook][game_key] = odds_data[sportsbook][odds_key]
                                odds_found = True
                                break
                
                # If no odds found, create entry with game data only
                if not odds_found:
                    for sportsbook in real_games:
                        real_games[sportsbook][game_key] = {
                            'away_team': game_info['away_team'],
                            'home_team': game_info['home_team'],
                            'start_time': game_info.get('start_time', datetime.now().isoformat()),
                            'away_team_odds': -110,  # Default odds
                            'home_team_odds': -110,  # Default odds
                            'ou_value': 220.5,      # Default over/under
                        }
                
                # Add game status and scores
                for sportsbook in real_games:
                    if game_key in real_games[sportsbook]:
                        game = real_games[sportsbook][game_key]
                        game.update({
                            'status': game_info.get('status', 'upcoming'),
                            'home_score': game_info.get('home_score', 0),
                            'away_score': game_info.get('away_score', 0),
                            'quarter': game_info.get('quarter', 1),
                            'time_remaining': game_info.get('time_remaining', '12:00'),
                        })
                        
                        # Calculate confidence if odds exist
                        if game['away_team_odds'] and game['home_team_odds']:
                            total_odds = abs(game['away_team_odds']) + abs(game['home_team_odds'])
                            game['away_confidence'] = round((abs(game['home_team_odds']) / total_odds) * 100)
                            game['home_confidence'] = round((abs(game['away_team_odds']) / total_odds) * 100)
                            
                            # Calculate EV
                            game['away_team_ev'] = calculate_ev(game['away_team_odds'], game['away_confidence'])
                            game['home_team_ev'] = calculate_ev(game['home_team_odds'], game['home_confidence'])
                        else:
                            # Default confidence values
                            game['away_confidence'] = 50
                            game['home_confidence'] = 50
                            game['away_team_ev'] = 0
                            game['home_team_ev'] = 0
                        
                        # Add over/under confidence (placeholder)
                        game['ou_confidence'] = 55
            
            return real_games
    except Exception as e:
        print(f"Error fetching live games: {str(e)}")
    return get_game_data()

def calculate_ev(odds, win_probability):
    """Calculate Expected Value"""
    try:
        if odds > 0:
            return round((odds/100 * (win_probability/100)) - ((1 - (win_probability/100)) * 1), 2)
        else:
            return round((100/abs(odds) * (win_probability/100)) - ((1 - (win_probability/100)) * 1), 2)
    except:
        return 0

def get_game_data():
    """Fallback data with today's date"""
    current_time = datetime.now().isoformat()
    three_hours_ago = (datetime.now() - timedelta(hours=3)).isoformat()
    one_hour_from_now = (datetime.now() + timedelta(hours=1)).isoformat()
    
    # These are placeholder games - in production this would be replaced with real data
    return {
        'fanduel': {
            'Bucks:76ers': {
                'away_team': 'Bucks',
                'home_team': '76ers',
                'away_team_odds': -110,
                'home_team_odds': -110,
                'away_team_ev': 2.5,
                'home_team_ev': 1.8,
                'away_confidence': 65,
                'home_confidence': 35,
                'ou_value': 220.5,
                'ou_pick': 'OVER',
                'ou_confidence': 58,
                'status': 'live',
                'home_score': 78,
                'away_score': 72,
                'quarter': 3,
                'time_remaining': '4:35',
                'start_time': three_hours_ago
            },
            'Knicks:Nets': {
                'away_team': 'Knicks',
                'home_team': 'Nets',
                'away_team_odds': +150,
                'home_team_odds': -170,
                'away_team_ev': 3.1,
                'home_team_ev': 0.8,
                'away_confidence': 58,
                'home_confidence': 42,
                'ou_value': 235.5,
                'ou_pick': 'UNDER',
                'ou_confidence': 62,
                'status': 'upcoming',
                'start_time': one_hour_from_now
            },
            'Heat:Celtics': {
                'away_team': 'Heat',
                'home_team': 'Celtics',
                'away_team_odds': +180,
                'home_team_odds': -220,
                'away_team_ev': 1.2,
                'home_team_ev': 0.5,
                'away_confidence': 40,
                'home_confidence': 60,
                'ou_value': 218.5,
                'ou_pick': 'OVER',
                'ou_confidence': 53,
                'status': 'completed',
                'home_score': 105,
                'away_score': 98,
                'start_time': three_hours_ago
            }
        },
        'draftkings': {},
        'betmgm': {}
    }

def background_odds_update():
    """Background task to update odds periodically"""
    while True:
        with app.app_context():
            try:
                if api_tracker.can_make_request():
                    odds_data = fetch_live_games()
                    if odds_data:
                        cache.set('current_odds', odds_data)
                        cache.set('last_updated', datetime.now().isoformat())
                        print(f"Updated odds data at {datetime.now()}")
                    else:
                        cache.set('current_odds', get_game_data())
                        cache.set('last_updated', datetime.now().isoformat())
            except Exception as e:
                print(f"Error in background update: {str(e)}")
                cache.set('current_odds', get_game_data())
                cache.set('last_updated', datetime.now().isoformat())
        time.sleep(Config.UPDATE_INTERVAL)

@app.before_request
def initialize():
    """Start background odds updating"""
    if not hasattr(app, 'background_started'):
        thread = threading.Thread(target=background_odds_update)
        thread.daemon = True
        thread.start()
        app.background_started = True

@app.route('/')
def index():
    try:
        odds_data = cache.get('current_odds')
        if not odds_data:
            odds_data = fetch_live_games()
            if odds_data:
                cache.set('current_odds', odds_data)
                cache.set('last_updated', datetime.now().isoformat())
            else:
                odds_data = get_game_data()
                cache.set('last_updated', datetime.now().isoformat())

        print(f"Serving odds data at {datetime.now()}")
        print("Available games:", list(odds_data.get('fanduel', {}).keys()))
        print(f"API requests made this month: {api_tracker.monthly_requests}/500")
        
        # Format last_updated for display in template
        last_updated = cache.get('last_updated')
        if last_updated:
            try:
                last_updated_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                formatted_last_updated = last_updated_time.strftime('%Y-%m-%d %I:%M %p')
            except (ValueError, TypeError):
                formatted_last_updated = "recently"
        else:
            formatted_last_updated = "recently"

        return render_template('index.html',
                            data=odds_data,
                            today=datetime.now().strftime("%Y-%m-%d"),
                            last_updated=formatted_last_updated)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return f"An error occurred: {str(e)}"

@app.route('/api/games')
def api_games():
    """API endpoint for getting game data via AJAX"""
    odds_data = cache.get('current_odds')
    if not odds_data:
        odds_data = get_game_data()
    
    last_updated = cache.get('last_updated') or datetime.now().isoformat()
    
    return jsonify({
        'data': odds_data,
        'last_updated': last_updated
    })

if __name__ == '__main__':
    app.run(debug=True)