import requests
from datetime import datetime
import pandas as pd  # Note: pandas is imported but not used in this specific class anymore.
from config.config import Config  # Assuming you have this config file setup

class NBADataManager:
    def __init__(self):
        """Initializes the NBADataManager with configuration."""
        self.config = Config()
        self.headers = {
            "x-rapidapi-key": self.config.RAPID_API_KEY,
            "x-rapidapi-host": "api-nba-v1.p.rapidapi.com"
        }
        # Define the base URL once
        self.base_url = "https://api-nba-v1.p.rapidapi.com"

    def get_todays_games(self):
        """
        Get today's NBA games, including status, scores, and optionally stats.
        Fetches game details like status, score, quarter, time remaining.
        Conditionally fetches team stats if the game is live or completed.
        For completed games, displays the final score and result.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"{self.base_url}/games/date/{today}"  # Use correct endpoint format

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            games_data = response.json()

            processed_games = {}

            for game in games_data.get('response', []):
                # Safely extract team names and IDs using .get()
                away_team_data = game.get('teams', {}).get('away', {})
                home_team_data = game.get('teams', {}).get('home', {})
                away_name = away_team_data.get('name', 'Unknown Away')
                home_name = home_team_data.get('name', 'Unknown Home')
                away_id = away_team_data.get('id')
                home_id = home_team_data.get('id')

                # Create game key
                game_key = f"{away_name}:{home_name}"

                # Determine game status
                game_status_data = game.get('status', {})
                status_short = game_status_data.get('short')
                
                # Process scores safely using .get()
                scores_data = game.get('scores', {})
                home_score = scores_data.get('home', {}).get('points', 0) or 0
                away_score = scores_data.get('away', {}).get('points', 0) or 0

                # Properly determine game status
                status = 'upcoming'
                result = None  # Initialize result field
                
                # Check if game is completed (status code 3 or specific strings like 'FT', 'F', etc.)
                if status_short == 3 or status_short in ['FT', 'F', 'Final']:
                    status = 'completed'
                    # Determine winner for completed games
                    if home_score > away_score:
                        result = f"{home_name} won by {home_score - away_score} points"
                    elif away_score > home_score:
                        result = f"{away_name} won by {away_score - home_score} points"
                    else:
                        result = "Game ended in a tie"
                
                # Check if game is in progress (status code 2 or specific strings)
                elif status_short == 2 or status_short in ['Q1', 'Q2', 'Q3', 'Q4', 'OT', 'HT', 'Live']:
                    status = 'live'

                # Get quarter and time safely using .get()
                quarter = game.get('periods', {}).get('current', 0)  # Default to 0 if not started
                time_remaining = game_status_data.get('clock', 'N/A')  # Default to N/A

                # Get start time safely
                start_time = game.get('date', {}).get('start', 'Unknown Time')

                # Create processed game entry
                game_data = {
                    'away_team': away_name,
                    'home_team': home_name,
                    'start_time': start_time,
                    'status': status,
                    'home_score': home_score,
                    'away_score': away_score,
                    'quarter': quarter,
                    'time_remaining': time_remaining,
                }
                
                # Add result for completed games
                if result:
                    game_data['result'] = result
                    game_data['final_score'] = f"{away_name} {away_score} - {home_score} {home_name}"
                
                # Initialize team stats as None
                game_data['away_team_stats'] = None
                game_data['home_team_stats'] = None

                # Add game to processed games
                processed_games[game_key] = game_data

                # Conditionally fetch team stats if game is not upcoming and IDs are available
                if status != 'upcoming' and away_id and home_id:
                    try:
                        # Get team stats for this game using the existing method
                        away_stats = self.get_team_stats(away_id)
                        home_stats = self.get_team_stats(home_id)

                        processed_games[game_key]['away_team_stats'] = away_stats
                        processed_games[game_key]['home_team_stats'] = home_stats
                    except Exception as e:
                        # Log error but continue processing other games
                        print(f"Error fetching team stats for game {game_key}: {str(e)}")
                elif status != 'upcoming' and (not away_id or not home_id):
                    print(f"Warning: Missing team ID(s) for game {game_key}. Cannot fetch stats.")

            return processed_games

        except requests.exceptions.RequestException as e:
            print(f"HTTP Error fetching NBA games: {str(e)}")
            return None
        except Exception as e:
            print(f"General Error processing NBA games: {str(e)}")
            return None

    def get_team_stats(self, team_id, season="2023-2024"):
        """
        Get detailed team statistics for a given team ID and season.
        Args:
            team_id (int): The ID of the team.
            season (str): The season year (e.g., "2023-2024").
        Returns:
            dict: Team statistics data, or None if an error occurs.
        """
        url = f"{self.base_url}/teams/statistics"
        params = {
            "id": team_id,
            "season": season
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            stats_data = response.json()
            
            # Extract the actual statistics from the response
            if 'response' in stats_data:
                return stats_data['response']
            return stats_data  # Return raw data if no 'response' field
            
        except requests.exceptions.RequestException as e:
            print(f"HTTP Error fetching team stats for ID {team_id}: {str(e)}")
            return None
        except Exception as e:
            print(f"General Error fetching team stats for ID {team_id}: {str(e)}")
            return None

    def calculate_win_probability(self, away_stats, home_stats):
        """
        Calculate win probabilities based on team statistics.
        Placeholder for a machine learning model or statistical calculation.
        NOTE: This method is NOT called by the updated get_todays_games method.
        Args:
            away_stats (dict): Statistics for the away team.
            home_stats (dict): Statistics for the home team.
        Returns:
            dict: Containing 'away' and 'home' win probability percentages.
        """
        # Placeholder for actual prediction model
        if not away_stats or not home_stats:
            print("Warning: Missing stats for probability calculation.")
            return {'away': 50, 'home': 50}  # Return default if stats are missing

        return {
            'away': 50,  # Placeholder value
            'home': 50   # Placeholder value
        }

# Example Usage
# if __name__ == "__main__":
#     data_manager = NBADataManager()
#     todays_games = data_manager.get_todays_games()
#
#     if todays_games:
#         print(f"Fetched {len(todays_games)} games for today:")
#         for game_key, game_info in todays_games.items():
#             print(f"\n--- Game: {game_key} ---")
#             print(f"  Start Time: {game_info['start_time']}")
#             print(f"  Status: {game_info['status']}")
#             
#             if game_info['status'] == 'completed':
#                 print(f"  FINAL: {game_info['final_score']}")
#                 print(f"  Result: {game_info['result']}")
#             elif game_info['status'] == 'live':
#                 print(f"  Current Score: {game_info['away_team']} {game_info['away_score']} - {game_info['home_score']} {game_info['home_team']}")
#                 print(f"  Quarter: {game_info['quarter']}, Time Left: {game_info['time_remaining']}")
#             else:  # upcoming
#                 print(f"  Matchup: {game_info['away_team']} @ {game_info['home_team']}")
#     else:
#         print("Could not fetch today's games.")