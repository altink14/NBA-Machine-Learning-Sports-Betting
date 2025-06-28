# src/DataProviders/SbrOddsProvider.py
from sbrscrape import Scoreboard
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SbrOddsProvider:
    """
    An intelligent odds provider that fetches NBA games.
    If no games are found for the current day, it automatically checks subsequent days.
    """
    def __init__(self, sportsbook="fanduel"):
        self.sportsbook = sportsbook
        self.games = self._fetch_games_with_fallback()

    def _fetch_games_with_fallback(self):
        """
        Tries to fetch games for today, and if none are found,
        iteratively checks the next few days.
        """
        today = datetime.now()
        # Look up to 7 days in the future for the next available games
        for i in range(7):
            check_date = today + timedelta(days=i)
            logger.info(f"Checking for NBA games on: {check_date.strftime('%Y-%m-%d')}")
            try:
                sb = Scoreboard(sport="NBA", date=check_date)
                if hasattr(sb, 'games') and sb.games:
                    logger.info(f"Found {len(sb.games)} games on {check_date.strftime('%Y-%m-%d')}.")
                    # Add game start time to each game object for the frontend
                    for game in sb.games:
                        game['game_start_time_utc'] = game.get('datetime')
                    return sb.games
            except Exception as e:
                logger.error(f"Failed to fetch games for {check_date.strftime('%Y-%m-%d')} due to an error: {e}")
        
        logger.warning("No games found within the next 7 days.")
        return []

    def get_odds(self):
        """
        Processes the fetched games to return odds in a structured dictionary.
        """
        dict_res = {}
        if not self.games:
            return dict_res

        for game in self.games:
            try:
                home_team_name = game['home_team'].replace("Los Angeles Clippers", "LA Clippers")
                away_team_name = game['away_team'].replace("Los Angeles Clippers", "LA Clippers")

                money_line_home_value = game.get('home_ml', {}).get(self.sportsbook)
                money_line_away_value = game.get('away_ml', {}).get(self.sportsbook)
                totals_value = game.get('total', {}).get(self.sportsbook)
                
                # Only include games that have odds from the specified sportsbook
                if money_line_home_value is not None and money_line_away_value is not None:
                    dict_res[f"{home_team_name}:{away_team_name}"] = {
                        'under_over_odds': totals_value,
                        home_team_name: {'money_line_odds': money_line_home_value},
                        away_team_name: {'money_line_odds': money_line_away_value},
                        'game_start_time_utc': game.get('game_start_time_utc') # Pass the start time
                    }
            except KeyError as e:
                logger.warning(f"Skipping a game due to missing key: {e}")
                continue
        return dict_res

