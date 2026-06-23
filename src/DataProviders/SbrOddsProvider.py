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
    def __init__(self, sportsbook="fanduel", sport="NBA"):
        self.sportsbook = sportsbook
        self.sport = sport
        self.games = self._fetch_games_with_fallback()

    def _fetch_games_with_fallback(self):
        """
        Fetches games for the requested sport.
        If the sport is NBA and none are found, falls back to WNBA.
        """
        import sbrscrape
        if "WNBA" not in sbrscrape.sport_dict:
            sbrscrape.sport_dict["WNBA"] = "wnba-basketball"
            
        today = datetime.now()
        requested_sport = self.sport.upper()

        # If sport is NBA, we check NBA first, then WNBA
        if requested_sport == 'NBA':
            # Check NBA for the next 7 days
            for i in range(7):
                check_date = today + timedelta(days=i)
                logger.info(f"Checking for NBA games on: {check_date.strftime('%Y-%m-%d')}")
                try:
                    sb = Scoreboard(sport="NBA", date=check_date)
                    if hasattr(sb, 'games') and sb.games:
                        logger.info(f"Found {len(sb.games)} NBA games on {check_date.strftime('%Y-%m-%d')}.")
                        for game in sb.games:
                            game['game_start_time_utc'] = game.get('datetime')
                            game['sport'] = 'NBA'
                        return sb.games
                except Exception as e:
                    logger.error(f"Failed to fetch NBA games for {check_date.strftime('%Y-%m-%d')} due to an error: {e}")

            # Fallback to WNBA
            logger.info("No NBA games found. Falling back to WNBA...")
            requested_sport = 'WNBA'

        # Fetch the requested sport (could be WNBA, MLB, etc.)
        logger.info(f"Checking for {requested_sport} games...")
        for i in range(7):
            check_date = today + timedelta(days=i)
            logger.info(f"Checking for {requested_sport} games on: {check_date.strftime('%Y-%m-%d')}")
            try:
                sb = Scoreboard(sport=requested_sport, date=check_date)
                if hasattr(sb, 'games') and sb.games:
                    logger.info(f"Found {len(sb.games)} {requested_sport} games on {check_date.strftime('%Y-%m-%d')}.")
                    for game in sb.games:
                        game['game_start_time_utc'] = game.get('datetime')
                        game['sport'] = requested_sport
                    return sb.games
            except Exception as e:
                logger.error(f"Failed to fetch {requested_sport} games for {check_date.strftime('%Y-%m-%d')} due to an error: {e}")
        
        logger.warning(f"No {requested_sport} games found within the next 7 days.")
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

