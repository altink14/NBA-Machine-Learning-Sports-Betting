from datetime import date
import json
from flask import Flask, render_template, jsonify
from functools import lru_cache
import subprocess, requests, re, time, sqlite3, os, unicodedata


@lru_cache()
def fetch_fanduel(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="fanduel")

@lru_cache()
def fetch_draftkings(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="draftkings")

@lru_cache()
def fetch_betmgm(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="betmgm")

def fetch_game_data(sportsbook="fanduel"):
    cmd = ["python", "main.py", "-xgb", f"-odds={sportsbook}"]
    stdout = subprocess.check_output(cmd, cwd="../").decode()
    data_re = re.compile(r'\n(?P<home_team>[\w ]+)(\((?P<home_confidence>[\d+\.]+)%\))? vs (?P<away_team>[\w ]+)(\((?P<away_confidence>[\d+\.]+)%\))?: (?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d+\.]+) (\((?P<ou_confidence>[\d+\.]+)%\))?', re.MULTILINE)
    ev_re = re.compile(r'(?P<team>[\w ]+) EV: (?P<ev>[-\d+\.]+)', re.MULTILINE)
    odds_re = re.compile(r'(?P<away_team>[\w ]+) \((?P<away_team_odds>-?\d+)\) @ (?P<home_team>[\w ]+) \((?P<home_team_odds>-?\d+)\)', re.MULTILINE)
    games = {}
    for match in data_re.finditer(stdout):
        game_dict = {'away_team': match.group('away_team').strip(),
                     'home_team': match.group('home_team').strip(),
                     'away_confidence': match.group('away_confidence'),
                     'home_confidence': match.group('home_confidence'),
                     'ou_pick': match.group('ou_pick'),
                     'ou_value': match.group('ou_value'),
                     'ou_confidence': match.group('ou_confidence')}
        for ev_match in ev_re.finditer(stdout):
            if ev_match.group('team') == game_dict['away_team']:
                game_dict['away_team_ev'] = ev_match.group('ev')
            if ev_match.group('team') == game_dict['home_team']:
                game_dict['home_team_ev'] = ev_match.group('ev')
        for odds_match in odds_re.finditer(stdout):
            if odds_match.group('away_team') == game_dict['away_team']:
                game_dict['away_team_odds'] = odds_match.group('away_team_odds')
            if odds_match.group('home_team') == game_dict['home_team']:
                game_dict['home_team_odds'] = odds_match.group('home_team_odds')

        print(json.dumps(game_dict, sort_keys=True, indent=4))
        games[f"{game_dict['away_team']}:{game_dict['home_team']}"] = game_dict
    return games


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


def get_bbr_db_connection():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "BasketballReference.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_name(name):
    if not name:
        return ""
    normalized = unicodedata.normalize('NFD', name)
    return "".join([c for c in normalized if unicodedata.category(c) != 'Mn']).lower().strip()


@app.route("/api/standings")
def get_standings():
    try:
        conn = get_bbr_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, a.ortg, a.drtg, a.nrtg, a.pace
            FROM team_standings s
            LEFT JOIN team_advanced_stats a ON s.team = a.team
            ORDER BY s.win_pct DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        standings = [dict(row) for row in rows]
        
        east = [t for t in standings if t["conference"] == "East"]
        west = [t for t in standings if t["conference"] == "West"]
        
        return jsonify({
            "success": True,
            "east": east,
            "west": west
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/api/matchup/<team1>/<team2>")
def get_matchup(team1, team2):
    try:
        conn = get_bbr_db_connection()
        
        def find_team_stats(team_name):
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.*, p.pts, p.ast, p.trb, p.stl, p.blk, p.tov, p.fg_pct, p.fg3_pct, p.ft_pct
                FROM team_advanced_stats a
                LEFT JOIN team_per_game_stats p ON a.team = p.team
                WHERE a.team = ?
            """, (team_name,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            
            clean_name = team_name.replace("LA ", "Los Angeles ").replace("NY ", "New York ").strip()
            cursor.execute("""
                SELECT a.*, p.pts, p.ast, p.trb, p.stl, p.blk, p.tov, p.fg_pct, p.fg3_pct, p.ft_pct
                FROM team_advanced_stats a
                LEFT JOIN team_per_game_stats p ON a.team = p.team
                WHERE a.team LIKE ? OR a.team LIKE ?
            """, (f"%{team_name}%", f"%{clean_name}%"))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
            
        t1_stats = find_team_stats(team1)
        t2_stats = find_team_stats(team2)
        conn.close()
        
        if not t1_stats or not t2_stats:
            return jsonify({
                "success": False,
                "error": f"Stats not found for one or both teams: {team1} ({'found' if t1_stats else 'missing'}), {team2} ({'found' if t2_stats else 'missing'})"
            })
            
        return jsonify({
            "success": True,
            "team1": t1_stats,
            "team2": t2_stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/api/player-summary/<player_name>")
def get_player_summary(player_name):
    try:
        conn = get_bbr_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM player_stats")
        rows = cursor.fetchall()
        conn.close()
        
        target = normalize_name(player_name)
        matched_player = None
        for row in rows:
            db_name_norm = normalize_name(row["name"])
            if db_name_norm == target or target in db_name_norm or db_name_norm in target:
                matched_player = dict(row)
                break
                
        if not matched_player:
            return jsonify({
                "success": False,
                "error": f"Player {player_name} not found in database"
            })
            
        return jsonify({
            "success": True,
            "player": matched_player
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/")
def index():
    fanduel = fetch_fanduel(ttl_hash=get_ttl_hash())
    draftkings = fetch_draftkings(ttl_hash=get_ttl_hash())
    betmgm = fetch_betmgm(ttl_hash=get_ttl_hash())

    return render_template('index.html', today=date.today(), data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm})




def get_player_data(team_abv):
    """Fetch player data for a given team abbreviation"""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster"
    headers = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    querystring = {"teamAbv": team_abv}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        
        if data.get('statusCode') == 200:
            formatted_players = []
            roster = data.get('body', {}).get('roster', [])
            
            for player in roster:
                # Format injury status
                injury_status = "Healthy"
                if player.get('injury'):
                    injury_info = player['injury']
                    if injury_info.get('designation'):
                        injury_status = injury_info['designation']
                        if injury_info.get('description'):
                            injury_status += f" - {injury_info['description']}"
                
                formatted_player = {
                    'name': player.get('longName'),
                    'shortName': player.get('shortName'),
                    'headshot': player.get('nbaComHeadshot'),
                    'injury': injury_status,
                    'position': player.get('pos'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'college': player.get('college'),
                    'experience': player.get('exp'),
                    'jerseyNum': player.get('jerseyNum'),
                    'playerId': player.get('playerID'),
                    'birthDate': player.get('bDay')
                }
                formatted_players.append(formatted_player)
            
            return {
                'success': True,
                'players': formatted_players
            }
        
        return {
            'success': False,
            'error': 'Failed to fetch team data'
        }
        
    except Exception as e:
        print(f"Error in get_player_data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)
    
    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        })
    
    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


    
@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    headers = {
        "x-rapidapi-key": "a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    
    # First get player info
    info_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo"
    info_querystring = {"playerID": player_id}
    
    # Then get game stats
    games_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
    games_querystring = {
        "playerID": player_id,
        "season": "2024",
    }
    
    try:
        # Get both responses
        info_response = requests.get(info_url, headers=headers, params=info_querystring)
        games_response = requests.get(games_url, headers=headers, params=games_querystring)
        
        info_data = info_response.json()
        games_data = games_response.json()
        
        if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
            # Process games data
            games = list(games_data['body'].values())
            games.sort(key=lambda x: x['gameID'], reverse=True)
            recent_games = games[:10]
            
            # Get player info
            player_info = info_data['body']
            
            # Format injury info
            injury_status = "Healthy"
            if player_info.get('injury'):
                injury_info = player_info['injury']
                injury_status = injury_info

            # Combine and return all data
            return jsonify({
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('longName'),
                    'position': player_info.get('pos'),
                    'number': player_info.get('jerseyNum'),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'team': player_info.get('team'),
                    'college': player_info.get('college'),
                    'experience': player_info.get('exp'),
                    'headshot': player_info.get('nbaComHeadshot'),
                    'injury': injury_status
                }
            })
            
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

        
team_abbreviations = {
    'Orlando Magic': 'ORL',
    'Minnesota Timberwolves': 'MIN',
    'Miami Heat': 'MIA',
    'Boston Celtics': 'BOS',
    'LA Clippers': 'LAC',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Atlanta Hawks': 'ATL',
    'Cleveland Cavaliers': 'CLE',
    'Toronto Raptors': 'TOR',
    'Washington Wizards': 'WAS',
    'Phoenix Suns': 'PHO',
    'San Antonio Spurs': 'SA',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHA',
    'Philadelphia 76ers': 'PHI',
    'New Orleans Pelicans': 'NO',
    'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL',
    'Houston Rockets': 'HOU',
    'Brooklyn Nets': 'BKN',
    'New York Knicks': 'NY',
    'Utah Jazz': 'UTA',
    'Oklahoma City Thunder': 'OKC',
    'Portland Trail Blazers': 'POR',
    'Indiana Pacers': 'IND',
    'Milwaukee Bucks': 'MIL',
    'Golden State Warriors': 'GS',
    'Memphis Grizzlies': 'MEM',
    'Los Angeles Lakers': 'LAL'
}