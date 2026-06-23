# scrape_bbref_deep.py
import requests
from bs4 import BeautifulSoup, Comment
import json
import time
import os
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def unwrap_comment_table(soup, table_id):
    table = soup.find('table', id=table_id)
    if table:
        return table
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if f'id="{table_id}"' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id=table_id)
            if table:
                return table
    return None

def safe_float(text):
    if not text:
        return 0.0
    text = text.strip().replace("%", "")
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except ValueError:
        return 0.0

def safe_int(text):
    if not text:
        return 0
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return 0

def scrape_team_advanced_stats(season=2026):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
    print(f"Fetching team advanced stats from: {url}")
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"Failed to fetch team stats: HTTP {r.status_code}")
            return []
            
        soup = BeautifulSoup(r.content, 'html.parser')
        table = unwrap_comment_table(soup, "advanced-team")
        if not table:
            print("Could not find '#advanced-team' table.")
            return []
            
        rows = table.find('tbody').find_all('tr')
        team_stats = []
        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue
            cols = row.find_all(['td', 'th'])
            if not cols:
                continue
            team_name = row.find('td', {'data-stat': 'team'}).text.strip().replace("*", "")
            
            def get_val(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return el.text.strip() if el else ""
                
            team_stats.append({
                "teamName": team_name,
                "offRating": safe_float(get_val('off_rtg')),
                "defRating": safe_float(get_val('def_rtg')),
                "netRating": safe_float(get_val('net_rtg')),
                "pace": safe_float(get_val('pace')),
                "fourFactors": {
                    "eFG": safe_float(get_val('efg_pct')),
                    "TOV": safe_float(get_val('tov_pct')),
                    "ORB": safe_float(get_val('orb_pct')),
                    "FT": safe_float(get_val('ft_rate'))
                }
            })
        return team_stats
    except Exception as e:
        print(f"Error scraping team stats: {e}")
        return []

def scrape_referee_stats(season=2026):
    url = f"https://www.basketball-reference.com/referees/{season}_register.html"
    print(f"Fetching referee stats from: {url}")
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = unwrap_comment_table(soup, "rs_raw")
        if not table and season == 2026:
            print("2026 referee table not found. Falling back to 2025...")
            url = "https://www.basketball-reference.com/referees/2025_register.html"
            r = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(r.content, 'html.parser')
            table = unwrap_comment_table(soup, "rs_raw")
            
        if not table:
            return []
            
        rows = table.find('tbody').find_all('tr')
        referee_stats = []
        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue
            ref_name_el = row.find('th', {'data-stat': 'referee'}) or row.find('td', {'data-stat': 'referee'})
            if not ref_name_el:
                continue
            ref_name = ref_name_el.text.strip()
            
            def get_val(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return el.text.strip() if el else ""
                
            games = safe_int(get_val('g'))
            if games < 5:
                continue
                
            referee_stats.append({
                "refereeName": ref_name,
                "games": games,
                "foulsPerGame": safe_float(get_val('pf_per_g')),
                "foulsRelative": get_val('pf_per_g_rel'),
                "ptsPerGame": safe_float(get_val('pts_per_g')),
                "ptsRelative": get_val('pts_per_g_rel'),
                "homeWinPct": safe_float(get_val('home_win_loss_pct'))
            })
        return referee_stats
    except Exception as e:
        print(f"Error scraping referee stats: {e}")
        return []

def scrape_player_per_game(season=2026):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    print(f"Fetching player per-game stats from: {url}")
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = unwrap_comment_table(soup, "per_game_stats")
        if not table:
            print("Failed to find '#per_game_stats' table.")
            return []
            
        rows = table.find('tbody').find_all('tr')
        players = []
        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue
            name_el = row.find('td', {'data-stat': 'name_display'}) or row.find('td', {'data-stat': 'player'})
            if not name_el:
                continue
            name = name_el.text.strip()
            
            def get_val(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return el.text.strip() if el else ""
                
            players.append({
                "name": name,
                "age": safe_int(get_val('age')),
                "team": get_val('team_name_abbr'),
                "pos": get_val('pos'),
                "games": safe_int(get_val('games')),
                "gamesStarted": safe_int(get_val('games_started')),
                "minutes": safe_float(get_val('mp_per_g')),
                "fgPct": safe_float(get_val('fg_pct')) * 100,
                "fg3Pct": safe_float(get_val('fg3_pct')) * 100,
                "ftPct": safe_float(get_val('ft_pct')) * 100,
                "efgPct": safe_float(get_val('efg_pct')) * 100,
                "pts": safe_float(get_val('pts_per_g')),
                "ast": safe_float(get_val('ast_per_g')),
                "trb": safe_float(get_val('trb_per_g')),
                "stl": safe_float(get_val('stl_per_g')),
                "blk": safe_float(get_val('blk_per_g')),
                "tov": safe_float(get_val('tov_per_g')),
                "pf": safe_float(get_val('pf_per_g')),
                "awards": get_val('awards')
            })
        return players
    except Exception as e:
        print(f"Error scraping player per-game: {e}")
        return []

def scrape_player_advanced(season=2026):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    print(f"Fetching player advanced stats from: {url}")
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = unwrap_comment_table(soup, "advanced")
        if not table:
            print("Failed to find '#advanced' table.")
            return []
            
        rows = table.find('tbody').find_all('tr')
        advanced = []
        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue
            name_el = row.find('td', {'data-stat': 'name_display'}) or row.find('td', {'data-stat': 'player'})
            if not name_el:
                continue
            name = name_el.text.strip()
            
            def get_val(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return el.text.strip() if el else ""
                
            advanced.append({
                "name": name,
                "age": safe_int(get_val('age')),
                "team": get_val('team_name_abbr'),
                "pos": get_val('pos'),
                "per": safe_float(get_val('per')),
                "tsPct": safe_float(get_val('ts_pct')) * 100,
                "orbPct": safe_float(get_val('orb_pct')),
                "drbPct": safe_float(get_val('drb_pct')),
                "trbPct": safe_float(get_val('trb_pct')),
                "astPct": safe_float(get_val('ast_pct')),
                "stlPct": safe_float(get_val('stl_pct')),
                "blkPct": safe_float(get_val('blk_pct')),
                "tovPct": safe_float(get_val('tov_pct')),
                "usgPct": safe_float(get_val('usg_pct')),
                "ows": safe_float(get_val('ows')),
                "dws": safe_float(get_val('dws')),
                "ws": safe_float(get_val('ws')),
                "ws48": safe_float(get_val('ws_per_48')),
                "obpm": safe_float(get_val('obpm')),
                "dbpm": safe_float(get_val('dbpm')),
                "bpm": safe_float(get_val('bpm')),
                "vorp": safe_float(get_val('vorp'))
            })
        return advanced
    except Exception as e:
        print(f"Error scraping player advanced stats: {e}")
        return []

def scrape_standings(season=2026):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
    print(f"Fetching standings from: {url}")
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        table_e = unwrap_comment_table(soup, "confs_standings_E")
        table_w = unwrap_comment_table(soup, "confs_standings_W")
        
        standings = {"East": [], "West": []}
        
        def parse_table(table):
            if not table:
                return []
            rows = table.find('tbody').find_all('tr')
            list_teams = []
            for row in rows:
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                team_name_el = row.find('th', {'data-stat': 'team_name'}) or row.find('td', {'data-stat': 'team_name'})
                if not team_name_el:
                    continue
                
                raw_team_name = team_name_el.text.strip()
                clean_name = re.sub(r'[*]+', '', raw_team_name).strip()
                clean_name = re.sub(r'\s*\(\d+\)\s*$', '', clean_name).strip()
                
                def get_val(stat_name):
                    el = row.find('td', {'data-stat': stat_name})
                    return el.text.strip() if el else ""
                    
                list_teams.append({
                    "teamName": clean_name,
                    "wins": safe_int(get_val('wins')),
                    "losses": safe_int(get_val('losses')),
                    "winLossPct": safe_float(get_val('win_loss_pct')),
                    "gb": get_val('gb'),
                    "ptsPerGame": safe_float(get_val('pts_per_g')),
                    "oppPtsPerGame": safe_float(get_val('opp_pts_per_g')),
                    "srs": safe_float(get_val('srs'))
                })
            return list_teams
            
        standings["East"] = parse_table(table_e)
        standings["West"] = parse_table(table_w)
        return standings
    except Exception as e:
        print(f"Error scraping standings: {e}")
        return {"East": [], "West": []}

def scrape_injuries():
    url = "https://www.basketball-reference.com/friv/injuries.fcgi"
    print(f"Fetching injuries from: {url}")
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"Failed to fetch injuries: HTTP {r.status_code}")
            return []
            
        soup = BeautifulSoup(r.content, 'html.parser')
        table = unwrap_comment_table(soup, "injuries")
        if not table:
            print("Failed to find '#injuries' table.")
            return []
            
        rows = table.find('tbody').find_all('tr')
        injuries = []
        for row in rows:
            if row.get('class') and 'thead' in row.get('class'):
                continue
            
            player_el = row.find('th', {'data-stat': 'player'}) or row.find('td', {'data-stat': 'player'})
            if not player_el:
                continue
            player_name = player_el.text.strip()
            
            def get_val(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return el.text.strip() if el else ""
                
            injuries.append({
                "player": player_name,
                "team": get_val('team_name'),
                "updateDate": get_val('date_update'),
                "description": get_val('note')
            })
        return injuries
    except Exception as e:
        print(f"Error scraping injuries: {e}")
        return []

def main():
    start_time = time.time()
    print("=== STARTING DEEP BASKETBALL-REFERENCE SCRAPE ===")
    
    seasons = [2026, 2025, 2024]
    combined_data = {
        "asOfTimestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seasons": {},
        "injuries": []
    }
    
    # 0. Injuries
    print("\n--- SCRAPING REAL-TIME INJURY REPORT ---")
    injuries = scrape_injuries()
    combined_data["injuries"] = injuries
    print(f"Scraped {len(injuries)} current injuries.")
    print("Sleeping 4s...")
    time.sleep(4.0)
    
    for season in seasons:
        print(f"\n--- SCRAPING SEASON: {season} ---")
        
        # 1. Standings
        standings = scrape_standings(season)
        print(f"Scraped standings: East={len(standings['East'])}, West={len(standings['West'])}")
        print("Sleeping 4s...")
        time.sleep(4.0)
        
        # 2. Team Advanced
        team_stats = scrape_team_advanced_stats(season)
        print(f"Scraped {len(team_stats)} team advanced metrics.")
        print("Sleeping 4s...")
        time.sleep(4.0)
        
        # 3. Referees
        ref_stats = scrape_referee_stats(season)
        print(f"Scraped {len(ref_stats)} referees.")
        print("Sleeping 4s...")
        time.sleep(4.0)
        
        # 4. Player Per Game
        player_pg = scrape_player_per_game(season)
        print(f"Scraped {len(player_pg)} per-game player rows.")
        print("Sleeping 4s...")
        time.sleep(4.0)
        
        # 5. Player Advanced
        player_adv = scrape_player_advanced(season)
        print(f"Scraped {len(player_adv)} advanced player rows.")
        
        # Merge
        print("Merging player statistics...")
        adv_dict = {}
        for p in player_adv:
            key = (p["name"], p["team"])
            adv_dict[key] = p
            
        merged_players = []
        for p in player_pg:
            key = (p["name"], p["team"])
            adv_info = adv_dict.get(key)
            if not adv_info:
                matches = [item for (name, team), item in adv_dict.items() if name == p["name"]]
                if matches:
                    adv_info = matches[0]
                    
            merged_p = {**p}
            if adv_info:
                for k, v in adv_info.items():
                    if k not in merged_p:
                        merged_p[k] = v
            else:
                merged_p["per"] = 0.0
                merged_p["tsPct"] = 0.0
                merged_p["orbPct"] = 0.0
                merged_p["drbPct"] = 0.0
                merged_p["trbPct"] = 0.0
                merged_p["astPct"] = 0.0
                merged_p["stlPct"] = 0.0
                merged_p["blkPct"] = 0.0
                merged_p["tovPct"] = 0.0
                merged_p["usgPct"] = 0.0
                merged_p["ows"] = 0.0
                merged_p["dws"] = 0.0
                merged_p["ws"] = 0.0
                merged_p["ws48"] = 0.0
                merged_p["obpm"] = 0.0
                merged_p["dbpm"] = 0.0
                merged_p["bpm"] = 0.0
                merged_p["vorp"] = 0.0
                
            merged_players.append(merged_p)
            
        print(f"Total merged players for {season}: {len(merged_players)}")
        
        combined_data["seasons"][str(season)] = {
            "standings": standings,
            "teamAdvancedStats": team_stats,
            "refereeStats": ref_stats,
            "players": merged_players
        }
        
        # Sleep between seasons to stay under limit
        if season != seasons[-1]:
            print("Sleeping 6s between seasons...")
            time.sleep(6.0)
            
    output_path = os.path.join(
        "c:/Users/altin/OneDrive/Documents/GitHub/basic-saas-starter/src/lib",
        "bbref_data_deep.json"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
    print(f"=== DEEP SCRAPE COMPLETE. Saved to: {output_path} ===")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
