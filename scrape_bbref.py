# scrape_bbref.py
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
    """Basketball-Reference wraps tables inside comments to speed up page loads. 
    This helper extracts and parses tables from HTML comments."""
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
            # Skip divider rows
            if row.get('class') and 'thead' in row.get('class'):
                continue
                
            cols = row.find_all(['td', 'th'])
            if not cols:
                continue
                
            # Extract team name and metrics
            team_name = row.find('td', {'data-stat': 'team'}).text.strip().replace("*", "")
            
            # Helper to safely parse float
            def get_float(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return float(el.text.strip()) if el and el.text.strip() else 0.0

            # Advanced metrics
            ortg = get_float('off_rtg')
            drtg = get_float('def_rtg')
            nrtg = get_float('net_rtg')
            pace = get_float('pace')
            
            # Four Factors
            efg_pct = get_float('efg_pct')
            tov_pct = get_float('tov_pct')
            orb_pct = get_float('orb_pct')
            ft_rate = get_float('ft_rate')
            
            team_stats.append({
                "teamName": team_name,
                "offRating": ortg,
                "defRating": drtg,
                "netRating": nrtg,
                "pace": pace,
                "fourFactors": {
                    "eFG": efg_pct,
                    "TOV": tov_pct,
                    "ORB": orb_pct,
                    "FT": ft_rate
                }
            })
            
        return team_stats
    except Exception as e:
        print(f"Error scraping team advanced stats: {e}")
        return []

def scrape_referee_stats(season=2026):
    url = f"https://www.basketball-reference.com/referees/{season}_register.html"
    print(f"Fetching referee stats from: {url}")
    
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = unwrap_comment_table(soup, "rs_raw")
        
        # Fall back to 2025 if 2026 table isn't present
        if not table and season == 2026:
            print("2026 referee table not found. Falling back to 2025 season...")
            url = "https://www.basketball-reference.com/referees/2025_register.html"
            r = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(r.content, 'html.parser')
            table = unwrap_comment_table(soup, "rs_raw")
            
        if not table:
            print("Could not find '#rs_raw' table on any season register page.")
            return []
            
        rows = table.find('tbody').find_all('tr')
        referee_stats = []
        
        for row in rows:
            # Skip header or group rows
            if row.get('class') and 'thead' in row.get('class'):
                continue
                
            cols = row.find_all(['td', 'th'])
            if not cols:
                continue
                
            ref_name_el = row.find('th', {'data-stat': 'referee'}) or row.find('td', {'data-stat': 'referee'})
            if not ref_name_el:
                continue
            ref_name = ref_name_el.text.strip()
            
            def get_int(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return int(el.text.strip()) if el and el.text.strip() else 0

            def get_float(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return float(el.text.strip()) if el and el.text.strip() else 0.0

            def get_string(stat_name):
                el = row.find('td', {'data-stat': stat_name})
                return el.text.strip() if el else ""

            games = get_int('g')
            if games < 5:
                continue # Skip inactive/low game referees
                
            fouls = get_float('pf_per_g')
            fouls_rel = get_string('pf_per_g_rel')
            pts = get_float('pts_per_g')
            pts_rel = get_string('pts_per_g_rel')
            
            # Parse home win pct (e.g. .560 -> 0.560)
            home_win_pct_str = get_string('home_win_loss_pct')
            home_win_pct = float(home_win_pct_str) if home_win_pct_str and home_win_pct_str != "." else 0.0
            
            referee_stats.append({
                "refereeName": ref_name,
                "games": games,
                "foulsPerGame": fouls,
                "foulsRelative": fouls_rel,
                "ptsPerGame": pts,
                "ptsRelative": pts_rel,
                "homeWinPct": home_win_pct
            })
            
        return referee_stats
    except Exception as e:
        print(f"Error scraping referee stats: {e}")
        return []

def main():
    start_time = time.time()
    print("Starting Basketball-Reference scraper...")
    
    # 1. Scrape Team Advanced Stats
    team_stats = scrape_team_advanced_stats(season=2026)
    
    # Stay below BBRef 20 requests per minute rate limit
    print("Pausing 5 seconds to respect rate limits...")
    time.sleep(5.0)
    
    # 2. Scrape Referee Stats
    ref_stats = scrape_referee_stats(season=2026)
    
    combined_data = {
        "asOfTimestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teamAdvancedStats": team_stats,
        "refereeStats": ref_stats
    }
    
    # Write to basic-saas-starter next.js folder
    output_path = os.path.join(
        "c:/Users/altin/OneDrive/Documents/GitHub/basic-saas-starter/src/lib",
        "bbref_data.json"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"Successfully scraped and wrote BBRef data to: {output_path}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
