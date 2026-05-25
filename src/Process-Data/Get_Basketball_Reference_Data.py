import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup, Comment

# Reconfigure encoding to avoid Windows console errors
sys.stdout.reconfigure(encoding='utf-8')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data", "BasketballReference")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_PATH = os.path.join(BASE_DIR, "Data", "BasketballReference.sqlite")

os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_team_name(name):
    """Remove trailing asterisks and seeding parentheticals (e.g. 'Boston Celtics* (1)' -> 'Boston Celtics')."""
    if not name:
        return ""
    # Strip non-breaking spaces
    name = name.replace("\xa0", " ").strip()
    # Strip asterisks
    name = name.replace("*", "")
    # Remove seed numbers like (1) or (15)
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    return name.strip()

import re

def get_page(url, filename, max_age_hours=24):
    """Fetch HTML page with local caching and rate-limiting."""
    cache_path = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=max_age_hours):
            print(f"Loading {filename} from cache...")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
                
    print(f"Fetching {url} from web...")
    # Be polite to Basketball Reference (max 20 requests per minute)
    time.sleep(3.5)
    
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    html = response.text
    
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return html

def parse_standings(soup):
    """Parse Eastern and Western conference standings."""
    standings = []
    
    for conf, table_id in [("East", "confs_standings_E"), ("West", "confs_standings_W")]:
        table = soup.find("table", id=table_id)
        if not table:
            print(f"Standings table {table_id} not found.")
            continue
            
        tbody = table.find("tbody")
        if not tbody:
            continue
            
        for row in tbody.find_all("tr"):
            if "class" in row.attrs and ("thead" in row["class"] or "over_header" in row["class"]):
                continue
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
                
            team_name = clean_team_name(cells[0].text)
            w = int(cells[1].text or 0)
            l = int(cells[2].text or 0)
            w_pct = float(cells[3].text or 0.0)
            gb = cells[4].text.strip()
            if gb == "—" or gb == "":
                gb = "0.0"
            gb = float(gb)
            
            psg = float(cells[5].text or 0.0)
            pag = float(cells[6].text or 0.0)
            srs = float(cells[7].text or 0.0)
            
            standings.append({
                "team": team_name,
                "conference": conf,
                "wins": w,
                "losses": l,
                "win_pct": w_pct,
                "gb": gb,
                "ppg": psg,
                "oppg": pag,
                "srs": srs
            })
            
    return standings

def parse_team_advanced(soup):
    """Parse advanced team stats table."""
    table = soup.find("table", id="advanced-team")
    if not table:
        print("Advanced team stats table not found.")
        return []
        
    tbody = table.find("tbody")
    if not tbody:
        return []
        
    advanced_stats = []
    for row in tbody.find_all("tr"):
        if "class" in row.attrs and ("thead" in row["class"] or "over_header" in row["class"]):
            continue
        cells = row.find_all(["th", "td"])
        if len(cells) < 15:
            continue
            
        team_name = clean_team_name(cells[1].text)
        # Skip League Average row
        if "League Average" in team_name or not team_name:
            continue
            
        age = float(cells[2].text or 0.0)
        pw = int(cells[5].text or 0)
        pl = int(cells[6].text or 0)
        mov = float(cells[7].text or 0.0)
        sos = float(cells[8].text or 0.0)
        srs = float(cells[9].text or 0.0)
        ortg = float(cells[10].text or 0.0)
        drtg = float(cells[11].text or 0.0)
        nrtg = float(cells[12].text or 0.0)
        pace = float(cells[13].text or 0.0)
        ftr = float(cells[14].text or 0.0)
        par3 = float(cells[15].text or 0.0)  # 3PAr
        ts_pct = float(cells[16].text or 0.0)
        
        # Four Factors (Offensive)
        efg_pct = float(cells[18].text or 0.0)
        tov_pct = float(cells[19].text or 0.0)
        orb_pct = float(cells[20].text or 0.0)
        ft_fga = float(cells[21].text or 0.0)
        
        # Four Factors (Defensive)
        def_efg_pct = float(cells[23].text or 0.0)
        def_tov_pct = float(cells[24].text or 0.0)
        drb_pct = float(cells[25].text or 0.0)
        def_ft_fga = float(cells[26].text or 0.0)
        
        advanced_stats.append({
            "team": team_name,
            "age": age,
            "pw": pw,
            "pl": pl,
            "mov": mov,
            "sos": sos,
            "srs": srs,
            "ortg": ortg,
            "drtg": drtg,
            "nrtg": nrtg,
            "pace": pace,
            "ftr": ftr,
            "par3": par3,
            "ts_pct": ts_pct,
            "efg_pct": efg_pct,
            "tov_pct": tov_pct,
            "orb_pct": orb_pct,
            "ft_fga": ft_fga,
            "def_efg_pct": def_efg_pct,
            "def_tov_pct": def_tov_pct,
            "drb_pct": drb_pct,
            "def_ft_fga": def_ft_fga
        })
        
    return advanced_stats

def parse_team_per_game(soup):
    """Parse team per game stats table."""
    table = soup.find("table", id="per_game-team")
    if not table:
        print("Per game team stats table not found.")
        return []
        
    tbody = table.find("tbody")
    if not tbody:
        return []
        
    per_game_stats = []
    for row in tbody.find_all("tr"):
        if "class" in row.attrs and ("thead" in row["class"] or "over_header" in row["class"]):
            continue
        cells = row.find_all(["th", "td"])
        if len(cells) < 10:
            continue
            
        team_name = clean_team_name(cells[1].text)
        if "League Average" in team_name or not team_name:
            continue
            
        games = int(cells[2].text or 0)
        mp = float(cells[3].text or 0.0)
        fg = float(cells[4].text or 0.0)
        fga = float(cells[5].text or 0.0)
        fg_pct = float(cells[6].text or 0.0)
        fg3 = float(cells[7].text or 0.0)
        fg3a = float(cells[8].text or 0.0)
        fg3_pct = float(cells[9].text or 0.0)
        fg2 = float(cells[10].text or 0.0)
        fg2a = float(cells[11].text or 0.0)
        fg2_pct = float(cells[12].text or 0.0)
        ft = float(cells[13].text or 0.0)
        fta = float(cells[14].text or 0.0)
        ft_pct = float(cells[15].text or 0.0)
        orb = float(cells[16].text or 0.0)
        drb = float(cells[17].text or 0.0)
        trb = float(cells[18].text or 0.0)
        ast = float(cells[19].text or 0.0)
        stl = float(cells[20].text or 0.0)
        blk = float(cells[21].text or 0.0)
        tov = float(cells[22].text or 0.0)
        pf = float(cells[23].text or 0.0)
        pts = float(cells[24].text or 0.0)
        
        per_game_stats.append({
            "team": team_name,
            "games": games,
            "mp": mp,
            "fg": fg,
            "fga": fga,
            "fg_pct": fg_pct,
            "fg3": fg3,
            "fg3a": fg3a,
            "fg3_pct": fg3_pct,
            "fg2": fg2,
            "fg2a": fg2a,
            "fg2_pct": fg2_pct,
            "ft": ft,
            "fta": fta,
            "ft_pct": ft_pct,
            "orb": orb,
            "drb": drb,
            "trb": trb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "tov": tov,
            "pf": pf,
            "pts": pts
        })
        
    return per_game_stats

def parse_player_per_game(html):
    """Parse player per game averages and return a dict keyed by Player Name/ID."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="per_game_stats")
    if not table:
        print("Player per game table not found.")
        return {}
        
    tbody = table.find("tbody")
    if not tbody:
        return {}
        
    players = {}
    for row in tbody.find_all("tr"):
        if "class" in row.attrs and ("thead" in row["class"] or "over_header" in row["class"]):
            continue
        cells = row.find_all(["th", "td"])
        if len(cells) < 29:
            continue
            
        player_cell = cells[1]
        player_name = player_cell.text.strip()
        
        # Get Player ID from link
        link = player_cell.find("a")
        if not link:
            continue
        href = link.get("href", "")
        # Link format: /players/d/doncilu01.html
        match = re.search(r"/players/\w/([^/.]+)\.html", href)
        if not match:
            continue
        player_id = match.group(1)
        
        # If player was traded, they might have multiple entries (TOT for total, and team entries).
        # We only keep the total stats or their main team stats. If they have TOT, keep that.
        team = cells[3].text.strip()
        
        age = int(cells[2].text or 0)
        pos = cells[4].text.strip()
        games = int(cells[5].text or 0)
        gs = int(cells[6].text or 0)
        mp = float(cells[7].text or 0.0)
        fg = float(cells[8].text or 0.0)
        fga = float(cells[9].text or 0.0)
        fg_pct = float(cells[10].text or 0.0) if cells[10].text.strip() else 0.0
        fg3 = float(cells[11].text or 0.0)
        fg3a = float(cells[12].text or 0.0)
        fg3_pct = float(cells[13].text or 0.0) if cells[13].text.strip() else 0.0
        ft = float(cells[18].text or 0.0)
        fta = float(cells[19].text or 0.0)
        ft_pct = float(cells[20].text or 0.0) if cells[20].text.strip() else 0.0
        orb = float(cells[21].text or 0.0)
        drb = float(cells[22].text or 0.0)
        trb = float(cells[23].text or 0.0)
        ast = float(cells[24].text or 0.0)
        stl = float(cells[25].text or 0.0)
        blk = float(cells[26].text or 0.0)
        tov = float(cells[27].text or 0.0)
        pts = float(cells[29].text or 0.0)
        
        # If player already exists in our dict and this entry is NOT 'TOT' (Total), skip it.
        # This keeps the overall seasonal stats for traded players.
        if player_id in players and team != "TOT":
            continue
            
        players[player_id] = {
            "player_id": player_id,
            "name": player_name,
            "age": age,
            "team": team,
            "pos": pos,
            "games": games,
            "gs": gs,
            "mp": mp,
            "fg": fg,
            "fga": fga,
            "fg_pct": fg_pct,
            "fg3": fg3,
            "fg3a": fg3a,
            "fg3_pct": fg3_pct,
            "ft": ft,
            "fta": fta,
            "ft_pct": ft_pct,
            "orb": orb,
            "drb": drb,
            "trb": trb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "tov": tov,
            "pts": pts,
            "per": 0.0,  # Default, populated from advanced stats
            "ts_pct": 0.0,
            "ws": 0.0,
            "bpm": 0.0,
            "vorp": 0.0
        }
        
    return players

def parse_player_advanced(html, players):
    """Parse player advanced stats and merge them into the players dict."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="advanced")
    if not table:
        print("Player advanced stats table not found.")
        return players
        
    tbody = table.find("tbody")
    if not tbody:
        return players
        
    for row in tbody.find_all("tr"):
        if "class" in row.attrs and ("thead" in row["class"] or "over_header" in row["class"]):
            continue
        cells = row.find_all(["th", "td"])
        if len(cells) < 28:
            continue
            
        player_cell = cells[1]
        link = player_cell.find("a")
        if not link:
            continue
        href = link.get("href", "")
        match = re.search(r"/players/\w/([^/.]+)\.html", href)
        if not match:
            continue
        player_id = match.group(1)
        
        team = cells[3].text.strip()
        
        # Merge if player exists and keep TOT if traded
        if player_id in players:
            if players[player_id]["team"] != "TOT" and team == "TOT":
                # If we parsed a team specific entry first but advanced has TOT, update team.
                players[player_id]["team"] = "TOT"
            elif players[player_id]["team"] == "TOT" and team != "TOT":
                # We already have TOT, ignore team-specific entry.
                pass
            
            per = float(cells[7].text or 0.0) if cells[7].text.strip() else 0.0
            ts_pct = float(cells[8].text or 0.0) if cells[8].text.strip() else 0.0
            ws = float(cells[21].text or 0.0) if cells[21].text.strip() else 0.0
            bpm = float(cells[26].text or 0.0) if cells[26].text.strip() else 0.0
            vorp = float(cells[27].text or 0.0) if cells[27].text.strip() else 0.0
            
            players[player_id]["per"] = per
            players[player_id]["ts_pct"] = ts_pct
            players[player_id]["ws"] = ws
            players[player_id]["bpm"] = bpm
            players[player_id]["vorp"] = vorp
            
    return players

def save_to_db(standings, team_advanced, team_per_game, players):
    """Save scraped data to SQLite database."""
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable journal mode for speed and safety
    cursor.execute("PRAGMA journal_mode=WAL")
    
    # 1. Table: team_standings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_standings (
            team TEXT PRIMARY KEY,
            conference TEXT,
            wins INTEGER,
            losses INTEGER,
            win_pct REAL,
            gb REAL,
            ppg REAL,
            oppg REAL,
            srs REAL
        )
    """)
    cursor.execute("DELETE FROM team_standings")
    for s in standings:
        cursor.execute("""
            INSERT OR REPLACE INTO team_standings 
            (team, conference, wins, losses, win_pct, gb, ppg, oppg, srs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (s["team"], s["conference"], s["wins"], s["losses"], s["win_pct"], s["gb"], s["ppg"], s["oppg"], s["srs"]))
        
    # 2. Table: team_advanced_stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_advanced_stats (
            team TEXT PRIMARY KEY,
            age REAL,
            pw INTEGER,
            pl INTEGER,
            mov REAL,
            sos REAL,
            srs REAL,
            ortg REAL,
            drtg REAL,
            nrtg REAL,
            pace REAL,
            ftr REAL,
            par3 REAL,
            ts_pct REAL,
            efg_pct REAL,
            tov_pct REAL,
            orb_pct REAL,
            ft_fga REAL,
            def_efg_pct REAL,
            def_tov_pct REAL,
            drb_pct REAL,
            def_ft_fga REAL
        )
    """)
    cursor.execute("DELETE FROM team_advanced_stats")
    for a in team_advanced:
        cursor.execute("""
            INSERT OR REPLACE INTO team_advanced_stats
            (team, age, pw, pl, mov, sos, srs, ortg, drtg, nrtg, pace, ftr, par3, ts_pct,
             efg_pct, tov_pct, orb_pct, ft_fga, def_efg_pct, def_tov_pct, drb_pct, def_ft_fga)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (a["team"], a["age"], a["pw"], a["pl"], a["mov"], a["sos"], a["srs"], a["ortg"], a["drtg"], a["nrtg"], a["pace"], a["ftr"], a["par3"], a["ts_pct"],
              a["efg_pct"], a["tov_pct"], a["orb_pct"], a["ft_fga"], a["def_efg_pct"], a["def_tov_pct"], a["drb_pct"], a["def_ft_fga"]))

    # 3. Table: team_per_game_stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_per_game_stats (
            team TEXT PRIMARY KEY,
            games INTEGER,
            mp REAL,
            fg REAL,
            fga REAL,
            fg_pct REAL,
            fg3 REAL,
            fg3a REAL,
            fg3_pct REAL,
            fg2 REAL,
            fg2a REAL,
            fg2_pct REAL,
            ft REAL,
            fta REAL,
            ft_pct REAL,
            orb REAL,
            drb REAL,
            trb REAL,
            ast REAL,
            stl REAL,
            blk REAL,
            tov REAL,
            pf REAL,
            pts REAL
        )
    """)
    cursor.execute("DELETE FROM team_per_game_stats")
    for p in team_per_game:
        cursor.execute("""
            INSERT OR REPLACE INTO team_per_game_stats
            (team, games, mp, fg, fga, fg_pct, fg3, fg3a, fg3_pct, fg2, fg2a, fg2_pct,
             ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, tov, pf, pts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (p["team"], p["games"], p["mp"], p["fg"], p["fga"], p["fg_pct"], p["fg3"], p["fg3a"], p["fg3_pct"], p["fg2"], p["fg2a"], p["fg2_pct"],
              p["ft"], p["fta"], p["ft_pct"], p["orb"], p["drb"], p["trb"], p["ast"], p["stl"], p["blk"], p["tov"], p["pf"], p["pts"]))

    # 4. Table: player_stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            player_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            team TEXT,
            pos TEXT,
            games INTEGER,
            gs INTEGER,
            mp REAL,
            fg REAL,
            fga REAL,
            fg_pct REAL,
            fg3 REAL,
            fg3a REAL,
            fg3_pct REAL,
            ft REAL,
            fta REAL,
            ft_pct REAL,
            orb REAL,
            drb REAL,
            trb REAL,
            ast REAL,
            stl REAL,
            blk REAL,
            tov REAL,
            pts REAL,
            per REAL,
            ts_pct REAL,
            ws REAL,
            bpm REAL,
            vorp REAL
        )
    """)
    cursor.execute("DELETE FROM player_stats")
    for p_id, p in players.items():
        cursor.execute("""
            INSERT OR REPLACE INTO player_stats
            (player_id, name, age, team, pos, games, gs, mp, fg, fga, fg_pct, fg3, fg3a, fg3_pct,
             ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, tov, pts, per, ts_pct, ws, bpm, vorp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (p["player_id"], p["name"], p["age"], p["team"], p["pos"], p["games"], p["gs"], p["mp"], p["fg"], p["fga"], p["fg_pct"], p["fg3"], p["fg3a"], p["fg3_pct"],
              p["ft"], p["fta"], p["ft_pct"], p["orb"], p["drb"], p["trb"], p["ast"], p["stl"], p["blk"], p["tov"], p["pts"], p["per"], p["ts_pct"], p["ws"], p["bpm"], p["vorp"]))

    conn.commit()
    conn.close()
    print("Database sync completed successfully!")

def main():
    print("Starting Basketball Reference scraper...")
    season = 2026
    
    # URLs
    summary_url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
    player_per_game_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    player_advanced_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    
    try:
        # 1. Scrape league summary
        summary_html = get_page(summary_url, f"NBA_{season}_summary.html")
        soup = BeautifulSoup(summary_html, "html.parser")
        
        print("Parsing standings...")
        standings = parse_standings(soup)
        print(f"Parsed {len(standings)} teams for standings.")
        
        print("Parsing team advanced stats...")
        team_advanced = parse_team_advanced(soup)
        print(f"Parsed {len(team_advanced)} teams for advanced stats.")
        
        print("Parsing team per-game stats...")
        team_per_game = parse_team_per_game(soup)
        print(f"Parsed {len(team_per_game)} teams for per-game stats.")
        
        # 2. Scrape player stats
        print("Fetching player per-game html...")
        player_per_game_html = get_page(player_per_game_url, f"NBA_{season}_per_game.html")
        print("Parsing player per-game stats...")
        players = parse_player_per_game(player_per_game_html)
        print(f"Parsed {len(players)} players (per-game).")
        
        print("Fetching player advanced html...")
        player_advanced_html = get_page(player_advanced_url, f"NBA_{season}_advanced.html")
        print("Parsing player advanced stats...")
        players = parse_player_advanced(player_advanced_html, players)
        print(f"Parsed advanced stats for players.")
        
        # 3. Save to database
        save_to_db(standings, team_advanced, team_per_game, players)
        print("Scraper run completed successfully!")
        
    except Exception as e:
        print(f"An error occurred in scraping: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
