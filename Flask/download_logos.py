import os
import requests
import json
from pathlib import Path

# Team IDs mapping 
team_ids = {
    "Atlanta Hawks": "1610612737",
    "Boston Celtics": "1610612738",
    "Brooklyn Nets": "1610612751",
    "Charlotte Hornets": "1610612766",
    "Chicago Bulls": "1610612741",
    "Cleveland Cavaliers": "1610612739",
    "Dallas Mavericks": "1610612742",
    "Denver Nuggets": "1610612743",
    "Detroit Pistons": "1610612765",
    "Golden State Warriors": "1610612744",
    "Houston Rockets": "1610612745",
    "Indiana Pacers": "1610612754",
    "LA Clippers": "1610612746",
    "Los Angeles Lakers": "1610612747",
    "Memphis Grizzlies": "1610612763",
    "Miami Heat": "1610612748",
    "Milwaukee Bucks": "1610612749",
    "Minnesota Timberwolves": "1610612750",
    "New Orleans Pelicans": "1610612740",
    "New York Knicks": "1610612752",
    "Oklahoma City Thunder": "1610612760",
    "Orlando Magic": "1610612753",
    "Philadelphia 76ers": "1610612755",
    "Phoenix Suns": "1610612756",
    "Portland Trail Blazers": "1610612757",
    "Sacramento Kings": "1610612758",
    "San Antonio Spurs": "1610612759",
    "Toronto Raptors": "1610612761",
    "Utah Jazz": "1610612762",
    "Washington Wizards": "1610612764"
}

def download_team_logos():
    # Create directories
    logo_dir = Path("static/img/team-logos")
    logo_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading logos to {logo_dir.absolute()}")
    
    # Download logos for all teams
    for team_name, team_id in team_ids.items():
        try:
            # Create normalized filename
            filename = team_name.lower().replace(" ", "_")
            file_path = logo_dir / f"{filename}.png"
            
            # Skip if already exists
            if file_path.exists():
                print(f"Logo for {team_name} already exists, skipping...")
                continue
                
            # Download from NBA CDN
            logo_url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
            print(f"Downloading {team_name} logo from {logo_url}")
            
            response = requests.get(logo_url)
            response.raise_for_status()  # Raise exception for 404s, etc.
            
            # Save the logo
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Successfully saved logo for {team_name}")
            
        except Exception as e:
            print(f"❌ Error downloading logo for {team_name}: {str(e)}")
    
    # Create a default logo for fallback
    try:
        default_path = logo_dir / "default.png"
        if not default_path.exists():
            print("Creating default logo...")
            default_url = "https://cdn.nba.com/logos/nba/1610612747/global/L/logo.svg"
            default_response = requests.get(default_url)
            with open(default_path, "wb") as f:
                f.write(default_response.content)
            print("✅ Default logo created")
    except Exception as e:
        print(f"❌ Error creating default logo: {str(e)}")
        
    # Save team ID mapping for reference
    try:
        with open(logo_dir / "team_ids.json", "w") as f:
            json.dump(team_ids, f, indent=2)
        print("✅ Saved team ID mapping file")
    except Exception as e:
        print(f"❌ Error saving team ID mapping: {str(e)}")
    
    print("\nLogo download process complete!")
    print(f"Total logos downloaded: {len(list(logo_dir.glob('*.png'))) - 1}")  # Exclude default.png

if __name__ == "__main__":
    download_team_logos()