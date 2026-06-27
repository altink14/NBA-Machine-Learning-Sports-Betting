import requests
import json

base_url = "http://localhost:8000"
game_id = "0022400061"

print(f"Testing backend endpoints for game {game_id}...")

# 1. Game Details
try:
    r = requests.get(f"{base_url}/api/games/{game_id}")
    print(f"\nGET /api/games/{game_id} -> Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Game Date: {data.get('game_date')}")
        print(f"  Season: {data.get('season')}")
        print(f"  Teams: {data.get('away_team', {}).get('name')} @ {data.get('home_team', {}).get('name')}")
        print(f"  Scores: {data.get('away_team', {}).get('score')} - {data.get('home_team', {}).get('score')}")
    else:
        print(f"  Error: {r.text}")
except Exception as e:
    print(f"  Request failed: {e}")

# 2. Play-by-Play
try:
    r = requests.get(f"{base_url}/api/games/{game_id}/play-by-play")
    print(f"\nGET /api/games/{game_id}/play-by-play -> Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Total PBP events: {len(data)}")
        if data:
            print("  First event:")
            print(json.dumps(data[0], indent=4))
    else:
        print(f"  Error: {r.text}")
except Exception as e:
    print(f"  Request failed: {e}")

# 3. Shot Chart
try:
    r = requests.get(f"{base_url}/api/games/{game_id}/shot-chart")
    print(f"\nGET /api/games/{game_id}/shot-chart -> Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Total shots: {len(data.get('shots', []))}")
        if data.get('shots'):
            print("  First shot:")
            print(json.dumps(data['shots'][0], indent=4))
    else:
        print(f"  Error: {r.text}")
except Exception as e:
    print(f"  Request failed: {e}")
