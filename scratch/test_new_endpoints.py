import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main_api import app

sys.stdout.reconfigure(encoding='utf-8')

client = TestClient(app)
game_id = "0022400001"

print(f"Testing game endpoints for game_id: {game_id}")

# 1. GET /api/games/{game_id}
print("\n--- GET /api/games/{game_id} ---")
r = client.get(f"/api/games/{game_id}")
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("Keys in response:", list(data.keys()))
    print("Home team score:", data["home_team"])
    print("Away team score:", data["away_team"])
    print("Status:", data["status"])
else:
    print("Error:", r.text)

# 2. GET /api/games/{game_id}/play-by-play
print("\n--- GET /api/games/{game_id}/play-by-play ---")
r = client.get(f"/api/games/{game_id}/play-by-play")
print("Status:", r.status_code)
if r.status_code == 200:
    events = r.json()
    print(f"Fetched {len(events)} events.")
    if events:
        print("First event:", events[0])
else:
    print("Error:", r.text)

# 3. GET /api/games/{game_id}/shot-chart
print("\n--- GET /api/games/{game_id}/shot-chart ---")
r = client.get(f"/api/games/{game_id}/shot-chart")
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print(f"Fetched {len(data.get('shots', []))} shots.")
    if data.get('shots'):
        print("First shot:", data['shots'][0])
else:
    print("Error:", r.text)
