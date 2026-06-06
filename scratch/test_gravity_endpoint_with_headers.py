from nba_api.stats.endpoints import GravityLeaders
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

try:
    print("Calling GravityLeaders with custom headers...")
    gl = GravityLeaders(season="2024-25", season_type_all_star="Regular Season", headers=headers)
    data = gl.get_dict()
    print("Success! Keys in response dict:", list(data.keys()))
    
    out_path = "C:/Users/altin/.gemini/antigravity/brain/692fbb13-ae06-4316-ac49-429ad1aa8b0b/scratch/gravity_endpoint_response.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved response to {out_path}")
except Exception as e:
    print("Error calling GravityLeaders:", e)
