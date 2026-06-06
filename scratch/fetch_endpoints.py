import requests
import json
import time

headers = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Origin": "https://www.nba.com",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

endpoints = {
    "gravity": "gravityleaders",
    "leverage": "leverageleaders",
    "shot-difficulty": "shotqualityleaders"
}

seasons = ["2024-25", "2025-26"]
season_types = ["Regular Season", "Preseason"]

for name, endpoint in endpoints.items():
    for season in seasons:
        for st in season_types:
            url = f"https://stats.nba.com/stats/{endpoint}"
            params = {
                "LeagueID": "00",
                "Season": season,
                "SeasonType": st
            }
            print(f"\nFetching {name} for {season} {st}...")
            try:
                # Add delay to be nice
                time.sleep(1.5)
                r = requests.get(url, headers=headers, params=params, timeout=15)
                print("Status Code:", r.status_code)
                if r.status_code == 200:
                    data = r.json()
                    out_name = f"C:/Users/altin/.gemini/antigravity/brain/692fbb13-ae06-4316-ac49-429ad1aa8b0b/scratch/{name}_{season.replace('-', '_')}_{st.replace(' ', '_')}.json"
                    with open(out_name, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"  Saved to {out_name}")
                    # Let's inspect some of the items in the response to understand headers
                    if "leaders" in data:
                        print("  Leaders count:", len(data["leaders"]))
                        if len(data["leaders"]) > 0:
                            print("  Sample item keys:", list(data["leaders"][0].keys()))
                    elif "results" in data:
                        print("  Results count:", len(data["results"]))
                    else:
                        print("  Response keys:", list(data.keys()))
                else:
                    print("  Failed:", r.text[:200])
            except Exception as e:
                print("  Error:", e)
