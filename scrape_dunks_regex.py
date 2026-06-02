import requests
import json
import re

url = "https://www.nba.com/stats/players/dunk-scores"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

try:
    response = requests.get(url, headers=headers, timeout=15)
    print("Status:", response.status_code)
    
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.DOTALL)
    if match:
        print("Found __NEXT_DATA__!")
        data = json.loads(match.group(1))
        props = data.get("props", {})
        page_props = props.get("pageProps", {})
        print("Keys in pageProps:", list(page_props.keys()))
        
        # Save json to file
        with open("dunk_data.json", "w", encoding="utf-8") as f:
            json.dump(page_props, f, indent=2)
        print("Wrote pageProps to dunk_data.json")
    else:
        print("No __NEXT_DATA__ found.")
        # Let's check for any script elements
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', response.text, re.DOTALL)
        print("Found script elements:", len(scripts))
except Exception as e:
    print("Error:", e)
