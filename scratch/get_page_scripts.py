import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

urls = {
    "shot-difficulty": "https://www.nba.com/inside-the-game/player/shot-difficulty",
    "leverage": "https://www.nba.com/inside-the-game/player/leverage"
}

for name, url in urls.items():
    print(f"\n=================== {name} ===================")
    r = requests.get(url, headers=headers)
    
    # Find all scripts with src
    scripts = re.findall(r'<script[^>]*src="([^"]+)"', r.text)
    print("Scripts with src:")
    for s in scripts:
        if "_next/static/chunks" in s:
            print("  ", s)
