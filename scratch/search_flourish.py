import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

urls = {
    "gravity": "https://www.nba.com/inside-the-game/player/gravity?SeasonType=Regular+Season",
    "shot-difficulty": "https://www.nba.com/inside-the-game/player/shot-difficulty",
    "leverage": "https://www.nba.com/inside-the-game/player/leverage"
}

for name, url in urls.items():
    print(f"\n=================== {name} ===================")
    r = requests.get(url, headers=headers)
    
    # Find all occurrences of flourish in the raw HTML text (case-insensitive)
    # Let's print the line and index of match, along with surrounding 200 characters
    for match in re.finditer(r'flourish', r.text, re.IGNORECASE):
        start = max(0, match.start() - 100)
        end = min(len(r.text), match.end() + 100)
        print(f"Match at {match.start()}: ...{r.text[start:end]}...")
