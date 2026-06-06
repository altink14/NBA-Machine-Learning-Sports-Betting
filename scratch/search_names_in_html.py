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
    print("HTML length:", len(r.text))
    for p in ["Jokic", "Curry", "LeBron", "Doncic", "Edwards", "Tatum"]:
        matches = list(re.finditer(p, r.text, re.IGNORECASE))
        if matches:
            print(f"Found player {p}: {len(matches)} times")
            for m in matches[:2]:
                start = max(0, m.start() - 50)
                end = min(len(r.text), m.end() + 50)
                print(f"  Match at {m.start()}: ...{r.text[start:end]}...")
        else:
            print(f"Player {p} NOT found")
