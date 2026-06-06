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
    
    # Let's search for iframe tags
    iframes = re.findall(r'<iframe[^>]*src="([^"]+)"', r.text)
    print("Iframes found:", iframes)
    
    # Search for any references to flourish
    flourish = re.findall(r'flourish', r.text, re.IGNORECASE)
    print("Flourish occurrences:", len(flourish))
    if flourish:
        # Print lines containing flourish
        for line in r.text.split("\n"):
            if "flourish" in line.lower():
                print("Flourish line:", line[:200])
                
    # Search for stats.nba.com
    stats_nba = re.findall(r'stats\.nba\.com[^\s\'"]*', r.text)
    print("stats.nba.com references:", list(set(stats_nba)))
    
    # Search for JSON URLs in the source
    json_urls = re.findall(r'https?://[^\s\'"]+\.json', r.text)
    print("JSON URLs:", list(set(json_urls))[:10])
