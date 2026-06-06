import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

urls = {
    "shot-difficulty-chunk": "https://www.nba.com/_next/static/chunks/42674-71e8ef5a4327a1c8.js",
    "leverage-chunk": "https://www.nba.com/_next/static/chunks/8187-e70e002ecd2ffd83.js"
}

for name, url in urls.items():
    print(f"\n=================== {name} ===================")
    r = requests.get(url, headers=headers)
    text = r.text
    print("JS length:", len(text))
    
    # Search for keywords like stats, endpoint, fetch, leaders, leaders, lvg, shdiff
    for keyword in ["stats", "leaders", "fetch", "difficulty", "leverage", "shot", "endpoint", "api"]:
        matches = [m.start() for m in re.finditer(keyword, text, re.IGNORECASE)]
        if matches:
            print(f"  Keyword '{keyword}' found {len(matches)} times")
            for idx in matches[:3]:
                start = max(0, idx - 80)
                end = min(len(text), idx + 80)
                print(f"    Match: ...{text[start:end]}...")
                
    # Find anything inside double quotes starting with a word and ending with leaders
    leaders_find = re.findall(r'"([a-zA-Z_]+leaders)"', text, re.IGNORECASE)
    if leaders_find:
        print("  Leaders patterns found:", list(set(leaders_find)))
