import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

# The javascript file for gravity page
js_url = "https://www.nba.com/_next/static/chunks/pages/inside-the-game/player/gravity-7663df4042baf7ae.js"

print("Fetching JS...")
r = requests.get(js_url, headers=headers)
print("JS length:", len(r.text))

# Let's search for endpoints, APIs, fetch, axios, etc.
# Find strings like "https://" or absolute paths like "/stats/" or "/api/"
urls = re.findall(r'https?://[^\s\'"]+', r.text)
print("URLs found in JS:", len(urls))
for u in urls[:10]:
    print("  ", u)

# Search for stats/api keywords
print("\nSearching for stats/api/nba keywords:")
for keyword in ["stats", "nba", "api", "inside-the-game", "second-spectrum", "aws"]:
    matches = [m.start() for m in re.finditer(keyword, r.text, re.IGNORECASE)]
    print(f"Keyword '{keyword}' found {len(matches)} times")
    if matches:
        for idx in matches[:3]:
            start = max(0, idx - 50)
            end = min(len(r.text), idx + 50)
            print(f"  Match: ...{r.text[start:end]}...")
            
# Let's look for any URLs or paths starting with /stats or stats.
paths = re.findall(r'"(/[a-zA-Z0-9_\-/]+)"', r.text)
print("\nSome paths found in JS:")
for p in list(set(paths))[:20]:
    print("  ", p)
