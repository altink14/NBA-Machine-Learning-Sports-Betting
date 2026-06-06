import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

url = "https://www.nba.com/_next/static/chunks/18285-f2b3907a5e5dd245.js"
r = requests.get(url, headers=headers)
text = r.text

print("Length of chunk text:", len(text))

# Let's search for fetch calls or similar
fetch_matches = [m.start() for m in re.finditer(r'fetch', text, re.IGNORECASE)]
print("\n'fetch' occurrences:", len(fetch_matches))
for idx in fetch_matches:
    print(f"  Fetch around {idx}: ...{text[idx-40:idx+80]}...")

# Search for any URL paths in strings (starts with slash and is followed by stats/data)
paths = re.findall(r'"(/[a-zA-Z0-9_\-/]+)"', text)
print("\nPaths found in chunk 18285:")
for p in list(set(paths)):
    print("  ", p)

# Search for "http" or "https" links
http_links = re.findall(r'https?://[a-zA-Z0-9_\-/\.\?=&]+', text)
print("\nHTTP links found:")
for link in list(set(http_links)):
    print("  ", link)

# Search for "gravity", "leverage", "shotDifficulty", "shot-difficulty"
for k in ["gravity", "leverage", "difficulty", "inside", "season", "query"]:
    matches = [m.start() for m in re.finditer(k, text, re.IGNORECASE)]
    print(f"\nKeyword '{k}' matches: {len(matches)}")
    for idx in matches[:5]:
        print(f"  Match around {idx}: ...{text[max(0, idx-60):min(len(text), idx+100)]}...")
