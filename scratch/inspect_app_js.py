import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

url = "https://www.nba.com/_next/static/chunks/pages/_app-e255db8780a7b930.js"
r = requests.get(url, headers=headers)
text = r.text

print("Length of app chunk:", len(text))

# Search for "js/data"
matches = [m.start() for m in re.finditer(r'js/data', text, re.IGNORECASE)]
print("\n'js/data' occurrences:", len(matches))
for idx in matches:
    print(f"  Match around {idx}: ...{text[idx-100:idx+200]}...")

# Search for "stats.nba.com"
nba_matches = [m.start() for m in re.finditer(r'stats\.nba\.com', text, re.IGNORECASE)]
print("\n'stats.nba.com' occurrences:", len(nba_matches))
for idx in nba_matches:
    print(f"  Match around {idx}: ...{text[idx-100:idx+200]}...")
