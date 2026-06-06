import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

chunks = [
    "webpack-5e113ee93deeb36a.js",
    "framework-ad7abf312d5d61d4.js",
    "main-196c53f13b153e43.js",
    "pages/_app-e255db8780a7b930.js",
    "68866-2b57036c0cd48a32.js",
    "87014-2d8cd66307c281aa.js",
    "60017-98a64da384ba44f5.js",
    "21293-4d1e0b9907a72bf1.js",
    "10038-9189cce9fcf3a9e6.js",
    "62458-2c193d77ef460d9c.js",
    "18285-f2b3907a5e5dd245.js"
]

for chunk in chunks:
    url = f"https://www.nba.com/_next/static/chunks/{chunk}"
    print(f"\n=================== Chunk {chunk} ===================")
    r = requests.get(url, headers=headers)
    print("JS length:", len(r.text))
    
    # Check for keywords
    for keyword in ["stats.nba.com", "second-spectrum", "aws", "inside-the-game", "stats-prod", "/api/"]:
        matches = [m.start() for m in re.finditer(keyword, r.text, re.IGNORECASE)]
        if matches:
            print(f"  Keyword '{keyword}' found {len(matches)} times")
            for idx in matches[:3]:
                start = max(0, idx - 80)
                end = min(len(r.text), idx + 80)
                print(f"    Match: ...{r.text[start:end]}...")
                
    # Search for any URL-like strings
    urls = re.findall(r'https?://[^\s\'"]+', r.text)
    if urls:
        print("  Found urls:", len(urls))
        for u in urls[:5]:
            print("    ", u)
