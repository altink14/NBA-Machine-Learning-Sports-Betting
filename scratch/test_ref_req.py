import requests
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

url = "https://www.basketball-reference.com/referees/2025_register.html"
print(f"Requesting: {url}")
r = requests.get(url, headers=HEADERS)
print(f"Status Code: {r.status_code}")
print(f"Content Length: {len(r.content)}")
title_match = re.search(r"<title>(.*?)</title>", r.text, re.IGNORECASE)
print(f"Title: {title_match.group(1) if title_match else 'None'}")
print(f"Number of tables in HTML text: {r.text.count('<table')}")
print(f"Number of tables in comments: {r.text.count('<!--')}")
