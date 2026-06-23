import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

urls = [
    "https://www.basketball-reference.com/referees/2024_2025.html",
    "https://www.basketball-reference.com/referees/2024-25.html",
    "https://www.basketball-reference.com/referees/2025.html",
    "https://www.basketball-reference.com/referees/NBA_2025.html",
    "https://www.basketball-reference.com/leagues/NBA_2025_referees.html",
    "https://www.basketball-reference.com/leagues/NBA_2025_referee.html"
]

for url in urls:
    r = requests.get(url, headers=HEADERS)
    print(f"URL: {url} -> Status Code: {r.status_code}, Length: {len(r.content)}")
