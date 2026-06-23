import requests
from bs4 import BeautifulSoup, Comment
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def inspect():
    url = "https://www.basketball-reference.com/referees/2025.html"
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    # Print direct tables
    tables = soup.find_all('table')
    print(f"Direct tables found: {[t.get('id') for t in tables]}")
    
    # Print commented tables
    commented_ids = []
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if '<table' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            commented_tables = comment_soup.find_all('table')
            commented_ids.extend([t.get('id') for t in commented_tables if t.get('id')])
            
    print(f"Commented tables found: {commented_ids}")
    
    # Try to find 'officians' or 'referees' or similar key in comments
    for cid in commented_ids:
        print(f"Found commented table ID: {cid}")

if __name__ == "__main__":
    inspect()
