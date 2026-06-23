import requests
from bs4 import BeautifulSoup, Comment

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def inspect():
    url = "https://www.basketball-reference.com/referees/2025_register.html"
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    table = soup.find('table', id='rs_raw')
    if not table:
        print("Table 'rs_raw' not found!")
        return
        
    rows = table.find_all('tr')
    print(f"Total rows in rs_raw: {len(rows)}")
    if len(rows) > 1:
        first_data_row = rows[2] # Skip header row
        print(f"Row data stats: {[td.get('data-stat') for td in first_data_row.find_all(['td', 'th'])]}")
        print(f"Row text content: {[td.text.strip() for td in first_data_row.find_all(['td', 'th'])]}")

if __name__ == "__main__":
    inspect()
