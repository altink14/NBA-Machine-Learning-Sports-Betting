import requests
from bs4 import BeautifulSoup, Comment

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_links():
    url = "https://www.basketball-reference.com/referees/"
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    links = soup.find_all('a')
    ref_links = []
    for link in links:
        href = link.get('href', '')
        if href.startswith('/referees/') and '99r.html' not in href:
            ref_links.append((link.text.strip(), href))
            
    print(f"Total seasonal/index links found: {len(ref_links)}")
    for text, href in ref_links:
        print(f"Link: {text} -> {href}")

if __name__ == "__main__":
    get_links()
