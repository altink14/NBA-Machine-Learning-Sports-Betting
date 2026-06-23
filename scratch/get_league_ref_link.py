import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_ref_link():
    url = "https://www.basketball-reference.com/leagues/NBA_2025.html"
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    links = soup.find_all('a')
    ref_links = []
    for link in links:
        href = link.get('href', '')
        if 'referee' in href or 'ref' in href.lower():
            ref_links.append((link.text.strip(), href))
            
    for text, href in ref_links:
        # Avoid console encoding crashes on Windows by encoding to ascii (ignore errors)
        text_clean = text.encode('ascii', errors='ignore').decode('ascii')
        print(f"Link text: '{text_clean}' -> URL: '{href}'")

if __name__ == "__main__":
    get_ref_link()
