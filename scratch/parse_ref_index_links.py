from bs4 import BeautifulSoup
import re

with open("scratch/ref_index.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), 'html.parser')
    
links = soup.find_all('a')
ref_links = set()
for link in links:
    href = link.get('href', '')
    if href.startswith('/referees/'):
        # Normalize and filter out individual profiles
        if not re.search(r'[a-z]{5,}\d{2}r\.html$', href):
            ref_links.add((link.text.strip(), href))

print(f"Found {len(ref_links)} referee index links:")
for text, href in sorted(ref_links):
    print(f"'{text}' -> {href}")
