from bs4 import BeautifulSoup

with open("scratch/ref_index.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), 'html.parser')
    
# Find all select options or menus
selects = soup.find_all('select')
print(f"Number of select elements: {len(selects)}")
for s in selects:
    print(f"Select name: {s.get('name') or s.get('id')}")
    for opt in s.find_all('option')[:10]:
        print(f"  Option: {opt.text.strip()} -> {opt.get('value')}")
        
# Print any tables on the page with their headers
tables = soup.find_all('table')
print(f"Tables on index page: {[t.get('id') for t in tables]}")
for t in tables:
    print(f"Table ID: {t.get('id')}")
    print(f"  Headers: {[th.text for th in t.find_all('th')[:10]]}")
