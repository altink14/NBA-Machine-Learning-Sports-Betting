import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

url = "https://www.basketball-reference.com/referees/"
r = requests.get(url, headers=HEADERS)
with open("scratch/ref_index.html", "w", encoding="utf-8") as f:
    f.write(r.text)
print("Saved ref index HTML to scratch/ref_index.html")
