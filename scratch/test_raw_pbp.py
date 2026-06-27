import requests

url = "https://stats.nba.com/stats/playbyplayv2"
params = {
    "GameID": "0022400001",
    "StartPeriod": 0,
    "EndPeriod": 10,
}
headers = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

try:
    r = requests.get(url, params=params, headers=headers, timeout=10)
    print("Status code:", r.status_code)
    print("Content-type:", r.headers.get("content-type"))
    if r.status_code == 200:
        data = r.json()
        print("Keys:", list(data.keys()))
        if "resultSets" in data:
            print("resultSets present.")
        elif "resultSet" in data:
            print("resultSet present.")
        else:
            print("Neither present. Raw data prefix:")
            print(str(data)[:1000])
    else:
        print("Response text:", r.text[:1000])
except Exception as e:
    print("Error:", e)
