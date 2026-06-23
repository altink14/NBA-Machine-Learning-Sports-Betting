import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.fanduel.com/"
}

EVENT_API_URL = "https://sbapi.nj.sportsbook.fanduel.com/api/event-page"

def test_tabs(event_id):
    tabs_to_test = ["player-points", "player-assists", "player-rebounds", "player-threes", "player-combos", "quick-bets"]
    for tab in tabs_to_test:
        params = {
            "betexRegion": "GBR",
            "capiJurisdiction": "intl",
            "currencyCode": "USD",
            "exchangeLocale": "en_US",
            "includePrices": "true",
            "language": "en",
            "priceHistory": "1",
            "regionCode": "NAMERICA",
            "_ak": "FhMFpcPWXMeyZxOx",
            "eventId": event_id,
            "tab": tab
        }
        try:
            r = requests.get(EVENT_API_URL, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                markets = data.get("attachments", {}).get("markets", {})
                print(f"Tab '{tab}': found {len(markets)} markets")
            else:
                print(f"Tab '{tab}' failed with status: {r.status_code}")
        except Exception as e:
            print(f"Tab '{tab}' error: {e}")

print("Testing WNBA Event 35737435...")
test_tabs("35737435")
