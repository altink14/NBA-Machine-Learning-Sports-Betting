import requests
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.fanduel.com/"
}

# The endpoint used by betfinder R package
API_URL = "https://sbapi.nj.sportsbook.fanduel.com/api/content-managed-page"

def fetch_sport_events(sport):
    params = {
        "betexRegion": "GBR",
        "capiJurisdiction": "intl",
        "currencyCode": "USD",
        "exchangeLocale": "en_US",
        "language": "en",
        "regionCode": "NAMERICA",
        "_ak": "FhMFpcPWXMeyZxOx",
        "page": "CUSTOM",
        "customPageId": sport
    }
    
    print(f"\nFetching FanDuel events for sport: {sport}...")
    try:
        r = requests.get(API_URL, params=params, headers=headers, timeout=10)
        print("Status code:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            # Save the raw JSON response to check its structure
            filename = f"scratch/fd_{sport}_managed_page.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved response to {filename}")
            
            # Print high-level overview of keys
            print("Main keys:", list(data.keys()))
            attachments = data.get("attachments", {})
            print("Attachment keys:", list(attachments.keys()))
            
            events = attachments.get("events", {})
            print(f"Number of events found: {len(events)}")
            
            # Print a few event summaries
            for event_id, event in list(events.items())[:3]:
                print(f"Event ID: {event_id} | Name: {event.get('name')} | Open Date: {event.get('openDate')}")
            
            return data
    except Exception as e:
        print("Error fetching sport:", e)
    return None

# Fetch MLB, WNBA, and NBA
fetch_sport_events("mlb")
fetch_sport_events("wnba")
fetch_sport_events("nba")
