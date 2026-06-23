import json

def parse_sport_json(sport):
    print(f"\n================= {sport.upper()} =================")
    with open(f"scratch/fd_{sport}_managed_page.json") as f:
        data = json.load(f)
        
    attachments = data.get("attachments", {})
    events = attachments.get("events", {})
    markets = attachments.get("markets", {})
    
    print(f"Events: {len(events)}")
    print(f"Markets: {len(markets)}")
    
    # Let's map markets to their events
    event_markets = {}
    for market_id, market in markets.items():
        event_id = str(market.get("eventId"))
        if event_id not in event_markets:
            event_markets[event_id] = []
        event_markets[event_id].append((market_id, market))
        
    # Print details for real game events
    for event_id, event in events.items():
        name = event.get("name")
        # Ignore futures/awards/player markets
        if any(x in name.lower() for x in ["futures", "awards", "player markets", "draft", "championship"]):
            continue
            
        print(f"\nGame: {name} (Event ID: {event_id})")
        print(f"Open Date: {event.get('openDate')}")
        
        game_markets = event_markets.get(event_id, [])
        print(f"Found {len(game_markets)} markets:")
        for m_id, m in game_markets:
            market_type = m.get("marketType")
            print(f"  - Market: {m.get('marketName')} | Type: {market_type} | ID: {m_id}")
            runners = m.get("runners", [])
            for runner in runners:
                runner_name = runner.get("runnerName")
                handicap = runner.get("handicap")
                # Price info
                win_runner_odds = runner.get("winRunnerOdds", {})
                american_odds = win_runner_odds.get("americanDisplayOdds", {}).get("americanOdds")
                print(f"    * Runner: {runner_name} | Handicap: {handicap} | Odds: {american_odds}")

parse_sport_json("wnba")
parse_sport_json("mlb")
