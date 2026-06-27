from nba_api.stats.endpoints import playbyplayv3

game_id = "0022400001"
try:
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    data = pbp.get_dict()
    print("Success V3!")
    print("Keys in V3 response:", list(data.keys()))
    if "playByPlay" in data:
        print("playByPlay keys:", list(data["playByPlay"].keys()))
        events = data["playByPlay"].get("actions", [])
        print(f"Number of events: {len(events)}")
        if events:
            print("First event sample:")
            print(events[0])
except Exception as e:
    import traceback
    print("Error:", e)
    traceback.print_exc()
