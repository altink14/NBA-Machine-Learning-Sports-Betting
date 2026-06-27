from nba_api.stats.endpoints import playbyplayv3

game_id = "0022400001"
try:
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    data = pbp.get_dict()
    game_data = data.get("game", {})
    print("Keys inside game:", list(game_data.keys()))
    for k, v in game_data.items():
        if isinstance(v, list):
            print(f"  {k}: list of length {len(v)}")
            if len(v) > 0:
                print(f"    First item keys: {list(v[0].keys())}")
                print(f"    First item sample: {v[0]}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        else:
            print(f"  {k}: {v}")
except Exception as e:
    print("Error:", e)
