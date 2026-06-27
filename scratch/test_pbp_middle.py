from nba_api.stats.endpoints import playbyplayv3

game_id = "0022400001"
try:
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    actions = pbp.get_dict().get("game", {}).get("actions", [])
    print("Events in the middle:")
    for idx, act in enumerate(actions[100:108]):
        print(f"  {act.get('clock')} | Q{act.get('period')} | {act.get('teamTricode')} | {act.get('playerName')} | {act.get('description')} | Score: {act.get('scoreHome')}-{act.get('scoreAway')}")
except Exception as e:
    print("Error:", e)
