from nba_api.stats.endpoints import playbyplayv2

game_id = "0022400001"
try:
    pbp = playbyplayv2.PlayByPlayV2(
        game_id=game_id,
        start_period=0,
        end_period=10
    )
    print("Success!")
    print("Response status:", pbp.nba_response.status_code)
    print("Response headers:", pbp.nba_response.headers)
except Exception as e:
    import traceback
    print("Error:", e)
    # Check if there is an nba_response object on the exception
    traceback.print_exc()
