from nba_api.stats.endpoints import GravityLeaders
import json
import inspect

print("GravityLeaders signature:")
print(inspect.signature(GravityLeaders.__init__))

# Let's try calling it for Regular Season 2024-25
try:
    print("\nCalling GravityLeaders...")
    gl = GravityLeaders(season="2024-25", season_type_all_star="Regular Season")
    # Let's inspect the returned JSON structure
    data = gl.get_dict()
    print("Success! Keys in response dict:", list(data.keys()))
    
    # Save the dictionary to scratch
    out_path = "C:/Users/altin/.gemini/antigravity/brain/692fbb13-ae06-4316-ac49-429ad1aa8b0b/scratch/gravity_endpoint_response.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved response to {out_path}")
    
    # Print the first item in the results if possible
    results = data.get("results", [])
    if results:
        print("\nFirst result item:", results[0])
    else:
        # Check other keys
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0:
                print(f"\nFirst item in key '{k}':", v[0])
except Exception as e:
    print("Error calling GravityLeaders:", e)
