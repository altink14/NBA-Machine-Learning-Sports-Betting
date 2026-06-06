import nba_api.stats.endpoints as ep
import inspect

all_endpoints = []
for name, obj in inspect.getmembers(ep):
    if inspect.isclass(obj):
        all_endpoints.append(name)

print("Total endpoints:", len(all_endpoints))

# Filter endpoints matching keywords
keywords = ["grav", "lev", "diff", "shot", "track", "leader"]
for kw in keywords:
    matches = [name for name in all_endpoints if kw.lower() in name.lower()]
    print(f"\nEndpoints matching '{kw}':")
    for m in matches:
        print("  ", m)
