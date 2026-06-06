import nba_api.stats.endpoints as ep
import inspect

try:
    source_file = inspect.getsourcefile(ep.GravityLeaders)
    print("Source file path:", source_file)
    source_code = inspect.getsource(ep.GravityLeaders)
    print("\nSource code preview:")
    print(source_code[:1000])
except Exception as e:
    print("Error:", e)
