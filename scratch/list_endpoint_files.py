import os

path = r"C:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting\venv\Lib\site-packages\nba_api\stats\endpoints"
files = [f for f in os.listdir(path) if f.endswith(".py") and not f.startswith("_")]
print("Total endpoint files:", len(files))
print(sorted(files))
