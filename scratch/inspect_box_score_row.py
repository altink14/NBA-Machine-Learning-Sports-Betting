import sqlite3
import json

db_path = "Data/TeamData.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT traditional_json FROM box_scores LIMIT 1")
row = cursor.fetchone()
if row and row[0]:
    data = json.loads(row[0])
    print("Top-level keys:", list(data.keys()))
    if "meta" in data:
        print("meta keys:", list(data["meta"].keys()))
        print("meta content:", data["meta"])
    p = data.get("boxScoreTraditional", {})
    # Recursively find any key with 'status' or 'date' in it
    def search_dict(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if "status" in k.lower() or "date" in k.lower() or "season" in k.lower():
                    print(f"Found {path}.{k}: {v}")
                search_dict(v, f"{path}.{k}")
        elif isinstance(d, list):
            for i, item in enumerate(d):
                search_dict(item, f"{path}[{i}]")
    search_dict(data)
else:
    print("No data.")
conn.close()
