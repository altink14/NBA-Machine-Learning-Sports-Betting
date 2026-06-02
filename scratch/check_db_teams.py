import sqlite3
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(project_root, 'Data', 'TeamData.sqlite')
print("DB Path:", db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' ORDER BY name DESC LIMIT 1")
table_row = cursor.fetchone()
if table_row:
    table_name = table_row[0]
    print("Table:", table_name)
    cursor.execute(f"SELECT DISTINCT TEAM_NAME FROM `{table_name}`")
    teams = [r[0] for r in cursor.fetchall()]
    print("Teams:", teams)
else:
    print("No tables found")
conn.close()
