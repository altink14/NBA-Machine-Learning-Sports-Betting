import sqlite3
import os

db_path = os.path.join('Data', 'TeamData.sqlite')
print("DB Path exists:", os.path.exists(db_path))

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in c.fetchall()]
print("Tables:", tables)

for table in tables[:10]:
    try:
        c.execute(f"SELECT COUNT(*) FROM `{table}`")
        cnt = c.fetchone()[0]
        print(f"Table {table}: {cnt} rows")
    except Exception as e:
        print(f"Error counting {table}: {e}")

try:
    c.execute("SELECT player_id, full_name, is_active FROM players LIMIT 5")
    print("Sample players:", c.fetchall())
except Exception as e:
    print("Error querying players:", e)

conn.close()
