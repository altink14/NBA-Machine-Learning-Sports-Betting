import sqlite3

db_path = "Data/TeamData.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables in SQLite database:")
for t in tables:
    cursor.execute(f"PRAGMA table_info('{t}')")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"  {t}: {cols}")
conn.close()
