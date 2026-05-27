import sqlite3
import pandas as pd

def inspect_database(db_path):
    print(f"=== Inspecting {db_path} ===")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print("Tables:", tables)
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM `{table}` LIMIT 2", conn)
        print(f"\nTable: {table}")
        print("Columns:", list(df.columns))
        print("First row:\n", df.head(1).to_dict(orient='records'))
    conn.close()

if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "Data/TeamData.sqlite"
    inspect_database(db)
