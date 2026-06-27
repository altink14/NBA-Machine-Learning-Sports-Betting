import sqlite3

db_path = "Data/TeamData.sqlite"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if column already exists
    cursor.execute("PRAGMA table_info(box_scores)")
    columns = [col[1] for col in cursor.fetchall()]
    if "pbp_json" not in columns:
        print("Adding pbp_json column to box_scores table...")
        cursor.execute("ALTER TABLE box_scores ADD COLUMN pbp_json TEXT")
        conn.commit()
        print("Column added successfully.")
    else:
        print("pbp_json column already exists.")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
