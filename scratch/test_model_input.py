import sqlite3
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(project_root, 'Data', 'TeamData.sqlite')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '202%' ORDER BY name DESC LIMIT 1")
table_name = cursor.fetchone()[0]
df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn, index_col="index")
conn.close()

# The game: home = 'Oklahoma City Thunder', away = 'San Antonio Spurs'
home_team = 'Oklahoma City Thunder'
away_team = 'San Antonio Spurs'

home_stats_rows = df[df['TEAM_NAME'] == home_team]
away_stats_rows = df[df['TEAM_NAME'] == away_team]

home_stats = home_stats_rows.iloc[0].copy()
away_stats = away_stats_rows.iloc[0].copy()

# Concatenate home and away team statistics
game_data = pd.concat([home_stats, away_stats.rename(index=lambda x: x + '.1')])
game_data['Days-Rest-Home'] = float(7)
game_data['Days-Rest-Away'] = float(7)

game_data_list = [game_data]
frame_ml = pd.DataFrame(game_data_list)
print("frame_ml shape:", frame_ml.shape)
print("frame_ml dtypes counts:\n", frame_ml.dtypes.value_counts())

cols_to_drop = [
    'TEAM_ID', 'TEAM_NAME', 'Date', 'index',
    'TEAM_ID.1', 'TEAM_NAME.1', 'Date.1', 'index.1',
    'Score', 'Home-Team-Win', 'OU-Cover', 'OU'
]
frame_for_model = frame_ml.drop(columns=[c for c in cols_to_drop if c in frame_ml.columns], errors='ignore')
print("frame_for_model shape:", frame_for_model.shape)

non_numeric = [col for col in frame_for_model.columns if not pd.api.types.is_numeric_dtype(frame_for_model[col])]
print("Non-numeric columns count:", len(non_numeric))
if non_numeric:
    print("Non-numeric columns:", non_numeric)
