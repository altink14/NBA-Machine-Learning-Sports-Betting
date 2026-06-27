import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
from src.Utils.nba_stats_client import get_client

sys.stdout.reconfigure(encoding='utf-8')

game_id = "0022400001"
try:
    client = get_client()
    events = client.play_by_play(game_id)
    print(f"Fetched {len(events)} events.")
except Exception as e:
    traceback.print_exc()
