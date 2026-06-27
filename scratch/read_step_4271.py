import json
import os

filepath = r"C:\Users\altin\.gemini\antigravity\brain\86ca385f-8377-4547-9698-ea02ae34656c\.system_generated\logs\transcript_full.jsonl"
if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                if data.get("step_index") == 4271 and data.get("type") == "USER_INPUT":
                    print(data.get("content"))
                    break
            except Exception as e:
                pass
else:
    print("Transcript not found.")
