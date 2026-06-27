import json
import os
import re

filepath = r"C:\Users\altin\.gemini\antigravity\brain\86ca385f-8377-4547-9698-ea02ae34656c\.system_generated\logs\transcript_full.jsonl"
if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                if data.get("type") == "USER_INPUT":
                    content = data.get("content", "")
                    # Match words like Phase 1/2/3/4/5
                    if re.search(r'Phase \d', content, re.IGNORECASE) or "master plan" in content.lower():
                        # Find unique mentions of phases in this step
                        matches = re.findall(r'Phase \d[a-z\-]*', content, re.IGNORECASE)
                        if matches:
                            print(f"Step {data.get('step_index')} (Line {idx}) mentions phases: {set(matches)}")
                            # Print first 200 chars of matching steps
                            print(f"Snippet: {content[:300].strip()}...\n")
            except Exception as e:
                pass
else:
    print("Transcript not found.")
