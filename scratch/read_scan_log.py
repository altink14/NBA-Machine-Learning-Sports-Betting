import os

log_file = r"C:\Users\altin\.gemini\antigravity\brain\86ca385f-8377-4547-9698-ea02ae34656c\.system_generated\tasks\task-6418.log"

if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        print(f.read())
else:
    print("Log file not found.")
