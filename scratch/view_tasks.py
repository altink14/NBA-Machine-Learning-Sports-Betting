import os

tasks_dir = r"C:\Users\altin\.gemini\antigravity\brain\86ca385f-8377-4547-9698-ea02ae34656c\.system_generated\tasks"
if os.path.exists(tasks_dir):
    files = sorted(os.listdir(tasks_dir))
    print(f"Task log files ({len(files)} total):")
    # Print the last 15 files
    for f in files[-15:]:
        full_path = os.path.join(tasks_dir, f)
        print(f"  {f} ({os.path.getsize(full_path)} bytes)")
else:
    print("Tasks dir not found.")
