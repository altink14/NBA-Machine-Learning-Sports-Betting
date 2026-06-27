import os

base_dir = r"C:\Users\altin\.gemini\antigravity\brain\86ca385f-8377-4547-9698-ea02ae34656c\.system_generated"

if os.path.exists(base_dir):
    for root, dirs, files in os.walk(base_dir):
        print(f"Directory: {root}")
        for f in files:
            full_path = os.path.join(root, f)
            try:
                print(f"  {f} ({os.path.getsize(full_path)} bytes)")
            except Exception as e:
                print(f"  {f} (error: {e})")
else:
    print("Base directory not found.")
