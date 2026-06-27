import os
import shutil

backend_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting"
web_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\basic-saas-starter"

to_delete = [
    # Backend files
    os.path.join(backend_dir, "scrape_bbref.py"),
    os.path.join(backend_dir, "scrape_bbref_deep.py"),
    os.path.join(backend_dir, "src", "Process-Data", "Get_Basketball_Reference_Data.py"),
    os.path.join(backend_dir, "Data", "BasketballReference.sqlite"),
    os.path.join(backend_dir, "Data", "BasketballReference"), # directory
    
    # Web files
    os.path.join(web_dir, "src", "lib", "bbref_data_deep.json"),
]

print("Starting file purge...")
for item in to_delete:
    if os.path.exists(item):
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"Deleted directory: {item}")
            else:
                os.remove(item)
                print(f"Deleted file: {item}")
        except Exception as e:
            print(f"Failed to delete {item}: {e}")
    else:
        print(f"Item not found (already deleted): {item}")

# Let's search for other bbref_*.json files in the web workspace.
for root, dirs, files in os.walk(web_dir):
    for f in files:
        if f.startswith("bbref_") and f.endswith(".json"):
            full_path = os.path.join(root, f)
            try:
                os.remove(full_path)
                print(f"Deleted additional BBR JSON: {full_path}")
            except Exception as e:
                print(f"Failed to delete {full_path}: {e}")
                
print("Purge completed.")
