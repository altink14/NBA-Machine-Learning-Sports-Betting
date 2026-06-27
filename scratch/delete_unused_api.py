import os
import shutil

web_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\basic-saas-starter"

to_delete = [
    os.path.join(web_dir, "scripts", "scrape_player.py"),
    os.path.join(web_dir, "src", "app", "api", "player-stats"),
]

for path in to_delete:
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Deleted directory: {path}")
            else:
                os.remove(path)
                print(f"Deleted file: {path}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
    else:
        print(f"Already deleted: {path}")
