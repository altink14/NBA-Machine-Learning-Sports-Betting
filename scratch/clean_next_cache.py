import os
import shutil

web_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\basic-saas-starter"
next_dir = os.path.join(web_dir, ".next")

if os.path.exists(next_dir):
    try:
        shutil.rmtree(next_dir)
        print("Successfully deleted .next directory.")
    except Exception as e:
        print(f"Failed to delete .next: {e}")
else:
    print(".next directory does not exist.")
