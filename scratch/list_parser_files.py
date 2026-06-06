import os
import re

dir_path = r"C:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting\venv\Lib\site-packages\nba_api\stats\endpoints\_parsers"
files = os.listdir(dir_path)
print("Files in _parsers folder:", len(files))
for f in files:
    if any(kw in f.lower() for kw in ["gravity", "leverage", "difficulty", "dunk", "leader"]):
        print("  Matching file:", f)
        # print the contents of the file
        with open(os.path.join(dir_path, f), "r", encoding="utf-8") as file_obj:
            print("--- Contents ---")
            print(file_obj.read())
            print("----------------")
