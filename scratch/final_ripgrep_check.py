import os
import re

backend_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting"
web_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\basic-saas-starter"

search_terms = [
    r"basketball-reference",
    r"basketball_reference",
    r"bbref_data",
    r"bbref_",
    r"get_bbr_slug",
    r"BasketballReference",
    r"scrape_bbref",
    r"antetgi01",
    r"brunsja01",
    r"jamesle01",
    r"townska01"
]

ignore_dirs = {
    ".git", "node_modules", ".next", "venv", "nba_cache", "scratch", "player_cache", ".agents", ".gemini",
    "__pycache__", "Data"
}

ignore_files = {
    "final_ripgrep_check.py", "purge_files.py", "delete_unused_api.py", "delete_flask.py", "read_scan_log.py"
}

compiled_regexes = [re.compile(term, re.IGNORECASE) for term in search_terms]

print("Scanning for Basketball-Reference references...")
matches_found = 0

def scan_dir(directory):
    global matches_found
    for root, dirs, files in os.walk(directory):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        
        for file in files:
            if file in ignore_files:
                continue
            
            # Skip common binary and build files
            if file.endswith(('.sqlite', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.zip', '.tar', '.gz', '.pyc', '.db')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        for term, regex in zip(search_terms, compiled_regexes):
                            if regex.search(line):
                                print(f"MATCH: {file_path}:{line_num} (Term: {term}) -> {line.strip()[:120]}")
                                matches_found += 1
            except Exception as e:
                pass

print("\n--- Scanning Backend ---")
scan_dir(backend_dir)

print("\n--- Scanning Web Frontend ---")
scan_dir(web_dir)

print(f"\nScan complete. Total matches found: {matches_found}")
