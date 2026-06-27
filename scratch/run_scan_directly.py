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
    "__pycache__", "Data", "nba_env", "venv"
}

ignore_files = {
    "final_ripgrep_check.py", "purge_files.py", "delete_unused_api.py", "delete_flask.py", "read_scan_log.py",
    "run_scan_directly.py", "scan_result.txt", "check_scan_progress.py", "view_tasks.py", "find_log_files.py"
}

compiled_regexes = [re.compile(term, re.IGNORECASE) for term in search_terms]

out_lines = ["Scanning for Basketball-Reference references...\n"]
matches_found = 0

def scan_dir(directory, name):
    global matches_found
    out_lines.append(f"\n--- Scanning {name} ---\n")
    for root, dirs, files in os.walk(directory):
        # Prune
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.') and "env" not in d]
        
        for file in files:
            if file in ignore_files:
                continue
            if file.endswith(('.sqlite', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.zip', '.tar', '.gz', '.pyc', '.db')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        for term, regex in zip(search_terms, compiled_regexes):
                            if regex.search(line):
                                match_str = f"MATCH: {file_path}:{line_num} (Term: {term}) -> {line.strip()[:120]}\n"
                                out_lines.append(match_str)
                                matches_found += 1
            except Exception as e:
                pass

scan_dir(backend_dir, "Backend")
scan_dir(web_dir, "Web Frontend")

out_lines.append(f"\nScan complete. Total matches found: {matches_found}\n")

# Write to file
result_file = os.path.join(os.path.dirname(__file__), "scan_result.txt")
with open(result_file, 'w', encoding='utf-8') as out_f:
    out_f.writelines(out_lines)

# Print result summary to stdout
print("".join(out_lines))
