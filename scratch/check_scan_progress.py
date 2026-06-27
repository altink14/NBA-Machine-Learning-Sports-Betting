import os

backend_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting"
web_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\basic-saas-starter"

ignore_dirs = {
    ".git", "node_modules", ".next", "venv", "nba_cache", "scratch", "player_cache", ".agents", ".gemini",
    "__pycache__", "Data"
}

def count_files(directory, name):
    total_dirs = 0
    total_files = 0
    subdirs = []
    for root, dirs, files in os.walk(directory):
        # Prune
        original_dirs = list(dirs)
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        pruned = set(original_dirs) - set(dirs)
        if pruned:
            # print(f"Pruned in {root}: {pruned}")
            pass
            
        total_dirs += len(dirs)
        total_files += len(files)
        
        # If there are files, check how many
        if len(files) > 100:
            print(f"Large file count in {root}: {len(files)} files")
            
    print(f"{name}: {total_dirs} directories, {total_files} files scanned (excluding pruned ones)")

print("Counting active files...")
count_files(backend_dir, "Backend")
count_files(web_dir, "Web Frontend")
