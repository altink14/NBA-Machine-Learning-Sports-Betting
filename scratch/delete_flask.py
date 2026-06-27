import os
import stat
import shutil

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

backend_dir = r"c:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting"
flask_dir = os.path.join(backend_dir, "Flask")

if os.path.exists(flask_dir):
    try:
        shutil.rmtree(flask_dir, onerror=remove_readonly)
        print(f"Deleted obsolete directory: {flask_dir}")
    except Exception as e:
        print(f"Failed to delete {flask_dir}: {e}")
else:
    print(f"Directory not found: {flask_dir}")
