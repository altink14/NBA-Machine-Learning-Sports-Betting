"""
bootstrap_db.py
===============
Production boot step: make sure the stats databases exist before the API
starts. Databases are data artifacts (too big for git), so in production
they are downloaded once from a snapshot archive and then live on the
mounted volume.

Environment:
  DB_SNAPSHOT_URL  — URL of a .tar.gz containing Data/*.sqlite
                     (e.g. a GitHub release asset). If unset, or the
                     databases already exist, this is a no-op.
"""

import os
import sys
import tarfile
import tempfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
MAIN_DB = os.path.join(DATA_DIR, "TeamData.sqlite")


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(MAIN_DB):
        size_mb = os.path.getsize(MAIN_DB) / 1e6
        print(f"[bootstrap] TeamData.sqlite present ({size_mb:.0f} MB) — skipping download.")
        return 0

    url = os.environ.get("DB_SNAPSHOT_URL", "").strip()
    if not url:
        print("[bootstrap] WARNING: no database and no DB_SNAPSHOT_URL set. "
              "The API will start but stats endpoints will be empty.")
        return 0

    print(f"[bootstrap] Downloading database snapshot from {url} ...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        urllib.request.urlretrieve(url, tmp_path)
        size_mb = os.path.getsize(tmp_path) / 1e6
        print(f"[bootstrap] Downloaded {size_mb:.0f} MB. Extracting ...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Only extract flat sqlite files into Data/ — no paths from the archive.
                if not member.isfile() or not member.name.endswith(".sqlite"):
                    continue
                member.name = os.path.basename(member.name)
                tar.extract(member, DATA_DIR)
                print(f"[bootstrap]   extracted {member.name}")
        print("[bootstrap] Done.")
        return 0
    except Exception as exc:
        print(f"[bootstrap] ERROR: snapshot download failed: {exc}")
        # Don't block the API from starting; endpoints will report emptiness.
        return 0
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
