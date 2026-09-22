"""
publish_db_snapshot.py
======================
Build the database archive production boots from, consistently.

WHY THIS EXISTS AND WHY `tar` WAS NOT ENOUGH. DEPLOY.md told you to run

    tar czf db-snapshot.tar.gz -C Data TeamData.sqlite OddsData.sqlite

against the live databases. TeamData.sqlite is in **WAL mode**. In WAL mode a
committed transaction lives in the `-wal` sidecar until something checkpoints
it into the main file, and that command copies the main file only. Run it in
the hour after the 9am job and the snapshot silently omits the newest data; run
it while a write is in flight and the copy can be torn. Either way you get a
plausible-looking tarball and no error, and you find out in production.

`VACUUM INTO` is the fix and it is not a workaround: SQLite defines it as
producing a consistent copy of the database as of the moment it runs, safe to
use while other connections are active, fully checkpointed and defragmented.
It also drops free pages, so the copy is usually smaller than the original --
which matters here, because this archive is downloaded on every cold boot.

WHAT GOES IN. Only what `main_api:app` reads at runtime:

  TeamData.sqlite   the archive and the model's feature source
  OddsData.sqlite   the prediction ledger

Not `dataset.sqlite` (the old tar command included it; it is training-only),
not `retrain_*.sqlite`, not the `test_*.sqlite` fixtures, and not
NflData.sqlite unless you pass --with-nfl: NFL is parked and unlinked, and it
is 1.28 GB of download on every boot for endpoints nothing calls.

This does NOT upload. Publishing is a deliberate act against a real release,
so it prints the exact command and stops.

Usage:
    venv/Scripts/python.exe publish_db_snapshot.py
    venv/Scripts/python.exe publish_db_snapshot.py --with-nfl --out D:/tmp
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "Data")

#: (filename, required) — what production actually opens.
RUNTIME_DBS = [("TeamData.sqlite", True), ("OddsData.sqlite", True)]
NFL_DB = ("NflData.sqlite", False)

RELEASE_TAG = "db-snapshot-v1"
ARCHIVE_NAME = "db-snapshot.tar.gz"


def mb(path: str) -> float:
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


def consistent_copy(src: str, dst: str) -> tuple[float, float]:
    """VACUUM INTO: a checkpointed, defragmented, point-in-time copy.

    Opened read-write because VACUUM INTO is a statement the source connection
    executes; it does not modify the source, but it is not available on a
    `mode=ro` connection.
    """
    if os.path.exists(dst):
        os.remove(dst)
    before = mb(src)
    conn = sqlite3.connect(src)
    try:
        conn.execute("PRAGMA busy_timeout = 120000")
        conn.execute("VACUUM INTO ?", (dst,))
    finally:
        conn.close()
    return before, mb(dst)


def verify(path: str) -> str:
    """Open the copy and make sure it is a database, not a pile of bytes."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        ok = conn.execute("PRAGMA quick_check").fetchone()[0]
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    finally:
        conn.close()
    return f"quick_check={ok}, {tables:,} tables"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the production database snapshot.")
    ap.add_argument("--out", default=tempfile.gettempdir(),
                    help="Where to write the archive (default: the temp dir).")
    ap.add_argument("--with-nfl", action="store_true",
                    help="Include NflData.sqlite. NFL is parked; leave this off.")
    ap.add_argument("--keep-copies", action="store_true",
                    help="Keep the intermediate .sqlite copies for inspection.")
    args = ap.parse_args()

    wanted = list(RUNTIME_DBS) + ([NFL_DB] if args.with_nfl else [])
    staging = tempfile.mkdtemp(prefix="db_snapshot_")
    archive = os.path.join(args.out, ARCHIVE_NAME)
    os.makedirs(args.out, exist_ok=True)

    print(f"staging: {staging}")
    print(f"archive: {archive}\n")

    copies = []
    for name, required in wanted:
        src = os.path.join(DATA_DIR, name)
        if not os.path.exists(src):
            if required:
                print(f"MISSING and required: {src}", file=sys.stderr)
                return 1
            print(f"  skipping {name} (not present)")
            continue
        dst = os.path.join(staging, name)
        print(f"  {name}: copying consistently ...", flush=True)
        t = time.time()
        before, after = consistent_copy(src, dst)
        shrink = (1 - after / before) * 100 if before else 0
        print(f"    {before:,.0f} MB -> {after:,.0f} MB "
              f"({shrink:.0f}% smaller, free pages dropped) "
              f"in {time.time() - t:.0f}s")
        print(f"    {verify(dst)}")
        copies.append(dst)

    print("\n  compressing ...", flush=True)
    t = time.time()
    with tarfile.open(archive, "w:gz") as tar:
        for path in copies:
            tar.add(path, arcname=os.path.basename(path))
    print(f"    {mb(archive):,.0f} MB in {time.time() - t:.0f}s")

    if not args.keep_copies:
        for path in copies:
            os.remove(path)
        os.rmdir(staging)

    print(f"\nSnapshot built {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.")
    print("It is NOT published. To publish:\n")
    print(f'    gh release upload {RELEASE_TAG} "{archive}" --clobber\n')
    print("Then make production pick it up: delete TeamData.sqlite from the volume,")
    print("or point DB_SNAPSHOT_URL at a new tag, and redeploy so bootstrap_db.py")
    print("downloads it. A running instance will NOT replace a database it already has.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
