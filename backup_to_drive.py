"""
backup_to_drive.py
==================
Back up everything in Data/ that git does not hold to another drive, verified.

    venv/Scripts/python.exe backup_to_drive.py              # to D:\\BettingBuddy-Backup\\<today>
    venv/Scripts/python.exe backup_to_drive.py --dest-root E:\\Backups

WHY. The repo lives under OneDrive, and OneDrive's quota is full, so nothing
here is being backed up (found 2026-09-23). The code is on GitHub; the data is
not: the prediction ledger (OddsData.sqlite, the public track record, which
can never be recreated), the 30-season archive (TeamData.sqlite), the NFL
archive, the sealed model's training databases, and Data/nba_cache, the raw
stats.nba.com responses the archive is rebuilt from. The owner keeps backups
on an external drive (D:, a WD My Passport on USB).

WHAT IT DOES
  - Databases: VACUUM INTO, which SQLite guarantees is a consistent copy even
    while the scheduled jobs are writing (a plain file copy of a WAL database
    can be torn or miss committed data), then PRAGMA quick_check on the copy.
  - Caches: packed into one .tar.gz each and re-read in full. D: is exFAT with
    1 MB blocks, so ~122k small cache files copied loose would take ~120 GB.
    Files whose full path passes Windows' 260-character limit are read through
    the long-path prefix instead of being skipped.
  - The sealed model's untracked working files (Models/candidate_2026-08/work).
  - README.txt with a SHA-256 for every file and restore steps.
  Nothing in the repo is modified. .env files (API keys) are never copied.

First full run 2026-09-23: 4.2 GB, about 20 minutes (TeamData ~5 min, the
cache ~13 min). Exit 0 = complete and verified; 1 = failed (the log says why);
2 = the destination drive is not connected.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import tarfile
import time
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))

#: Databases worth keeping. Test fixtures (test_*.sqlite) are not.
DATABASES = ["OddsData.sqlite", "TeamData.sqlite", "NflData.sqlite",
             "retrain_features.sqlite", "retrain_training.sqlite", "dataset.sqlite"]
SMALL_FILES = ["nba-2023-UTC.csv", "nba-2024-UTC.csv", "nba-2025-UTC.csv"]
MIN_FREE_BYTES = 12 * 1024 ** 3

_B = chr(92)  # a backslash, built rather than typed: shells mangle "\\?\"


def _long(path: str) -> str:
    """Windows long-path form, so >260-character cache names can be opened."""
    if os.name != "nt" or path.startswith(_B + _B):
        return path
    return _B + _B + "?" + _B + os.path.abspath(path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Backup:
    def __init__(self, repo: str, dest: str):
        self.repo = repo
        self.data = os.path.join(repo, "Data")
        self.dest = dest
        self.rows = []  # (name, bytes, sha256, note)

    def log(self, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        with open(os.path.join(self.dest, "backup.log"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def database(self, name: str) -> None:
        src = os.path.join(self.data, name)
        if not os.path.exists(src):
            self.log(f"{name}: not present, skipped")
            return
        dst = os.path.join(self.dest, name)
        t0 = time.time()
        con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            con.execute("VACUUM INTO ?", (dst,))
        finally:
            con.close()
        chk = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        try:
            verdict = chk.execute("PRAGMA quick_check").fetchone()[0]
            tables = {r[0] for r in chk.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            extra = ""
            if "predictions_log" in tables:
                extra += f", predictions_log {chk.execute('SELECT COUNT(*) FROM predictions_log').fetchone()[0]} rows"
            if "ledger" in tables:
                extra += f", ledger {chk.execute('SELECT COUNT(*) FROM ledger').fetchone()[0]} rows"
            if name == "TeamData.sqlite" and "box_scores" in tables:
                extra += f", {chk.execute('SELECT COUNT(DISTINCT game_id) FROM box_scores').fetchone()[0]:,} games"
        finally:
            chk.close()
        if verdict != "ok":
            raise RuntimeError(f"{name}: quick_check on the copy said {verdict!r}")
        size = os.path.getsize(dst)
        self.rows.append((name, size, _sha256(dst), f"quick_check ok, {len(tables)} tables{extra}"))
        self.log(f"{name}: {size / 1024 ** 2:,.1f} MB, quick_check ok{extra} ({time.time() - t0:.0f}s)")

    def small_file(self, name: str) -> None:
        src = os.path.join(self.data, name)
        if os.path.exists(src):
            dst = os.path.join(self.dest, name)
            shutil.copy2(src, dst)
            self.rows.append((name, os.path.getsize(dst), _sha256(dst), "copied as-is"))

    def archive(self, src_dir: str, archive: str, root: str, what: str) -> None:
        if not os.path.isdir(src_dir):
            self.log(f"{archive}: {src_dir} missing, skipped")
            return
        t0 = time.time()
        dst = os.path.join(self.dest, archive)
        packed, missing = 0, 0
        walk_root = _long(src_dir)
        with tarfile.open(dst, "w:gz", compresslevel=6) as tar:
            for cur, _dirs, files in os.walk(walk_root):
                for f in sorted(files):
                    if ".tmp" in f:          # a writer's half-finished file
                        continue
                    full = os.path.join(cur, f)
                    rel = os.path.relpath(full, walk_root)
                    try:
                        tar.add(full, arcname=os.path.join(root, rel).replace(_B, "/"), recursive=False)
                        packed += 1
                    except FileNotFoundError:
                        missing += 1         # deleted between listing and reading
                    if packed and packed % 20000 == 0:
                        self.log(f"  {root}: {packed:,} files packed")
        with tarfile.open(dst, "r:gz") as tar:
            held = sum(1 for t in tar if t.isfile())
        if held != packed:
            raise RuntimeError(f"{archive}: packed {packed} files but the archive holds {held}")
        size = os.path.getsize(dst)
        note = f"{packed:,} files, re-read and verified; {what}"
        if missing:
            note += f" ({missing} vanished while packing: a live cache being refreshed)"
        self.rows.append((archive, size, _sha256(dst), note))
        self.log(f"{archive}: {packed:,} files, {size / 1024 ** 2:,.1f} MB ({time.time() - t0:.0f}s)")

    def readme(self) -> None:
        total = sum(r[1] for r in self.rows)
        with open(os.path.join(self.dest, "README.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"Betting Buddy data backup, {datetime.now():%Y-%m-%d %H:%M}\n"
                     f"From: {self.data} (and Models{_B}candidate_2026-08{_B}work)\n"
                     "Databases copied with VACUUM INTO and checked with PRAGMA quick_check;\n"
                     "every archive re-read in full. NOT included: .env files (API keys).\n\nFILES\n")
            for name, size, digest, note in self.rows:
                fh.write(f"  {name:32} {size / 1024 ** 2:>10,.1f} MB  sha256 {digest}\n      {note}\n")
            fh.write(f"\nTOTAL {total / 1024 ** 3:.2f} GB\n\n"
                     "HOW TO RESTORE (stop the scheduled tasks and the API first)\n"
                     "  1. Copy the .sqlite files into the backend's Data folder, replacing the\n"
                     "     damaged ones; delete any matching -wal / -shm files there first.\n"
                     "  2. Unpack the caches into Data:   tar -xzf nba_cache.tar.gz -C <backend>\\Data\n"
                     "  3. Unpack the model work files:\n"
                     "       tar -xzf candidate_2026-08_work.tar.gz -C <backend>\\Models\\candidate_2026-08\n"
                     "  4. Before trusting a file, compare its hash with the list above:\n"
                     "       certutil -hashfile TeamData.sqlite SHA256\n")
        self.log(f"DONE: {len(self.rows)} files, {total / 1024 ** 3:.2f} GB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Back up Data/ to another drive, verified.")
    ap.add_argument("--dest-root", default=r"D:" + _B + "BettingBuddy-Backup",
                    help="Folder that receives one dated subfolder per run (default D:\\BettingBuddy-Backup)")
    ap.add_argument("--repo", default=REPO, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    drive = os.path.splitdrive(os.path.abspath(args.dest_root))[0] or args.dest_root
    if not os.path.exists(drive + _B if drive.endswith(":") else drive):
        print(f"Destination drive {drive} is not connected. Plug it in and run again.")
        return 2
    os.makedirs(args.dest_root, exist_ok=True)
    free = shutil.disk_usage(args.dest_root).free
    if free < MIN_FREE_BYTES:
        print(f"Only {free / 1024 ** 3:.1f} GB free at {args.dest_root}; need about 12 GB.")
        return 1

    # One folder per run; a second run the same day gets its own, never overwrites.
    stamp = datetime.now().strftime("%Y-%m-%d")
    dest = os.path.join(args.dest_root, stamp)
    n = 2
    while os.path.exists(dest):
        dest = os.path.join(args.dest_root, f"{stamp}_{n}")
        n += 1
    os.makedirs(dest)

    b = Backup(args.repo, dest)
    start = time.time()
    try:
        b.log(f"Backup of {b.data} -> {dest}")
        for name in DATABASES:
            b.database(name)
        for name in SMALL_FILES:
            b.small_file(name)
        b.archive(os.path.join(b.data, "nba_cache"), "nba_cache.tar.gz", "nba_cache",
                  "every raw stats.nba.com response; the source for rebuilding the archive")
        b.archive(os.path.join(b.data, "nfl_cache"), "nfl_cache.tar.gz", "nfl_cache", "NFL download cache")
        b.archive(os.path.join(args.repo, "Models", "candidate_2026-08", "work"),
                  "candidate_2026-08_work.tar.gz", "work",
                  "untracked sealed-model working files (validation predictions etc.)")
        b.readme()
    except Exception as exc:
        b.log(f"FAILED: {exc}. This folder is incomplete: {dest}")
        return 1
    b.log(f"Finished in {(time.time() - start) / 60:.1f} min: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
