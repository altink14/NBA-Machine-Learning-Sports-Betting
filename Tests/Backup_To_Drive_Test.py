"""backup_to_drive.py copies a fake repo's data completely and verifiably.

Everything runs in temp directories; the real Data/ and D: are never touched.
"""

import os
import shutil
import sqlite3
import tarfile
import tempfile
import unittest

import backup_to_drive as btd


class BackupTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="backup_test_")
        self.repo = os.path.join(self.tmp, "repo")
        data = os.path.join(self.repo, "Data")
        os.makedirs(os.path.join(data, "nba_cache"))
        os.makedirs(os.path.join(self.repo, "Models", "candidate_2026-08", "work"))
        c = sqlite3.connect(os.path.join(data, "OddsData.sqlite"))
        c.execute("CREATE TABLE predictions_log (id INTEGER PRIMARY KEY, v TEXT)")
        c.execute("CREATE TABLE ledger (id INTEGER PRIMARY KEY)")
        c.execute("INSERT INTO predictions_log (v) VALUES ('a'), ('b')")
        c.commit()
        c.execute("PRAGMA journal_mode=WAL")  # a live WAL database, like TeamData
        c.execute("INSERT INTO predictions_log (v) VALUES ('committed, still in the WAL')")
        c.commit()
        self.live = c  # held open, so the last row sits in the -wal file
        for i in range(5):
            with open(os.path.join(data, "nba_cache", f"box_{i}.json"), "w") as fh:
                fh.write('{"x": %d}' % i)
        with open(os.path.join(data, "nba_cache", "box_9.json.tmp"), "w") as fh:
            fh.write("half-written")
        with open(os.path.join(self.repo, "Models", "candidate_2026-08", "work", "valpreds.npy"), "wb") as fh:
            fh.write(b"\x00" * 64)
        with open(os.path.join(data, "nba-2025-UTC.csv"), "w") as fh:
            fh.write("date,home,away\n")
        self.dest_root = os.path.join(self.tmp, "backups")
        self.old_min = btd.MIN_FREE_BYTES
        btd.MIN_FREE_BYTES = 0

    def tearDown(self):
        btd.MIN_FREE_BYTES = self.old_min
        self.live.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_backup(self):
        return btd.main(["--dest-root", self.dest_root, "--repo", self.repo])

    def only_folder(self):
        (name,) = os.listdir(self.dest_root)
        return os.path.join(self.dest_root, name)

    def test_complete_and_verified(self):
        self.assertEqual(self.run_backup(), 0)
        out = self.only_folder()
        c = sqlite3.connect(os.path.join(out, "OddsData.sqlite"))
        try:
            # VACUUM INTO includes the row still sitting in the WAL file.
            self.assertEqual(c.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0], 3)
        finally:
            c.close()
        with tarfile.open(os.path.join(out, "nba_cache.tar.gz")) as t:
            names = sorted(m.name for m in t if m.isfile())
        self.assertEqual(names, [f"nba_cache/box_{i}.json" for i in range(5)],
                         "every cache file, and not the half-written .tmp")
        with tarfile.open(os.path.join(out, "candidate_2026-08_work.tar.gz")) as t:
            self.assertEqual([m.name for m in t if m.isfile()], ["work/valpreds.npy"])
        readme = open(os.path.join(out, "README.txt"), encoding="utf-8").read()
        self.assertIn("predictions_log 3 rows", readme)
        self.assertIn("HOW TO RESTORE", readme)
        self.assertNotIn(".env", [n for n in os.listdir(out)])

    def test_a_second_run_never_overwrites_the_first(self):
        self.assertEqual(self.run_backup(), 0)
        self.assertEqual(self.run_backup(), 0)
        self.assertEqual(len(os.listdir(self.dest_root)), 2)

    def test_the_source_is_not_modified(self):
        before = sorted(os.listdir(os.path.join(self.repo, "Data", "nba_cache")))
        self.run_backup()
        self.assertEqual(sorted(os.listdir(os.path.join(self.repo, "Data", "nba_cache"))), before)

    def test_a_missing_drive_is_exit_2(self):
        self.assertEqual(btd.main(["--dest-root", "Q:" + chr(92) + "nope", "--repo", self.repo]), 2)


if __name__ == "__main__":
    unittest.main()
