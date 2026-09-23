"""The nba_cache on disk: gzip writes, both-format reads, and the compactor.

Everything runs in throwaway temp directories. No network: endpoint classes
are fakes, and nothing here touches Data/nba_cache.
"""

import gzip
import importlib.util
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.Utils import nba_stats_client as nsc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, *rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compact = _load("compact_nba_cache", ["compact_nba_cache.py"])
ingest = _load("ingest_game_summaries", ["src", "Process-Data", "ingest_game_summaries.py"])

PAYLOAD = {"resultSets": [{"name": "X", "headers": ["A"], "rowSet": [[1], [2]]}], "pad": "x" * 5000}


def age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


class _TempCache(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="nba_cache_test_"))
        self._patch = mock.patch.object(nsc, "_CACHE_ROOT", self.dir)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def names(self):
        return sorted(p.name for p in self.dir.iterdir())

    def write_plain(self, name, data=PAYLOAD):
        p = self.dir / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def write_gz(self, name, data=PAYLOAD):
        p = self.dir / name
        p.write_bytes(gzip.compress(json.dumps(data).encode("utf-8")))
        return p


class ClientCacheTest(_TempCache):

    def test_writes_gzip_atomically_and_reads_it_back(self):
        path = nsc._cache_path("boxscoreadvancedv3", {"game_id": "0022400001"})
        self.assertEqual(path.name, "boxscoreadvancedv3_game_id=0022400001.json")
        nsc._write_cache(path, PAYLOAD)
        self.assertEqual(self.names(), ["boxscoreadvancedv3_game_id=0022400001.json.gz"])  # no temp left
        self.assertEqual(nsc._read_cache(path, None), PAYLOAD)
        with gzip.open(nsc.gz_path(path), "rt", encoding="utf-8") as f:
            self.assertEqual(json.load(f), PAYLOAD)

    def test_existing_plain_json_still_reads(self):
        p = self.write_plain("playbyplayv3_game_id=1.json")
        self.assertEqual(nsc._read_cache(p, None), PAYLOAD)

    def test_newest_copy_wins_and_a_tie_goes_to_gzip(self):
        old, new = {"v": "old"}, {"v": "new"}
        plain = self.write_plain("k.json", old)
        gz = self.write_gz("k.json.gz", new)
        t = time.time() - 100
        os.utime(plain, (t, t)); os.utime(gz, (t, t))
        self.assertEqual(nsc._read_cache(plain, None), new)      # tie -> gzip
        age(gz, 500)                                             # an old-format writer refreshed .json
        self.assertEqual(nsc._read_cache(plain, None), old)

    def test_truncated_gzip_is_a_miss_never_partial_data(self):
        gz = self.dir / "k.json.gz"
        full = gzip.compress(json.dumps(PAYLOAD).encode("utf-8"))
        gz.write_bytes(full[: len(full) // 2])
        self.assertIsNone(nsc._read_cache(self.dir / "k.json", None))
        gz.write_bytes(full[:-4])  # stream intact but the size trailer is gone
        self.assertIsNone(nsc._read_cache(self.dir / "k.json", None))
        gz.write_bytes(b"not gzip at all")
        self.assertIsNone(nsc._read_cache(self.dir / "k.json", None))

    def test_corrupt_gzip_falls_back_to_a_plain_copy(self):
        self.write_plain("k.json", {"v": "plain"})
        (self.dir / "k.json.gz").write_bytes(b"\x1f\x8b garbage")
        self.assertEqual(nsc._read_cache(self.dir / "k.json", None), {"v": "plain"})

    def test_ttl_applies_to_each_copy(self):
        gz = self.write_gz("k.json.gz", {"v": "gz"})
        age(gz, 7200)
        self.assertIsNone(nsc._read_cache(self.dir / "k.json", 3600))
        self.write_plain("k.json", {"v": "plain"})  # fresh
        self.assertEqual(nsc._read_cache(self.dir / "k.json", 3600), {"v": "plain"})
        self.assertIsNone(nsc._read_cache(self.dir / "k.json", 0))  # ttl=0 always refetches

    def test_write_supersedes_a_plain_copy(self):
        self.write_plain("k.json", {"v": "stale"})
        nsc._write_cache(self.dir / "k.json", {"v": "fresh"})
        self.assertEqual(self.names(), ["k.json.gz"])
        self.assertEqual(nsc._read_cache(self.dir / "k.json", None), {"v": "fresh"})

    def test_failed_write_leaves_the_old_entry_and_no_temp(self):
        self.write_gz("k.json.gz", {"v": "keep"})
        nsc._write_cache(self.dir / "k.json", {"bad": object()})  # not serialisable
        self.assertEqual(self.names(), ["k.json.gz"])
        self.assertEqual(nsc._read_cache(self.dir / "k.json", None), {"v": "keep"})

    def test_fetch_uses_the_cache_and_refetches_over_a_corrupt_file(self):
        calls = []

        class FakeEndpoint:
            def __init__(self, **kw):
                calls.append(kw)

            def get_dict(self):
                return {"game": {"actions": [{"n": len(calls)}]}}

        client = nsc.NBAStatsClient(rate_delay=0)
        params = {"game_id": "0022400001"}
        first = client._fetch("playbyplayv3", FakeEndpoint, params, ttl=None)
        again = client._fetch("playbyplayv3", FakeEndpoint, params, ttl=None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, again)

        gz = nsc.gz_path(nsc._cache_path("playbyplayv3", params))
        gz.write_bytes(gz.read_bytes()[:10])  # a partial file on disk
        third = client._fetch("playbyplayv3", FakeEndpoint, params, ttl=None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(third, {"game": {"actions": [{"n": 2}]}})
        self.assertEqual(nsc._read_cache(gz, None), third)


class IngestSummariesReaderTest(_TempCache):

    def test_lists_each_game_once_across_both_formats(self):
        self.write_plain("boxscoresummaryv2_game_id=0020300001.json", {"v": 1})
        self.write_gz("boxscoresummaryv2_game_id=0020300002.json.gz", {"v": 2})
        self.write_plain("boxscoresummaryv2_game_id=0020300003.json", {"v": "old"})
        self.write_gz("boxscoresummaryv2_game_id=0020300003.json.gz", {"v": 3})
        self.write_plain("boxscoresummaryv3_game_id=0020300001.json")  # other endpoint
        got = ingest.cached_summaries(str(self.dir))
        self.assertEqual([g for g, _ in got], ["0020300001", "0020300002", "0020300003"])
        self.assertEqual([ingest.load_summary(p) for _, p in got], [{"v": 1}, {"v": 2}, {"v": 3}])

    def test_a_corrupt_gz_falls_back_and_nothing_readable_raises(self):
        self.write_plain("boxscoresummaryv2_game_id=1.json", {"v": "plain"})
        (self.dir / "boxscoresummaryv2_game_id=1.json.gz").write_bytes(b"junk")
        self.assertEqual(ingest.load_summary(str(self.dir / "boxscoresummaryv2_game_id=1.json")), {"v": "plain"})
        (self.dir / "boxscoresummaryv2_game_id=2.json.gz").write_bytes(b"junk")
        with self.assertRaises(Exception):
            ingest.load_summary(str(self.dir / "boxscoresummaryv2_game_id=2.json"))


class CompactorTest(_TempCache):

    def make_old(self, name, data=PAYLOAD, seconds=3600):
        p = self.write_plain(name, data)
        age(p, seconds)
        return p

    def run_apply(self, **kw):
        args = dict(prefix=None, limit=0, min_age_s=600, level=6, log_path=str(self.dir.parent / (self.dir.name + ".log")))
        args.update(kw)
        try:
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                return compact.apply(str(self.dir), **args)
        finally:
            try:
                os.unlink(args["log_path"])
            except (OSError, TypeError):
                pass

    def snapshot(self):
        return {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in self.dir.iterdir()}

    def test_dry_run_changes_nothing(self):
        self.make_old("playbyplayv3_game_id=1.json")
        self.make_old("boxscoreadvancedv3_game_id=1.json")
        self.write_plain("leaguegamelog_season=2025-26.json")  # recent
        before = self.snapshot()
        out = io.StringIO()
        tot = compact.dry_run(str(self.dir), None, 40, 600, 6, 1, out=out)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(tot["files"], 3)
        self.assertEqual(tot["recent"], 1)
        self.assertLess(tot["proj"], tot["bytes"])
        self.assertIn("Nothing was written", out.getvalue())

    def test_apply_is_lossless_keeps_mtime_and_the_client_reads_it(self):
        p = self.make_old("playbyplayv3_game_id=1.json")
        original, mtime = p.read_bytes(), p.stat().st_mtime_ns
        stats = self.run_apply()
        self.assertEqual(stats["converted"], 1)
        self.assertEqual(self.names(), ["playbyplayv3_game_id=1.json.gz"])
        gz = self.dir / "playbyplayv3_game_id=1.json.gz"
        self.assertEqual(gzip.decompress(gz.read_bytes()), original)
        self.assertEqual(gz.stat().st_mtime_ns, mtime)  # TTL clock unchanged
        self.assertEqual(nsc._read_cache(p, None), PAYLOAD)
        self.assertLess(stats["bytes_after"], stats["bytes_before"])

    def test_recently_touched_files_are_skipped(self):
        self.write_plain("scoreboardv3_game_date=2026-10-21.json")
        stats = self.run_apply()
        self.assertEqual(stats["skipped_recent"], 1)
        self.assertEqual(self.names(), ["scoreboardv3_game_date=2026-10-21.json"])

    def test_resume_finishes_an_identical_twin_and_leaves_a_different_one(self):
        same = self.make_old("a.json")
        gz_same = self.dir / "a.json.gz"
        gz_same.write_bytes(gzip.compress(same.read_bytes()))
        self.make_old("b.json", {"v": "plain"})
        self.write_gz("b.json.gz", {"v": "different"})
        stats = self.run_apply()
        self.assertEqual(stats["finished_twin"], 1)
        self.assertEqual(stats["twin_differs"], 1)
        self.assertEqual(self.names(), ["a.json.gz", "b.json", "b.json.gz"])

    def test_stale_temp_files_are_swept_and_fresh_ones_kept(self):
        self.make_old("a.json")
        stale = self.dir / ("a.json.gz.999" + compact.TEMP_SUFFIX)
        stale.write_bytes(b"half")
        age(stale, 3600)
        fresh = self.dir / ("z.json.gz.998" + compact.TEMP_SUFFIX)
        fresh.write_bytes(b"in flight")
        stats = self.run_apply()
        self.assertEqual(stats["stale_temps_removed"], 1)
        self.assertEqual(self.names(), ["a.json.gz", fresh.name])

    def test_a_gz_that_appears_mid_conversion_is_not_overwritten(self):
        p = self.make_old("a.json")
        real = compact.rename_no_clobber

        def racing(src, dst):
            Path(dst).write_bytes(gzip.compress(b'{"v": "newer writer"}'))
            return real(src, dst)

        with mock.patch.object(compact, "rename_no_clobber", side_effect=racing):
            stats = self.run_apply()
        self.assertEqual(stats["race_gz_appeared"], 1)
        self.assertEqual(self.names(), ["a.json", "a.json.gz"])
        self.assertEqual(p.read_bytes(), json.dumps(PAYLOAD).encode())
        self.assertEqual(gzip.decompress((self.dir / "a.json.gz").read_bytes()), b'{"v": "newer writer"}')

    def test_a_json_rewritten_mid_conversion_is_kept_and_nothing_is_placed(self):
        p = self.make_old("a.json")
        real = gzip.compress

        def writer_strikes(data, compresslevel=9):
            p.write_text('{"v": "rewritten"}', encoding="utf-8")
            return real(data, compresslevel=compresslevel)

        with mock.patch.object(compact.gzip, "compress", side_effect=writer_strikes):
            stats = self.run_apply(min_age_s=0)
        self.assertEqual(stats["changed_during"], 1)
        self.assertEqual(self.names(), ["a.json"])
        self.assertEqual(p.read_text(encoding="utf-8"), '{"v": "rewritten"}')

    def test_failed_verification_touches_nothing(self):
        p = self.make_old("a.json")
        before = self.snapshot()
        with mock.patch.object(compact.gzip, "decompress", return_value=b"wrong"):
            stats = self.run_apply()
        self.assertEqual(stats["verify_failed"], 1)
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(p.exists())

    def test_a_locked_json_is_left_for_the_next_run(self):
        self.make_old("a.json")
        real_unlink = os.unlink

        def locked(path, *a, **kw):
            if str(path).endswith(".json"):
                raise PermissionError("in use")
            return real_unlink(path, *a, **kw)

        with mock.patch.object(compact.os, "unlink", side_effect=locked):
            stats = self.run_apply()
        self.assertEqual(stats["json_locked"], 1)
        self.assertEqual(self.names(), ["a.json", "a.json.gz"])
        stats = self.run_apply()
        self.assertEqual(stats["finished_twin"], 1)
        self.assertEqual(self.names(), ["a.json.gz"])

    def test_limit_and_prefix(self):
        for i in range(3):
            self.make_old(f"playbyplayv3_game_id={i}.json")
        self.make_old("boxscoreadvancedv3_game_id=0.json")
        stats = self.run_apply(prefix="playbyplayv3", limit=2)
        self.assertEqual(stats["converted"], 2)
        left = [n for n in self.names() if n.endswith(".json")]
        self.assertEqual(left, ["boxscoreadvancedv3_game_id=0.json", "playbyplayv3_game_id=2.json"])

    def test_cli_defaults_to_dry_run(self):
        self.make_old("a.json")
        before = self.snapshot()
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            rc = compact.main(["--cache-dir", str(self.dir)])
        self.assertEqual(rc, 0)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
