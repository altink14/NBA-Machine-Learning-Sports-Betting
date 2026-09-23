"""
compact_nba_cache.py
====================
Convert the plain `.json` files in Data/nba_cache to `.json.gz`, one file at a
time, losslessly. The cache is the raw source the archive is rebuilt from, so
this never deletes information: a `.json` is removed only after its `.gz`
twin is on disk AND decompresses to exactly the original bytes.

    python compact_nba_cache.py                    # dry run (the default): projected savings
    python compact_nba_cache.py --apply            # convert everything eligible
    python compact_nba_cache.py --apply --prefix playbyplayv3 --limit 1000

BEFORE --apply: every process that reads the cache must be running a
nba_stats_client.py that reads `.json.gz` (this branch or later). A process
still on the old client (a uvicorn started earlier, a backfill in flight, the
scheduled daily update from an old checkout) sees a converted entry as a cache
miss and re-downloads it from stats.nba.com - the exact thing the cache exists
to prevent. Restart them first.

Per file (--apply):
  1. skip it if it was modified in the last --min-age-minutes (an active
     writer may still be refreshing it);
  2. gzip the bytes to a temp file in the same folder, fsync, read the temp
     file back and check it decompresses to the original bytes;
  3. copy the original's modification time onto it, so TTL'd entries expire
     exactly when they would have (a converted 24 h entry does not become
     "fresh" again);
  4. check the original has not changed, move the temp into place WITHOUT
     overwriting (if a `.gz` appeared meanwhile, a newer writer won - keep its
     copy and ours is discarded), check the original once more, and only then
     remove the `.json`.

Resumable: every file is independent. A run that dies part-way leaves each
file converted or not; a `.json` whose `.gz` twin already holds identical
bytes (a crash between steps 4's rename and delete) is finished off on the
next run, and our own stale temp files are swept. A `.json` whose `.gz` twin
holds DIFFERENT bytes is left alone and reported: the reader takes the newer
of the two, and a person should look at it.

Progress is appended to --log (default: compact_nba_cache.log next to the
cache folder) and summarised at the end. No network, ever.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import os
import random
import sys
import time
from datetime import datetime

DEFAULT_CACHE = os.environ.get("NBA_CACHE_DIR", os.path.join("Data", "nba_cache"))
TEMP_SUFFIX = ".compact.tmp"


def prefix_of(name: str) -> str:
    return name.split("_", 1)[0]


def mb(n: float) -> str:
    return f"{n / 1e6:,.1f}"


def scan(cache_dir: str, prefix: str | None = None):
    """(plain .json entries, set of .json.gz names, our temp files), via os.scandir."""
    plain, gz, temps = [], set(), []
    with os.scandir(cache_dir) as it:
        for e in it:
            if not e.is_file(follow_symlinks=False):
                continue
            n = e.name
            if n.endswith(TEMP_SUFFIX):
                temps.append(e)
                continue
            if prefix and prefix_of(n) != prefix:
                continue
            if n.endswith(".json"):
                plain.append(e)
            elif n.endswith(".json.gz"):
                gz.add(n)
    plain.sort(key=lambda e: e.name)
    return plain, gz, temps


def rename_no_clobber(src: str, dst: str) -> None:
    """Move src to dst, raising FileExistsError if dst exists. Atomic on NTFS
    (os.rename refuses an existing target on Windows) and POSIX (link+unlink)."""
    if os.name == "nt":
        os.rename(src, dst)
    else:
        os.link(src, dst)
        os.unlink(src)


def same_file_state(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_size == b.st_size and a.st_mtime_ns == b.st_mtime_ns


# --------------------------------------------------------------------- dry run

def dry_run(cache_dir: str, prefix: str | None, sample: int, min_age_s: float,
            level: int, seed: int, out=None) -> dict:
    out = out or sys.stdout
    t0 = time.time()
    plain, gz, temps = scan(cache_dir, prefix)
    now = time.time()
    by = collections.defaultdict(list)
    for e in plain:
        by[prefix_of(e.name)].append(e)
    rng = random.Random(seed)

    print(f"compact_nba_cache DRY RUN - {cache_dir} - {datetime.now():%Y-%m-%d %H:%M}", file=out)
    print(f"{len(plain):,} plain .json files, {len(gz):,} .json.gz already, "
          f"{len(temps)} leftover temp file(s); scanned in {time.time() - t0:.1f}s", file=out)
    print(f"sample: up to {sample} files per endpoint, gzip level {level}\n", file=out)
    print("| endpoint | .json files | MB now | sample ratio (n) | projected MB | saves MB | recent (skipped) | has .gz twin |", file=out)
    print("|---|---:|---:|---:|---:|---:|---:|---:|", file=out)

    tot = collections.Counter()
    rows = sorted(by.items(), key=lambda kv: -sum(e.stat().st_size for e in kv[1]))
    for pfx, entries in rows:
        size = sum(e.stat().st_size for e in entries)
        recent = sum(1 for e in entries if now - e.stat().st_mtime < min_age_s)
        twins = sum(1 for e in entries if e.name + ".gz" in gz)
        raw = comp = 0
        picked = rng.sample(entries, min(sample, len(entries)))
        for e in picked:
            try:
                with open(e.path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            raw += len(data)
            comp += len(gzip.compress(data, compresslevel=level))
        ratio = comp / raw if raw else 1.0
        proj = size * ratio
        print(f"| {pfx} | {len(entries):,} | {mb(size)} | {ratio:.3f} ({len(picked)}) | {mb(proj)} | "
              f"{mb(size - proj)} | {recent:,} | {twins:,} |", file=out)
        tot["files"] += len(entries)
        tot["bytes"] += size
        tot["proj"] += proj
        tot["recent"] += recent
        tot["twins"] += twins

    saved = tot["bytes"] - tot["proj"]
    pct = 100 * saved / tot["bytes"] if tot["bytes"] else 0.0
    print(f"\nTOTAL {tot['files']:,} files, {mb(tot['bytes'])} MB -> about {mb(tot['proj'])} MB "
          f"(saves about {mb(saved)} MB, {pct:.1f}%)", file=out)
    print(f"{tot['recent']:,} file(s) modified in the last {min_age_s / 60:.0f} min would be skipped this run.", file=out)
    print("Projection is sample-based. Nothing was written. Re-run with --apply to convert.", file=out)
    return dict(tot)


# ----------------------------------------------------------------------- apply

class Log:
    def __init__(self, path: str | None, verbose: bool):
        self.f = open(path, "a", encoding="utf-8") if path else None
        self.verbose = verbose

    def __call__(self, msg: str, echo: bool = False) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()
        if echo or self.verbose:
            print(line, flush=True)

    def close(self):
        if self.f:
            self.f.close()


def convert_one(path: str, level: int, min_age_s: float, stats: collections.Counter, log: Log) -> str:
    """Convert one .json and return the outcome name. Only the byte totals
    are added to `stats` here; the caller counts the outcome."""
    gz = path + ".gz"
    tmp = f"{gz}.{os.getpid()}{TEMP_SUFFIX}"
    try:
        st0 = os.stat(path)
    except FileNotFoundError:
        return "vanished"
    if time.time() - st0.st_mtime < min_age_s:
        return "skipped_recent"

    with open(path, "rb") as f:
        original = f.read()
    if len(original) != st0.st_size:
        return "changed_during"

    # A twin already exists: an interrupted earlier run, or a newer writer.
    if os.path.exists(gz):
        try:
            with open(gz, "rb") as f:
                same = gzip.decompress(f.read()) == original
        except Exception:
            same = False
        if not same:
            log(f"TWIN-DIFFERS {os.path.basename(path)} (left both; readers take the newer)")
            return "twin_differs"
        st1 = os.stat(path)
        if not same_file_state(st0, st1):
            return "changed_during"
        os.unlink(path)
        stats["bytes_before"] += st0.st_size
        stats["bytes_after"] += os.path.getsize(gz)
        log(f"FINISHED {os.path.basename(path)} (identical .gz already present)")
        return "finished_twin"

    placed = False
    try:
        payload = gzip.compress(original, compresslevel=level)
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, "rb") as f:
            if gzip.decompress(f.read()) != original:
                log(f"VERIFY-FAILED {os.path.basename(path)}", echo=True)
                return "verify_failed"
        os.utime(tmp, ns=(st0.st_atime_ns, st0.st_mtime_ns))

        if not same_file_state(st0, os.stat(path)):
            return "changed_during"
        try:
            rename_no_clobber(tmp, gz)
        except FileExistsError:
            log(f"RACE {os.path.basename(path)} (.gz appeared mid-conversion; kept it)")
            return "race_gz_appeared"
        placed = True

        # The original changed after we read it: our .gz is stale. Ours to remove.
        try:
            st2 = os.stat(path)
        except FileNotFoundError:
            st2 = None
        if st2 is None or not same_file_state(st0, st2):
            os.unlink(gz)
            placed = False
            return "changed_during"

        try:
            os.unlink(path)
        except PermissionError:
            # Held open by another process. Both copies are identical, so
            # readers are fine; the next run deletes it via the twin path.
            log(f"LOCKED {os.path.basename(path)} (.gz placed; .json left for next run)")
            return "json_locked"
        stats["bytes_before"] += st0.st_size
        stats["bytes_after"] += len(payload)
        log(f"OK {os.path.basename(path)} {st0.st_size} -> {len(payload)}")
        return "converted"
    finally:
        if not placed and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def apply(cache_dir: str, prefix: str | None, limit: int, min_age_s: float, level: int,
          log_path: str | None, verbose: bool = False, progress_every: float = 30.0) -> collections.Counter:
    log = Log(log_path, verbose)
    stats: collections.Counter = collections.Counter()
    t0 = time.time()
    try:
        plain, _, temps = scan(cache_dir, prefix)
        log(f"START apply cache={cache_dir} prefix={prefix or '*'} limit={limit or 'none'} "
            f"min_age={min_age_s / 60:.0f}min level={level}: {len(plain):,} .json candidates", echo=True)
        for e in temps:  # our own leftovers from a run that died
            try:
                if time.time() - e.stat().st_mtime >= min_age_s:
                    os.unlink(e.path)
                    stats["stale_temps_removed"] += 1
            except OSError:
                pass

        last_report = time.time()
        attempted = 0
        for e in plain:
            if limit and attempted >= limit:
                break
            try:
                outcome = convert_one(e.path, level, min_age_s, stats, log)
            except Exception as exc:  # one bad file must not end the run
                outcome = "error"
                log(f"ERROR {e.name}: {exc}", echo=True)
            stats[outcome] += 1
            if outcome != "skipped_recent":
                attempted += 1
            if time.time() - last_report >= progress_every:
                last_report = time.time()
                log(f"PROGRESS {attempted:,}/{len(plain):,} converted={stats['converted']:,} "
                    f"saved={mb(stats['bytes_before'] - stats['bytes_after'])} MB", echo=True)
    except KeyboardInterrupt:
        log("INTERRUPTED - safe to re-run; every file is either converted or untouched", echo=True)
        stats["interrupted"] = 1
    finally:
        saved = stats["bytes_before"] - stats["bytes_after"]
        outcomes = {k: v for k, v in sorted(stats.items()) if not k.startswith("bytes_")}
        log(f"SUMMARY {time.time() - t0:.0f}s {outcomes} "
            f"bytes {mb(stats['bytes_before'])} MB -> {mb(stats['bytes_after'])} MB (saved {mb(saved)} MB)",
            echo=True)
        log.close()
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report projected savings (the default)")
    mode.add_argument("--apply", action="store_true", help="convert .json -> .json.gz")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--prefix", help="only this endpoint (filename prefix before the first _)")
    p.add_argument("--limit", type=int, default=0, help="stop after N files (0 = all)")
    p.add_argument("--min-age-minutes", type=float, default=10.0,
                   help="skip files modified more recently than this (default 10)")
    p.add_argument("--level", type=int, default=6, help="gzip level (default 6, as the client writes)")
    p.add_argument("--sample", type=int, default=40, help="dry run: files sampled per endpoint")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--log", help="progress log (default: compact_nba_cache.log beside the cache dir)")
    p.add_argument("--verbose", action="store_true", help="echo every file's outcome")
    a = p.parse_args(argv)

    if not os.path.isdir(a.cache_dir):
        print(f"no such cache dir: {a.cache_dir}", file=sys.stderr)
        return 2
    min_age_s = a.min_age_minutes * 60
    if not a.apply:
        dry_run(a.cache_dir, a.prefix, a.sample, min_age_s, a.level, a.seed)
        return 0
    log_path = a.log or os.path.join(os.path.dirname(os.path.abspath(a.cache_dir)), "compact_nba_cache.log")
    stats = apply(a.cache_dir, a.prefix, a.limit, min_age_s, a.level, log_path, a.verbose)
    return 1 if stats["error"] or stats["verify_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
