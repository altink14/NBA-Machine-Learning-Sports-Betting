"""
publish_commitments.py
======================
Publish the pick commitments' hashes somewhere we do not control, which is
what turns "the record has not been edited" into something nobody has to take
our word for (src/Utils/ledger_commit.py, step 4 of its docstring).

    venv/Scripts/python.exe publish_commitments.py             # print only
    venv/Scripts/python.exe publish_commitments.py --publish   # really publish

WHAT IT PUBLISHES
One JSON line per commitment, appended to COMMITMENTS_FILE in a git
repository, then committed and pushed. Each line carries the commitment's
hashes, size and timing, never its nonce or its picks, so publishing before
tip-off reveals nothing about the picks:

  {"chain_sha256": ..., "committed_at": ..., "first_tipoff_utc": ...,
   "log_date": ..., "method_version": ..., "n_picks": ..., "pick_ids": [...],
   "picks_sha256": ..., "prev_chain_sha256": ..., "seq": ...}

(keys sorted, compact separators, one line each, in chain order). The file
only ever grows: before appending, every line already in it must equal the
commitment the database holds at that position. If it does not, nothing is
written and the run fails, because either the published record or ours has
been changed and a person must find out which.

WHAT PROVES THE TIME
A git commit's own date is whatever this machine says. The independent
evidence is the host's record of when the push arrived (GitHub shows it in
the repository's activity), and the fact that anyone who pulled the file
keeps a copy we cannot change.

IT DOES NOTHING UNLESS BOTH ARE TRUE
  1. it is run with --publish, and
  2. LEDGER_PUBLISH_REPO (environment or .env) names a local clone of the
     public repository, with push access already set up.
Without --publish it prints what it would publish. With --publish and no
target it prints SKIPPED and exits 0, as push_ledger.py does before deploy.
The daily and hourly jobs run it with --publish, so the owner's switch-on is
setting LEDGER_PUBLISH_REPO (DEPLOY.md, "Publishing the pick commitments").

Exit codes: 0 printed, published, nothing new, or skipped; 1 anything else.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from typing import Callable, List, Optional

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

from src.Utils import ledger_commit  # noqa: E402

ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")
COMMITMENTS_FILE = "nba/commitments.jsonl"
PUBLISHED_FIELDS = ("method_version", "log_date", "seq", "n_picks", "pick_ids", "committed_at",
                    "first_tipoff_utc", "picks_sha256", "prev_chain_sha256", "chain_sha256")


class PublishRefused(Exception):
    """The published file and the database disagree, or the chain is broken."""


def public_lines(conn: sqlite3.Connection) -> List[str]:
    """Every commitment as the line it is published as, in chain order."""
    if not ledger_commit.has_table(conn):
        return []
    check = ledger_commit.verify(conn)
    bad = [e for e in check["entries"] if e["chain"] != "ok" or e["rows"] != "match"]
    if bad:
        raise PublishRefused(
            f"{len(bad)} commitment(s) do not verify (first: {bad[0]['log_date']} #{bad[0]['seq']}); "
            "run commit_ledger.py --verify. A broken chain is never published.")
    old = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM ledger_commitments ORDER BY id").fetchall()
    finally:
        conn.row_factory = old
    out = []
    for r in rows:
        obj = {f: (json.loads(r[f]) if f == "pick_ids" else r[f]) for f in PUBLISHED_FIELDS}
        out.append(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return out


def new_lines(all_lines: List[str], published: List[str]) -> List[str]:
    """What to append: refuse unless the published file is a prefix of ours."""
    if len(published) > len(all_lines):
        raise PublishRefused(
            f"the published file has {len(published)} line(s) but the database only "
            f"{len(all_lines)} commitment(s). Something was removed here; nothing written.")
    for i, (theirs, ours) in enumerate(zip(published, all_lines), start=1):
        if theirs != ours:
            raise PublishRefused(
                f"published line {i} differs from the database's commitment {i}. Either the "
                "public copy or ours was changed; nothing written.")
    return all_lines[len(published):]


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          timeout=120, encoding="utf-8", errors="replace")


def publish(conn: sqlite3.Connection, target: Optional[str], do_publish: bool,
            git: Callable[..., subprocess.CompletedProcess] = _git) -> int:
    lines = public_lines(conn)

    if not do_publish:
        print(f"DRY RUN: {len(lines)} commitment line(s) would be published to "
              f"{COMMITMENTS_FILE} in {target or '(LEDGER_PUBLISH_REPO not set)'}:")
        for ln in lines:
            print(ln)
        return 0
    if not target:
        print("SKIPPED: LEDGER_PUBLISH_REPO not set (no public copy configured yet)")
        return 0
    if not os.path.isdir(os.path.join(target, ".git")):
        print(f"FAILED: LEDGER_PUBLISH_REPO={target} is not a git clone")
        return 1

    pulled = git(target, "pull", "--ff-only")
    if pulled.returncode != 0:
        print(f"FAILED: git pull --ff-only in {target}: {(pulled.stderr or pulled.stdout).strip()[-300:]}")
        return 1
    # What is already public is what the pushed history holds, not whatever
    # is in the working file: a run that died between writing and committing
    # must not make its lines look published.
    shown = git(target, "show", f"HEAD:{COMMITMENTS_FILE}")
    published = ([ln for ln in (shown.stdout or "").splitlines() if ln.strip()]
                 if shown.returncode == 0 else [])
    fresh = new_lines(lines, published)
    if not fresh:
        # Still push: an earlier run may have committed and then failed to push.
        r = git(target, "push")
        if r.returncode != 0:
            print(f"FAILED at git push: {(r.stderr or r.stdout).strip()[-300:]}")
            return 1
        print(f"NOTHING NEW: all {len(lines)} commitment(s) are already published")
        return 0

    path = os.path.join(target, *COMMITMENTS_FILE.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for ln in published + fresh:
            fh.write(ln + "\n")
    last = json.loads(fresh[-1])
    msg = (f"NBA pick commitments through {last['log_date']} #{last['seq']} "
           f"(chain head {last['chain_sha256'][:16]})")
    for step in (("add", COMMITMENTS_FILE), ("commit", "-m", msg), ("push",)):
        r = git(target, *step)
        if r.returncode != 0:
            print(f"FAILED at git {step[0]}: {(r.stderr or r.stdout).strip()[-300:]}")
            return 1
    print(f"PUBLISHED {len(fresh)} new commitment(s); chain head {last['chain_sha256']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--publish", action="store_true",
                    help="Actually append, commit and push (needs LEDGER_PUBLISH_REPO).")
    args = ap.parse_args()
    load_dotenv(os.path.join(REPO, ".env"))
    target = (os.environ.get("LEDGER_PUBLISH_REPO") or "").strip() or None
    if not os.path.exists(ODDS_DB):
        print(f"FAILED: {ODDS_DB} does not exist")
        return 1
    conn = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
    try:
        return publish(conn, target, args.publish)
    except PublishRefused as exc:
        print(f"REFUSED: {exc}")
        return 1
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
