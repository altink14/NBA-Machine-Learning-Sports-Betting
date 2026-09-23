"""
push_ledger.py
==============
Send this machine's prediction ledger to the public server (DEPLOY.md 3a,
option A), then prove the server now holds exactly what was sent.

The home PC is the ledger's only writer. This script is how the record
reaches the page people read. It runs after every scheduled job that writes
to OddsData.sqlite, and by hand whenever you like; it is idempotent.

    venv/Scripts/python.exe push_ledger.py            # push and verify
    venv/Scripts/python.exe push_ledger.py --dry-run  # build + fingerprint only

Environment (.env):
  LEDGER_SYNC_URL     the public backend's base URL, e.g. https://api.example.com
  LEDGER_SYNC_SECRET  must equal the server's LEDGER_SYNC_SECRET

With either unset it says SKIPPED and exits 0: until the site is deployed
there is nowhere to push, and that is not a failure.

Exit codes: 0 pushed and verified (or skipped), 1 anything else. A refusal
from the server (409) means the two copies disagree about the past. That is
never retried or forced: read the message, find out why, fix the cause.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

from src.Utils import ledger_sync  # noqa: E402

ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")


def build_copy(dest: str) -> None:
    """A consistent point-in-time copy, safe while other jobs are writing."""
    src = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
    try:
        src.execute("VACUUM INTO ?", (dest,))
    finally:
        src.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the copy and print its fingerprint; send nothing.")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    load_dotenv(os.path.join(REPO, ".env"))
    url = (os.environ.get("LEDGER_SYNC_URL") or "").strip().rstrip("/")
    secret = (os.environ.get("LEDGER_SYNC_SECRET") or "").strip()

    if not os.path.exists(ODDS_DB):
        print(f"FAILED: {ODDS_DB} does not exist")
        return 1
    if not args.dry_run and (not url or not secret):
        print("SKIPPED: LEDGER_SYNC_URL / LEDGER_SYNC_SECRET not set (no public server yet)")
        return 0

    tmpdir = tempfile.mkdtemp(prefix="ledger_push_")
    copy = os.path.join(tmpdir, "OddsData.sqlite")
    try:
        build_copy(copy)
        ledger_sync.check_upload(copy)
        conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            local = ledger_sync.fingerprint(conn)
        finally:
            conn.close()
        summary = ", ".join(f"{t} {v['rows']}" for t, v in local.items())

        if args.dry_run:
            print(f"DRY RUN: would push {summary}")
            print(json.dumps(local, indent=2))
            return 0

        with open(copy, "rb") as fh:
            payload = gzip.compress(fh.read(), compresslevel=6)

        import requests
        endpoint = f"{url}/api/admin/ledger/sync"
        headers = {"X-Ledger-Sync-Secret": secret,
                   "Content-Type": "application/octet-stream"}
        resp = None
        for attempt in (1, 2, 3):
            try:
                resp = requests.post(endpoint, data=payload, headers=headers, timeout=args.timeout)
            except requests.RequestException as exc:
                # Network trouble is worth a retry; a refusal (below) never is.
                print(f"attempt {attempt}: could not reach {url}: {exc}")
                if attempt < 3:
                    time.sleep(10 * attempt)
                continue
            if resp.status_code >= 500 and attempt < 3:
                print(f"attempt {attempt}: server answered {resp.status_code}; retrying")
                time.sleep(10 * attempt)
                continue
            break
        if resp is None:
            print("FAILED: the server could not be reached; the public record is behind, not wrong")
            return 1
        if resp.status_code == 409:
            print(f"REFUSED by the server: {resp.json().get('detail')}")
            print("The public copy and this one disagree about the past. Nothing was written.")
            return 1
        if resp.status_code != 200:
            print(f"FAILED: HTTP {resp.status_code}: {resp.text[:300]}")
            return 1

        body = resp.json()
        remote = body.get("fingerprint") or {}
        mismatched = [t for t in local if remote.get(t) != local[t]]
        if mismatched:
            print(f"MISMATCH after push in {mismatched}: the server accepted the upload but does "
                  "not hold what was sent.")
            for t in mismatched:
                print(f"  {t}: sent {local[t]}  server {remote.get(t)}")
            return 1
        changes = ", ".join(f"{t} +{v['inserted']} new/{v['updated']} updated"
                            for t, v in (body.get("tables") or {}).items())
        print(f"PUBLISHED and verified: {summary} ({changes}); fingerprints match")
        return 0
    finally:
        for f in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, f))
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
