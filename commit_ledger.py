"""
commit_ledger.py
================
Write a SHA-256 commitment to every logged pick no commitment covers yet,
chained to the one before (src/Utils/ledger_commit.py explains the method and
the exact canonical form).

    venv/Scripts/python.exe commit_ledger.py            # commit pending picks
    venv/Scripts/python.exe commit_ledger.py --verify   # recheck the whole chain

Runs in the 9am daily_update.py right after the picks are logged, and in the
hourly job, so a pick logged later in the day is committed within the hour and
still before its game. Idempotent: with nothing new it writes nothing and says
so. Home PC only (the ledger's single writer, DEPLOY.md 3a): it refuses to run
where PREDICTIONS_SOURCE=ledger, which is how the public server is set up.

Exit codes: 0 committed, nothing to commit, or verified; 1 anything else.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

from src.Utils import ledger_commit  # noqa: E402

ODDS_DB = os.path.join(REPO, "Data", "OddsData.sqlite")


def run(db_path: str = ODDS_DB, verify_only: bool = False) -> int:
    if (os.environ.get("PREDICTIONS_SOURCE") or "").strip().lower() == "ledger":
        print("REFUSED: this machine serves the mirrored ledger (PREDICTIONS_SOURCE=ledger). "
              "Only the home PC writes commitments; they arrive here through push_ledger.py.")
        return 1
    if not os.path.exists(db_path):
        print(f"FAILED: {db_path} does not exist")
        return 1
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        has_log = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions_log'").fetchone()
        if verify_only:
            check = ledger_commit.verify(conn)
            bad = [e for e in check["entries"] if e["chain"] != "ok" or e["rows"] != "match"]
            for e in bad:
                print(f"BROKEN {e['log_date']} #{e['seq']}: chain {e['chain']}, rows {e['rows']}")
            if bad:
                print(f"VERIFY FAILED: {len(bad)} of {check['length']} commitment(s) do not verify")
                return 1
            print(f"VERIFIED: {check['length']} commitment(s), chain head {check['head'] or '(empty)'}")
            return 0
        if not has_log:
            print("NOTHING TO COMMIT: predictions_log does not exist yet")
            return 0
        written = ledger_commit.commit_pending(conn)
        if not written:
            print("NOTHING TO COMMIT: every logged pick is already committed")
            return 0
        for w in written:
            timing = ("before its first tip-off" if w["before_first_tipoff"]
                      else "AFTER its first tip-off (proves no later edit, not pre-game timing)")
            print(f"COMMITTED {w['log_date']} #{w['seq']}: {w['n_picks']} pick(s), "
                  f"picks {w['picks_sha256'][:16]}..., chain {w['chain_sha256'][:16]}..., {timing}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="Recheck every commitment against the chain and the current rows; write nothing.")
    args = ap.parse_args()
    load_dotenv(os.path.join(REPO, ".env"))
    return run(verify_only=args.verify)


if __name__ == "__main__":
    sys.exit(main())
