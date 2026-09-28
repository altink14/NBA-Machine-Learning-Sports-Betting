"""The test suite as CI runs it: everything except the modules that need the
local archive (2026-09-28).

The full suite is the local gate:
    venv/Scripts/python.exe -m unittest discover -s Tests -p "*_Test.py"
It reads Data/TeamData.sqlite and friends, which live only on the home PC
(several GB, and not ours to publish). A CI checkout has none of it, and
these modules assert against the real archive without a skip guard, so on
a clean checkout they fail for want of data, not for a bug. Measured on a
`git archive HEAD` copy: 37 failures/errors, all in the modules below.

CI runs the rest. A module belongs here only if it cannot run without the
archive; the better fix is a `skipUnless(os.path.exists(REAL_DB))` guard in
the test itself (as Season_History_Test and Api_Contracts_Test have), after
which it comes off this list.

    python Tests/run_ci.py            exit 0 when the suite passes
"""
import os
import sys
import unittest

NEEDS_LOCAL_ARCHIVE = {
    "Availability_Test",
    "Backfill_Exit_Test",
    "Game_Flow_Test",
    "PlayIn_Handling_Test",
    "Plausible_Value_Fixes_Test",
    "Player_Impact_Test",
    "Retrain_Features_Test",
    "Schedule_Cup_Rest_Test",
}


def _flatten(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _flatten(t)
        else:
            yield t


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    sys.path.insert(0, root)
    # As `python -m unittest discover -s Tests` does: Tests/ is the top level
    # (it has no __init__.py), the repo root is on the path for main_api.
    loaded = unittest.defaultTestLoader.discover(os.path.join(root, "Tests"), pattern="*_Test.py")
    keep = unittest.TestSuite()
    skipped_modules = set()
    for test in _flatten(loaded):
        module = type(test).__module__.split(".")[-1]
        if module in NEEDS_LOCAL_ARCHIVE:
            skipped_modules.add(module)
            continue
        keep.addTest(test)
    print(f"Leaving out {len(skipped_modules)} archive-dependent module(s): {', '.join(sorted(skipped_modules))}")
    result = unittest.TextTestRunner(verbosity=1).run(keep)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
