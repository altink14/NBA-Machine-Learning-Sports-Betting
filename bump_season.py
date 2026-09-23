"""
bump_season.py
==============
Move the three "current season" constants to the new season, in one step.

    venv/Scripts/python.exe bump_season.py                 # show what would change
    venv/Scripts/python.exe bump_season.py --apply         # change it
    venv/Scripts/python.exe bump_season.py --to 2026-27    # a specific season

Run it the morning of opening night (preflight_opening_night.py flags the
backend constant as WRONG from that day on). It edits:

  backend   main_api.py                      CURRENT_SEASON = "2025-26"
  frontend  src/lib/nba-api.ts               export const CURRENT_SEASON = '2025-26';
  frontend  src/lib/archive-seasons.ts       const CURRENT_END_YEAR = 2026; // 2025-26

Three hand edits in two repos on the busiest morning of the year is how one
gets missed; a missed one serves last season silently. It only moves forward
by exactly one season unless --to says otherwise, and refuses if any file is
not in the shape it expects rather than guess. It does not commit, restart
anything or touch OPENING_NIGHT (that is next fall's date, set once the league
announces it); it prints those steps instead.

The other season-dependent scripts (ingest_players.py,
ingest_player_bios_bulk.py, daily_update.py) derive the season from the date.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date

BACKEND = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(os.path.dirname(BACKEND), "basic-saas-starter")

SEASON = re.compile(r"^(\d{4})-(\d{2})$")


def season_for(today: date) -> str:
    start = today.year if today.month >= 10 else today.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def next_season(season: str) -> str:
    start = int(season[:4]) + 1
    return f"{start}-{(start + 1) % 100:02d}"


def targets(backend: str, frontend: str):
    """(path, regex with one group = the current value, formatter)"""
    return [
        (os.path.join(backend, "main_api.py"),
         re.compile(r'^CURRENT_SEASON = "(\d{4}-\d{2})"$', re.M),
         lambda s: f'CURRENT_SEASON = "{s}"'),
        (os.path.join(frontend, "src", "lib", "nba-api.ts"),
         re.compile(r"^export const CURRENT_SEASON = '(\d{4}-\d{2})';$", re.M),
         lambda s: f"export const CURRENT_SEASON = '{s}';"),
        (os.path.join(frontend, "src", "lib", "archive-seasons.ts"),
         re.compile(r"^const CURRENT_END_YEAR = (\d{4}); // \d{4}-\d{2}$", re.M),
         lambda s: f"const CURRENT_END_YEAR = {int(s[:4]) + 1}; // {s}"),
    ]


def read_current(path: str, pattern: re.Pattern) -> str:
    """The season a file currently says, as 'YYYY-YY'."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    hits = pattern.findall(text)
    if len(hits) != 1:
        raise SystemExit(f"{path}: expected exactly one match for {pattern.pattern!r}, found {len(hits)}")
    v = hits[0]
    if len(v) == 4:                       # CURRENT_END_YEAR
        end = int(v)
        return f"{end - 1}-{end % 100:02d}"
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bump the current-season constants in both repos.")
    ap.add_argument("--to", help="Target season, e.g. 2026-27 (default: one after the current)")
    ap.add_argument("--apply", action="store_true", help="Write the change (default: show it)")
    ap.add_argument("--backend", default=BACKEND, help=argparse.SUPPRESS)
    ap.add_argument("--frontend", default=FRONTEND, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.frontend):
        print(f"Frontend repo not found at {args.frontend}")
        return 1
    files = targets(args.backend, args.frontend)
    current = {path: read_current(path, pat) for path, pat, _ in files}
    values = set(current.values())
    if len(values) != 1:
        print("The three constants disagree; fix by hand first:")
        for path, v in current.items():
            print(f"  {v}  {path}")
        return 1
    now = values.pop()
    target = args.to or next_season(now)
    m = SEASON.match(target)
    if not m or (int(m.group(1)) + 1) % 100 != int(m.group(2)):
        print(f"--to must look like 2026-27, got {target!r}")
        return 1
    if target <= now:
        print(f"Refusing to move backwards or stay put: {now} -> {target}")
        return 1
    if not args.to and target != season_for(date.today()):
        print(f"Note: today's calendar season is {season_for(date.today())}; "
              f"bumping {now} -> {target} anyway.")

    print(f"{'Applying' if args.apply else 'Would change'}: {now} -> {target}")
    for path, pat, fmt in files:
        with open(path, encoding="utf-8", newline="") as fh:
            text = fh.read()
        new = pat.sub(lambda _m: fmt(target), text, count=1)
        print(f"  {path}")
        if args.apply:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new)
    if not args.apply:
        print("Nothing written. Run again with --apply.")
        return 0

    after = {path: read_current(path, pat) for path, pat, _ in files}
    assert set(after.values()) == {target}, after
    print(f"""
Done: all three read {target}. Next:
  1. frontend: npx tsc --noEmit && npm run lint, then commit src/lib/nba-api.ts
     and src/lib/archive-seasons.ts and push
  2. backend: commit main_api.py and push (git push origin bettingbuddy-2.0:bettingbuddy2.0)
  3. restart the API (and redeploy, once deployed) so it serves {target}
  4. run preflight_opening_night.py: the season-label checks should now pass
  5. later, once the league announces it: set OPENING_NIGHT in
     preflight_opening_night.py to next fall's date (the preseason guard reads it)""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
