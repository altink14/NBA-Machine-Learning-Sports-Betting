"""
fix_wnba_snapshots_tagged_nba.py
================================
One-off correction: odds_snapshots rows labelled sport='NBA' whose teams are
WNBA teams. DRY RUN BY DEFAULT; nothing is written without --apply.

WHAT HAPPENED. main_api's PredictionRunner scrapes odds through
SbrOddsProvider, which falls back from NBA to WNBA when there are no NBA
games -- i.e. all summer. Until commit a49a956c (2026-08-08, "Secure the API
and fix the days-rest feature fed to the model") the runner labelled the
snapshot with the sport it had ASKED for, `snapshot_odds(odds_data,
self.sportsbook, self.sport)`, so every WNBA board scraped on a /predictions
call was written as sport='NBA'. That commit switched it to the sport the
provider actually resolved. The rows it left behind are ids 1-23, captured
2026-07-07 to 2026-07-10, all fanduel, all with a NULL start time (the
provider never read the tip-off; see SbrOddsProvider._tipoff). Every WNBA row
written since 2026-08-16 carries sport='WNBA'. Found by audit, confirmed
read-only 2026-09-23.

WHY IT MATTERS EVEN THOUGH NO NBA GAME MATCHES THEM. Anything that counts or
lists "NBA snapshots" by sport -- the archive's own stats, the line-movement
reader, the NBA repair's cross-check -- sees 23 NBA rows for games the NBA
never played.

WHAT THIS CHANGES. Only the `sport` column, from 'NBA' to 'WNBA', only on
rows whose home AND away team are both WNBA franchises, and only if every one
of those rows is still exactly as the dry run printed it. Nothing is deleted.
odds_snapshots has no immutability triggers (it is observations, not picks),
and ledger_sync's mirror keys it on (captured_at, sportsbook, game_key), so
the correction reaches the public copy as an ordinary update on the next
push rather than being refused.

Usage:
    venv/Scripts/python.exe fix_wnba_snapshots_tagged_nba.py            # dry run
    venv/Scripts/python.exe fix_wnba_snapshots_tagged_nba.py --apply    # the owner's call
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")

#: The fifteen WNBA franchises of 2026 (Portland Fire and Toronto Tempo joined
#: that season). None of these names is also an NBA team's.
WNBA_TEAMS = frozenset({
    "Atlanta Dream", "Chicago Sky", "Connecticut Sun", "Dallas Wings",
    "Golden State Valkyries", "Indiana Fever", "Las Vegas Aces", "Los Angeles Sparks",
    "Minnesota Lynx", "New York Liberty", "Phoenix Mercury", "Portland Fire",
    "Seattle Storm", "Toronto Tempo", "Washington Mystics",
})

_COLS = "id, captured_at, sportsbook, game_key, home_team, away_team, home_ml, away_ml, ou_line"


def find(conn: sqlite3.Connection) -> List[tuple]:
    ph = ",".join("?" * len(WNBA_TEAMS))
    teams = sorted(WNBA_TEAMS)
    return conn.execute(
        f"SELECT {_COLS} FROM odds_snapshots WHERE sport = 'NBA' "
        f"AND home_team IN ({ph}) AND away_team IN ({ph}) ORDER BY id",
        teams + teams).fetchall()


def half_wnba(conn: sqlite3.Connection) -> List[tuple]:
    """NBA-labelled rows with exactly one WNBA team: not ours to guess at."""
    ph = ",".join("?" * len(WNBA_TEAMS))
    teams = sorted(WNBA_TEAMS)
    return conn.execute(
        f"SELECT {_COLS} FROM odds_snapshots WHERE sport = 'NBA' "
        f"AND ((home_team IN ({ph})) + (away_team IN ({ph}))) = 1 ORDER BY id",
        teams + teams).fetchall()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Relabel WNBA odds snapshots stored as sport='NBA'.")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--apply", action="store_true", help="Write the change. Without it, only print.")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no database at {args.db}")
        return 1
    uri = f"file:{args.db}" + ("" if args.apply else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    try:
        rows = find(conn)
        odd = half_wnba(conn)
        print(f"{len(rows)} row(s) in odds_snapshots are labelled sport='NBA' but both teams are "
              f"WNBA franchises. Each would change sport 'NBA' -> 'WNBA', nothing else:")
        print(f"  {'id':>5}  {'captured_at':<28} {'book':<10} game_key")
        for r in rows:
            print(f"  {r[0]:>5}  {r[1]:<28} {r[2]:<10} {r[3]}   (ml {r[6]}/{r[7]}, total {r[8]})")
        if odd:
            print(f"\nNOT touched: {len(odd)} NBA-labelled row(s) with exactly one WNBA team, "
                  f"which this script will not guess about:")
            for r in odd:
                print(f"  {r[0]:>5}  {r[1]:<28} {r[2]:<10} {r[3]}")
        if not rows:
            print("nothing to do")
            return 0
        if not args.apply:
            print("\nDRY RUN: nothing was written. Re-run with --apply to make exactly these "
                  "changes.")
            return 0

        ids = [r[0] for r in rows]
        with conn:  # one transaction; any mismatch rolls the lot back
            n = conn.execute(
                f"UPDATE odds_snapshots SET sport = 'WNBA' WHERE sport = 'NBA' "
                f"AND id IN ({','.join('?' * len(ids))})", ids).rowcount
            if n != len(ids):
                raise RuntimeError(f"expected to relabel {len(ids)} row(s), matched {n}; "
                                   f"rolled back, nothing changed")
            left = find(conn)
            if left:
                raise RuntimeError(f"{len(left)} row(s) still mislabelled after the update; "
                                   f"rolled back")
        print(f"\nAPPLIED: {n} row(s) relabelled 'NBA' -> 'WNBA'.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
