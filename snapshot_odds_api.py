"""
snapshot_odds_api.py
====================
Archive the NBA odds board from The Odds API into odds_snapshots.

USAGE
  venv\\Scripts\\python.exe snapshot_odds_api.py            # one snapshot
  venv\\Scripts\\python.exe snapshot_odds_api.py --loop 45  # every 45 minutes, forever

SETUP
  1. Create a key at the-odds-api.com (free tier: 500 credits/month).
  2. Set ODDS_API_KEY in the environment this runs under.

QUOTA MATH, so nobody discovers it the hard way in November:
  one run = 3 credits (3 markets x 1 region).
  - Once daily (the daily_update step): ~90 credits/month -> FREE tier.
  - Every 45 min, 12h/day in season: ~1,440/month -> needs the $59 tier.
  The remaining quota is logged on every run.

THE OCTOBER STEP-UP (before opening night):
  schtasks /create /tn "BettingBuddy Odds Snapshots" ^
    /tr "<venv python> <this file> --loop 45" /sc onstart
  ...or run it in a terminal during slates. Closing lines can only be
  captured live; missed weeks are unrecoverable.
"""

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.Utils.odds_api_client import OddsApiError, snapshot_nba_board  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(REPO_ROOT, "odds_snapshots.log")),
    ],
)
logger = logging.getLogger("snapshot_odds_api")

DB_PATH = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot the NBA odds board from The Odds API.")
    ap.add_argument("--loop", type=int, metavar="MINUTES",
                    help="Keep running, snapshotting every N minutes.")
    ap.add_argument("--bookmakers", help="Comma-separated book keys (default: all US books).")
    args = ap.parse_args()

    while True:
        try:
            summary = snapshot_nba_board(DB_PATH, bookmakers=args.bookmakers)
            logger.info(
                "OK: %s events, %s rows written, %s unchanged, quota remaining %s",
                summary["events"], summary["written"], summary["unchanged"],
                summary["quota_remaining"],
            )
        except OddsApiError as e:
            logger.error("%s", e)
            if not args.loop:
                return 1
        except Exception as e:  # a hiccup in a loop must not kill the recorder
            logger.error("Snapshot failed: %s", e, exc_info=True)
            if not args.loop:
                return 1
        if not args.loop:
            return 0
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    raise SystemExit(main())
