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

SCHEDULE-AWARE BY DEFAULT (2026-09-19). The loop used to fire every N minutes
regardless of whether anything was about to tip, which paid full price to
discover there was nothing to watch AND captured whenever it happened to wake
up rather than near the close. It now checks the schedule first, which is
free: /events costs 0 credits (verified -- x-requests-last was 0 and the
remaining count did not move), so we can ask "is a game about to start?" as
often as we like and spend only when the answer is yes.

The ladder, same shape as the NFL recorder:

  tip within 30 min  -> capture, at most every 10 min  (this is the close)
  tip within 3 h     -> at most every 90 min
  tip within 30 h    -> at most every 12 h
  nothing within 30h -> no API call at all

QUOTA MATH, so nobody discovers it the hard way in November:
  one capture = 3 credits (3 markets x 1 region); schedule checks are free.
  - A blind 45-min loop, 12h/day in season: ~1,440/month -> the $59 tier.
  - Schedule-aware on a normal NBA night (games clustered into 2-3 tip times):
    roughly 6-10 captures a night, ~600-900/month -> fits the $30 tier, and
    lands within 30 minutes of each tip instead of wherever the clock fell.
  Remaining quota is logged on every run. --force overrides the ladder.

MISSED CAPTURES ARE REPAIRABLE, unlike an injury status. The Odds API keeps
historical snapshots back to 2020-06-06 at 5-minute resolution from September
2022, so a slate this missed can be bought back at 10 credits per market per
region. See docs/sources/odds-providers.md in the frontend repo. That is a
repair for accidents, not a substitute: a reconstructed price is one we read
out of an archive afterwards, not one we watched, and anything built on it has
to say so.
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Load the repo's .env explicitly, by path. Task Scheduler and a bare terminal
# both start this with an environment that has no ODDS_API_KEY in it; without
# this line the odds step only worked when main_api happened to be imported
# first (it calls load_dotenv), which is an accident, not a design.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from src.Utils.odds_api_client import (  # noqa: E402
    OddsApiError, fetch_nba_events, snapshot_nba_board,
)

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

#: Stop capturing when the month's remaining credits fall this low.
#:
#: NBA and NFL share one API key. From 20 October both recorders run on the
#: same 15-minute job, and an NBA night is the heavier of the two -- games
#: cluster into a few tip times, so the closing window fires repeatedly. With
#: no floor, a busy basketball week could drain the month and leave the NFL
#: silently uncaptured on the Sunday, which is the expensive failure: NFL
#: closing lines feed a ledger that is already live.
#:
#: The NFL recorder has had a floor of 60 from the start; this matches it, so
#: whichever sport hits the wall first leaves the same reserve for the other.
#: --force overrides, for when a human decides one slate matters more.
QUOTA_FLOOR = 60

#: (minutes until the nearest tip, minimum minutes between captures).
#: Checked in order; the first window the nearest tip falls into wins.
#:
#: DO NOT NARROW THE 30-MINUTE WINDOW TO SAVE CREDITS. It is the obvious
#: optimisation and it has been measured: `forecast_odds_credits.py --compare`
#: replays a real season through both. At 15 minutes the ladder makes 36%
#: fewer calls and, if every scheduled run fires, produces closing lines
#: indistinguishable from these -- 1.0 minutes before tip either way. The
#: saving is entirely insurance. Allow 5% of runs to miss, which is ordinary
#: for a scheduled task on a machine that sleeps, and the narrow ladder leaves
#: 5.2% of tips with a stale close against 0.2%, its worst close sliding from
#: 26 minutes early to 89. A price from 89 minutes out is not a close, and CLV
#: measured against one is not CLV. The second capture inside this window is
#: the retry; it is doing its job on exactly the nights nothing looks wrong.
CAPTURE_LADDER = ((30, 10), (180, 90), (1800, 720))


def _minutes_to_nearest_tip(events) -> float:
    """Minutes until the next tip-off, or inf if nothing is scheduled."""
    now = datetime.now(timezone.utc)
    soonest = None
    for e in events:
        ct = e.get("commence_time")
        if not ct:
            continue
        try:
            t = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
        except ValueError:
            continue
        if t > now and (soonest is None or t < soonest):
            soonest = t
    return float("inf") if soonest is None else (soonest - now).total_seconds() / 60.0


def _minutes_since_last_capture() -> float:
    """How long since we last wrote a snapshot, from the archive itself."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            row = conn.execute("SELECT MAX(captured_at) FROM odds_snapshots").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return float("inf")
    if not row or not row[0]:
        return float("inf")
    try:
        last = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds() / 60.0


def should_capture(events) -> tuple:
    """(capture?, why). The whole point of not looping blindly lives here."""
    mins_out = _minutes_to_nearest_tip(events)
    if mins_out == float("inf"):
        return False, "no NBA game is scheduled"
    since = _minutes_since_last_capture()
    for window, cooldown in CAPTURE_LADDER:
        if mins_out <= window:
            label = "closing window" if window == 30 else f"tip in {mins_out/60:.1f}h"
            if since >= cooldown:
                return True, f"{label}, last capture {since:.0f} min ago"
            return False, f"{label} but captured {since:.0f} min ago (cooldown {cooldown})"
    return False, f"nearest tip is {mins_out/60:.1f}h away"


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot the NBA odds board from The Odds API.")
    ap.add_argument("--loop", type=int, metavar="MINUTES",
                    help="Keep running, snapshotting every N minutes.")
    ap.add_argument("--bookmakers", help="Comma-separated book keys (default: all US books).")
    ap.add_argument("--force", action="store_true",
                    help="Capture regardless of the schedule ladder.")
    ap.add_argument("--blind", action="store_true",
                    help="Skip the free schedule check entirely (the old behaviour). "
                         "Costs credits to learn there is nothing to watch.")
    args = ap.parse_args()

    while True:
        try:
            if not args.blind and not args.force:
                events, eq = fetch_nba_events()          # free: 0 credits
                ok, why = should_capture(events)
                if not ok:
                    logger.info("skipping: %s (%d event(s) on the board, schedule check "
                                "cost %s credits)", why, len(events), eq.get("last") or "0")
                    if not args.loop:
                        return 0
                    time.sleep(args.loop * 60)
                    continue
                remaining = eq.get("remaining")
                if remaining is not None and int(remaining) < QUOTA_FLOOR and not args.force:
                    logger.warning(
                        "NOT capturing despite %s: only %s credits remain, below the floor of "
                        "%d. The reserve is held so the other sport's recorder is not starved "
                        "by this one -- they share a key. Raise the tier or wait for the reset.",
                        why, remaining, QUOTA_FLOOR)
                    if not args.loop:
                        return 0
                    time.sleep(args.loop * 60)
                    continue
                logger.info("capturing: %s", why)
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
