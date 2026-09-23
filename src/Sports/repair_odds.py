"""
repair_odds.py (cross-sport)
============================
Rebuilds market snapshots for games whose live capture was missed, from The
Odds API's historical archive, and marks every row it writes as
`provenance = 'reconstructed'`.

WHY THIS EXISTS. The odds recorder runs every 15 minutes on a home PC. A PC
that sleeps through a Sunday used to mean that slate's closing lines were gone
for good. They are not: The Odds API keeps odds snapshots back to 2020-06-06
at 10-minute resolution, and 5-minute from September 2022. This script is the
repair path for that, and only for that.

WHAT IT IS NOT. It is not a substitute for the recorder and it must not become
one. A reconstructed price is one we read out of a vendor's archive after the
fact; an observed price is one we watched. Both are honest, they are not the
same claim, and closing line value has to state which it rests on. That is why
every row written here carries 'reconstructed' and why `seal()` refuses to let
a backfill outrank a price we watched. If you find yourself running this
routinely instead of keeping the machine awake, the product has quietly gotten
worse in a way no number will show you.

IT COSTS REAL MONEY AND THE FREE TIER CANNOT DO IT. Historical odds are a paid
feature, and each call costs 10 credits per market per region, against 1 for a
live call. One repaired kickoff time is 30 credits at our three markets. The
script therefore refuses to spend anything without --apply, prints the bill
first, stops at --max-credits, and will not drive the account below the same
quota floor the recorder respects.

WHAT IT REFUSES TO DO.

  - It will not write a snapshot for a game that has not kicked off. Those are
    the live recorder's job, and a "historical" price for a future game is a
    contradiction.
  - It will not write a snapshot timestamped at or after kickoff. The archive
    returns the closest snapshot at or before the timestamp you ask for, which
    is usually what you want, but if that lands after kickoff the row is
    dropped rather than sealed into a closing line that was never a close.
  - It will not duplicate a snapshot we already hold for the same game, book,
    market and moment, so re-running is safe.
  - It never touches outcomes. It reads prices and schedules only, which is
    what keeps it usable on a sport whose season is under seal.

Usage:
    # what is missing, and what repairing it would cost -- spends nothing
    venv/Scripts/python.exe src/Sports/repair_odds.py --sport nfl --since 2026-09-20

    # actually do it
    venv/Scripts/python.exe src/Sports/repair_odds.py --sport nfl --since 2026-09-20 --apply

    # a single game, and a tighter budget
    venv/Scripts/python.exe src/Sports/repair_odds.py --game 2026_02_CAR_ATL --apply --max-credits 60
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import requests  # noqa: E402

from src.Sports.core_schema import INGEST_VERSION, ensure_core_schema, record_run  # noqa: E402
from src.Sports.odds_recorder import (  # noqa: E402
    DEFAULT_MARKETS,
    DEFAULT_REGIONS,
    ODDS_API_BASE,
    QUOTA_FLOOR,
    SPORTS,
    api_key,
    ensure_snapshot_schema,
    match_games,
    rows_from,
    seal,
)

logger = logging.getLogger("repair_odds")

#: The archive does not go back further than this. Asking for an earlier
#: timestamp returns nothing useful, so we say so rather than spending credits
#: to find out. Documented by The Odds API: snapshots from 2020-06-06, at
#: 10-minute intervals, and 5-minute from September 2022.
ARCHIVE_START = "2020-06-06T00:00:00+00:00"

#: How far before kickoff to ask for. The archive returns the closest snapshot
#: at or earlier than the timestamp requested, so asking a little before
#: kickoff gets the last price that existed while the game could still be bet.
#: Two minutes is inside the 5-minute grid, so in practice this returns the
#: final pre-kickoff snapshot.
LOOKBACK_MINUTES = 2

#: 10 credits per market per region, per the API's documented pricing.
HISTORICAL_MULTIPLIER = 10

#: Default window, in days, when no --since is given. This is a repair tool
#: for runs we missed recently, not a way to buy history. Every game before
#: the recorder existed looks like a "gap" to the query below, and without a
#: default window an innocent-looking invocation would price up the entire
#: archive. Reaching further back is a deliberate purchase with its own
#: decision attached -- see docs/sources/odds-providers.md, open question 1 --
#: so it has to be asked for by date.
DEFAULT_WINDOW_DAYS = 14

#: Unattended defaults, used by the daily job. Deliberately much tighter than
#: the manual ones. A script that spends money on a timer should repair the
#: slate that just happened and nothing else: three days covers a weekend the
#: machine slept through plus the Thursday and Monday games, and 90 credits is
#: three kickoff times. A backlog bigger than that is worked down a little each
#: night rather than bought in one unsupervised gulp, and anything older than
#: the window stays a decision a person makes.
UNATTENDED_WINDOW_DAYS = 3
UNATTENDED_MAX_CREDITS = 90

#: Headroom left for the live recorder on top of its own floor. Repair is the
#: lower priority of the two: a price we can still watch beats one we would
#: have to buy back later.
UNATTENDED_RESERVE = 120


class RepairUnavailable(RuntimeError):
    """We cannot repair right now. Not a bug, and not worth a red daily job."""


class HistoricalUnavailable(RepairUnavailable):
    """The plan cannot read historical odds. Expected on the free tier."""


class QuotaExhausted(RepairUnavailable):
    """No credits left this month."""


def credits_per_call(markets: str, regions: str) -> int:
    return HISTORICAL_MULTIPLIER * len(markets.split(",")) * len(regions.split(","))


def gaps(conn: sqlite3.Connection, since: Optional[str], until: Optional[str],
         game_id: Optional[str]) -> List[Tuple[str, str]]:
    """Games that have kicked off but hold no pre-kickoff snapshot of our own.

    That is the definition of a missed capture. A game with even one
    pre-kickoff snapshot is not a gap: the recorder was alive and the seal has
    something honest to work with, even if it is not as close to kickoff as we
    would like. Repairing those too would spend credits to slightly improve a
    price we already watched, and would risk a reconstructed row displacing an
    observed one.
    """
    now = datetime.now(timezone.utc).isoformat()
    sql = """
        SELECT g.game_id, g.date_utc
        FROM games g
        WHERE g.date_utc IS NOT NULL
          AND g.date_utc < ?
          AND g.date_utc >= ?
          AND NOT EXISTS (
                SELECT 1 FROM market_line_snapshots s
                WHERE s.game_id = g.game_id AND s.captured_at < g.date_utc)
    """
    args: List[str] = [now, ARCHIVE_START]
    if since:
        sql += " AND g.date_utc >= ?"
        args.append(since)
    if until:
        sql += " AND g.date_utc <= ?"
        args.append(until)
    if game_id:
        sql += " AND g.game_id = ?"
        args.append(game_id)
    sql += " ORDER BY g.date_utc"
    return [(r[0], r[1]) for r in conn.execute(sql, args).fetchall()]


def fetch_historical(sport_key: str, when: str, markets: str, regions: str,
                     bookmakers: Optional[str]) -> Tuple[str, List[Dict], Dict[str, Optional[str]]]:
    """One snapshot from the archive. Returns (snapshot_timestamp, events, quota)."""
    params = {
        "apiKey": api_key(),
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": when,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    r = requests.get(f"{ODDS_API_BASE}/historical/sports/{sport_key}/odds",
                     params=params, timeout=60)
    quota = {"remaining": r.headers.get("x-requests-remaining"),
             "used": r.headers.get("x-requests-used"),
             "last": r.headers.get("x-requests-last")}
    if r.status_code in (401, 403):
        raise HistoricalUnavailable(
            "The Odds API refused the historical endpoint (HTTP %d). Historical odds are a "
            "paid feature; a free-tier key cannot read them. This is the repair path's one "
            "hard dependency -- see docs/sources/odds-providers.md. Nothing was written."
            % r.status_code)
    if r.status_code == 422:
        raise SystemExit(f"The API rejected the timestamp {when} (HTTP 422). The archive starts "
                         f"{ARCHIVE_START}. Nothing was written.")
    if r.status_code == 429:
        raise QuotaExhausted(f"Quota exhausted. Remaining={quota['remaining']}. Nothing further "
                             f"was written; re-run after the quota resets or raise the tier.")
    r.raise_for_status()
    body = r.json()
    return body.get("timestamp"), (body.get("data") or []), quota


def remaining_quota() -> Optional[int]:
    """Credits left, read from a call that costs none. None if it cannot be read."""
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key()}, timeout=30)
        v = r.headers.get("x-requests-remaining")
        return int(v) if v is not None else None
    except Exception:
        return None


def already_have(conn: sqlite3.Connection, game_id: str, captured_at: str) -> bool:
    return conn.execute(
        "SELECT EXISTS(SELECT 1 FROM market_line_snapshots WHERE game_id = ? "
        "AND captured_at = ?)", (game_id, captured_at)).fetchone()[0] == 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild missed market snapshots from The Odds API's historical archive.")
    ap.add_argument("--sport", default="nfl", choices=sorted(SPORTS))
    ap.add_argument("--since", help="Only games kicking off at or after this ISO date.")
    ap.add_argument("--until", help="Only games kicking off at or before this ISO date.")
    ap.add_argument("--game", help="Repair exactly one game_id.")
    ap.add_argument("--markets", default=DEFAULT_MARKETS)
    ap.add_argument("--regions", default=DEFAULT_REGIONS)
    ap.add_argument("--bookmakers", help="Comma-separated book keys (default: all in region).")
    ap.add_argument("--max-credits", type=int, default=300,
                    help="Refuse to start if the estimate exceeds this. Default 300.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually spend credits and write rows. Without it, nothing is called.")
    ap.add_argument("--unattended", action="store_true",
                    help="Scheduled-job mode: a 3-day window, a 90-credit cap, extra quota "
                         "reserved for the live recorder, a backlog worked down a slice at a "
                         "time, and exit 0 for anything that is merely 'cannot right now'.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = SPORTS[args.sport]
    conn = sqlite3.connect(cfg["db"], timeout=120)
    ensure_core_schema(conn)
    ensure_snapshot_schema(conn)

    window_days = UNATTENDED_WINDOW_DAYS if args.unattended else DEFAULT_WINDOW_DAYS
    if args.unattended and ap.get_default("max_credits") == args.max_credits:
        args.max_credits = UNATTENDED_MAX_CREDITS

    since = args.since
    if since is None and args.game is None:
        since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        older = len(gaps(conn, None, since, None))
        logger.info("no --since given, so looking back %d days (from %s).",
                    window_days, since[:10])
        if older:
            logger.info("%d older game(s) also lack a snapshot, mostly from before the recorder "
                        "existed. Those are a purchase, not a repair: name a date with --since "
                        "if you really mean to buy them.", older)

    missing = gaps(conn, since, args.until, args.game)
    if not missing:
        logger.info("no gaps: every finished game in range already has a pre-kickoff snapshot")
        conn.close()
        return 0

    # One archive call returns every game the sport had listed at that instant,
    # so games sharing a kickoff time share a call. This is why a whole missed
    # Sunday costs about as much as a single missed game.
    by_kickoff: Dict[str, List[str]] = {}
    for gid, kickoff in missing:
        by_kickoff.setdefault(kickoff, []).append(gid)

    per_call = credits_per_call(args.markets, args.regions)
    estimate = per_call * len(by_kickoff)

    logger.info("%d game(s) with no pre-kickoff snapshot, across %d distinct kickoff time(s)",
                len(missing), len(by_kickoff))
    for kickoff in sorted(by_kickoff):
        logger.info("   %s  %s", kickoff, ", ".join(sorted(by_kickoff[kickoff])))
    logger.info("estimate: %d call(s) x %d credits = %d credits (%s x %s at %dx historical rate)",
                len(by_kickoff), per_call, estimate, args.markets, args.regions,
                HISTORICAL_MULTIPLIER)

    if estimate > args.max_credits:
        if not args.unattended:
            logger.error("estimate %d exceeds --max-credits %d. Narrow the range with "
                         "--since/--until or raise the budget deliberately. Nothing was called.",
                         estimate, args.max_credits)
            conn.close()
            return 1
        # Unattended: do not stall forever on a backlog. Take the most recent
        # kickoffs that fit tonight's budget and leave the rest for tomorrow,
        # newest first because a fresh gap is the one most likely to matter.
        affordable = max(0, args.max_credits // per_call)
        if affordable == 0:
            logger.warning("budget %d cannot afford even one call at %d credits; nothing done.",
                           args.max_credits, per_call)
            conn.close()
            return 0
        keep = sorted(by_kickoff, reverse=True)[:affordable]
        deferred = len(by_kickoff) - len(keep)
        by_kickoff = {k: by_kickoff[k] for k in keep}
        estimate = per_call * len(by_kickoff)
        logger.info("budget %d credits: repairing the %d most recent kickoff time(s) tonight "
                    "(%d credits), leaving %d for a later run.",
                    args.max_credits, len(by_kickoff), estimate, deferred)

    if not args.apply:
        logger.info("dry run: nothing called, nothing written, no credits spent. "
                    "Re-run with --apply to repair.")
        conn.close()
        return 0

    if args.unattended:
        # The live recorder has first claim on the month's credits. Repair only
        # proceeds if the whole bill fits above the recorder's floor plus a
        # reserve, because a price we can still watch beats one we would have
        # to buy back.
        have = remaining_quota()
        need = estimate + QUOTA_FLOOR + UNATTENDED_RESERVE
        # An UNKNOWN balance is not permission to spend. This used to test
        # `have is not None and have < need`, so whenever the balance could
        # not be read -- a network blip, a 401, a renamed header -- the whole
        # guard was skipped and the job spent anyway, unattended, with no
        # line explaining why. The unattended path is the one with no human
        # to notice, so it refuses; a manual run can still override.
        if have is None:
            logger.warning("skipping repair: could not read the remaining quota, and an "
                           "unattended run does not spend credits it cannot account "
                           "for. It will try again tomorrow.")
            conn.close()
            return 0
        if have < need:
            logger.warning("skipping repair: %d credits remain, and spending %d would leave "
                           "less than the recorder's floor (%d) plus its reserve (%d). The "
                           "live capture matters more than the backfill.",
                           have, estimate, QUOTA_FLOOR, UNATTENDED_RESERVE)
            conn.close()
            return 0

    started_at = datetime.now(timezone.utc).isoformat()
    written = skipped_dupe = skipped_late = 0
    calls = 0
    remaining: Optional[str] = None

    interrupted = None
    try:
        for kickoff in sorted(by_kickoff):
            ko = datetime.fromisoformat(kickoff)
            when = (ko - timedelta(minutes=LOOKBACK_MINUTES)).isoformat().replace("+00:00", "Z")

            snap_ts, events, quota = fetch_historical(
                cfg["odds_api_key"], when, args.markets, args.regions, args.bookmakers)
            calls += 1
            remaining = quota.get("remaining")
            logger.info("kickoff %s: asked for %s, archive returned %s (%d event(s), "
                        "cost %s, remaining %s)",
                        kickoff, when, snap_ts, len(events), quota.get("last"), remaining)

            if not snap_ts:
                logger.warning("   no snapshot timestamp returned; skipping")
                continue
            # The archive gives the closest snapshot at or before what we asked
            # for. If that still lands after kickoff, it is not a pre-kickoff
            # price and must not become a closing line.
            if snap_ts >= kickoff:
                logger.warning("   snapshot %s is not before kickoff %s; refusing to write it",
                               snap_ts, kickoff)
                skipped_late += 1
                continue

            match = match_games(conn, events, cfg)
            wanted = set(by_kickoff[kickoff])
            rows = [r for r in rows_from(events, match, snap_ts, "the-odds-api (historical)")
                    if r[0] in wanted]
            rows = [r for r in rows if not already_have(conn, r[0], snap_ts)]
            if not rows:
                logger.info("   nothing new to write for this kickoff")
                skipped_dupe += 1
                continue

            conn.executemany(
                "INSERT INTO market_line_snapshots (game_id, captured_at, commence_time, book, "
                "market_type, line, price_home, price_away, price_over, price_under, source, "
                "ingest_version, provenance) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'reconstructed')", rows)
            conn.commit()
            written += len(rows)
            logger.info("   wrote %d reconstructed snapshot row(s)", len(rows))

            if remaining is not None and int(remaining) < QUOTA_FLOOR:
                logger.warning("quota remaining %s is below the floor of %d; stopping here so the "
                               "live recorder is not starved. Re-run to continue.",
                               remaining, QUOTA_FLOOR)
                break

    except RepairUnavailable as exc:
        # Not a bug: the plan cannot read history, or the month's credits
        # are gone. Whatever was written before this point stays written and
        # still gets sealed below. A scheduled run says so and exits clean,
        # because a daily job that reports failure every night for a reason
        # nobody can act on tonight is a job people stop reading.
        interrupted = str(exc)
        (logger.warning if args.unattended else logger.error)('%s', interrupted)

    sealed = seal(conn)
    record_run(conn, "market_line_snapshots", "the-odds-api (historical)",
               f"{ODDS_API_BASE}/historical/sports/{cfg['odds_api_key']}/odds",
               started_at, datetime.now(timezone.utc).isoformat(), written,
               notes=f"repair: calls={calls} kickoffs={len(by_kickoff)} written={written} "
                     f"dupe={skipped_dupe} late={skipped_late} sealed={sealed} "
                     f"quota_remaining={remaining}")

    logger.info("SUMMARY calls=%d written=%d (reconstructed) sealed=%d quota_remaining=%s%s",
                calls, written, sealed, remaining,
                "  [stopped early: see above]" if interrupted else "")
    if written:
        logger.info("These prices were read from an archive, not watched. Any closing line "
                    "value computed from them must say so; the provenance column carries it.")
    conn.close()
    # Unattended runs only fail on something a person could actually fix
    # tonight. A missing paid tier is not that.
    if interrupted and not args.unattended:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
