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

THE NBA PATH (--sport nba, added 2026-09-23). Same guards, three differences
forced by where the NBA's data lives:

  - THE SCHEDULE COMES FROM ESPN. The NFL decides what is missing against its
    own `games` table. The NBA has no table of tip-off times: the archive holds
    finished games by date only, and cdn.nba.com's schedule answers 403 from
    the home network (verified 2026-09-23). ESPN's public scoreboard, one free
    request per day in the window, carries every game's scheduled tip in UTC
    and its season type. If it cannot be read the run FAILS (exit 1, attended
    or not): "we could not see the schedule" must never be reported as
    "nothing was missed". Preseason and All-Star games are not repaired.

  - THE SNAPSHOTS LIVE IN OddsData `odds_snapshots`, written by
    snapshot_odds_api.py (and, for one book, by main_api's SBR path), keyed by
    'Home:Away' full names. A repaired row goes into that same table with
    `provenance = 'reconstructed'`, which grade_predictions.price_clv() already
    carries through as `closing_provenance`. There is no seal step: the NBA's
    CLV reads odds_snapshots directly.

  - "COVERED" MEANS A SNAPSHOT WITHIN 3 HOURS OF TIP, not merely before it.
    The NBA board lists games weeks ahead (a Christmas game was on the board
    on 22 September), and daily_update takes one board snapshot every morning,
    so under the NFL's any-pre-kickoff rule almost no missed evening would
    ever count as missed: its "close" would be a price from 9 a.m. or from
    last month. Three hours is the recorder's second rung (it captures at
    least every 90 minutes inside it), so a snapshot there means the recorder
    was alive for the approach to tip. NBA_COVERED_WITHIN_MINUTES is the knob;
    widening it spends less and accepts staler closes.

Usage:
    # what is missing, and what repairing it would cost -- spends nothing
    venv/Scripts/python.exe src/Sports/repair_odds.py --sport nfl --since 2026-09-20

    # actually do it
    venv/Scripts/python.exe src/Sports/repair_odds.py --sport nfl --since 2026-09-20 --apply

    # a single game, and a tighter budget
    venv/Scripts/python.exe src/Sports/repair_odds.py --game 2026_02_CAR_ATL --apply --max-credits 60

    # the NBA: what is missing over the last 14 days, spending nothing
    venv/Scripts/python.exe src/Sports/repair_odds.py --sport nba
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
from src.Utils.odds_api_client import (  # noqa: E402
    ensure_snapshot_schema as ensure_nba_snapshot_schema,
    events_to_rows,
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


def fit_budget(by_kickoff: Dict[str, List[str]], per_call: int, max_credits: int,
               unattended: bool) -> Tuple[Optional[Dict[str, List[str]]], int, str]:
    """Trim a repair plan to the budget. Returns (plan, estimate, verdict).

    verdict is 'ok', 'over_budget' (attended: refuse, the caller exits 1) or
    'unaffordable' (unattended: not even one call fits, exit 0). Shared by both
    sports so the NBA inherits the NFL's rules rather than a copy of them.
    """
    estimate = per_call * len(by_kickoff)
    if estimate <= max_credits:
        return by_kickoff, estimate, "ok"
    if not unattended:
        logger.error("estimate %d exceeds --max-credits %d. Narrow the range with "
                     "--since/--until or raise the budget deliberately. Nothing was called.",
                     estimate, max_credits)
        return None, estimate, "over_budget"
    # Unattended: do not stall forever on a backlog. Take the most recent
    # kickoffs that fit tonight's budget and leave the rest for tomorrow,
    # newest first because a fresh gap is the one most likely to matter.
    affordable = max(0, max_credits // per_call)
    if affordable == 0:
        logger.warning("budget %d cannot afford even one call at %d credits; nothing done.",
                       max_credits, per_call)
        return None, 0, "unaffordable"
    keep = sorted(by_kickoff, reverse=True)[:affordable]
    deferred = len(by_kickoff) - len(keep)
    plan = {k: by_kickoff[k] for k in keep}
    estimate = per_call * len(plan)
    logger.info("budget %d credits: repairing the %d most recent kickoff time(s) tonight "
                "(%d credits), leaving %d for a later run.",
                max_credits, len(plan), estimate, deferred)
    return plan, estimate, "ok"


def unattended_quota_refusal(estimate: int) -> Optional[str]:
    """Why an unattended run must not spend `estimate`, or None if it may.

    The live recorder has first claim on the month's credits. Repair only
    proceeds if the whole bill fits above the recorder's floor plus a
    reserve, because a price we can still watch beats one we would have
    to buy back.
    """
    have = remaining_quota()
    need = estimate + QUOTA_FLOOR + UNATTENDED_RESERVE
    # An UNKNOWN balance is not permission to spend. This used to test
    # `have is not None and have < need`, so whenever the balance could
    # not be read -- a network blip, a 401, a renamed header -- the whole
    # guard was skipped and the job spent anyway, unattended, with no
    # line explaining why. The unattended path is the one with no human
    # to notice, so it refuses; a manual run can still override.
    if have is None:
        return ("skipping repair: could not read the remaining quota, and an unattended run "
                "does not spend credits it cannot account for. It will try again tomorrow.")
    if have < need:
        return ("skipping repair: %d credits remain, and spending %d would leave less than the "
                "recorder's floor (%d) plus its reserve (%d). The live capture matters more "
                "than the backfill." % (have, estimate, QUOTA_FLOOR, UNATTENDED_RESERVE))
    return None


# ---------------------------------------------------------------------------
# NFL
# ---------------------------------------------------------------------------

def _run_nfl(args: argparse.Namespace, window_days: int) -> int:
    cfg = SPORTS[args.sport]
    conn = sqlite3.connect(cfg["db"], timeout=120)
    ensure_core_schema(conn)
    ensure_snapshot_schema(conn)

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

    logger.info("%d game(s) with no pre-kickoff snapshot, across %d distinct kickoff time(s)",
                len(missing), len(by_kickoff))
    for kickoff in sorted(by_kickoff):
        logger.info("   %s  %s", kickoff, ", ".join(sorted(by_kickoff[kickoff])))
    logger.info("estimate: %d call(s) x %d credits = %d credits (%s x %s at %dx historical rate)",
                len(by_kickoff), per_call, per_call * len(by_kickoff), args.markets,
                args.regions, HISTORICAL_MULTIPLIER)

    plan, estimate, verdict = fit_budget(by_kickoff, per_call, args.max_credits, args.unattended)
    if plan is None:
        conn.close()
        return 1 if verdict == "over_budget" else 0
    by_kickoff = plan

    if not args.apply:
        logger.info("dry run: nothing called, nothing written, no credits spent. "
                    "Re-run with --apply to repair.")
        conn.close()
        return 0

    if args.unattended:
        refusal = unattended_quota_refusal(estimate)
        if refusal:
            logger.warning("%s", refusal)
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


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------

NBA_ODDS_API_KEY = "basketball_nba"
NBA_ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

#: ESPN season types worth buying a close for. 1 is preseason and 4 the
#: All-Star break: neither is predicted, and neither is worth credits.
NBA_REPAIRABLE_SEASON_TYPES = {2: "regular season", 3: "postseason", 5: "play-in"}

#: A game with a snapshot inside this many minutes before tip is not a gap.
#: See "COVERED" in the module docstring for why this is not the NFL's rule.
NBA_COVERED_WITHIN_MINUTES = 180

#: How far an archive event's commence_time may sit from ESPN's scheduled tip
#: and still be the same game. Teams meet two to four times a season, so the
#: team names alone would also match the rematch.
NBA_SAME_GAME_HOURS = 6

#: ESPN statuses for a game that never started; no close exists to buy.
_NEVER_PLAYED = ("STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED")


class ScheduleUnavailable(RuntimeError):
    """We could not read the schedule, so we cannot know what was missed.

    Deliberately NOT a RepairUnavailable: those exit 0 unattended, and a run
    that could not look must never read the same as a run that found nothing.
    """


def canon_team(name: Optional[str]) -> str:
    """One spelling per team for matching. ESPN (and the SBR provider) say
    'LA Clippers'; The Odds API says 'Los Angeles Clippers'. Rows are still
    written with The Odds API's own names, exactly as the live recorder does."""
    n = (name or "").strip()
    return "Los Angeles Clippers" if n == "LA Clippers" else n


def canon_key(game_key: Optional[str]) -> str:
    home, _, away = (game_key or "").partition(":")
    return f"{canon_team(home)}:{canon_team(away)}"


def parse_utc(ts: Optional[str]) -> Optional[datetime]:
    """Any timestamp this table holds, as an aware UTC datetime. The
    recorder writes '+00:00', ESPN and The Odds API write 'Z', and main_api's
    SBR path wrote naive utcnow() -- all three are UTC."""
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def fetch_espn_day(day: str) -> Dict:
    """ESPN's NBA scoreboard for one US date (YYYYMMDD). Free. Raises
    ScheduleUnavailable on anything but a 200 carrying an 'events' list:
    an unreadable day must not look like a day with no games."""
    try:
        r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": day}, timeout=30)
    except requests.RequestException as exc:
        raise ScheduleUnavailable(f"ESPN scoreboard for {day}: {exc}") from exc
    if r.status_code != 200:
        raise ScheduleUnavailable(f"ESPN scoreboard for {day}: HTTP {r.status_code}")
    try:
        body = r.json()
    except ValueError as exc:
        raise ScheduleUnavailable(f"ESPN scoreboard for {day}: body is not JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("events"), list):
        raise ScheduleUnavailable(f"ESPN scoreboard for {day}: no 'events' list; the feed's "
                                  f"shape has changed")
    return body


def nba_schedule(since: datetime, until: datetime,
                 fetch_day=fetch_espn_day) -> Tuple[List[Dict], List[Dict]]:
    """(repairable games, every event seen) tipping in [since, until).

    One request per US date. The window is widened by a day at the front
    because an evening tip in the east is already tomorrow in UTC.
    """
    games: List[Dict] = []
    seen: List[Dict] = []
    day = (since - timedelta(days=1)).date()
    while day <= until.date():
        body = fetch_day(day.strftime("%Y%m%d"))
        for e in body["events"]:
            try:
                comp = (e.get("competitions") or [{}])[0]
                sides = {c.get("homeAway"): (c.get("team") or {}).get("displayName")
                         for c in comp.get("competitors") or []}
                tip = parse_utc(e.get("date") or comp.get("date"))
                stype = (e.get("season") or {}).get("type")
                status = (((e.get("status") or {}).get("type")) or {}).get("name") or ""
            except (AttributeError, IndexError, TypeError) as exc:
                raise ScheduleUnavailable(f"ESPN scoreboard for {day}: unreadable event "
                                          f"{e.get('id') if isinstance(e, dict) else e!r}: "
                                          f"{exc}") from exc
            if not (tip and sides.get("home") and sides.get("away")):
                raise ScheduleUnavailable(f"ESPN scoreboard for {day}: event {e.get('id')} has "
                                          f"no tip time or no teams")
            g = {"event_id": str(e.get("id")), "tip": tip, "season_type": stype,
                 "status": status, "home": sides["home"], "away": sides["away"],
                 "key": canon_key(f"{sides['home']}:{sides['away']}")}
            if not (since <= tip < until):
                continue
            seen.append(g)
            if stype in NBA_REPAIRABLE_SEASON_TYPES and status not in _NEVER_PLAYED:
                games.append(g)
        day += timedelta(days=1)
    # A game can be listed on two adjacent US dates' boards; keep one.
    uniq = {g["event_id"]: g for g in games}
    return sorted(uniq.values(), key=lambda g: g["tip"]), seen


def nba_snapshots(conn: sqlite3.Connection, since: datetime) -> List[Tuple[str, datetime, Optional[datetime]]]:
    """(canonical game_key, captured_at, game_start) for NBA snapshots that
    could cover a game in the window. Parsed in Python because the table
    mixes three timestamp spellings."""
    floor = (since - timedelta(days=2)).date().isoformat()
    out = []
    for key, cap, start in conn.execute(
            "SELECT game_key, captured_at, game_start_time_utc FROM odds_snapshots "
            "WHERE sport = 'NBA' AND captured_at >= ?", (floor,)):
        c = parse_utc(cap)
        if c is not None:
            out.append((canon_key(key), c, parse_utc(start)))
    return out


def nba_gaps(conn: sqlite3.Connection, schedule: List[Dict], since: datetime) -> List[Dict]:
    """Scheduled games with no snapshot of ours in the last
    NBA_COVERED_WITHIN_MINUTES before tip. Any provenance counts, so a game
    already repaired is not bought twice."""
    snaps = nba_snapshots(conn, since)
    window = timedelta(minutes=NBA_COVERED_WITHIN_MINUTES)
    return [g for g in schedule
            if not any(k == g["key"] and g["tip"] - window <= c < g["tip"]
                       for k, c, _ in snaps)]


def nba_unlisted(conn: sqlite3.Connection, seen: List[Dict], since: datetime,
                 until: datetime) -> List[Tuple[str, datetime]]:
    """Games our own recorder saw tipping in the window that ESPN's full
    list for those days (every season type) does not contain.

    The cross-check on the schedule we are trusting. ESPN returning a valid
    but empty day is the one failure its shape checks cannot catch, and it is
    exactly the failure that would read as "nothing to repair".
    """
    listed = [(g["key"], g["tip"]) for g in seen]
    same = timedelta(hours=NBA_SAME_GAME_HOURS)
    return sorted({(k, s) for k, _, s in nba_snapshots(conn, since)
                   if s is not None and since <= s < until
                   and not any(k == lk and abs(s - lt) <= same for lk, lt in listed)},
                  key=lambda x: x[1])


def already_have_nba(conn: sqlite3.Connection, book: str, game_key: str, captured_at: str) -> bool:
    return conn.execute(
        "SELECT EXISTS(SELECT 1 FROM odds_snapshots WHERE sport = 'NBA' AND sportsbook = ? "
        "AND game_key = ? AND captured_at = ?)", (book, game_key, captured_at)).fetchone()[0] == 1


def nba_rows_for(events: List[Dict], wanted: List[Dict], snap: datetime) -> List[Dict]:
    """The archive's rows for the games we came for, and only those.

    An event must match a wanted game on both teams AND sit within
    NBA_SAME_GAME_HOURS of its scheduled tip (so a rematch is not mistaken for
    it), and the snapshot must predate the event's own commence_time as well
    as ESPN's tip, so an in-play price can never be written as a close.
    """
    keep = []
    for ev in events:
        key = canon_key(f"{ev.get('home_team')}:{ev.get('away_team')}")
        start = parse_utc(ev.get("commence_time"))
        for g in wanted:
            if key != g["key"] or start is None:
                continue
            if abs(start - g["tip"]) > timedelta(hours=NBA_SAME_GAME_HOURS):
                continue
            if snap >= start:
                logger.warning("   %s: archive snapshot %s is not before its commence_time %s; "
                               "refusing it", key, snap.isoformat(), start.isoformat())
                continue
            keep.append(ev)
            break
    return events_to_rows(keep)


def _run_nba(args: argparse.Namespace, window_days: int) -> int:
    if args.game:
        logger.error("--game is NFL-only (it takes our NFL game_id). For the NBA, name the "
                     "window with --since/--until.")
        return 2

    now = datetime.now(timezone.utc)
    if args.since:
        since = parse_utc(args.since)
    else:
        since = now - timedelta(days=window_days)
        logger.info("no --since given, so looking back %d days (from %s). Earlier NBA games are "
                    "not checked: buying them is a purchase, not a repair.",
                    window_days, since.date().isoformat())
    until = min(parse_utc(args.until) or now, now)
    if since is None or since >= until:
        logger.error("empty or unreadable window: since=%s until=%s", args.since, args.until)
        return 2
    since = max(since, parse_utc(ARCHIVE_START))

    try:
        schedule, seen = nba_schedule(since, until)
    except ScheduleUnavailable as exc:
        # Loud on purpose, unattended included. If we cannot read the
        # schedule we do not know whether anything was missed, and "no gaps"
        # would be a claim we had not checked.
        logger.error("could not read the NBA schedule, so gaps were NOT checked: %s", exc)
        return 1

    # Detection only reads, so it reads through a read-only connection: a dry
    # run cannot write by construction, and neither can a run that finds
    # nothing. A writable connection is opened only once there is something
    # to write and the quota guard has agreed to pay for it.
    ro = sqlite3.connect(f"file:{NBA_ODDS_DB}?mode=ro", uri=True, timeout=120)
    try:
        unlisted = nba_unlisted(ro, seen, since, until)
        if unlisted:
            logger.warning("%d game(s) our recorder saw tipping in the window are not in ESPN's "
                           "schedule, so they were not checked for gaps: %s", len(unlisted),
                           "; ".join(f"{k} at {s.isoformat()}" for k, s in unlisted[:5]))
            if not seen:
                logger.error("ESPN listed no NBA game of any kind for days on which our own "
                             "recorder watched one tip. That schedule cannot be trusted, so "
                             "gaps were NOT checked.")
                return 1
        if not schedule:
            logger.info("no gaps: no regular-season, play-in or playoff NBA game tipped between "
                        "%s and %s (%d event(s) of other kinds). Nothing to repair; 0 credits "
                        "spent.", since.isoformat(timespec="minutes"),
                        until.isoformat(timespec="minutes"), len(seen))
            return 0
        missing = nba_gaps(ro, schedule, since)
    finally:
        ro.close()
    return _repair_nba(args, schedule, missing)


def _repair_nba(args: argparse.Namespace, schedule: List[Dict], missing: List[Dict]) -> int:
    if not missing:
        logger.info("no gaps: all %d NBA game(s) that tipped in range have a snapshot within %d "
                    "minutes of tip", len(schedule), NBA_COVERED_WITHIN_MINUTES)
        return 0

    by_tip: Dict[str, List[Dict]] = {}
    for g in missing:
        by_tip.setdefault(g["tip"].isoformat(), []).append(g)
    per_call = credits_per_call(args.markets, args.regions)

    logger.info("%d of %d NBA game(s) have no snapshot within %d minutes of tip, across %d "
                "distinct tip time(s)", len(missing), len(schedule), NBA_COVERED_WITHIN_MINUTES,
                len(by_tip))
    for tip in sorted(by_tip):
        logger.info("   %s  %s", tip, ", ".join(sorted(g["key"] for g in by_tip[tip])))
    logger.info("estimate: %d call(s) x %d credits = %d credits (%s x %s at %dx historical rate)",
                len(by_tip), per_call, per_call * len(by_tip), args.markets, args.regions,
                HISTORICAL_MULTIPLIER)

    plan, estimate, verdict = fit_budget(by_tip, per_call, args.max_credits, args.unattended)
    if plan is None:
        return 1 if verdict == "over_budget" else 0

    if not args.apply:
        logger.info("dry run: nothing called, nothing written, no credits spent. "
                    "Re-run with --apply to repair.")
        return 0

    if args.unattended:
        refusal = unattended_quota_refusal(estimate)
        if refusal:
            logger.warning("%s", refusal)
            return 0

    conn = sqlite3.connect(NBA_ODDS_DB, timeout=120)
    try:
        ensure_nba_snapshot_schema(conn)
        return _write_nba(conn, args, plan)
    finally:
        conn.close()


def _write_nba(conn: sqlite3.Connection, args: argparse.Namespace,
               plan: Dict[str, List[Dict]]) -> int:
    written = skipped_dupe = skipped_late = calls = 0
    remaining: Optional[str] = None
    interrupted = None
    try:
        for tip_iso in sorted(plan):
            tip = datetime.fromisoformat(tip_iso)
            when = (tip - timedelta(minutes=LOOKBACK_MINUTES)).isoformat().replace("+00:00", "Z")
            snap_ts, events, quota = fetch_historical(
                NBA_ODDS_API_KEY, when, args.markets, args.regions, args.bookmakers)
            calls += 1
            remaining = quota.get("remaining")
            logger.info("tip %s: asked for %s, archive returned %s (%d event(s), cost %s, "
                        "remaining %s)", tip_iso, when, snap_ts, len(events), quota.get("last"),
                        remaining)
            snap = parse_utc(snap_ts)
            if snap is None:
                logger.warning("   no usable snapshot timestamp returned; skipping")
                continue
            if snap >= tip:
                logger.warning("   snapshot %s is not before tip %s; refusing to write it",
                               snap_ts, tip_iso)
                skipped_late += 1
                continue

            captured_at = snap.isoformat()
            usable = nba_rows_for(events, plan[tip_iso], snap)
            found = {canon_key(r["game_key"]) for r in usable}
            lost = [g["key"] for g in plan[tip_iso] if g["key"] not in found]
            if lost:
                logger.warning("   the archive had no usable price for: %s", ", ".join(lost))
            rows = [r for r in usable
                    if not already_have_nba(conn, r["sportsbook"], r["game_key"], captured_at)]
            if not rows:
                logger.info("   nothing new to write for this tip")
                skipped_dupe += 1
                continue
            conn.executemany(
                "INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, home_team, "
                "away_team, home_ml, away_ml, ou_line, game_start_time_utc, spread_home, "
                "spread_home_price, spread_away_price, ou_over_price, ou_under_price, provenance) "
                "VALUES (?, 'NBA', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reconstructed')",
                [(captured_at, r["sportsbook"], r["game_key"], r["home_team"], r["away_team"],
                  r["home_ml"], r["away_ml"], r["ou_line"], r["game_start_time_utc"],
                  r["spread_home"], r["spread_home_price"], r["spread_away_price"],
                  r["ou_over_price"], r["ou_under_price"]) for r in rows])
            conn.commit()
            written += len(rows)
            logger.info("   wrote %d reconstructed snapshot row(s)", len(rows))

            if remaining is not None and int(remaining) < QUOTA_FLOOR:
                logger.warning("quota remaining %s is below the floor of %d; stopping here so the "
                               "live recorder is not starved. Re-run to continue.",
                               remaining, QUOTA_FLOOR)
                break
    except RepairUnavailable as exc:
        interrupted = str(exc)
        (logger.warning if args.unattended else logger.error)('%s', interrupted)

    logger.info("SUMMARY sport=nba calls=%d written=%d (reconstructed) dupe=%d late=%d "
                "quota_remaining=%s%s", calls, written, skipped_dupe, skipped_late, remaining,
                "  [stopped early: see above]" if interrupted else "")
    if written:
        logger.info("These prices were read from an archive, not watched. grade_predictions "
                    "carries provenance into closing_provenance; CLV settled on them must say so.")
    if interrupted and not args.unattended:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild missed market snapshots from The Odds API's historical archive.")
    ap.add_argument("--sport", default="nfl", choices=sorted(set(SPORTS) | {"nba"}))
    ap.add_argument("--since", help="Only games kicking off at or after this ISO date.")
    ap.add_argument("--until", help="Only games kicking off at or before this ISO date.")
    ap.add_argument("--game", help="Repair exactly one game_id (NFL only).")
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
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    window_days = UNATTENDED_WINDOW_DAYS if args.unattended else DEFAULT_WINDOW_DAYS
    if args.unattended and ap.get_default("max_credits") == args.max_credits:
        args.max_credits = UNATTENDED_MAX_CREDITS

    if args.sport == "nba":
        return _run_nba(args, window_days)
    return _run_nfl(args, window_days)


if __name__ == "__main__":
    sys.exit(main())
