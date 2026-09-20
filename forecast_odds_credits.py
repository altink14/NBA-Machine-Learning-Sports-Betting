"""
forecast_odds_credits.py
========================
Replays a real season of tip-off times through the live capture ladders and
counts what The Odds API would actually bill us, instead of estimating it.

WHY THIS EXISTS. The handoff carried a figure of "600-900 credits a month"
for the NBA recorder, arrived at by reasoning about a typical night. It was
low by roughly half, and the gap mattered: the free tier is 500 credits a
month and both sports share one key. An estimate that is wrong by 2x is the
difference between "we should upgrade at some point" and "the recorder stops
five days after opening night".

So this stops estimating. It walks a real schedule minute by minute, applies
the same ladder the recorder applies, at the same times the Windows task
actually fires (:14, :29, :44, :59), and counts the captures. A capture is 3
credits: three markets, one region.

WHAT IT CANNOT KNOW. The 2026-27 NBA schedule is not out, so `--nba-shift`
slides last season's tip-off times forward to line opening night up with the
real date. Tip times are what drive the ladder and they barely move year to
year, but the output is a forecast, not a bill. The Odds API also resets on
the billing date rather than the 1st, so a month boundary here is an
approximation of a month boundary there.

WHAT IT FOUND, 2026-09-20 (`--compare`). The current ladder costs about 8,200
credits across an NBA season plus an NFL season -- a worst month of 1,563,
which is 3.1x the free allowance -- and the free 500 runs dry on roughly
25 October, five days after the NBA returns.

Narrowing the closing window from 30 minutes to 15 looks like free money: it
cuts 36% of the captures and, with every scheduled run firing, the closing
lines come out identical (median 1.0 minutes before tip either way). Do not
do it. Run `--compare` with a failure rate: the second capture inside a
30-minute window is not waste, it is the retry. At a 5% missed-run rate the
narrow ladder leaves 5.2% of tips with a stale close against 0.2%, and its
worst close is 89 minutes early rather than 26. A price from 89 minutes out
is not a closing line, and CLV computed against one is not CLV.

The conclusion is that no safe tuning fits inside the free tier, which is
why the $30 / 20,000-credit tier is the fix rather than a nice-to-have.

Usage:
    venv/Scripts/python.exe forecast_odds_credits.py
    venv/Scripts/python.exe forecast_odds_credits.py --month 2026-10
    venv/Scripts/python.exe forecast_odds_credits.py --compare --fail-rate 0.05
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

NBA_SCHEDULE_CSV = os.path.join(REPO_ROOT, "Data", "nba-2025-UTC.csv")
NFL_DB = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")

#: Three markets (h2h, spreads, totals) in one region (us). Both recorders
#: request the same set, so both cost the same per capture.
CREDITS_PER_CAPTURE = 3

#: The free allowance, and the first paid tier.
FREE_MONTHLY = 500
PAID_MONTHLY = 20_000

#: When "BettingBuddy Odds Recorder" actually fires, read off the live logs.
#: Not every 15 minutes in the abstract -- these minutes past the hour.
TICKS = (14, 29, 44, 59)

#: Must match CAPTURE_LADDER in snapshot_odds_api.py and the ladder in
#: src/Sports/odds_recorder.py. If those change and this does not, the
#: forecast is fiction.
LIVE_LADDER = ((30, 10), (180, 90), (1800, 720))
NARROW_LADDER = ((15, 10), (180, 90), (1800, 720))

#: 2025-26 opened 2025-10-21; 2026-27 opens 2026-10-20. 364 days keeps the
#: weekday alignment, which is what actually shapes a slate.
DEFAULT_NBA_SHIFT_DAYS = 364


def nba_tipoffs(shift_days: int) -> list[datetime]:
    """Last season's real tip-off times, slid forward onto this season."""
    shift = timedelta(days=shift_days)
    out: list[datetime] = []
    with open(NBA_SCHEDULE_CSV, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            stamp = (row.get("Date") or "").strip()
            if not stamp:
                continue
            out.append(datetime.strptime(stamp, "%d/%m/%Y %H:%M")
                       .replace(tzinfo=timezone.utc) + shift)
    return sorted(out)


def nfl_kickoffs(season: int) -> list[datetime]:
    """Real kickoff times out of our own archive. Times only -- no scores.

    The 2026 season is the sealed evaluation window for the NFL model. A
    kickoff time is not an outcome, so reading it here is fine, and this
    query deliberately selects nothing else.
    """
    conn = sqlite3.connect(f"file:{NFL_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT date_utc FROM games WHERE season = ? AND date_utc IS NOT NULL",
            (season,)).fetchall()
    finally:
        conn.close()
    return sorted(datetime.fromisoformat(r[0]) for r in rows)


def replay(tips: list[datetime], ladder, fail_rate: float = 0.0,
           rng: random.Random | None = None):
    """Walk the schedule tick by tick, applying the ladder. Returns capture times.

    `fail_rate` drops runs at random, which is how a scheduled task on a home
    machine behaves: sleep, a reboot, a network blip. The ladder's redundancy
    only shows up when something is allowed to go wrong.
    """
    if not tips:
        return []
    rng = rng or random.Random(0)
    start = tips[0] - timedelta(days=2)
    end = tips[-1] + timedelta(hours=6)
    last: datetime | None = None
    captures: list[datetime] = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour <= end:
        for minute in TICKS:
            now = hour.replace(minute=minute)
            if fail_rate and rng.random() < fail_rate:
                continue
            nxt = next((t for t in tips if t > now), None)
            if nxt is None:
                continue
            mins_out = (nxt - now).total_seconds() / 60.0
            since = 1e9 if last is None else (now - last).total_seconds() / 60.0
            for window, cooldown in ladder:
                if mins_out <= window:
                    if since >= cooldown:
                        captures.append(now)
                        last = now
                    break
        hour += timedelta(hours=1)
    return captures


def close_gaps(tips: list[datetime], captures: list[datetime]) -> list[float]:
    """Minutes between each distinct tip-off and the last capture before it.

    This is the number that decides whether we hold a closing line or merely
    a line. `inf` means we hold nothing from before that tip at all.
    """
    gaps = []
    for tip in sorted(set(tips)):
        before = [c for c in captures if c < tip]
        gaps.append((tip - before[-1]).total_seconds() / 60.0 if before else float("inf"))
    return gaps


def by_month(captures: list[datetime]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for c in captures:
        out[c.strftime("%Y-%m")] += 1
    return dict(out)


def report_season(nba: list[datetime], nfl: list[datetime]) -> None:
    nba_caps = replay(nba, LIVE_LADDER)
    nfl_caps = replay(nfl, LIVE_LADDER)
    nba_m, nfl_m = by_month(nba_caps), by_month(nfl_caps)

    print(f"NBA: {len(nba)} games, {len(nba_caps)} captures, "
          f"{len(nba_caps) * CREDITS_PER_CAPTURE} credits")
    print(f"NFL: {len(nfl)} games, {len(nfl_caps)} captures, "
          f"{len(nfl_caps) * CREDITS_PER_CAPTURE} credits")
    print()
    print(f"{'month':9s} {'NBA':>7s} {'NFL':>7s} {'total':>7s}")
    worst = 0
    for month in sorted(set(nba_m) | set(nfl_m)):
        a = nba_m.get(month, 0) * CREDITS_PER_CAPTURE
        b = nfl_m.get(month, 0) * CREDITS_PER_CAPTURE
        worst = max(worst, a + b)
        over = "  over the free 500" if a + b > FREE_MONTHLY else ""
        print(f"{month:9s} {a:7d} {b:7d} {a + b:7d}{over}")
    total = (len(nba_caps) + len(nfl_caps)) * CREDITS_PER_CAPTURE
    print()
    print(f"worst month {worst} credits -- {worst / FREE_MONTHLY:.1f}x the free {FREE_MONTHLY}, "
          f"{100 * worst / PAID_MONTHLY:.0f}% of the paid {PAID_MONTHLY:,}")
    print(f"whole overlap {total} credits")
    for gaps, name in ((close_gaps(nba, nba_caps), "NBA"), (close_gaps(nfl, nfl_caps), "NFL")):
        finite = [g for g in gaps if g != float("inf")]
        print(f"{name} closing lines: median {statistics.median(finite):.1f} min before tip, "
              f"worst {max(finite):.1f}, tips with nothing before them: "
              f"{sum(1 for g in gaps if g == float('inf'))}")


def report_month(nba: list[datetime], nfl: list[datetime], month: str) -> None:
    """Day by day through one month, to find the date the allowance dies."""
    year, mon = (int(x) for x in month.split("-"))
    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    end = (start + timedelta(days=32)).replace(day=1)

    per_day: dict = defaultdict(lambda: [0, 0])
    for i, tips in enumerate((nba, nfl)):
        for c in replay([t for t in tips if start - timedelta(days=2) <= t <= end + timedelta(days=1)],
                        LIVE_LADDER):
            if start <= c < end:
                per_day[c.date()][i] += 1

    print(f"{month}, day by day. Free allowance {FREE_MONTHLY}.\n")
    print(f"{'date':12s} {'NBA':>4s} {'NFL':>4s} {'credits':>8s} {'running':>8s}")
    running, dry = 0, None
    day = start.date()
    while day < end.date():
        a, b = per_day.get(day, [0, 0])
        cost = (a + b) * CREDITS_PER_CAPTURE
        running += cost
        if dry is None and running > FREE_MONTHLY:
            dry = day
        if cost:
            print(f"{str(day):12s} {a:4d} {b:4d} {cost:8d} {running:8d}"
                  + ("   <- the free allowance runs out here" if day == dry else ""))
        day += timedelta(days=1)
    print(f"\n{month} total {running} credits.")
    print(f"The free tier runs dry on {dry}." if dry
          else "Stays inside the free tier.")


def report_compare(nba: list[datetime], fail_rate: float, trials: int) -> None:
    """The narrow window looks cheaper. Show what it costs when a run is missed."""
    window = [t for t in nba if t < nba[0] + timedelta(days=60)]
    print(f"NBA, first 60 days, {trials} trials at a {fail_rate:.0%} missed-run rate.")
    print("A close more than 20 minutes early is not a closing line.\n")
    for label, ladder in (("30-min window (live) ", LIVE_LADDER),
                          ("15-min window (narrow)", NARROW_LADDER)):
        counts, stale, worst = [], [], []
        for seed in range(trials):
            caps = replay(window, ladder, fail_rate, random.Random(seed))
            gaps = close_gaps(window, caps)
            finite = [g for g in gaps if g != float("inf")]
            counts.append(len(caps))
            stale.append(100 * sum(1 for g in gaps if g > 20) / len(gaps))
            worst.append(max(finite) if finite else 0)
        print(f"  {label}  {statistics.mean(counts):6.0f} captures  "
              f"{statistics.mean(counts) * CREDITS_PER_CAPTURE:7.0f} credits  "
              f"{statistics.mean(stale):5.1f}% of tips closed stale  "
              f"worst close {statistics.mean(worst):6.1f} min early")
    if fail_rate == 0:
        print("\nAt a 0% failure rate the two are indistinguishable and the narrow one is "
              "cheaper. Re-run with --fail-rate 0.05 before concluding anything from that.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Forecast Odds API credits by replaying a real schedule through the ladder.")
    ap.add_argument("--month", help="Day-by-day burn for one month, e.g. 2026-10.")
    ap.add_argument("--compare", action="store_true",
                    help="Compare the live ladder against a narrower closing window.")
    ap.add_argument("--fail-rate", type=float, default=0.05,
                    help="Fraction of scheduled runs that do not fire (--compare only).")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--nba-shift", type=int, default=DEFAULT_NBA_SHIFT_DAYS,
                    help="Days to slide the NBA schedule forward.")
    ap.add_argument("--nfl-season", type=int, default=2026)
    args = ap.parse_args()

    nba = nba_tipoffs(args.nba_shift)
    nfl = nfl_kickoffs(args.nfl_season)
    if not nba or not nfl:
        print("No schedule to replay.", file=sys.stderr)
        return 1
    print(f"NBA tip-offs from {NBA_SCHEDULE_CSV} shifted {args.nba_shift} days "
          f"(opening night {nba[0]:%Y-%m-%d})")
    print(f"NFL kickoffs: {args.nfl_season} season, {nfl[0]:%Y-%m-%d} to {nfl[-1]:%Y-%m-%d}")
    print(f"{CREDITS_PER_CAPTURE} credits a capture, runs at :{', :'.join(str(t) for t in TICKS)}\n")

    if args.compare:
        report_compare(nba, args.fail_rate, args.trials)
    elif args.month:
        report_month(nba, nfl, args.month)
    else:
        report_season(nba, nfl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
