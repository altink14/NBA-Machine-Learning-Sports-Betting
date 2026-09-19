"""
leakage_tests.py (NFL)
======================
The five controls from section 7 of
`docs/sports/nfl/MODEL_PREREGISTRATION_v1.md`. These must pass before any
model is fitted, and they run again before the single sealed evaluation.

Leakage is the failure mode that does not look like one. A model that has seen
the future does not crash, does not warn, and does not produce anything
obviously wrong. It produces a number that is too good, and a team that wants
to believe it. This project has already paid for that lesson once, with a
model-versus-market comparison drawn from seasons inside the training window.

So these tests are adversarial: each one tries to PROVE the pipeline leaks, and
passes only when it cannot.

    venv/Scripts/python.exe src/Sports/nfl/leakage_tests.py

Exit code 0 when every control holds, 1 otherwise.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.nfl.features import (  # noqa: E402
    build_frame, MODEL_COLUMNS, ODDS_DENYLIST, SEALED_SEASONS,
    TRAIN_SEASONS, TUNE_SEASONS, ELO_START, ELO_SEASON_REGRESSION, TeamState,
)

FAILS: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{name}: {detail}")


def control_1_as_of_date(rows: List[Dict[str, Any]]) -> None:
    """A row must be identical when the rest of history has not happened yet.

    This is the strongest statement available: rebuild the frame with every
    game from the target kickoff onward deleted, and demand that the target
    row comes out byte-identical. If any feature peeked forward, even by one
    game, the two runs disagree.
    """
    print("\n=== Control 1: features use only games that had already kicked off ===")
    random.seed(20260919)
    sample = random.sample([r for r in rows if r["season"] >= "2005"], 12)
    mismatches = []
    for target in sample:
        # The cutoff must INCLUDE the target itself, or there is nothing to
        # compare. Appending "Z" makes the string compare inclusive of the
        # exact timestamp without changing any real value. Games that kick off
        # at the same moment come along too, which is correct rather than a
        # loophole: they involve different teams, and control 1b proves it.
        truncated, _ = build_frame(max_kickoff=target["kickoff_utc"] + "Z",
                                   seasons=(target["season"],))
        got = next((r for r in truncated if r["game_id"] == target["game_id"]), None)
        if got is None:
            mismatches.append(f"{target['game_id']} vanished when truncated")
            continue
        for col in MODEL_COLUMNS:
            a, b = target[col], got[col]
            if a is None and b is None:
                continue
            if a is None or b is None or (isinstance(a, float) and abs(a - b) > 1e-9) or \
               (not isinstance(a, float) and a != b):
                mismatches.append(f"{target['game_id']}.{col}: full={a} truncated={b}")
                break
    check(f"{len(sample)} sampled rows rebuild identically from truncated history",
          not mismatches, "; ".join(mismatches[:3]))

    # Control 1b: simultaneous kickoffs cannot contaminate each other, because
    # a team plays one game at a time. Asserted rather than assumed.
    same_time = {}
    for r in rows:
        same_time.setdefault(r["kickoff_utc"], []).append(r)
    clashes = []
    for k, group in same_time.items():
        if len(group) < 2:
            continue
        teams = [t for r in group for t in (r["home_team"], r["away_team"])]
        if len(teams) != len(set(teams)):
            clashes.append(k)
    check("no team appears twice at one kickoff time", not clashes,
          f"{len(clashes)} clashing timestamps")


def control_2_no_future_rows(rows: List[Dict[str, Any]]) -> None:
    """A team's very first game can have no history behind it."""
    print("\n=== Control 2: no row carries history it could not have had ===")
    seen: set = set()
    first_rows = []
    for r in sorted(rows, key=lambda x: x["kickoff_utc"]):
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            if t not in seen:
                seen.add(t)
                first_rows.append((r, side))
    bad = []
    for r, side in first_rows:
        if r["season"] != "1999":
            continue    # only 1999 debuts are genuinely history-free
        if r[f"elo_{side}"] != ELO_START:
            bad.append(f"{r[f'{side}_team']} debut elo {r[f'elo_{side}']}")
        if r[f"epa_off_{side}_8"] is not None:
            bad.append(f"{r[f'{side}_team']} debut already has rolling EPA")
        if r[f"games_played_{side}"] != 0:
            bad.append(f"{r[f'{side}_team']} debut games_played {r[f'games_played_{side}']}")
    check("teams start their first archived game with no history",
          not bad, "; ".join(bad[:4]))

    # Week 1 of any season must show a zeroed record, never last year's.
    wk1 = [r for r in rows if r["week"] == 1 and r["season_type"] == "REG"]
    carried = [r["game_id"] for r in wk1
               if r["games_played_home"] != 0 or r["games_played_away"] != 0]
    check("week 1 rows carry no win-loss record from the previous season",
          not carried, f"{len(carried)} rows do")


def control_3_season_boundary(rows: List[Dict[str, Any]]) -> None:
    """Ratings must regress toward the mean between seasons, not carry whole.

    The earlier version of this test compared the rating ENTERING a team's
    last game of one season with the rating entering its first of the next,
    and failed on 75 of 693 team-seasons. That was the test being wrong, not
    the pipeline: the rating in a row is the PRE-game value, so the last
    game's own result had not been applied yet, and the comparison was
    meaningless. Replaced with two things that are actually provable.
    """
    print("\n=== Control 3: season boundaries regress rather than leak ===")

    # (a) The mechanism itself, as a unit. Exact arithmetic, no data.
    unit_ok = []
    for start in (1700.0, 1300.0, 1500.0, 1800.0):
        st = TeamState()
        st.elo = start
        st.new_season()
        expected = ELO_START + (start - ELO_START) * (1.0 - ELO_SEASON_REGRESSION)
        unit_ok.append(abs(st.elo - expected) < 1e-9 and abs(st.elo - ELO_START) <= abs(start - ELO_START))
    check("new_season() regresses ratings toward the mean by exactly the stated fraction",
          all(unit_ok), f"{sum(unit_ok)}/{len(unit_ok)} cases")

    # (b) The consequence, distributionally: week 1 ratings must be more
    # tightly packed than late-season ones, because every team was pulled
    # toward 1500 in the interim.
    def spread(sample: List[float]) -> float:
        if len(sample) < 2:
            return 0.0
        m = sum(sample) / len(sample)
        return (sum((x - m) ** 2 for x in sample) / (len(sample) - 1)) ** 0.5

    wk1 = [r[f"elo_{s}"] for r in rows if r["week"] == 1 and r["season_type"] == "REG"
           for s in ("home", "away")]
    late = [r[f"elo_{s}"] for r in rows if r["week"] and r["week"] >= 15
            and r["season_type"] == "REG" for s in ("home", "away")]
    sd1, sd15 = spread(wk1), spread(late)
    check("week 1 ratings are more tightly packed than week 15+ ratings",
          sd1 < sd15, f"sd {sd1:.1f} vs {sd15:.1f}")

    # (c) And no team may enter week 1 with a record.
    carried = [r["game_id"] for r in rows if r["week"] == 1 and r["season_type"] == "REG"
               and (r["winpct_home"] is not None or r["winpct_away"] is not None)]
    check("no team enters week 1 carrying a win percentage", not carried,
          f"{len(carried)} rows do")


def control_4_sealed_window() -> None:
    """The sealed window must be unreachable without the explicit flag."""
    print("\n=== Control 4: the sealed window is physically guarded ===")
    default_rows, _ = build_frame()
    present = sorted({r["season"] for r in default_rows} & set(SEALED_SEASONS))
    check("a default build contains no sealed season", not present, f"found {present}")

    refused = False
    try:
        build_frame(seasons=(SEALED_SEASONS[0],))
    except PermissionError:
        refused = True
    check("asking for a sealed season without the flag raises", refused)

    live_present = sorted({r["season"] for r in default_rows} & {"2026"})
    check("a default build contains no live season", not live_present, f"found {live_present}")

    allowed = sorted({r["season"] for r in default_rows})
    expected = sorted(set(TRAIN_SEASONS) | set(TUNE_SEASONS))
    check("a default build contains exactly train plus tune",
          allowed == expected, f"{len(allowed)} seasons, {allowed[0]} to {allowed[-1]}")


def control_5_no_odds() -> None:
    """No feature may be derived from a price. Checked by name and by value."""
    print("\n=== Control 5: no odds anywhere in the feature path ===")
    hits = [c for c in MODEL_COLUMNS
            if any(bad in c.lower() for bad in ODDS_DENYLIST)]
    check("no model column name suggests a market quantity", not hits, f"{hits}")

    # And by source: the feature builder must not read the odds tables at all.
    src = open(os.path.join(REPO_ROOT, "src", "Sports", "nfl", "features.py"),
               encoding="utf-8").read().lower()
    touched = [t for t in ("market_lines", "market_line_snapshots", "odds_snapshots")
               if t in src.split("odds_denylist")[0]]
    check("the feature builder never queries an odds table", not touched, f"{touched}")


def main() -> int:
    print("Leakage controls for NFL model v1")
    print("Pre-registration sealed at df7f55696c8b606cfd35435279f0db4d1f703224")
    rows, _ = build_frame()
    print(f"\nframe: {len(rows)} rows, {len(MODEL_COLUMNS)} model features")

    control_1_as_of_date(rows)
    control_2_no_future_rows(rows)
    control_3_season_boundary(rows)
    control_4_sealed_window()
    control_5_no_odds()

    print("\n" + "=" * 62)
    if FAILS:
        print(f"LEAKAGE CONTROLS FAILED: {len(FAILS)}")
        for f in FAILS:
            print(f"  - {f}")
        print("\nDo not train. A model fitted on a leaking frame produces a number")
        print("that is too good and a story about why it is real.")
        return 1
    print("ALL LEAKAGE CONTROLS HOLD. The frame is safe to fit on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
