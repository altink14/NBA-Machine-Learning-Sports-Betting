"""
nba_clv.py
==========
Closing line value for the NBA prediction log (predictions_log), from our own
odds archive (odds_snapshots + the odds_polls / odds_seen heartbeat).

WHY CLV AND NOT ROI. Whether a pick won says little until hundreds are
settled. Whether the price we logged was better than the market's last price
before tip says something after a few dozen, and it does not depend on who
won. It is the honest route to ever claiming an edge. It is still EVIDENCE,
not profit: a pick can beat the close and lose, and nobody has measured our
return on anything.

WHAT "THE CLOSE" MEANS HERE (method bb-nba-clv-v1). Written down so the
number means one thing:

  Tip        the tip-off logged with the pick (predictions_log.game_start_time_utc,
             frozen before the game). If the game's box score is not on that
             tip's US Eastern date (the grader allows +/-1 day, so a game moved
             a day can still be graded), the game was moved or postponed and
             there is no close for this pick: 'tip_moved'. The odds feed cannot
             be the witness: the recorder writes a row only when a PRICE
             changes, so a re-stamped start time with the same prices leaves
             no trace in odds_snapshots.

  This game  odds rows for the pick's game_key whose start time is within
             TIP_MOVED_TOLERANCE_MINUTES of the logged tip. game_key
             ('Home:Away') is not unique per event: playoff games 1 and 2 share
             a home court. Rows for another start are another event, and while
             one is on the board the heartbeat (keyed on game_key too) cannot
             say which event it saw, so it is not used for that pick (rule 3
             below applies).

  A book's   that book's latest moneyline row for this game captured strictly
  close      before tip (where two rows share a timestamp, the one we watched
             wins over a rebuilt one, the same rule as odds_recorder.seal()).
             The recorders write a row only when a price CHANGES, so the row's
             own time is when the price appeared, not the last time anyone saw
             it. The time that matters is when we last KNEW it still stood:

               1. the heartbeat's last sighting of that book's moneyline, if it
                  is after the row and before tip;
               2. if the market was still listed AFTER tip, the latest
                  successful whole-board poll that covered the book between
                  the row and tip (the recorder would have written a new row
                  had the price changed at that poll);
               3. otherwise the row's own capture time (archives from before
                  the heartbeat, and reconstructed rows, whose captured_at is
                  "when the price was true").

             If that moment is more than CLOSE_WINDOW_MINUTES before tip, the
             book has NO close: 'close_too_early'. 30 minutes is the recorder's
             closing window (snapshot_odds_api.CAPTURE_LADDER); a price from
             further out is not a close and CLV against it is not CLV. Nothing
             is ever estimated or interpolated.

  Pulled     a successful whole-board poll that covered the book ran after we
             last knew the price and before tip, and did not list it; or the
             book's last pre-tip row has no moneyline. The book took the market
             down: 'book_pulled', and the book is left out of the consensus.

  Consensus  the median, over every book with a close, of that book's no-vig
             (Shin, src/Utils/devig.py) probability for our side. Needs at
             least MIN_CONSENSUS_BOOKS books, else 'too_few_books'.

  Same book  the pick's own book (predictions_log.sportsbook), never another
             one standing in for it.

THE FOUR NUMBERS (stored as fractions; the API turns them into points / %):

  clv                  dec(price we logged) / dec(same book's close, our side) - 1
  clv_prob             no-vig P(our side) at the same book's close
                       - no-vig P(our side) from the pair we logged
  clv_consensus_prob   consensus no-vig P(our side) at the close
                       - no-vig P(our side) from the pair we logged
  clv_consensus_price  dec(price we logged) * consensus P(our side) - 1: what the
                       price we took was worth at the consensus closing
                       probability. Vig is in the price we took, so taking the
                       closing price itself scores about minus the margin.

  Positive = we beat the close. A pick needs both logged prices.

WHEN A PICK IS SETTLED, AND ONCE. Only after tip and only once the grader has
found the game's result (so a postponed game waits rather than being priced
against a game that did not happen). A pick
whose book and consensus both priced settles at once. One missing a close for
a reason a repair could still fix (no capture in the window, too few books)
waits REPAIR_GRACE_HOURS after tip, because repair_odds.py can buy a missed
capture back for three days, then settles with its reason. Settling writes the
CLV columns only, in one UPDATE guarded by `clv_status IS NULL`: a settled row
is never re-priced, and no pick column is ever touched (the table's triggers
would refuse it anyway). Reconstructed closes are stored with their provenance
and reported apart from watched ones, never pooled.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.Utils import devig
from src.Utils.odds_board import _Heartbeat, parse_ts

logger = logging.getLogger(__name__)

METHOD = "bb-nba-clv-v1"

#: The recorder's closing window (snapshot_odds_api.CAPTURE_LADDER[0][0]); a
#: test pins the two together. A confirmation older than this is not a close.
CLOSE_WINDOW_MINUTES = 30.0
#: An odds row whose start time is further than this from the logged tip
#: belongs to another event with the same game_key. Genuine reschedules and
#: playoff rematches move days; this only absorbs feed rounding.
TIP_MOVED_TOLERANCE_MINUTES = 30.0
#: Another event with the same game_key captured this close to our tip makes
#: the heartbeat ambiguous for this pick.
SAME_KEY_GUARD_HOURS = 48.0
#: repair_odds.py --unattended works a three-day window, so a missed capture
#: can still arrive as a reconstructed row until then.
REPAIR_GRACE_HOURS = 72.0
#: Fewer books than this is not a market consensus.
MIN_CONSENSUS_BOOKS = 3

#: Reasons a close is missing that a later repair could still change.
_REPAIRABLE = frozenset({"no_capture", "close_too_early", "too_few_books"})

#: Columns this module writes, all additive. The first six predate it
#: (grade_predictions._CLV_COLUMNS, 2026-09-19) and keep their meaning, except
#: closing_minutes_before_tip, which is now measured from the confirmation
#: rather than from the row's first appearance (the honest distance).
CLV_COLUMNS: List[Tuple[str, str]] = [
    ("closing_home_ml", "REAL"),
    ("closing_away_ml", "REAL"),
    ("closing_captured_at", "TEXT"),
    ("closing_minutes_before_tip", "REAL"),
    ("closing_provenance", "TEXT"),
    ("clv", "REAL"),
    ("closing_confirmed_at", "TEXT"),
    ("closing_confirmed_by", "TEXT"),       # heartbeat | poll | captured
    ("clv_prob", "REAL"),
    ("clv_status", "TEXT"),                 # priced | the reason there is no same-book CLV
    ("consensus_close_prob", "REAL"),
    ("consensus_books", "INTEGER"),
    ("consensus_provenance", "TEXT"),       # observed | mixed
    ("clv_consensus_prob", "REAL"),
    ("clv_consensus_price", "REAL"),
    ("clv_consensus_status", "TEXT"),       # priced | the reason there is no consensus CLV
    ("clv_method", "TEXT"),
    ("clv_settled_at", "TEXT"),
]


def ensure_columns(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions_log)")}
    if not have:
        return
    for col, decl in CLV_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE predictions_log ADD COLUMN {col} {decl}")
            logger.info("predictions_log: added column %s", col)


def american_to_decimal(price: Any) -> Optional[float]:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if -100.0 < p < 100.0:
        return None
    return 1.0 + (p / 100.0 if p > 0 else 100.0 / -p)


def no_vig_home(home_ml: Any, away_ml: Any) -> Optional[float]:
    """No-vig P(home) from one book's moneyline pair, or None."""
    dh, da = american_to_decimal(home_ml), american_to_decimal(away_ml)
    if dh is None or da is None:
        return None
    try:
        return float(devig.fair_probs([dh, da])[0])
    except (ValueError, RuntimeError):
        return None


def _clippers(game_key: str) -> str:
    # The prediction path writes "LA Clippers", The Odds API "Los Angeles
    # Clippers" (same rule as grade_predictions and closing_lines).
    return (game_key or "").replace("Los Angeles Clippers", "LA Clippers")


def _provenance(row) -> str:
    try:
        return row["provenance"] or "observed"
    except (IndexError, KeyError):
        return "observed"


def _minutes(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 60.0


def book_close(rows: List[Any], beat: Optional[_Heartbeat], book: str,
               tip: datetime) -> Dict[str, Any]:
    """One book's close for one game. `rows` are that book's rows for THIS
    event, any order. `beat` None = the heartbeat cannot be trusted for this
    game (another event shares its key), so only capture times count."""
    pre = [r for r in rows if (parse_ts(r["captured_at"]) or tip) < tip]
    if not pre:
        return {"status": "no_capture"}
    # Watched beats rebuilt at the same instant: observed sorts last.
    pre.sort(key=lambda r: (parse_ts(r["captured_at"]),
                            1 if _provenance(r) == "observed" else 0, r["id"]))
    last = pre[-1]
    recorded = parse_ts(last["captured_at"])
    out: Dict[str, Any] = {"book": book, "row_captured_at": last["captured_at"],
                           "provenance": _provenance(last)}
    if last["home_ml"] is None or last["away_ml"] is None:
        out["status"] = "book_pulled"
        out["pulled_at"] = last["captured_at"]
        return out

    seen, covering = None, []
    if beat is not None:
        keys = {last["game_key"], _clippers(last["game_key"]),
                last["game_key"].replace("LA Clippers", "Los Angeles Clippers")}
        sightings = [beat.seen[(k, book, "ml")] for k in keys if (k, book, "ml") in beat.seen]
        seen = max(sightings) if sightings else None
        covering = [p[0] for p in beat.polls if beat._covers(p, book, "ml")]
    if seen is None or seen < tip:
        known = max(recorded, seen) if seen else recorded
        missed = [t for t in covering if known < t < tip]
        if missed:
            out["status"] = "book_pulled"
            out["pulled_at"] = min(missed).isoformat()
            return out

    if seen is not None and recorded <= seen < tip:
        confirmed, how = seen, "heartbeat"
    elif seen is not None and seen >= tip:
        # Listed after tip, so the market survived to tip-off. Every covering
        # poll in between would have written a row had the price moved.
        polls = [t for t in covering if recorded <= t < tip]
        confirmed, how = (max(polls), "poll") if polls else (recorded, "captured")
    else:
        confirmed, how = recorded, "captured"

    gap = _minutes(tip, confirmed)
    out.update({"home_ml": last["home_ml"], "away_ml": last["away_ml"],
                "confirmed_at": confirmed.isoformat(), "confirmed_by": how,
                "minutes_before_tip": round(gap, 2)})
    out["status"] = "ok" if gap <= CLOSE_WINDOW_MINUTES else "close_too_early"
    return out


def evaluate(pick: Any, game_rows: List[Any], beat: _Heartbeat, now: datetime,
             played_on_date: Optional[bool]) -> Dict[str, Any]:
    """Everything about one pick's CLV, and whether it is final.

    `game_rows`: every NBA odds row for the pick's game_key (either Clippers
    spelling). `played_on_date`: whether the archive holds this game's box
    score on the logged tip's Eastern date (None = could not look).
    Returns {"final": bool, "reason": str, "values": {column: value}}.
    Pure: reads nothing but its arguments, so the tests drive it directly.
    """
    tip = parse_ts(pick["game_start_time_utc"])
    if tip is None or now < tip:
        return {"final": False, "reason": "not_started", "values": {}}
    if not pick["actual_winner"]:
        # Not graded: either the box score is not in yet, or the game was not
        # played near that date. Either way there is nothing to price yet.
        return {"final": False, "reason": "awaiting_result", "values": {}}
    if played_on_date is None:
        return {"final": False, "reason": "played_unknown", "values": {}}

    def settle(book_status: str, cons_status: str, **vals) -> Dict[str, Any]:
        return {"final": True, "reason": book_status,
                "values": {"clv_status": book_status, "clv_consensus_status": cons_status,
                           "clv_method": f"{METHOD}/{devig.ACTIVE_METHOD}",
                           "clv_settled_at": now.replace(microsecond=0).isoformat(), **vals}}

    home, away, winner = pick["home_team"], pick["away_team"], pick["predicted_winner"]
    if winner not in (home, away):
        return settle("no_pick", "no_pick")
    on_home = winner == home
    taken = pick["home_ml"] if on_home else pick["away_ml"]
    p_taken_home = no_vig_home(pick["home_ml"], pick["away_ml"])
    dec_taken = american_to_decimal(taken)
    if p_taken_home is None or dec_taken is None:
        return settle("no_logged_price", "no_logged_price")
    p_taken = p_taken_home if on_home else 1.0 - p_taken_home

    # Graded (the grader allows +/-1 day) but not on the logged date: the game
    # was moved, and the market we logged against is not the one that closed.
    if not played_on_date:
        return settle("tip_moved", "tip_moved")

    this_game, other_event_near = [], False
    for r in game_rows:
        start, cap = parse_ts(r["game_start_time_utc"]), parse_ts(r["captured_at"])
        if start is None or cap is None:
            continue  # legacy rows with no start time cannot be placed in a game
        if abs(_minutes(start, tip)) <= TIP_MOVED_TOLERANCE_MINUTES:
            this_game.append(r)
        elif abs(_minutes(cap, tip)) <= SAME_KEY_GUARD_HOURS * 60:
            other_event_near = True
    usable_beat = None if other_event_near else beat

    by_book: Dict[str, List[Any]] = {}
    for r in this_game:
        by_book.setdefault(r["sportsbook"], []).append(r)
    closes = {b: book_close(rows, usable_beat, b, tip) for b, rows in by_book.items()}

    vals: Dict[str, Any] = {}
    mine = closes.get(pick["sportsbook"], {"status": "no_capture"})
    book_status = mine["status"]
    if "row_captured_at" in mine:
        vals["closing_captured_at"] = mine["row_captured_at"]
        vals["closing_provenance"] = mine["provenance"]
    if "confirmed_at" in mine:
        # Recorded even when too early, so the distance travels with the NULL.
        vals.update({"closing_home_ml": mine["home_ml"], "closing_away_ml": mine["away_ml"],
                     "closing_confirmed_at": mine["confirmed_at"],
                     "closing_confirmed_by": mine["confirmed_by"],
                     "closing_minutes_before_tip": mine["minutes_before_tip"]})
    if book_status == "ok":
        close_side = mine["home_ml"] if on_home else mine["away_ml"]
        dec_close = american_to_decimal(close_side)
        p_close_home = no_vig_home(mine["home_ml"], mine["away_ml"])
        if dec_close is None or p_close_home is None:
            book_status = "bad_close_price"
        else:
            p_close = p_close_home if on_home else 1.0 - p_close_home
            vals["clv"] = dec_taken / dec_close - 1.0
            vals["clv_prob"] = p_close - p_taken
            book_status = "priced"

    fair = []
    provs = set()
    for c in closes.values():
        if c["status"] != "ok":
            continue
        ph = no_vig_home(c["home_ml"], c["away_ml"])
        if ph is None:
            continue
        fair.append(ph if on_home else 1.0 - ph)
        provs.add(c["provenance"])
    vals["consensus_books"] = len(fair)
    if len(fair) >= MIN_CONSENSUS_BOOKS:
        p_cons = float(median(fair))
        vals.update({"consensus_close_prob": p_cons,
                     "consensus_provenance": "observed" if provs == {"observed"} else "mixed",
                     "clv_consensus_prob": p_cons - p_taken,
                     "clv_consensus_price": dec_taken * p_cons - 1.0})
        cons_status = "priced"
    else:
        cons_status = "too_few_books"

    waiting = [s for s in (book_status, cons_status) if s in _REPAIRABLE]
    if waiting and now < tip + timedelta(hours=REPAIR_GRACE_HOURS):
        return {"final": False, "reason": waiting[0], "values": {}}
    return settle(book_status, cons_status, **vals)


def _game_rows(conn: sqlite3.Connection, game_key: str) -> List[Any]:
    return conn.execute(
        "SELECT * FROM odds_snapshots WHERE sport = 'NBA' AND "
        "REPLACE(game_key, 'Los Angeles Clippers', 'LA Clippers') = ?",
        (_clippers(game_key),)).fetchall()


def price_all(conn: sqlite3.Connection, played: Callable[[Any], Optional[bool]],
              now: Optional[datetime] = None) -> Dict[str, int]:
    """Settle every pick that can be settled. Returns counts by outcome.

    `played(pick)` says whether the archive holds the game's box score on the
    logged tip's Eastern date (None = could not look). Writes only the CLV
    columns of rows whose clv_status is still NULL, in one transaction.
    """
    now = now or datetime.now(timezone.utc)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)
    counts: Dict[str, int] = {}
    beat = _Heartbeat(conn, "NBA")
    picks = conn.execute(
        "SELECT * FROM predictions_log WHERE clv_status IS NULL AND sport = 'NBA'").fetchall()
    with conn:
        for pick in picks:
            if pick["clv"] is not None:
                # Priced by the pre-2026-09-28 code, which had no status. Left
                # exactly as it is: a settled figure is never re-priced.
                counts["legacy_priced"] = counts.get("legacy_priced", 0) + 1
                continue
            played_on_date = played(pick) if pick["actual_winner"] else None
            res = evaluate(pick, _game_rows(conn, pick["game_key"]), beat, now, played_on_date)
            key = ("settled_" if res["final"] else "waiting_") + res["reason"]
            counts[key] = counts.get(key, 0) + 1
            if not res["final"]:
                continue
            cols = [c for c, _ in CLV_COLUMNS if c in res["values"]]
            conn.execute(
                f"UPDATE predictions_log SET {', '.join(f'{c} = ?' for c in cols)} "
                "WHERE id = ? AND clv_status IS NULL",
                [res["values"][c] for c in cols] + [pick["id"]])
    logger.info("closing line value (%s): %s", METHOD, counts or "nothing to price")
    return counts


# --- Read side: the track record's CLV section ------------------------------

def _t95(n: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except Exception:  # pragma: no cover - scipy is installed on both machines
        return 1.96


def _mean_ci(xs: List[float]) -> Dict[str, Any]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None}
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": m, "ci95": None}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    half = _t95(n) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "ci95": [m - half, m + half]}


def _wilson(k: int, n: int) -> Optional[List[float]]:
    if n == 0:
        return None
    z = 1.959964
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def _block(values: List[float], scale: float, digits: int) -> Dict[str, Any]:
    """Mean with a t-interval, and the share above zero with a Wilson interval."""
    mc = _mean_ci(values)
    n = mc["n"]
    beat = sum(1 for v in values if v > 1e-12)
    ties = sum(1 for v in values if abs(v) <= 1e-12)
    share_ci = _wilson(beat, n)
    r = lambda v: None if v is None else round(v * scale, digits)  # noqa: E731
    return {
        "n": n,
        "mean": r(mc["mean"]),
        "ci95": [r(mc["ci95"][0]), r(mc["ci95"][1])] if mc["ci95"] else None,
        "beat_close": beat,
        "ties": ties,
        "beat_share_pct": round(100.0 * beat / n, 1) if n else None,
        "beat_share_ci95": [round(100 * share_ci[0], 1), round(100 * share_ci[1], 1)]
        if share_ci else None,
    }


def summary(conn: sqlite3.Connection, now: Optional[datetime] = None,
            exclude_model: Optional[str] = None) -> Dict[str, Any]:
    """The CLV section of /track-record. Never writes.

    Watched closes and reconstructed ones are reported apart, never pooled.
    Empty (n = 0, means None) until picks have settled; the columns may not
    exist yet on a copy that has never been priced.
    """
    now = now or datetime.now(timezone.utc)
    conn.row_factory = sqlite3.Row
    definition = {
        "method": METHOD,
        "devig": devig.ACTIVE_METHOD,
        "close_window_minutes": CLOSE_WINDOW_MINUTES,
        "tip_moved_tolerance_minutes": TIP_MOVED_TOLERANCE_MINUTES,
        "same_key_guard_hours": SAME_KEY_GUARD_HOURS,
        "repair_grace_hours": REPAIR_GRACE_HOURS,
        "min_consensus_books": MIN_CONSENSUS_BOOKS,
    }
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions_log)")}
    base = {"definition": definition, "logged_started": 0, "settled": 0, "waiting": 0,
            "no_clv_reasons": {}, "consensus": _block([], 100, 2),
            "same_book": {"prob": _block([], 100, 2), "price": _block([], 100, 2)},
            "reconstructed": {"consensus_n": 0, "same_book_n": 0},
            "median_minutes_before_tip": None, "generated_at": now.isoformat()}
    if not have or "clv_status" not in have:
        return base
    where = "sport = 'NBA'"
    params: List[Any] = []
    if exclude_model:
        where += " AND (model IS NULL OR model != ?)"
        params.append(exclude_model)
    rows = conn.execute(f"SELECT * FROM predictions_log WHERE {where}", params).fetchall()
    started = [r for r in rows if (parse_ts(r["game_start_time_utc"]) or now) < now]
    settled = [r for r in started if r["clv_status"] is not None]

    cons_obs, cons_rec = [], 0
    book_prob, book_price, book_rec, gaps = [], [], 0, []
    reasons: Dict[str, int] = {}
    for r in settled:
        if r["clv_consensus_status"] == "priced":
            if r["consensus_provenance"] == "observed":
                cons_obs.append(r["clv_consensus_prob"])
            else:
                cons_rec += 1
        else:
            k = f"consensus:{r['clv_consensus_status']}"
            reasons[k] = reasons.get(k, 0) + 1
        if r["clv_status"] == "priced":
            if (r["closing_provenance"] or "observed") == "observed":
                book_prob.append(r["clv_prob"])
                book_price.append(r["clv"])
                if r["closing_minutes_before_tip"] is not None:
                    gaps.append(r["closing_minutes_before_tip"])
            else:
                book_rec += 1
        else:
            k = f"same_book:{r['clv_status']}"
            reasons[k] = reasons.get(k, 0) + 1

    base.update({
        "logged_started": len(started),
        "settled": len(settled),
        "waiting": len(started) - len(settled),
        "no_clv_reasons": reasons,
        # Points of no-vig probability; the headline figure.
        "consensus": _block(cons_obs, 100, 2),
        "same_book": {"prob": _block(book_prob, 100, 2),   # points
                      "price": _block(book_price, 100, 2)},  # percent of price
        "reconstructed": {"consensus_n": cons_rec, "same_book_n": book_rec},
        "median_minutes_before_tip": round(median(gaps), 1) if gaps else None,
    })
    return base
