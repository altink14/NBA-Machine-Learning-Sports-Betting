"""
ledger.py (cross-sport)
=======================
The prediction ledger. One table, every sport, append-only, and the rule that
gives the whole product its argument:

    NO PREDICTION IS EVER DISPLAYED THAT WAS NOT WRITTEN DOWN BEFORE THE EVENT
    STARTED.

That is not a convention here. It is a CHECK constraint. A row whose
`created_at` is not strictly earlier than `event_start_utc` cannot be inserted
at all; SQLite rejects it. Backfilling a prediction is therefore not a thing a
careless script can do by accident, or a motivated person can do on purpose
without leaving the schema visibly altered in version control.

WHAT MAY CHANGE AFTER INSERT. Only the grading fields: the result, when it was
graded, what graded it, and the closing-line columns that make closing line
value computable. Every prediction field is frozen by a trigger. A correction
is a NEW row that references the old one through `supersedes_id`; nothing is
ever edited into a better story.

WHY CLV IS HERE FROM THE FIRST ROW. Return on investment needs hundreds of
graded bets before it means anything. Closing line value means something after
a few dozen, and it is the metric that says whether we were on the right side
of a price. `price_taken` records what we could actually have got at the moment
we predicted, not a backfilled number, which is why the odds recorder had to
exist before this did.

The NBA's older `predictions_log` table is untouched and still has zero rows;
this is the cross-sport replacement, and the NBA moves onto it separately.

Usage:
    from src.Sports.ledger import ensure_ledger, write_prediction, grade_pending
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEDGER_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")

# --------------------------------------------------------------------------
# SEALED COMPETITIONS
#
# A sealed competition is one whose outcomes a pre-registration forbids us to
# look at until its single evaluation. Right now that is the 2026 NFL season
# (MODEL_PREREGISTRATION_v2.md, sealed at commit a88f762).
#
# Predictions for a sealed competition are still written and still graded --
# that is the whole forward test, and the results must be in the database when
# the evaluation finally runs. What must not happen is any RUNNING TOTAL of
# those outcomes reaching a human before that day, because a win-loss tally is
# exactly the kind of aggregate that voided v1. MODEL_V1_VOIDED.md rejects the
# "it was only an aggregate" defence in as many words.
#
# So the reporting functions below collapse win/loss/push into a single
# `settled` count for sealed competitions. The count of finished games carries
# no information the schedule does not already give away; the split does.
#
# `reveal_sealed_outcomes=True` exists for exactly one caller that does not
# exist yet: the single evaluation, after the season ends. Passing it before
# then is the breach, and it has to be typed on purpose.
# --------------------------------------------------------------------------
SEALED_COMPETITION_PREFIXES = ("nfl-2026-",)


def is_sealed(competition_id: Optional[str]) -> bool:
    """True if this competition's outcomes are under seal today."""
    if not competition_id:
        return False
    return competition_id.startswith(SEALED_COMPETITION_PREFIXES)


SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sport           TEXT NOT NULL,
    league          TEXT NOT NULL,
    competition_id  TEXT,
    game_id         TEXT NOT NULL,
    market_type     TEXT NOT NULL,      -- moneyline | spread | total
    side            TEXT NOT NULL,      -- the pick, e.g. 'home' | 'away' | 'over'
    line            REAL,
    price_taken     INTEGER,            -- American odds we could actually have had
    book            TEXT,
    model_version   TEXT NOT NULL,
    model_seal      TEXT,               -- pre-registration commit hash
    model_prob      REAL NOT NULL,      -- our probability for `side`
    fair_prob       REAL,               -- market's no-vig probability, when known
    ev              REAL,               -- expected value per unit staked
    kelly_fraction  REAL,
    recommended_unit REAL,

    created_at      TEXT NOT NULL,      -- when WE wrote it. Immutable.
    event_start_utc TEXT NOT NULL,
    lock_time_utc   TEXT,

    closing_line    REAL,
    closing_price   INTEGER,
    clv             REAL,

    result          TEXT NOT NULL DEFAULT 'pending',  -- pending|win|loss|push|void
    graded_at       TEXT,
    grading_source  TEXT,
    is_shadow       INTEGER NOT NULL DEFAULT 1,       -- 1 until a model has passed its gates
    supersedes_id   INTEGER,
    notes           TEXT,

    -- The constraint the whole product rests on. Not a comment, not a code
    -- path: the database refuses the row.
    CHECK (created_at < event_start_utc),
    CHECK (result IN ('pending', 'win', 'loss', 'push', 'void')),
    CHECK (model_prob >= 0.0 AND model_prob <= 1.0),
    UNIQUE (game_id, market_type, side, model_version)
);
CREATE INDEX IF NOT EXISTS idx_ledger_sport   ON ledger(sport, created_at);
CREATE INDEX IF NOT EXISTS idx_ledger_pending ON ledger(result, event_start_utc);
CREATE INDEX IF NOT EXISTS idx_ledger_game    ON ledger(game_id);

-- Prediction fields are frozen. Only grading and closing-line columns move.
CREATE TRIGGER IF NOT EXISTS ledger_prediction_is_immutable
BEFORE UPDATE ON ledger
WHEN OLD.sport != NEW.sport OR OLD.game_id != NEW.game_id
  OR OLD.market_type != NEW.market_type OR OLD.side != NEW.side
  OR OLD.model_version != NEW.model_version OR OLD.model_prob != NEW.model_prob
  OR OLD.created_at != NEW.created_at OR OLD.event_start_utc != NEW.event_start_utc
  OR IFNULL(OLD.price_taken, -99999) != IFNULL(NEW.price_taken, -99999)
  OR IFNULL(OLD.line, -99999) != IFNULL(NEW.line, -99999)
BEGIN
    SELECT RAISE(ABORT, 'ledger predictions are immutable; write a new row with supersedes_id');
END;

-- A graded row is settled. Re-grading it differently is how records get
-- quietly improved, so it is refused.
CREATE TRIGGER IF NOT EXISTS ledger_no_regrade
BEFORE UPDATE ON ledger
WHEN OLD.result != 'pending' AND NEW.result != OLD.result
BEGIN
    SELECT RAISE(ABORT, 'a graded ledger row cannot be regraded');
END;

CREATE TRIGGER IF NOT EXISTS ledger_no_delete
BEFORE DELETE ON ledger
BEGIN
    SELECT RAISE(ABORT, 'the ledger is append-only');
END;
"""


# Added after the first rows were written. Additive only.
_LEDGER_ADDED_COLUMNS = [
    ("closing_book", "TEXT"),
    ("closing_provenance", "TEXT"),
]


def ensure_ledger(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(ledger)")}
    for col, decl in _LEDGER_ADDED_COLUMNS:
        if have and col not in have:
            conn.execute(f"ALTER TABLE ledger ADD COLUMN {col} {decl}")
    conn.commit()


def american_to_decimal(price: Optional[int]) -> Optional[float]:
    if price is None:
        return None
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / -price)


#: Which closing price settles which side of which market.
_SIDE_COLUMN = {
    ("moneyline", "home"): "price_home", ("moneyline", "away"): "price_away",
    ("spread", "home"): "price_home", ("spread", "away"): "price_away",
    ("total", "over"): "price_over", ("total", "under"): "price_under",
}


def apply_clv(conn: sqlite3.Connection,
              closes: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[str, int]:
    """Attach closing line value to every prediction that can carry one.

    WHAT CLV IS, AND WHY IT IS HERE RATHER THAN ROI. Return on investment needs
    hundreds of settled bets before it says anything; closing line value says
    something after a few dozen, because it asks a question that does not
    depend on who won: did we take a better price than the market's last one?
    A model that consistently beats the close is finding something. A model
    that does not is, at best, agreeing with the market slowly.

    THE ARITHMETIC. Both prices become decimal odds and

        clv = (decimal we took / decimal at the close) - 1

    so +0.02 means the price we got was 2% better than the close on the same
    side of the same market. It is deliberately NOT a profit figure and must
    never be presented as one: beating the close is evidence about price, not
    money, and this project does not claim ROI.

    PROVENANCE TRAVELS WITH IT. Each row records which book closed it and
    whether that closing price was `observed`, `reconstructed` or
    `third_party`. CLV computed against a price we watched and CLV computed
    against one we bought back from an archive are different claims, and the
    row now says which it is instead of leaving it to whoever reads the
    average.

    ONLY AFTER KICKOFF. A prediction whose game has not started is skipped.
    `is_closing` alone is not proof of a close: the nflverse schedule file
    carries lines for scheduled games and our ingest marks them closing, so
    the flag can be true days before there is anything to close. Time is the
    check that cannot be wrong.

    WHICH BOOK CLOSES IT. `closes` maps (game_id, market_type) to every book
    that closed that market. We settle against the BEST close for the side we
    backed -- the longest price still available at the bell. That is the
    conservative choice, not a generous one: a better closing price makes our
    CLV smaller, so picking a worse book would flatter the number for free.

    Rows already carrying a clv are left alone: like grading, this settles
    once.
    """
    counts = {"priced": 0, "no_close": 0, "no_price_taken": 0,
              "not_started": 0, "already": 0}
    now = datetime.now(timezone.utc).isoformat()
    rows = conn.execute(
        "SELECT id, game_id, market_type, side, price_taken, clv, event_start_utc FROM ledger"
    ).fetchall()

    for rid, gid, market, side, taken, existing, starts in rows:
        # A game that has not started has no close, whatever a table says. The
        # upstream schedule file publishes lines for scheduled games and our
        # ingest flags them is_closing=1, so a row that looks like a close is
        # sitting there days early. Settling against it would invent closing
        # line value out of a price the market has not finished moving.
        if starts and starts > now:
            counts["not_started"] += 1
            continue
        if existing is not None:
            counts["already"] += 1
            continue
        if taken is None:
            # No price when we predicted means no CLV, ever. We will not
            # substitute a later price and call it the one we could have had.
            counts["no_price_taken"] += 1
            continue
        candidates = closes.get((gid, market)) or []
        col = _SIDE_COLUMN.get((market, side))
        dec_taken = american_to_decimal(taken)
        best = None
        if col and dec_taken is not None:
            for cand in candidates:
                dec = american_to_decimal(cand.get(col))
                if dec is not None and dec > 1.0 and (best is None or dec > best[0]):
                    best = (dec, cand)
        if best is None:
            counts["no_close"] += 1
            continue
        dec_close, close = best
        closing_price = close.get(col)
        conn.execute(
            "UPDATE ledger SET closing_line = ?, closing_price = ?, clv = ?, "
            "closing_book = ?, closing_provenance = ? WHERE id = ?",
            (close.get("line"), closing_price, (dec_taken / dec_close) - 1.0,
             close.get("book"), close.get("provenance"), rid))
        counts["priced"] += 1

    conn.commit()
    return counts


def write_prediction(conn: sqlite3.Connection, *, sport: str, league: str, game_id: str,
                     market_type: str, side: str, model_version: str, model_prob: float,
                     event_start_utc: str, competition_id: Optional[str] = None,
                     line: Optional[float] = None, price_taken: Optional[int] = None,
                     book: Optional[str] = None, model_seal: Optional[str] = None,
                     fair_prob: Optional[float] = None, ev: Optional[float] = None,
                     kelly_fraction: Optional[float] = None,
                     recommended_unit: Optional[float] = None,
                     lock_time_utc: Optional[str] = None, is_shadow: bool = True,
                     supersedes_id: Optional[int] = None,
                     notes: Optional[str] = None) -> Optional[int]:
    """Insert one prediction. Returns its id, or None if it already existed.

    Raises sqlite3.IntegrityError if the event has already started, which is
    the point: there is no argument you can make to this function that gets a
    late prediction into the record.
    """
    now = datetime.now(timezone.utc).isoformat()
    if now >= event_start_utc:
        raise ValueError(
            f"refusing to write a prediction for {game_id}: the event started at "
            f"{event_start_utc} and it is now {now}. A prediction made after kickoff "
            f"is not a prediction.")
    try:
        cur = conn.execute(
            "INSERT INTO ledger (sport, league, competition_id, game_id, market_type, side, "
            "line, price_taken, book, model_version, model_seal, model_prob, fair_prob, ev, "
            "kelly_fraction, recommended_unit, created_at, event_start_utc, lock_time_utc, "
            "is_shadow, supersedes_id, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sport, league, competition_id, game_id, market_type, side, line, price_taken,
             book, model_version, model_seal, model_prob, fair_prob, ev, kelly_fraction,
             recommended_unit, now, event_start_utc, lock_time_utc,
             1 if is_shadow else 0, supersedes_id, notes))
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError as exc:
        if "UNIQUE" in str(exc):
            return None      # already predicted; not an error
        raise


def grade_pending(conn: sqlite3.Connection, results: Dict[str, Dict[str, Any]],
                  grading_source: str = "archive",
                  reveal_sealed_outcomes: bool = False) -> Dict[str, int]:
    """Settle pending rows whose games have finished.

    `results` maps game_id to {'home_score', 'away_score', 'status'}. A game
    that was abandoned or never played is voided rather than guessed at: the
    2022 Bills-Bengals game is the reason that branch exists.

    The rows are graded either way. What `reveal_sealed_outcomes` controls is
    only what this function TELLS YOU: with it false, which is always outside
    the single evaluation, a sealed competition's wins, losses and pushes come
    back as one `settled` number. See the note beside SEALED_COMPETITION_PREFIXES.
    """
    now = datetime.now(timezone.utc).isoformat()
    counts = {"win": 0, "loss": 0, "push": 0, "settled": 0,
              "void": 0, "still_pending": 0}
    rows = conn.execute(
        "SELECT id, game_id, market_type, side, line, competition_id "
        "FROM ledger WHERE result = 'pending'"
    ).fetchall()

    for rid, gid, market, side, line, comp in rows:
        r = results.get(gid)
        if not r or r.get("status") != "final" or r.get("home_score") is None:
            if r and r.get("status") in ("cancelled", "abandoned"):
                conn.execute("UPDATE ledger SET result='void', graded_at=?, grading_source=? "
                             "WHERE id=?", (now, grading_source, rid))
                counts["void"] += 1
            else:
                counts["still_pending"] += 1
            continue

        hs, as_ = r["home_score"], r["away_score"]
        outcome: Optional[str] = None
        if market == "moneyline":
            if hs == as_:
                outcome = "push"
            else:
                won = (side == "home" and hs > as_) or (side == "away" and as_ > hs)
                outcome = "win" if won else "loss"
        elif market == "spread" and line is not None:
            margin = (hs - as_) if side == "home" else (as_ - hs)
            adj = margin + (line if side == "home" else -line)
            outcome = "push" if abs(adj) < 1e-9 else ("win" if adj > 0 else "loss")
        elif market == "total" and line is not None:
            total = hs + as_
            if abs(total - line) < 1e-9:
                outcome = "push"
            else:
                over = total > line
                outcome = "win" if (over == (side == "over")) else "loss"
        if outcome is None:
            counts["still_pending"] += 1
            continue
        conn.execute("UPDATE ledger SET result=?, graded_at=?, grading_source=? WHERE id=?",
                     (outcome, now, grading_source, rid))
        if is_sealed(comp) and not reveal_sealed_outcomes:
            counts["settled"] += 1
        else:
            counts[outcome] += 1

    conn.commit()
    return counts


def health(conn: sqlite3.Connection, sport: Optional[str] = None,
           reveal_sealed_outcomes: bool = False) -> Dict[str, Any]:
    """The monitoring view: silence here means the pipeline has stopped.

    Monitoring needs to know that rows are being written and graded, not how
    they turned out, so a sealed competition's outcomes arrive here collapsed
    into `settled` unless the caller explicitly asks otherwise.
    """
    where = "WHERE sport = ?" if sport else ""
    args = (sport,) if sport else ()
    total = conn.execute(f"SELECT COUNT(*) FROM ledger {where}", args).fetchone()[0]
    today = datetime.now(timezone.utc).date().isoformat()
    written = conn.execute(
        f"SELECT COUNT(*) FROM ledger {where or 'WHERE 1=1'} AND DATE(created_at) = ?",
        args + (today,)).fetchone()[0]
    graded = conn.execute(
        f"SELECT COUNT(*) FROM ledger {where or 'WHERE 1=1'} AND DATE(graded_at) = ?",
        args + (today,)).fetchone()[0]
    stale = conn.execute(
        f"SELECT COUNT(*) FROM ledger {where or 'WHERE 1=1'} AND result = 'pending' "
        f"AND event_start_utc < ?",
        args + (datetime.now(timezone.utc).isoformat(),)).fetchone()[0]
    by_result: Dict[str, int] = {}
    for result, comp, n in conn.execute(
            f"SELECT result, competition_id, COUNT(*) FROM ledger {where} "
            f"GROUP BY result, competition_id", args).fetchall():
        key = result
        if (result in ("win", "loss", "push")
                and is_sealed(comp) and not reveal_sealed_outcomes):
            key = "settled"
        by_result[key] = by_result.get(key, 0) + n
    return {"total": total, "written_today": written, "graded_today": graded,
            "pending_past_kickoff": stale, "by_result": by_result}
