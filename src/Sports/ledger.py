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
from typing import Any, Dict, List, Optional

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


def ensure_ledger(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


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
