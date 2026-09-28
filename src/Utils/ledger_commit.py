"""
Commit to each day's picks with a hash, and chain the hashes, so anyone can
check that the track record was not edited after the fact.

WHY THIS EXISTS
predictions_log already refuses a changed pick, a regrade and a deletion
(its triggers), and the public server re-applies those rules on every sync
(ledger_sync.py). That makes the record honest as long as you trust whoever
runs the database, because whoever runs the database can drop a trigger.
A commitment removes the need for that trust, one step at a time:

  1. Before the day's games, the home PC hashes the day's logged picks
     together with a random secret (the nonce) and stores the hash here.
     The hash says nothing about the picks; the nonce is what stops anyone
     guessing them from it (a one-game day has only a few hundred plausible
     pick/confidence combinations, which a bare hash would give away).
  2. Each commitment's own hash also covers the previous commitment's hash,
     so the entries form a chain. Rewriting any past pick changes that day's
     hash, which changes every chain hash after it.
  3. Once every game in a commitment has tipped off, the nonce and the
     picks are revealed (GET /api/ledger/commitments) and anyone can
     recompute the hash, in their browser or with sha256sum.
  4. Once the chain hashes are also published somewhere we do not control
     (publish_commitments.py; off until the owner configures a target), the
     record is provable against us too: a rewritten chain would no longer
     match the copy the world already saw.

Until step 4 is switched on, a commitment proves the record is internally
consistent and has not been edited since it was committed, provided the whole
chain was not rebuilt. The page says exactly that.

CANONICAL FORM, method "bb-ledger-commit-v1" (do not change; a new form gets
a new version string and old rows keep verifying under the old one)

  A commitment covers a set of predictions_log rows: every row with one
  log_date that no earlier commitment covers, excluding the market-implied
  placeholder model (which the public record excludes too). Rows are taken in
  ascending id order. Each row becomes a JSON array of exactly these fields:

      [id, log_date, sport, sportsbook, game_key, game_start_time_utc,
       logged_at, model, predicted_winner, winner_confidence, home_ml, away_ml]

  id is a JSON integer. Every other value is a JSON string or null. The three
  numbers (winner_confidence in percent, home_ml and away_ml in American odds)
  are written as strings with exactly six decimals, Python format(x, ".6f"),
  e.g. "67.230000" and "-150.000000", so no language's float printing can
  make the same number hash two ways. Missing values are null.

  The preimage is the UTF-8 encoding of this JSON array, with no whitespace
  (separators "," and ":") and non-ASCII characters left as they are:

      ["bb-ledger-commit-v1", log_date, seq, nonce, [row, row, ...]]

  seq is 1 for a day's first commitment and counts up for picks logged later
  that day (a game added to the slate after the morning run). nonce is 64 hex
  characters from secrets.token_hex(32).

      picks_sha256 = SHA-256(preimage), lowercase hex

  The chain link is the SHA-256 of the same kind of JSON array:

      ["bb-ledger-commit-v1", "chain", prev_chain_sha256, log_date, seq,
       n_picks, committed_at, picks_sha256]

  prev_chain_sha256 is the previous commitment's chain_sha256 (commitments in
  id order), or 64 zeros for the first. committed_at is UTC,
  'YYYY-MM-DDTHH:MM:SS+00:00', the same form as the ledger's own timestamps.

  JSON.stringify in a browser produces byte-identical output for these arrays
  (strings, small integers, null), which is what the page's verifier relies on.

WHAT IS DELIBERATELY NOT IN THE HASH
  Grading (actual_winner, closing lines, CLV) is written after the game and
  is not a claim made before it; the over/under columns are a withdrawn pick
  that is never served; why_json is the explanation, not the pick. Expected
  value is arithmetic on the probability and the odds, which are both in.

THE SINGLE-WRITER RULE
  Only the home PC writes commitments (DEPLOY.md 3a: it is the only machine
  that logs picks). The public server receives them through ledger_sync and
  never writes its own; commit_ledger.py refuses to run with
  PREDICTIONS_SOURCE=ledger, which is how the public server is configured.
  The table is append-only, enforced by triggers here and again on the server.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

METHOD_VERSION = "bb-ledger-commit-v1"
GENESIS = "0" * 64

#: main_api.SIMULATED_MODEL_TAG, repeated so this module does not have to
#: import the whole API. Such rows are never written, and the public record
#: filters them anyway; a test pins the two strings together.
SIMULATED_MODEL_TAG = "implied_probability_sim"

CANONICAL_FIELDS = (
    "id", "log_date", "sport", "sportsbook", "game_key", "game_start_time_utc",
    "logged_at", "model", "predicted_winner", "winner_confidence", "home_ml", "away_ml",
)
_NUMERIC_FIELDS = frozenset({"winner_confidence", "home_ml", "away_ml"})

#: The guard triggers ledger_sync requires on the server before it will write.
GUARD_TRIGGERS = frozenset({"ledger_commitments_no_update", "ledger_commitments_no_delete"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_commitments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_date TEXT NOT NULL,
    seq INTEGER NOT NULL,
    n_picks INTEGER NOT NULL CHECK (n_picks > 0),
    pick_ids TEXT NOT NULL,            -- JSON array of predictions_log ids, ascending
    first_tipoff_utc TEXT NOT NULL,
    last_tipoff_utc TEXT NOT NULL,     -- the nonce and picks are revealed after this
    nonce TEXT NOT NULL,               -- secret until last_tipoff_utc; see the module docstring
    picks_sha256 TEXT NOT NULL,
    prev_chain_sha256 TEXT NOT NULL,
    chain_sha256 TEXT NOT NULL UNIQUE,
    committed_at TEXT NOT NULL,
    method_version TEXT NOT NULL,
    UNIQUE (log_date, seq)
);

-- A commitment is a statement made at a moment. Changing one afterwards is
-- the offence it exists to expose, so the table refuses it outright.
CREATE TRIGGER IF NOT EXISTS ledger_commitments_no_update
BEFORE UPDATE ON ledger_commitments
BEGIN
    SELECT RAISE(ABORT, 'a ledger commitment is immutable');
END;

CREATE TRIGGER IF NOT EXISTS ledger_commitments_no_delete
BEFORE DELETE ON ledger_commitments
BEGIN
    SELECT RAISE(ABORT, 'ledger commitments are append-only');
END;
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the table and its guards if they are not there. Additive only."""
    conn.executescript(SCHEMA)


def has_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger_commitments'"
    ).fetchone() is not None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "id":
        return int(value)
    if field in _NUMERIC_FIELDS:
        return format(float(value), ".6f")
    return str(value)


def canonical_row(row: Any) -> list:
    """One predictions_log row (sqlite3.Row or dict) as its canonical array."""
    return [canonical_value(f, row[f]) for f in CANONICAL_FIELDS]


def preimage(log_date: str, seq: int, nonce: str, rows: List[list]) -> str:
    """The exact text that picks_sha256 hashes. `rows` are canonical arrays."""
    return _dumps([METHOD_VERSION, log_date, int(seq), nonce, rows])


def picks_hash(log_date: str, seq: int, nonce: str, rows: List[list]) -> str:
    return _sha256(preimage(log_date, seq, nonce, rows))


def chain_hash(prev: str, log_date: str, seq: int, n_picks: int,
               committed_at: str, picks_sha256: str) -> str:
    return _sha256(_dumps([METHOD_VERSION, "chain", prev, log_date, int(seq),
                           int(n_picks), committed_at, picks_sha256]))


def _rows_by_id(conn: sqlite3.Connection, ids: Iterable[int]) -> Dict[int, Any]:
    ids = list(ids)
    if not ids:
        return {}
    old = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        out = {}
        # Chunked: SQLite caps bound parameters (999 on older builds).
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            for r in conn.execute(
                    f"SELECT {', '.join(CANONICAL_FIELDS)} FROM predictions_log "
                    f"WHERE id IN ({', '.join('?' for _ in chunk)})", chunk):
                out[int(r["id"])] = r
        return out
    finally:
        conn.row_factory = old


def _commitments(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    old = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM ledger_commitments ORDER BY id")]
    finally:
        conn.row_factory = old


def commit_pending(conn: sqlite3.Connection, now: Optional[str] = None,
                   nonce_source=None) -> List[Dict[str, Any]]:
    """Commit every logged pick no commitment covers yet. Returns what was written.

    One commitment per log_date that has uncovered picks, in date order, all
    inside one IMMEDIATE transaction so a pick logged by a concurrent request
    is either wholly in this run or wholly left for the next. No picks means
    nothing is written: an empty commitment would claim a slate that did not
    exist. `now` and `nonce_source` exist for tests.
    """
    ensure_schema(conn)
    now = now or _utc_now()
    nonce_source = nonce_source or (lambda: secrets.token_hex(32))
    old_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = _commitments(conn)
            covered = set()
            for c in existing:
                covered.update(json.loads(c["pick_ids"]))
            prev = existing[-1]["chain_sha256"] if existing else GENESIS

            pending: Dict[str, List[int]] = {}
            for rid, log_date in conn.execute(
                    "SELECT id, log_date FROM predictions_log "
                    "WHERE model IS NULL OR model != ? ORDER BY id", (SIMULATED_MODEL_TAG,)):
                if rid not in covered:
                    pending.setdefault(log_date, []).append(int(rid))

            written = []
            for log_date in sorted(pending):
                ids = pending[log_date]
                rows = _rows_by_id(conn, ids)
                canon = [canonical_row(rows[i]) for i in ids]
                tips = sorted(rows[i]["game_start_time_utc"] for i in ids)
                seq = (conn.execute("SELECT MAX(seq) FROM ledger_commitments WHERE log_date = ?",
                                    (log_date,)).fetchone()[0] or 0) + 1
                nonce = nonce_source()
                p_hash = picks_hash(log_date, seq, nonce, canon)
                c_hash = chain_hash(prev, log_date, seq, len(ids), now, p_hash)
                conn.execute(
                    "INSERT INTO ledger_commitments (log_date, seq, n_picks, pick_ids, "
                    "first_tipoff_utc, last_tipoff_utc, nonce, picks_sha256, prev_chain_sha256, "
                    "chain_sha256, committed_at, method_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (log_date, seq, len(ids), _dumps(ids), tips[0], tips[-1], nonce, p_hash,
                     prev, c_hash, now, METHOD_VERSION))
                written.append({"log_date": log_date, "seq": seq, "n_picks": len(ids),
                                "picks_sha256": p_hash, "chain_sha256": c_hash,
                                "committed_at": now, "first_tipoff_utc": tips[0],
                                "before_first_tipoff": now < tips[0]})
                prev = c_hash
            conn.execute("COMMIT")
            return written
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.isolation_level = old_isolation


def verify(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Recheck every commitment against the chain and against today's rows.

    Per entry: "chain" says whether its link recomputes and follows the one
    before it; "rows" says whether the picks it covers, as the database holds
    them NOW, still hash to what was committed ("match", "changed" or
    "missing"). Needs the nonces, so it runs where the database is.
    """
    if not has_table(conn):
        return {"length": 0, "head": None, "intact": True, "entries": []}
    entries = []
    prev = GENESIS
    intact = True
    for c in _commitments(conn):
        link_ok = (c["prev_chain_sha256"] == prev and c["chain_sha256"] == chain_hash(
            c["prev_chain_sha256"], c["log_date"], c["seq"], c["n_picks"],
            c["committed_at"], c["picks_sha256"]))
        ids = json.loads(c["pick_ids"])
        rows = _rows_by_id(conn, ids)
        if len(rows) != len(ids) or len(ids) != c["n_picks"]:
            rows_state = "missing"
        else:
            canon = [canonical_row(rows[i]) for i in ids]
            rows_state = ("match" if picks_hash(c["log_date"], c["seq"], c["nonce"], canon)
                          == c["picks_sha256"] else "changed")
        if not link_ok or rows_state != "match":
            intact = False
        entries.append({"id": c["id"], "log_date": c["log_date"], "seq": c["seq"],
                        "chain": "ok" if link_ok else "broken", "rows": rows_state})
        prev = c["chain_sha256"]
    return {"length": len(entries), "head": prev if entries else None,
            "intact": intact, "entries": entries}


def public_view(conn: sqlite3.Connection, now: Optional[str] = None,
                since: Optional[str] = None) -> Dict[str, Any]:
    """What GET /api/ledger/commitments serves.

    A commitment's nonce and picks are revealed only once EVERY game it
    covers has tipped off, judged against both the stored last tip-off and
    the rows' own tip-off times (whichever is later), the same rule
    /api/prediction-log uses to seal a pick. Before that it shows only its
    hashes, its size and its timing, which say nothing about the picks.
    """
    now = now or _utc_now()
    check = verify(conn)
    if not has_table(conn):
        return {"now": now, "chain": {k: check[k] for k in ("length", "head", "intact")},
                "commitments": []}
    state = {e["id"]: e for e in check["entries"]}
    out = []
    for c in _commitments(conn):
        if since and c["log_date"] < since:
            continue
        ids = json.loads(c["pick_ids"])
        rows = _rows_by_id(conn, ids)
        last_tip = max([c["last_tipoff_utc"]] + [r["game_start_time_utc"] for r in rows.values()])
        revealed = now >= last_tip
        entry = {
            "id": c["id"], "log_date": c["log_date"], "seq": c["seq"],
            "n_picks": c["n_picks"], "pick_ids": ids,
            "committed_at": c["committed_at"],
            "first_tipoff_utc": c["first_tipoff_utc"], "last_tipoff_utc": last_tip,
            "committed_before_first_tipoff": c["committed_at"] < c["first_tipoff_utc"],
            "method_version": c["method_version"],
            "picks_sha256": c["picks_sha256"],
            "prev_chain_sha256": c["prev_chain_sha256"],
            "chain_sha256": c["chain_sha256"],
            "server_check": {"chain": state[c["id"]]["chain"], "rows": state[c["id"]]["rows"]},
            "revealed": revealed,
            "nonce": None,
            "picks": None,
        }
        if revealed and len(rows) == len(ids):
            entry["nonce"] = c["nonce"]
            entry["picks"] = [canonical_row(rows[i]) for i in ids]
        out.append(entry)
    return {"now": now, "chain": {k: check[k] for k in ("length", "head", "intact")},
            "commitments": out}
