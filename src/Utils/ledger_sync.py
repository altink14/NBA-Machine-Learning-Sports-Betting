"""
Mirror the home PC's ledger onto the public server without giving the server a
way to change the past.

WHY THIS EXISTS (DEPLOY.md section 3a)
The home PC is the only machine that can make, record and grade picks:
stats.nba.com refuses cloud IPs, so only it has fresh team stats, and it is
where the closing-line recorder and the grader run. The public track record,
though, is served by a cloud machine. Something has to carry the record from
one to the other, and that something must not become the easiest way to
rewrite it.

HOW
The home PC uploads a consistent copy of OddsData.sqlite. The server does NOT
swap files. It merges the upload row by row, inside one transaction, into its
own copy, and its own copy still carries the triggers that forbid changing a
pick, regrading a result or deleting a row. So the rules the ledger enforces
at home are enforced again, independently, on the public copy. Anything they
refuse aborts the whole sync and nothing is written.

Checks the triggers cannot make are made here, before anything is written:
  - every row the server already has must still be in the upload (a missing
    row is a deletion, and the no-delete trigger only sees DELETE statements);
  - a row id must mean the same game on both sides (two machines each writing
    their own "first" pick produce this, and there is no honest merge);
  - the ledger tables on the server must still have their guard triggers.

After a sync the server returns a fingerprint of what it now holds, and the
home PC compares it with the fingerprint of what it sent. Equal means the
public record is byte-for-byte the home record.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Dict, List, Optional

#: The tables carried to the server, with the triggers each one must have
#: there before we will write into it. odds_snapshots is observations, not
#: predictions, so it has no immutability triggers of its own; the shrink
#: check below still refuses an upload that has lost any of them.
SYNCED_TABLES: Dict[str, frozenset] = {
    "predictions_log": frozenset({
        "predictions_log_immutable", "predictions_log_no_regrade", "predictions_log_no_delete"}),
    "ledger": frozenset({
        "ledger_prediction_is_immutable", "ledger_no_regrade", "ledger_no_delete"}),
    "odds_snapshots": frozenset(),
}

#: The natural key of each table: the thing that says which game a row is
#: about. The same id with a different natural key means the two copies grew
#: apart, which is refused rather than guessed at.
NATURAL_KEYS: Dict[str, tuple] = {
    "predictions_log": ("log_date", "sportsbook", "game_key"),
    "ledger": ("game_id", "market_type", "side", "model_version"),
    "odds_snapshots": ("captured_at", "sportsbook", "game_key"),
}


class SyncRefused(Exception):
    """The upload would have changed the past, or could not be trusted."""


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> List[tuple]:
    """[(name, declared_type)] in table order."""
    return [(r[1], r[2]) for r in conn.execute(f"PRAGMA {schema}.table_info({_q(table)})")]


def _has_table(conn: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    return conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _triggers(conn: sqlite3.Connection, table: str, schema: str = "main") -> set:
    return {r[0] for r in conn.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='trigger' AND tbl_name=?", (table,))}


def fingerprint(conn: sqlite3.Connection, schema: str = "main",
                columns: Optional[Dict[str, List[str]]] = None) -> Dict[str, dict]:
    """Row count, highest id and a SHA-256 of every row, per synced table.

    Columns are hashed in sorted-name order and rows in id order, so two
    copies with the same content produce the same digest whatever order their
    columns were added in. `columns` pins the column set (the server hashes
    exactly the columns the upload had, so an extra column of its own cannot
    make an identical record look different).
    """
    out: Dict[str, dict] = {}
    for table in SYNCED_TABLES:
        if not _has_table(conn, table, schema):
            continue
        cols = sorted(columns[table]) if columns and table in columns else \
            sorted(c for c, _ in _columns(conn, table, schema))
        h = hashlib.sha256()
        n = 0
        for row in conn.execute(
                f"SELECT {', '.join(_q(c) for c in cols)} FROM {schema}.{_q(table)} ORDER BY id"):
            h.update(json.dumps(list(row), separators=(",", ":"), default=str).encode("utf-8"))
            h.update(b"\n")
            n += 1
        max_id = conn.execute(f"SELECT MAX(id) FROM {schema}.{_q(table)}").fetchone()[0]
        out[table] = {"rows": n, "max_id": max_id, "sha256": h.hexdigest()}
    return out


def check_upload(path: str) -> None:
    """Refuse a file that is not a sound SQLite database holding a ledger."""
    try:
        up = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise SyncRefused(f"upload is not a readable SQLite file: {exc}")
    try:
        try:
            verdict = up.execute("PRAGMA quick_check").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise SyncRefused(f"upload is not a SQLite database: {exc}")
        if verdict != "ok":
            raise SyncRefused(f"upload failed quick_check: {verdict}")
        if not _has_table(up, "predictions_log"):
            raise SyncRefused("upload has no predictions_log table; wrong file?")
    finally:
        up.close()


def merge(conn: sqlite3.Connection, upload_path: str) -> dict:
    """Merge the upload into `conn`'s database. All or nothing.

    `conn` must be the server's OddsData connection with its ledger schema
    already ensured by the caller (main_api does that, so a server booted from
    an old snapshot gets the guarded shape first). Returns per-table counts and
    the post-merge fingerprint; raises SyncRefused, having written nothing,
    for anything that would alter or lose a row the server already holds.
    """
    check_upload(upload_path)
    old_isolation = conn.isolation_level
    conn.isolation_level = None  # explicit BEGIN/COMMIT below
    conn.execute("ATTACH DATABASE ? AS up", (upload_path,))
    try:
        tables = [t for t in SYNCED_TABLES if _has_table(conn, t, "up")]

        # Refuse to write into a ledger table that has lost its guards: the
        # whole point is that the server enforces the rules independently.
        for t in tables:
            if not _has_table(conn, t):
                if SYNCED_TABLES[t]:
                    raise SyncRefused(f"server has no {t} table; ensure its schema first")
                continue  # unguarded table, created inside the transaction below
            missing = SYNCED_TABLES[t] - _triggers(conn, t)
            if missing:
                raise SyncRefused(f"server {t} is missing its guard trigger(s) {sorted(missing)}; "
                                  "refusing to write into an unguarded ledger")

        conn.execute("BEGIN IMMEDIATE")
        try:
            report = {}
            hashed_columns = {}
            for t in tables:
                if not _has_table(conn, t):
                    # Inside the transaction, so a refused sync leaves no
                    # empty table behind either.
                    conn.execute(conn.execute(
                        "SELECT sql FROM up.sqlite_master WHERE type='table' AND name=?",
                        (t,)).fetchone()[0])
                up_cols = _columns(conn, t, "up")
                server_names = {c for c, _ in _columns(conn, t)}
                added = []
                for name, decl in up_cols:
                    if name not in server_names:
                        # Columns only ever get ADDED to these tables (grading
                        # grew closing_* and provenance that way), so a newer
                        # home copy may carry one the server's older copy lacks.
                        conn.execute(f"ALTER TABLE main.{_q(t)} ADD COLUMN {_q(name)} {decl}")
                        added.append(name)
                names = [c for c, _ in up_cols]
                hashed_columns[t] = names
                key = NATURAL_KEYS[t]
                tq = _q(t)

                lost = conn.execute(
                    f"SELECT COUNT(*) FROM main.{tq} s WHERE NOT EXISTS "
                    f"(SELECT 1 FROM up.{tq} u WHERE u.id = s.id)").fetchone()[0]
                if lost:
                    raise SyncRefused(
                        f"{t}: the upload is missing {lost} row(s) the public copy already has. "
                        "That is a deletion; the public record only grows.")

                moved = conn.execute(
                    f"SELECT COUNT(*) FROM main.{tq} s JOIN up.{tq} u ON u.id = s.id WHERE "
                    + " OR ".join(f"s.{_q(k)} IS NOT u.{_q(k)}" for k in key)).fetchone()[0]
                if moved:
                    raise SyncRefused(
                        f"{t}: {moved} row id(s) refer to a different game on each side. The two "
                        "copies were written independently; there is no honest merge.")

                col_list = ", ".join(_q(c) for c in names)
                inserted = conn.execute(
                    f"INSERT INTO main.{tq} ({col_list}) SELECT {col_list} FROM up.{tq} u "
                    f"WHERE u.id NOT IN (SELECT id FROM main.{tq}) ORDER BY u.id").rowcount

                others = [c for c in names if c != "id"]
                differs = " OR ".join(f"s.{_q(c)} IS NOT u.{_q(c)}" for c in others) or "0"
                changed_ids = [r[0] for r in conn.execute(
                    f"SELECT s.id FROM main.{tq} s JOIN up.{tq} u ON u.id = s.id WHERE {differs}")]
                # One UPDATE per changed row, so the triggers judge each row
                # exactly as they would a local grading write. A pick that
                # changed raises here and the whole transaction is undone.
                set_clause = ", ".join(f"{_q(c)} = (SELECT {_q(c)} FROM up.{tq} u WHERE u.id = ?)"
                                       for c in others)
                for rid in changed_ids:
                    conn.execute(f"UPDATE main.{tq} SET {set_clause} WHERE id = ?",
                                 [rid] * len(others) + [rid])

                report[t] = {"inserted": inserted, "updated": len(changed_ids),
                             "columns_added": added}
            conn.execute("COMMIT")
        except sqlite3.DatabaseError as exc:
            conn.execute("ROLLBACK")
            raise SyncRefused(f"the public copy refused the upload: {exc}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

        fp = fingerprint(conn, columns=hashed_columns)
        for t in report:
            report[t]["rows"] = fp[t]["rows"]
        return {"tables": report, "fingerprint": fp}
    finally:
        conn.execute("DETACH DATABASE up")
        conn.isolation_level = old_isolation
