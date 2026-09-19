"""
ingest_officials.py (NFL)
=========================
Officiating crews, from the nflverse-data `officials` release (CC-BY-4.0).
22,012 assignments across 3,045 games and 317 officials, 2015 to 2026, with
each official's own stable numeric id and jersey number.

WHY ONLY 2015, WHEN A 1999 FILE EXISTS. Lee Sharpe's `nfldata` repository
carries full seven-person crews back to 1999, which is sixteen more seasons.
We do not ingest it, because that repository has no licence file, and no
licence means all rights reserved, not "probably fine". We chose the NFL over
basketball precisely because its data is cleanly licensed; helping ourselves to
an unlicensed file the moment it was convenient would throw that away for a
decade of referee assignments. It is recorded in docs/sources/REJECTED.md with
the reason, and it becomes available the moment the author adds a licence or
answers an email.

For 1999 to 2014 we still know the REFEREE of every game, because games.csv
carries the name under CC-BY-4.0. We do not know the other six officials.
Pages must say which era they are describing.

THE STANDING RULE, inherited from the NBA officials work and non-negotiable:
this stays descriptive. Crews are assigned, not drawn at random. Senior
officials work nationally televised games, playoff games and rivalries, which
differ in pace, stakes and penalty rate before anyone throws a flag. Any
difference we show is an association with the games an official is GIVEN, never
evidence about how they call them, and it is never a betting angle. With a few
hundred officials on a table, some will clear their interval by chance alone,
and the page says so.

IDS. The file keys games by the GSIS 10-digit id, which our `games` table
carries as `external_ids.old_game_id`. We join on that, never on a date and a
team name.

Usage:
    venv/Scripts/python.exe src/Sports/nfl/ingest_officials.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.Sports.core_schema import ensure_core_schema, record_run, INGEST_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nfl.ingest_officials")

SOURCE = "nflverse-data officials (CC-BY-4.0)"
ENDPOINT = "https://github.com/nflverse/nflverse-data/releases/download/officials/officials.csv"
DB_PATH = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
UA = {"User-Agent": "BettingBuddy/1.0 (archive ingest)"}


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest NFL officiating crews.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("downloading %s", ENDPOINT)
    with urllib.request.urlopen(urllib.request.Request(ENDPOINT, headers=UA), timeout=180) as r:
        text = r.read().decode("utf-8", errors="replace")

    conn = sqlite3.connect(args.db, timeout=120)
    conn.row_factory = sqlite3.Row
    ensure_core_schema(conn)

    # GSIS 10-digit id -> our canonical game id.
    by_gsis: Dict[str, str] = {}
    for row in conn.execute("SELECT game_id, external_ids FROM games WHERE external_ids IS NOT NULL"):
        try:
            ext = json.loads(row["external_ids"])
        except (TypeError, ValueError):
            continue
        old = ext.get("old_game_id")
        if old:
            by_gsis[str(old)] = row["game_id"]
    logger.info("%d games available for the GSIS crosswalk", len(by_gsis))

    fetched_at = datetime.now(timezone.utc).isoformat()
    persons: Dict[str, tuple] = {}
    links = []
    unmatched = 0

    for r in csv.DictReader(io.StringIO(text)):
        oid, gsis = _s(r.get("official_id")), _s(r.get("game_id"))
        if not (oid and gsis):
            continue
        gid = by_gsis.get(gsis)
        if not gid:
            unmatched += 1
            continue
        pid = f"nfl-off-{oid}"
        name = _s(r.get("official_name"))
        persons.setdefault(pid, (
            pid, "football", "official", name,
            (name or "").split(" ")[0] or None,
            " ".join((name or "").split(" ")[1:]) or None,
            None, None,
            json.dumps({"nflverse_official_id": oid, "jersey": _s(r.get("jersey_number"))}),
            SOURCE, ENDPOINT, fetched_at, INGEST_VERSION))
        links.append((gid, pid, _s(r.get("position"))))

    conn.executemany("INSERT OR REPLACE INTO persons VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     list(persons.values()))
    conn.executemany("INSERT OR REPLACE INTO officials VALUES (?,?,?)", links)
    conn.commit()

    record_run(conn, "officials", SOURCE, ENDPOINT, started_at,
               datetime.now(timezone.utc).isoformat(), len(links),
               notes=f"officials={len(persons)} unmatched_games={unmatched}")

    games_covered = conn.execute("SELECT COUNT(DISTINCT game_id) FROM officials").fetchone()[0]
    logger.info("SUMMARY assignments=%d officials=%d games=%d unmatched=%d",
                len(links), len(persons), games_covered, unmatched)
    if unmatched:
        logger.warning("%d assignment(s) referenced a GSIS game id not in our schedule", unmatched)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
