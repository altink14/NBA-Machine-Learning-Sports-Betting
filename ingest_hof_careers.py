"""
ingest_hof_careers.py
=====================
Links Hall of Fame inductees to their NBA person_id and pulls their career
totals, so the Hall can be described by what its members actually did rather
than by reputation.

Two things this deliberately does NOT do:

1. It does not force a match. Only 149 of the 243 inducted players appear in the
   NBA's all-time index, and that is correct - the Naismith Hall enshrines WNBA
   players, Globetrotters, and international figures who never played an NBA
   game. Those inductees get no career row and are reported as unmatched rather
   than fuzzy-matched into somebody else's statistics.

2. It does not treat the resulting numbers as "the Hall of Fame standard". They
   describe the NBA players in the Hall, which is a different and narrower
   population, and the page built on this has to say so.

Run after ingest_hall_of_fame.py:
    venv/Scripts/python.exe ingest_hof_careers.py
"""

import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.Utils.nba_stats_client import get_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hof_careers")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS hof_career_totals (
    person_id INTEGER PRIMARY KEY,
    slug TEXT,
    name TEXT,
    class_year INTEGER,
    from_year INTEGER,
    to_year INTEGER,
    seasons INTEGER,
    gp INTEGER, min REAL, pts INTEGER, reb INTEGER, ast INTEGER,
    stl INTEGER, blk INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER,
    ftm INTEGER, fta INTEGER,
    ppg REAL, rpg REAL, apg REAL,
    fetched_at TEXT
)
"""


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace(".", "").replace("'", "'").split())


def main() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA)
        inductees = conn.execute(
            "SELECT slug, name, class_year FROM hof_inductees WHERE category = 'Player'"
        ).fetchall()
        if not inductees:
            logger.error("No inducted players found - run ingest_hall_of_fame.py first.")
            return 1

        client = get_client()
        logger.info("Fetching the NBA all-time player index...")
        # Names are not unique in the index - Patrick Ewing and Patrick Ewing Jr.
        # both appear as "Patrick Ewing", and a plain dict lets the last one win,
        # which credited the Hall of Famer with his son's 3 career points. Keep
        # every candidate and choose per inductee below.
        index: dict = {}
        for r in client.common_all_players(season="2025-26", is_only_current_season=0):
            key = norm(r.get("DISPLAY_FIRST_LAST"))
            if key:
                index.setdefault(key, []).append(r)

        def choose(candidates, class_year):
            """Pick the candidate whose career could actually have been enshrined.

            A player cannot be inducted before he debuted, so any candidate whose
            first season postdates the class year is the wrong person. Among what
            is left, the longest career wins - a Hall of Famer is essentially
            never the shorter of two same-named careers.
            """
            plausible = [
                c for c in candidates
                if not (class_year and c.get("FROM_YEAR") and int(c["FROM_YEAR"]) > class_year)
            ]
            pool = plausible or []
            if not pool:
                return None
            return max(
                pool,
                key=lambda c: (int(c.get("TO_YEAR") or 0) - int(c.get("FROM_YEAR") or 0)),
            )

        matched, unmatched, rejected = [], [], []
        for row in inductees:
            candidates = index.get(norm(row["name"]))
            if not candidates:
                unmatched.append(row)
                continue
            hit = choose(candidates, row["class_year"])
            if hit is None:
                rejected.append(row["name"])
                continue
            if len(candidates) > 1:
                logger.info(
                    "  %s: %d players share that name, took the one active %s-%s",
                    row["name"], len(candidates), hit.get("FROM_YEAR"), hit.get("TO_YEAR"),
                )
            matched.append((row, hit))
        if rejected:
            logger.info(
                "%d inductee(s) had only implausible name matches and were skipped: %s",
                len(rejected), ", ".join(rejected),
            )
        logger.info(
            "%d of %d inducted players matched an NBA person_id (%d never played in the NBA).",
            len(matched), len(inductees), len(unmatched),
        )

        now = datetime.now(timezone.utc).isoformat()
        written = skipped = 0
        for i, (row, hit) in enumerate(matched, 1):
            pid = hit["PERSON_ID"]
            if conn.execute(
                "SELECT 1 FROM hof_career_totals WHERE person_id = ?", (pid,)
            ).fetchone():
                skipped += 1
                continue
            try:
                sets = client.player_career_stats(pid, per_mode="Totals")
            except Exception as exc:
                logger.warning("  %s (%s): career fetch failed - %s", row["name"], pid, exc)
                continue

            career = sets.get("CareerTotalsRegularSeason") or []
            regular = sets.get("SeasonTotalsRegularSeason") or []
            if not career:
                logger.warning("  %s: no career totals returned", row["name"])
                continue
            t = career[0]

            gp = t.get("GP") or 0
            conn.execute(
                """
                INSERT OR REPLACE INTO hof_career_totals
                    (person_id, slug, name, class_year, from_year, to_year, seasons,
                     gp, min, pts, reb, ast, stl, blk, fgm, fga, fg3m, ftm, fta,
                     ppg, rpg, apg, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pid, row["slug"], row["name"], row["class_year"],
                    int(hit.get("FROM_YEAR") or 0) or None,
                    int(hit.get("TO_YEAR") or 0) or None,
                    len(regular),
                    gp, t.get("MIN"), t.get("PTS"), t.get("REB"), t.get("AST"),
                    t.get("STL"), t.get("BLK"), t.get("FGM"), t.get("FGA"),
                    t.get("FG3M"), t.get("FTM"), t.get("FTA"),
                    round((t.get("PTS") or 0) / gp, 1) if gp else None,
                    round((t.get("REB") or 0) / gp, 1) if gp else None,
                    round((t.get("AST") or 0) / gp, 1) if gp else None,
                    now,
                ),
            )
            written += 1
            if written % 20 == 0:
                conn.commit()
                logger.info("  %d/%d fetched...", i, len(matched))

        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM hof_career_totals").fetchone()[0]
        logger.info(
            "Done. %d written, %d already present. hof_career_totals holds %d players.",
            written, skipped, total,
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
