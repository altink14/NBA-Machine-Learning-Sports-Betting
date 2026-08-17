"""
ingest_draft_bios.py
====================
Fills `player_bio` for recent draft classes, so the draft board can show what
nba.com's board shows: position, height, weight and country alongside the pick.

draft_history carries the pick, the team and the school, and nothing about the
player. The remaining columns come from commonplayerinfo, one request each, which
is why this is an ingest rather than a per-request fetch - sixty requests to
render a page would be rude to the upstream and slow for the reader.

player_bio is reused rather than a new draft-specific table, because it already
has exactly these columns and the player detail page already reads it. Filling it
here improves those pages as a side effect.

Bios are only fetched for players who do not already have one, so a repeat run
costs nothing and a new class costs sixty requests once.

Run standalone, or let refresh_registry.py call it weekly:
    venv/Scripts/python.exe ingest_draft_bios.py [--seasons 2]
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("draft_bios")

DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")


def _fetch_bio(player_id: int):
    from nba_api.stats.endpoints import commonplayerinfo

    d = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=60).get_dict()
    rs = d["resultSets"][0]
    if not rs["rowSet"]:
        return None
    return dict(zip(rs["headers"], rs["rowSet"][0]))


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _combine_measurements(season: int) -> dict:
    """Height and weight from the draft combine, keyed by player id.

    A second source because commonplayerinfo leaves measurements blank for some
    players - Bryce Hopkins was drafted 49th in 2026 with no height or weight on
    his bio - and because the combine is published BEFORE bios exist, which is
    exactly the window each June when a new class lands and the board is busiest.
    """
    from nba_api.stats.endpoints import draftcombineplayeranthro

    try:
        d = draftcombineplayeranthro.DraftCombinePlayerAnthro(
            season_year=str(season), timeout=60
        ).get_dict()
    except Exception as exc:
        logger.warning("Combine measurements unavailable for %s: %s", season, exc)
        return {}
    rs = d["resultSets"][0]
    idx = {h: i for i, h in enumerate(rs["headers"])}
    out = {}
    for r in rs["rowSet"]:
        pid = r[idx["PLAYER_ID"]]
        if pid is None:
            continue
        weight = r[idx["WEIGHT"]] if "WEIGHT" in idx else None
        height = r[idx.get("HEIGHT_W_SHOES_FT_IN", -1)] if "HEIGHT_W_SHOES_FT_IN" in idx else None
        out[int(pid)] = {
            # Combine weights carry a decimal ("218.80"); the board shows pounds.
            "weight": str(round(float(weight))) if weight else None,
            "height": height or None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=2,
                    help="How many recent draft classes to fill (default 2).")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        seasons = [
            r["season"] for r in conn.execute(
                "SELECT DISTINCT season FROM draft_history ORDER BY season DESC LIMIT ?",
                (args.seasons,),
            )
        ]
        if not seasons:
            logger.error("No draft classes on record - run ingest_draft.py first.")
            return 1
        logger.info("Filling bios for draft classes: %s", seasons)

        placeholders = ",".join("?" * len(seasons))
        missing = conn.execute(
            f"""
            SELECT d.person_id, d.player_name, d.season
            FROM draft_history d
            LEFT JOIN player_bio b ON b.player_id = d.person_id
            WHERE d.season IN ({placeholders})
              AND (b.player_id IS NULL
                   OR b.height IS NULL OR b.height = ''
                   OR b.weight IS NULL OR b.weight = '')
            ORDER BY d.season DESC, d.overall_pick ASC
            """,
            seasons,
        ).fetchall()
        logger.info("%d picks need a bio or measurements.", len(missing))
        if not missing:
            return 0

        # Combine data is fetched once per season, not per player.
        combine = {}
        for season in {row["season"] for row in missing}:
            combine.update(_combine_measurements(season))
        if combine:
            logger.info("Combine measurements on hand for %d players.", len(combine))

        now = datetime.now(timezone.utc).isoformat()
        written = failed = 0
        for i, row in enumerate(missing, 1):
            pid = row["person_id"]
            try:
                info = _fetch_bio(pid)
            except Exception as exc:
                logger.warning("  %s (%s): %s", row["player_name"], pid, exc)
                failed += 1
                continue
            if not info:
                # A drafted player who never signed has no bio record. Counted,
                # not invented.
                logger.info("  %s: no bio published", row["player_name"])
                failed += 1
                continue

            # Whichever measurement the bio leaves blank, take the combine's.
            fallback = combine.get(pid, {})
            height = info.get("HEIGHT") or fallback.get("height")
            weight = info.get("WEIGHT") or fallback.get("weight")

            conn.execute(
                """
                INSERT INTO player_bio (
                    player_id, full_name, first_name, last_name, team_id, team_abbr,
                    jersey, position, height, weight, birth_date, country, school,
                    draft_year, draft_round, draft_number, years_experience,
                    is_active, fetched_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(player_id) DO UPDATE SET
                    position=excluded.position,
                    height=COALESCE(NULLIF(excluded.height, ''), player_bio.height),
                    weight=COALESCE(NULLIF(excluded.weight, ''), player_bio.weight),
                    country=excluded.country,
                    school=excluded.school, fetched_at=excluded.fetched_at
                """,
                (
                    pid,
                    info.get("DISPLAY_FIRST_LAST"),
                    info.get("FIRST_NAME"),
                    info.get("LAST_NAME"),
                    _int(info.get("TEAM_ID")),
                    info.get("TEAM_ABBREVIATION"),
                    info.get("JERSEY"),
                    info.get("POSITION"),
                    height,
                    weight,
                    info.get("BIRTHDATE"),
                    info.get("COUNTRY"),
                    info.get("SCHOOL"),
                    _int(info.get("DRAFT_YEAR")),
                    _int(info.get("DRAFT_ROUND")),
                    _int(info.get("DRAFT_NUMBER")),
                    _int(info.get("SEASON_EXP")) or 0,
                    1 if info.get("ROSTERSTATUS") in ("Active", 1, "1") else 0,
                    now,
                ),
            )
            written += 1
            if written % 15 == 0:
                conn.commit()
                logger.info("  %d/%d...", i, len(missing))
        conn.commit()

        total = conn.execute("SELECT COUNT(*) FROM player_bio").fetchone()[0]
        logger.info(
            "Done. %d bios written, %d unavailable. player_bio holds %d rows.",
            written, failed, total,
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
