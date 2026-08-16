"""
ingest_hall_of_fame.py
======================
Ingests the Naismith Memorial Basketball Hall of Fame inductee list into
TeamData.sqlite as `hof_inductees`.

Why a scraper: there is no Hall of Fame API. nba_api has no endpoint for it,
ESPN publishes none, and the local player_awards table covers four players. The
alternative was writing the list from memory, which would be wrong somewhere and
would go stale every September without anyone noticing - so it is scraped from
the Hall's own site, with the source URL and fetch time stored per row.

The list includes players, coaches, referees, contributors and whole teams. That
is the Hall's own scope and it is preserved rather than filtered down to players,
with the category kept so a caller can filter.

Run once, and again each September after the new class is enshrined:
    venv/Scripts/python.exe ingest_hall_of_fame.py
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hof_ingest")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
SOURCE_URL = "https://www.hoophall.com/hall-of-famers"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS hof_inductees (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    class_year INTEGER,
    sort_name TEXT,
    headshot_url TEXT,
    profile_url TEXT,
    source_url TEXT,
    fetched_at TEXT
)
"""

# A class year outside this range means the page changed shape and the parse is
# reading something that is not a year. 1959 is the Hall's first class.
MIN_YEAR, MAX_YEAR = 1959, 2100


def parse(html: str):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for li in soup.select("li.inductee-result"):
        link = li.find("a", href=True)
        name_el = li.find("h3")
        if not link or not name_el:
            continue
        name = name_el.get_text(strip=True)
        if not name:
            continue

        href = link["href"]
        slug = href.rstrip("/").rsplit("/", 1)[-1]

        year = None
        raw_year = li.get("data-class-sort")
        if raw_year and raw_year.isdigit():
            year = int(raw_year)
        else:
            p = li.find("p")
            if p:
                m = re.search(r"(\d{4})", p.get_text())
                if m:
                    year = int(m.group(1))
        if year is not None and not (MIN_YEAR <= year <= MAX_YEAR):
            logger.warning("Ignoring implausible class year %s for %s", year, name)
            year = None

        pill = li.find(class_="pill")
        img = li.find("img")
        src = img.get("src") if img else None
        if src and src.startswith("/"):
            src = "https://www.hoophall.com" + src

        out.append({
            "slug": slug,
            "name": name,
            "category": pill.get_text(strip=True).title() if pill else None,
            "class_year": year,
            "sort_name": li.get("data-alpha-sort") or name.lower(),
            "headshot_url": src,
            "profile_url": href,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Parse and report, write nothing.")
    args = ap.parse_args()

    logger.info("Fetching %s", SOURCE_URL)
    resp = requests.get(SOURCE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    rows = parse(resp.text)

    # The Hall has enshrined over 400 people and teams. A parse returning a
    # handful means the markup moved, and overwriting a good table with it would
    # be worse than failing.
    if len(rows) < 300:
        logger.error(
            "Only %d inductees parsed, expected 400+. The page markup has probably "
            "changed; refusing to overwrite the table.", len(rows)
        )
        return 1

    with_year = sum(1 for r in rows if r["class_year"])
    logger.info(
        "Parsed %d inductees (%d with a class year), %d categories, classes %s-%s",
        len(rows), with_year, len({r["category"] for r in rows if r["category"]}),
        min((r["class_year"] for r in rows if r["class_year"]), default="?"),
        max((r["class_year"] for r in rows if r["class_year"]), default="?"),
    )
    if args.dry_run:
        for r in rows[:5]:
            logger.info("  %s | %s | %s", r["name"], r["category"], r["class_year"])
        return 0

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(SCHEMA)
        conn.executemany(
            """
            INSERT INTO hof_inductees
                (slug, name, category, class_year, sort_name, headshot_url,
                 profile_url, source_url, fetched_at)
            VALUES (:slug, :name, :category, :class_year, :sort_name, :headshot_url,
                    :profile_url, :source_url, :fetched_at)
            ON CONFLICT(slug) DO UPDATE SET
                name=excluded.name, category=excluded.category,
                class_year=excluded.class_year, sort_name=excluded.sort_name,
                headshot_url=excluded.headshot_url, profile_url=excluded.profile_url,
                source_url=excluded.source_url, fetched_at=excluded.fetched_at
            """,
            [{**r, "source_url": SOURCE_URL, "fetched_at": now} for r in rows],
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM hof_inductees").fetchone()[0]
        logger.info("hof_inductees now holds %d rows.", total)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
