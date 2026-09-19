"""
core_schema.py
==============
The cross-sport core schema. Every sport we add implements these tables with
these names, and extends them with sport-specific tables. A sport that adds
`nfl_games` alongside `games` has forked the schema and will fork the frontend
next; that is the failure mode this module exists to prevent.

ONE DATABASE PER SPORT (Data/NflData.sqlite, Data/TeamData.sqlite for the NBA,
and so on) plus the shared Data/OddsData.sqlite for the prediction ledger.
Rationale: independent backfills, independent snapshots, independent deploy
sizes, no write-lock contention between sports, and one sport's corruption
cannot take the others down.

DEVIATIONS FROM THE BLUEPRINT'S SKETCH, and why:

1. Provenance is carried as four columns ON each table (`source`,
   `source_endpoint`, `fetched_at`, `ingest_version`) rather than a separate
   row-per-row `provenance` table, which would roughly double the row count of
   the archive for no query anyone will run. A `ingest_runs` table records one
   row per ingest run per table, so "when did this table last change, from
   what, with which code version" is still answerable.

2. `market_lines` carries `price_over` / `price_under` next to
   `price_home` / `price_away` / `price_draw`, so one table covers two-way
   markets, three-way markets (soccer) and totals without a sub-table.

3. Times: `date_utc` is the kickoff/tip in UTC; `local_date` and `local_time`
   are what the venue's clock said; `tz` names the zone. Half the bugs in
   sports data are timezone bugs, so all three are stored and none is derived
   at read time.

Idempotency is enforced by primary keys and unique indexes, never by
application logic: re-running any ingest must produce zero duplicates.
"""

from __future__ import annotations

import logging
import sqlite3

#: Bump when an ingest's parsing logic changes, so rows written by a buggy
#: version can be found and reprocessed: SELECT ... WHERE ingest_version < N.

logger = logging.getLogger(__name__)

INGEST_VERSION = 1

CORE_SCHEMA = """
-- One row per competition-season-phase, e.g. NFL 2024 regular season.
CREATE TABLE IF NOT EXISTS competitions (
    competition_id  TEXT PRIMARY KEY,
    sport           TEXT NOT NULL,
    league          TEXT NOT NULL,
    season          TEXT NOT NULL,
    season_type     TEXT NOT NULL,
    start_date      TEXT,
    end_date        TEXT,
    source          TEXT,
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER
);

CREATE TABLE IF NOT EXISTS teams (
    team_id         TEXT PRIMARY KEY,   -- canonical, e.g. 'nfl-KC'
    sport           TEXT NOT NULL,
    league          TEXT NOT NULL,
    abbrev          TEXT NOT NULL,
    name            TEXT,
    conference      TEXT,
    division        TEXT,
    first_season    TEXT,
    last_season     TEXT,               -- null while active
    external_ids    TEXT,               -- JSON: {"espn": ..., "pfr": ...}
    source          TEXT,
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER
);

CREATE TABLE IF NOT EXISTS venues (
    venue_id        TEXT PRIMARY KEY,
    sport           TEXT,
    name            TEXT,
    city            TEXT,
    state           TEXT,
    country         TEXT,
    lat             REAL,
    lon             REAL,
    tz              TEXT,
    roof_type       TEXT,               -- outdoors | dome | closed | open
    surface         TEXT,
    altitude_m      REAL,
    capacity        INTEGER,
    source          TEXT,
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER
);

-- Players, coaches and officials share one table; `role` separates them.
CREATE TABLE IF NOT EXISTS persons (
    person_id       TEXT PRIMARY KEY,   -- canonical, source id where one exists
    sport           TEXT NOT NULL,
    role            TEXT NOT NULL,      -- player | coach | official
    full_name       TEXT,
    first_name      TEXT,
    last_name       TEXT,
    dob             TEXT,
    debut_date      TEXT,
    external_ids    TEXT,               -- JSON crosswalk
    source          TEXT,
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER
);

CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,
    sport           TEXT NOT NULL,
    league          TEXT NOT NULL,
    competition_id  TEXT,
    season          TEXT NOT NULL,
    season_type     TEXT NOT NULL,
    week            INTEGER,
    date_utc        TEXT,               -- kickoff/tip in UTC, null if unknown
    local_date      TEXT NOT NULL,      -- what the venue's calendar said
    local_time      TEXT,
    tz              TEXT,
    home_team_id    TEXT NOT NULL,
    away_team_id    TEXT NOT NULL,
    status          TEXT,               -- scheduled | final | postponed
    home_score      INTEGER,
    away_score      INTEGER,
    venue_id        TEXT,
    neutral_site    INTEGER DEFAULT 0,
    attendance      INTEGER,
    broadcast       TEXT,
    external_ids    TEXT,               -- JSON: {"gsis": ..., "pfr": ..., "espn": ...}
    source          TEXT,
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_games_season   ON games(season, season_type);
CREATE INDEX IF NOT EXISTS idx_games_date     ON games(local_date);
CREATE INDEX IF NOT EXISTS idx_games_home     ON games(home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away     ON games(away_team_id);

-- Quarters, halves, innings, periods: whatever the sport divides a game into.
CREATE TABLE IF NOT EXISTS game_periods (
    game_id         TEXT NOT NULL,
    period          INTEGER NOT NULL,
    label           TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    PRIMARY KEY (game_id, period)
);

-- Who was there and in what state: started, active, inactive, did not play.
CREATE TABLE IF NOT EXISTS participations (
    game_id         TEXT NOT NULL,
    person_id       TEXT NOT NULL,
    team_id         TEXT,
    role            TEXT,
    started         INTEGER,
    status          TEXT,               -- active | inactive | dnp
    reason          TEXT,
    PRIMARY KEY (game_id, person_id)
);

CREATE TABLE IF NOT EXISTS officials (
    game_id         TEXT NOT NULL,
    person_id       TEXT NOT NULL,
    position        TEXT,
    PRIMARY KEY (game_id, person_id)
);
CREATE INDEX IF NOT EXISTS idx_officials_person ON officials(person_id);

-- Event-level play data: one row per play, pitch, shot or possession. Kept
-- deliberately thin and sport-agnostic; each sport adds a wide extension
-- table (nfl_plays, nba_pbp_events) keyed on the same (game_id, sequence)
-- so cross-sport surfaces read `events` and sport pages join the extension.
CREATE TABLE IF NOT EXISTS events (
    game_id         TEXT NOT NULL,
    sequence        INTEGER NOT NULL,   -- play_id / action_number, ordered within the game
    period          INTEGER,
    clock_seconds   REAL,               -- seconds REMAINING in the game, sport-normalised
    event_type      TEXT,               -- pass | run | kickoff | shot | pitch | ...
    team_id         TEXT,               -- the team in possession / acting
    actor_id        TEXT,               -- primary actor, source's person id
    description     TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    PRIMARY KEY (game_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_events_type  ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_actor ON events(actor_id);

-- One row per game per book per market. `is_closing` marks the last price
-- before the event started; `captured_at` is when WE saw it, which is null
-- for a historical archive that did not record it. A line whose book and
-- capture time are unknown must say so rather than imply a precision we
-- do not have.
CREATE TABLE IF NOT EXISTS market_lines (
    game_id         TEXT NOT NULL,
    book            TEXT NOT NULL,
    market_type     TEXT NOT NULL,      -- spread | total | moneyline | ...
    line            REAL,
    price_home      INTEGER,
    price_away      INTEGER,
    price_draw      INTEGER,
    price_over      INTEGER,
    price_under     INTEGER,
    captured_at     TEXT,
    is_closing      INTEGER,
    -- How we came to know this price, strongest claim first. No default: a
    -- writer has to say, and the trigger below refuses anything else.
    --   'observed'      our recorder read it off the live market at captured_at
    --   'reconstructed' we pulled a timestamped snapshot out of a vendor's
    --                   historical archive after the fact and applied our own
    --                   last-before-kickoff rule to it
    --   'third_party'   someone else's record of the close, where we chose
    --                   neither the moment nor the book -- nflverse games.csv
    --                   is this, and its own docs do not say which book or
    --                   when, which is why it is the weakest of the three
    -- These are not interchangeable. "We watched this price and beat it" and
    -- "an archive says this price existed and we beat it" are different
    -- claims, and closing line value has to state which one it rests on.
    provenance      TEXT,
    source          TEXT,
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER,
    PRIMARY KEY (game_id, book, market_type)
);

-- NULL is checked explicitly: `NULL NOT IN (...)` is NULL, not true, so a
-- bare NOT IN would let an unstated provenance through.
CREATE TRIGGER IF NOT EXISTS market_lines_provenance_valid
BEFORE INSERT ON market_lines
WHEN NEW.provenance IS NULL
  OR NEW.provenance NOT IN ('observed', 'reconstructed', 'third_party')
BEGIN
    SELECT RAISE(ABORT,
      'market_lines.provenance must be observed, reconstructed or third_party');
END;
CREATE INDEX IF NOT EXISTS idx_market_game ON market_lines(game_id);

CREATE TABLE IF NOT EXISTS rest_travel (
    game_id         TEXT NOT NULL,
    team_id         TEXT NOT NULL,
    days_rest       INTEGER,
    games_last_7    INTEGER,
    games_last_14   INTEGER,
    miles_travelled REAL,
    tz_shift_hours  REAL,
    PRIMARY KEY (game_id, team_id)
);

CREATE TABLE IF NOT EXISTS weather (
    game_id         TEXT PRIMARY KEY,
    temp_f          REAL,
    wind_mph        REAL,
    wind_dir_deg    REAL,
    precip_mm       REAL,
    humidity_pct    REAL,
    roof_state      TEXT,
    source          TEXT,               -- 'games.csv' (stadium reading) or 'open-meteo' (reanalysis)
    source_endpoint TEXT,
    fetched_at      TEXT,
    ingest_version  INTEGER
);

-- One row per ingest run per table: what ran, from where, how much changed.
CREATE TABLE IF NOT EXISTS ingest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name      TEXT NOT NULL,
    source          TEXT,
    source_endpoint TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    rows_written    INTEGER,
    ingest_version  INTEGER,
    notes           TEXT
);
"""


# Columns added after the first databases were built. Additive only: a column
# is appended if missing, never renamed or dropped, so an existing archive
# migrates in place instead of being rebuilt.
_ADDED_COLUMNS = [
    ("market_lines", "provenance", "TEXT"),
]

# Backfilling provenance for rows written before the column existed. This is
# deliberately not a blanket default. The 19,991 closing lines already in
# market_lines came from nflverse's games.csv, which we never watched, so
# stamping them 'observed' would invent exactly the provenance the column was
# added to protect. Unknown sources fall to 'third_party', the weakest of the
# three, because the safe direction to be wrong in is to understate what we
# know rather than overstate it.
_PROVENANCE_BACKFILL = """
UPDATE market_lines SET provenance = CASE
    WHEN source LIKE 'the-odds-api%%observed%%'      THEN 'observed'
    WHEN source LIKE 'the-odds-api%%reconstructed%%' THEN 'reconstructed'
    WHEN source LIKE 'the-odds-api%%'                THEN 'observed'
    ELSE 'third_party'
END
WHERE provenance IS NULL
"""


def ensure_core_schema(conn: sqlite3.Connection) -> None:
    """Create every core table if absent, and apply additive migrations.

    Safe to call on every run. Where a new column needs a value for rows that
    predate it, it is derived from what those rows actually say about
    themselves -- see _PROVENANCE_BACKFILL -- rather than given a flattering
    default.
    """
    conn.executescript(CORE_SCHEMA)
    migrated = False
    for table, col, decl in _ADDED_COLUMNS:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if have and col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
            logger.info("%s: added column %s", table, col)
            migrated = True
    if migrated or conn.execute(
            "SELECT EXISTS(SELECT 1 FROM market_lines WHERE provenance IS NULL)").fetchone()[0]:
        n = conn.execute(_PROVENANCE_BACKFILL.replace("%%", "%")).rowcount
        if n:
            logger.info("market_lines: backfilled provenance for %d row(s) from source", n)
    conn.commit()


def record_run(conn: sqlite3.Connection, table_name: str, source: str, endpoint: str,
               started_at: str, finished_at: str, rows_written: int,
               notes: str = "") -> None:
    conn.execute(
        "INSERT INTO ingest_runs (table_name, source, source_endpoint, started_at, "
        "finished_at, rows_written, ingest_version, notes) VALUES (?,?,?,?,?,?,?,?)",
        (table_name, source, endpoint, started_at, finished_at, rows_written,
         INGEST_VERSION, notes),
    )
    conn.commit()
