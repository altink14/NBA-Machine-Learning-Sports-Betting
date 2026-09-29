"""
ESPN box scores: the fallback source for NEW games when stats.nba.com cannot
be asked.

WHY (2026-09-28)
nba.com's edge blocked this PC after a 12,000-request backfill, and the
outbound guard (nba_outbound_guard.py) keeps every stats.nba.com request
refused while the breaker is open. Every input of the served model is built
from the box-score archive (candidate_live: rolling form, Elo, rest) or from
a team-stats snapshot that team_stats_from_archive.py can rebuild from that
archive. So the picks only depend on nba.com through one thing: new games
landing in the archive. This module is the second way in. ESPN's public site
API answers from this PC.

WHAT IT STORES, AND WHERE
Never in `box_scores`. That table is nba.com's own JSON, and half the site
parses its advanced box score (possessions, ratings, player tracking), none
of which ESPN publishes. ESPN games go in their own tables, every row carries
source = 'espn', and a reader has to ask for them by name:

  espn_box_scores  one row per ESPN event: the nba.com game id it maps to,
                   date, teams, score, whether it counts, and the trimmed raw
                   payload (gzip JSON) so it can be re-parsed later.
  espn_team_box    per team: the counting stats the model reads.
  espn_player_box  per player, for the record (nothing reads it yet).

The model's readers (candidate_live, team_stats_from_archive, the days-rest
lookup) take ESPN rows ONLY for games that are not in box_scores. When
nba.com answers again and the backfill stores the real box score, the nba.com
row wins and the ESPN row is simply shadowed; nothing is deleted.

WHAT ESPN CAN AND CANNOT REPRODUCE (measured 2026-09-28 against our nba.com
box scores on 146 games of 2024-25 and 2025-26 -- regular season, play-in,
Finals, five overtimes; fixtures in Tests/fixtures/espn)
  - points, FGM/FGA, 3PM/3PA, FTM/FTA, OREB/DREB/REB, AST, STL, BLK and
    personal fouls: ESPN's team totals are the same numbers in all 292
    team-games, and every player's counting stats in all 3,148 player-games.
  - player plus-minus: 2 of 3,148 differ (substitution timing).
  - turnovers: players' turnovers plus ESPN's team turnovers is the team
    total the model uses (nba.com's advanced box score recovers the same
    number). ESPN once sent a negative team count; that is read as zero.
  - team minutes: ESPN has no team clock; 240 + 25 per overtime is used. That
    is what nba.com reports for every regulation game; the ~0.1% of nba.com
    games carrying odd clocks (e.g. '239:55') cannot be reproduced.
  - player minutes: ESPN gives whole minutes only.

WHICH GAMES COUNT
A regular-season game counts only when it maps to an nba.com game id whose
prefix is 002 (a regular-season game) through the archive, the league game
log or the mirrored nba.com schedule. That rule is what keeps the NBA Cup
final (ESPN files it as regular season; nba.com's standings and team stats do
not count it) and any exhibition out of the model's inputs. Play-in and
playoff games count even when no id is known yet (the mirrored schedule
predates the bracket); they are keyed 'espn:<event id>' until nba.com's id
appears.

POLITENESS
Only site.api.espn.com is ever contacted, at least MIN_GAP_SECONDS apart,
and a run stops after MAX_REQUESTS_PER_RUN. A summary already stored is never
fetched again.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SOURCE = "espn"
PARSER_VERSION = 1

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
ALLOWED_HOSTS = frozenset({"site.api.espn.com", "site.web.api.espn.com"})

MIN_GAP_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUESTS_PER_RUN = 150

#: ESPN's abbreviation where it differs from nba.com's tricode.
ESPN_ABBR_TO_NBA = {"NY": "NYK", "GS": "GSW", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}

#: ESPN season types (header.season.type / event.season.type).
ESPN_SEASON_TYPES = {1: "Preseason", 2: "Regular Season", 3: "Playoffs", 4: "All-Star", 5: "PlayIn"}

#: nba.com game-id prefixes (characters 1-3) and what they are.
NBA_ID_SEASON_TYPES = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}

#: ESPN game notes that mark a game nba.com does not count in team stats.
_EXCLUDED_NOTES = ("nba cup championship", "all-star")

_SPLIT_STATS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("FGM", "FGA"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("FG3M", "FG3A"),
    "freeThrowsMade-freeThrowsAttempted": ("FTM", "FTA"),
}
_PLAIN_TEAM_STATS = {
    "offensiveRebounds": "OREB", "defensiveRebounds": "DREB", "totalRebounds": "REB",
    "assists": "AST", "steals": "STL", "blocks": "BLK",
    "turnovers": "TOV_PLAYERS", "teamTurnovers": "TOV_TEAM_RAW", "fouls": "PF",
}
TEAM_STATS = ("MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
              "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "TOV_PLAYERS", "TOV_TEAM_RAW", "PF")
PLAYER_STATS = ("MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS")
_PLAYER_KEYS = {
    "minutes": "MIN", "points": "PTS", "rebounds": "REB", "assists": "AST", "turnovers": "TOV",
    "steals": "STL", "blocks": "BLK", "offensiveRebounds": "OREB", "defensiveRebounds": "DREB",
    "fouls": "PF", "plusMinus": "PLUS_MINUS",
}

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - no tz database on this platform
    _ET = None


class EspnUnavailable(RuntimeError):
    """ESPN could not be read (network, non-200, budget spent). Never cached."""


class EspnParseError(ValueError):
    """The payload is not a finished game in the shape this parser knows."""


# ---------------------------------------------------------------------------
# HTTP: one polite client
# ---------------------------------------------------------------------------
class EspnClient:
    """Paced, host-restricted GETs with a per-run request ceiling."""

    def __init__(self, max_requests: int = MAX_REQUESTS_PER_RUN,
                 min_gap: float = MIN_GAP_SECONDS,
                 http_get: Optional[Callable[..., Any]] = None):
        self.max_requests = max_requests
        self.min_gap = min_gap
        self.requests_made = 0
        self._last = 0.0
        self._lock = threading.Lock()
        self._scoreboards: Dict[str, Dict] = {}
        if http_get is None:
            import requests
            http_get = requests.get
        self._get = http_get

    def get_json(self, url: str, params: Dict[str, str]) -> Dict:
        host = urlparse(url).hostname
        if host not in ALLOWED_HOSTS:
            raise EspnUnavailable(f"refusing to contact {host}: not an ESPN site API host")
        with self._lock:
            if self.requests_made >= self.max_requests:
                raise EspnUnavailable(f"ESPN request ceiling reached ({self.max_requests} this run)")
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self.requests_made += 1
            try:
                # No custom User-Agent: ESPN answered a descriptive one with
                # HTTP 403 on 2026-09-28 and the library default with 200.
                r = self._get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            except Exception as exc:
                raise EspnUnavailable(f"{url} {params}: {type(exc).__name__}: {exc}") from exc
            finally:
                self._last = time.time()
        if getattr(r, "status_code", None) != 200:
            raise EspnUnavailable(f"{url} {params}: HTTP {getattr(r, 'status_code', None)}")
        try:
            body = r.json()
        except ValueError as exc:
            raise EspnUnavailable(f"{url} {params}: body is not JSON") from exc
        if not isinstance(body, dict):
            raise EspnUnavailable(f"{url} {params}: body is not an object")
        return body

    def scoreboard(self, day: date) -> Dict:
        key = day.strftime("%Y%m%d")
        if key not in self._scoreboards:
            body = self.get_json(SCOREBOARD_URL, {"dates": key, "limit": "50"})
            if not isinstance(body.get("events"), list):
                raise EspnUnavailable(f"ESPN scoreboard {key}: no 'events' list")
            self._scoreboards[key] = body
        return self._scoreboards[key]

    def summary(self, event_id: str) -> Dict:
        return self.get_json(SUMMARY_URL, {"event": str(event_id)})


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------
def et_date(iso_utc: str) -> str:
    """ESPN's UTC timestamp ('2026-01-06T00:00Z') as the US Eastern date,
    which is how nba.com and box_scores date a game."""
    d = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    if _ET is not None:
        return d.astimezone(_ET).date().isoformat()
    # No tz database: 5 hours covers EST; an EDT evening tip is still that date.
    return (d.astimezone(timezone.utc) - timedelta(hours=5)).date().isoformat()


def _int(v: Any) -> int:
    s = str(v).strip()
    if not re.fullmatch(r"[+-]?\d+", s):
        raise EspnParseError(f"not an integer: {v!r}")
    return int(s)


def _split(v: Any) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(v))
    if not m:
        raise EspnParseError(f"not a made-attempted pair: {v!r}")
    return int(m.group(1)), int(m.group(2))


def parse_summary(payload: Dict) -> Dict[str, Any]:
    """A finished ESPN game summary -> plain dict. Raises EspnParseError.

    Returns {event_id, date_utc, game_date, espn_season_year, espn_season_type,
    notes, periods, status, teams: {home: {...}, away: {...}},
    players: [{...}]}, team dicts keyed by TEAM_STATS plus espn_team_id/abbr.
    """
    try:
        header = payload["header"]
        comp = header["competitions"][0]
        box = payload["boxscore"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EspnParseError(f"missing header/boxscore: {exc}") from exc
    status = ((comp.get("status") or {}).get("type") or {})
    if not status.get("completed") or status.get("name") != "STATUS_FINAL":
        raise EspnParseError(f"game is not final ({status.get('name')})")

    competitors = {c.get("homeAway"): c for c in comp.get("competitors") or []}
    if set(competitors) != {"home", "away"}:
        raise EspnParseError("expected one home and one away competitor")
    periods = {len(c.get("linescores") or []) for c in competitors.values()}
    if len(periods) != 1 or min(periods) < 4:
        raise EspnParseError(f"unreadable period count {periods}")
    n_periods = periods.pop()

    by_team_id = {}
    for t in box.get("teams") or []:
        by_team_id[str(t["team"]["id"])] = t
    teams: Dict[str, Dict[str, Any]] = {}
    for side, c in competitors.items():
        tid = str(c["team"]["id"])
        t = by_team_id.get(tid)
        if t is None:
            raise EspnParseError(f"no team box for ESPN team {tid}")
        row: Dict[str, Any] = {"espn_team_id": tid, "abbr": c["team"].get("abbreviation"),
                               "name": c["team"].get("displayName")}
        stats = {s.get("name"): s.get("displayValue") for s in t.get("statistics") or []}
        for key, (made, att) in _SPLIT_STATS.items():
            if key not in stats:
                raise EspnParseError(f"team {tid} has no {key}")
            row[made], row[att] = _split(stats[key])
        for key, col in _PLAIN_TEAM_STATS.items():
            if key not in stats:
                raise EspnParseError(f"team {tid} has no {key}")
            row[col] = _int(stats[key])
        # The team total the model reads: players' turnovers plus team
        # turnovers (shot clock, 8-second...), which is what nba.com's advanced
        # box score recovers. Not ESPN's own `totalTurnovers`: on 2026-04-10
        # (IND-PHI) ESPN carried teamTurnovers = -3, an impossible count, and
        # its total came out 3 short of nba.com's. A negative team count is
        # read as none; the raw value is kept in TOV_TEAM_RAW.
        row["TOV"] = row["TOV_PLAYERS"] + max(row["TOV_TEAM_RAW"], 0)
        row["PTS"] = _int(c.get("score"))
        # Team minutes: ESPN publishes no team clock. Regulation is 5 x 48,
        # each overtime 5 x 5, which is what nba.com reports for all but a
        # handful of games in thirty seasons.
        row["MIN"] = 240 + 25 * (n_periods - 4)
        teams[side] = row

    players: List[Dict[str, Any]] = []
    team_side = {teams[s]["espn_team_id"]: s for s in teams}
    for block in box.get("players") or []:
        tid = str(block["team"]["id"])
        side = team_side.get(tid)
        if side is None:
            raise EspnParseError(f"player block for unknown team {tid}")
        stat_block = (block.get("statistics") or [{}])[0]
        keys = stat_block.get("keys") or []
        totals = stat_block.get("totals") or []
        if totals and "points" in keys and _int(totals[keys.index("points")]) != teams[side]["PTS"]:
            raise EspnParseError(f"team {tid} player points do not add up to its score")
        for a in stat_block.get("athletes") or []:
            ath = a.get("athlete") or {}
            p = {"side": side, "espn_team_id": tid, "espn_athlete_id": str(ath.get("id")),
                 "name": ath.get("displayName"), "starter": bool(a.get("starter")),
                 "did_not_play": bool(a.get("didNotPlay")) or not a.get("stats")}
            vals = a.get("stats") or []
            if vals and len(vals) == len(keys):
                for k, v in zip(keys, vals):
                    if k in _SPLIT_STATS:
                        made, att = _SPLIT_STATS[k]
                        p[made], p[att] = _split(v)
                    elif k in _PLAYER_KEYS:
                        p[_PLAYER_KEYS[k]] = _int(v) if str(v).strip() not in ("", "--") else None
            players.append(p)

    notes = [str(n.get("headline") or "") for n in (comp.get("notes") or [])]
    season = header.get("season") or {}
    return {
        "event_id": str(header.get("id") or comp.get("id")),
        "date_utc": comp.get("date"),
        "game_date": et_date(comp.get("date")),
        "espn_season_year": season.get("year"),
        "espn_season_type": season.get("type"),
        "notes": notes,
        "periods": n_periods,
        "status": status.get("name"),
        "teams": teams,
        "players": players,
    }


def trimmed_payload(payload: Dict) -> bytes:
    """What is kept of a summary: the box score and the header (the rest is
    news, odds and play-by-play), gzipped."""
    keep = {"header": payload.get("header"), "boxscore": payload.get("boxscore")}
    return gzip.compress(json.dumps(keep, separators=(",", ":")).encode("utf-8"))


def season_label_from_espn_year(year: Optional[int]) -> Optional[str]:
    """ESPN names a season by the year it ends: 2026 -> '2025-26'."""
    if not year:
        return None
    return f"{int(year) - 1}-{str(int(year))[-2:]}"


def season_from_nba_id(game_id: str) -> Tuple[Optional[str], Optional[str]]:
    """'0022500502' -> ('2025-26', 'Regular Season'); unknown prefixes -> (None, None)."""
    if not re.fullmatch(r"00\d{8}", game_id or ""):
        return None, None
    yy = int(game_id[3:5])
    start = 1900 + yy if yy >= 46 else 2000 + yy
    return f"{start}-{str(start + 1)[-2:]}", NBA_ID_SEASON_TYPES.get(game_id[:3])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS espn_box_scores (
    espn_event_id  TEXT PRIMARY KEY,
    game_id        TEXT,               -- nba.com game id, or NULL when none is known yet
    id_source      TEXT,               -- where game_id came from: box_scores | game_results | schedule
    model_game_id  TEXT NOT NULL,      -- game_id, else 'espn:<event id>'
    season         TEXT,
    season_type    TEXT,
    game_date      TEXT NOT NULL,      -- US Eastern date, as box_scores dates games
    home_team_id   INTEGER NOT NULL,   -- nba.com team ids
    away_team_id   INTEGER NOT NULL,
    home_pts       INTEGER NOT NULL,
    away_pts       INTEGER NOT NULL,
    periods        INTEGER NOT NULL,
    usable         INTEGER NOT NULL,   -- 1 = may enter model inputs; 0 = held, not used
    exclude_reason TEXT,
    source         TEXT NOT NULL DEFAULT 'espn' CHECK (source = 'espn'),
    source_url     TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,      -- UTC ISO
    parser_version INTEGER NOT NULL,
    payload        BLOB NOT NULL       -- gzip(JSON {header, boxscore}) as ESPN sent it
);
CREATE INDEX IF NOT EXISTS espn_box_scores_date ON espn_box_scores (game_date);
CREATE INDEX IF NOT EXISTS espn_box_scores_game ON espn_box_scores (game_id);
CREATE TABLE IF NOT EXISTS espn_team_box (
    espn_event_id TEXT NOT NULL,
    team_id       INTEGER NOT NULL,
    is_home       INTEGER NOT NULL,
    MIN INTEGER, PTS INTEGER, FGM INTEGER, FGA INTEGER, FG3M INTEGER, FG3A INTEGER,
    FTM INTEGER, FTA INTEGER, OREB INTEGER, DREB INTEGER, REB INTEGER, AST INTEGER,
    STL INTEGER, BLK INTEGER,
    TOV INTEGER,            -- players + team turnovers (a negative team count counts as 0)
    TOV_PLAYERS INTEGER,    -- ESPN turnovers: players only
    TOV_TEAM_RAW INTEGER,   -- ESPN teamTurnovers exactly as sent (it has been negative)
    PF INTEGER,
    source TEXT NOT NULL DEFAULT 'espn' CHECK (source = 'espn'),
    PRIMARY KEY (espn_event_id, team_id)
);
CREATE TABLE IF NOT EXISTS espn_player_box (
    espn_event_id   TEXT NOT NULL,
    team_id         INTEGER NOT NULL,
    espn_athlete_id TEXT NOT NULL,
    player_id       INTEGER,            -- nba.com person id when the name matched one player; else NULL
    player_name     TEXT,
    starter         INTEGER,
    did_not_play    INTEGER,
    MIN INTEGER,                        -- whole minutes: ESPN publishes no seconds
    PTS INTEGER, FGM INTEGER, FGA INTEGER, FG3M INTEGER, FG3A INTEGER, FTM INTEGER, FTA INTEGER,
    OREB INTEGER, DREB INTEGER, REB INTEGER, AST INTEGER, STL INTEGER, BLK INTEGER, TOV INTEGER,
    PF INTEGER, PLUS_MINUS INTEGER,
    source TEXT NOT NULL DEFAULT 'espn' CHECK (source = 'espn'),
    PRIMARY KEY (espn_event_id, espn_athlete_id)
);
"""


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def has_espn_tables(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='espn_box_scores'"
                        ).fetchone() is not None


# ---------------------------------------------------------------------------
# Mapping to nba.com
# ---------------------------------------------------------------------------
def nba_team_ids(conn: sqlite3.Connection) -> Dict[str, int]:
    """nba.com tricode -> team id, from team_metadata."""
    return {str(abbr): int(tid) for tid, abbr in
            conn.execute("SELECT team_id, abbreviation FROM team_metadata WHERE abbreviation IS NOT NULL")}


def espn_to_nba_team(abbr: str, tricodes: Dict[str, int]) -> Optional[int]:
    return tricodes.get(ESPN_ABBR_TO_NBA.get(abbr, abbr))


def _schedule_games(conn: sqlite3.Connection, season: str) -> List[Dict[str, Any]]:
    """Games in the mirrored nba.com schedule (scheduleleaguev2) for a season."""
    try:
        row = conn.execute("SELECT payload FROM nba_response_mirror WHERE endpoint='scheduleleaguev2' "
                           "AND season=? ORDER BY fetched_at DESC LIMIT 1", (season,)).fetchone()
    except sqlite3.OperationalError:
        return []
    if not row:
        return []
    try:
        body = json.loads(gzip.decompress(row[0]))
    except Exception:
        return []
    out = []
    for gd in ((body.get("leagueSchedule") or {}).get("gameDates") or []):
        for g in gd.get("games") or []:
            est = str(g.get("gameDateEst") or "")[:10]
            out.append({"game_id": g.get("gameId"), "date": est,
                        "home": (g.get("homeTeam") or {}).get("teamId"),
                        "away": (g.get("awayTeam") or {}).get("teamId")})
    return out


def map_game_id(conn: sqlite3.Connection, game_date: str, home_id: int, away_id: int,
                season_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """The nba.com game id for (US Eastern date, home, away), and where it was found."""
    r = conn.execute("SELECT game_id FROM box_scores WHERE game_date=? AND home_team_id=? AND away_team_id=?",
                     (game_date, home_id, away_id)).fetchone()
    if r:
        return str(r[0]), "box_scores"
    try:
        rows = conn.execute(
            "SELECT h.game_id FROM game_results h JOIN game_results a ON a.game_id=h.game_id "
            "WHERE h.game_date=? AND h.team_id=? AND a.team_id=? AND h.matchup LIKE '%vs.%'",
            (game_date, home_id, away_id)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if len(rows) == 1:
        return str(rows[0][0]), "game_results"
    seasons = [season_hint] if season_hint else []
    for s in seasons:
        hits = [g for g in _schedule_games(conn, s)
                if g["date"] == game_date and g["home"] == home_id and g["away"] == away_id]
        if len(hits) == 1 and hits[0]["game_id"]:
            return str(hits[0]["game_id"]), "schedule"
    return None, None


def season_for_game_date(game_date: str) -> str:
    """Season label for a US Eastern date: August on opens the next season."""
    y, m = int(game_date[:4]), int(game_date[5:7])
    start = y if m >= 8 else y - 1
    return f"{start}-{str(start + 1)[-2:]}"


def classify(parsed: Dict[str, Any], game_id: Optional[str]) -> Tuple[Optional[str], Optional[str], bool, Optional[str]]:
    """(season, season_type, usable, exclude_reason) for a parsed game."""
    if any(any(x in n.lower() for x in _EXCLUDED_NOTES) for n in parsed.get("notes") or []):
        return (season_label_from_espn_year(parsed.get("espn_season_year")), None, False,
                f"ESPN note {parsed['notes']}: nba.com team stats do not count this game")
    espn_type = ESPN_SEASON_TYPES.get(parsed.get("espn_season_type"))
    if game_id:
        season, stype = season_from_nba_id(game_id)
        if stype is None:
            return season, None, False, f"nba.com id {game_id} is not a regular-season, play-in or playoff game"
        return season, stype, True, None
    season = season_label_from_espn_year(parsed.get("espn_season_year"))
    if espn_type in ("Playoffs", "PlayIn"):
        return season, espn_type, True, None
    return (season, espn_type, False,
            "no nba.com game id for this date and these teams; a regular-season game must map to "
            "nba.com's schedule so that nothing nba.com does not count can enter the standings")


# ---------------------------------------------------------------------------
# Storing
# ---------------------------------------------------------------------------
def _player_ids(conn: sqlite3.Connection) -> Dict[str, List[Tuple[int, Optional[int]]]]:
    from src.Utils.espn_injuries import normalize_name
    out: Dict[str, List[Tuple[int, Optional[int]]]] = {}
    for pid, name, team in conn.execute("SELECT player_id, full_name, last_team_id FROM players"):
        out.setdefault(normalize_name(name), []).append((int(pid), team))
    return out


def _match_player(index, name: str, team_id: int) -> Optional[int]:
    from src.Utils.espn_injuries import normalize_name
    cands = index.get(normalize_name(name or "")) or []
    if len(cands) == 1:
        return cands[0][0]
    on_team = [pid for pid, t in cands if t == team_id]
    return on_team[0] if len(on_team) == 1 else None


def store_game(conn: sqlite3.Connection, payload: Dict, fetched_at: Optional[str] = None,
               tricodes: Optional[Dict[str, int]] = None, player_index=None,
               scoreboard_notes: Iterable[str] = ()) -> Dict[str, Any]:
    """Parse one summary and write it to the three tables, in one transaction.
    Returns the espn_box_scores row as a dict (without the payload).

    `scoreboard_notes`: the event's notes from the day's scoreboard. The
    summary's own header has carried none for the NBA Cup final, while the
    scoreboard says 'NBA Cup Championship', so both are read.
    """
    parsed = parse_summary(payload)
    parsed["notes"] = list(parsed.get("notes") or []) + [n for n in scoreboard_notes if n]
    tricodes = tricodes if tricodes is not None else nba_team_ids(conn)
    ids = {}
    for side in ("home", "away"):
        tid = espn_to_nba_team(parsed["teams"][side]["abbr"], tricodes)
        if tid is None:
            raise EspnParseError(f"ESPN team {parsed['teams'][side]['abbr']!r} has no nba.com team id")
        ids[side] = tid
    hint = season_label_from_espn_year(parsed.get("espn_season_year")) or season_for_game_date(parsed["game_date"])
    game_id, id_source = map_game_id(conn, parsed["game_date"], ids["home"], ids["away"], hint)
    season, stype, usable, reason = classify(parsed, game_id)
    row = dict(
        espn_event_id=parsed["event_id"], game_id=game_id, id_source=id_source,
        model_game_id=game_id or f"espn:{parsed['event_id']}",
        season=season, season_type=stype, game_date=parsed["game_date"],
        home_team_id=ids["home"], away_team_id=ids["away"],
        home_pts=parsed["teams"]["home"]["PTS"], away_pts=parsed["teams"]["away"]["PTS"],
        periods=parsed["periods"], usable=int(usable), exclude_reason=reason, source=SOURCE,
        source_url=f"{SUMMARY_URL}?event={parsed['event_id']}",
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
        parser_version=PARSER_VERSION)
    if player_index is None:
        player_index = _player_ids(conn)
    ensure_tables(conn)
    with conn:
        conn.execute(
            f"INSERT OR REPLACE INTO espn_box_scores ({', '.join(row)}, payload) "
            f"VALUES ({', '.join('?' * len(row))}, ?)", list(row.values()) + [trimmed_payload(payload)])
        conn.execute("DELETE FROM espn_team_box WHERE espn_event_id=?", (row["espn_event_id"],))
        conn.execute("DELETE FROM espn_player_box WHERE espn_event_id=?", (row["espn_event_id"],))
        for side in ("home", "away"):
            t = parsed["teams"][side]
            conn.execute(
                f"INSERT INTO espn_team_box (espn_event_id, team_id, is_home, {', '.join(TEAM_STATS)}) "
                f"VALUES (?, ?, ?, {', '.join('?' * len(TEAM_STATS))})",
                [row["espn_event_id"], ids[side], int(side == "home")] + [t[c] for c in TEAM_STATS])
        for p in parsed["players"]:
            tid = ids[p["side"]]
            conn.execute(
                f"INSERT OR REPLACE INTO espn_player_box (espn_event_id, team_id, espn_athlete_id, player_id, "
                f"player_name, starter, did_not_play, {', '.join(PLAYER_STATS)}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, {', '.join('?' * len(PLAYER_STATS))})",
                [row["espn_event_id"], tid, p["espn_athlete_id"], _match_player(player_index, p["name"], tid),
                 p["name"], int(p["starter"]), int(p["did_not_play"])] + [p.get(c) for c in PLAYER_STATS])
    return row


def ingest_dates(conn: sqlite3.Connection, days: Iterable[date],
                 client: Optional[EspnClient] = None) -> Dict[str, Any]:
    """Fetch and store every FINAL game on these US dates that the archive lacks.

    A game already in box_scores (nba.com) or already stored from ESPN is not
    fetched. Never raises for one bad game or day: each lands in `failed` and
    the rest carry on. Raises EspnUnavailable only if no day could be read at
    all, so a caller cannot mistake "could not look" for "nothing to do".
    """
    client = client or EspnClient()
    ensure_tables(conn)
    tricodes = nba_team_ids(conn)
    players = _player_ids(conn)
    out: Dict[str, Any] = {"days": [], "final_events": 0, "in_box_scores": 0, "already_from_espn": 0,
                           "stored": [], "excluded": [], "failed": [], "not_final": 0}
    days = list(days)
    unreadable = 0
    for day in days:
        try:
            sb = client.scoreboard(day)
        except EspnUnavailable as exc:
            unreadable += 1
            out["failed"].append({"day": day.isoformat(), "error": str(exc)})
            continue
        out["days"].append(day.isoformat())
        for ev in sb.get("events") or []:
            comp = (ev.get("competitions") or [{}])[0]
            st = ((comp.get("status") or {}).get("type") or {})
            if not st.get("completed") or st.get("name") != "STATUS_FINAL":
                out["not_final"] += 1
                continue
            out["final_events"] += 1
            eid = str(ev.get("id"))
            if conn.execute("SELECT 1 FROM espn_box_scores WHERE espn_event_id=?", (eid,)).fetchone():
                out["already_from_espn"] += 1
                continue
            sides = {c.get("homeAway"): espn_to_nba_team((c.get("team") or {}).get("abbreviation"), tricodes)
                     for c in comp.get("competitors") or []}
            gd = et_date(comp.get("date") or ev.get("date"))
            if sides.get("home") and sides.get("away") and conn.execute(
                    "SELECT 1 FROM box_scores WHERE game_date=? AND home_team_id=? AND away_team_id=?",
                    (gd, sides["home"], sides["away"])).fetchone():
                out["in_box_scores"] += 1
                continue
            notes = [str(n.get("headline") or "") for n in comp.get("notes") or []]
            try:
                row = store_game(conn, client.summary(eid), tricodes=tricodes, player_index=players,
                                 scoreboard_notes=notes)
            except (EspnUnavailable, EspnParseError, sqlite3.Error, KeyError, TypeError) as exc:
                out["failed"].append({"event": eid, "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            (out["stored"] if row["usable"] else out["excluded"]).append(
                {k: row[k] for k in ("espn_event_id", "model_game_id", "game_date", "season_type",
                                     "home_team_id", "away_team_id", "exclude_reason")})
    out["requests"] = client.requests_made
    if days and unreadable == len(days):
        raise EspnUnavailable(f"no ESPN scoreboard could be read for {len(days)} day(s): "
                              f"{out['failed'][0]['error'] if out['failed'] else ''}")
    return out


# ---------------------------------------------------------------------------
# Reading: ESPN games as model rows
# ---------------------------------------------------------------------------
#: The ESPN games that stand in for nba.com: usable, and not held by
#: box_scores under either their date and teams or their nba.com id (a
#: postponed game can sit in box_scores under its original date).
_STANDS_IN = (
    "e.usable = 1 "
    "AND NOT EXISTS (SELECT 1 FROM box_scores b WHERE b.game_date = e.game_date "
    "                AND b.home_team_id = e.home_team_id AND b.away_team_id = e.away_team_id) "
    "AND (e.game_id IS NULL OR NOT EXISTS (SELECT 1 FROM box_scores b2 WHERE b2.game_id = e.game_id))")

TEAM_GAME_COLUMNS = ("game_id", "season", "season_type", "game_date", "team_id", "team_name", "is_home",
                     "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST",
                     "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "OPP_PTS")


def espn_team_game_rows(conn: sqlite3.Connection, season: Optional[str] = None):
    """Usable ESPN games that box_scores does not hold, one row per team, in
    backtest_model.load_team_games()'s column shape plus `source`.

    team_name is left None; merge_team_games fills it from the team's own
    nba.com rows so every reader sees one spelling per franchise.
    """
    import pandas as pd
    cols = list(TEAM_GAME_COLUMNS) + ["W", "L", "PLUS_MINUS", "source"]
    if not has_espn_tables(conn):
        return pd.DataFrame(columns=cols)
    q = ("SELECT e.model_game_id, e.season, e.season_type, e.game_date, t.team_id, t.is_home, "
         "t.MIN, t.FGM, t.FGA, t.FG3M, t.FG3A, t.FTM, t.FTA, t.OREB, t.DREB, t.REB, t.AST, t.TOV, "
         "t.STL, t.BLK, o.BLK, t.PF, o.PF, t.PTS, o.PTS "
         "FROM espn_box_scores e "
         "JOIN espn_team_box t ON t.espn_event_id = e.espn_event_id "
         "JOIN espn_team_box o ON o.espn_event_id = e.espn_event_id AND o.team_id <> t.team_id "
         "WHERE " + _STANDS_IN)
    params: List[Any] = []
    if season:
        q += " AND e.season = ?"
        params.append(season)
    rows = conn.execute(q, params).fetchall()
    recs = []
    for r in rows:
        (gid, s, st, gd, tid, home, mn, fgm, fga, f3m, f3a, ftm, fta, oreb, dreb, reb, ast, tov,
         stl, blk, blka, pf, pfd, pts, opp) = r
        recs.append(dict(game_id=gid, season=s, season_type=st, game_date=gd, team_id=int(tid),
                         team_name=None, is_home=bool(home), MIN=float(mn), FGM=fgm, FGA=fga,
                         FG3M=f3m, FG3A=f3a, FTM=ftm, FTA=fta, OREB=oreb, DREB=dreb, REB=reb, AST=ast,
                         TOV=tov, STL=stl, BLK=blk, BLKA=blka, PF=pf, PFD=pfd, PTS=pts, OPP_PTS=opp,
                         W=int(pts > opp), L=int(pts <= opp), PLUS_MINUS=pts - opp, source=SOURCE))
    return pd.DataFrame(recs, columns=cols)


def merge_team_games(conn: sqlite3.Connection, tg, season: Optional[str] = None):
    """nba.com team-game rows (load_team_games' frame) plus usable ESPN rows
    for games it lacks. Returns `tg` itself, untouched, when there are none,
    so a day on which nba.com answered is bit-for-bit what it always was.

    A game is never counted twice: an ESPN game box_scores holds, under its
    date and teams or under its nba.com id, is left out.
    """
    import pandas as pd
    extra = espn_team_game_rows(conn, season)
    if extra.empty:
        return tg
    # espn_team_game_rows already left out every game box_scores holds (by
    # date and teams, and by nba.com id). This drops only an id `tg` has
    # that the database did not (a caller's own frame).
    extra = extra[~extra.game_id.isin(set(tg.game_id))] if len(tg) else extra
    if extra.empty:
        return tg
    # A team cannot play twice on one date. If an ESPN row lands on a date
    # where `tg` already has that team, one of the two dates is wrong, and it
    # has been box_scores': postponed 2025-26 games are stored under their
    # original dates (e.g. 0022501111 MEM-DAL under 2026-04-01, played
    # 2026-03-12). Both rows are real games, so both are kept, and it is said.
    have = set(zip(tg.game_date.astype(str), tg.team_id.astype(int))) if len(tg) else set()
    clash = [(d, int(t)) for d, t in zip(extra.game_date, extra.team_id) if (d, int(t)) in have]
    if clash:
        logger.warning("ESPN game(s) share a date with another game of the same team in box_scores "
                       "(a misdated box score?): %s", clash[:5])
    names = (tg.sort_values(["game_date", "game_id"]).groupby("team_id").team_name.last().to_dict()
             if len(tg) else {})
    if len(tg) == 0 or not set(extra.team_id) <= set(names):
        missing = set(extra.team_id) - set(names)
        names.update({int(t): n for t, n in conn.execute(
            "SELECT team_id, full_name FROM team_metadata") if int(t) in missing})
    extra = extra.copy()
    extra["team_name"] = extra.team_id.map(lambda t: names.get(int(t)))
    base = tg.copy()
    if "source" not in base.columns:
        base["source"] = "nba.com"
    merged = pd.concat([base, extra[base.columns.intersection(extra.columns)]], ignore_index=True)
    logger.warning("Model inputs include %d ESPN-sourced team-game row(s) (%d game(s)) that the "
                   "nba.com archive does not hold yet.", len(extra), extra.game_id.nunique())
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def latest_game_date(conn: sqlite3.Connection) -> Optional[str]:
    """Newest game date across box_scores and the ESPN games standing in for
    nba.com: exactly the rows merge_team_games adds, so a reader that compares
    this with its own newest date never sees a difference it cannot close."""
    a = conn.execute("SELECT MAX(game_date) FROM box_scores").fetchone()[0]
    if not has_espn_tables(conn):
        return a
    b = conn.execute("SELECT MAX(e.game_date) FROM espn_box_scores e WHERE " + _STANDS_IN).fetchone()[0]
    dates = [x for x in (a, b) if x]
    return max(dates) if dates else None


def stand_in_count(conn: sqlite3.Connection) -> int:
    """How many ESPN games are standing in for nba.com right now (0 normally)."""
    if not has_espn_tables(conn):
        return 0
    return int(conn.execute("SELECT COUNT(*) FROM espn_box_scores e WHERE " + _STANDS_IN).fetchone()[0])


def team_date_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """[{'team': full name, 'd': date}] for each team in each ESPN game
    standing in for nba.com: the days-rest lookup's shape (main_api
    PredictionRunner._last_game_dates). Empty on a normal day."""
    if not has_espn_tables(conn):
        return []
    return [{"team": t, "d": d} for t, d in conn.execute(
        "SELECT m.full_name, e.game_date FROM espn_box_scores e "
        "JOIN team_metadata m ON m.team_id = e.home_team_id OR m.team_id = e.away_team_id "
        "WHERE " + _STANDS_IN)]


def recent_sources(conn: sqlite3.Connection, since: str) -> Dict[str, Any]:
    """Which source each game on or after `since` came from (for job_health):
    nba.com box scores, ESPN games standing in for them, ESPN games nba.com
    has since supplied (shadowed), and ESPN games held but not counted."""
    nba = conn.execute("SELECT COUNT(*) FROM box_scores WHERE game_date >= ?", (since,)).fetchone()[0]
    out = {"since": since, "nba_com": int(nba), "espn_only": 0, "espn_shadowed": 0,
           "espn_excluded": 0, "espn_only_games": [], "excluded_games": []}
    if not has_espn_tables(conn):
        return out
    for mgid, gd, usable, stands_in, reason in conn.execute(
            "SELECT e.model_game_id, e.game_date, e.usable, (" + _STANDS_IN + "), e.exclude_reason "
            "FROM espn_box_scores e WHERE e.game_date >= ? ORDER BY e.game_date, e.model_game_id", (since,)):
        if not usable:
            out["espn_excluded"] += 1
            out["excluded_games"].append(f"{gd} {mgid}: {reason}")
        elif stands_in:
            out["espn_only"] += 1
            out["espn_only_games"].append(f"{gd} {mgid}")
        else:
            out["espn_shadowed"] += 1
    return out
