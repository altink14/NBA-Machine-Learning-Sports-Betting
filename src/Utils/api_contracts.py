"""Response contracts for the endpoints the frontend reads most (2026-09-28).

WHY THIS EXISTS. The audits kept finding the same bug: a page reading a field
the API does not send (`games` where the row has `gp`, `team_abbreviation`
where it has `team_abbr`). JavaScript turns the missing field into
`undefined`, the page prints NaN or falls back to a plausible default, and
nothing fails. Almost every endpoint here returns an untyped dict, so the
OpenAPI schema said nothing about the JSON and the frontend could not check
itself against it.

These models describe the JSON those endpoints ALREADY return. They are
attached to the OpenAPI document only (`install`); the handlers and their
serialisation are untouched, so a response is byte-for-byte what it was
before (verified with TestClient on 268 responses when this was added).
The frontend generates TypeScript from the document (`npm run gen:api`, file
`src/types/api.generated.ts`), so a field renamed here, or read there
without existing, is a tsc error instead of a NaN.

A model that is only documentation drifts, so Tests/Api_Contracts_Test.py
runs real responses through these models with extra="forbid" and strict
types: a key added, renamed or removed in a handler fails the suite until
the contract (and then the frontend) is updated with it.

Conventions: a key that is always present but can be null is
`Optional[X]` with no default (required, nullable). A key that is sometimes
ABSENT, never null, is `X = None` (TypeScript: `key?: X`; the default is
never validated, so strict validation still refuses an explicit null). Unknown is null, never
0: a statistic is nullable wherever the handler or the source can leave it
unknown. Identity fields the archive always fills (names, abbreviations,
dates, a team's conference) are declared non-null; the test checks that
against every season, so if one ever arrives null the suite says so.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyObject(Contract):
    """`{}`: what a handler returns where it has nothing (no row for that season)."""


# --- Players -----------------------------------------------------------------

class PlayerSearchHit(Contract):
    player_id: int
    full_name: str
    first_name: str
    last_name: str
    is_active: int
    from_year: Optional[int]
    to_year: Optional[int]


class PlayerBio(Contract):
    """The profile header. Unknown text fields are the string "N/A"."""
    fullName: str
    position: str
    heightWeight: str
    team: str
    born: str
    college: str
    experience: str
    jersey: str
    country: str
    draft_year: Optional[int]
    draft_round: Optional[int]
    draft_number: Optional[int]
    active: bool
    instagram: None
    nicknames: None


class _SeasonCounting(Contract):
    gp: int
    gs: Optional[int]
    min: float
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    reb: int
    ast: int
    stl: int
    blk: int
    tov: int
    pf: int
    pts: int
    fg_pct: Optional[float]
    fg3_pct: Optional[float]
    ft_pct: Optional[float]


class PlayerSeasonLine(_SeasonCounting):
    """A player's whole regular season (all stints summed when he was traded).

    `id` is the stored row's id, present only for a one-team season; a
    combined line has no row and team_id is null, team_abbr "MIA/PHX" style.
    """
    id: int = None   # absent (not null) on a combined line
    player_id: int
    season: str
    season_type: str
    team_id: Optional[int]
    team_abbr: str


class PlayerGameLogTotals(Contract):
    """/api/players/{id} totals when no season-totals row exists yet but game
    logs do (a season in progress before its totals are built). A DIFFERENT
    shape from PlayerSeasonLine: `games`, not `gp`, and per-game averages."""
    games: int
    pts: int
    ast: int
    reb: int
    min: float
    pts_per_game: float
    ast_per_game: float
    reb_per_game: float


class PlayerSeasonAdvancedLine(Contract):
    """Usage, ratings and pace are nba.com's whole-season figures; null when
    nba.com never supplied them. TS%, eFG%, TOV% come from the totals."""
    id: int = None   # absent (not null) on a combined line
    player_id: int
    season: str
    season_type: str
    team_id: Optional[int]
    ts_pct: Optional[float]
    usg_pct: Optional[float]
    off_rating: Optional[float]
    def_rating: Optional[float]
    net_rating: Optional[float]
    ast_pct: Optional[float]
    reb_pct: Optional[float]
    efg_pct: Optional[float]
    tov_pct: Optional[float]
    pace: Optional[float]


class PlayerProfile(Contract):
    id: int
    player_id: int
    full_name: str
    first_name: str
    last_name: str
    is_active: int
    bio: PlayerBio
    totals: Union[PlayerSeasonLine, PlayerGameLogTotals, EmptyObject]
    advanced: Union[PlayerSeasonAdvancedLine, EmptyObject]


class PlayerGameLogRow(Contract):
    id: int
    game_id: str
    player_id: int
    team_id: int
    game_date: str
    season: str
    season_type: str
    team_abbr: str
    opp_abbr: str
    is_home: int
    starter: int
    min: float
    fgm: int
    fga: int
    fg_pct: float
    fg3m: int
    fg3a: int
    fg3_pct: float
    ftm: int
    fta: int
    ft_pct: float
    oreb: int
    dreb: int
    reb: int
    ast: int
    stl: int
    blk: int
    tov: int
    pf: int
    pts: int
    plus_minus: float


class CareerSeasonRow(_SeasonCounting):
    """One team-season: a one-team season, or one stint of a traded season
    (whose usage, ratings and pace are null: nba.com reports them per season)."""
    id: int
    player_id: int
    season: str
    season_type: str
    team_id: int
    team_abbr: str
    ts_pct: Optional[float]
    efg_pct: Optional[float]
    tov_pct: Optional[float]
    usg_pct: Optional[float]
    off_rating: Optional[float]
    def_rating: Optional[float]
    net_rating: Optional[float]
    ast_pct: Optional[float]
    reb_pct: Optional[float]
    pace: Optional[float]


class CareerTotalRow(_SeasonCounting):
    """The TOT row of a traded season, listed before its stints."""
    player_id: int
    season: str
    season_type: str
    team_id: None
    team_abbr: Literal["TOT"]
    teams: Optional[str]
    is_total: Literal[True]
    ts_pct: Optional[float]
    efg_pct: Optional[float]
    tov_pct: Optional[float]
    usg_pct: Optional[float]
    off_rating: Optional[float]
    def_rating: Optional[float]
    net_rating: Optional[float]
    ast_pct: Optional[float]
    reb_pct: Optional[float]
    pace: Optional[float]


CareerRow = Union[CareerTotalRow, CareerSeasonRow]


class CareerOfficialRow(Contract):
    """nba.com's official career table. Counting stats the league did not keep
    in a player's era (steals and blocks before 1973-74, turnovers before
    1977-78, starts, threes before 1979-80) are null."""
    player_id: int
    season: str
    season_type: str
    team_abbr: str
    player_age: Optional[float]
    gp: Optional[int]
    gs: Optional[int]
    min: Optional[float]
    fgm: Optional[int]
    fga: Optional[int]
    fg_pct: Optional[float]
    fg3m: Optional[int]
    fg3a: Optional[int]
    fg3_pct: Optional[float]
    ftm: Optional[int]
    fta: Optional[int]
    ft_pct: Optional[float]
    oreb: Optional[int]
    dreb: Optional[int]
    reb: Optional[int]
    ast: Optional[int]
    stl: Optional[int]
    blk: Optional[int]
    tov: Optional[int]
    pf: Optional[int]
    pts: Optional[int]
    is_career_total: int
    fetched_at: Optional[str]


class CareerOfficial(Contract):
    player_id: int
    seasons: List[CareerOfficialRow]
    career_totals: Dict[str, CareerOfficialRow]


class PlayerStatsRow(Contract):
    """/api/player-stats from the archive (player_season_stats: nba.com's
    per-game dashboard, one row per player-season). `gs` has never been
    filled (null on every row). `player_name` is absent when the player is
    not in the directory.

    A season with no archived rows returns an empty list (since 2026-09-28;
    it used to fetch nba.com live and answer in a different shape)."""
    player_id: int
    season: str
    season_type: str
    team_id: int
    team_abbr: str
    player_name: str = None   # absent when the player is not in the directory
    gp: int
    gs: Optional[int]
    min: float
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    pf: float
    fgm: float
    fga: float
    fg_pct: float
    fg3m: float
    fg3a: float
    fg3_pct: float
    ftm: float
    fta: float
    ft_pct: float
    oreb: float
    dreb: float
    plus_minus: float
    ts_pct: float
    usg_pct: float
    off_rating: float
    def_rating: float
    net_rating: float
    ast_pct: float
    reb_pct: float
    efg_pct: float
    tov_pct: float
    pace: float


# --- Leaders -----------------------------------------------------------------

class LeaderMinMakes(Contract):
    fg_pct: int
    fg3_pct: int
    ft_pct: int


class LeaderRules(Contract):
    team_games: int
    min_games: int
    min_makes: LeaderMinMakes


class LeaderRow(_SeasonCounting):
    """One player's season (stints summed; team_abbr "MIA/PHX" when traded,
    team_id the lowest of his teams' ids)."""
    player_id: int
    team_id: int
    full_name: str
    team_abbr: str
    fantasy: float


class LeaderBoard(Contract):
    season: str
    season_type: str
    rank: Literal["per_game", "total"]
    rules: LeaderRules
    boards: Dict[str, List[LeaderRow]]


# --- Teams -------------------------------------------------------------------

class TeamSeasonAdvanced(Contract):
    """team_season_advanced: ratings from Dean Oliver estimated possessions
    (1-3 points off nba.com's official figures)."""
    id: int
    team_id: int
    season: str
    season_type: str
    games: int
    wins: int
    losses: int
    win_pct: float
    pace: float
    off_rating: float
    def_rating: float
    net_rating: float
    efg_pct: float
    tov_pct: float
    orb_pct: float
    ft_rate: float
    ts_pct: float
    srs: float
    sos: float
    computed_at: Optional[str]


class LeagueTeamAdvanced(TeamSeasonAdvanced):
    full_name: str
    abbreviation: str
    conference: str
    division: str


class TeamAdvancedSeason(TeamSeasonAdvanced):
    team_name: str
    abbreviation: str


class StandingsRow(LeagueTeamAdvanced):
    """conference is the team's conference THAT season; division is null before
    2004-05 and for a team whose conference that season differs from today's.
    srs clips margins at +/-srs_margin_cap (capped_games, srs_uncapped say
    what that did). postseason is null until the season's bracket is known."""
    division: Optional[str]
    srs_margin_cap: float
    capped_games: Optional[int]
    srs_uncapped: Optional[float]
    postseason_known: bool
    postseason: Optional[Literal["playoffs", "play_in_to_playoffs", "play_in", "none"]]


class TeamGameRow(Contract):
    game_id: str
    game_date: str
    season: str
    season_type: str
    pts: int
    opp_pts: int
    opp_abbr: str
    opp_name: str
    is_home: int
    wl: Literal["W", "L"]


class RosterRow(Contract):
    """A player's totals for THIS team only (a traded player appears on both)."""
    player_id: int
    full_name: str
    first_name: str
    last_name: str
    gp: int
    min: float
    pts: int
    reb: int
    ast: int
    jersey: Optional[str]
    position: Optional[str]


# --- Games -------------------------------------------------------------------

class DateGameSide(Contract):
    team_id: int
    name: str
    abbr: str
    pts: int


class DateGame(Contract):
    """A side is ABSENT (not null) when the archive holds only the other
    team's row for the game."""
    game_id: str
    game_date: str
    season: str
    season_type: str
    home: DateGameSide = None
    away: DateGameSide = None


class GamesByDate(Contract):
    date: str
    games: List[DateGame]
    prev_date: Optional[str]
    next_date: Optional[str]


class GameTeam(Contract):
    name: str
    abbreviation: str
    score: Optional[int]


class GameDetail(Contract):
    game_id: str
    game_date: str
    season: str
    season_type: str
    home_team_id: int
    away_team_id: int
    home_team: GameTeam
    away_team: GameTeam
    status: Literal["Final", "Scheduled"]


# --- Draft, awards, seasons ------------------------------------------------------

class DraftPick(Contract):
    """organization is the common school name (UCLA); organization_official is
    nba.com's spelling. position..current_team come from player_bio, filled
    for recent classes only: null means not on file, not "none"."""
    person_id: int
    player_name: str
    season: int
    round_number: int
    round_pick: int
    overall_pick: int
    team_id: int
    team_city: str
    team_name: str
    team_abbreviation: str
    organization: Optional[str]
    organization_official: Optional[str]
    organization_type: str
    fetched_at: Optional[str]
    in_database: int
    position: Optional[str]
    height: Optional[str]
    weight: Optional[str]
    country: Optional[str]
    current_team: Optional[str]
    moved_to: Optional[str]


class DraftClass(Contract):
    year: int
    count: int
    rounds: List[int]
    bios_available: int
    current_team_known: int
    current_team_unknown: int
    moved_count: int
    picks: List[DraftPick]


class ReferenceCoverage(Contract):
    players_checked: int
    players_total: int
    complete: bool
    note: Optional[str]


class AwardWinner(Contract):
    season: str
    player_id: int
    full_name: Optional[str]
    team: Optional[str]
    team_number: Optional[str]


class AwardWinners(Contract):
    award: str
    season: Optional[str]
    winners: List[AwardWinner]
    coverage: ReferenceCoverage
    source: str


class SeasonTeamRef(Contract):
    """abbr/name as the league game log recorded them THAT season (SEA,
    Washington Bullets); link_abbr is today's abbreviation, for team URLs."""
    team_id: int
    abbr: str
    name: str
    link_abbr: Optional[str]


class SeriesTeam(SeasonTeamRef):
    wins: int


class BestRecordTeam(SeasonTeamRef):
    wins: int
    losses: int


class GamesPerTeam(Contract):
    min: Optional[int]
    max: Optional[int]


class SeasonMvp(Contract):
    player_id: int
    name: Optional[str]
    team: Optional[str]


class SeasonSummary(Contract):
    """champion/runner_up/finals_result are null until the final is decided."""
    season: str
    champion: Optional[SeriesTeam]
    runner_up: Optional[SeriesTeam]
    finals_result: Optional[str]
    playoffs_complete: bool
    best_record: List[BestRecordTeam]
    games_per_team: GamesPerTeam
    teams: int
    mvp: Optional[SeasonMvp]
    notes: List[str]


class SeasonHistory(Contract):
    seasons: List[SeasonSummary]
    source: str


# --- Track record ------------------------------------------------------------

class PredictionLogRow(Contract):
    """A logged pre-game prediction. Before tip-off (sealed_until_tipoff) the
    pick, confidence, EVs and reasons are null. The over/under pick columns
    are written but never served (withdrawn 2026-09-19); ou_line is the
    market's total, a fact.

    The handler serves `SELECT *`, so the closing-line (CLV) columns appear
    only once the grader has added them to the table (src/Utils/nba_clv.py
    CLV_COLUMNS): each is ABSENT before that migration and null until the
    game is priced. Tests/Api_Contracts_Test.py holds this list to the
    table's own column sources."""
    id: int
    logged_at: str
    log_date: str
    sport: str
    sportsbook: str
    game_key: str
    home_team: str
    away_team: str
    game_start_time_utc: str
    home_ml: Optional[float]
    away_ml: Optional[float]
    ou_line: Optional[float]
    predicted_winner: Optional[str]
    winner_confidence: Optional[float]
    ev_home: Optional[float]
    ev_away: Optional[float]
    model: Optional[str]
    actual_winner: Optional[str]
    actual_total: Optional[float]
    why_json: Optional[str]
    sealed_until_tipoff: bool
    closing_home_ml: Optional[float] = None
    closing_away_ml: Optional[float] = None
    closing_captured_at: Optional[str] = None
    closing_minutes_before_tip: Optional[float] = None
    closing_provenance: Optional[str] = None
    clv: Optional[float] = None
    closing_confirmed_at: Optional[str] = None
    closing_confirmed_by: Optional[str] = None
    clv_prob: Optional[float] = None
    clv_status: Optional[str] = None
    consensus_close_prob: Optional[float] = None
    consensus_books: Optional[int] = None
    consensus_provenance: Optional[str] = None
    clv_consensus_prob: Optional[float] = None
    clv_consensus_price: Optional[float] = None
    clv_consensus_status: Optional[str] = None
    clv_method: Optional[str] = None
    clv_settled_at: Optional[str] = None


class PredictionLogSummary(Contract):
    graded: int
    moneyline_correct: int
    moneyline_pct: Optional[float]


class PredictionLog(Contract):
    days: int
    count: int
    summary: PredictionLogSummary
    predictions: List[PredictionLogRow]


# --- The map the OpenAPI document and the tests read ---------------------------

#: GET path (as FastAPI declares it) -> the type of its 200 response.
CONTRACTS: Dict[str, Any] = {
    "/api/players/search": List[PlayerSearchHit],
    "/api/players/{id}": PlayerProfile,
    "/api/players/by-slug/{slug}": PlayerProfile,          # same handler
    "/api/players/{id}/game-log": List[PlayerGameLogRow],
    "/api/players/{id}/career": List[CareerRow],
    "/api/players/{id}/career-official": CareerOfficial,
    "/api/player-stats": List[PlayerStatsRow],
    "/api/stats/leaders/board": LeaderBoard,
    "/api/stats/leaders": List[LeaderRow],                  # one board, same rows
    "/api/stats/standings": List[StandingsRow],
    "/api/seasons/{year}": List[StandingsRow],              # same handler
    "/api/teams/advanced": List[LeagueTeamAdvanced],
    "/api/teams/{abbr}/advanced": List[TeamAdvancedSeason],
    "/api/teams/{abbr}/games": List[TeamGameRow],
    "/api/teams/{abbr}/roster": List[RosterRow],
    "/api/games/by-date/{game_date}": GamesByDate,
    "/api/games/{game_id}": GameDetail,
    "/api/draft/{year}": DraftClass,
    "/api/awards/winners": AwardWinners,
    "/api/season-history": SeasonHistory,
    "/api/prediction-log": PredictionLog,
}

_ADAPTERS: Dict[str, TypeAdapter] = {}


def adapter(path: str) -> TypeAdapter:
    if path not in _ADAPTERS:
        _ADAPTERS[path] = TypeAdapter(CONTRACTS[path])
    return _ADAPTERS[path]


def validate(path: str, payload: Any) -> None:
    """Raise pydantic.ValidationError unless `payload` matches the contract
    exactly: no unknown keys, no missing ones, no int where a str belongs."""
    adapter(path).validate_python(payload, strict=True)


def install(app) -> None:
    """Attach CONTRACTS to the app's OpenAPI document. Only the document
    changes: responses are serialised by the handlers exactly as before."""
    from fastapi.openapi.utils import get_openapi

    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, openapi_version=app.openapi_version,
                             description=app.description, routes=app.routes)
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for path, tp in CONTRACTS.items():
            op = schema.get("paths", {}).get(path, {}).get("get")
            if op is None:
                raise RuntimeError(f"api_contracts: {path} is not a GET route on this app")
            js = adapter(path).json_schema(ref_template="#/components/schemas/{model}")
            defs = js.pop("$defs", {})
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                # A top-level model comes back inline; name it like the rest.
                defs[tp.__name__], js = js, {"$ref": f"#/components/schemas/{tp.__name__}"}
            for name, definition in defs.items():
                if name in components and components[name] != definition:
                    raise RuntimeError(f"api_contracts: schema name {name} collides with an existing one")
                components[name] = definition
            ok = op.setdefault("responses", {}).setdefault("200", {"description": "Successful Response"})
            ok.setdefault("content", {}).setdefault("application/json", {})["schema"] = js
        app.openapi_schema = schema
        return schema

    app.openapi = openapi
