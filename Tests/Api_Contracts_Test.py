"""The response contracts match what the endpoints really send (2026-09-28).

src/Utils/api_contracts.py describes the JSON of the endpoints the frontend
reads most; the frontend generates its TypeScript from that description, so
a page reading a field the API does not send fails tsc instead of printing
NaN. A description nobody checks drifts, so:

- the shape tests (no database) hold the contracts to the app: every path is
  a real GET route, the OpenAPI document carries every schema, validation is
  strict (unknown key, missing key, wrong type all refused), and the
  prediction-log row lists exactly the table's columns;
- against the real archive when it is present, real responses from every
  contracted endpoint, across old and new seasons, validate with no extra or
  missing key. Rename a field in a handler and this fails until the contract,
  and then the frontend (`npm run gen:api`), move with it.

Nothing here reaches stats.nba.com: the one handler that would (a stale
official career) is mocked, and any non-loopback connection raises.
"""
import os
import socket
import sqlite3
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from pydantic import ValidationError

import main_api
from src.Utils import api_contracts as ac

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DB = os.path.join(HERE, "Data", "TeamData.sqlite")

_real_connect = socket.socket.connect


def _loopback_only(self, addr, *a, **k):
    host = addr[0] if isinstance(addr, tuple) else addr
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise OSError(f"Api_Contracts_Test: no network ({addr})")
    return _real_connect(self, addr, *a, **k)


class ContractShapeTest(unittest.TestCase):

    def test_every_contract_is_a_get_route_of_the_app(self):
        gets = {r.path for r in main_api.app.routes if "GET" in getattr(r, "methods", set())}
        self.assertEqual(sorted(p for p in ac.CONTRACTS if p not in gets), [])

    def test_the_openapi_document_carries_every_schema(self):
        main_api.app.openapi_schema = None
        doc = main_api.app.openapi()
        names = set(doc["components"]["schemas"])
        for path in ac.CONTRACTS:
            schema = doc["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
            self.assertTrue(schema, path)

        def refs(node):
            if isinstance(node, dict):
                if "$ref" in node:
                    yield node["$ref"].rsplit("/", 1)[-1]
                for v in node.values():
                    yield from refs(v)
            elif isinstance(node, list):
                for v in node:
                    yield from refs(v)
        self.assertEqual(sorted(set(refs(doc)) - names), [])

    def test_validation_is_strict(self):
        good = {"player_id": 1, "full_name": "A B", "first_name": "A", "last_name": "B",
                "is_active": 1, "from_year": None, "to_year": 2020}
        ac.validate("/api/players/search", [good])
        for bad in ({**good, "team_abbreviation": "BOS"},              # a key the contract lacks
                    {k: v for k, v in good.items() if k != "to_year"},  # a key gone missing
                    {**good, "is_active": True},                        # bool for int
                    {**good, "player_id": "1"}):                        # str for int
            with self.assertRaises(ValidationError):
                ac.validate("/api/players/search", [bad])

    def test_a_sometimes_absent_key_is_never_null(self):
        day = {"date": "2026-03-10", "prev_date": None, "next_date": None,
               "games": [{"game_id": "1", "game_date": "2026-03-10", "season": "2025-26",
                          "season_type": "Regular Season"}]}
        ac.validate("/api/games/by-date/{game_date}", day)     # a side absent: allowed
        day["games"][0]["home"] = None
        with self.assertRaises(ValidationError):                # a side null: never sent
            ac.validate("/api/games/by-date/{game_date}", day)

    def test_the_prediction_log_row_is_the_tables_columns(self):
        c = sqlite3.connect(":memory:")
        c.executescript(main_api._PREDICTION_LOG_SCHEMA)
        base = {r[1] for r in c.execute("PRAGMA table_info(predictions_log)")} | {"why_json"}
        served = base - {"ou_prediction", "ou_confidence"}    # withdrawn, popped by the handler
        try:
            from src.Utils import nba_clv
            clv = {col for col, _ in nba_clv.CLV_COLUMNS}
        except ImportError:
            clv = set()
        # So do the grade-source columns (grade_predictions.GRADE_COLUMNS, 2026-09-29).
        import grade_predictions
        clv |= {col for col, _ in grade_predictions.GRADE_COLUMNS}
        fields = ac.PredictionLogRow.model_fields
        required = {n for n, f in fields.items() if f.is_required()}
        self.assertEqual(required, served | {"sealed_until_tipoff"})
        # The closing-line columns arrive with the grader's migration: absent until then.
        self.assertTrue(clv <= set(fields) - required, sorted(clv - set(fields)))
        self.assertEqual(set(fields) - required - clv, set() if clv else set(fields) - required)


# Size, not existence: any handler's sqlite3.connect() creates an empty file.
HAVE_ARCHIVE = os.path.exists(REAL_DB) and os.path.getsize(REAL_DB) > 100_000_000


@unittest.skipUnless(HAVE_ARCHIVE, "real archive not present")
class RealResponsesMatchTheContractsTest(unittest.TestCase):
    """Old and new seasons, traded players, a pre-shot-clock-era career."""

    SEASONS = ("2025-26", "2011-12", "1996-97")
    PLAYERS = (2544, 201935, 1628389, 600003)   # LeBron, Harden (traded), Adebayo, Cousy (no 1950-51 minutes)

    @classmethod
    def setUpClass(cls):
        cls._patches = [
            mock.patch.object(socket.socket, "connect", _loopback_only),
            mock.patch.object(main_api, "_ensure_career_official", lambda conn, pid: None),
            # A profile whose bio is not cached asks nba.com; this makes it skip.
            mock.patch.object(main_api, "commonplayerinfo", None),
        ]
        for p in cls._patches:
            p.start()
        cls._limiter_was = getattr(main_api.limiter, "enabled", None)
        if cls._limiter_was is not None:
            main_api.limiter.enabled = False   # 20/minute on two of these routes
        cls.c = TestClient(main_api.app)

    @classmethod
    def tearDownClass(cls):
        if cls._limiter_was is not None:
            main_api.limiter.enabled = cls._limiter_was
        for p in cls._patches:
            p.stop()

    def urls(self):
        out = []
        add = lambda tpl, url: out.append((tpl, url))
        for q in ("lebron", "jordan", "doncic"):
            add("/api/players/search", f"/api/players/search?q={q}")
        for s in self.SEASONS:
            add("/api/stats/standings", f"/api/stats/standings?season={s}")
            add("/api/stats/leaders/board", f"/api/stats/leaders/board?season={s}&categories=pts,reb,fg_pct,fg3_pct,fantasy")
            add("/api/teams/advanced", f"/api/teams/advanced?season={s}")
            add("/api/player-stats", f"/api/player-stats?season={s}")
            add("/api/teams/{abbr}/games", f"/api/teams/BOS/games?season={s}")
            add("/api/teams/{abbr}/roster", f"/api/teams/NOP/roster?season={s}")
        add("/api/stats/standings", "/api/stats/standings?season=2003-04&season_type=Playoffs")
        add("/api/seasons/{year}", "/api/seasons/2002-03")
        add("/api/stats/leaders", "/api/stats/leaders?category=ast&season=2019-20&rank=per_game")
        add("/api/teams/{abbr}/advanced", "/api/teams/OKC/advanced")
        add("/api/teams/{abbr}/advanced", "/api/teams/MEM/advanced?season=2024-25")
        for p in self.PLAYERS:
            add("/api/players/{id}", f"/api/players/{p}")
            add("/api/players/{id}/career", f"/api/players/{p}/career")
            add("/api/players/{id}/career-official", f"/api/players/{p}/career-official")
        add("/api/players/{id}", "/api/players/201935?season=2022-23")   # a traded season, combined
        add("/api/players/by-slug/{slug}", "/api/players/by-slug/lebron-james-2544?season=2011-12")
        add("/api/players/{id}/game-log", "/api/players/2544/game-log?season=2024-25")
        add("/api/players/{id}/game-log", "/api/players/1628389/game-log?season=2024-25&season_type=Playoffs")
        for d in ("2026-03-10", "1997-01-15", "2025-07-04"):
            add("/api/games/by-date/{game_date}", f"/api/games/by-date/{d}")
        for g in ("0022500938", "0042400401"):
            add("/api/games/{game_id}", f"/api/games/{g}")
        for y in (2025, 1996, 1985):
            add("/api/draft/{year}", f"/api/draft/{y}")
        add("/api/awards/winners", "/api/awards/winners?award=mvp")
        add("/api/awards/winners", "/api/awards/winners?award=all-nba&season=2023-24")
        add("/api/season-history", "/api/season-history")
        add("/api/prediction-log", "/api/prediction-log?days=3650")
        return out

    def test_real_responses_validate_against_their_contracts(self):
        failures, seen = [], set()
        for tpl, url in self.urls():
            r = self.c.get(url)
            if r.status_code != 200:
                failures.append(f"{url}: HTTP {r.status_code}")
                continue
            seen.add(tpl)
            try:
                ac.validate(tpl, r.json())
            except ValidationError as e:
                errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])
                failures.append(f"{url}: {errs}")
        self.assertEqual(failures, [])
        self.assertEqual(sorted(set(ac.CONTRACTS) - seen), [], "every contract exercised")

    def test_game_log_columns_the_contract_calls_non_null_have_no_nulls(self):
        # PlayerGameLogRow declares every box-score column non-null; a sample
        # cannot prove that for 787k rows, the table can.
        cols = [n for n, f in ac.PlayerGameLogRow.model_fields.items()
                if f.is_required() and n not in ("season", "season_type", "team_abbr", "opp_abbr", "is_home")]
        con = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
        try:
            have = {r[1] for r in con.execute("PRAGMA table_info(player_game_log)")}
            where = " OR ".join(f"{c} IS NULL" for c in cols if c in have)
            n = con.execute(f"SELECT COUNT(*) FROM player_game_log WHERE {where}").fetchone()[0]
        finally:
            con.close()
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
