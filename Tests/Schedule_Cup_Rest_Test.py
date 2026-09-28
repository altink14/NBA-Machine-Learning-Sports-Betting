"""Schedule, NBA Cup and Schedule Spots fixes (fix round 2, 2026-09-27).

- Cup knockout slots arrive with null teams; the schedule printed them as
  "None None at None None" at "12:00 AM".
- The team dropdown offered the London Lions (a preseason opponent) in the
  regular-season view, and broadcaster counts mixed preseason in.
- The "80 games a team / 1,206 games" counts are explained from the feed.
- The schedule season was a literal '2026-27' in three places bump_season.py
  does not move; it is now derived from CURRENT_SEASON.
- The Cup header typed "one knockout round"; the structure is now computed.
- Schedule Spots took rest days from team_game_advanced.game_date, which keeps
  a postponed game's ORIGINAL date; game_results has the date it was played.

The feed is mocked (no nba.com). The rest test uses a temp database, plus one
READ-ONLY check against the archive's settled 2025-26 season.
"""
import json
import sqlite3
import types
import unittest
from datetime import date
from unittest import mock

import main_api


class _Client:
    client = None

    @classmethod
    def get(cls, path, **params):
        if cls.client is None:
            from fastapi.testclient import TestClient
            cls.client = TestClient(main_api.app)
        return cls.client.get(path, params=params)


def _team(tri, city, name, tid):
    return {"teamId": tid, "teamTricode": tri, "teamCity": city, "teamName": name}


_TBD = {"teamId": 0, "teamName": None, "teamCity": None, "teamTricode": None}
_PRIME = {"nationalBroadcasters": [{"broadcasterMedia": "tv", "broadcasterDisplay": "Prime Video"}]}


def _game(gid, est, home, away, sub="", label="", subtype="", time="1900-01-01T19:00:00Z",
          status="7:00 pm ET", tv=None, arena="Arena"):
    return {"gameId": gid, "gameDateTimeEst": est, "gameTimeEst": time, "gameStatusText": status,
            "gameLabel": label, "gameSubLabel": sub, "gameSubtype": subtype,
            "homeTeam": home, "awayTeam": away, "broadcasters": tv or {}, "arenaName": arena}


BOS = _team("BOS", "Boston", "Celtics", 1)
NYK = _team("NYK", "New York", "Knicks", 2)
MIA = _team("MIA", "Miami", "Heat", 3)
LON = _team("LON", "London", "Lions", 99)

# A miniature season: three regular-season games with teams (ids 1-3), one Cup
# quarterfinal slot with no teams (id 5, so id 4 is "not scheduled yet"), a
# preseason game against the London Lions, and the Cup final (006).
FEED = {"gameDates": [
    {"gameDate": "10/03/2026", "games": [
        _game("0012600001", "2026-10-03T19:00:00Z", BOS, LON, tv=_PRIME)]},
    {"gameDate": "10/21/2026", "games": [
        _game("0022600001", "2026-10-21T19:00:00Z", BOS, NYK, tv=_PRIME),
        _game("0022600002", "2026-10-21T19:30:00Z", NYK, MIA, sub="East Group B",
              label="Emirates NBA Cup", subtype="in-season")]},
    {"gameDate": "10/22/2026", "games": [
        _game("0022600003", "2026-10-22T19:00:00Z", MIA, BOS)]},
    {"gameDate": "12/04/2026", "games": [
        _game("0022600005", "2026-12-04T00:00:00Z", _TBD, _TBD, sub="Quarterfinal",
              label="Emirates NBA Cup", subtype="in-season-knockout",
              time="0001-01-01T00:00:00Z", status="TBD", tv=_PRIME, arena="")]},
    {"gameDate": "12/11/2026", "games": [
        _game("0062600001", "2026-12-11T00:00:00Z", _TBD, _TBD, sub="Championship",
              label="Emirates NBA Cup", subtype="in-season-knockout",
              time="0001-01-01T00:00:00Z", status="TBD", tv=_PRIME, arena="Hinkle Fieldhouse")]},
]}


def _patched():
    fake = types.SimpleNamespace(schedule_league_v2=lambda season: FEED)
    return mock.patch("src.Utils.nba_stats_client.get_client", lambda *a, **k: fake)


class TestScheduleFeed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with _patched():
            r = _Client.get("/api/schedule", season="2026-27", season_type="Regular Season")
        assert r.status_code == 200, r.text
        cls.body = r.json()

    def test_knockout_slot_never_prints_none(self):
        self.assertNotIn("None", json.dumps(self.body))
        slot = [g for d in self.body["dates"] for g in d["games"] if g["game_id"] == "0022600005"][0]
        self.assertTrue(slot["teams_tbd"])
        self.assertTrue(slot["time_tbd"])
        self.assertEqual(slot["sub_label"], "Quarterfinal")
        self.assertIsNone(slot["home"]["name"])
        self.assertIsNone(slot["home"]["team_id"])
        self.assertIsNone(slot["arena"])

    def test_a_named_game_keeps_its_names_and_time(self):
        g = [g for d in self.body["dates"] for g in d["games"] if g["game_id"] == "0022600001"][0]
        self.assertEqual(g["home"]["name"], "Boston Celtics")
        self.assertFalse(g["time_tbd"])
        self.assertFalse(g["teams_tbd"])

    def test_regular_season_dropdown_has_no_preseason_opponent(self):
        tris = [t["tricode"] for t in self.body["options"]["teams"]]
        self.assertEqual(tris, ["BOS", "MIA", "NYK"])

    def test_broadcaster_counts_are_scoped_to_the_view(self):
        # Prime Video: one regular-season game and the knockout slot; the
        # preseason game and the 006 final are not in this view.
        self.assertEqual(self.body["options"]["broadcasters"], [{"name": "Prime Video", "games": 2}])

    def test_regular_season_summary_explains_the_counts(self):
        rs = self.body["regular_season"]
        self.assertEqual(rs["games"], 4)
        self.assertEqual(rs["games_with_teams"], 3)
        self.assertEqual([k["stage"] for k in rs["knockout_slots_tbd"]], ["Quarterfinal"])
        self.assertEqual((rs["per_team_min"], rs["per_team_max"]), (2, 2))
        self.assertEqual(rs["per_team_full_season"], 82)
        self.assertEqual(rs["unscheduled_game_ids"], 1)
        self.assertEqual(rs["unscheduled_id_range"], [4, 4])

    def test_preseason_view_still_lists_its_opponent(self):
        with _patched():
            r = _Client.get("/api/schedule", season="2026-27", season_type="Preseason")
        self.assertIn("LON", [t["tricode"] for t in r.json()["options"]["teams"]])


class TestScheduleSeason(unittest.TestCase):

    def test_summer_and_preseason_show_the_next_season(self):
        with mock.patch.object(main_api, "CURRENT_SEASON", "2025-26"):
            self.assertEqual(main_api._schedule_season(date(2026, 9, 27)), "2026-27")
            self.assertEqual(main_api._schedule_season(date(2026, 7, 1)), "2026-27")
            self.assertEqual(main_api._schedule_season(date(2026, 6, 30)), "2025-26")

    def test_after_the_opening_night_bump_it_stays_put(self):
        with mock.patch.object(main_api, "CURRENT_SEASON", "2026-27"):
            self.assertEqual(main_api._schedule_season(date(2026, 10, 21)), "2026-27")
            self.assertEqual(main_api._schedule_season(date(2027, 4, 12)), "2026-27")
            self.assertEqual(main_api._schedule_season(date(2027, 8, 20)), "2027-28")

    def test_no_literal_season_default_left_on_the_schedule_routes(self):
        import inspect
        for fn in (main_api.get_league_schedule, main_api.get_nba_cup):
            self.assertIsNone(inspect.signature(fn).parameters["season"].default)


class TestCupStructure(unittest.TestCase):

    def test_structure_is_counted_from_the_fixtures(self):
        with _patched():
            r = _Client.get("/api/cup", season="2026-27")
        self.assertEqual(r.status_code, 200, r.text)
        st = r.json()["structure"]
        self.assertEqual([x["stage"] for x in st["knockout_rounds"]], ["Quarterfinal", "Championship"])
        self.assertEqual(st["groups"], 1)
        self.assertEqual(st["knockout_games_in_standings"], 1)   # the 002 slot, not the 006 final
        ko = r.json()["knockout"]
        self.assertEqual([k["counts_in_standings"] for k in ko], [True, False])
        self.assertTrue(all(k["time_tbd"] for k in ko))


class _KeepOpen:
    def __init__(self, conn):
        self._c = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


class TestRestDatesFromGameResults(unittest.TestCase):

    def _db(self):
        c = sqlite3.connect(":memory:", check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.executescript("""
            CREATE TABLE team_game_advanced (team_id INT, opp_team_id INT, game_id TEXT, game_date TEXT,
                                             pts INT, opp_pts INT, season TEXT, season_type TEXT);
            CREATE TABLE team_metadata (team_id INT, abbreviation TEXT, full_name TEXT);
            CREATE TABLE box_scores (game_id TEXT, home_team_id INT);
            CREATE TABLE game_results (game_id TEXT, team_id INT, game_date TEXT);
            INSERT INTO team_metadata VALUES (1, 'AAA', 'Team A'), (2, 'BBB', 'Team B');
        """)
        # Three games. Game 2 was postponed: team_game_advanced keeps its
        # original date (Jan 2, a back-to-back after Jan 1), game_results has
        # the date it was played (Jan 10).
        games = [("g1", "2026-01-01", "2026-01-01"), ("g2", "2026-01-02", "2026-01-10"),
                 ("g3", "2026-01-05", "2026-01-05")]
        for gid, adv_date, played in games:
            c.execute("INSERT INTO box_scores VALUES (?, 1)", (gid,))
            for tid, opp in ((1, 2), (2, 1)):
                c.execute("INSERT INTO team_game_advanced VALUES (?,?,?,?,?,?, '2025-26', 'Regular Season')",
                          (tid, opp, gid, adv_date, 100 + tid, 100 + opp))
                c.execute("INSERT INTO game_results VALUES (?,?,?)", (gid, tid, played))
        return c

    def test_postponed_game_counts_on_its_make_up_date(self):
        c = self._db()
        rows = main_api._load_rest_rows(c, "2025-26", "Regular Season")
        a = [(d["game_id"], d["game_date"], d["rest_days"]) for d in rows if d["team_id"] == 1]
        # Played order: g1 (Jan 1), g3 (Jan 5, 3 days off), g2 (Jan 10, 4 days off).
        # From team_game_advanced it would have read g2 as a back-to-back.
        self.assertEqual(a, [("g1", "2026-01-01", None), ("g3", "2026-01-05", 3), ("g2", "2026-01-10", 4)])

    def test_endpoint_reports_team_games_and_the_computed_b2b_cost(self):
        c = self._db()
        with mock.patch.object(main_api, "get_db_conn", lambda: _KeepOpen(c)):
            body = _Client.get("/api/stats/rest", season="2025-26", season_type="Regular Season").json()
        self.assertEqual((body["games"], body["team_games"], body["team_games_with_rest"]), (3, 6, 4))
        self.assertEqual(body["b2b_effect"]["b2b_games"], 0)
        self.assertIsNone(body["b2b_effect"]["margin_cost"])


class TestRestArchive2025_26(unittest.TestCase):
    """Settled history, read-only: recounted from game_results by hand
    (441 back-to-backs, 213-228; the old team_game_advanced dates gave 432)."""

    def test_back_to_backs_match_the_game_log(self):
        body = _Client.get("/api/stats/rest", season="2025-26", season_type="Regular Season").json()
        b2b = next(r for r in body["by_rest"] if r["label"] == "Back-to-back")
        self.assertEqual((b2b["games"], b2b["wins"], b2b["losses"]), (441, 213, 228))
        self.assertEqual(body["team_games_with_rest"], sum(r["games"] for r in body["by_rest"]))
        self.assertEqual(body["b2b_effect"]["margin_cost"], 1.44)


if __name__ == "__main__":
    unittest.main()
