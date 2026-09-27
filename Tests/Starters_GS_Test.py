"""Games started come from the starters, not from "has a position".

Found 2026-09-27 (nav audit bug 17): before 2017-18 nba.com's box score gives
every player who played his roster position, and the pipeline read a
position as a start, so GS was a copy of GP (2009-10: 22,243 league starts
instead of 12,300; Harden's rookie year 76 GS instead of 0). nba.com lists
each team's five starters first; that is the rule now, in the ingest and in
repair_starters.py. Temp databases only.
"""
import json
import sqlite3
import unittest

import repair_starters
from src.Utils import nba_pipeline


def _player(pid, minutes="20:00", position="G"):
    return {"personId": pid, "firstName": "P", "familyName": str(pid), "position": position,
            "statistics": {"minutes": minutes, "points": 2}}


def _box(home_players, away_players, home=1, away=2):
    return json.dumps({"boxScoreTraditional": {
        "homeTeam": {"teamId": home, "players": home_players},
        "awayTeam": {"teamId": away, "players": away_players},
    }})


class FirstFiveTest(unittest.TestCase):
    def test_first_five_listed_are_the_starters_even_when_everyone_has_a_position(self):
        home = [_player(i) for i in range(1, 10)]          # 2009-10 style: all positions filled
        away = [_player(i) for i in range(11, 19)]
        got = repair_starters.first_five(_box(home, away))
        self.assertEqual(got[1], {1, 2, 3, 4, 5})
        self.assertEqual(got[2], {11, 12, 13, 14, 15})

    def test_a_first_five_that_did_not_all_play_is_unknown_not_guessed(self):
        home = [_player(1), _player(2), _player(3), _player(4), _player(5, minutes="")] + [_player(6)]
        got = repair_starters.first_five(_box(home, [_player(i) for i in range(11, 17)]))
        self.assertIsNone(got[1])
        self.assertEqual(got[2], {11, 12, 13, 14, 15})

    def test_gs_is_unknown_when_any_game_is(self):
        self.assertEqual(repair_starters._gs([1, 0, 1]), 2)
        self.assertIsNone(repair_starters._gs([1, None, 1]))


class IngestRuleTest(unittest.TestCase):
    def _conn(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, full_name TEXT, first_name TEXT, "
                  "last_name TEXT, is_active INTEGER)")
        c.execute("""CREATE TABLE player_game_log (id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, player_id INTEGER,
                     team_id INTEGER, game_date TEXT, min REAL, fgm INTEGER, fga INTEGER, fg_pct REAL, fg3m INTEGER,
                     fg3a INTEGER, fg3_pct REAL, ftm INTEGER, fta INTEGER, ft_pct REAL, oreb INTEGER, dreb INTEGER,
                     reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER, tov INTEGER, pf INTEGER, pts INTEGER,
                     plus_minus REAL, starter INTEGER DEFAULT 0, UNIQUE(game_id, player_id))""")
        return c

    def test_bench_players_with_a_position_are_not_starters(self):
        c = self._conn()
        dnp = {"personId": 99, "firstName": "P", "familyName": "99", "position": "",
               "statistics": {"minutes": "", "points": 0}}
        players = [_player(i) for i in range(1, 9)] + [dnp]   # DNP last, skipped
        nba_pipeline.save_players_and_game_log(c, "0020900001", "2009-10-27", players, 1)
        rows = dict(c.execute("SELECT player_id, starter FROM player_game_log").fetchall())
        self.assertEqual(sum(rows.values()), 5)
        self.assertEqual({p for p, s in rows.items() if s}, {1, 2, 3, 4, 5})
        self.assertNotIn(99, rows)


class RepairPlanTest(unittest.TestCase):
    def test_plan_turns_gp_copies_into_real_starts(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE box_scores (game_id TEXT, season TEXT, season_type TEXT, home_team_id INTEGER, "
                  "traditional_json TEXT)")
        c.execute("CREATE TABLE player_game_log (id INTEGER PRIMARY KEY, game_id TEXT, player_id INTEGER, "
                  "team_id INTEGER, game_date TEXT, starter INTEGER)")
        c.execute("CREATE TABLE team_game_advanced (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT, "
                  "pts INTEGER, opp_pts INTEGER)")
        c.execute("CREATE TABLE player_season_totals (id INTEGER PRIMARY KEY, player_id INTEGER, season TEXT, "
                  "season_type TEXT, team_id INTEGER, gp INTEGER, gs INTEGER)")
        c.execute("CREATE TABLE player_splits (id INTEGER PRIMARY KEY, player_id INTEGER, season TEXT, "
                  "season_type TEXT, split_type TEXT, split_value TEXT, gp INTEGER, gs INTEGER)")
        home = [_player(i) for i in range(1, 8)]
        away = [_player(i) for i in range(11, 17)]
        c.execute("INSERT INTO box_scores VALUES ('G1', '2009-10', 'Regular Season', 1, ?)", (_box(home, away),))
        c.execute("INSERT INTO team_game_advanced VALUES ('G1', 1, '2009-10', 'Regular Season', 100, 90)")
        c.execute("INSERT INTO team_game_advanced VALUES ('G1', 2, '2009-10', 'Regular Season', 90, 100)")
        log_id = 0
        for pid in range(1, 8):
            log_id += 1
            c.execute("INSERT INTO player_game_log VALUES (?, 'G1', ?, 1, '2009-10-27', 1)", (log_id, pid))
            c.execute("INSERT INTO player_season_totals VALUES (?, ?, '2009-10', 'Regular Season', 1, 1, 1)",
                      (pid, pid))
            c.execute("INSERT INTO player_splits VALUES (?, ?, '2009-10', 'Regular Season', 'Location', 'Home', 1, 1)",
                      (pid, pid))
        p = repair_starters.plan(c)
        # players 6 and 7 came off the bench: their flag, GS and split GS drop to 0
        self.assertEqual(sorted(c.execute("SELECT player_id FROM player_game_log WHERE id IN (%s)"
                                          % ",".join(map(str, p["flag_updates"]))).fetchall()), [(6,), (7,)])
        self.assertEqual(sorted((rid, new) for rid, _, new in p["total_updates"]), [(6, 0), (7, 0)])
        self.assertEqual(sorted((rid, new) for rid, _, new in p["split_updates"]), [(6, 0), (7, 0)])
        self.assertEqual(p["unknown_team_games"], 0)


if __name__ == "__main__":
    unittest.main()
