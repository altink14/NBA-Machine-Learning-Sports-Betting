"""Season History: champion and bracket from the league game log (2026-09-28).

The Season History index was 30 identical "ARCHIVED" cards and the season
page promised "playoff results" it did not have. src/Utils/season_history.py
builds each bracket from game_results. These tests pin the rules that make
it trustworthy: rounds from the order of opponents (pre-2001-02 playoff ids
carry no round), a pre-2002-03 first round is best-of-five, an unfinished
final names no champion, and - against the real archive when it is present -
all 30 champions agree with nba.com's own "NBA Champion" award rows.
"""
import os
import sqlite3
import unittest

from src.Utils import season_history as sh

REAL_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "TeamData.sqlite")


def _db():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE game_results (game_id TEXT, team_id INTEGER, season TEXT, season_type TEXT, "
              "game_date TEXT, team_abbr TEXT, team_name TEXT, matchup TEXT, wl TEXT, pts INTEGER)")
    return c


_seq = [0]


def _game(c, season, date, home, away, home_won, pts=(100, 90)):
    """One playoff game, both sides, with a sequence id like pre-2001 ids."""
    _seq[0] += 1
    gid = f"004{_seq[0]:07d}"
    hp, ap = pts if home_won else (pts[1], pts[0])
    for me, opp, is_home, won, p in ((home, away, True, home_won, hp), (away, home, False, not home_won, ap)):
        c.execute("INSERT INTO game_results VALUES (?,?,?,?,?,?,?,?,?,?)", (
            gid, me, season, "Playoffs", date, f"T{me}", f"Team {me}",
            f"T{me} vs. T{opp}" if is_home else f"T{me} @ T{opp}", "W" if won else "L", p))


def _series(c, season, day, a, b, a_wins, b_wins):
    """a and b play; a is home in every game (enough for these tests)."""
    d = day
    for _ in range(a_wins):
        _game(c, season, f"1999-{d // 28 + 4:02d}-{d % 28 + 1:02d}", a, b, True)
        d += 1
    for _ in range(b_wins):
        _game(c, season, f"1999-{d // 28 + 4:02d}-{d % 28 + 1:02d}", a, b, False)
        d += 1
    return d


def _bracket(c, season, r1_wins=(3, 2), final=(4, 1)):
    """A 16-team bracket: 1 beats 16, 2 beats 15, ... winners pair off by seed."""
    day = 0
    teams = list(range(1, 17))
    r1 = [(teams[i], teams[15 - i]) for i in range(8)]
    for a, b in r1:
        day = max(day, _series(c, season, 0, a, b, *r1_wins))
    alive = [a for a, _ in r1]
    rounds = [(4, 1), (4, 2)]
    start = day
    for wins in rounds:
        pairs = [(alive[i], alive[len(alive) - 1 - i]) for i in range(len(alive) // 2)]
        day = start
        for a, b in pairs:
            day = max(day, _series(c, season, start, a, b, *wins))
        start = day
        alive = [a for a, _ in pairs]
    a, b = alive
    _series(c, season, start, a, b, *final)
    return a, b


class SeasonHistoryTest(unittest.TestCase):

    def test_champion_and_rounds_from_opponent_order(self):
        c = _db()
        self.addCleanup(c.close)
        champ, runner = _bracket(c, "1998-99")
        b = sh.playoff_bracket(c, "1998-99")
        self.assertEqual(b["series_per_round"], {"1": 8, "2": 4, "3": 2, "4": 1})
        self.assertTrue(b["bracket_complete"])
        self.assertEqual(b["champion"]["team_id"], champ)
        self.assertEqual(b["runner_up"]["team_id"], runner)
        self.assertEqual(b["finals_result"], "4-1")

    def test_pre_2002_first_round_is_best_of_five(self):
        c = _db()
        self.addCleanup(c.close)
        _bracket(c, "1998-99", r1_wins=(3, 2))
        b = sh.playoff_bracket(c, "1998-99")
        r1 = [s for s in b["series"] if s["round"] == 1]
        self.assertTrue(all(s["best_of"] == 5 and s["complete"] and s["result"] == "3-2" for s in r1))
        later = [s for s in b["series"] if s["round"] > 1]
        self.assertTrue(all(s["best_of"] == 7 for s in later))

    def test_from_2002_03_three_wins_in_round_one_is_unfinished(self):
        c = _db()
        self.addCleanup(c.close)
        _bracket(c, "2002-03", r1_wins=(3, 2))
        b = sh.playoff_bracket(c, "2002-03")
        self.assertFalse(b["bracket_complete"])
        self.assertIsNone(b["champion"])

    def test_unfinished_final_names_no_champion(self):
        c = _db()
        self.addCleanup(c.close)
        _bracket(c, "1998-99", final=(3, 2))
        b = sh.playoff_bracket(c, "1998-99")
        final = next(s for s in b["series"] if s["round"] == 4)
        self.assertFalse(final["complete"])
        self.assertIsNone(b["champion"])
        self.assertIsNone(b["finals_result"])

    def test_games_carry_scores_and_home_team(self):
        c = _db()
        self.addCleanup(c.close)
        _game(c, "1998-99", "1999-05-01", 1, 2, True, pts=(101, 99))
        g = sh.playoff_bracket(c, "1998-99")["series"][0]["games"][0]
        self.assertEqual((g["home_team_id"], g["winner_team_id"], g["winner_pts"], g["loser_pts"]), (1, 1, 101, 99))

    def test_short_season_note_uses_games_per_team(self):
        shape = {"teams": 29, "games_per_team_min": 50, "games_per_team_max": 50}
        self.assertEqual(sh.season_notes("1998-99", shape, []),
                         ["A 50-game regular season, shortened by the lockout."])
        self.assertEqual(sh.season_notes("2005-06", {"teams": 30, "games_per_team_min": 82,
                                                     "games_per_team_max": 82}, []), [])

    @unittest.skipUnless(os.path.exists(REAL_DB), "archive database not present")
    def test_every_archived_champion_matches_nba_award_rows(self):
        c = sqlite3.connect(REAL_DB)
        c.row_factory = sqlite3.Row
        self.addCleanup(c.close)
        award = {r[0]: r[1] for r in c.execute(
            "SELECT season, team FROM player_awards WHERE description = 'NBA Champion' GROUP BY season, team")}
        seasons = [r[0] for r in c.execute(
            "SELECT DISTINCT season FROM team_season_advanced WHERE season_type = 'Regular Season'")]
        checked = 0
        for s in seasons:
            if s not in award:
                continue
            b = sh.playoff_bracket(c, s)
            self.assertTrue(b["bracket_complete"], s)
            self.assertEqual(b["champion"]["name"], award[s], s)
            checked += 1
        self.assertGreaterEqual(checked, 30)


if __name__ == "__main__":
    unittest.main()
