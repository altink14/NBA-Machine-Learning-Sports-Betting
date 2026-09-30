"""ingest_game_summaries.py falls through to the cached v3 summary when v2 is the empty shell.

Since 2025-04-10 boxscoresummaryv2 answers every game with the pre-game
placeholder (status 1, NULL line score, no attendance, "0:00", no inactives).
No network and no production data: the cache is a temp directory of
hand-written summaries and the database is a throwaway file.

Run from the repo root:
    venv/Scripts/python.exe -m unittest Tests.Game_Summaries_V3_Test
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location(
    "ingest_game_summaries_under_test",
    os.path.join(REPO, "src", "Process-Data", "ingest_game_summaries.py"))
ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest)

HOME, AWAY = 1610612763, 1610612743
LS_HEADERS = (["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]
              + [f"PTS_QTR{i}" for i in range(1, 5)] + ingest.OT_COLS + ["PTS"])


def v2(gid, final=True, tv=None):
    """A v2 summary: the real thing, or (final=False) the post-2025-04-10 shell."""
    def ls(tid, abbr, qs, pts):
        return [gid, tid, abbr, *qs, *([0] * 10), pts] if final else [gid, tid, abbr, *([None] * 15)]
    return {"resultSets": [
        {"name": "GameSummary",
         "headers": ["GAME_ID", "GAME_STATUS_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                     "NATL_TV_BROADCASTER_ABBREVIATION"],
         "rowSet": [[gid, 3 if final else 1, HOME, AWAY, tv]]},
        {"name": "GameInfo", "headers": ["GAME_DATE", "ATTENDANCE", "GAME_TIME"],
         "rowSet": [["x", 18000 if final else None, "2:10" if final else "0:00"]]},
        {"name": "LineScore", "headers": LS_HEADERS,
         "rowSet": [ls(HOME, "MEM", [30, 30, 30, 35], 125), ls(AWAY, "DEN", [30, 30, 31, 27], 118)]},
        {"name": "InactivePlayers",
         "headers": ["PLAYER_ID", "FIRST_NAME", "LAST_NAME", "JERSEY_NUM", "TEAM_ID", "TEAM_ABBREVIATION"],
         "rowSet": [[1, "Old", "Listing", "9 ", HOME, "MEM"]] if final else []},
    ]}


def v3(gid, home_periods=(31, 29, 39, 26), home_score=125, status=3, national=("NBC", "Peacock")):
    def team(tid, tri, periods, score, inactive):
        return {"teamId": tid, "teamTricode": tri, "score": score,
                "periods": [{"period": i, "periodType": "REGULAR" if i <= 4 else "OVERTIME", "score": s}
                            for i, s in enumerate(periods, 1)],
                "inactives": [{"personId": inactive, "firstName": "Ja", "familyName": "Morant",
                               "jerseyNum": "12  "}]}
    return {"boxScoreSummary": {
        "gameId": gid, "gameStatus": status, "homeTeamId": HOME, "awayTeamId": AWAY,
        "duration": "2:22", "attendance": 15612,
        "broadcasters": {"nationalBroadcasters": [{"broadcastDisplay": b} for b in national]},
        "homeTeam": team(HOME, "MEM", list(home_periods), home_score, 1629630),
        "awayTeam": team(AWAY, "DEN", [30, 30, 31, 27], 118, 203932),
    }}


class IngestV3FallThroughTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bb_summaries_test_")
        self.cache = os.path.join(self.tmp, "nba_cache")
        os.makedirs(self.cache)
        self.db = os.path.join(self.tmp, "TeamData.sqlite")
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE box_scores (game_id TEXT PRIMARY KEY)")
        # A table from before the source column existed, holding an old shell.
        c.execute("CREATE TABLE game_info (game_id TEXT PRIMARY KEY, attendance INTEGER, game_time TEXT, "
                  "natl_tv TEXT, home_team_id INTEGER, visitor_team_id INTEGER, ingested_at TEXT)")
        c.commit()
        c.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def put(self, gid, v2_doc=None, v3_doc=None):
        c = sqlite3.connect(self.db)
        c.execute("INSERT OR IGNORE INTO box_scores VALUES (?)", (gid,))
        c.commit()
        c.close()
        for prefix, doc in ((ingest.SUMMARY_PREFIX, v2_doc), (ingest.V3_PREFIX, v3_doc)):
            if doc is not None:
                with open(os.path.join(self.cache, f"{prefix}{gid}.json"), "w", encoding="utf-8") as f:
                    json.dump(doc, f)

    def run_ingest(self):
        self.assertEqual(ingest.main(["--db", self.db, "--cache-dir", self.cache]), 0)
        c = sqlite3.connect(self.db)
        self.addCleanup(c.close)
        c.row_factory = sqlite3.Row
        return c

    def test_a_v2_shell_is_filled_from_v3(self):
        self.put("0022500651", v2("0022500651", final=False, tv="Peacock"), v3("0022500651"))
        c = self.run_ingest()
        info = c.execute("SELECT * FROM game_info").fetchone()
        self.assertEqual((info["attendance"], info["game_time"], info["natl_tv"], info["source"]),
                         (15612, "2:22", "NBC/Peacock", "boxscoresummaryv3"))
        self.assertEqual((info["home_team_id"], info["visitor_team_id"]), (HOME, AWAY))
        home = c.execute("SELECT * FROM game_line_scores WHERE team_id = ?", (HOME,)).fetchone()
        self.assertEqual([home[k] for k in ("q1", "q2", "q3", "q4", "ot1", "ot10", "pts")],
                         [31, 29, 39, 26, 0, 0, 125])
        self.assertEqual(sorted((r["player_id"], r["jersey_num"]) for r in c.execute("SELECT * FROM game_inactives")),
                         [(203932, "12"), (1629630, "12")])

    def test_overtime_lands_in_the_ot_columns(self):
        self.put("0022500001", v2("0022500001", final=False),
                 v3("0022500001", home_periods=(30, 30, 30, 28, 9, 12), home_score=139))
        home = self.run_ingest().execute(
            "SELECT ot1, ot2, ot3, pts FROM game_line_scores WHERE team_id = ?", (HOME,)).fetchone()
        self.assertEqual(tuple(home), (9, 12, 0, 139))

    def test_a_real_v2_answer_is_kept_even_with_v3_cached(self):
        self.put("0020700757", v2("0020700757", tv="NBA TV"), v3("0020700757"))
        c = self.run_ingest()
        info = c.execute("SELECT * FROM game_info").fetchone()
        self.assertEqual((info["attendance"], info["game_time"], info["natl_tv"], info["source"]),
                         (18000, "2:10", "NBA TV", "boxscoresummaryv2"))
        self.assertEqual([r["player_id"] for r in c.execute("SELECT * FROM game_inactives")], [1])

    def test_periods_that_do_not_add_up_leave_the_shell(self):
        self.put("0022500002", v2("0022500002", final=False), v3("0022500002", home_score=126))
        c = self.run_ingest()
        info = c.execute("SELECT * FROM game_info").fetchone()
        self.assertEqual((info["attendance"], info["game_time"], info["source"]),
                         (None, "0:00", "boxscoresummaryv2"))
        self.assertIsNone(c.execute("SELECT pts FROM game_line_scores LIMIT 1").fetchone()[0])

    def test_a_v3_that_is_not_final_or_about_another_game_is_not_used(self):
        self.put("0022500003", v2("0022500003", final=False), v3("0022500003", status=2))
        self.put("0022500004", v2("0022500004", final=False), v3("0022500999"))
        c = self.run_ingest()
        self.assertEqual({r[0] for r in c.execute("SELECT source FROM game_info")}, {"boxscoresummaryv2"})

    def test_no_national_tv_in_v3_keeps_the_known_pregame_value(self):
        self.put("0022500005", v2("0022500005", final=False, tv="ESPN"), v3("0022500005", national=()))
        self.put("0022500006", v2("0022500006", final=False, tv="TBD"), v3("0022500006", national=()))
        c = self.run_ingest()
        tv = dict(c.execute("SELECT game_id, natl_tv FROM game_info").fetchall())
        self.assertEqual(tv, {"0022500005": "ESPN", "0022500006": None})

    def test_a_v3_only_game_is_ingested_and_rerunning_changes_nothing(self):
        self.put("0042500401", v3_doc=v3("0042500401"))
        first = [tuple(r) for r in self.run_ingest().execute("SELECT * FROM game_line_scores ORDER BY team_id")]
        c = self.run_ingest()
        self.assertEqual([tuple(r) for r in c.execute("SELECT * FROM game_line_scores ORDER BY team_id")], first)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM game_inactives").fetchone()[0], 2)

    def test_season_reads_only_that_seasons_games(self):
        self.put("0022500651", v2("0022500651", final=False), v3("0022500651"))
        self.put("0022600001", v2("0022600001", final=False), v3("0022600001"))
        c = sqlite3.connect(self.db)
        c.execute("ALTER TABLE box_scores ADD COLUMN season TEXT")
        c.execute("UPDATE box_scores SET season = CASE WHEN game_id LIKE '00225%' THEN '2025-26' "
                  "ELSE '2026-27' END")
        c.commit()
        c.close()
        self.assertEqual(ingest.main(["--db", self.db, "--cache-dir", self.cache, "--season", "2026-27"]), 0)
        c = sqlite3.connect(self.db)
        self.addCleanup(c.close)
        self.assertEqual([r[0] for r in c.execute("SELECT game_id FROM game_info")], ["0022600001"])
        self.assertEqual({r[0] for r in c.execute("SELECT game_id FROM game_line_scores")}, {"0022600001"})

    def test_the_daily_job_runs_right_after_officials_and_reports_coverage(self):
        import refresh_registry as rr
        names = [j.name for j in rr.JOBS]
        self.assertEqual(names.index("game_summaries"), names.index("officials") + 1)
        self.put("0022600001", v2("0022600001", final=False), v3("0022600001"))
        self.put("0022600002", v2("0022600002", final=False), v3("0022600002", home_score=126))
        c = sqlite3.connect(self.db)
        c.execute("ALTER TABLE box_scores ADD COLUMN season TEXT")
        c.execute("UPDATE box_scores SET season = '2026-27'")
        c.commit()
        c.close()
        ingest.main(["--db", self.db, "--cache-dir", self.cache, "--season", "2026-27"])
        self.assertEqual(rr._summary_coverage(self.db, "2026-27"),
                         "2026-27: 1 of 2 games with a line score (1 from v3), 1 empty")

    def test_dry_run_writes_nothing(self):
        self.put("0022500651", v2("0022500651", final=False), v3("0022500651"))
        self.assertEqual(ingest.main(["--db", self.db, "--cache-dir", self.cache, "--dry-run"]), 0)
        c = sqlite3.connect(self.db)
        self.addCleanup(c.close)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM game_info").fetchone()[0], 0)
        self.assertNotIn("source", [r[1] for r in c.execute("PRAGMA table_info(game_info)")])


if __name__ == "__main__":
    unittest.main()
