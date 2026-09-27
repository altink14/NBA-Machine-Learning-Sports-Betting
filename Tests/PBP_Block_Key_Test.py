"""pbp_events keeps both rows of a split play (2026-09-27 fix).

playbyplayv3 reports a blocked shot as two rows sharing one actionNumber
(the MISS, then the BLOCK) and a steal the same way (the Turnover, then the
STEAL). Keyed on (game_id, action_number) with INSERT OR REPLACE, the second
row erased the first: 86,077 blocked misses and 137,078 stolen turnovers
were missing, so every PBP FG% read high. The key is now the feed's
actionId.
"""
import sqlite3
import unittest
from unittest import mock

import backfill_pbp as bp


def _act(action_id, number, atype, desc, clock="PT11M08.00S", fg=0, result="", person=1, **kw):
    a = {"actionId": action_id, "actionNumber": number, "clock": clock, "period": 1,
         "teamId": 10, "teamTricode": "AAA", "personId": person, "playerName": "X",
         "actionType": atype, "subType": "", "description": desc, "xLegacy": None,
         "yLegacy": None, "shotDistance": None, "shotValue": 2 if fg else 0,
         "shotResult": result, "isFieldGoal": fg, "scoreHome": "", "scoreAway": ""}
    a.update(kw)
    return a


# A miss and its block share actionNumber 11; a turnover and its steal share
# 51; the made shot at id 5 was entered late, so its actionNumber (361) is
# higher than the rebound that follows it (129).
ACTIONS = [
    _act(1, 11, "Missed Shot", "MISS Gardner 1' Driving Reverse Dunk Shot", fg=1, result="Missed"),
    _act(2, 11, "", "Sarr BLOCK (1 BLK)", person=2),
    _act(3, 51, "Turnover", "Carrington Bad Pass Turnover (P2.T3)", clock="PT07M33.00S"),
    _act(4, 51, "", "Larsson STEAL (1 STL)", clock="PT07M33.00S", person=2),
    _act(5, 361, "Made Shot", "Smith 2' Driving Layup (2 PTS)", clock="PT02M38.00S", fg=1,
         result="Made", scoreHome="2", scoreAway="0"),
    _act(6, 129, "Rebound", "Gardner REBOUND (Off:1 Def:0)", clock="PT02M28.00S"),
]


def _db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(bp.SCHEMA)
    return conn


class SplitPlayKeyTest(unittest.TestCase):

    def test_block_and_miss_both_survive(self):
        conn = _db()
        self.addCleanup(conn.close)
        bp.write_game(conn, bp.rows_for("G1", ACTIONS))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM pbp_events").fetchone()[0], 6)
        fga = conn.execute("SELECT COUNT(*) FROM pbp_events WHERE is_field_goal = 1").fetchone()[0]
        self.assertEqual(fga, 2, "the blocked miss must count as an attempt")
        descs = [r[0] for r in conn.execute(
            "SELECT description FROM pbp_events WHERE action_number = 11 ORDER BY action_id")]
        self.assertEqual(descs, ["MISS Gardner 1' Driving Reverse Dunk Shot", "Sarr BLOCK (1 BLK)"])

    def test_turnover_and_steal_both_survive(self):
        conn = _db()
        self.addCleanup(conn.close)
        bp.write_game(conn, bp.rows_for("G1", ACTIONS))
        types = [r[0] for r in conn.execute(
            "SELECT action_type FROM pbp_events WHERE action_number = 51 ORDER BY action_id")]
        self.assertEqual(types, ["Turnover", ""])

    def test_action_id_is_the_feed_order(self):
        conn = _db()
        self.addCleanup(conn.close)
        bp.write_game(conn, bp.rows_for("G1", ACTIONS))
        by_id = [r[0] for r in conn.execute("SELECT action_type FROM pbp_events ORDER BY action_id")]
        self.assertEqual(by_id, [a["actionType"] for a in ACTIONS])

    def test_missing_action_id_falls_back_to_position_for_the_whole_game(self):
        acts = [dict(a) for a in ACTIONS]
        del acts[3]["actionId"]
        ids = [r[2] for r in bp.rows_for("G1", acts)]
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6])

    def test_a_real_collision_fails_the_game_and_writes_nothing(self):
        conn = _db()
        self.addCleanup(conn.close)
        acts = ACTIONS + [_act(6, 999, "Foul", "dupe id")]
        with self.assertRaises(sqlite3.IntegrityError):
            bp.write_game(conn, bp.rows_for("G1", acts))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM pbp_events").fetchone()[0], 0)

    def test_old_key_is_detected(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        self.assertTrue(bp.has_action_id(conn), "no table yet is fine")
        conn.execute("CREATE TABLE pbp_events (game_id TEXT, action_number INTEGER, "
                     "PRIMARY KEY (game_id, action_number))")
        self.assertFalse(bp.has_action_id(conn))


class RebuildFromCacheTest(unittest.TestCase):
    """--rebuild re-derives held games from the disk cache and keeps the old table."""

    def _old_table(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE pbp_events (game_id TEXT NOT NULL, action_number INTEGER NOT NULL, "
                     "description TEXT, PRIMARY KEY (game_id, action_number))")
        conn.execute("CREATE INDEX idx_pbp_game ON pbp_events(game_id)")
        conn.execute("INSERT INTO pbp_events VALUES ('G1', 11, 'Sarr BLOCK (1 BLK)')")
        conn.commit()
        self.addCleanup(conn.close)
        return conn

    def _patch_cache(self, payloads):
        from src.Utils import nba_stats_client as c
        return mock.patch.multiple(
            c,
            _cache_path=lambda ep, params: params["game_id"],
            cache_candidates=lambda key: [(key, None)] if key in payloads else [],
            load_cache_file=lambda key: {"game": {"actions": payloads[key]}},
        )

    def test_rebuild_swaps_in_every_action_and_keeps_the_old_table(self):
        conn = self._old_table()
        with self._patch_cache({"G1": ACTIONS}):
            self.assertEqual(bp.rebuild_from_cache(conn), 0)
        self.assertTrue(bp.has_action_id(conn))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM pbp_events").fetchone()[0], 6)
        backups = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pbp_events_pre_%'")]
        self.assertEqual(len(backups), 1)
        self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {backups[0]}").fetchone()[0], 1)
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='pbp_events'")}
        self.assertTrue({"idx_pbp_game", "idx_pbp_person", "idx_pbp_type"} <= idx)

    def test_an_uncached_game_stops_before_changing_anything(self):
        conn = self._old_table()
        with self._patch_cache({}):
            self.assertEqual(bp.rebuild_from_cache(conn), 1)
        self.assertFalse(bp.has_action_id(conn))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM pbp_events").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
