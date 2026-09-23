"""
NBA_Injury_Recorder_Test.py
===========================
Guards src/Sports/nba/poll_injuries.py, the hourly NBA injury recorder.

The rules pinned here are the ones whose failure would not look like a
failure: a dead feed recorded as a quiet hour, a player who came back still
reading as Out, and a broken payload parsed into "nobody is injured". No test
touches the network; the feed is a dict built below.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Sports.nba import poll_injuries as P  # noqa: E402


def _entry(pid, name, status, date="2026-10-20T15:00Z", type_="Knee", ret=None, comment=None):
    return {
        "id": f"inj-{pid}",
        "status": status,
        "date": date,
        "shortComment": comment or f"{name} is {status}.",
        "athlete": {
            "displayName": name,
            "position": {"abbreviation": "G"},
            "links": [{"href": f"https://www.espn.com/nba/player/_/id/{pid}/x"}],
        },
        "details": {"type": type_, "location": "Leg", "detail": "Soreness", "side": "Left",
                    "returnDate": ret},
    }


def _payload(*teams):
    """teams: (display_name, [entries])"""
    return {
        "timestamp": "2026-10-20T16:00:00Z",
        "status": "success",
        "season": {"year": 2027, "displayName": "2026-27"},
        "injuries": [{"id": str(i), "displayName": name, "injuries": list(es)}
                     for i, (name, es) in enumerate(teams)],
    }


TATUM = ("101", "Jayson Tatum")
DONCIC = ("202", "Luka Dončić")
HARDEN = ("303", "James Harden")


def _standard(tatum_status="Out", with_harden=True):
    teams = [("Boston Celtics", [_entry(*TATUM, tatum_status)]),
             ("Los Angeles Lakers", [_entry(*DONCIC, "Day-To-Day")])]
    if with_harden:
        teams.append(("LA Clippers", [_entry(*HARDEN, "Out")]))
    return _payload(*teams)


class _Base(unittest.TestCase):

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="nba_inj_test_"), "team.sqlite")
        self.conn = sqlite3.connect(self.db)
        self.conn.executescript("""
            CREATE TABLE players (player_id INTEGER, full_name TEXT, is_active INTEGER);
            INSERT INTO players VALUES (1628369, 'Jayson Tatum', 1), (1629029, 'Luka Doncic', 1),
                                       (201935, 'James Harden', 1);
            CREATE TABLE team_metadata (full_name TEXT, nickname TEXT, abbreviation TEXT);
            INSERT INTO team_metadata VALUES ('Boston Celtics', 'Celtics', 'BOS'),
                                             ('Los Angeles Lakers', 'Lakers', 'LAL'),
                                             ('Los Angeles Clippers', 'Clippers', 'LAC');
        """)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def poll(self, payload, **kw):
        return P.run(self.conn, fetch=lambda: (payload, 200), **kw)

    def obs(self):
        cur = self.conn.cursor()
        cur.row_factory = sqlite3.Row
        return cur.execute("SELECT * FROM nba_injury_observations ORDER BY id").fetchall()

    def polls(self):
        cur = self.conn.cursor()
        cur.row_factory = sqlite3.Row
        return cur.execute("SELECT * FROM nba_injury_polls ORDER BY id").fetchall()


class TestChangeLog(_Base):

    def test_first_poll_records_everyone_as_new_and_joins_to_our_ids(self):
        res = self.poll(_standard())
        self.assertEqual(res["status"], "ok")
        rows = self.obs()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["change_kind"] for r in rows}, {"new"})
        by = {r["player_name"]: r for r in rows}
        self.assertEqual(by["Jayson Tatum"]["nba_player_id"], 1628369)
        self.assertEqual(by["Jayson Tatum"]["team_abbr"], "BOS")
        # accent folded, and ESPN's 'LA Clippers' resolved to our LAC
        self.assertEqual(by["Luka Dončić"]["nba_player_id"], 1629029)
        self.assertEqual(by["James Harden"]["team_abbr"], "LAC")
        self.assertEqual(by["Jayson Tatum"]["player_id"], "101")   # from the player-card link
        self.assertEqual(by["Jayson Tatum"]["season"], "2026-27")
        self.assertTrue(all(r["poll_id"] == self.polls()[0]["id"] for r in rows))

    def test_an_unchanged_feed_writes_no_observation_but_still_records_the_poll(self):
        self.poll(_standard())
        res = self.poll(_standard())
        self.assertEqual(res["rows_written"], 0)
        self.assertEqual(len(self.obs()), 3)
        polls = self.polls()
        self.assertEqual([p["status"] for p in polls], ["ok", "ok"])
        # the quiet hour is distinguishable from a missing hour
        self.assertEqual(polls[1]["entries_seen"], 3)

    def test_a_status_change_is_recorded_with_what_it_was(self):
        self.poll(_standard("Day-To-Day"))
        self.poll(_standard("Out"))
        last = self.obs()[-1]
        self.assertEqual((last["player_name"], last["change_kind"], last["report_status"],
                          last["prev_status"]), ("Jayson Tatum", "changed", "Out", "Day-To-Day"))

    def test_a_redated_report_is_a_new_observation(self):
        """A reaffirmed 'Out' for a new game is information, not churn."""
        self.poll(_standard())
        p = _standard()
        p["injuries"][0]["injuries"][0]["date"] = "2026-10-22T15:00Z"
        res = self.poll(p)
        self.assertEqual((res["changed"], res["rows_written"]), (1, 1))

    def test_a_retouched_comment_alone_is_not_a_change(self):
        self.poll(_standard())
        p = _standard()
        p["injuries"][0]["injuries"][0]["shortComment"] = "Reworded by an editor."
        self.assertEqual(self.poll(p)["rows_written"], 0)

    def test_a_player_who_leaves_the_report_is_cleared_not_left_as_out(self):
        self.poll(_standard())
        res = self.poll(_standard(with_harden=False))
        self.assertEqual(res["cleared"], 1)
        last = self.obs()[-1]
        self.assertEqual((last["player_name"], last["change_kind"], last["report_status"],
                          last["prev_status"], last["nba_player_id"]),
                         ("James Harden", "cleared", None, "Out", 201935))
        # and only once: the next poll does not clear him again
        self.assertEqual(self.poll(_standard(with_harden=False))["rows_written"], 0)

    def test_a_cleared_player_who_returns_to_the_report_is_new_again(self):
        self.poll(_standard())
        self.poll(_standard(with_harden=False))
        res = self.poll(_standard())
        self.assertEqual(res["new"], 1)
        last = self.obs()[-1]
        self.assertEqual((last["player_name"], last["change_kind"], last["prev_status"]),
                         ("James Harden", "new", None))

    def test_snapshot_mode_writes_every_row(self):
        self.poll(_standard())
        res = self.poll(_standard(), snapshot=True)
        self.assertEqual(res["rows_written"], 3)
        self.assertEqual({r["change_kind"] for r in self.obs()[3:]}, {"snapshot"})

    def test_an_entry_without_an_athlete_id_is_keyed_by_name(self):
        p = _payload(("Boston Celtics", [_entry(*TATUM, "Out")]))
        p["injuries"][0]["injuries"][0]["athlete"]["links"] = []
        self.poll(p)
        self.assertEqual(self.obs()[0]["player_key"], "name:jayson tatum")
        self.assertIsNone(self.obs()[0]["player_id"])


class TestFailuresAreFailures(_Base):

    def assertFailedPoll(self, res, needle):
        self.assertEqual(res["status"], "failed")
        polls = self.polls()
        self.assertEqual(polls[-1]["status"], "failed")
        self.assertIn(needle, polls[-1]["error"])

    def test_a_dead_feed_is_a_failed_poll_and_clears_nobody(self):
        self.poll(_standard())
        n = len(self.obs())

        def boom():
            raise P.FeedError("HTTP 503 from espn", http_status=503)

        res = P.run(self.conn, fetch=boom)
        self.assertFailedPoll(res, "503")
        self.assertEqual(self.polls()[-1]["http_status"], 503)
        self.assertEqual(len(self.obs()), n, "a feed we did not read must not write anything")

    def test_a_payload_whose_shape_changed_is_a_failure_not_an_empty_report(self):
        self.assertFailedPoll(self.poll({"timestamp": "x", "teams": []}), "shape")
        self.assertEqual(self.obs(), [])

    def test_a_feed_that_says_it_failed_is_believed(self):
        p = _standard()
        p["status"] = "error"
        self.assertFailedPoll(self.poll(p), "status")

    def test_a_non_object_payload_is_a_failure(self):
        self.assertFailedPoll(self.poll(["not", "a", "dict"]), "not an object")

    def test_an_unexpected_parser_error_is_still_recorded(self):
        p = _standard()
        p["injuries"][0]["injuries"][0]["athlete"] = "garbage"
        res = self.poll(p)
        self.assertFailedPoll(res, "unexpected")

    def test_an_empty_feed_after_a_full_one_is_refused(self):
        self.poll(_standard())
        res = self.poll(_payload())
        self.assertFailedPoll(res, "zero entries")
        self.assertEqual(len(self.obs()), 3, "nobody may be cleared on a suspect empty feed")

    def test_allow_empty_accepts_it_and_clears_everyone(self):
        self.poll(_standard())
        res = self.poll(_payload(), allow_empty=True)
        self.assertEqual((res["status"], res["cleared"]), ("ok", 3))

    def test_an_empty_feed_with_nothing_known_is_just_empty(self):
        res = self.poll(_payload())
        self.assertEqual((res["status"], res["rows_written"]), ("ok", 0))
        self.assertEqual(self.polls()[0]["entries_seen"], 0)


class TestAppendOnly(_Base):

    def test_observations_and_polls_cannot_be_edited_or_deleted(self):
        self.poll(_standard())
        for sql in ("UPDATE nba_injury_observations SET report_status='Active'",
                    "DELETE FROM nba_injury_observations",
                    "UPDATE nba_injury_polls SET status='ok'",
                    "DELETE FROM nba_injury_polls"):
            with self.assertRaises(sqlite3.IntegrityError, msg=sql):
                self.conn.execute(sql)


class TestMain(_Base):
    """The exit code is what Task Scheduler and run_scheduled.py read."""

    def _resp(self, status, body):
        r = mock.Mock()
        r.status_code = status
        r.json.return_value = body
        return r

    def test_exit_0_on_a_good_poll(self):
        self.conn.close()
        with mock.patch.object(P.requests, "get", return_value=self._resp(200, _standard())):
            self.assertEqual(P.main(["--db", self.db]), 0)
        self.conn = sqlite3.connect(self.db)
        self.assertEqual(len(self.obs()), 3)

    def test_exit_1_and_a_failed_row_when_espn_is_down(self):
        self.conn.close()
        with mock.patch.object(P.requests, "get", return_value=self._resp(503, None)):
            self.assertEqual(P.main(["--db", self.db]), 1)
        with mock.patch.object(P.requests, "get",
                               side_effect=P.requests.ConnectionError("no route")):
            self.assertEqual(P.main(["--db", self.db]), 1)
        self.conn = sqlite3.connect(self.db)
        self.assertEqual([p["status"] for p in self.polls()], ["failed", "failed"])
        self.assertEqual(self.obs(), [])

    def test_no_user_agent_override_is_sent(self):
        """ESPN 403s custom user agents; see the module docstring."""
        self.conn.close()
        with mock.patch.object(P.requests, "get", return_value=self._resp(200, _standard())) as g:
            P.main(["--db", self.db])
        self.assertNotIn("headers", g.call_args.kwargs)


class TestSeason(unittest.TestCase):

    def test_league_year_boundaries(self):
        from datetime import datetime
        self.assertEqual(P.season_for(datetime(2026, 7, 1)), "2026-27")
        self.assertEqual(P.season_for(datetime(2027, 6, 30)), "2026-27")
        self.assertEqual(P.season_for(datetime(2026, 6, 30)), "2025-26")


if __name__ == "__main__":
    unittest.main()
