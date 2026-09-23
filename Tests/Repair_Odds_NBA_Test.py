"""
Repair_Odds_NBA_Test.py
=======================
Guards the NBA path of src/Sports/repair_odds.py, and the budget/quota rules
it now shares with the NFL path.

Everything network-shaped goes through one fake `requests.get` that answers
for ESPN's scoreboard and The Odds API, and records every URL it was asked
for -- so "spent 0 credits" is asserted as "never called The Odds API", not
inferred from a log line. No test touches a real database.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Sports import repair_odds as R  # noqa: E402

NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)
#: A slate that tipped yesterday, inside the unattended 3-day window.
TIP_A = NOW - timedelta(days=1, hours=2)
TIP_B = TIP_A + timedelta(minutes=30)
TIP_C = TIP_A + timedelta(hours=1)
TIP_D = TIP_A + timedelta(hours=2)


def _z(d):
    return d.strftime("%Y-%m-%dT%H:%MZ")


def espn_event(eid, home, away, tip, stype=2, status="STATUS_FINAL"):
    return {"id": str(eid), "date": _z(tip), "season": {"type": stype},
            "status": {"type": {"name": status}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": home}},
                {"homeAway": "away", "team": {"displayName": away}}]}]}


def odds_event(home, away, tip, ml=(-150, 130)):
    return {"id": f"{home}-{tip.isoformat()}", "home_team": home, "away_team": away,
            "commence_time": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bookmakers": [{"key": "fanduel", "markets": [
                {"key": "h2h", "outcomes": [{"name": home, "price": ml[0]},
                                            {"name": away, "price": ml[1]}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 225.5, "price": -110},
                                               {"name": "Under", "point": 225.5, "price": -110}]},
            ]}]}


class FakeNet:
    """Stands in for requests.get. `espn` maps YYYYMMDD -> events (missing
    days are empty); `historical` maps the requested date -> (status, body)."""

    def __init__(self, espn=None, historical=None, remaining="400", espn_status=200,
                 espn_body=None):
        self.espn = espn or {}
        self.historical = historical or {}
        self.remaining = remaining
        self.espn_status = espn_status
        self.espn_body = espn_body
        self.calls = []

    def odds_calls(self):
        return [u for u, _ in self.calls if "the-odds-api" in u]

    def historical_calls(self):
        return [p.get("date") for u, p in self.calls if "/historical/" in u]

    def __call__(self, url, params=None, timeout=None, **kw):
        params = params or {}
        self.calls.append((url, params))
        r = mock.Mock()
        r.headers = {}
        if "espn.com" in url:
            r.status_code = self.espn_status
            body = self.espn_body if self.espn_body is not None else \
                {"leagues": [], "events": self.espn.get(params.get("dates"), [])}
            r.json.return_value = body
            return r
        if self.remaining is not None:
            r.headers = {"x-requests-remaining": self.remaining, "x-requests-used": "100",
                         "x-requests-last": "30"}
        if url.endswith("/sports"):
            r.status_code = 200
            r.json.return_value = []
            return r
        if "/historical/sports/basketball_nba/odds" in url:
            status, body = self.historical.get(params.get("date"), (200, {"timestamp": None,
                                                                          "data": []}))
            r.status_code = status
            r.json.return_value = body
            r.raise_for_status = mock.Mock()
            return r
        raise AssertionError(f"unexpected URL {url}")


def _espn_days(*events_with_tips):
    """Group ESPN events by every US date they could be asked for."""
    out = {}
    for ev, tip in events_with_tips:
        for d in (tip - timedelta(hours=5)).date(), tip.date():
            out.setdefault(d.strftime("%Y%m%d"), [])
            if ev not in out[d.strftime("%Y%m%d")]:
                out[d.strftime("%Y%m%d")].append(ev)
    return out


class _Base(unittest.TestCase):

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="repair_nba_"), "OddsData.sqlite")
        conn = sqlite3.connect(self.db)
        # the table as main_api creates it; the client's migration adds the rest
        conn.execute("""CREATE TABLE odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, captured_at TEXT NOT NULL,
            sport TEXT NOT NULL, sportsbook TEXT NOT NULL, game_key TEXT NOT NULL,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL, home_ml REAL, away_ml REAL,
            ou_line REAL, game_start_time_utc TEXT,
            provenance TEXT NOT NULL DEFAULT 'observed')""")
        R.ensure_nba_snapshot_schema(conn)
        conn.commit()
        conn.close()
        patches = [mock.patch.object(R, "NBA_ODDS_DB", self.db),
                   mock.patch.dict(os.environ, {"ODDS_API_KEY": "test-key-not-real"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def snap(self, home, away, captured, sport="NBA", book="fanduel", provenance="observed"):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                     "home_team, away_team, home_ml, away_ml, provenance) "
                     "VALUES (?,?,?,?,?,?,-120,100,?)",
                     (captured, sport, book, f"{home}:{away}", home, away, provenance))
        conn.commit()
        conn.close()

    def rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute("SELECT * FROM odds_snapshots ORDER BY id").fetchall()
        finally:
            conn.close()

    def run_main(self, net, *argv):
        with mock.patch.object(R.requests, "get", side_effect=net):
            return R.main(["--sport", "nba", *argv])


class TestOffseasonAndFailures(_Base):

    def test_offseason_finds_nothing_and_never_calls_the_odds_api(self):
        net = FakeNet()
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual(net.odds_calls(), [], "an empty schedule must cost 0 credits")
        self.assertGreaterEqual(len(net.calls), 3, "each day of the window is looked at")
        self.assertEqual(self.rows(), [])

    def test_preseason_and_postponed_games_are_not_repaired(self):
        net = FakeNet(espn=_espn_days(
            (espn_event(1, "Boston Celtics", "Miami Heat", TIP_A, stype=1), TIP_A),
            (espn_event(2, "Utah Jazz", "Denver Nuggets", TIP_A, status="STATUS_POSTPONED"),
             TIP_A)))
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual(net.odds_calls(), [])

    def test_an_unreadable_schedule_fails_loudly_even_unattended(self):
        for net in (FakeNet(espn_status=503), FakeNet(espn_body={"code": 400}),
                    FakeNet(espn_body=["not", "a", "dict"])):
            with self.assertLogs(R.logger, "ERROR") as logs:
                self.assertEqual(self.run_main(net, "--unattended", "--apply"), 1)
            self.assertIn("NOT checked", "\n".join(logs.output))
            self.assertEqual(net.odds_calls(), [])

    def test_a_network_error_reading_the_schedule_fails(self):
        def boom(url, **kw):
            raise R.requests.ConnectionError("no route")
        with mock.patch.object(R.requests, "get", side_effect=boom):
            self.assertEqual(R.main(["--sport", "nba", "--unattended"]), 1)

    def test_game_flag_is_refused_for_the_nba(self):
        self.assertEqual(R.main(["--sport", "nba", "--game", "0022600001"]), 2)


class TestWhatCountsAsMissing(_Base):

    def gaps_for(self, *events_with_tips):
        schedule, seen = R.nba_schedule(NOW - timedelta(days=3), NOW,
                                        fetch_day=lambda d: {"events": _espn_days(
                                            *events_with_tips).get(d, [])})
        conn = sqlite3.connect(self.db)
        try:
            self.unlisted = R.nba_unlisted(conn, seen, NOW - timedelta(days=3), NOW)
            return [g["key"] for g in R.nba_gaps(conn, schedule, NOW - timedelta(days=3))]
        finally:
            conn.close()

    def test_a_snapshot_inside_three_hours_of_tip_covers_the_game(self):
        self.snap("Boston Celtics", "Miami Heat", (TIP_A - timedelta(minutes=50)).isoformat())
        self.assertEqual(self.gaps_for(
            (espn_event(1, "Boston Celtics", "Miami Heat", TIP_A), TIP_A)), [])

    def test_a_morning_snapshot_does_not_count_as_a_close(self):
        self.snap("Boston Celtics", "Miami Heat", (TIP_A - timedelta(hours=10)).isoformat())
        self.assertEqual(self.gaps_for(
            (espn_event(1, "Boston Celtics", "Miami Heat", TIP_A), TIP_A)),
            ["Boston Celtics:Miami Heat"])

    def test_a_snapshot_after_tip_does_not_count(self):
        self.snap("Boston Celtics", "Miami Heat", (TIP_A + timedelta(minutes=5)).isoformat())
        self.assertEqual(len(self.gaps_for(
            (espn_event(1, "Boston Celtics", "Miami Heat", TIP_A), TIP_A))), 1)

    def test_all_three_timestamp_spellings_are_understood(self):
        base = TIP_A - timedelta(minutes=20)
        self.snap("Boston Celtics", "Miami Heat", base.strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.snap("Utah Jazz", "Denver Nuggets", base.replace(tzinfo=None).isoformat())  # SBR path
        self.assertEqual(self.gaps_for(
            (espn_event(1, "Boston Celtics", "Miami Heat", TIP_A), TIP_A),
            (espn_event(2, "Utah Jazz", "Denver Nuggets", TIP_A), TIP_A)), [])

    def test_clippers_spellings_match(self):
        self.snap("Los Angeles Clippers", "Phoenix Suns",
                  (TIP_A - timedelta(minutes=20)).isoformat())
        self.assertEqual(self.gaps_for(
            (espn_event(1, "LA Clippers", "Phoenix Suns", TIP_A), TIP_A)), [])

    def test_a_wnba_row_does_not_cover_an_nba_game(self):
        self.snap("Boston Celtics", "Miami Heat", (TIP_A - timedelta(minutes=20)).isoformat(),
                  sport="WNBA")
        self.assertEqual(len(self.gaps_for(
            (espn_event(1, "Boston Celtics", "Miami Heat", TIP_A), TIP_A))), 1)

    def test_a_game_our_recorder_saw_but_espn_did_not_list_is_reported(self):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                     "home_team, away_team, game_start_time_utc) VALUES (?,?,?,?,?,?,?)",
                     ((TIP_A - timedelta(minutes=20)).isoformat(), "NBA", "fanduel",
                      "Utah Jazz:Denver Nuggets", "Utah Jazz", "Denver Nuggets", _z(TIP_A)))
        conn.commit()
        conn.close()
        self.gaps_for()
        self.assertEqual([k for k, _ in self.unlisted], ["Utah Jazz:Denver Nuggets"])
        # and a preseason listing of the same game satisfies the cross-check
        self.gaps_for((espn_event(9, "Utah Jazz", "Denver Nuggets", TIP_A, stype=1), TIP_A))
        self.assertEqual(self.unlisted, [])

    def test_an_empty_espn_schedule_contradicted_by_our_recorder_fails(self):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO odds_snapshots (captured_at, sport, sportsbook, game_key, "
                     "home_team, away_team, game_start_time_utc) VALUES (?,?,?,?,?,?,?)",
                     ((TIP_A - timedelta(minutes=20)).isoformat(), "NBA", "fanduel",
                      "Utah Jazz:Denver Nuggets", "Utah Jazz", "Denver Nuggets", _z(TIP_A)))
        conn.commit()
        conn.close()
        net = FakeNet()     # ESPN answers 200 with no events, every day
        with self.assertLogs(R.logger, "ERROR"):
            self.assertEqual(self.run_main(net, "--unattended", "--apply"), 1)
        self.assertEqual(net.odds_calls(), [])


class TestRepair(_Base):

    def slate(self):
        """Two games at TIP_A: one covered, one missed."""
        self.snap("Boston Celtics", "Miami Heat", (TIP_A - timedelta(minutes=15)).isoformat())
        return _espn_days((espn_event(1, "Boston Celtics", "Miami Heat", TIP_A), TIP_A),
                          (espn_event(2, "LA Clippers", "Phoenix Suns", TIP_A), TIP_A))

    def when(self, tip):
        return (tip - timedelta(minutes=R.LOOKBACK_MINUTES)).isoformat().replace("+00:00", "Z")

    def archive(self, tip, *events, snap=None):
        snap = snap or (tip - timedelta(minutes=5))
        return {self.when(tip): (200, {"timestamp": snap.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       "data": list(events)})}

    def test_dry_run_prices_it_and_calls_nothing(self):
        net = FakeNet(espn=self.slate())
        self.assertEqual(self.run_main(net, "--unattended"), 0)
        self.assertEqual(net.odds_calls(), [])
        self.assertEqual(len(self.rows()), 1)

    def test_writes_only_the_missed_game_as_reconstructed(self):
        rematch = odds_event("Los Angeles Clippers", "Phoenix Suns", TIP_A + timedelta(days=20))
        covered = odds_event("Boston Celtics", "Miami Heat", TIP_A)
        missed = odds_event("Los Angeles Clippers", "Phoenix Suns", TIP_A, ml=(-200, 170))
        days = self.slate()
        net = FakeNet(espn=days, historical=self.archive(TIP_A, rematch, covered, missed))
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual(net.historical_calls(), [self.when(TIP_A)])
        new = [r for r in self.rows() if r["provenance"] == "reconstructed"]
        self.assertEqual(len(new), 1)
        r = new[0]
        self.assertEqual((r["sport"], r["sportsbook"], r["game_key"], r["home_ml"], r["away_ml"],
                          r["ou_line"]),
                         ("NBA", "fanduel", "Los Angeles Clippers:Phoenix Suns", -200, 170, 225.5))
        self.assertEqual(r["captured_at"], (TIP_A - timedelta(minutes=5)).isoformat())
        self.assertLess(r["captured_at"], TIP_A.isoformat())

        # Re-running finds the game covered now and spends nothing.
        net2 = FakeNet(espn=days, historical=self.archive(TIP_A, rematch, covered, missed))
        self.assertEqual(self.run_main(net2, "--unattended", "--apply"), 0)
        self.assertEqual(net2.odds_calls(), [])
        self.assertEqual(len(self.rows()), 2)

    def test_a_snapshot_at_or_after_tip_is_refused(self):
        net = FakeNet(espn=self.slate(), historical=self.archive(
            TIP_A, odds_event("Los Angeles Clippers", "Phoenix Suns", TIP_A),
            snap=TIP_A + timedelta(minutes=1)))
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual([r for r in self.rows() if r["provenance"] == "reconstructed"], [])

    def test_an_unknown_quota_is_not_permission_to_spend(self):
        net = FakeNet(espn=self.slate(), remaining=None)
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual(net.historical_calls(), [])

    def test_too_little_quota_leaves_it_for_the_live_recorder(self):
        net = FakeNet(espn=self.slate(), remaining=str(30 + R.QUOTA_FLOOR + R.UNATTENDED_RESERVE - 1))
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual(net.historical_calls(), [])

    def test_free_tier_401_is_a_warning_unattended_and_an_error_by_hand(self):
        hist = {self.when(TIP_A): (401, {})}
        days = self.slate()
        net = FakeNet(espn=days, historical=hist)
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        net = FakeNet(espn=days, historical=hist)
        self.assertEqual(self.run_main(net, "--apply", "--since",
                                       (NOW - timedelta(days=3)).isoformat()), 1)
        self.assertEqual([r for r in self.rows() if r["provenance"] == "reconstructed"], [])

    def test_unattended_repairs_the_three_newest_tips_and_defers_the_rest(self):
        days = _espn_days(*[(espn_event(i, h, "Miami Heat", t), t) for i, (h, t) in enumerate(
            [("Boston Celtics", TIP_A), ("Utah Jazz", TIP_B), ("Denver Nuggets", TIP_C),
             ("Chicago Bulls", TIP_D)])])
        net = FakeNet(espn=days)
        self.assertEqual(self.run_main(net, "--unattended", "--apply"), 0)
        self.assertEqual(sorted(net.historical_calls()),
                         sorted(self.when(t) for t in (TIP_B, TIP_C, TIP_D)))

    def test_attended_over_budget_refuses(self):
        days = _espn_days(*[(espn_event(i, h, "Miami Heat", t), t) for i, (h, t) in enumerate(
            [("Boston Celtics", TIP_A), ("Utah Jazz", TIP_B)])])
        net = FakeNet(espn=days)
        self.assertEqual(self.run_main(net, "--apply", "--max-credits", "40", "--since",
                                       (NOW - timedelta(days=3)).isoformat()), 1)
        self.assertEqual(net.odds_calls(), [])

    def test_a_dry_run_never_opens_the_database_for_writing(self):
        net = FakeNet(espn=self.slate())
        before = os.path.getmtime(self.db)
        with mock.patch.object(R, "ensure_nba_snapshot_schema",
                               side_effect=AssertionError("dry run opened a writer")):
            self.assertEqual(self.run_main(net, "--unattended"), 0)
            self.assertEqual(self.run_main(net), 0)
        self.assertEqual(os.path.getmtime(self.db), before)
        self.assertEqual(net.odds_calls(), [])


class TestSharedBudget(unittest.TestCase):

    def test_fit_budget(self):
        plan = {"t1": ["a"], "t2": ["b"], "t3": ["c"], "t4": ["d"]}
        self.assertEqual(R.fit_budget(plan, 30, 300, False), (plan, 120, "ok"))
        self.assertEqual(R.fit_budget(plan, 30, 90, False)[2], "over_budget")
        trimmed, est, verdict = R.fit_budget(plan, 30, 90, True)
        self.assertEqual((sorted(trimmed), est, verdict), (["t2", "t3", "t4"], 90, "ok"))
        self.assertEqual(R.fit_budget(plan, 30, 20, True), (None, 0, "unaffordable"))

    def test_quota_refusal(self):
        with mock.patch.object(R, "remaining_quota", return_value=None):
            self.assertIn("could not read", R.unattended_quota_refusal(30))
        with mock.patch.object(R, "remaining_quota", return_value=100):
            self.assertIn("100 credits remain", R.unattended_quota_refusal(30))
        with mock.patch.object(R, "remaining_quota", return_value=1000):
            self.assertIsNone(R.unattended_quota_refusal(30))


class TestNflPathStillWorks(unittest.TestCase):
    """The NFL flow was moved into _run_nfl; pin that it still decides gaps
    from its own schedule and refuses to spend on a dry run."""

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="repair_nfl_"), "NflData.sqlite")
        conn = sqlite3.connect(self.db)
        R.ensure_core_schema(conn)
        R.ensure_snapshot_schema(conn)
        for i, h in enumerate((TIP_A, TIP_B)):
            conn.execute("INSERT INTO games (game_id, sport, league, season, season_type, "
                         "date_utc, local_date, home_team_id, away_team_id) "
                         "VALUES (?, 'football', 'NFL', '2020', 'REG', ?, ?, 'nfl-KC', 'nfl-BUF')",
                         (f"g{i}", h.isoformat(), h.date().isoformat()))
        conn.commit()
        conn.close()
        p = mock.patch.dict(R.SPORTS["nfl"], {"db": self.db})
        p.start()
        self.addCleanup(p.stop)

    def test_dry_run_and_budget(self):
        with mock.patch.object(R.requests, "get", side_effect=AssertionError("no network")):
            self.assertEqual(R.main(["--sport", "nfl", "--unattended"]), 0)
            self.assertEqual(R.main(["--sport", "nfl", "--max-credits", "40", "--since",
                                     (NOW - timedelta(days=3)).isoformat()]), 1)


if __name__ == "__main__":
    unittest.main()
