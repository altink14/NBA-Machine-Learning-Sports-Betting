"""
job_health.py
=============
One read-only answer to "is every scheduled job actually working?"

WHY THIS EXISTS. From 22 August to 21 September 2026 the sealed model the site
advertised was not the model serving picks: it had stopped loading, the code
fell back to the old one, and every scheduled run still reported success. The
same shape keeps recurring one layer over -- a grader handed a table where no
game had ever finished, a daily job that pinged a server nobody started, a
15-minute job that fired a third as often as it was set to because the PC was
asleep. Each time the job's own exit code said "fine". A job can be wrong in
three ways its exit code cannot see: it did not run at all, it ran and did
nothing, or it ran and wrote something stale that looks plausible.

So this checks each job three ways and reports the worst:

  1. TASK SCHEDULER'S OWN RECORD -- is the task there, enabled, set to the
     cadence we expect, when did it last start, what did it return, how many
     runs has Windows itself counted as missed.
  2. THE JOB'S OWN LOG -- did the run that Windows started actually get as far
     as writing its log, did it finish, which steps failed, and how many of
     the runs due in the last 24 hours really happened.
  3. THE DATA IT IS SUPPOSED TO LEAVE BEHIND -- the newest injury poll, odds
     heartbeat, team-stats snapshot, play-by-play for every archived game,
     pending picks past kickoff, the newest backup. This is the layer that
     catches "ran and did nothing".

READ-ONLY. It opens every database with mode=ro, reads logs, and asks
PowerShell for Get-ScheduledTask / Get-ScheduledTaskInfo. It never creates,
changes, runs or disables a task, and it never calls stats.nba.com or any odds
API. The only files it writes are its own report (logs/job_health.json and
logs/job_health.txt) and, with --notify, the note of what it last sent.

UNKNOWN IS NOT OK. If a check cannot read its evidence -- a table that does not
exist yet, a log that is missing, Task Scheduler unavailable -- it says
"unknown" and the overall status is at least a warning. A monitor that turns
"could not look" into "fine" is the bug this file exists to catch.

NOTIFICATIONS are pluggable and OFF unless configured (see notify()). Setting
one environment variable in .env and running this with --notify on a schedule
is the whole switch-on; the steps are in docs/JOB_HEALTH.md.

Usage:
    venv/Scripts/python.exe job_health.py            # report, write logs/job_health.*
    venv/Scripts/python.exe job_health.py --json     # print the JSON instead of prose
    venv/Scripts/python.exe job_health.py --notify   # also send to configured channels

Exit code: 1 when anything is failing, else 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
TEAM_DB = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
ODDS_DB = os.path.join(REPO_ROOT, "Data", "OddsData.sqlite")
NFL_DB = os.path.join(REPO_ROOT, "Data", "NflData.sqlite")
DAILY_UPDATE_LOG = os.path.join(REPO_ROOT, "daily_update.log")
REPORT_JSON = os.path.join(LOG_DIR, "job_health.json")
REPORT_TXT = os.path.join(LOG_DIR, "job_health.txt")
NOTIFY_STATE = os.path.join(LOG_DIR, "job_health_notified.json")

OK, WARN, FAIL, OFF, UNKNOWN = "ok", "warn", "fail", "off", "unknown"
#: Worst wins. OFF (not expected / not configured yet) never raises the
#: overall status; UNKNOWN always does, to at least a warning.
_RANK = {OK: 0, OFF: 0, UNKNOWN: 1, WARN: 2, FAIL: 3}

#: Play-by-play exists from this season on (PBP_FIRST_SEASON in main_api.py).
PBP_FIRST_SEASON = "2019-20"

#: Task Scheduler result codes that are not failures.
_SCHED_RUNNING = 267009       # 0x41301 the task is running right now
_SCHED_NEVER_RAN = 267011     # 0x41303 the task has not run yet
_SCHED_CODES = {
    0: "success",
    1: "the job reported a failure (exit 1)",
    2: "the job was started with a bad command line (exit 2)",
    _SCHED_RUNNING: "running now",
    _SCHED_NEVER_RAN: "has never run",
    267014: "stopped by a user or by the task's time limit",
    2147750687: "an instance was already running, so this one was not started",
    2147942402: "the program was not found (file moved?)",
    3221225786: "killed (Ctrl+C / the console was closed)",
}


@dataclass(frozen=True)
class TaskSpec:
    """One Task Scheduler task and what we expect of it."""
    job_id: str
    task_name: str
    label: str
    every: timedelta          # how often it is set to run
    warn_after: timedelta     # no start for this long: a run was missed
    fail_after: timedelta     # no start for this long: it has stopped
    log_path: Optional[str]   # the job's own log
    log_kind: str             # "scheduled:<mode>" or "daily_update"
    run_timeout: timedelta    # a start older than this with no finish = crashed
    trigger: str              # what the task definition should say: "PT15M", "PT1H", "daily"
    what: str                 # one line: what it does, for the summary
    optional: bool = False    # not created yet by design: absent = "off", not "fail"


TASKS: List[TaskSpec] = [
    TaskSpec("odds_recorder", "BettingBuddy Odds Recorder", "Odds recorder",
             timedelta(minutes=15), timedelta(minutes=35), timedelta(hours=2),
             os.path.join(LOG_DIR, "scheduled_frequent.log"), "scheduled:frequent",
             timedelta(minutes=35), "PT15M",
             "NFL and NBA closing lines, captured as kickoff approaches"),
    TaskSpec("hourly", "BettingBuddy Injuries And Predictions", "Injuries and predictions",
             timedelta(hours=1), timedelta(minutes=80), timedelta(hours=4),
             os.path.join(LOG_DIR, "scheduled_hourly.log"), "scheduled:hourly",
             timedelta(hours=1), "PT1H",
             "NFL and NBA injury recorders, NFL picks, ledger publish"),
    TaskSpec("ledger_grading", "BettingBuddy Ledger Grading", "Ledger grading",
             timedelta(days=1), timedelta(hours=27), timedelta(hours=51),
             os.path.join(LOG_DIR, "scheduled_daily.log"), "scheduled:daily",
             timedelta(hours=2), "daily",
             "NFL results, grading, odds repair, closing-line value"),
    TaskSpec("daily_update", "BettingBuddy Daily Data Update", "Daily data update",
             timedelta(days=1), timedelta(hours=27), timedelta(hours=51),
             DAILY_UPDATE_LOG, "daily_update",
             timedelta(hours=2), "daily",
             "NBA box scores, play-by-play, team stats, NBA picks, odds board, preflight"),
    # This checker's own schedule, which the owner creates to switch
    # notifications on (docs/JOB_HEALTH.md). Until then it is "off".
    TaskSpec("job_health", "BettingBuddy Job Health", "Job health checker",
             timedelta(hours=1), timedelta(minutes=80), timedelta(hours=4),
             None, "none", timedelta(minutes=10), "PT1H",
             "This report, with --notify: sends problems to the phone", optional=True),
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _utc(dt: datetime) -> datetime:
    """Aware UTC. A naive datetime is taken as this machine's local time,
    which is what the logs write (logging's asctime is local)."""
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(timezone.utc)


def _parse_iso(value: Any) -> Optional[datetime]:
    """An ISO timestamp from a database row, as aware UTC. Naive = UTC,
    because every recorder in this repo writes datetime.now(timezone.utc)."""
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(s[:19])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ago(then: Optional[datetime], now: datetime) -> str:
    if then is None:
        return "never"
    secs = (now - then).total_seconds()
    if secs < 0:
        return "in the future (clock skew?)"
    if secs < 90:
        return f"{int(secs)} s ago"
    if secs < 5400:
        return f"{int(secs // 60)} min ago"
    if secs < 72 * 3600:
        return f"{secs / 3600:.1f} h ago"
    return f"{secs / 86400:.1f} days ago"


def _local(dt: Optional[datetime]) -> Optional[str]:
    """Local wall-clock time for prose ("2026-09-26 09:30")."""
    return dt.astimezone().strftime("%Y-%m-%d %H:%M") if dt else None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds") if dt else None


def _every(td: timedelta) -> str:
    mins = int(td.total_seconds() // 60)
    if mins < 60:
        return f"every {mins} min"
    if mins < 1440:
        return "every hour" if mins == 60 else f"every {mins // 60} h"
    return "every day" if mins == 1440 else f"every {mins // 1440} days"


def worst(statuses: Iterable[str]) -> str:
    out = OK
    for s in statuses:
        if _RANK.get(s, 1) > _RANK[out]:
            out = s
    return out


def _check(cid: str, name: str, status: str, summary: str, **evidence) -> Dict[str, Any]:
    return {"id": cid, "name": name, "status": status, "summary": summary,
            "evidence": {k: v for k, v in evidence.items() if v is not None}}


def _ro(path: str) -> sqlite3.Connection:
    """Read-only connection. Raises if the file is missing (mode=ro never creates)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _tail_lines(path: str, max_bytes: int = 1_500_000) -> List[str]:
    """The last ~1.5 MB of a log, as lines. The logs only ever grow; reading
    the whole of a months-old log on every API request would not."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[1:] if size > max_bytes else lines   # first line may be cut


# --------------------------------------------------------------------------- #
# Calendar: when is silence expected?
# --------------------------------------------------------------------------- #

def _opening_night() -> Optional[date]:
    try:
        from preflight_opening_night import OPENING_NIGHT
        return OPENING_NIGHT
    except Exception:
        return None


def current_nba_season(today: date) -> str:
    """Same rule as daily_update.current_season (October starts a new one)."""
    start = today.year if today.month >= 10 else today.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def nba_games_expected(today: date, opening: Optional[date] = None) -> bool:
    """Opening night through June. Mirrors daily_update._nba_games_expected,
    which cannot be imported here: importing daily_update attaches a file
    handler to daily_update.log in whatever process imports it."""
    if today.month in (7, 8, 9):
        return False
    if today.month == 10:
        opening = opening or _opening_night()
        return today >= opening if opening and opening.year == today.year else today.day >= 20
    return True


def nba_new_season_waiting(today: date, opening: Optional[date] = None) -> bool:
    """1 October until two days after opening night: the season label has
    rolled over, the new season has no games, and refresh_team_stats writes
    nothing. Silence from the snapshot is correct then."""
    if today.month != 10:
        return False
    opening = opening or _opening_night()
    if not opening or opening.year != today.year:
        return today.day < 26
    return (today - opening).days < 2


def nfl_season_on(today: date) -> bool:
    """September through early February."""
    return today.month >= 9 or today.month <= 2


# --------------------------------------------------------------------------- #
# 1. Task Scheduler (read-only)
# --------------------------------------------------------------------------- #

_PS_QUERY = r"""
$ErrorActionPreference = 'Stop'
try { $tasks = @(Get-ScheduledTask -TaskName 'BettingBuddy*') } catch { $tasks = @() }
$out = foreach ($t in $tasks) {
  $i = $t | Get-ScheduledTaskInfo
  $fmt = { param($d) if ($d -and $d.Year -gt 2000) { $d.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') } else { $null } }
  [pscustomobject]@{
    name = $t.TaskName
    state = [string]$t.State
    last_run = & $fmt $i.LastRunTime
    last_result = $i.LastTaskResult
    next_run = & $fmt $i.NextRunTime
    missed_runs = $i.NumberOfMissedRuns
    logon_type = [string]$t.Principal.LogonType
    wake_to_run = $t.Settings.WakeToRun
    start_when_available = $t.Settings.StartWhenAvailable
    disallow_on_batteries = $t.Settings.DisallowStartIfOnBatteries
    triggers = @($t.Triggers | ForEach-Object { [pscustomobject]@{
        kind = $_.CimClass.CimClassName; repeat = $_.Repetition.Interval;
        days_interval = $_.DaysInterval; enabled = $_.Enabled } })
    action = (@($t.Actions | ForEach-Object { ($_.Execute + ' ' + $_.Arguments).Trim() }) -join ' ; ')
  }
}
ConvertTo-Json -InputObject @($out) -Depth 4 -Compress
"""


def read_task_scheduler(runner: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """{"available": bool, "tasks": {name: info}, "error": str|None}.

    Only asks; never changes anything. On a machine with no Task Scheduler
    (the future Linux server) this says so, and every task check is
    "unknown", because the jobs run on the home PC and this machine cannot
    see them.
    """
    if os.name != "nt" and runner is None:
        return {"available": False, "tasks": {},
                "error": "Task Scheduler is Windows-only; the jobs run on the home PC"}
    run = runner or subprocess.run
    try:
        r = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS_QUERY],
                capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"available": False, "tasks": {}, "error": f"could not ask Task Scheduler: {exc}"}
    if r.returncode != 0:
        return {"available": False, "tasks": {},
                "error": f"Task Scheduler query failed: {(r.stderr or '').strip()[-300:]}"}
    try:
        raw = json.loads((r.stdout or "").strip() or "[]")
    except ValueError:
        return {"available": False, "tasks": {}, "error": "Task Scheduler answered something that is not JSON"}
    if isinstance(raw, dict):
        raw = [raw]
    return {"available": True, "error": None,
            "tasks": {t.get("name"): t for t in raw if isinstance(t, dict) and t.get("name")}}


def _trigger_matches(spec: TaskSpec, info: Dict[str, Any]) -> Optional[str]:
    """None when the definition's trigger is what we expect, else why not."""
    trig = info.get("triggers") or []
    if isinstance(trig, dict):
        trig = [trig]
    enabled = [t for t in trig if t.get("enabled") is not False]
    if not enabled:
        return "the task has no enabled trigger, so it will never start"
    if spec.trigger == "daily":
        if any((t.get("kind") or "").endswith("DailyTrigger") and (t.get("days_interval") or 1) == 1
               for t in enabled):
            return None
        return "the task is no longer set to run every day"
    if any((t.get("repeat") or "") == spec.trigger for t in enabled):
        return None
    got = ", ".join(sorted({str(t.get("repeat") or t.get("kind")) for t in enabled}))
    return f"the task repeats {got}, not {spec.trigger}"


def check_scheduler(spec: TaskSpec, sched: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    name = f"Task Scheduler: {spec.task_name}"
    if not sched.get("available"):
        return _check("scheduler", name, UNKNOWN, sched.get("error") or "Task Scheduler unavailable")
    info = sched["tasks"].get(spec.task_name)
    if info is None:
        return _check("scheduler", name, FAIL,
                      "The task does not exist in Task Scheduler, so this job never runs.")
    last = _parse_iso(info.get("last_run"))
    nxt = _parse_iso(info.get("next_run"))
    code = info.get("last_result")
    meaning = _SCHED_CODES.get(code, f"error code {code}")
    missed = info.get("missed_runs") or 0
    ev = dict(state=info.get("state"), last_run=_iso(last), last_result=code,
              last_result_meaning=meaning, next_run=_iso(nxt), missed_runs=missed,
              logon_type=info.get("logon_type"), wake_to_run=info.get("wake_to_run"),
              start_when_available=info.get("start_when_available"), action=info.get("action"))

    problems: List[tuple] = []
    if (info.get("state") or "").lower() == "disabled":
        problems.append((FAIL, "The task is DISABLED, so it will not run."))
    bad_trigger = _trigger_matches(spec, info)
    if bad_trigger:
        problems.append((FAIL, f"Its schedule changed: {bad_trigger}."))
    if last is None:
        problems.append((FAIL, "Windows has no record of it ever running."))
    else:
        gap = now - last
        if gap > spec.fail_after:
            problems.append((FAIL, f"Last started {_local(last)} ({_ago(last, now)}); it should run "
                                   f"{_every(spec.every)}, so it has stopped running."))
        elif gap > spec.warn_after:
            problems.append((WARN, f"Last started {_local(last)} ({_ago(last, now)}); it should run "
                                   f"{_every(spec.every)}, so at least one run was missed."))
    if code not in (0, _SCHED_RUNNING, None):
        problems.append((FAIL, f"The last run returned {code}: {meaning}."))
    if nxt is not None and now - nxt > timedelta(minutes=10) and (info.get("state") or "").lower() != "disabled":
        problems.append((WARN, f"Its next run was due {_local(nxt)} and has not happened."))
    if missed and not problems and spec.every >= timedelta(days=1):
        # Windows' count of runs it skipped since the last one. For a daily job
        # one skip is a lost morning, so it is always worth saying.
        problems.append((WARN, f"Windows counts {missed} missed run(s) since the last one."))

    if problems:
        status = worst(p[0] for p in problems)
        text = " ".join(p[1] for p in problems)
        if missed and "missed run" not in text:
            text += f" Windows counts {missed} missed run(s)."
        return _check("scheduler", name, status, text, **ev)
    running = " It is running now." if code == _SCHED_RUNNING else ""
    return _check("scheduler", name, OK,
                  f"Last started {_local(last)} ({_ago(last, now)}), result {code} ({meaning}); "
                  f"next {_local(nxt) or 'not scheduled'}.{running}", **ev)


# --------------------------------------------------------------------------- #
# 2. The job's own log
# --------------------------------------------------------------------------- #

_STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - (\w+) - (.*)$")


def _stamped(lines: Iterable[str]):
    for ln in lines:
        m = _STAMP.match(ln)
        if m:
            yield _utc(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")), m.group(2), m.group(3)


def parse_scheduled_log(lines: List[str], mode: str) -> List[Dict[str, Any]]:
    """Every run in a run_scheduled.py log: start, finish, clean or not,
    and each step's last word ("odds recorder (NBA): ok | ...")."""
    runs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    start_re = re.compile(rf"^=== {re.escape(mode)} run starting")
    end_re = re.compile(rf"^=== {re.escape(mode)} run finished (clean|with (\d+) failure\(s\): (.*)) ===$")
    step_re = re.compile(r"^(.+?): (ok|exit -?\d+|timed out after .*?)(?: \| (.*))?$")
    for t, level, msg in _stamped(lines):
        if start_re.match(msg):
            cur = {"started": t, "finished": None, "clean": None, "failures": [], "steps": {}}
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = end_re.match(msg)
        if m:
            cur["finished"] = t
            cur["clean"] = m.group(1) == "clean"
            cur["failures"] = [s.strip() for s in (m.group(3) or "").split(",") if s.strip()]
            cur = None
            continue
        m = step_re.match(msg)
        if m:
            cur["steps"][m.group(1)] = {"ok": m.group(2) == "ok" and level == "INFO",
                                        "result": m.group(2), "detail": (m.group(3) or "")[:400]}
        elif level == "ERROR" and ":" in msg:
            # run_scheduled logs an unexpected exception as "name: message".
            name, _, rest = msg.partition(":")
            cur["steps"][name.strip()] = {"ok": False, "result": "error", "detail": rest.strip()[:400]}
    return runs


def parse_daily_update_log(lines: List[str]) -> List[Dict[str, Any]]:
    """Every run in daily_update.log, with its closing line's verdicts."""
    runs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    ok_re = re.compile(r"^=== Daily update finished OK \((.*)\) ===$")
    err_re = re.compile(r"^=== Daily update finished WITH ERRORS: (.*) ===$")
    for t, level, msg in _stamped(lines):
        if msg.startswith("=== Daily update starting"):
            cur = {"started": t, "finished": None, "clean": None, "failures": [],
                   "verdicts": {}, "errors": []}
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = ok_re.match(msg)
        if m:
            cur["finished"], cur["clean"] = t, True
            for part in m.group(1).split(","):
                k, _, v = part.partition(":")
                if v:
                    cur["verdicts"][k.strip()] = v.strip()
            cur = None
            continue
        m = err_re.match(msg)
        if m:
            cur["finished"], cur["clean"] = t, False
            cur["failures"] = [s.strip() for s in m.group(1).split(",") if s.strip()]
            cur = None
            continue
        if level == "ERROR" and len(cur["errors"]) < 12:
            cur["errors"].append(msg[:300])
    return runs


def _runs_for(spec: TaskSpec) -> Optional[List[Dict[str, Any]]]:
    if not spec.log_path or not os.path.exists(spec.log_path):
        return None
    lines = _tail_lines(spec.log_path)
    if spec.log_kind == "daily_update":
        return parse_daily_update_log(lines)
    return parse_scheduled_log(lines, spec.log_kind.split(":", 1)[1])


def check_log(spec: TaskSpec, runs: Optional[List[Dict[str, Any]]], now: datetime,
              sched_last: Optional[datetime]) -> Dict[str, Any]:
    name = f"Log: {os.path.basename(spec.log_path or '')}"
    if runs is None:
        return _check("log", name, UNKNOWN if sched_last is None else FAIL,
                      f"The job's log {spec.log_path} does not exist"
                      + (", although Task Scheduler says it ran. The program is probably "
                         "crashing before it can write a line." if sched_last else "."))
    if not runs:
        return _check("log", name, UNKNOWN, "The log holds no run of this job.")
    last = runs[-1]
    started = last["started"]
    problems: List[tuple] = []
    ev: Dict[str, Any] = {"last_started": _iso(started), "last_finished": _iso(last["finished"])}

    # Windows started it, but no run appeared in the log: python or the venv
    # is broken, or the script dies on import. Exit codes often still read 0
    # for a console that closed, so this is the check that sees it.
    if sched_last and sched_last - started > timedelta(minutes=10) \
            and now - sched_last > timedelta(minutes=10):
        problems.append((FAIL, f"Task Scheduler started it at {_local(sched_last)}, but the newest "
                               f"run in the log began {_local(started)}: the program did not get "
                               "as far as logging. Run the task's command by hand to see why."))

    if last["finished"] is None:
        if now - started > spec.run_timeout:
            problems.append((FAIL, f"The run that began {_local(started)} never finished: it "
                                   "crashed or was killed partway."))
        else:
            ev["running"] = True
    elif not last["clean"]:
        failed = last.get("failures") or []
        problems.append((FAIL, f"The last run ({_local(started)}) finished WITH ERRORS in: "
                               f"{', '.join(failed) or 'unnamed steps'}."))
        detail = [f"{k}: {v['detail'][:160]}" for k, v in (last.get("steps") or {}).items() if not v["ok"]]
        detail += last.get("errors") or []
        if detail:
            ev["errors"] = detail[:8]

    # Coverage: runs that actually started in the last 24 h against the number
    # due. This is how the PC sleeping through half its schedule was found.
    if spec.every < timedelta(days=1):
        since = now - timedelta(hours=24)
        n = sum(1 for r in runs if r["started"] >= since)
        due = int(timedelta(hours=24) / spec.every)
        pct = 100.0 * n / due
        ev["runs_last_24h"], ev["runs_due_24h"] = n, due
        gap = _longest_gap(runs, since, now)
        gap_txt = ""
        if gap and gap["hours"] * 60 >= 2 * spec.every.total_seconds() / 60:
            ev["longest_gap"] = gap
            gap_txt = (f" Longest silence: {_local(_parse_iso(gap['from']))} to "
                       f"{_local(_parse_iso(gap['to']))} ({gap['hours']:.1f} h).")
        if pct < 75:
            problems.append((FAIL, f"Only {n} of {due} runs due in the last 24 h happened ({pct:.0f}%): "
                                   "the PC was asleep, off or signed out for long stretches, and "
                                   "every missed run is a closing line or injury update we never saw."
                                   + gap_txt))
        elif pct < 90:
            problems.append((WARN, f"{n} of {due} runs due in the last 24 h happened ({pct:.0f}%)." + gap_txt))

    if spec.log_kind == "daily_update" and last["clean"]:
        v = last.get("verdicts") or {}
        ev["verdicts"] = v
        for key in ("preflight", "audit"):
            val = (v.get(key) or "").strip()
            if val and val.lower() != "clean":
                problems.append((WARN, f"The run finished OK, but its {key} said \"{val}\" "
                                       "(this does not change the task's exit code)."))

    if problems:
        return _check("log", name, worst(p[0] for p in problems),
                      " ".join(p[1] for p in problems), **ev)
    fin = last["finished"]
    txt = (f"Last run began {_local(started)} and finished clean" +
           (f" at {_local(fin)}" if fin else " (still running)") + ".")
    if "runs_last_24h" in ev:
        txt += f" {ev['runs_last_24h']} of {ev['runs_due_24h']} runs due in 24 h happened."
    return _check("log", name, OK, txt, **ev)


def _longest_gap(runs: List[Dict[str, Any]], since: datetime, now: datetime) -> Optional[Dict[str, Any]]:
    """The longest stretch in the window with no run starting. It runs from
    the last start before the window when there is one (so a silence that
    began yesterday reads from when it began) and up to now (so a job that
    has stopped shows the silence it is in)."""
    before = [r["started"] for r in runs if r["started"] < since]
    starts = [max(before) if before else since] + sorted(
        r["started"] for r in runs if r["started"] >= since) + [now]
    best = max(zip(starts, starts[1:]), key=lambda ab: ab[1] - ab[0], default=None)
    if not best:
        return None
    return {"from": _iso(best[0]), "to": _iso(best[1]),
            "hours": round((best[1] - best[0]).total_seconds() / 3600, 2)}


def _step(runs: Optional[List[Dict[str, Any]]], step: str) -> Optional[tuple]:
    """(run start, step dict) for the newest run that mentions the step."""
    for r in reversed(runs or []):
        if step in (r.get("steps") or {}):
            return r["started"], r["steps"][step]
    return None


# --------------------------------------------------------------------------- #
# 3. What each job leaves behind
# --------------------------------------------------------------------------- #

def check_nba_injury_polls(now: datetime, db: str = TEAM_DB) -> Dict[str, Any]:
    name = "NBA injury recorder: newest poll"
    try:
        conn = _ro(db)
    except Exception as exc:
        return _check("nba_injuries", name, UNKNOWN, f"Cannot open {db}: {exc}")
    try:
        if not _has_table(conn, "nba_injury_polls"):
            return _check("nba_injuries", name, UNKNOWN, "nba_injury_polls does not exist.")
        last = conn.execute("SELECT started_at, status, error FROM nba_injury_polls "
                            "ORDER BY id DESC LIMIT 1").fetchone()
        since = (now - timedelta(hours=24)).isoformat()
        n_ok, n_fail = conn.execute(
            "SELECT SUM(status='ok'), SUM(status='failed') FROM nba_injury_polls WHERE started_at >= ?",
            (since,)).fetchone()
    finally:
        conn.close()
    if not last:
        return _check("nba_injuries", name, FAIL, "The recorder has never written a poll.")
    t = _parse_iso(last["started_at"])
    ev = dict(last_poll=_iso(t), last_status=last["status"], ok_last_24h=n_ok or 0,
              failed_last_24h=n_fail or 0)
    if last["status"] != "ok":
        return _check("nba_injuries", name, FAIL,
                      f"The newest poll ({_local(t)}) FAILED: ESPN's feed could not be read. "
                      "An hour we could not see is gone for good.", error=last["error"], **ev)
    gap = now - t if t else None
    if gap is None or gap > timedelta(hours=4):
        return _check("nba_injuries", name, FAIL, f"Newest poll {_ago(t, now)}; it should poll hourly.", **ev)
    if gap > timedelta(minutes=80):
        return _check("nba_injuries", name, WARN, f"Newest poll {_ago(t, now)}; an hourly poll was missed.", **ev)
    return _check("nba_injuries", name, OK,
                  f"Newest poll {_ago(t, now)}, ok; {n_ok or 0} ok and {n_fail or 0} failed in 24 h.", **ev)


def check_nfl_injury_polls(now: datetime, runs, db: str = NFL_DB) -> Dict[str, Any]:
    """The NFL recorder exits 0 even when a source fails (see its docstring),
    so the run's 'ok' is not evidence. Its ingest_runs row is, and the
    wrapper's tail shows an ESPN failure if there was one."""
    name = "NFL injury recorder: newest poll"
    try:
        conn = _ro(db)
    except Exception as exc:
        return _check("nfl_injuries", name, UNKNOWN, f"Cannot open {db}: {exc}")
    try:
        if not _has_table(conn, "ingest_runs"):
            return _check("nfl_injuries", name, UNKNOWN, "NflData has no ingest_runs table.")
        row = conn.execute("SELECT started_at, rows_written FROM ingest_runs WHERE "
                           "table_name='nfl_injury_observations' AND source_endpoint='poll' "
                           "ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if not row:
        return _check("nfl_injuries", name, FAIL, "The recorder has never recorded a poll.")
    t = _parse_iso(row["started_at"])
    ev = dict(last_poll=_iso(t), rows_written=row["rows_written"])
    st = _step(runs, "injury recorder (NFL)")
    if st and "fetch failed" in (st[1].get("detail") or ""):
        return _check("nfl_injuries", name, WARN,
                      f"The run at {_local(st[0])} said a source failed (\"{st[1]['detail'][:120]}\"), "
                      "though the recorder still exits 0.", **ev)
    gap = now - t if t else None
    if gap is None or gap > timedelta(hours=4):
        return _check("nfl_injuries", name, FAIL, f"Newest poll {_ago(t, now)}; it should poll hourly.", **ev)
    if gap > timedelta(minutes=80):
        return _check("nfl_injuries", name, WARN, f"Newest poll {_ago(t, now)}; an hourly poll was missed.", **ev)
    return _check("nfl_injuries", name, OK, f"Newest poll {_ago(t, now)}. (It cannot tell us if the "
                  "nflverse half failed; only ESPN failures reach the log.)", **ev)


def check_nfl_odds_captures(now: datetime, db: str = NFL_DB) -> Dict[str, Any]:
    name = "NFL odds recorder: newest capture"
    if not nfl_season_on(now.date()):
        return _check("nfl_odds", name, OFF, "NFL offseason: no kickoffs, so no captures are expected.")
    try:
        conn = _ro(db)
        try:
            if not _has_table(conn, "market_line_snapshots"):
                return _check("nfl_odds", name, UNKNOWN, "market_line_snapshots does not exist.")
            last = conn.execute("SELECT MAX(captured_at) FROM market_line_snapshots").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return _check("nfl_odds", name, UNKNOWN, f"Cannot read {db}: {exc}")
    t = _parse_iso(last)
    # Captures happen only as a kickoff approaches; the longest normal gap in
    # a season week is Monday night to Thursday night (~3 days).
    if t is None or now - t > timedelta(days=8):
        return _check("nfl_odds", name, FAIL, f"Newest NFL line captured {_ago(t, now)}, in season.",
                      last_capture=_iso(t))
    if now - t > timedelta(days=4):
        return _check("nfl_odds", name, WARN, f"Newest NFL line captured {_ago(t, now)}; a normal week "
                      "has a kickoff at least every 4 days.", last_capture=_iso(t))
    return _check("nfl_odds", name, OK, f"Newest NFL line captured {_ago(t, now)}.", last_capture=_iso(t))


def check_nfl_grading(now: datetime, db: str = ODDS_DB) -> Dict[str, Any]:
    """A pick still 'pending' long after its game is the grader silently
    doing nothing (2026-09-21: fourteen finished games, 'ok' every morning)."""
    name = "NFL ledger: picks graded after their games"
    try:
        conn = _ro(db)
        try:
            if not _has_table(conn, "ledger"):
                return _check("nfl_grading", name, UNKNOWN, "The ledger table does not exist.")
            cutoff = (now - timedelta(hours=36)).isoformat()
            stale, oldest = conn.execute(
                "SELECT COUNT(*), MIN(event_start_utc) FROM ledger WHERE result='pending' "
                "AND event_start_utc < ?", (cutoff,)).fetchone()
            pending, total, last_graded = conn.execute(
                "SELECT SUM(result='pending'), COUNT(*), MAX(graded_at) FROM ledger").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _check("nfl_grading", name, UNKNOWN, f"Cannot read the ledger: {exc}")
    ev = dict(stale_pending=stale, oldest_stale_kickoff=oldest, pending=pending or 0, total=total,
              last_graded=last_graded)
    if stale:
        return _check("nfl_grading", name, FAIL,
                      f"{stale} pick(s) are still ungraded more than 36 h after kickoff (oldest "
                      f"{str(oldest)[:16]}). The grader is running but not grading.", **ev)
    return _check("nfl_grading", name, OK,
                  f"No pick is waiting more than 36 h after kickoff ({pending or 0} pending of {total}).", **ev)


def check_nba_predictions_log(now: datetime, db: str = ODDS_DB) -> Dict[str, Any]:
    name = "NBA picks logged before tip-off (predictions_log)"
    try:
        conn = _ro(db)
        try:
            if not _has_table(conn, "predictions_log"):
                return _check("nba_predictions", name, UNKNOWN, "predictions_log does not exist.")
            n, last = conn.execute("SELECT COUNT(*), MAX(logged_at) FROM predictions_log "
                                   "WHERE UPPER(COALESCE(sport, 'NBA')) = 'NBA'").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _check("nba_predictions", name, UNKNOWN, f"Cannot read predictions_log: {exc}")
    t = _parse_iso(last)
    ev = dict(rows=n, last_logged=_iso(t))
    if not nba_games_expected(now.date()):
        on = _opening_night()
        return _check("nba_predictions", name, OFF,
                      f"No NBA games until opening night{f' ({on})' if on else ''}; {n} row(s) "
                      "logged so far, which is correct.", **ev)
    # In season a slate is almost every day; the All-Star break is the one
    # long gap, so the bar is two and a half days, not one.
    if t is None or now - t > timedelta(hours=60):
        return _check("nba_predictions", name, FAIL,
                      f"The newest NBA pick was logged {_ago(t, now)}. Picks not logged before tip-off "
                      "can never be logged; check the daily update's prediction step.", **ev)
    if now - t > timedelta(hours=30):
        return _check("nba_predictions", name, WARN, f"The newest NBA pick was logged {_ago(t, now)}.", **ev)
    return _check("nba_predictions", name, OK, f"The newest NBA pick was logged {_ago(t, now)}.", **ev)


def check_team_stats_snapshot(now: datetime, db: str = TEAM_DB) -> Dict[str, Any]:
    name = "Team-stats snapshot the model predicts from"
    try:
        conn = _ro(db)
        try:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name GLOB "
                               "'[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]' ORDER BY name DESC LIMIT 1").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _check("team_stats", name, UNKNOWN, f"Cannot open {db}: {exc}")
    if not row:
        return _check("team_stats", name, FAIL, "There is no dated team-stats snapshot at all.")
    snap = datetime.strptime(row[0], "%Y-%m-%d").date()
    today = now.astimezone().date()
    age = (today - snap).days
    ev = dict(newest=row[0], age_days=age)
    if nba_new_season_waiting(today):
        return _check("team_stats", name, OFF if age > 1 else OK,
                      f"Newest snapshot {row[0]} ({age} day(s) old). Between 1 October and opening "
                      "night the refresh writes nothing, which is correct.", **ev)
    if age > 3:
        return _check("team_stats", name, FAIL, f"Newest snapshot {row[0]} is {age} days old; the daily "
                      "update writes one every morning. The model is predicting from stale stats.", **ev)
    if age > 1:
        return _check("team_stats", name, WARN, f"Newest snapshot {row[0]} is {age} days old.", **ev)
    return _check("team_stats", name, OK, f"Newest snapshot {row[0]} ({age} day(s) old).", **ev)


def check_archive_freshness(now: datetime, db: str = TEAM_DB) -> Dict[str, Any]:
    """Box scores and play-by-play for the current season.

    The box-score backfill fails the task when a game does not land, so this
    mostly watches play-by-play, which is a separate fetch and was in no
    scheduled job until 2026-09-24 (six play-in games went missing that way).
    """
    name = "NBA archive: box scores and play-by-play"
    today = now.astimezone().date()
    season = current_nba_season(today)
    try:
        conn = _ro(db)
        try:
            # The season label rolls over on 1 October; until opening night
            # the newest season with games is the one to watch.
            have = conn.execute("SELECT MAX(season) FROM box_scores").fetchone()[0]
            if have and have < season:
                season = have
            games, newest = conn.execute(
                "SELECT COUNT(*), MAX(game_date) FROM box_scores WHERE season=?", (season,)).fetchone()
            missing: List[str] = []
            if season >= PBP_FIRST_SEASON and _has_table(conn, "pbp_events"):
                cutoff = (today - timedelta(days=2)).isoformat()
                missing = [r[0] for r in conn.execute(
                    "SELECT b.game_id FROM box_scores b WHERE b.season=? AND b.game_date <= ? "
                    "AND NOT EXISTS (SELECT 1 FROM pbp_events p WHERE p.game_id=b.game_id) "
                    "ORDER BY b.game_id", (season, cutoff))]
        finally:
            conn.close()
    except Exception as exc:
        return _check("archive", name, UNKNOWN, f"Cannot read the archive: {exc}")
    ev = dict(season=season, games=games, newest_game=newest, missing_pbp=len(missing),
              missing_pbp_ids=missing[:10] or None)
    problems: List[tuple] = []
    if missing:
        problems.append((FAIL if nba_games_expected(today) else WARN,
                         f"{len(missing)} {season} game(s) older than two days have a box score but no "
                         f"play-by-play (e.g. {', '.join(missing[:3])}): on/off, clutch, runs and win "
                         "probability are missing them."))
    if nba_games_expected(today) and newest:
        age = (today - datetime.strptime(str(newest)[:10], "%Y-%m-%d").date()).days
        if age > 7:
            problems.append((FAIL, f"The newest {season} box score is from {newest}, {age} days ago, in season."))
        elif age > 3:
            problems.append((WARN, f"The newest {season} box score is from {newest}, {age} days ago."))
    if problems:
        return _check("archive", name, worst(p[0] for p in problems), " ".join(p[1] for p in problems), **ev)
    tail = "" if nba_games_expected(today) else " (offseason: no new games expected)"
    return _check("archive", name, OK, f"{season}: {games} games, newest {newest}; every game older "
                  f"than two days has play-by-play{tail}.", **ev)


#: How far back the box-score source check looks.
SOURCE_WINDOW_DAYS = 14


def check_box_score_sources(now: datetime, db: str = TEAM_DB) -> Dict[str, Any]:
    """Where the model's recent inputs came from: nba.com, or ESPN standing in.

    Added 2026-09-28 with the ESPN fallback (src/Utils/espn_boxscore.py) and
    the archive rebuild of the team-stats snapshot (team_stats_from_archive.py).
    Both keep the picks going while stats.nba.com refuses us, and both are
    exact to what the model reads; but a pick made from them was not made
    from nba.com's own numbers, and that must be visible, not assumed. WARN,
    not FAIL: the inputs are sound, the source is the news.
    """
    name = "Model inputs: which source each recent game came from"
    today = now.astimezone().date()
    since = (today - timedelta(days=SOURCE_WINDOW_DAYS)).isoformat()
    try:
        from src.Utils import espn_boxscore
        conn = _ro(db)
        try:
            rep = espn_boxscore.recent_sources(conn, since)
            newest = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name GLOB "
                                  "'[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]' ORDER BY name DESC LIMIT 1").fetchone()
            snap = None
            if newest and _has_table(conn, "team_stats_snapshot_source"):
                snap = conn.execute("SELECT source, games_through, n_games, n_games_espn FROM "
                                    "team_stats_snapshot_source WHERE table_name=?", (newest[0],)).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return _check("input_sources", name, UNKNOWN, f"Cannot read the archive: {exc}")
    ev = dict(since=since, nba_com_games=rep["nba_com"], espn_standing_in=rep["espn_only"],
              espn_superseded_by_nba_com=rep["espn_shadowed"], espn_held_not_counted=rep["espn_excluded"],
              espn_games=rep["espn_only_games"][:20] or None,
              newest_snapshot=newest[0] if newest else None,
              snapshot_source=(snap[0] if snap else "nba.com leaguedashteamstats"),
              snapshot_espn_games=(snap[3] if snap else 0))
    notes: List[str] = []
    if rep["espn_only"]:
        notes.append(f"{rep['espn_only']} game(s) since {since} are ESPN box scores standing in for "
                     f"nba.com's (e.g. {', '.join(rep['espn_only_games'][:3])}); nba.com has "
                     f"{rep['nba_com']}.")
    if snap:
        notes.append(f"The newest team-stats snapshot {newest[0]} was rebuilt from the archive, not "
                     f"read from nba.com ({snap[2]} games through {snap[1]}, {snap[3]} of them from ESPN).")
    if notes:
        return _check("input_sources", name, WARN, " ".join(notes), **ev)
    tail = f" {rep['espn_shadowed']} earlier ESPN stand-in(s) have since been replaced by nba.com's." \
        if rep["espn_shadowed"] else ""
    games = (f"no games since {since}" if not rep["nba_com"]
             else f"since {since}, all {rep['nba_com']} game(s) are nba.com box scores")
    return _check("input_sources", name, OK,
                  f"No ESPN stand-ins: {games}, and the newest team-stats snapshot is nba.com's.{tail}", **ev)


def check_odds_heartbeat(now: datetime, db: str = ODDS_DB) -> Dict[str, Any]:
    """The daily update's NBA board snapshot leaves an odds_polls row whether
    it succeeds or fails (the heartbeat added 2026-09-27)."""
    name = "NBA odds board: newest poll (odds_polls)"
    try:
        conn = _ro(db)
        try:
            if not _has_table(conn, "odds_polls"):
                snap = conn.execute("SELECT MAX(captured_at) FROM odds_snapshots").fetchone()[0] \
                    if _has_table(conn, "odds_snapshots") else None
                return _check("odds_heartbeat", name, UNKNOWN,
                              "odds_polls does not exist yet. The first board poll made by the "
                              "heartbeat code (committed 2026-09-27 22:46) creates it; expect it after "
                              "the next 9 AM daily update. Until then there is no record of polls that "
                              "saw no change." + (f" Newest changed price: {str(snap)[:16]} UTC." if snap else ""),
                              newest_changed_price=snap)
            rows = conn.execute("SELECT polled_at, status, error FROM odds_polls WHERE sport='NBA' "
                                "ORDER BY id DESC LIMIT 1").fetchone()
            ok_last = conn.execute("SELECT MAX(polled_at) FROM odds_polls WHERE sport='NBA' AND status='ok'"
                                   ).fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return _check("odds_heartbeat", name, UNKNOWN, f"Cannot read odds_polls: {exc}")
    if not rows:
        return _check("odds_heartbeat", name, UNKNOWN, "odds_polls exists but holds no NBA poll yet.")
    t, ok_t = _parse_iso(rows["polled_at"]), _parse_iso(ok_last)
    ev = dict(last_poll=_iso(t), last_status=rows["status"], last_ok_poll=_iso(ok_t), error=rows["error"])
    if rows["status"] == "failed":
        return _check("odds_heartbeat", name, FAIL if nba_games_expected(now.date()) else WARN,
                      f"The newest board poll ({_local(t)}) FAILED: {str(rows['error'] or '')[:160]}", **ev)
    if ok_t is None or now - ok_t > timedelta(hours=51):
        return _check("odds_heartbeat", name, FAIL, f"The newest successful board poll was {_ago(ok_t, now)}; "
                      "the daily update polls every morning.", **ev)
    if now - ok_t > timedelta(hours=27):
        return _check("odds_heartbeat", name, WARN, f"The newest successful board poll was {_ago(ok_t, now)}.", **ev)
    return _check("odds_heartbeat", name, OK, f"Newest board poll {_ago(t, now)}, {rows['status']}.", **ev)


def check_periodic_ingests(now: datetime, db: str = TEAM_DB) -> Dict[str, Any]:
    """refresh_registry.py's datasets (Hall of Fame, player directory, officials...)."""
    name = "Periodic ingests (refresh_registry)"
    try:
        from refresh_registry import JOBS as REG
        intervals = {j.name: j.interval_days for j in REG}
    except Exception:
        intervals = {}
    try:
        conn = _ro(db)
        try:
            if not _has_table(conn, "ingest_runs"):
                return _check("ingests", name, UNKNOWN, "ingest_runs does not exist.")
            rows = {r["name"]: dict(r) for r in conn.execute("SELECT * FROM ingest_runs")}
        finally:
            conn.close()
    except Exception as exc:
        return _check("ingests", name, UNKNOWN, f"Cannot read ingest_runs: {exc}")
    problems: List[tuple] = []
    for job in sorted(set(intervals) | set(rows)):
        r = rows.get(job)
        if r is None:
            problems.append((WARN, f"{job} has never run."))
            continue
        if r.get("last_status") != "ok":
            problems.append((FAIL, f"{job} FAILED {r.get('consecutive_failures') or 1} time(s) in a row: "
                                   f"{str(r.get('last_detail') or '')[:140]}"))
            continue
        t = _parse_iso(r.get("last_run_at"))
        every = intervals.get(job)
        if every and t and now - t > timedelta(days=every, hours=27):
            problems.append((WARN, f"{job} last ran {_ago(t, now)}; it is due every {every} day(s)."))
    ev = {"jobs": {k: {"last_run_at": v.get("last_run_at"), "last_status": v.get("last_status"),
                       "every_days": intervals.get(k)} for k, v in rows.items()}}
    if problems:
        return _check("ingests", name, worst(p[0] for p in problems), " ".join(p[1] for p in problems), **ev)
    return _check("ingests", name, OK, f"All {len(rows)} registered datasets ran ok within their interval.", **ev)


def _preflight_item(line: str) -> str:
    """'FAIL  label -- detail — advice' -> 'label (detail)'."""
    body = line.split("FAIL", 1)[1].strip()
    label, _, detail = body.partition(" -- ")
    detail = detail.split(" — ")[0].strip()
    return f"{label} ({detail[:90]})" if detail else label


def check_preflight(now: datetime, path: Optional[str] = None) -> Dict[str, Any]:
    """logs/preflight_latest.txt, written by every preflight run."""
    path = path or os.path.join(LOG_DIR, "preflight_latest.txt")
    name = "Opening-night preflight (latest report)"
    if not os.path.exists(path):
        return _check("preflight", name, UNKNOWN, f"{path} does not exist; the preflight has not run.")
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    head = lines[0] if lines else ""
    m = re.search(r"preflight (\S+) .*?(\d+) pass, (\d+) not yet, (\d+) wrong", head)
    if not m:
        return _check("preflight", name, UNKNOWN, "The preflight report has no summary line.")
    try:
        # preflight stamps datetime.now().isoformat(): local wall-clock time.
        t = _utc(datetime.fromisoformat(m.group(1)))
    except ValueError:
        t = None
    wrong = [ln.strip() for ln in lines[1:] if ln.strip().startswith("FAIL")]
    ev = dict(ran=_iso(t), passed=int(m.group(2)), not_yet=int(m.group(3)), wrong=int(m.group(4)),
              wrong_items=wrong or None)
    if t and now - t > timedelta(hours=51):
        return _check("preflight", name, WARN, f"The newest preflight report is {_ago(t, now)}; the daily "
                      "update runs it every morning.", **ev)
    if wrong:
        return _check("preflight", name, WARN,
                      f"{len(wrong)} check(s) WRONG at {_local(t)}: " +
                      "; ".join(_preflight_item(w) for w in wrong) +
                      ". Full list: logs/preflight_latest.txt.", **ev)
    return _check("preflight", name, OK, f"{m.group(2)} pass, {m.group(3)} not yet, 0 wrong ({_local(t)}).", **ev)


def check_ledger_publish(runs) -> Dict[str, Any]:
    name = "Ledger publish to the public server (push_ledger.py)"
    st = _step(runs, "publish ledger")
    if st is None:
        return _check("ledger_publish", name, UNKNOWN, "No run of the hourly job mentions the publish step.")
    t, step = st
    detail = step.get("detail") or ""
    if step["ok"] and detail.startswith("SKIPPED"):
        return _check("ledger_publish", name, OFF, "Not switched on: no public server is configured "
                      "(LEDGER_SYNC_URL / LEDGER_SYNC_SECRET). Correct until the site is deployed.",
                      last_run=_iso(t))
    if step["ok"]:
        return _check("ledger_publish", name, OK, f"Published at {_local(t)}: {detail[:160]}", last_run=_iso(t))
    return _check("ledger_publish", name, FAIL, f"The publish at {_local(t)} failed: {detail[:200]}. "
                  "The public track record is behind the home copy.", last_run=_iso(t))


def check_backup(now: datetime, root: Optional[str] = None) -> Dict[str, Any]:
    """backup_to_drive.py is run by hand. OneDrive syncs nothing (quota full),
    so this drive is the only copy of the ledger that exists anywhere else."""
    root = root or os.environ.get("BACKUP_ROOT") or ("D:" + os.sep + "BettingBuddy-Backup")
    name = "Backup to the external drive (backup_to_drive.py, by hand)"
    if not os.path.isdir(root):
        return _check("backup", name, UNKNOWN, f"{root} is not reachable (drive unplugged?). Cannot tell "
                      "how old the newest backup is.", root=root)
    dated = sorted(d for d in os.listdir(root)
                   if re.match(r"^\d{4}-\d\d-\d\d", d) and os.path.exists(os.path.join(root, d, "README.txt")))
    if not dated:
        return _check("backup", name, FAIL, f"No completed backup in {root}.", root=root)
    last = datetime.strptime(dated[-1][:10], "%Y-%m-%d").date()
    age = (now.astimezone().date() - last).days
    ev = dict(root=root, newest=dated[-1], age_days=age, count=len(dated))
    if age > 14:
        return _check("backup", name, FAIL, f"The newest backup is {age} days old ({dated[-1]}). The home PC "
                      "holds the only other copy of the ledger. Run backup_to_drive.py.", **ev)
    if age > 7:
        return _check("backup", name, WARN, f"The newest backup is {age} days old ({dated[-1]}); weekly is "
                      "the plan. Run backup_to_drive.py.", **ev)
    return _check("backup", name, OK, f"Newest backup {dated[-1]} ({age} day(s) old).", **ev)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def _job(spec: TaskSpec, sched: Dict[str, Any], now: datetime,
         extra: Callable[[Optional[list]], List[Dict[str, Any]]]) -> Dict[str, Any]:
    info = (sched.get("tasks") or {}).get(spec.task_name) or {}
    if spec.optional and sched.get("available") and not info:
        return {"id": spec.job_id, "name": spec.label, "kind": "scheduled", "task_name": spec.task_name,
                "what": spec.what, "expected_every": _every(spec.every), "last_run": None,
                "last_result": None, "last_result_meaning": None, "next_run": None,
                "overdue": False, "failing": False, "status": OFF,
                "summary": "Not switched on: no scheduled task runs this checker yet, so nothing is "
                           "sent anywhere. See docs/JOB_HEALTH.md.",
                "checks": [_check("scheduler", f"Task Scheduler: {spec.task_name}", OFF,
                                  "Not created yet (optional; see docs/JOB_HEALTH.md).")]}
    sched_check = check_scheduler(spec, sched, now)
    sched_last = _parse_iso(info.get("last_run")) if info else None
    runs, checks = None, [sched_check]
    if spec.log_path:      # the checker's own task has no log but its report
        try:
            runs = _runs_for(spec)
            checks.append(check_log(spec, runs, now, sched_last))
        except Exception as exc:
            checks.append(_check("log", "Log", UNKNOWN, f"Could not read the log: {exc}"))
    for fn in extra(runs):
        checks.append(fn)
    status = worst(c["status"] for c in checks)
    last_log = runs[-1] if runs else None
    return {
        "id": spec.job_id, "name": spec.label, "kind": "scheduled", "task_name": spec.task_name,
        "what": spec.what, "expected_every": _every(spec.every),
        "last_run": _iso(sched_last or (last_log or {}).get("started")),
        "last_result": info.get("last_result"),
        "last_result_meaning": _SCHED_CODES.get(info.get("last_result"), None) if info else None,
        "next_run": _iso(_parse_iso(info.get("next_run"))),
        "overdue": sched_check["status"] in (WARN, FAIL) and "Last started" in sched_check["summary"],
        "failing": any(c["status"] == FAIL for c in checks),
        "status": status,
        # The headline is the worst check's own words; a healthy job's is the
        # scheduler line ("last started ..., result 0").
        "summary": next(c["summary"] for c in checks if c["status"] == status)
        if status not in (OK, OFF) else sched_check["summary"],
        "checks": checks,
    }


def _safe(fn: Callable[[], Dict[str, Any]], cid: str, name: str) -> Dict[str, Any]:
    """One broken check must not hide the others."""
    try:
        return fn()
    except Exception as exc:
        return _check(cid, name, UNKNOWN, f"The check itself crashed: {type(exc).__name__}: {exc}")


_SECRET_IN_TEXT = re.compile(r"((?:api[_-]?key|token|secret|password)=)[^&\s'\")]+", re.IGNORECASE)


def _scrub(value: Any) -> Any:
    """Every string in the report with key=... values redacted. The checks
    quote log lines and stored errors, and a requests error quotes its URL,
    query string and API key included (found 2026-09-28: the Odds API key
    appeared in this report, which /api/admin/health serves)."""
    if isinstance(value, str):
        return _SECRET_IN_TEXT.sub(r"REDACTED", value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def collect(now: Optional[datetime] = None, sched: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole report as a dict, secrets redacted. Read-only."""
    return _scrub(_collect(now, sched))


def _collect(now: Optional[datetime] = None, sched: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = _utc(now) if now else datetime.now(timezone.utc)
    sched = sched if sched is not None else read_task_scheduler()

    extras: Dict[str, Callable[[Optional[list]], List[Dict[str, Any]]]] = {
        "odds_recorder": lambda runs: [
            _safe(lambda: check_nfl_odds_captures(now), "nfl_odds", "NFL odds captures")],
        "hourly": lambda runs: [
            _safe(lambda: check_nba_injury_polls(now), "nba_injuries", "NBA injuries"),
            _safe(lambda: check_nfl_injury_polls(now, runs), "nfl_injuries", "NFL injuries"),
            _safe(lambda: check_ledger_publish(runs), "ledger_publish", "Ledger publish")],
        "ledger_grading": lambda runs: [
            _safe(lambda: check_nfl_grading(now), "nfl_grading", "NFL grading")],
        "daily_update": lambda runs: [
            _safe(lambda: check_team_stats_snapshot(now), "team_stats", "Team stats"),
            _safe(lambda: check_archive_freshness(now), "archive", "Archive"),
            _safe(lambda: check_box_score_sources(now), "input_sources", "Model input sources"),
            _safe(lambda: check_nba_predictions_log(now), "nba_predictions", "NBA predictions"),
            _safe(lambda: check_odds_heartbeat(now), "odds_heartbeat", "Odds heartbeat"),
            _safe(lambda: check_periodic_ingests(now), "ingests", "Periodic ingests"),
            _safe(lambda: check_preflight(now), "preflight", "Preflight")],
    }
    jobs = [_job(spec, sched, now, extras.get(spec.job_id, lambda runs: [])) for spec in TASKS]

    backup = _safe(lambda: check_backup(now), "backup", "Backup")
    jobs.append({"id": "backup", "name": "Data backup", "kind": "manual", "task_name": None,
                 "what": "Ledger, archive and model data copied to the external drive",
                 "expected_every": "every week (run by hand)", "last_run": None, "last_result": None,
                 "last_result_meaning": None, "next_run": None,
                 "overdue": backup["status"] in (WARN, FAIL), "failing": backup["status"] == FAIL,
                 "status": backup["status"], "summary": backup["summary"], "checks": [backup]})

    # A BettingBuddy task nobody told this file about is unmonitored by
    # definition; say so rather than ignore it.
    known = {s.task_name for s in TASKS}
    for tname, info in sorted((sched.get("tasks") or {}).items()):
        if tname not in known:
            jobs.append({"id": "unmonitored:" + tname, "name": tname, "kind": "scheduled", "task_name": tname,
                         "what": info.get("action"), "expected_every": None,
                         "last_run": info.get("last_run"), "last_result": info.get("last_result"),
                         "last_result_meaning": _SCHED_CODES.get(info.get("last_result")),
                         "next_run": info.get("next_run"), "overdue": None, "failing": None,
                         "status": WARN, "summary": "A BettingBuddy task that job_health.py does not know "
                         "about, so nothing checks it. Add it to TASKS.", "checks": []})

    notes = []
    for info in (sched.get("tasks") or {}).values():
        if (info.get("logon_type") or "").lower() == "interactive":
            notes.append("Every BettingBuddy task is set to run only while you are logged in "
                         "(\"Interactive only\"). After a reboot, nothing runs until someone signs in.")
            break

    overall = worst(j["status"] for j in jobs)
    counts = {s: sum(1 for j in jobs if j["status"] == s) for s in (FAIL, WARN, UNKNOWN, OFF, OK)}
    return {"generated_at": _iso(now), "generated_local": _local(now), "host": socket.gethostname(),
            "overall": overall, "counts": counts, "scheduler_available": bool(sched.get("available")),
            "scheduler_error": sched.get("error"), "notes": notes, "jobs": jobs}


_BADGE = {OK: "OK     ", WARN: "WARN   ", FAIL: "FAIL   ", OFF: "OFF    ", UNKNOWN: "UNKNOWN"}


def summarize(report: Dict[str, Any]) -> str:
    """Plain English, one block per job, problems first."""
    c = report["counts"]
    head = (f"Job health, {report['generated_local']} on {report['host']}: "
            f"{report['overall'].upper()} ({c[FAIL]} failing, {c[WARN]} warning, {c[UNKNOWN]} unknown, "
            f"{c[OK]} ok, {c[OFF]} not expected yet)")
    out = [head, "=" * min(len(head), 100)]
    if not report["scheduler_available"]:
        out.append(f"Task Scheduler could not be read: {report['scheduler_error']}")
    order = sorted(report["jobs"], key=lambda j: -_RANK.get(j["status"], 1))
    for j in order:
        out.append("")
        out.append(f"{_BADGE.get(j['status'], j['status'])}  {j['name']} ({j['expected_every'] or 'unknown cadence'})"
                   f" - {j['what'] or ''}")
        for ch in j["checks"]:
            if j["status"] == OK and ch["status"] == OK and ch["id"] not in ("scheduler",):
                continue   # a healthy job gets one line, not seven
            out.append(f"   {ch['status'].upper():<7} {ch['name']}: {ch['summary']}")
        if not j["checks"]:
            out.append(f"   {j['summary']}")
    for n in report.get("notes") or []:
        out.append("")
        out.append("NOTE: " + n)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Notifications: pluggable, and nothing happens unless configured
# --------------------------------------------------------------------------- #

def _problems(report: Dict[str, Any]) -> List[str]:
    lines = []
    for j in report["jobs"]:
        for ch in j["checks"] or [{"status": j["status"], "name": j["name"], "summary": j["summary"]}]:
            if ch["status"] in (FAIL, WARN, UNKNOWN):
                lines.append(f"{ch['status'].upper()} {j['name']} / {ch['name']}: {ch['summary']}")
    return lines


def _fingerprint(report: Dict[str, Any]) -> str:
    """What is wrong, not how long it has been wrong: the same problem an hour
    later must not page again, a new one must."""
    keys = sorted(f"{j['id']}/{ch['id']}/{ch['status']}" for j in report["jobs"]
                  for ch in j["checks"] if ch["status"] in (FAIL, WARN, UNKNOWN))
    return hashlib.sha256("|".join(keys).encode()).hexdigest()[:16]


def _http_post(url: str, data: bytes, headers: Dict[str, str], timeout: int = 15) -> int:
    import urllib.request
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (owner-configured URL)
        return resp.status


def notify(report: Dict[str, Any], env: Optional[Dict[str, str]] = None,
           post: Optional[Callable[[str, Optional[bytes], Dict[str, str]], int]] = None,
           state_path: str = NOTIFY_STATE, now: Optional[datetime] = None,
           repeat_after: timedelta = timedelta(hours=12)) -> Dict[str, Any]:
    """Send the report's problems to whatever channels are configured.

    NOTHING IS SENT unless at least one of these is set (in .env or the
    environment):

      JOB_HEALTH_NTFY_TOPIC   a private ntfy.sh topic name; install the ntfy
                              app on the phone and subscribe to it. Optional
                              JOB_HEALTH_NTFY_SERVER for a self-hosted server.
      JOB_HEALTH_EMAIL_TO     an address, sent through Resend with
                              RESEND_API_KEY (and RESEND_FROM if set).
      JOB_HEALTH_PING_URL     a dead-man's-switch URL (healthchecks.io style):
                              pinged on every run, "<url>/fail" when failing.
                              This is the only channel that notices when the
                              PC itself is off, because the silence is the alert.

    A problem is sent once, again when the set of problems changes, and again
    every 12 hours while it lasts; recovery sends one "all clear".
    """
    env = dict(os.environ if env is None else env)
    post = post or _http_post
    now = _utc(now) if now else datetime.now(timezone.utc)
    topic = (env.get("JOB_HEALTH_NTFY_TOPIC") or "").strip()
    email_to = (env.get("JOB_HEALTH_EMAIL_TO") or "").strip()
    resend = (env.get("RESEND_API_KEY") or "").strip()
    ping = (env.get("JOB_HEALTH_PING_URL") or "").strip()
    result: Dict[str, Any] = {"configured": [], "sent": [], "errors": [], "reason": None}
    if topic:
        result["configured"].append("ntfy")
    if email_to and resend:
        result["configured"].append("email")
    if ping:
        result["configured"].append("ping")
    if not result["configured"]:
        result["reason"] = "no channel configured (see notify() / docs/JOB_HEALTH.md)"
        return result

    failing = report["overall"] == FAIL
    if ping:
        try:
            post(ping.rstrip("/") + ("/fail" if failing else ""), None, {})
            result["sent"].append("ping")
        except Exception as exc:
            result["errors"].append(f"ping: {exc}")

    try:
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        state = {}
    problems = _problems(report)
    fp = _fingerprint(report) if problems else "clean"
    last_fp, last_at = state.get("fingerprint"), _parse_iso(state.get("sent_at"))
    if not problems:
        due = last_fp not in (None, "clean")     # one all-clear after a problem
    else:
        due = fp != last_fp or last_at is None or now - last_at >= repeat_after
    if not due:
        result["reason"] = "nothing new to say"
        return result

    if problems:
        title = f"BettingBuddy jobs: {report['overall'].upper()} ({len(problems)} problem(s))"
        body = "\n".join(problems[:12]) + (f"\n...and {len(problems) - 12} more" if len(problems) > 12 else "")
    else:
        title, body = "BettingBuddy jobs: all clear", "Every scheduled job is healthy again."
    body += f"\n\n{report['generated_local']} on {report['host']}. Full report: logs/job_health.txt"

    if topic:
        server = (env.get("JOB_HEALTH_NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
        try:
            post(f"{server}/{topic}", body.encode("utf-8"),
                 {"Title": title.encode("ascii", "replace").decode(),
                  "Priority": "high" if failing else "default", "Tags": "warning" if problems else "white_check_mark"})
            result["sent"].append("ntfy")
        except Exception as exc:
            result["errors"].append(f"ntfy: {exc}")
    if email_to and resend:
        payload = {"from": env.get("RESEND_FROM") or "Betting Buddy <onboarding@resend.dev>",
                   "to": [e.strip() for e in email_to.split(",") if e.strip()],
                   "subject": title, "text": body}
        try:
            post("https://api.resend.com/emails", json.dumps(payload).encode("utf-8"),
                 {"Authorization": f"Bearer {resend}", "Content-Type": "application/json"})
            result["sent"].append("email")
        except Exception as exc:
            result["errors"].append(f"email: {exc}")

    if any(s in result["sent"] for s in ("ntfy", "email")):
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump({"fingerprint": fp, "sent_at": _iso(now), "channels": result["sent"]}, fh)
        except OSError as exc:
            result["errors"].append(f"state: {exc}")
    return result


def write_report(report: Dict[str, Any], json_path: str = REPORT_JSON, txt_path: str = REPORT_TXT) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    for path, text in ((json_path, json.dumps(report, indent=2, default=str)), (txt_path, summarize(report))):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)   # a reader never sees half a file


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only health report for every scheduled job.")
    ap.add_argument("--json", action="store_true", help="print the JSON report instead of the summary")
    ap.add_argument("--notify", action="store_true", help="send problems to the configured channels")
    ap.add_argument("--no-write", action="store_true", help="do not write logs/job_health.*")
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
    except Exception:
        pass
    report = collect()
    if not args.no_write:
        write_report(report)
    if args.notify:
        report["notified"] = notify(report)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(json.dumps(report, indent=2, default=str) if args.json else summarize(report))
    if args.notify and not args.json:
        n = report["notified"]
        print(f"notify: sent {n['sent'] or 'nothing'}" + (f" ({n['reason']})" if n.get("reason") else "")
              + (f"; errors {n['errors']}" if n["errors"] else ""))
    return 1 if report["overall"] == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
