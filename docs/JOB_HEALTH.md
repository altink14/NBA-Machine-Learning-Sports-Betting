# Job health (`job_health.py`)

A read-only check of every scheduled job on the home PC. For each job it reports the
last run, the last result, the expected cadence, and whether the job is overdue or
failing. It gets that from three places: Task Scheduler's own record, the job's log,
and the data the job should have written (injury polls, odds heartbeat, team-stats
snapshot, play-by-play, ungraded picks, the newest backup). It never creates,
changes, starts or disables a task, and it makes no network call unless you turn
notifications on.

```
venv/Scripts/python.exe job_health.py            # plain-English report; writes logs/job_health.json + .txt
venv/Scripts/python.exe job_health.py --json     # the JSON instead
venv/Scripts/python.exe job_health.py --notify   # also send problems to the configured channels
```

Exit code 1 means something is failing. `unknown` means the checker could not look.
It never counts as ok.

| Job (Task Scheduler name) | Cadence | Evidence beyond the exit code |
|---|---|---|
| BettingBuddy Odds Recorder | 15 min | runs in the last 24 h, the longest silence, newest NFL line captured |
| BettingBuddy Injuries And Predictions | hourly | newest NBA and NFL injury poll (a failed NBA poll fails the check), ledger publish |
| BettingBuddy Ledger Grading | daily 9:30 | NFL picks still pending 36 h after kickoff |
| BettingBuddy Daily Data Update | daily 9:00 | preflight and audit verdicts (these never change the exit code), team-stats age, play-by-play gaps, NBA picks logged (from opening night), odds_polls heartbeat, refresh_registry datasets |
| Backup (by hand) | weekly | newest completed folder on the backup drive |

## The admin page

`/admin/health` in the frontend shows the same report. Only `ADMIN_EMAILS` can open it;
anyone else gets a 404. It reads `GET /api/admin/health`, which requires the
`X-Internal-Key` header.

**To switch it on:** set the same random value as `INTERNAL_API_KEY` in the backend
`.env` and the frontend `.env.local`, then restart both. With the key unset, the
endpoint answers 503. The same key also exempts the frontend's server-side renderer
from the per-IP rate limits (see `backend-fetch.ts`), which production needs anyway.

## Phone notifications (off until configured)

Nothing is sent unless one of these is set in the backend `.env`, **and** the checker
runs with `--notify`:

- `JOB_HEALTH_NTFY_TOPIC=<a long random name>` pushes to the ntfy app
  (https://ntfy.sh). Install the app and subscribe to that topic. Anyone who knows the
  topic name can read it, so make the name long and random.
  `JOB_HEALTH_NTFY_SERVER` points it at a self-hosted server.
- `JOB_HEALTH_EMAIL_TO=you@example.com` together with `RESEND_API_KEY` (and optionally
  `RESEND_FROM`) sends email through Resend.
- `JOB_HEALTH_PING_URL=https://hc-ping.com/<uuid>` is a dead-man's switch
  (healthchecks.io). The checker pings it every run and adds `/fail` when something is
  failing. **This is the only channel that notices when the PC itself is off or
  asleep**, because the alert is the missing ping.

A problem is sent once, again when the set of problems changes, and again every 12 h
while it lasts. When things recover, one "all clear" is sent.

**The one step (after the env var):** give the checker a schedule of its own. Run this
once from an elevated prompt:

```
schtasks /create /tn "BettingBuddy Job Health" /sc hourly /mo 1 /st 00:50 ^
  /tr "\"C:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting\venv\Scripts\python.exe\" \"C:\Users\altin\OneDrive\Documents\GitHub\NBA-Machine-Learning-Sports-Betting\job_health.py\" --notify"
```

It is a separate task on purpose. Placed inside the hourly job, it could not report
that the hourly job had stopped. The report already knows about this task: it shows
"off" until the task exists, and after that it is checked like the others.
