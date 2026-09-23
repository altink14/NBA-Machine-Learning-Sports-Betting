# Deploying BettingBuddy

> **Read this first — measured 2026-09-22.** The numbers below this box were
> written against a much smaller database and are corrected in place, but one
> thing has to be done before any of it works:
>
> **The published snapshot is stale and far too small.** The release asset
> holds a **349 MB** `TeamData.sqlite` from **9 August 2026**. The live one is
> **2,928 MB**. Deploying today would put a database in production that
> predates the served model's supporting data, the 1996-2001 backfill, three
> seasons of ingest, the `predictions_log` rebuild that added its constraints
> and triggers, and the odds `provenance` column. **Re-publish the snapshot
> (section 3) before the first deploy, not after.**
>
> | | measured |
> |---|---|
> | `TeamData.sqlite` live | 2.93 GB |
> | `OddsData.sqlite` | 4.5 MB |
> | `NflData.sqlite` | 1.28 GB — **skip it**, NFL is parked and unlinked |
> | Disk to provision | **~3 GB persistent volume**, not ephemeral (the 9am job writes daily) |
> | Model cold start | ~55-110s, warms in background when `WARM_MODEL_ON_START=true` |
> | Peak RAM on model load | **255 MB** (was 1,831 MB before the 2026-09-22 chunked read) |
>
> Do not trim `TeamData.sqlite` to make it fit. 1.6 GB of it is the raw
> box-score JSON the model parses on load, and the 4,025 dated tables going
> back to 2007-10-31 are the leakage-safe snapshots `retrain_features` builds
> every feature from — it refuses to start without them.

Two services: this repo (FastAPI backend) → **Railway**, the frontend
(`basic-saas-starter`) → **Vercel**. Both deploy from GitHub branch
`bettingbuddy2.0` / `bettingbuddy-2.0`.

## 0. Free option: Render (recommended to start)

1. https://render.com → sign in with GitHub → **New → Blueprint** →
   select `altink14/NBA-Machine-Learning-Sports-Betting` (branch
   `bettingbuddy2.0`). The `render.yaml` configures everything, including
   `DB_SNAPSHOT_URL` — just click Apply.
2. Wait for the first deploy (build + a **407 MB** snapshot download, not
   the 47 MB this used to say), then hit
   `https://<render-url>/health`.
3. Free-tier tradeoffs, and they are worse than they look for THIS app:
   no persistent disk, sleeps after 15 idle minutes, 512MB RAM. Every wake
   re-downloads the whole snapshot AND pays the model cold start, so the
   first request after an idle period is minutes, not seconds. The model
   itself now peaks at 255 MB so 512 MB is no longer arithmetically
   impossible, but the free plan is the wrong shape regardless — pay for an
   instance with a volume that stays warm.
4. After the frontend deploys, set `CORS_ORIGINS` to the Vercel URL in
   the Render dashboard.

## 1. Backend → Railway (paid alternative, ~$5/mo, no cold starts)

1. https://railway.app → sign in with GitHub → **New Project → Deploy from
   GitHub repo** → `altink14/NBA-Machine-Learning-Sports-Betting`, branch
   `bettingbuddy2.0`. Railway auto-detects the `Dockerfile`.
2. **Add a Volume** (service → Settings → Volumes): mount path `/app/Data`,
   **at least 4 GB** — the databases are ~2.93 GB today and the daily job
   grows them. 1 GB, which this used to say, will not hold TeamData alone.
3. **Variables** (service → Variables):
   - `DB_SNAPSHOT_URL` = `https://github.com/altink14/NBA-Machine-Learning-Sports-Betting/releases/download/db-snapshot-v1/db-snapshot.tar.gz`
   - `CORS_ORIGINS` = your Vercel URL once you have it (e.g. `https://bettingbuddy.vercel.app`)
   - Optional: `ODDS_API_KEY`, `DEFAULT_SPORTSBOOK` (copy from local `.env`)
4. Settings → Networking → **Generate Domain**. Note the URL —
   that's your `NEXT_PUBLIC_NBA_API_URL`.
5. First boot downloads the snapshot into the volume automatically
   (`bootstrap_db.py`). Check logs for `[bootstrap] Done.`, then hit
   `https://<railway-url>/health`.

## 2. Frontend → Vercel (~5 min)

1. https://vercel.com → sign in with GitHub → **Add New Project** →
   import `altink14/basic-saas-starter`, branch `bettingbuddy-2.0`.
2. Environment variables — copy every key from the local `.env`, plus:
   - `NEXT_PUBLIC_NBA_API_URL` = the Railway URL from step 1.4
   - `NEXT_PUBLIC_SITE_URL` = the Vercel production URL
3. Deploy. Then update Supabase (Auth → URL Configuration) to add the
   Vercel URL as Site URL / redirect, and point the Stripe webhook at
   `https://<vercel-url>/api/webhooks/stripe` (new signing secret →
   update `STRIPE_WEBHOOK_SECRET` in Vercel).

## 2b. Sync the Stripe catalogue BEFORE taking any money

Checked 2026-09-22: `products` and `prices` in Supabase are **empty**, and
`subscriptions.price_id` has a validated, non-deferrable foreign key to
`prices.id`. The only things that fill those tables are the webhook's
`product.created` / `price.created` handlers, and **Stripe does not re-fire
creation events for objects that already existed when a webhook endpoint is
registered.**

So the first real sale would go: checkout succeeds, card is charged, Stripe
sends `customer.subscription.created`, the upsert hits a foreign key violation,
no subscription row is written, `isUserSubscribed()` returns false — a paying
customer with no access, and Stripe retrying the webhook for days.

After the frontend deploys, set `ADMIN_SYNC_SECRET` (32+ random chars) in
Vercel, then run once:

```
curl -X POST https://<vercel-url>/api/admin/sync-stripe-catalog   -H "x-admin-secret: $ADMIN_SYNC_SECRET"
```

It returns `{"synced":{"products":N,"prices":M},"ok":true}`. It is idempotent —
re-run it whenever you add a product or price in the Stripe dashboard. With the
secret unset the route refuses, so an unconfigured deploy cannot expose it.

Verify before selling: `select count(*) from prices;` should be non-zero.

## 3. Data freshness in production (read this)

**stats.nba.com blocks most cloud-datacenter IPs**, so the daily backfill
generally cannot run on Railway. The working pipeline:

- The Windows scheduled task ("BettingBuddy Daily Data Update", 9 AM)
  keeps refreshing the LOCAL database as before.
- To push fresh data to production, re-publish the snapshot and redeploy:

  ```
  venv/Scripts/python.exe publish_db_snapshot.py
  ```

  It prints the `gh release upload` command and stops; publishing stays a
  deliberate act. **Do not go back to `tar czf` on the live files.**
  TeamData.sqlite is in WAL mode, so a committed transaction can still be
  sitting in the `-wal` sidecar that the tar does not copy — run it after the
  9am job and the snapshot silently omits the newest data, or mid-write and
  the copy is torn. Either way it looks fine and fails in production. The
  script uses `VACUUM INTO`, which SQLite guarantees is a consistent
  point-in-time copy even with other connections active, then runs
  `quick_check` on the result. Last measured: 2,929 MB → 2,799 MB → **407 MB**
  compressed, in about nine minutes.

  then delete `TeamData.sqlite` from the Railway volume (or bump
  `DB_SNAPSHOT_URL` to a new tag) and redeploy so bootstrap re-downloads.
- Live odds (sbrscrape) and ESPN feeds (scores/news/injuries) are fetched
  at request time and usually work from cloud IPs; if sbrscrape gets
  blocked in practice, odds-dependent features degrade gracefully.

## 3a. Where the prediction ledger lives (UNRESOLVED -- settle before deploy)

Found 2026-09-22. The ledger is `predictions_log` in `OddsData.sqlite`, and as
things stand a deploy creates TWO of them:

| | Home PC | Production server |
|---|---|---|
| Writes predictions | 9am job (`daily_update.py`) and any `/predictions` visit | any `/predictions` visit |
| Records closing lines | yes, every 15 minutes | no |
| Grades results + CLV | yes, 9am job | no |
| What `/track-record` shows | nothing (not public) | this copy |

So on the live site the public record would be rows written whenever a
visitor happened to load the picks, built on team stats only as fresh as the
last snapshot, and never graded. The complete, graded record would sit on
the home PC where nobody can see it.

Two guards landed with this note; neither changes how the home PC behaves:

- `bootstrap_db.py` never overwrites an existing `OddsData.sqlite`. Before
  this, the refresh in section 3 (delete TeamData, redeploy) re-extracted the
  ledger from the snapshot and silently deleted every row production had
  written since. The no-delete triggers cannot see a whole file being replaced.
- `LOG_PREDICTIONS_ON_REQUEST=false` stops `/predictions` visits writing to the
  ledger. Set it on every server that is not the ledger's single writer.

The decision still needed is which machine is the single writer:

- **A. Home PC writes, production mirrors (recommended).** Everything that
  already works stays where it is. After the 9am job, the home PC pushes
  `OddsData.sqlite` (about 5 MB) to production through a key-protected
  upload that refuses any file missing or altering a row production already
  has, so the public record can only grow. Production shows the picks that
  were logged instead of recomputing them on older team stats: the pick a
  visitor sees is the pick on the record. Cost: the upload route, the push
  step and its checks, roughly a day. Risk: if the home PC is off or offline, the record
  is late; it is never wrong.
- **B. Production writes.** Move the recorder, grader and logging to the
  server. Blocked by the same thing as section 3: stats.nba.com refuses
  cloud IPs, so the server's team stats (and so its picks) are only as fresh
  as the last snapshot.
- **C. The home PC IS the backend,** exposed through a tunnel. One copy, no
  sync, but the whole site goes down whenever the home PC sleeps, restarts or
  loses Wi-Fi.

## Env inventory

The authoritative, annotated list now lives in `.env.example` in each repo —
those are kept in sync with the code. Summary:

Backend: `DB_SNAPSHOT_URL`, `CORS_ORIGINS`, **`API_KEY`** (not optional once you
charge for anything — unset, `/predictions` is served to anyone who finds the
host, which is the product), `RATE_LIMIT_DEFAULT?`,
`RATE_LIMIT_GLOBAL?`, `RATE_LIMIT_EXPENSIVE?`, `RATE_LIMIT_UPSTREAM?`,
`NBA_CACHE_DIR?`, `WARM_MODEL_ON_START` (set it to `true` in production —
otherwise the first prediction after every restart pays the model's cold
load and a 30s gateway gives up first), `PORT` (set by the platform).
`ODDS_API_KEY` and `DEFAULT_SPORTSBOOK` are listed in `render.yaml` but read by
no Python in this repo — setting them does nothing today.

Frontend: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`,
`SUPABASE_SERVICE_ROLE_KEY`, `GEMINI_API_KEY`, `STRIPE_SECRET_KEY`,
`NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`, `STRIPE_WEBHOOK_SECRET`,
`NEXT_PUBLIC_STRIPE_PRICE_*` (4x), `NEXT_PUBLIC_NBA_API_URL`,
`NEXT_PUBLIC_SITE_URL`, `NBA_API_KEY?`, `ADMIN_SYNC_SECRET` (required once, see 2b — without it the first sale charges the card and grants nothing), `PREMIUM_BYPASS?` (local only).

## Security notes — read before going live

- **`CORS_ORIGINS` is not in `render.yaml` on purpose** (step 0.4). Until you set
  it in the dashboard the backend falls back to `localhost:3000`, and the
  deployed frontend's requests will be rejected by the browser. This is the
  single easiest way to ship a broken production site.
- **`API_KEY` protects an allowlist, not everything**: `/predictions`,
  `/api/parlay/evaluate`, `/api/line-movements`. The public reference endpoints
  are keyless by design — they're fetched directly from the browser. Set
  `NBA_API_KEY` on Vercel to the same value; the frontend attaches the header
  only from server-side route handlers.
- **Rate limits need real client IPs.** The Dockerfile passes
  `--forwarded-allow-ips="*"`; without it every request behind the platform
  proxy looks like one IP and `RATE_LIMIT_GLOBAL` becomes one shared bucket for
  all users. If you ever override the start command, keep that flag.
- **Limits are in-process.** They reset on restart and are per-instance. Fine
  for a single instance; needs Redis (`storage_uri`) if you scale out.
