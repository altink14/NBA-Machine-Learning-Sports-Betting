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
2. Wait for the first deploy (build + snapshot download — budget for a
   ~600 MB-plus archive, not the 47 MB this used to say), then hit
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

## 3. Data freshness in production (read this)

**stats.nba.com blocks most cloud-datacenter IPs**, so the daily backfill
generally cannot run on Railway. The working pipeline:

- The Windows scheduled task ("BettingBuddy Daily Data Update", 9 AM)
  keeps refreshing the LOCAL database as before.
- To push fresh data to production, re-publish the snapshot and redeploy:

  ```
  tar czf %TEMP%\db-snapshot.tar.gz -C Data TeamData.sqlite OddsData.sqlite
  gh release upload db-snapshot-v1 %TEMP%\db-snapshot.tar.gz --clobber
  ```

  then delete `TeamData.sqlite` from the Railway volume (or bump
  `DB_SNAPSHOT_URL` to a new tag) and redeploy so bootstrap re-downloads.
- Live odds (sbrscrape) and ESPN feeds (scores/news/injuries) are fetched
  at request time and usually work from cloud IPs; if sbrscrape gets
  blocked in practice, odds-dependent features degrade gracefully.

## Env inventory

The authoritative, annotated list now lives in `.env.example` in each repo —
those are kept in sync with the code. Summary:

Backend: `DB_SNAPSHOT_URL`, `CORS_ORIGINS`, `API_KEY?`, `RATE_LIMIT_DEFAULT?`,
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
`NEXT_PUBLIC_SITE_URL`, `NBA_API_KEY?`, `PREMIUM_BYPASS?` (local only).

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
