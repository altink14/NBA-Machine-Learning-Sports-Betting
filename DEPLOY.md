# Deploying BettingBuddy

Two services: this repo (FastAPI backend) → **Railway**, the frontend
(`basic-saas-starter`) → **Vercel**. Both deploy from GitHub branch
`bettingbuddy2.0` / `bettingbuddy-2.0`.

## 0. Free option: Render (recommended to start)

1. https://render.com → sign in with GitHub → **New → Blueprint** →
   select `altink14/NBA-Machine-Learning-Sports-Betting` (branch
   `bettingbuddy2.0`). The `render.yaml` configures everything, including
   `DB_SNAPSHOT_URL` — just click Apply.
2. Wait for the first deploy (build + 47MB snapshot download), then hit
   `https://<render-url>/health`.
3. Free-tier tradeoffs: sleeps after 15 idle minutes (first request after
   that takes ~60-90s while it wakes and re-downloads the snapshot), and
   512MB RAM. If it ever OOMs under load, either upgrade ($7/mo, which
   also kills cold starts) or move to Hugging Face Spaces (free, 16GB).
4. After the frontend deploys, set `CORS_ORIGINS` to the Vercel URL in
   the Render dashboard.

## 1. Backend → Railway (paid alternative, ~$5/mo, no cold starts)

1. https://railway.app → sign in with GitHub → **New Project → Deploy from
   GitHub repo** → `altink14/NBA-Machine-Learning-Sports-Betting`, branch
   `bettingbuddy2.0`. Railway auto-detects the `Dockerfile`.
2. **Add a Volume** (service → Settings → Volumes): mount path `/app/Data`,
   1 GB. This is where the databases live.
3. **Variables** (service → Variables):
   - `DB_SNAPSHOT_URL` = `https://github.com/altink14/NBA-Machine-Learning-Sports-Betting/releases/download/db-snapshot-v1/db-snapshot.tar.gz`
   - `CORS_ORIGINS` = your Vercel URL once you have it (e.g. `https://bettingbuddy.vercel.app`)
   - Optional: `ODDS_API_KEY`, `DEFAULT_SPORTSBOOK` (copy from local `.env`)
4. Settings → Networking → **Generate Domain**. Note the URL —
   that's your `NEXT_PUBLIC_NBA_API_URL`.
5. First boot downloads the 47 MB snapshot into the volume automatically
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
  tar czf %TEMP%\db-snapshot.tar.gz -C Data TeamData.sqlite OddsData.sqlite dataset.sqlite
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
`NBA_CACHE_DIR?`, `PORT` (set by the platform).
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
