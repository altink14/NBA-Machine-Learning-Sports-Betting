# BettingBuddy stats backend
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; wheels cover pandas/numpy/xgboost on 3.11-slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The stats databases are NOT in the image (315MB+, live data). On boot,
# bootstrap_db.py downloads the latest snapshot from DB_SNAPSHOT_URL into
# the mounted volume at /app/Data if TeamData.sqlite is missing.
ENV PORT=8000
EXPOSE 8000

# --forwarded-allow-ips="*" is required for per-IP rate limiting to work.
# uvicorn already enables proxy_headers by default, but it only trusts
# X-Forwarded-For when the immediate peer is in forwarded_allow_ips, which
# defaults to 127.0.0.1. Behind Render/Railway the edge proxy is NOT on
# loopback, so without this every visitor is seen as the same address and one
# crawler would exhaust the shared rate-limit bucket for the whole userbase.
# "*" is safe here because the container is only reachable through the
# platform's proxy - a client cannot connect directly to spoof the header.
# Equivalent alternative if you prefer config over code: set the env var
# FORWARDED_ALLOW_IPS=* in the service dashboard (uvicorn reads it directly).
CMD ["sh", "-c", "python bootstrap_db.py && uvicorn main_api:app --host 0.0.0.0 --port ${PORT} --forwarded-allow-ips=\"*\""]
