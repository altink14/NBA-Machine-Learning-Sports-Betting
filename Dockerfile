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

CMD ["sh", "-c", "python bootstrap_db.py && uvicorn main_api:app --host 0.0.0.0 --port ${PORT}"]
